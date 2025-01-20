import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import xgboost as xgb
import cProfile, pstats
import psutil  
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# ===== Added TensorBoard Import =====
from torch.utils.tensorboard import SummaryWriter

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("results/logs/train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage=""):
    """
    Logs the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_gb = mem_info.rss / (1024 ** 3)
    logger.info(f"Memory Usage at {stage}: {mem_usage_gb:.2f} GB")

######################################################################
# Utility: Create Necessary Directories
######################################################################

def create_directories():
    """
    Create necessary directories if they don't exist.
    """
    directories = ["results/models", "results/logs", "results/logs/tensorboard", "results"]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory checked/created: {dir_path}")

######################################################################
# Utility: Downcast DataFrame for Memory Optimization
######################################################################

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numerical columns to the most efficient data type.
    """
    float_cols = df.select_dtypes(include=['float']).columns
    int_cols = df.select_dtypes(include=['int']).columns

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df

######################################################################
# Utility: Clean & Check DataFrame
######################################################################

def clean_dataframe(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Replace inf/-inf with fill_value, fill NaNs with fill_value.
    """
    df = df.replace([np.inf, -np.inf], fill_value)
    df = df.fillna(fill_value)
    return df

def check_infinite_values(df, name: str):
    """
    Check and log if there are any infinite values in the DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric_df).all().all():
        inf_count = np.isinf(numeric_df).sum().sum()
        logger.warning(f"{name} contains {inf_count} inf/-inf values in numeric columns.")
    else:
        logger.info(f"No infinite values found in numeric columns of {name}.")

######################################################################
# Feature Selection
######################################################################

def select_features(X_train, y_train, X_val, k=50):
    """
    Selects the top k features based on ANOVA F-test.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    
    selected_features = [feature for bool, feature in zip(selector.get_support(), X_train.columns) if bool]
    
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    
    logger.info(f"Selected top {k} features: {selected_features}")
    
    return X_train_selected, X_val_selected, selected_features

######################################################################
# Weighted Loss Function Definitions
######################################################################

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(weight=class_weights, **kwargs)

def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets.
    """
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)

######################################################################
# Dataset Definition
######################################################################

class CryptoSlidingWindowDataset(Dataset):
    def __init__(self, df, window_size=5, has_target=True):
        super().__init__()
        self.window_size = window_size
        self.has_target = has_target

        # Sort by timestamp if present
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        if self.has_target:
            if "target" not in df.columns:
                raise ValueError("DataFrame must have 'target' when has_target=True.")
            self.targets = df["target"].values

        feature_cols = [c for c in df.columns if c not in ["timestamp", "target"]]
        feature_array = df[feature_cols].values

        self.X = []
        if self.has_target:
            self.y = []
        else:
            self.y = None

        for i in range(len(feature_array) - window_size):
            seq = feature_array[i:i+window_size]
            self.X.append(seq)
            if self.has_target:
                # label is the last step in the window
                label = self.targets[i + window_size - 1]
                self.y.append(label)

        self.X = np.array(self.X, dtype=np.float32)
        if self.has_target:
            self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_target:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

######################################################################
# Model Definitions
######################################################################

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        dec_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        energy = self.v(torch.tanh(self.W(encoder_outputs) + self.U(dec_hidden_expanded)))
        attn_weights = torch.softmax(energy, dim=1)
        context = torch.sum(encoder_outputs * attn_weights, dim=1)
        return context, attn_weights

class LSTMAttnClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3, num_layers=2,
                 num_classes=2, use_attention=True, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        if use_attention:
            self.attention = BahdanauAttention(hidden_dim * (2 if bidirectional else 1))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            dec_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            dec_hidden = hn[-1]
        
        if self.use_attention:
            context, _ = self.attention(dec_hidden, lstm_out)
        else:
            context = lstm_out[:, -1, :]
        context = self.dropout(context)
        logits = self.fc(context)
        return logits

######################################################################
# Training Functions
######################################################################

def train_xgb(trial, X_train, y_train, X_val, y_val, writer=None, fold=1):
    """
    Train an XGBoost model with hyperparameter optimization using Optuna.
    """
    # Debugging: Print XGBoost version and file path
    print(f"XGBoost Version: {xgb.__version__}")
    
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",  
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "n_estimators": 1000,  
        "tree_method": "auto",  
        "n_jobs": -1,
    }

    from xgboost.callback import EarlyStopping

    model = xgb.XGBClassifier(
        **param,
        callbacks=[EarlyStopping(rounds=50, save_best=True)]
    )

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except TypeError as e:
        print(f"TypeError encountered: {e}")
        raise e

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")  # Compute Macro-Averaged F1

    # ===== Added TensorBoard Logging for XGBoost =====
    if writer:
        writer.add_scalar(f"Fold_{fold}/XGBoost_F1", f1, trial.number)
        for param_name, param_value in trial.params.items():
            writer.add_scalar(f"Fold_{fold}/Hyperparameters/{param_name}", param_value, trial.number)

    return f1

def train_lstm(trial, train_df, val_df, window_size=5, writer=None, fold=1):
    """
    Train an LSTM model with hyperparameter optimization using Optuna.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"LSTM training on device: {device}")

    train_dataset = CryptoSlidingWindowDataset(train_df, window_size=window_size)
    val_dataset = CryptoSlidingWindowDataset(val_df, window_size=window_size)

    # Hyperparameter suggestions
    batch_size = trial.suggest_int("batch_size", 64, 256)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    input_dim = train_dataset[0][0].shape[1]
    num_classes = len(np.unique(train_df["target"]))

    # Model, Loss, Optimizer, Scheduler
    class_weights = calculate_class_weights(train_df["target"].values).to(device)
    criterion = WeightedCrossEntropyLoss(class_weights=class_weights)

    model = LSTMAttnClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        bidirectional=bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    epochs = 50
    best_val_f1 = 0

    # ===== Added TensorBoard Writer for LSTM =====
    # Initialize per-fold writer if needed
    # (Alternatively, use the main writer passed as argument)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=(device.type == "mps")):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with autocast(device_type=device.type, enabled=(device.type == "mps")):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        scheduler.step(val_f1)

        # Report intermediate objective value to Optuna
        trial.report(val_f1, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
            if writer:
                writer.add_scalar(f"Fold_{fold}/LSTM_Prune", epoch + 1, trial.number)
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        # ===== Added TensorBoard Logging for LSTM =====
        if writer:
            writer.add_scalar(f"Fold_{fold}/LSTM_Train_Loss", train_loss, epoch)
            writer.add_scalar(f"Fold_{fold}/LSTM_Val_F1", val_f1, epoch)
            writer.add_scalar(f"Fold_{fold}/LSTM_Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Manual Logging of Learning Rates
        current_lrs = scheduler.get_last_lr()
        logger.info(f"Epoch {epoch+1}: Current Learning Rates: {current_lrs}")

    return best_val_f1

######################################################################
# Stacking Feature Importance Plot
######################################################################

def plot_meta_feature_importance(meta_model, feature_names):
    """
    Plots feature importance for stacking meta-model.
    """
    if isinstance(meta_model, xgb.XGBClassifier):
        importance = meta_model.feature_importances_
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importance = importance[sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_features, sorted_importance)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Meta-Model Feature Importance")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("results/models/meta_model_feature_importance.png")
        plt.close()
    elif isinstance(meta_model, (RandomForestClassifier, LogisticRegression)):
        if hasattr(meta_model, 'coef_'):
            importances = meta_model.coef_[0]
        else:
            importances = meta_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importance = importances[sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_features, sorted_importance)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Meta-Model Feature Importance")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("results/models/meta_model_feature_importance.png")
        plt.close()
    else:
        logger.warning("Unsupported meta-model type for feature importance plotting.")

######################################################################
# OOF Predictions for XGBoost
######################################################################

def generate_oof_predictions_xgb(combined_df, folds, features, label_col="target", writer=None):
    """
    Generate out-of-fold (OOF) predictions for XGBoost.
    """
    oof_preds = np.zeros(len(combined_df))
    
    for fold, (train_idx, val_idx) in enumerate(folds, 1):  
        logger.info(f"Training XGBoost Fold {fold}/{len(folds)}")
        train_fold = combined_df.iloc[train_idx]
        val_fold = combined_df.iloc[val_idx]
        
        X_train = train_fold[features]
        y_train = train_fold[label_col].values
        X_val = val_fold[features]
        y_val = val_fold[label_col].values
        
        # Define storage and study name
        storage_path = "sqlite:///results/models/optuna_xgb.db"
        study_name = f"xgb_study_fold{fold}"
        
        # Optimize hyperparameters with Optuna
        study = optuna.create_study(
            direction="maximize",  # Maximizing F1
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True
        )
        study.optimize(lambda trial: train_xgb(trial, X_train, y_train, X_val, y_val, writer=writer, fold=fold), n_trials=50, timeout=600)
        
        logger.info(f"Best parameters for Fold {fold}: {study.best_params}")
        logger.info(f"Best F1 for Fold {fold}: {study.best_value:.4f}")  
        
        # Train final model with best hyperparameters
        best_params = study.best_params
        best_params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": sum(y_train == 0) / sum(y_train == 1),
            "n_estimators": 1000, 
            "tree_method": "auto",  
            "n_jobs": -1,
        })
        
        from xgboost.callback import EarlyStopping
        
        model = xgb.XGBClassifier(
            **best_params,
            callbacks=[EarlyStopping(rounds=50, save_best=True)]
        )
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Generate binary predictions
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds  # Store predictions as binary for F1
    
    return oof_preds

######################################################################
# OOF Predictions for LSTM
######################################################################

def generate_oof_predictions_lstm(combined_df, folds, features, window_size=5, label_col="target", writer=None):
    """
    Generate out-of-fold (OOF) predictions for LSTM.
    """
    oof_preds = np.zeros(len(combined_df))
    
    for fold, (train_idx, val_idx) in enumerate(folds, 1):  
        logger.info(f"Training LSTM Fold {fold}/{len(folds)}")
        train_fold = combined_df.iloc[train_idx]
        val_fold = combined_df.iloc[val_idx]
        
        # Define storage and study name
        storage_path = "sqlite:///results/models/optuna_lstm.db"
        study_name = f"lstm_study_fold{fold}"
        
        # Create or load the Optuna study
        study = optuna.create_study(
            direction="maximize", 
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True
        )
        
        # Optimize hyperparameters with Optuna by passing the trial to train_lstm
        study.optimize(
            lambda trial: train_lstm(trial, train_fold, val_fold, window_size=window_size, writer=writer, fold=fold),
            n_trials=50,
            timeout=1800  # Adjust timeout as needed
        )
        
        logger.info(f"Best parameters for Fold {fold}: {study.best_params}")
        logger.info(f"Best F1 for Fold {fold}: {study.best_value:.4f}")
        
        # Train final model with best hyperparameters
        best_params = study.best_params
        
        lstm_model = LSTMAttnClassifier(
            input_dim=len(features),
            hidden_dim=best_params.get("hidden_dim", 256),
            num_layers=best_params.get("num_layers", 2),
            dropout=best_params.get("dropout", 0.3),
            num_classes=len(np.unique(train_fold["target"])),
            bidirectional=best_params.get("bidirectional", True),
        ).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=best_params.get("learning_rate", 1e-3))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        # Prepare datasets and loaders
        train_dataset = CryptoSlidingWindowDataset(train_fold, window_size=window_size, has_target=True)
        val_dataset = CryptoSlidingWindowDataset(val_fold, window_size=window_size, has_target=True)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=best_params.get("batch_size", 128), 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=best_params.get("batch_size", 128), 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        # Calculate class weights
        class_weights = calculate_class_weights(train_fold["target"].values).to(
            torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )
        criterion = WeightedCrossEntropyLoss(class_weights=class_weights)
        
        # Training loop
        epochs = 50
        best_val_f1 = 0
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        for epoch in range(epochs):
            lstm_model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=(device.type == "mps")):
                    outputs = lstm_model(X_batch)
                    loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
    
            lstm_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    with autocast(device_type=device.type, enabled=(device.type == "mps")):
                        outputs = lstm_model(X_batch)
                        loss = criterion(outputs, y_batch)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
    
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            scheduler.step(val_f1)
    
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
    
            # ===== Added TensorBoard Logging for LSTM =====
            if writer:
                writer.add_scalar(f"Fold_{fold}/LSTM_Train_Loss", train_loss, epoch)
                writer.add_scalar(f"Fold_{fold}/LSTM_Val_F1", val_f1, epoch)
                writer.add_scalar(f"Fold_{fold}/LSTM_Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    
            # Manual Logging of Learning Rates
            current_lrs = scheduler.get_last_lr()
            logger.info(f"Epoch {epoch+1}: Current Learning Rates: {current_lrs}")
        
        # Generate binary predictions
        lstm_model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                with autocast(device_type=device.type, enabled=(device.type == "mps")):
                    outputs = lstm_model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        oof_preds[val_idx] = all_preds
    
        # Cleanup
        del lstm_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    return oof_preds

######################################################################
# Stacking Model Training
######################################################################

def train_stacking_model_cv(oof_xgb, oof_lstm, y, model_type="rf", n_jobs=2, n_splits=5, writer=None):
    """
    Train a stacking meta-learner using cross-validated OOF predictions.
    """
    X_meta = np.vstack((oof_xgb, oof_lstm)).T
    y_meta = y
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    meta_preds = np.zeros(len(y_meta))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_meta), 1):
        logger.info(f"Training meta-model Fold {fold}/{n_splits}")
        X_train_fold, X_val_fold = X_meta[train_idx], X_meta[val_idx]
        y_train_fold, y_val_fold = y_meta[train_idx], y_meta[val_idx]
        
        if model_type.lower() == "xgb":
            scale_pos_weight = sum(y_train_fold == 0) / sum(y_train_fold == 1)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",  
                "scale_pos_weight": scale_pos_weight,
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 100,
                "n_jobs": n_jobs,
            }
            from xgboost.callback import EarlyStopping
            meta_model = xgb.XGBClassifier(
                **params,
                callbacks=[EarlyStopping(rounds=10, save_best=True)]
            )
            meta_model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )
            preds = meta_model.predict(X_val_fold)
            
            # ===== Optional TensorBoard Logging for Meta-Model =====
            if writer:
                writer.add_scalar(f"Meta-Model/Fold_{fold}/XGBoost_F1", f1_score(y_val_fold, preds, average="macro"), fold)
        
        elif model_type.lower() == "lr":
            meta_model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=n_jobs)
            meta_model.fit(X_train_fold, y_train_fold)
            preds = meta_model.predict(X_val_fold)
            
            if writer:
                writer.add_scalar(f"Meta-Model/Fold_{fold}/LR_F1", f1_score(y_val_fold, preds, average="macro"), fold)
        
        elif model_type.lower() == "rf":
            meta_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=n_jobs)
            meta_model.fit(X_train_fold, y_train_fold)
            preds = meta_model.predict(X_val_fold)
            
            if writer:
                writer.add_scalar(f"Meta-Model/Fold_{fold}/RF_F1", f1_score(y_val_fold, preds, average="macro"), fold)
        
        else:
            raise ValueError("Unsupported meta-model type. Choose 'xgb', 'lr', or 'rf'.")
        
        meta_preds[val_idx] = preds
        
        # Optionally, save each fold's meta-model
        # joblib.dump(meta_model, f"results/models/meta_model_fold{fold}.pkl")
    
    # Evaluate overall meta-model performance
    evaluate_and_print_metrics(y_meta, meta_preds, prefix="Meta-learner")
    
    # ===== Optional TensorBoard Logging for Meta-Model =====
    if writer:
        overall_f1 = f1_score(y_meta, meta_preds, average="macro")
        writer.add_scalar("Meta-Model/Overall_F1", overall_f1, 0)
    
    # Train final meta-model on the entire dataset
    if model_type.lower() == "xgb":
        scale_pos_weight = sum(y_meta == 0) / sum(y_meta == 1)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 200,
            "n_jobs": n_jobs,
        }
        from xgboost.callback import EarlyStopping
        final_meta_model = xgb.XGBClassifier(
            **params,
            callbacks=[EarlyStopping(rounds=10, save_best=True)]
        )
        final_meta_model.fit(X_meta, y_meta, verbose=False)
        joblib.dump(final_meta_model, "results/models/stacking_meta_model_final_xgb.pkl")
        
        if writer:
            preds = final_meta_model.predict(X_meta)
            overall_f1_final = f1_score(y_meta, preds, average="macro")
            writer.add_scalar("Meta-Model/Final_XGB_F1", overall_f1_final, 0)
        
    elif model_type.lower() == "lr":
        final_meta_model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=n_jobs)
        final_meta_model.fit(X_meta, y_meta)
        joblib.dump(final_meta_model, "results/models/stacking_meta_model_final_lr.pkl")
        
        if writer:
            preds = final_meta_model.predict(X_meta)
            overall_f1_final = f1_score(y_meta, preds, average="macro")
            writer.add_scalar("Meta-Model/Final_LR_F1", overall_f1_final, 0)
        
    elif model_type.lower() == "rf":
        final_meta_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=n_jobs)
        final_meta_model.fit(X_meta, y_meta)
        joblib.dump(final_meta_model, "results/models/stacking_meta_model_final_rf.pkl")
        
        if writer:
            preds = final_meta_model.predict(X_meta)
            overall_f1_final = f1_score(y_meta, preds, average="macro")
            writer.add_scalar("Meta-Model/Final_RF_F1", overall_f1_final, 0)
    
    return final_meta_model

######################################################################
# Additional Metric Printing
######################################################################

def evaluate_and_print_metrics(y_true, y_pred, prefix="Model"):
    """
    Computes and prints confusion matrix, recall, precision, F1, and ROC AUC for the given predictions.
    Also prints top-level info about ROC curve and precision-recall curve thresholds.
    """
    # 1. Metrics
    cm = confusion_matrix(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro")

    # If classes are [0,1] and both present, compute ROC AUC
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float('nan')

    logger.info(f"[{prefix}] Confusion Matrix:\n{cm}")
    logger.info(f"[{prefix}] Recall={rec:.4f}  Precision={prec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # 2. ROC Curve (truncated display)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_pred)
    logger.info(
        f"[{prefix}] ROC curve points (showing first 5):\n"
        f"   FPR={fpr[:5]}, TPR={tpr[:5]}, TH={roc_thresh[:5]} (truncated)"
    )
    
    # 3. Precision-Recall Curve (truncated display)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_pred)
    logger.info(
        f"[{prefix}] Precision-Recall points (showing first 5):\n"
        f"   Precision={pr_prec[:5]}, Recall={pr_rec[:5]}, TH={pr_thresh[:5]} (truncated)"
    )

######################################################################
# Run the Main Function
######################################################################

def main():
    # Confirm the updated script is running
    print("Running the updated train_model.py script.")
    
    # Ensure multiprocessing uses 'spawn' method for macOS and Windows compatibility
    if sys.platform.startswith("darwin") or sys.platform.startswith("win"):
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass 

    # Create necessary directories
    create_directories()

    # ===== Initialize TensorBoard SummaryWriter =====
    writer = SummaryWriter(log_dir="results/logs/tensorboard")
    logger.info("Initialized TensorBoard SummaryWriter.")

    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    logger.info("Loading train & val data")
    log_memory_usage("After Loading Data")
    
    train_csv = "/Users/mchildress/Code/my_crypto_prediction/data/intermediate/train_fe.csv"
    val_csv   = "/Users/mchildress/Code/my_crypto_prediction/data/intermediate/val_fe.csv"

    # Load data with specified dtypes
    dtype_mapping = {'target': 'int32'}
    train_df = pd.read_csv(train_csv, dtype=dtype_mapping)
    val_df = pd.read_csv(val_csv, dtype=dtype_mapping)

    log_memory_usage("After Loading CSVs")

    # Clean & check
    train_df = clean_dataframe(train_df)
    val_df = clean_dataframe(val_df)
    check_infinite_values(train_df, "train_df")
    check_infinite_values(val_df, "val_df")

    log_memory_usage("After Cleaning Data")

    # Feature Selection
    train_features = [c for c in train_df.columns if c not in ("timestamp", "target")]
    val_features = [c for c in val_df.columns if c not in ("timestamp", "target")]

    X_train = train_df[train_features]
    y_train = train_df["target"].values
    X_val = val_df[val_features]
    y_val = val_df["target"].values

    # Feature Selection
    k = min(50, X_train.shape[1])  
    X_train_selected, X_val_selected, selected_features = select_features(
        X_train, y_train, X_val, k=k
    )
    logger.info(f"Selected Features: {selected_features}")

    # Initialize TimeSeriesSplit and convert folds to a list
    tscv = TimeSeriesSplit(n_splits=5)
    folds = list(tscv.split(train_df))

    # Train XGBoost with Optuna
    logger.info("Training XGBoost with Optuna...")
    oof_xgb = generate_oof_predictions_xgb(train_df, folds, selected_features, label_col="target", writer=writer)
    logger.info("XGBoost OOF predictions completed.")

    # Train LSTM with Optuna
    logger.info("Training LSTM with Optuna...")
    oof_lstm = generate_oof_predictions_lstm(train_df, folds, selected_features, window_size=5, label_col="target", writer=writer)
    logger.info("LSTM OOF predictions completed.")

    log_memory_usage("After Generating OOF Predictions")

    # ===== Train Stacking Meta-learner =====
    logger.info("===== Training Stacking Meta-learner =====")
    y_combined = train_df["target"].values
    meta_model = train_stacking_model_cv(
        oof_xgb=oof_xgb,
        oof_lstm=oof_lstm,
        y=y_combined,
        model_type="rf",
        n_jobs=2,
        n_splits=5,
        writer=writer
    )
    logger.info("Stacking meta-learner training completed.")

    log_memory_usage("After Training Stacking Meta-learner")

    # ===== Feature Importance Plot =====
    plot_meta_feature_importance(meta_model, feature_names=["XGBoost", "LSTM"])

    logger.info(
        "All done! XGB+LSTM trained with weighted loss functions and Optuna hyperparameter tuning."
    )

    # ===== Close TensorBoard SummaryWriter =====
    writer.close()
    logger.info("Closed TensorBoard SummaryWriter.")

    # Stop profiler
    profiler.disable()
    profile_path = "results/profile.out"
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.dump_stats(profile_path)
    logger.info(f"Profiling stats saved to {profile_path}")

######################################################################
# Run the Main Function
######################################################################

if __name__ == "__main__":
    main()