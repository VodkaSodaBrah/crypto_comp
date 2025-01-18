# src/models/train_model.py

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import gc  # Garbage collection

import optuna
from optuna.pruners import MedianPruner

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import KFold

import xgboost as xgb
from xgboost import DMatrix, train as xgb_train

# For data balancing
from imblearn.over_sampling import SMOTE

import cProfile, pstats
import concurrent.futures
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################################################
# Utility: Create Necessary Directories
########################################################################

def create_directories():
    """
    Create necessary directories if they don't exist.
    """
    directories = ["results/models", "results"]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory checked/created: {dir_path}")

########################################################################
# Additional Metric Printing
########################################################################

def evaluate_and_print_metrics(y_true, y_prob, prefix="Model"):
    """
    Computes and prints confusion matrix, recall, precision, F1, and ROC AUC for the given predictions.
    Also prints top-level info about ROC curve and precision-recall curve thresholds.
    """
    # 1. Binary predictions at 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # 2. Metrics
    cm = confusion_matrix(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro")

    # If classes are [0,1] and both present, compute ROC AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    logger.info(f"[{prefix}] Confusion Matrix:\n{cm}")
    logger.info(f"[{prefix}] Recall={rec:.4f}  Precision={prec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # 3. ROC Curve (truncated display)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    logger.info(
        f"[{prefix}] ROC curve points (showing first 5):\n"
        f"   FPR={fpr[:5]}, TPR={tpr[:5]}, TH={roc_thresh[:5]} (truncated)"
    )
    
    # 4. Precision-Recall Curve (truncated display)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_prob)
    logger.info(
        f"[{prefix}] Precision-Recall points (showing first 5):\n"
        f"   Precision={pr_prec[:5]}, Recall={pr_rec[:5]}, TH={pr_thresh[:5]} (truncated)"
    )

########################################################################
# Utility: Clean & Check
########################################################################

def clean_dataframe(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Replace inf/-inf with fill_value, fill NaNs with fill_value.
    """
    df = df.replace([np.inf, -np.inf], fill_value)
    df = df.fillna(fill_value)
    return df

def check_infinite_values(df, name: str):
    numeric_df = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric_df).all().all():
        inf_count = np.isinf(numeric_df).sum().sum()
        logger.warning(f"{name} contains {inf_count} inf/-inf values in numeric columns.")
    else:
        logger.info(f"No infinite values found in numeric columns of {name}.")

########################################################################
# Data Balancing (SMOTE)
########################################################################

def rebalance_data(df):
    """
    Physically oversample the minority class in the training DataFrame using SMOTE.
    Leaves the validation/test sets untouched.
    """
    # Drop 'timestamp' if present, keep 'target' aside
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain 'target' for SMOTE balancing.")
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]

    X = df[feature_cols].copy()
    y = df["target"].copy()

    logger.info("Applying SMOTE to rebalance classes in the training set...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"SMOTE done. Original shape={X.shape}, new shape={X_res.shape}.")

    # Rebuild the DataFrame
    balanced_df = pd.DataFrame(X_res, columns=feature_cols)
    balanced_df["target"] = y_res.values

    return balanced_df

########################################################################
# 1. CryptoSlidingWindowDataset
########################################################################

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

########################################################################
# 2. LSTMAttnClassifier (with Attention)
########################################################################

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
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, num_layers=1,
                 num_classes=2, use_attention=True, bidirectional=False):
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

########################################################################
# 3. XGBoost + Optuna (Persistent Study + MedianPruner)
########################################################################

def train_xgb_optuna(X_train, y_train, X_val, y_val, num_boost_round=200, n_jobs=4):
    """
    Train an XGBoost model with Optuna hyperparam search (MedianPruner).
    Continues an existing study if available. 
    Uses SMOTE-based rebalancing (optional) + scale_pos_weight for class imbalance.
    """
    # Pruner
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    # Compute scale_pos_weight
    num_pos = sum(y_train == 1)
    num_neg = sum(y_train == 0)
    spw = num_neg / max(num_pos, 1)

    study_name = "xgb_study"
    storage_path = "sqlite:///results/optuna_xgb.db"

    try:
        # Attempt to load an existing XGB study
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        logger.info(f"Loaded existing Optuna study: {study_name}")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="maximize",
            pruner=pruner
        )
        logger.info(f"Created new Optuna study: {study_name}")

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "eta": trial.suggest_float("eta", 1e-4, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 20.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 20.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 15),
            "scale_pos_weight": spw,
            "n_jobs": n_jobs,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        val_preds = model.predict(dval, iteration_range=(0, model.best_iteration+1))
        score = f1_score(y_val, (val_preds > 0.5).astype(int), average="macro")

        # Prune if trial is unpromising
        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return score

    study.optimize(objective, n_trials=1000, n_jobs=n_jobs)
    logger.info(f"XGB Optuna best params: {study.best_params}")
    logger.info(f"XGB Optuna best F1: {study.best_value:.4f}")

    # Retrain final model
    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1,
        "scale_pos_weight": spw,
        "n_jobs": n_jobs
    })

    dtrain_final = xgb.DMatrix(X_train, label=y_train)
    dval_final = xgb.DMatrix(X_val, label=y_val)
    final_model = xgb.train(
        best_params,
        dtrain_final,
        num_boost_round=2000,
        evals=[(dval_final, "val")],
        early_stopping_rounds=30,
        verbose_eval=True
    )

    val_preds_final = final_model.predict(dval_final, iteration_range=(0, final_model.best_iteration+1))
    evaluate_and_print_metrics(y_val, val_preds_final, prefix="XGBoost")

    return final_model

########################################################################
# 4. LSTM + Optuna (Persistent Study + MedianPruner)
########################################################################

def train_lstm_optuna(
    train_df, val_df,
    window_size=5,
    base_epochs=3,
    final_epochs=5,
    batch_size=32,  # Reduced batch size
    accumulation_steps=4,  # For gradient accumulation
    hidden_dim_range=(32, 96),  # Adjusted to smaller range
    use_attention=True,
    resume_from_best=True
):
    """
    Train LSTM with attention + Optuna, using MedianPruner for pruning.
    Resumes from existing 'lstm_study' if available. 
    Optionally start from the best trial's params (resume_from_best).
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"LSTM Optuna on device: {device}")

    # Median pruner
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study_name = "lstm_study"
    storage_path = "sqlite:///results/optuna_lstm.db"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
        logger.info(f"Loaded existing Optuna study: {study_name}")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="maximize",
            pruner=pruner
        )
        logger.info(f"Created new Optuna study: {study_name}")

    if resume_from_best and len(study.trials) > 0:
        best_trial = study.best_trial
        logger.info(f"Starting from best trial parameters: {best_trial.params}")

        def fixed_params_trial(_trial):
            # Fix best params as a trial
            return {
                "hidden_dim": best_trial.params["hidden_dim"],
                "dropout": best_trial.params["dropout"],
                "num_layers": best_trial.params["num_layers"],
                "lr": best_trial.params["lr"],
                "bidirectional": best_trial.params.get("bidirectional", False),
            }

        study.enqueue_trial(fixed_params_trial(None))

    train_dataset = CryptoSlidingWindowDataset(train_df, window_size=window_size, has_target=True)
    val_dataset   = CryptoSlidingWindowDataset(val_df,   window_size=window_size, has_target=True)

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", hidden_dim_range[0], hidden_dim_range[1], step=32)
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])

        model = LSTMAttnClassifier(
            input_dim=train_dataset[0][0].shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            use_attention=use_attention,
            bidirectional=bidirectional
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        loader_val   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Gradient Accumulation
        optimizer.zero_grad()

        # Base training with gradient accumulation
        for ep in range(base_epochs):
            model.train()
            for i, (Xb, yb) in enumerate(loader_train):
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=(device.type=="mps")):
                    logits = model(Xb)
                    loss = criterion(logits, yb)
                    loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader_train):
                    optimizer.step()
                    optimizer.zero_grad()

        # Evaluate on val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in loader_val:
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=(device.type=="mps")):
                    out = model(Xb)
                    preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")

        # Cleanup to free memory
        del model
        del optimizer
        torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()
        gc.collect()

        # Report & prune
        trial.report(val_f1, step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_f1

    study.optimize(objective, n_trials=50, n_jobs=1)  # n_jobs=1 inside each process
    logger.info(f"LSTM Optuna best params: {study.best_params}")
    logger.info(f"LSTM Optuna best F1: {study.best_value:.4f}")

    # Retrain final model
    hp = study.best_params
    model = LSTMAttnClassifier(
        input_dim=train_dataset[0][0].shape[1],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
        num_layers=hp["num_layers"],
        use_attention=use_attention,
        bidirectional=hp.get("bidirectional", False)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    loader_val   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for ep in range(final_epochs):
        model.train()
        for i, (Xb, yb) in enumerate(loader_train):
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type=="mps")):
                logits = model(Xb)
                loss = criterion(logits, yb)
                loss = loss / accumulation_steps
            loss.backward()
            optimizer.step()

        # Evaluate after each epoch
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for Xb, yb in loader_val:
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=(device.type=="mps")):
                    out = model(Xb)
                    probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        evaluate_and_print_metrics(np.array(val_labels), np.array(val_probs), prefix=f"LSTM(Epoch={ep+1})")

    # Save the final model
    torch.save(model.state_dict(), "results/models/lstm_optuna_final.pth")
    logger.info("Final LSTM model saved.")

    # Cleanup
    del model
    del optimizer
    torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()
    gc.collect()

    return model, study.best_params

########################################################################
# 5. Out-of-Fold (OOF) Predictions for Stacking
########################################################################

def generate_oof_predictions_xgb(df, folds, features, label_col="target"):
    """
    Generate Out-of-Fold (OOF) predictions using XGBoost for stacking.

    Args:
        df (pd.DataFrame): The combined training and validation DataFrame.
        folds (list): List of (train_idx, val_idx) tuples from KFold.
        features (list): List of feature column names.
        label_col (str): Name of the target column.

    Returns:
        np.ndarray: OOF predictions for each instance.
    """
    oof_preds = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"XGB Fold {fold + 1}/{len(folds)}")
        
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx][label_col]
        X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx][label_col]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Retrieve the best parameters from the Optuna study
        study = optuna.load_study(study_name="xgb_study", storage="sqlite:///results/optuna_xgb.db")
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "scale_pos_weight": y_train.value_counts()[0] / y_train.value_counts()[1],
            "n_jobs": 4,  # Adjust based on your system
        })
        
        # Train the model
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        
        # Predict on validation set
        preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof_preds[val_idx] = preds
        
        logger.info(f"Fold {fold + 1} completed. Best iteration: {model.best_iteration}")
    
    return oof_preds

def generate_oof_predictions_lstm(df, folds, window_size=5, label_col="target"):
    """
    Generate Out-of-Fold (OOF) predictions using LSTM for stacking.

    Args:
        df (pd.DataFrame): The combined training and validation DataFrame.
        folds (list): List of (train_idx, val_idx) tuples from KFold.
        window_size (int): The size of the sliding window.
        label_col (str): Name of the target column.

    Returns:
        np.ndarray: OOF predictions for each instance.
    """
    oof_preds = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"LSTM Fold {fold + 1}/{len(folds)}")
        
        train_subset = df.iloc[train_idx].reset_index(drop=True)
        val_subset = df.iloc[val_idx].reset_index(drop=True)
        
        # Load the best parameters from the Optuna study
        study = optuna.load_study(study_name="lstm_study", storage="sqlite:///results/optuna_lstm.db")
        best_params = study.best_params.copy()
        
        model = LSTMAttnClassifier(
            input_dim=train_subset[features].shape[1],
            hidden_dim=best_params["hidden_dim"],
            dropout=best_params["dropout"],
            num_layers=best_params["num_layers"],
            use_attention=best_params["use_attention"],
            bidirectional=best_params.get("bidirectional", False)
        ).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        
        train_dataset = CryptoSlidingWindowDataset(train_subset, window_size=window_size, has_target=True)
        val_dataset = CryptoSlidingWindowDataset(val_subset, window_size=window_size, has_target=True)
        
        loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)  # Reduced batch size
        loader_val = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)  # Reduced batch size
        
        # Train the model (base epochs)
        model.train()
        optimizer.zero_grad()
        for epoch in range(3):  # Base epochs
            for i, (X_batch, y_batch) in enumerate(loader_train):
                X_batch, y_batch = X_batch.to(model.lstm.weight_ih_l0.device, non_blocking=True), y_batch.to(model.lstm.weight_ih_l0.device, non_blocking=True)
                with torch.autocast(device_type=model.lstm.weight_ih_l0.device.type, enabled=(model.lstm.weight_ih_l0.device.type=="mps")):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss = loss / 4  # Accumulation steps
                loss.backward()
                
                if (i + 1) % 4 == 0 or (i + 1) == len(loader_train):
                    optimizer.step()
                    optimizer.zero_grad()

        # Evaluate on validation set
        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader_val:
                X_batch = X_batch.to(model.lstm.weight_ih_l0.device, non_blocking=True)
                with torch.autocast(device_type=model.lstm.weight_ih_l0.device.type, enabled=(model.lstm.weight_ih_l0.device.type=="mps")):
                    out = model(X_batch)
                    probs = torch.softmax(out, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
        
        oof_preds[val_idx] = all_preds
        logger.info(f"Fold {fold + 1} completed.")
        
        # Cleanup to free memory
        del model
        del optimizer
        torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()
        gc.collect()
    
    return oof_preds

########################################################################
# 6. Train a Stacking Meta-learner
########################################################################

def train_stacking_model(oof_xgb, oof_lstm, y, model_type="xgb", n_jobs=4):
    """
    Train a stacking meta-learner using OOF predictions.

    Args:
        oof_xgb (np.ndarray): OOF predictions from XGBoost.
        oof_lstm (np.ndarray): OOF predictions from LSTM.
        y (np.ndarray): True target values.
        model_type (str): Type of meta-model ('xgb' or 'lr').
        n_jobs (int): Number of parallel jobs.

    Returns:
        object: Trained meta-model.
    """
    # Combine OOF predictions as features
    X_meta = np.vstack((oof_xgb, oof_lstm)).T
    y_meta = y
    
    logger.info(f"Training stacking meta-learner using {model_type.upper()}")

    if model_type.lower() == "xgb":
        dmeta = xgb.DMatrix(X_meta, label=y_meta)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "eta": 0.1,
            "max_depth": 3,
            "n_jobs": n_jobs,
        }
        meta_model = xgb.train(
            params,
            dmeta,
            num_boost_round=1000,
            evals=[(dmeta, "meta")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
    elif model_type.lower() == "lr":
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression(max_iter=1000, n_jobs=n_jobs)
        meta_model.fit(X_meta, y_meta)
    else:
        raise ValueError("Unsupported meta-model type. Choose 'xgb' or 'lr'.")

    # Evaluate meta-model
    if model_type.lower() == "xgb":
        meta_preds = meta_model.predict(dmeta)
    else:
        meta_preds = meta_model.predict_proba(X_meta)[:, 1]
    
    evaluate_and_print_metrics(y_meta, meta_preds, prefix="Meta-learner")

    # Save the meta-model
    if model_type.lower() == "xgb":
        meta_model.save_model("results/models/stacking_meta_model.json")
    elif model_type.lower() == "lr":
        joblib.dump(meta_model, "results/models/stacking_meta_model.pkl")
    
    return meta_model

########################################################################
# 7. Top-Level Wrapper Functions
########################################################################

def run_xgb_wrapper(train_csv, val_csv, train_features, val_features):
    """
    Wrapper function to train XGBoost. Intended to be called by ProcessPoolExecutor.
    """
    logger.info("Starting XGBoost training in child process...")
    try:
        # Load data within the process
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        train_df = clean_dataframe(train_df)
        val_df = clean_dataframe(val_df)
        train_df = rebalance_data(train_df)
        
        # Ensure feature sets match
        if set(train_features) != set(val_features):
            missing_in_val = set(train_features) - set(val_features)
            missing_in_train = set(val_features) - set(train_features)
            logger.warning(
                f"Feature mismatch! Train missing in val: {missing_in_val}, "
                f"Val missing in train: {missing_in_train}"
            )
        else:
            logger.info("Train/Val feature sets match within child process.")
        
        # Train XGBoost
        xgb_model = train_xgb_optuna(train_df[train_features].values, train_df["target"].values,
                                     val_df[val_features].values, val_df["target"].values,
                                     num_boost_round=200, n_jobs=4)
        xgb_model.save_model("results/models/xgb_optuna_final.json")
        logger.info("XGBoost training completed in child process.")
        return "XGBoost Training Completed"
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        raise e

def run_lstm_wrapper(train_csv, val_csv, window_size=5, base_epochs=3, final_epochs=5,
                    batch_size=32, accumulation_steps=4, hidden_dim_range=(32, 96),
                    use_attention=True, resume_from_best=True):
    """
    Wrapper function to train LSTM. Intended to be called by ProcessPoolExecutor.
    """
    logger.info("Starting LSTM training in child process...")
    try:
        # Load data within the process
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        train_df = clean_dataframe(train_df)
        val_df = clean_dataframe(val_df)
        train_df = rebalance_data(train_df)
        
        # Train LSTM
        lstm_model, lstm_params = train_lstm_optuna(
            train_df, val_df, window_size=window_size,
            base_epochs=base_epochs, final_epochs=final_epochs, batch_size=batch_size,
            accumulation_steps=accumulation_steps, hidden_dim_range=hidden_dim_range,
            use_attention=use_attention, resume_from_best=resume_from_best
        )
        torch.save(lstm_model.state_dict(), "results/models/lstm_optuna_final.pth")
        logger.info("LSTM training completed in child process.")
        return "LSTM Training Completed"
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        raise e

########################################################################
# 8. Main Routine (with cProfile)
########################################################################

def main():
    # Create necessary directories
    create_directories()

    # Start cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    logger.info("Loading train & val data")
    train_csv = "data/intermediate/train_fe.csv"
    val_csv   = "data/intermediate/val_fe.csv"
    train_df  = pd.read_csv(train_csv)
    val_df    = pd.read_csv(val_csv)

    # Clean & check
    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)
    check_infinite_values(train_df, "train_df")
    check_infinite_values(val_df,   "val_df")

    # Rebalance only the training set
    train_df = rebalance_data(train_df)

    # Verify feature sets after rebalancing
    train_features = [c for c in train_df.columns if c not in ("timestamp", "target")]
    val_features   = [c for c in val_df.columns   if c not in ("timestamp", "target")]
    if set(train_features) != set(val_features):
        missing_in_val = set(train_features) - set(val_features)
        missing_in_train = set(val_features) - set(train_features)
        logger.warning(
            f"Feature mismatch! Train missing in val: {missing_in_val}, "
            f"Val missing in train: {missing_in_train}"
        )
    else:
        logger.info("Train/Val feature sets match.")

    # ===== XGB + Optuna (with Pruning) and LSTM + Optuna (with Pruning) =====
    # Define the file paths to pass to child processes
    train_processed_csv = train_csv  # Assuming SMOTE is done in main and saved to train_fe.csv
    val_processed_csv = val_csv        # Assuming validation data is unchanged

    # Execute both studies in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_xgb = executor.submit(run_xgb_wrapper, train_processed_csv, val_processed_csv, train_features, val_features)
        future_lstm = executor.submit(run_lstm_wrapper, train_processed_csv, val_processed_csv)

        # Wait for both to complete
        try:
            xgb_result = future_xgb.result()
            lstm_result = future_lstm.result()
        except Exception as e:
            logger.error(f"Error during parallel training: {e}")
            sys.exit(1)

    # ===== Generating OOF Predictions for Stacking =====
    logger.info("===== Generating OOF for Stacking =====")
    combined_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    features = [c for c in combined_df.columns if c not in ("timestamp","target")]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(combined_df))

    # Generate OOF predictions
    logger.info("Generating OOF predictions for XGBoost...")
    oof_xgb = generate_oof_predictions_xgb(combined_df, folds, features, label_col="target")
    
    logger.info("Generating OOF predictions for LSTM...")
    oof_lstm = generate_oof_predictions_lstm(combined_df, folds, window_size=5, label_col="target")

    # ===== Train Stacking Meta-learner =====
    logger.info("===== Training Stacking Meta-learner =====")
    y_combined = combined_df["target"].values
    meta_model = train_stacking_model(oof_xgb, oof_lstm, y_combined, model_type="xgb", n_jobs=4)

    logger.info(
        "All done! XGB+LSTM + attention trained with SMOTE rebalancing, median pruning, "
        "and cProfile profiling."
    )

    # Stop profiler
    profiler.disable()
    profile_path = "results/profile.out"
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.dump_stats(profile_path)
    logger.info(f"Profiling stats saved to {profile_path}")

if __name__ == "__main__":
    main()