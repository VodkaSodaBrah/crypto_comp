import os
import pandas as pd
import numpy as np
import optuna  

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from xgboost import DMatrix, train as xgb_train
# from imblearn.over_sampling import RandomOverSampler

from src.features.feature_engineering import add_technical_indicators_inline

print("DEBUG: Optuna imported successfully!")

###############################################################################
# 1. Baseline XGBoost
###############################################################################
def train_xgboost(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/xgb_model.json"
):
    """
    Baseline XGBoost training without Optuna.
    Includes 'timestamp' in features by default now.
    """
    print("[INFO] Running baseline XGBoost (no Optuna).")

    # 1. Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # 2. Keep 'timestamp' as a feature.
    # Include 'timestamp' as a feature during training
    non_feature_cols = ["target"]  # Do NOT drop 'timestamp'

    X_train = train_df.drop(columns=non_feature_cols, errors="ignore")
    y_train = train_df["target"]
    X_val = val_df.drop(columns=non_feature_cols, errors="ignore")
    y_val = val_df["target"]

    # Convert 'timestamp' to a numeric representation (e.g., timestamp to epoch time)
    X_train["timestamp"] = pd.to_datetime(X_train["timestamp"]).astype(int) / 10**9
    X_val["timestamp"] = pd.to_datetime(X_val["timestamp"]).astype(int) / 10**9

    # Debugging: Check feature sample after conversion
    print("[DEBUG] Training features sample with 'timestamp':")
    print(X_train.head())

    # Debug: Print target distribution
    print("[XGBoost DEBUG] Training target distribution:")
    print(y_train.value_counts(normalize=True))
    print("[XGBoost DEBUG] Validation target distribution:")
    print(y_val.value_counts(normalize=True))

    # Debug: Print sample of feature data
    print("[XGBoost DEBUG] Training feature sample:")
    print(X_train.head())

    # 3. Convert all columns to numeric
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4. Create DMatrices
    dtrain = DMatrix(X_train, label=y_train)
    dval   = DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1
    }

    # Train XGBoost
    eval_list = [(dtrain, "train"), (dval, "val")]
    model = xgb_train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=eval_list,
        early_stopping_rounds=10
    )

    # Evaluate
    best_iter = model.best_iteration
    val_preds = model.predict(dval, iteration_range=(0, best_iter + 1))
    val_preds_bin = (val_preds > 0.5).astype(int)
    score = f1_score(y_val, val_preds_bin, average="macro")
    print(f"[XGBoost DMatrix] Validation Macro-F1: {score:.4f}")

    # Save
    model.save_model(model_path)
    print(f"[XGBoost DMatrix] Model saved to {model_path}")


###############################################################################
# 2. XGBoost with Optuna
###############################################################################
def train_xgboost_with_optuna(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/xgb_optuna_model.json",
    num_trials=200
):
    """
    XGBoost training with Optuna hyperparameter search.
    Searches 'num_trials' times to find the best hyperparams.
    Also keeps 'timestamp' as a feature.
    """
    print(f"[INFO] Running XGBoost with Optuna for hyperparam tuning. Trials={num_trials}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Keep 'timestamp' as a feature:
    non_feature_cols = ["target"]
    X_train = train_df.drop(columns=non_feature_cols, errors="ignore")
    y_train = train_df["target"]
    X_val = val_df.drop(columns=non_feature_cols, errors="ignore")
    y_val = val_df["target"]

    # Debug: target distribution
    print("[XGBoost+Optuna DEBUG] Training target distribution:")
    print(y_train.value_counts(normalize=True))
    print("[XGBoost+Optuna DEBUG] Validation target distribution:")
    print(y_val.value_counts(normalize=True))

    # Debug: feature sample
    print("[XGBoost+Optuna DEBUG] Training feature sample:")
    print(X_train.head())

    # Convert to numeric
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    dtrain = DMatrix(X_train, label=y_train)
    dval   = DMatrix(X_val, label=y_val)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 0.01, 10.0),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        }

        model = xgb_train(
            params=params,
            dtrain=dtrain,
            num_boost_round=200,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        best_iter = model.best_iteration
        val_preds = model.predict(dval, iteration_range=(0, best_iter + 1))
        val_preds_bin = (val_preds > 0.5).astype(int)
        return f1_score(y_val, val_preds_bin, average="macro")

    # Run study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    print(f"[Optuna] Best hyperparameters: {study.best_params}")
    print(f"[Optuna] Best Macro-F1 Score: {study.best_value:.4f}")

    # Retrain with best params
    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"
    best_params["verbosity"] = 1

    final_model = xgb_train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    best_iter = final_model.best_iteration
    val_preds = final_model.predict(dval, iteration_range=(0, best_iter + 1))
    val_preds_bin = (val_preds > 0.5).astype(int)
    final_score = f1_score(y_val, val_preds_bin, average="macro")

    print(f"[Optuna XGBoost] Final Validation Macro-F1: {final_score:.4f}")
    final_model.save_model(model_path)
    print(f"[Optuna XGBoost] Optimized model saved to {model_path}")


###############################################################################
# 3. Standard LSTM + Dropout
###############################################################################
class CryptoDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        # Keep 'timestamp' in the features if desired
        self.features = df.drop("target", axis=1).values
        self.labels = df["target"].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.0, num_layers=1, num_classes=2):
        super().__init__()
        """
        A stacked LSTM if num_layers > 1.
        """
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        We assume seq_len=1 or a short multi-step for each sample.
        """
        out, (hn, cn) = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits


def train_lstm(
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    model_path="results/models/lstm_model.pth",
    epochs=5,
    batch_size=128,
    lr=1e-3,
    hidden_dim=64,
    dropout=0.0,
    num_layers=1
):
    """
    Baseline LSTM training with optional dropout and stacking (num_layers).
    No hyperparameter search here by default.
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Inline feature engineering
    train_df = add_technical_indicators_inline(train_df, window=5)
    val_df = add_technical_indicators_inline(val_df, window=5)

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    # Debug: Print target distribution
    print("[LSTM DEBUG] Training target distribution:")
    print(train_df["target"].value_counts(normalize=True))
    print("[LSTM DEBUG] Validation target distribution:")
    print(val_df["target"].value_counts(normalize=True))

    # Debug: Print sample data
    print("[LSTM DEBUG] Training data sample (before numeric conversion):")
    print(train_df.head())

    # Convert columns to numeric
    train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    val_df   = val_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("[LSTM DEBUG] Training data sample after numeric conversion:")
    print(train_df.head())

    # Create datasets & loaders
    train_dataset = CryptoDataset(train_df)
    val_dataset   = CryptoDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    input_dim = train_df.drop("target", axis=1).shape[1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.unsqueeze(1).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.unsqueeze(1).to(device)
                y_val = y_val.to(device)
                logits = model(x_val)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_val.cpu().numpy())

        val_score = f1_score(val_labels, val_preds, average='macro')
        print(f"[LSTM] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val F1: {val_score:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[LSTM] Model saved to {model_path}")


###############################################################################
# 4. Multi-Step (Sliding Window) LSTM
###############################################################################
class CryptoSlidingWindowDataset(Dataset):
    """
    Creates sequences of length 'window_size' from the data, with the label
    being the next step's target or the last of the window (depending on approach).
    """
    def __init__(self, df, window_size=5):
        super().__init__()
        self.window_size = window_size

        # Sort by timestamp if needed, ensuring chronological order
        if 'timestamp' in df.columns:
            df = df.sort_values("timestamp")

        self.targets = df["target"].values

        # Drop non-feature columns
        feature_cols = [c for c in df.columns if c not in ["timestamp", "target"]]
        feature_array = df[feature_cols].values

        self.X, self.y = [], []
        for i in range(len(feature_array) - window_size):
            seq = feature_array[i:i + window_size]
            label = self.targets[i + window_size - 1]
            self.X.append(seq)
            self.y.append(label)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm_multistep(
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    model_path="results/models/lstm_multistep.pth",
    window_size=5,
    hidden_dim=64,
    dropout=0.0,
    num_layers=1,
    epochs=10,
    batch_size=128,
    lr=1e-3
):
    """
    Multi-step LSTM training using a sliding window approach.
    Stacked LSTM + dropout are supported.
    """
    print(f"[INFO] Running multi-step LSTM with window_size={window_size}")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_df = add_technical_indicators_inline(train_df, window=5)
    val_df   = add_technical_indicators_inline(val_df, window=5)

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    val_df   = val_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    train_dataset = CryptoSlidingWindowDataset(train_df, window_size=window_size)
    val_dataset   = CryptoSlidingWindowDataset(val_df,   window_size=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sample_seq, _ = train_dataset[0]
    input_dim = sample_seq.shape[1]

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(seq_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seq_batch, y_batch in val_loader:
                seq_batch = seq_batch.to(device)
                y_batch   = y_batch.to(device)
                logits = model(seq_batch)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_score = f1_score(val_labels, val_preds, average="macro")
        print(f"[LSTM Multistep] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_score:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[LSTM Multistep] Model saved to {model_path}")


###############################################################################
# 5. Main Script Execution
###############################################################################
if __name__ == "__main__":
    # Switches for XGBoost:
    use_optuna_for_xgb = True   # If True -> train_xgboost_with_optuna, else train_xgboost

    # Switches for LSTM:
    use_multistep_lstm   = True   # If True -> multi-step approach
    use_optuna_for_lstm  = False  # If True -> train_lstm_with_optuna

    # 1. XGBoost
    if use_optuna_for_xgb:
        print("DEBUG: Using Optuna for XGBoost training.")  # Add this
        train_xgboost_with_optuna(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/xgb_optuna_model.json",
            num_trials=200
        )
    else:
        print("DEBUG: Using baseline XGBoost training.")  # Add this
        train_xgboost(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/xgb_model.json"
        )

    # 2. LSTM (baseline or multi-step)
    if use_optuna_for_lstm:
        pass
    else:
        if use_multistep_lstm:
            train_lstm_multistep(
                train_csv="data/intermediate/train.csv",
                val_csv="data/intermediate/val.csv",
                model_path="results/models/lstm_multistep.pth",
                window_size=5,   
                hidden_dim=64,   
                dropout=0.1,     
                num_layers=2,    
                epochs=10,       
                batch_size=128,
                lr=1e-3
            )
        else:
            train_lstm(
                train_csv="data/intermediate/train.csv",
                val_csv="data/intermediate/val.csv",
                model_path="results/models/lstm_model.pth",
                epochs=5,
                batch_size=128,
                lr=1e-3,
                hidden_dim=64,
                dropout=0.0,
                num_layers=1
            )