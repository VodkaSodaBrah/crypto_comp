import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
import optuna

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

from xgboost import DMatrix, train as xgb_train
from src.features.feature_engineering import add_technical_indicators_inline

if __name__ == "__main__":
    print("DEBUG: Optuna imported successfully!")

###############################################################################
# Utility Classes & Functions
###############################################################################
class CryptoSlidingWindowDataset(Dataset):
    """
    Creates sequences of length 'window_size' from the data, 
    with the label being the last item in the window (or next step).
    """
    def __init__(self, df, window_size=5):
        super().__init__()
        self.window_size = window_size

        # Sort by timestamp to maintain chronological order if timestamp is included
        if 'timestamp' in df.columns:
            df = df.sort_values("timestamp")

        self.targets = df["target"].values

        # Build feature array excluding 'target'
        feature_cols = [c for c in df.columns if c not in ["timestamp", "target"]]
        feature_array = df[feature_cols].values

        self.X, self.y = [], []
        for i in range(len(feature_array) - window_size):
            seq = feature_array[i : i + window_size]
            label = self.targets[i + window_size - 1]  
            self.X.append(seq)
            self.y.append(label)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


###############################################################################
# Attention Module (Bahdanau)
###############################################################################
class BahdanauAttention(nn.Module):
    """
    Simple Bahdanau-style additive attention for demonstration.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch, hidden_dim)
        encoder_outputs: (batch, seq_len, hidden_dim)
        returns: 
           - context (batch, hidden_dim)
           - attention_weights (batch, seq_len, 1)
        """
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = self.v(
            torch.tanh(self.W(encoder_outputs) + self.U(decoder_hidden_expanded))
        )  # (batch, seq_len, 1)

        attention_weights = torch.softmax(energy, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(encoder_outputs * attention_weights, dim=1)  # (batch, hidden_dim)
        return context, attention_weights


###############################################################################
# LSTM with optional Attention
###############################################################################
class LSTMAttnClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        dropout=0.2,
        num_layers=1,
        num_classes=2,
        use_attention=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if self.use_attention:
            self.attention = BahdanauAttention(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)
        """
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)
        decoder_hidden = hn[-1]            # shape: (batch, hidden_dim)

        if self.use_attention:
            context, _ = self.attention(decoder_hidden, lstm_out)
        else:
            context = lstm_out[:, -1, :]  

        context = self.dropout(context)
        logits = self.fc(context)
        return logits


###############################################################################
# 1. Baseline XGBoost (no Optuna)
###############################################################################
def train_xgboost(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/xgb_model.json"
):
    """
    Baseline XGBoost training without Optuna.
    """
    print("[INFO] Running baseline XGBoost (no Optuna).")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    non_feature_cols = ["target"]
    X_train = train_df.drop(columns=non_feature_cols, errors="ignore")
    y_train = train_df["target"]
    X_val = val_df.drop(columns=non_feature_cols, errors="ignore")
    y_val = val_df["target"]

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    dtrain = DMatrix(X_train, label=y_train)
    dval   = DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1
    }

    eval_list = [(dtrain, "train"), (dval, "val")]
    model = xgb_train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=eval_list,
        early_stopping_rounds=10
    )

    best_iter = model.best_iteration
    val_preds = model.predict(dval, iteration_range=(0, best_iter + 1))
    val_preds_bin = (val_preds > 0.5).astype(int)
    score = f1_score(y_val, val_preds_bin, average="macro")
    print(f"[XGBoost DMatrix] Validation Macro-F1: {score:.4f}")

    model.save_model(model_path)
    print(f"[XGBoost DMatrix] Model saved to {model_path}")


###############################################################################
# 2. XGBoost with Optuna
###############################################################################
def train_xgboost_with_optuna(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/xgb_optuna_model.json",
    num_trials=50
):
    """
    XGBoost training with Optuna hyperparameter search.
    """
    print(f"[INFO] Running XGBoost with Optuna for hyperparam tuning. Trials={num_trials}")

    import xgboost as xgb
    from optuna.pruners import MedianPruner

    pruner = MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_study",
        storage="sqlite:///optuna_xgb_study.db",
        load_if_exists=True,
        pruner=pruner
    )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    non_feature_cols = ["target"]
    X_train = train_df.drop(columns=non_feature_cols, errors="ignore")
    y_train = train_df["target"]
    X_val = val_df.drop(columns=non_feature_cols, errors="ignore")
    y_val = val_df["target"]

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "eta": trial.suggest_float("eta", 0.01, 0.5),  
            "max_depth": trial.suggest_int("max_depth", 3, 15),  
            "subsample": trial.suggest_float("subsample", 0.3, 1.0), 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), 
            "lambda": trial.suggest_float("lambda", 0.01, 20.0),
            "alpha": trial.suggest_float("alpha", 0.01, 20.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 15)
    }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
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

    study.optimize(objective, n_trials=num_trials)
    print(f"[Optuna] Best hyperparameters: {study.best_params}")
    print(f"[Optuna] Best Macro-F1 Score: {study.best_value:.4f}")

    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"
    best_params["verbosity"] = 1

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)

    final_model = xgb.train(
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
# 3. Multi-Step LSTM with optional Attention (MPS only)
###############################################################################
def train_lstm_multistep(
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    model_path="results/models/lstm_multistep.pth",
    window_size=5,
    hidden_dim=64,
    dropout=0.2,
    num_layers=1,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    use_attention=False,
    num_workers=4
):
    """
    Multi-step LSTM training (sliding window).
    Optionally with attention and stacking (num_layers).
    Uses MPS if available.
    """
    print(f"[INFO] Running multi-step LSTM with window_size={window_size}, use_attention={use_attention}")

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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    sample_seq, _ = train_dataset[0]
    input_dim = sample_seq.shape[1]

    model = LSTMAttnClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        use_attention=use_attention
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)

            optimizer.zero_grad()
            # Mixed precision for MPS
            with torch.autocast(device_type="mps", enabled=(device != "cpu")):
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
                with torch.autocast(device_type="mps", enabled=(device != "cpu")):
                    logits = model(seq_batch)
                    preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_score = f1_score(val_labels, val_preds, average="macro")
        print(f"[LSTM Multistep] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_score:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[LSTM Multistep] Model saved to {model_path}")


###############################################################################
# 4. Multi-Step LSTM with Optuna (MPS only)
###############################################################################
def train_lstm_multistep_with_optuna(
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    model_path="results/models/lstm_multistep_optuna.pth",
    window_size=5,
    num_trials=20,
    base_epochs=5,
    final_epochs=10,
    batch_size=128,
    num_workers=4
):
    """
    Multi-step LSTM with sliding window + Optuna hyperparameter tuning.
    Continues the same study in a persistent DB, so old param combos aren't retried.
    Uses MPS if available.
    """
    print(f"[INFO] Running multi-step LSTM with Optuna, window_size={window_size}, trials={num_trials}")

    from optuna.pruners import MedianPruner
    pruner = MedianPruner(n_startup_trials=5)

    study = optuna.create_study(
        direction="maximize",
        study_name="lstm_study",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        pruner=pruner
    )

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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    from torch.utils.data import DataLoader

    def objective(trial):
        hidden_dim   = trial.suggest_int("hidden_dim", 32, 128, step=32)
        num_layers   = trial.suggest_int("num_layers", 1, 3)
        dropout      = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        use_attention= trial.suggest_categorical("use_attention", [False, True])

        model = LSTMAttnClassifier(
            input_dim=train_dataset[0][0].shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            use_attention=use_attention
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Basic training loop
        for ep in range(base_epochs):
            model.train()
            for seq_batch, y_batch in train_loader:
                seq_batch = seq_batch.to(device)
                y_batch   = y_batch.to(device)

                optimizer.zero_grad()
                with torch.autocast(device_type="mps", enabled=(device != "cpu")):
                    logits = model(seq_batch)
                    loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seq_batch, y_batch in val_loader:
                seq_batch = seq_batch.to(device)
                y_batch   = y_batch.to(device)
                with torch.autocast(device_type="mps", enabled=(device != "cpu")):
                    logits = model(seq_batch)
                    preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        return f1_score(val_labels, val_preds, average="macro")

    study.optimize(objective, n_trials=num_trials)
    print("[Optuna LSTM] Best Params:", study.best_params)
    print("[Optuna LSTM] Best Value (Macro-F1):", study.best_value)

    hp = study.best_params
    print(f"[Optuna LSTM] Using best hyperparams: {hp}")

    final_model = LSTMAttnClassifier(
        input_dim=train_dataset[0][0].shape[1],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
        num_layers=hp["num_layers"],
        use_attention=hp["use_attention"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=hp["lr"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for ep in range(final_epochs):
        final_model.train()
        for seq_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="mps", enabled=(device != "cpu")):
                logits = final_model(seq_batch)
                loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

    # Final validation
    final_model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for seq_batch, y_batch in val_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)
            with torch.autocast(device_type="mps", enabled=(device != "cpu")):
                logits = final_model(seq_batch)
                preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    final_f1 = f1_score(val_labels, val_preds, average="macro")
    print(f"[Optuna LSTM] Final Retrained Macro-F1: {final_f1:.4f}")
    torch.save(final_model.state_dict(), model_path)
    print(f"[Optuna LSTM] Optimized model saved to {model_path}")


###############################################################################
# 5. Ensemble: Combine XGBoost + LSTM
###############################################################################
def ensemble_predictions(
    test_csv="data/final/test_fe.csv",
    xgb_model_path="results/models/xgb_optuna_model.json",
    lstm_model_path="results/models/lstm_multistep_optuna.pth",
    window_size=5,
    xgb_f1=0.4597,  
    lstm_f1=0.664,  
    output_csv="results/models/ensemble_predictions.csv"
):
    """
    Loads both XGBoost and LSTM models, predicts on test set, 
    combines predictions via weighted average based on validation F1 scores, and writes them to CSV.
    """
    print("[INFO] Running ensemble predictions...")
    import xgboost as xgb

    # 1. Load and preprocess test data
    test_df = pd.read_csv(test_csv)
    test_df = add_technical_indicators_inline(test_df.copy(), window=5)
    test_df.dropna(inplace=True)

    # XGBoost predictions
    non_feature_cols = ["target"]
    X_test_xgb = test_df.drop(columns=non_feature_cols, errors="ignore")
    X_test_xgb = X_test_xgb.apply(pd.to_numeric, errors="coerce").fillna(0)
    dtest = xgb.DMatrix(X_test_xgb)

    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    xgb_preds = xgb_model.predict(dtest)
    xgb_preds_bin = (xgb_preds > 0.5).astype(int)

    # LSTM predictions
    from torch.utils.data import DataLoader
    test_dataset = CryptoSlidingWindowDataset(test_df, window_size=window_size)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sample_seq, _ = test_dataset[0]
    input_dim = sample_seq.shape[1]

    lstm_model = LSTMAttnClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        dropout=0.1,
        num_layers=2,
        use_attention=True
    ).to(device)

    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()

    lstm_preds = []
    with torch.no_grad():
        for seq_batch, _ in test_loader:
            seq_batch = seq_batch.to(device)
            with torch.autocast(device_type="mps", enabled=(device != "cpu")):
                logits = lstm_model(seq_batch)
                preds = torch.argmax(logits, dim=1)
            lstm_preds.extend(preds.cpu().numpy())
    lstm_preds = np.array(lstm_preds)

    # Dynamically calculate weights based on validation F1 scores
    total_f1 = xgb_f1 + lstm_f1
    xgb_weight = xgb_f1 / total_f1
    lstm_weight = lstm_f1 / total_f1

    print(f"[INFO] Ensemble Weights | XGBoost: {xgb_weight:.2f}, LSTM: {lstm_weight:.2f}")

    # Weighted averaging approach
    min_len = min(len(xgb_preds_bin), len(lstm_preds))
    ensemble_preds = np.round(
        xgb_weight * xgb_preds_bin[:min_len] + lstm_weight * lstm_preds[:min_len]
    ).astype(int)

    output_df = pd.DataFrame({
        "row_id": range(min_len),
        "target": ensemble_preds
    })
    output_df.to_csv(output_csv, index=False)
    print(f"Ensemble predictions saved to {output_csv}")

    return ensemble_preds


###############################################################################
# 6. Main Script Execution
###############################################################################
if __name__ == "__main__":
    use_optuna_for_xgb = True
    use_multistep_lstm = True
    use_optuna_for_lstm = True
    use_attention_lstm = False

    # 1. XGBoost
    if use_optuna_for_xgb:
        train_xgboost_with_optuna(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/xgb_optuna_model.json",
            num_trials=250  
        )
    else:
        train_xgboost(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/xgb_model.json"
        )

    # 2. LSTM
    if use_optuna_for_lstm:
        train_lstm_multistep_with_optuna(
            train_csv="data/intermediate/train.csv",
            val_csv="data/intermediate/val.csv",
            model_path="results/models/lstm_multistep_optuna.pth",
            window_size=5,
            num_trials=100,  
            base_epochs=5,
            final_epochs=10,
            batch_size=128,
            num_workers=4
        )
    else:
        if use_multistep_lstm:
            train_lstm_multistep(
                train_csv="data/intermediate/train.csv",
                val_csv="data/intermediate/val.csv",
                model_path="results/models/lstm_multistep.pth",
                window_size=5,
                hidden_dim=128,
                dropout=0.2,
                num_layers=2,
                epochs=10,
                batch_size=128,
                lr=1e-3,
                use_attention=use_attention_lstm,
                num_workers=4
            )
        else:
            print("You selected single-step LSTM approach. (Not fully shown in this snippet.)")

    # 3. Ensemble predictions (XGBoost + LSTM) on final test data
    ensemble_predictions(
        test_csv="data/final/test_fe.csv",
        xgb_model_path="results/models/xgb_optuna_model.json",
        lstm_model_path="results/models/lstm_multistep_optuna.pth",
        window_size=5,
        output_csv="results/models/ensemble_predictions.csv"
    )

    print("Pipeline complete!")