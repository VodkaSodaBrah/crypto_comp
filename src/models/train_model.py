import os
import sys
import pandas as pd
import numpy as np
import optuna

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # <-- TensorBoard integration
from sklearn.metrics import f1_score

import xgboost as xgb
from xgboost import DMatrix, train as xgb_train

from src.features.feature_engineering import add_technical_indicators_inline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
    def __init__(self, df, window_size=5, has_target=True):
        super().__init__()
        self.window_size = window_size
        self.has_target = has_target

        # Sort by timestamp to maintain chronological order if timestamp is included
        if 'timestamp' in df.columns:
            df = df.sort_values("timestamp")

        if self.has_target:
            if 'target' not in df.columns:
                raise ValueError("DataFrame must contain 'target' column when has_target=True.")
            self.targets = df["target"].values

        # Build feature array excluding 'target' and 'timestamp'
        feature_cols = [c for c in df.columns if c not in ["timestamp", "target"]]
        feature_array = df[feature_cols].values

        self.X = []
        self.y = [] if self.has_target else None
        for i in range(len(feature_array) - window_size):
            seq = feature_array[i : i + window_size]
            if self.has_target:
                label = self.targets[i + window_size - 1]  
                self.y.append(label)
            self.X.append(seq)

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
# 1. Baseline XGBoost (no Optuna) + TensorBoard
###############################################################################
def train_xgboost(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/xgb_model.json"
):
    """
    Baseline XGBoost training without Optuna, with TensorBoard logging.
    """
    print("[INFO] Running baseline XGBoost (no Optuna).")
    writer = SummaryWriter(log_dir="runs/xgboost_baseline_experiment")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Exclude 'timestamp' from features
    non_feature_cols = ["target", "timestamp"]
    X_train = train_df.drop(columns=non_feature_cols, errors="ignore")
    y_train = train_df["target"]
    X_val = val_df.drop(columns=non_feature_cols, errors="ignore")
    y_val = val_df["target"]

    # Debugging: Print feature lists
    train_features = X_train.columns.tolist()
    val_features = X_val.columns.tolist()
    print(f"[DEBUG] XGBoost Train features ({len(train_features)}): {train_features}")
    print(f"[DEBUG] XGBoost Validation features ({len(val_features)}): {val_features}")
    assert set(train_features) == set(val_features), "Feature mismatch between train and validation sets for XGBoost."

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

    dtrain = DMatrix(X_train, label=y_train)
    dval   = DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1,
        "n_jobs": 8  
    }

    eval_list = [(dtrain, "train"), (dval, "val")]

    # Evals result for logging
    evals_result = {}
    model = xgb_train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=eval_list,
        early_stopping_rounds=10,
        evals_result=evals_result
    )

    # Log the training/validation loss per iteration
    train_loss_history = evals_result["train"]["logloss"]
    val_loss_history = evals_result["val"]["logloss"]
    for i, (train_loss, val_loss) in enumerate(zip(train_loss_history, val_loss_history)):
        writer.add_scalar("XGB_Loss/Train", train_loss, i)
        writer.add_scalar("XGB_Loss/Validation", val_loss, i)

    best_iter = model.best_iteration
    val_preds = model.predict(dval, iteration_range=(0, best_iter + 1))
    val_preds_bin = (val_preds > 0.5).astype(int)
    score = f1_score(y_val, val_preds_bin, average="macro")

    # Log the final Macro-F1
    writer.add_scalar("XGB_F1/Validation", score, best_iter)
    print(f"[XGBoost DMatrix] Validation Macro-F1: {score:.4f}")

    model.save_model(model_path)
    print(f"[XGBoost DMatrix] Model saved to {model_path}")

    writer.close()

###############################################################################
# 2. XGBoost with Optuna + TensorBoard
###############################################################################
def train_xgboost_with_optuna(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/xgb_optuna_model.json",
    num_trials=1
):
    """
    XGBoost training with Optuna hyperparameter search + TensorBoard logging.
    """
    print(f"[INFO] Running XGBoost with Optuna for hyperparam tuning. Trials={num_trials}")
    writer = SummaryWriter(log_dir="runs/xgboost_optuna_experiment")

    from optuna.pruners import MedianPruner
    pruner = MedianPruner(n_startup_trials=1)
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_study",
        storage="sqlite:///optuna_xgb_study.db",
        load_if_exists=True,
        pruner=pruner
    )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Exclude 'timestamp' from features
    non_feature_cols = ["target", "timestamp"]
    X_train = train_df.drop(columns=non_feature_cols, errors="ignore")
    y_train = train_df["target"]
    X_val = val_df.drop(columns=non_feature_cols, errors="ignore")
    y_val = val_df["target"]

    # Debugging: Print feature lists
    train_features = X_train.columns.tolist()
    val_features = X_val.columns.tolist()
    print(f"[DEBUG] XGBoost Optuna Train features ({len(train_features)}): {train_features}")
    print(f"[DEBUG] XGBoost Optuna Validation features ({len(val_features)}): {val_features}")
    assert set(train_features) == set(val_features), "Feature mismatch between train and validation sets for XGBoost Optuna."

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
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 15),
            "n_jobs": 8
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals_result = {}
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=200,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=False,
            evals_result=evals_result
        )

        best_iter_ = model.best_iteration
        val_preds = model.predict(dval, iteration_range=(0, best_iter_ + 1))
        val_preds_bin = (val_preds > 0.5).astype(int)
        score_ = f1_score(y_val, val_preds_bin, average="macro")

        return score_

    # Run Optuna search
    study.optimize(objective, n_trials=num_trials, n_jobs=8)

    print(f"[Optuna] Best hyperparameters: {study.best_params}")
    print(f"[Optuna] Best Macro-F1 Score: {study.best_value:.4f}")

    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"
    best_params["verbosity"] = 1

    dtrain_final = xgb.DMatrix(X_train, label=y_train)
    dval_final   = xgb.DMatrix(X_val, label=y_val)

    # Final model with best params
    final_evals_result = {}
    final_model = xgb.train(
        params=best_params,
        dtrain=dtrain_final,
        num_boost_round=300,
        evals=[(dtrain_final, "train"), (dval_final, "val")],
        early_stopping_rounds=10,
        verbose_eval=True,
        evals_result=final_evals_result
    )

    # Log per-iteration info for final model
    train_loss_history = final_evals_result["train"]["logloss"]
    val_loss_history = final_evals_result["val"]["logloss"]
    for i, (t_loss, v_loss) in enumerate(zip(train_loss_history, val_loss_history)):
        writer.add_scalar("XGB_Optuna_Loss/Train", t_loss, i)
        writer.add_scalar("XGB_Optuna_Loss/Validation", v_loss, i)

    best_iter_final = final_model.best_iteration
    val_preds_final = final_model.predict(dval_final, iteration_range=(0, best_iter_final + 1))
    val_preds_bin_final = (val_preds_final > 0.5).astype(int)
    final_score = f1_score(y_val, val_preds_bin_final, average="macro")

    writer.add_scalar("XGB_Optuna_F1/Validation", final_score, best_iter_final)
    print(f"[Optuna XGBoost] Final Validation Macro-F1: {final_score:.4f}")
    final_model.save_model(model_path)
    print(f"[Optuna XGBoost] Optimized model saved to {model_path}")

    writer.close()


###############################################################################
# 3. Multi-Step LSTM (MPS) + TensorBoard
###############################################################################
def train_lstm_multistep(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
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
    Multi-step LSTM training (sliding window) with TensorBoard logging.
    Uses MPS if available.
    """
    print(f"[INFO] Running multi-step LSTM with window_size={window_size}, use_attention={use_attention}")
    writer = SummaryWriter(log_dir="runs/lstm_multistep_experiment")

    # Load feature-engineered data
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    train_dataset = CryptoSlidingWindowDataset(train_df, window_size=window_size)
    val_dataset   = CryptoSlidingWindowDataset(val_df, window_size=window_size)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Log model graph (dummy input)
    dummy_input = torch.randn(1, window_size, input_dim).to(device)
    writer.add_graph(model, dummy_input)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="mps", enabled=(device.type == "mps")):
                logits = model(seq_batch)
                loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("LSTM_Multistep/Loss_Train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for seq_batch, y_batch in val_loader:
                seq_batch = seq_batch.to(device)
                y_batch   = y_batch.to(device)
                with torch.autocast(device_type="mps", enabled=(device.type == "mps")):
                    logits = model(seq_batch)
                    loss = criterion(logits, y_batch)
                    preds = torch.argmax(logits, dim=1)

                val_loss += loss.item()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        writer.add_scalar("LSTM_Multistep/Loss_Validation", avg_val_loss, epoch)
        writer.add_scalar("LSTM_Multistep/F1_Validation", val_f1, epoch)

        print(f"[LSTM Multistep] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[LSTM Multistep] Model saved to {model_path}")

    writer.close()


###############################################################################
# 4. Multi-Step LSTM with Optuna (MPS) + TensorBoard
###############################################################################
def train_lstm_multistep_with_optuna(
    train_csv="data/intermediate/train_fe.csv",
    val_csv="data/intermediate/val_fe.csv",
    model_path="results/models/lstm_multistep_optuna.pth",
    window_size=5,
    num_trials=1,
    base_epochs=5,
    final_epochs=10,
    batch_size=128,
    num_workers=4
):
    """
    Multi-step LSTM with sliding window + Optuna hyperparameter tuning + TensorBoard logging.
    Uses MPS if available.
    """
    print(f"[INFO] Running multi-step LSTM with Optuna, window_size={window_size}, trials={num_trials}")

    from torch.utils.tensorboard import SummaryWriter
    from optuna.pruners import MedianPruner

    writer_optuna = SummaryWriter(log_dir="runs/lstm_multistep_optuna_experiment")
    pruner = MedianPruner(n_startup_trials=1)

    study = optuna.create_study(
        direction="maximize",
        study_name="lstm_study",
        storage="sqlite:///optuna_lstm_study.db",
        load_if_exists=True,
        pruner=pruner
    )

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_df.dropna(inplace=True)
    val_df.dropna(inplace=True)

    train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    val_df   = val_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    train_dataset = CryptoSlidingWindowDataset(train_df, window_size=window_size)
    val_dataset   = CryptoSlidingWindowDataset(val_df, window_size=window_size)

    # ----- FIX: Use torch.device here instead of a string -----
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    def objective(trial):
        hidden_dim   = trial.suggest_int("hidden_dim", 32, 128, step=32)
        num_layers   = trial.suggest_int("num_layers", 1, 3)
        dropout      = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        use_attention= trial.suggest_categorical("use_attention", [False, True])

        print(f"[Optuna LSTM] Starting trial with params: {trial.params}")

        model = LSTMAttnClassifier(
            input_dim=train_dataset[0][0].shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            use_attention=use_attention
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )

        # Basic training loop for base_epochs
        for ep in range(base_epochs):
            model.train()
            total_loss = 0
            for seq_batch, y_batch in train_loader:
                seq_batch = seq_batch.to(device)
                y_batch   = y_batch.to(device)

                optimizer.zero_grad()
                # ----- Use device.type for autocast -----
                with torch.autocast(device_type="mps", enabled=(device.type == "mps")):
                    logits = model(seq_batch)
                    loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"[Optuna LSTM] Trial {trial.number} Epoch {ep+1}/{base_epochs} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seq_batch, y_batch in val_loader:
                seq_batch = seq_batch.to(device)
                y_batch   = y_batch.to(device)
                with torch.autocast(device_type="mps", enabled=(device.type == "mps")):
                    logits = model(seq_batch)
                    preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        current_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"[Optuna LSTM] Trial {trial.number} | Validation Macro-F1: {current_f1:.4f}")

        return current_f1

    # Conduct hyperparameter search
    study.optimize(objective, n_trials=num_trials, n_jobs=8)

    print("[Optuna LSTM] Best Params:", study.best_params)
    print("[Optuna LSTM] Best Value (Macro-F1):", study.best_value)

    hp = study.best_params
    print(f"[Optuna LSTM] Using best hyperparams: {hp}")

    # ------------------- RETRAIN with best hyperparams ------------------- #
    final_model = LSTMAttnClassifier(
        input_dim=train_dataset[0][0].shape[1],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
        num_layers=hp["num_layers"],
        use_attention=hp["use_attention"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=hp["lr"])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # Log final model graph
    dummy_input = torch.randn(1, window_size, train_dataset[0][0].shape[1]).to(device)
    writer_optuna.add_graph(final_model, dummy_input)

    for ep in range(final_epochs):
        final_model.train()
        total_loss = 0
        for seq_batch, y_batch in train_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="mps", enabled=(device.type == "mps")):
                logits = final_model(seq_batch)
                loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer_optuna.add_scalar("LSTM_Optuna/Loss_Train_Retrain", avg_train_loss, ep)
        print(f"[Optuna LSTM] Retrain Epoch {ep+1}/{final_epochs} | Train Loss: {avg_train_loss:.4f}")

    # Final validation
    final_model.eval()
    val_preds, val_labels = [], []
    val_loss = 0
    with torch.no_grad():
        for seq_batch, y_batch in val_loader:
            seq_batch = seq_batch.to(device)
            y_batch   = y_batch.to(device)
            with torch.autocast(device_type="mps", enabled=(device.type == "mps")):
                logits = final_model(seq_batch)
                loss = criterion(logits, y_batch)
                preds = torch.argmax(logits, dim=1)

            val_loss += loss.item()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    final_f1 = f1_score(val_labels, val_preds, average="macro")
    writer_optuna.add_scalar("LSTM_Optuna/Loss_Validation_Retrain", avg_val_loss, final_epochs)
    writer_optuna.add_scalar("LSTM_Optuna/F1_Validation_Retrain", final_f1, final_epochs)

    print(f"[Optuna LSTM] Final Retrained Macro-F1: {final_f1:.4f}")

    # Save train features for alignment checks
    train_df = pd.read_csv(train_csv)
    train_df.dropna(inplace=True)
    train_features = train_df.drop(columns=["target", "timestamp"], errors="ignore").columns.tolist()

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "hyperparameters": {
            "hidden_dim": hp["hidden_dim"],
            "num_layers": hp["num_layers"],
            "dropout": hp["dropout"],
            "input_dim": train_dataset[0][0].shape[1],
            "use_attention": hp["use_attention"],
            "train_features": train_features
        }
    }, model_path)
    print(f"[Optuna LSTM] Optimized model saved to {model_path}")

    writer_optuna.close()

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

    # Load and preprocess test data
    full_test_df = pd.read_csv(test_csv)
    full_test_df = add_technical_indicators_inline(full_test_df.copy(), window=5)
    full_test_df.dropna(inplace=True)

    # ---------------- Subset for XGBoost ----------------
    xgb_train_df = pd.read_csv("data/intermediate/train_fe.csv")
    xgb_features = xgb_train_df.drop(columns=["target", "timestamp"], errors="ignore").columns.tolist()
    xgb_test_df = full_test_df[xgb_features].copy()
    print(f"[DEBUG] XGB test features: {len(xgb_test_df.columns)} columns")

    X_test_xgb = xgb_test_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    dtest = DMatrix(X_test_xgb)

    import xgboost as xgb
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    xgb_preds = xgb_model.predict(dtest)
    xgb_preds_bin = (xgb_preds > 0.5).astype(int)

    # ---------------- Subset for LSTM ----------------
    checkpoint = torch.load(lstm_model_path, map_location="cpu")
    lstm_features = checkpoint["hyperparameters"]["train_features"]
    lstm_test_df = full_test_df[list(set(lstm_features) & set(full_test_df.columns))].copy()
    print(f"[DEBUG] LSTM test features: {len(lstm_test_df.columns)} columns")

    test_dataset = CryptoSlidingWindowDataset(lstm_test_df, window_size=window_size, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Fix the device initialization
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_dim = test_dataset[0].shape[1]
    print(f"[DEBUG] LSTM input_dim: {input_dim}")

    lstm_model = LSTMAttnClassifier(
        input_dim=input_dim,
        hidden_dim=checkpoint["hyperparameters"]["hidden_dim"],
        dropout=checkpoint["hyperparameters"]["dropout"],
        num_layers=checkpoint["hyperparameters"]["num_layers"],
        use_attention=checkpoint["hyperparameters"]["use_attention"]
    ).to(device)

    lstm_model.load_state_dict(checkpoint["model_state_dict"])
    lstm_model.eval()

    lstm_preds = []
    with torch.no_grad():
        for seq_batch in test_loader:
            seq_batch = seq_batch.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "mps")):
                logits = lstm_model(seq_batch)
                preds = torch.argmax(logits, dim=1)
            lstm_preds.extend(preds.cpu().numpy())
    lstm_preds = np.array(lstm_preds)

    # ---------------- Weighted Ensemble ----------------
    total_f1 = xgb_f1 + lstm_f1
    xgb_weight = xgb_f1 / total_f1
    lstm_weight = lstm_f1 / total_f1

    print(f"[INFO] Ensemble Weights | XGBoost: {xgb_weight:.2f}, LSTM: {lstm_weight:.2f}")
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

    # --------------- Train XGBoost ---------------
    if use_optuna_for_xgb:
        train_xgboost_with_optuna(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/xgb_optuna_model.json",
            num_trials=1  
        )
    else:
        train_xgboost(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/xgb_model.json"
        )

    # --------------- Train LSTM ---------------
    if use_optuna_for_lstm:
        train_lstm_multistep_with_optuna(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
            model_path="results/models/lstm_multistep_optuna.pth",
            window_size=5,
            num_trials=1,
            base_epochs=5,
            final_epochs=10,
            batch_size=128,
            num_workers=4
        )
    elif use_multistep_lstm:
        train_lstm_multistep(
            train_csv="data/intermediate/train_fe.csv",
            val_csv="data/intermediate/val_fe.csv",
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

    # --------------- Debug Feature Alignment ---------------
    train_df = pd.read_csv("data/intermediate/train_fe.csv")
    val_df = pd.read_csv("data/intermediate/val_fe.csv")

    train_features = train_df.drop(columns=["target", "timestamp"], errors="ignore").columns
    print(f"[DEBUG] Train features: {len(train_features)} columns")

    val_features = val_df.drop(columns=["target", "timestamp"], errors="ignore").columns
    print(f"[DEBUG] Validation features: {len(val_features)} columns")

    if not set(train_features) == set(val_features):
        print("[WARNING] Feature mismatch between train and validation datasets!")
        print(f"Train-only features: {set(train_features) - set(val_features)}")
        print(f"Validation-only features: {set(val_features) - set(train_features)}")

    # --------------- Final Ensemble on Test Data ---------------
    test_df = pd.read_csv("data/final/test_fe.csv")  # Ensure test data is loaded
    print(f"[INFO] Test data loaded with {test_df.shape[0]} rows and {test_df.shape[1]} columns")

    test_features = test_df.drop(columns=["target", "timestamp"], errors="ignore").columns
    print(f"[DEBUG] Test features before alignment: {len(test_features)} columns")
    print(f"Test feature names: {test_features.tolist()}")

    # Align test set features with training features
    test_df = test_df[list(set(train_features) & set(test_df.columns))].copy()
    print(f"[DEBUG] Test features after alignment: {test_df.shape[1]} columns")

    if not set(train_features) == set(test_df.columns):
        print("[WARNING] Feature mismatch between train and test datasets!")
        print(f"Train-only features: {set(train_features) - set(test_df.columns)}")
        print(f"Test-only features: {set(test_df.columns) - set(train_features)}")

    # Perform ensemble predictions
    ensemble_predictions(
        test_csv="data/final/test_fe.csv",
        xgb_model_path="results/models/xgb_optuna_model.json",
        lstm_model_path="results/models/lstm_multistep_optuna.pth",
        window_size=5,
        xgb_f1=0.4597,  
        lstm_f1=0.664,  
        output_csv="results/models/ensemble_predictions.csv"
    )

    print("Pipeline complete!")