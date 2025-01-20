import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utils for preprocessing
def add_rsi(df, window=14, price_col="close"):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    roll_gain = gain.rolling(window=window).mean()
    roll_loss = loss.rolling(window=window).mean().replace(0, 1e-6)
    rs = roll_gain / roll_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(0)
    return df

def add_lag_features(df, price_col="close", num_lags=1):
    for lag in range(1, num_lags + 1):
        df[f"{price_col}_lag{lag}"] = df[price_col].shift(lag)
    df.fillna(0, inplace=True)
    return df

def add_technical_indicators(df, window=5, bb_window=20, price_col="close"):
    df[f"{price_col}_rolling_mean"] = df[price_col].rolling(window=window).mean()
    rolling_mean = df[price_col].rolling(bb_window).mean()
    rolling_std = df[price_col].rolling(bb_window).std().replace(0, 1e-6)
    df[f"{price_col}_bb_upper"] = rolling_mean + (rolling_std * 2)
    df[f"{price_col}_bb_lower"] = rolling_mean - (rolling_std * 2)
    df.fillna(0, inplace=True)
    return df

def preprocess_and_feature_engineer(in_csv, out_csv):
    """
    Preprocess and feature engineer the data.
    """
    df = pd.read_csv(in_csv)

    # Add features
    df = add_technical_indicators(df)
    df = add_rsi(df)
    df = add_lag_features(df)

    # Ensure all columns except "target" are numeric
    for col in df.columns:
        if col != "target":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaNs and infinite values with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Save the processed DataFrame
    df.to_csv(out_csv, index=False)
    logger.info(f"Processed data saved to {out_csv}")

# Dataset
class CryptoDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.targets is not None:
            return x, self.targets[idx]
        return x

# Kaggle Submission
def save_predictions_for_kaggle(model, test_loader, output_csv="submission.csv"):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            batch_size, seq_length = X_batch.size(0), X_batch.size(1)

            time_features_batch = torch.zeros(batch_size, seq_length, 1).to(X_batch.device)
            observed_mask_batch = torch.ones(batch_size, seq_length, 1).to(X_batch.device)

            outputs = model(
                past_values=X_batch,
                past_time_features=time_features_batch,
                past_observed_mask=observed_mask_batch,
            )["logits"]

            # Use the last prediction for submission
            predictions.extend(outputs[:, -1].cpu().numpy())

    submission = pd.DataFrame({"Id": range(len(predictions)), "Target": predictions})
    submission.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")

# Model training with Optuna
def train_time_series_transformer_with_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        embed_dim = trial.suggest_int("embed_dim", 32, 128, step=32)
        num_heads = trial.suggest_int("num_heads", 2, 8, step=2)
        encoder_layers = trial.suggest_int("encoder_layers", 1, 4)
        decoder_layers = trial.suggest_int("decoder_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        config = TimeSeriesTransformerConfig(
            prediction_length=y_train.shape[1],
            context_length=X_train.shape[1],
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dropout=dropout,
        )
        model = TimeSeriesTransformerForPrediction(config).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        train_loader = DataLoader(CryptoDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(CryptoDataset(X_val, y_val), batch_size=32, shuffle=False)

        for epoch in range(5):  # Training loop
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Ensure consistent tensor dimensions
                batch_size, seq_length = X_batch.size(0), X_batch.size(1)
                time_features_batch = torch.zeros(batch_size, config.context_length, 1).to(X_batch.device)
                observed_mask_batch = torch.ones(batch_size, config.context_length, 1).to(X_batch.device)

                outputs = model(
                    past_values=X_batch,
                    past_time_features=time_features_batch,
                    past_observed_mask=observed_mask_batch,
                )["logits"]

                loss = criterion(outputs[:, -config.prediction_length:], y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    time_features_batch = torch.zeros(batch_size, config.context_length, 1).to(X_batch.device)
                    observed_mask_batch = torch.ones(batch_size, config.context_length, 1).to(X_batch.device)

                    outputs = model(
                        past_values=X_batch,
                        past_time_features=time_features_batch,
                        past_observed_mask=observed_mask_batch,
                    )["logits"]
                    val_loss += criterion(outputs[:, -config.prediction_length:], y_batch).item()

            trial.report(val_loss / len(val_loader), step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return val_loss / len(val_loader)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best loss: {study.best_value}")
    return study.best_params

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Path to training CSV.")
    parser.add_argument("--val_csv", required=True, help="Path to validation CSV.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    args = parser.parse_args()

    train_csv, val_csv = args.train_csv, args.val_csv
    train_processed = "data/intermediate/train_fe.csv"
    val_processed = "data/intermediate/val_fe.csv"

    preprocess_and_feature_engineer(train_csv, train_processed)
    preprocess_and_feature_engineer(val_csv, val_processed)

    train_data = pd.read_csv(train_processed)
    val_data = pd.read_csv(val_processed)

    X_train = torch.tensor(train_data.drop(columns=["target"]).values, dtype=torch.float32)
    y_train = torch.tensor(train_data["target"].values, dtype=torch.float32).unsqueeze(-1)
    X_val = torch.tensor(val_data.drop(columns=["target"]).values, dtype=torch.float32)
    y_val = torch.tensor(val_data["target"].values, dtype=torch.float32).unsqueeze(-1)

    best_params = train_time_series_transformer_with_optuna(X_train, y_train, X_val, y_val, args.n_trials)

    # Save model for Kaggle competition
    test_data = pd.read_csv("data/intermediate/test_fe.csv")  # Example test file
    X_test = torch.tensor(test_data.values, dtype=torch.float32)
    test_loader = DataLoader(CryptoDataset(X_test), batch_size=32, shuffle=False)
    save_predictions_for_kaggle(best_params, test_loader)