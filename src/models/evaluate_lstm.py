import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.train_model import LSTMAttnClassifier
import argparse
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src.models.train_model import CryptoSlidingWindowDataset

def evaluate_lstm(data_csv, model_path, window_size=5):
    # Load test data
    df = pd.read_csv(data_csv)
    if "target" not in df.columns:
        raise ValueError("The 'target' column is missing in the dataset.")

    # Create sliding window dataset
    test_dataset = CryptoSlidingWindowDataset(df, window_size=window_size, has_target=True)  # Fixed
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Load the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = LSTMAttnClassifier(
        input_dim=len(checkpoint["hyperparameters"]["train_features"]),  # Use the length of features
        hidden_dim=checkpoint["hyperparameters"]["hidden_dim"],
        dropout=checkpoint["hyperparameters"]["dropout"],
        num_layers=checkpoint["hyperparameters"]["num_layers"],
        use_attention=checkpoint["hyperparameters"]["use_attention"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate Predictions
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    # Compute F1 Score
    f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"F1 Score on {data_csv}: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM model on test data.")
    parser.add_argument("--data", type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved LSTM model.")
    parser.add_argument("--window_size", type=int, default=5, help="Sliding window size.")
    args = parser.parse_args()

    evaluate_lstm(args.data, args.model, args.window_size)