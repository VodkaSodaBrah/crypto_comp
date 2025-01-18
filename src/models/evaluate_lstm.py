import argparse
import pandas as pd
import torch
from sklearn.metrics import f1_score
import numpy as np
import logging
import os
from torch.utils.data import DataLoader
from src.models.train_model import CryptoSlidingWindowDataset, LSTMAttnClassifier  # Adjust import as necessary
import joblib

def evaluate_lstm(data_csv, model_path="results/models/lstm_multistep_optuna.pth", window_size=5, train_features=None, save_predictions=False, output_path="results/predictions/lstm_test_predictions.csv"):
    """
    Evaluates the LSTM model on the provided dataset and optionally saves predictions.

    Args:
        data_csv (str): Path to the test dataset CSV.
        model_path (str): Path to the saved LSTM model.
        window_size (int): The size of the sliding window used during training.
        train_features (list of str): List of training feature names for alignment.
        save_predictions (bool): Whether to save predictions to a CSV file.
        output_path (str): Path to save the predictions CSV.
    """
    try:
        # Load test data
        df = pd.read_csv(data_csv)
        if "target" not in df.columns:
            logging.warning("The 'target' column is missing in the test dataset.")

        # Load scaler and feature list
        scaler = joblib.load("results/scalers/lstm_scaler.pkl")
        train_features_list = joblib.load("results/scalers/lstm_features.pkl")
        logging.info("Loaded scaler and training feature list for LSTM.")

        # Align features with training dataset
        if train_features_list:
            # Ensure all training features are present
            missing_features = set(train_features_list) - set(df.columns)
            if missing_features:
                logging.warning(f"The following training features are missing in the test set and will be filled with zeros: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0  # Fill missing features with 0 or another appropriate value

            # Reorder columns to match training features
            try:
                lstm_test_df = df[train_features_list].copy()
            except KeyError as e:
                logging.error(f"KeyError during feature alignment: {e}")
                raise
        else:
            logging.warning("No train_features provided. Proceeding without feature alignment.")
            lstm_test_df = df.copy()

        # Feature Scaling
        X_test_scaled = scaler.transform(lstm_test_df)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=train_features_list)

        # Create Dataset
        test_dataset = CryptoSlidingWindowDataset(X_test_scaled, window_size=window_size, has_target=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        # Device configuration
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Initialize LSTM model with best hyperparameters
        checkpoint = torch.load(model_path, map_location=device)
        model = LSTMAttnClassifier(
            input_dim=test_dataset[0].shape[1],
            hidden_dim=checkpoint["hyperparameters"]["hidden_dim"],
            dropout=checkpoint["hyperparameters"]["dropout"],
            num_layers=checkpoint["hyperparameters"]["num_layers"],
            use_attention=checkpoint["hyperparameters"]["use_attention"],
            bidirectional=checkpoint["hyperparameters"]["bidirectional"],
            debug=False
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logging.info(f"LSTM model loaded from {model_path}")

        # Generate LSTM predictions
        lstm_preds = []
        with torch.no_grad():
            for seq_batch in test_loader:
                seq_batch = seq_batch.to(device)
                logits = model(seq_batch)
                preds = torch.argmax(logits, dim=1)
                lstm_preds.extend(preds.cpu().numpy())
        lstm_preds = np.array(lstm_preds)

        # Compute F1 Score if 'target' exists
        if "target" in df.columns:
            y_test = df["target"].values
            score = f1_score(y_test, lstm_preds, average="macro")
            logging.info(f"F1 Score on {data_csv}: {score:.4f}")
        else:
            logging.info(f"No 'target' column in {data_csv}. Skipping F1 Score computation.")

        # Save Predictions if required
        if save_predictions:
            # Check if 'id' exists; if not, generate it
            if 'id' in df.columns:
                submission_ids = df['id'].values
            else:
                # Generate sequential 'id's starting from 1
                submission_ids = np.arange(1, len(lstm_preds) + 1)
                logging.warning("The 'id' column is missing in the test dataset. Generated sequential 'id's.")

            # Create a DataFrame for predictions
            submission = pd.DataFrame({
                'id': submission_ids,
                'prediction': lstm_preds
            })

            # Save to CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            submission.to_csv(output_path, index=False)
            logging.info(f"Predictions saved to {output_path}")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please ensure the model file, scaler, and dataset exist.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM model on test data.")
    parser.add_argument("--data", type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument("--model", type=str, default="results/models/lstm_multistep_optuna.pth", help="Path to the saved LSTM model.")
    parser.add_argument("--window_size", type=int, default=5, help="Sliding window size used during training.")
    parser.add_argument("--train_features", type=str, nargs="+", default=None, help="List of training features for alignment.")
    parser.add_argument("--save_predictions", action='store_true', help="Flag to save predictions to a CSV file.")
    parser.add_argument("--output_path", type=str, default="results/predictions/lstm_test_predictions.csv", help="Path to save the predictions CSV.")
    args = parser.parse_args()

    evaluate_lstm(args.data, args.model, args.window_size, args.train_features, args.save_predictions, args.output_path)