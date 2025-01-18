import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import numpy as np
import logging
import os

def evaluate(data_csv, model_path="results/models/xgb_optuna_model.json", train_features=None, save_predictions=False, output_path="results/predictions/xgb_test_predictions.csv"):
    """
    Evaluates the XGBoost model on the provided dataset and optionally saves predictions.

    Args:
        data_csv (str): Path to the test dataset CSV.
        model_path (str): Path to the saved XGBoost model.
        train_features (list of str): List of training feature names for alignment.
        save_predictions (bool): Whether to save predictions to a CSV file.
        output_path (str): Path to save the predictions CSV.
    """
    try:
        # Load test data
        df = pd.read_csv(data_csv)
        if "target" not in df.columns:
            raise ValueError("The 'target' column is missing in the dataset.")

        # Prepare Features and Target
        X_test = df.drop(columns=["target"], errors="ignore")
        y_test = df["target"]

        # Drop 'timestamp' if present
        if "timestamp" in X_test.columns:
            X_test = X_test.drop(columns=["timestamp"], errors="ignore")

        # Align features with training dataset
        if train_features:
            # Ensure the order matches the training features
            common_features = [feature for feature in train_features if feature in X_test.columns]
            missing_features = set(train_features) - set(common_features)
            if missing_features:
                logging.warning(f"The following training features are missing in the test set and will be filled with zeros: {missing_features}")
                for feature in missing_features:
                    X_test[feature] = 0  # Fill missing features with 0 or another appropriate value
            X_test = X_test[train_features].copy()
        else:
            logging.warning("No train_features provided. Proceeding without feature alignment.")

        # Convert all columns to numeric
        X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Load the model
        model = xgb.Booster()
        model.load_model(model_path)

        # Create DMatrix for XGBoost
        dtest = xgb.DMatrix(X_test)

        # Generate Predictions
        preds = model.predict(dtest)
        preds_bin = (preds > 0.5).astype(int)

        # Compute F1 Score
        score = f1_score(y_test, preds_bin, average="macro")
        logging.info(f"F1 Score on {data_csv}: {score:.4f}")

        # Save Predictions if required
        if save_predictions:
            # Check if 'id' exists; if not, generate it
            if 'id' in df.columns:
                submission_ids = df['id'].values
            else:
                # Generate sequential 'id's starting from 1
                submission_ids = np.arange(1, len(preds_bin) + 1)
                logging.warning("The 'id' column is missing in the test dataset. Generated sequential 'id's.")

            # Create a DataFrame for predictions
            submission = pd.DataFrame({
                'id': submission_ids,
                'prediction': preds_bin
            })

            # Save to CSV
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            submission.to_csv(output_path, index=False)
            logging.info(f"Predictions saved to {output_path}")

    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please ensure the model file and dataset exist.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model on test data.")
    parser.add_argument("--data", type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument("--model", type=str, default="results/models/xgb_optuna_model.json", help="Path to the saved XGBoost model.")
    parser.add_argument("--train_features", type=str, nargs="+", default=None, help="List of training features for alignment.")
    parser.add_argument("--save_predictions", action='store_true', help="Flag to save predictions to a CSV file.")
    parser.add_argument("--output_path", type=str, default="results/predictions/xgb_test_predictions.csv", help="Path to save the predictions CSV.")
    args = parser.parse_args()

    evaluate(args.data, args.model, args.train_features, args.save_predictions, args.output_path)