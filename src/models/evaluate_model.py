import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

def evaluate(data_csv, model_path="results/models/xgb_model.json"):
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
        print(f"F1 Score on {data_csv}: {score:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the model file and dataset exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model on test data.")
    parser.add_argument("--data", type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument("--model", type=str, default="results/models/xgb_model.json", help="Path to the saved XGBoost model.")
    args = parser.parse_args()

    evaluate(args.data, args.model)