import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

def evaluate(data_csv, model_path="results/models/xgb_model.json"):
    """
    Evaluate the XGBoost model on a test dataset.

    Args:
        data_csv (str): Path to the test data CSV.
        model_path (str): Path to the saved XGBoost model.

    Returns:
        None
    """
    try:
        # Load test data
        df = pd.read_csv(data_csv)
        if "target" not in df.columns:
            raise ValueError("The 'target' column is missing in the dataset.")
        
        # Drop timestamp column if it exists
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        X_test = df.drop(columns=["target"], errors="ignore")
        y_test = df["target"]

        # Ensure numeric conversion
        X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Load the XGBoost model as Booster
        model = xgb.Booster()
        model.load_model(model_path)

        # Convert test data to DMatrix
        dtest = xgb.DMatrix(X_test)

        # Predict probabilities and convert to binary predictions
        preds = model.predict(dtest)
        preds_bin = (preds > 0.5).astype(int)

        # Evaluate F1 score
        score = f1_score(y_test, preds_bin, average="macro")
        print(f"F1 Score on {data_csv}: {score:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the model file and dataset exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")