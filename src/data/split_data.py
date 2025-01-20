# src/data/split_data.py

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def time_series_split(
    source_csv="data/raw/train.csv",
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    test_csv="data/final/test.csv",
    train_end="2020-01-31",
    val_end="2020-06-30",
    required_test_rows=909617
):
    """
    Splits the dataset into train, validation, and test sets based on timestamp ranges.
    Ensures the test set contains the required number of rows as specified.

    Parameters:
        source_csv (str): Path to the source dataset.
        train_csv (str): Path to save the train split.
        val_csv (str): Path to save the validation split.
        test_csv (str): Path to save the test split.
        train_end (str): End date for the training data.
        val_end (str): End date for the validation data.
        required_test_rows (int): Number of rows required in the test set.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    logger.info(f"Reading source CSV: {source_csv}")
    df = pd.read_csv(source_csv)

    # Validate data
    if "timestamp" not in df.columns:
        raise KeyError("Missing 'timestamp' column in the dataset.")
    if df.empty:
        raise ValueError("The input dataset is empty.")

    # Convert timestamp to datetime and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    if df["timestamp"].isnull().any():
        raise ValueError("Timestamp conversion resulted in NaT values. Check input format.")
    logger.info(f"Total rows before split: {len(df)}")
    df.sort_values("timestamp", inplace=True)
    logger.info("Data sorted by timestamp.")

    # Perform initial splits
    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test_df = df[df["timestamp"] > val_end]

    # Ensure the test dataset has the required number of rows
    if len(test_df) > required_test_rows:
        test_df = test_df[-required_test_rows:]  # Take the last `required_test_rows`
    elif len(test_df) < required_test_rows:
        missing_rows = required_test_rows - len(test_df)
        logger.warning(f"Test set is short by {missing_rows} rows. Adjusting splits...")
        if len(val_df) > missing_rows:
            extra_rows = val_df[-missing_rows:]
            val_df = val_df[:-missing_rows]
            test_df = pd.concat([extra_rows, test_df], ignore_index=True)
            logger.info(f"Added {missing_rows} rows from validation to test set.")
        else:
            raise ValueError(f"Insufficient rows in validation to create a test set of {required_test_rows} rows.")

    # Save the splits
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info(f"Train set: {len(train_df)} rows saved to {train_csv}")
    logger.info(f"Validation set: {len(val_df)} rows saved to {val_csv}")
    logger.info(f"Test set: {len(test_df)} rows saved to {test_csv}")

if __name__ == "__main__":
    time_series_split()