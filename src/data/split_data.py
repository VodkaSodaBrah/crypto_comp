import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def time_series_split(
    source_csv="/Users/mchildress/Code/my_crypto_prediction/data/raw/train.csv",
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    test_csv="data/final/test.csv",
    train_end="2020-01-31",  # Moved earlier to allocate more rows to the test set
    val_end="2020-06-30"     # Moved earlier to allocate more rows to the test set
):
    """
    Splits the dataset into train, validation, and test based on timestamp ranges.
    Ensures the test set contains exactly 909,617 rows as required for submission.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    logger.info(f"Reading source CSV: {source_csv}")
    df = pd.read_csv(source_csv)

    # Convert timestamp to datetime and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    logger.info(f"Total rows before split: {len(df)}")
    df.sort_values("timestamp", inplace=True)
    logger.info("Data sorted by timestamp.")

    # Perform initial splits
    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test_df = df[df["timestamp"] > val_end]

    # Ensure the test dataset has exactly 909,617 rows
    if len(test_df) > 909617:
        test_df = test_df[-909617:]  # Take the last 909,617 rows
    elif len(test_df) < 909617:
        missing_rows = 909617 - len(test_df)
        logger.warning(f"Test set is short by {missing_rows} rows. Adjusting splits...")
        # Pull extra rows from validation dataset if needed
        if len(val_df) > missing_rows:
            extra_rows = val_df[-missing_rows:]
            val_df = val_df[:-missing_rows]
            test_df = pd.concat([extra_rows, test_df], ignore_index=True)
            logger.info(f"Added {missing_rows} rows from validation to test set.")
        else:
            raise ValueError(f"Insufficient rows in dataset to create test set of 909,617 rows!")

    # Print insights about the splits
    logger.info(f"Train range: <= {train_end}")
    logger.info(f"Train set rows: {len(train_df)}")
    logger.info(f"Sample Train timestamps: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

    logger.info(f"Validation range: > {train_end} and <= {val_end}")
    logger.info(f"Validation set rows: {len(val_df)}")
    logger.info(f"Sample Validation timestamps: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")

    logger.info(f"Test range: > {val_end}")
    logger.info(f"Test set rows: {len(test_df)}")
    logger.info(f"Sample Test timestamps: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}\n")

    # Save the splits
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info(f"Saved train split to: {train_csv}")
    logger.info(f"Saved val split to:   {val_csv}")
    logger.info(f"Saved test split to:  {test_csv}")

if __name__ == "__main__":
    time_series_split()