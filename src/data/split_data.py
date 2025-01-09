import pandas as pd
import os

def time_series_split(
    source_csv="/Users/mchildress/Code/my_crypto_prediction/data/raw/train.csv",
    train_csv="data/intermediate/train.csv",
    val_csv="data/intermediate/val.csv",
    test_csv="data/final/test.csv",
    train_end="2021-12-31",
    val_end="2022-03-31"
):
    """
    Splits the dataset into train, validation, and test based on timestamp ranges.

    1. Train: <= train_end
    2. Validation: train_end < timestamp <= val_end
    3. Test: > val_end
    """
    if not os.path.exists(os.path.dirname(train_csv)):
        os.makedirs(os.path.dirname(train_csv))
    if not os.path.exists(os.path.dirname(test_csv)):
        os.makedirs(os.path.dirname(test_csv))

    print(f"Reading source CSV: {source_csv}")
    df = pd.read_csv(source_csv)

    # Convert Unix timestamps to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    print(f"Total rows before split: {len(df)}")

    # Sort by timestamp for consistency
    df.sort_values("timestamp", inplace=True)

    # Split data
    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test_df = df[df["timestamp"] > val_end]

    print(f"Train range: <= {train_end} | Rows: {len(train_df)}")
    print(f"Val range: > {train_end} and <= {val_end} | Rows: {len(val_df)}")
    print(f"Test range: > {val_end} | Rows: {len(test_df)}\n")

    # Save splits to CSV
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved train split to: {train_csv}")
    print(f"Saved val split to:   {val_csv}")
    print(f"Saved test split to:  {test_csv}")

if __name__ == "__main__":
    time_series_split()