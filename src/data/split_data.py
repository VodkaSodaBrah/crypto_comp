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

    print(f"[INFO] Reading source CSV: {source_csv}")
    df = pd.read_csv(source_csv)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    print(f"[INFO] Total rows before split: {len(df)}")

    df.sort_values("timestamp", inplace=True)
    print("[INFO] Data sorted by timestamp.")

    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test_df = df[df["timestamp"] > val_end]

    # Print insights about the splits
    print(f"[INFO] Train range: <= {train_end}")
    print(f"[INFO] Train set rows: {len(train_df)}")
    print(f"Sample Train timestamps: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

    print(f"[INFO] Validation range: > {train_end} and <= {val_end}")
    print(f"[INFO] Validation set rows: {len(val_df)}")
    print(f"Sample Validation timestamps: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")

    print(f"[INFO] Test range: > {val_end}")
    print(f"[INFO] Test set rows: {len(test_df)}")
    print(f"Sample Test timestamps: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}\n")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"[INFO] Saved train split to: {train_csv}")
    print(f"[INFO] Saved val split to:   {val_csv}")
    print(f"[INFO] Saved test split to:  {test_csv}")

if __name__ == "__main__":
    time_series_split()