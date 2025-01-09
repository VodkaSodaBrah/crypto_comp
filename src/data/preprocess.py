import pandas as pd
import os

def preprocess_data(
    in_csv,
    out_csv,
    chunksize=100_000,
    cleaning_fn=None,
    fillna_method="ffill",
    dropna=True
):
    """
    Reads the data from in_csv in chunks, applies basic cleaning, and writes
    the results to out_csv. This helps limit memory usage for large datasets.
    """
    if os.path.exists(out_csv):
        os.remove(out_csv)

    reader = pd.read_csv(in_csv, chunksize=chunksize)

    for chunk_idx, df in enumerate(reader):
        try:
            # Basic cleaning
            print(f"[DEBUG] Chunk {chunk_idx + 1} original data sample:")
            print(df.head())  # Debugging: Inspect initial chunk

            df.drop_duplicates(inplace=True)
            df.fillna(method=fillna_method, inplace=True)
            if dropna:
                df.dropna(inplace=True)
            
            # Ensure all columns are numeric except 'timestamp'
            for col in df.columns:
                if col != "timestamp":
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df.fillna(0, inplace=True)  # Fill any NaNs after numeric conversion

            if cleaning_fn:
                df = cleaning_fn(df)

            # Debugging: Check data types and sample after cleaning
            print(f"[DEBUG] Chunk {chunk_idx + 1} dtypes after cleaning:")
            print(df.dtypes)
            print(f"[DEBUG] Chunk {chunk_idx + 1} data sample after cleaning:")
            print(df.head())

            # Save (append) to the output CSV
            header = chunk_idx == 0
            df.to_csv(out_csv, index=False, header=header, mode="a")

            print(f"Processed chunk {chunk_idx + 1} with {len(df)} rows")
        except Exception as e:
            print(f"Error processing chunk {chunk_idx + 1}: {e}")
            continue

    print(f"Preprocessed data saved to: {out_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess CSV data.")
    parser.add_argument("--in_csv", required=True, help="Input CSV file path")
    parser.add_argument("--out_csv", required=True, help="Output CSV file path")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for processing")
    args = parser.parse_args()

    preprocess_data(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        chunksize=args.chunksize
    )