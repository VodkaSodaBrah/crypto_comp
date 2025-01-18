# src/data/preprocess.py

import pandas as pd
import os
import time
import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(in_csv, out_csv, chunksize=100_000):
    """
    Reads data from in_csv in chunks, applies basic cleaning, and writes to out_csv.
    Does not drop any rows; infinite and NaN values are replaced with 0.
    """

    if os.path.exists(out_csv):
        os.remove(out_csv)
        logger.info(f"Existing output file {out_csv} removed.")

    reader = pd.read_csv(in_csv, chunksize=chunksize)

    for chunk_idx, df in enumerate(reader):
        start_time = time.time()
        try:
            # Convert columns to numeric if possible
            for col in df.columns:
                if col not in ["timestamp", "target"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            header = (chunk_idx == 0)
            df.to_csv(out_csv, index=False, header=header, mode="a")
            logger.info(
                f"Processed chunk {chunk_idx + 1} with {len(df)} rows in "
                f"{time.time() - start_time:.2f}s."
            )
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
            continue

    logger.info(f"Preprocessed data saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CSV data.")
    parser.add_argument("--in_csv", type=str, required=True, help="Input CSV file path.")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for processing.")
    args = parser.parse_args()

    preprocess_data(args.in_csv, args.out_csv, chunksize=args.chunksize)