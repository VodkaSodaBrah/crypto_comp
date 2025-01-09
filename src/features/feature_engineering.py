import pandas as pd
import numpy as np
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def add_technical_indicators(in_csv, out_csv, window=5):
    if os.path.exists(out_csv):
        os.remove(out_csv)

    df = pd.read_csv(in_csv)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    numeric_cols = [
        "timestamp",   # <--- Keep it here
        "open", "high", "low", "close",
        "volume", "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume"
    ]

    # Convert numeric columns (incl. timestamp)
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .replace(r"[^0-9.\-]+", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna(0, inplace=True)

    print("\n==== DEBUG: dtypes after numeric conversion ====")
    print(df[numeric_cols].dtypes)
    print("================================================\n")

    # Technical indicators
    df["return"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(window=window).mean()
    df["volatility_5"] = df["return"].rolling(window=window).std()
    df["ema_5"] = df["close"].ewm(span=window, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_window = 20
    df["bb_mid"] = df["close"].rolling(window=bb_window).mean()
    df["bb_std"] = df["close"].rolling(window=bb_window).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    df.dropna(inplace=True)

    df.to_csv(out_csv, index=False)
    logger.info(f"Feature-engineered data saved to: {out_csv}")


def add_technical_indicators_inline(df, window=5):
    # Optionally keep 'timestamp' or derive additional columns inside this function too.
    # If your timestamp is still in df, do something like:
    # df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df["return"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(window=window).mean()
    df["volatility_5"] = df["return"].rolling(window=window).std()
    df["ema_5"] = df["close"].ewm(span=window, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering script.")
    parser.add_argument("--in_csv", type=str, required=True, help="Input CSV file path.")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--window", type=int, default=5, help="Window size for rolling calculations.")
    args = parser.parse_args()

    add_technical_indicators(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        window=args.window
    )