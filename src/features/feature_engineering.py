# src/features/feature_engineering.py

import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_rsi(df, window=14, price_col="close"):
    """
    Adds RSI (Relative Strength Index) to df: 
    RSI = 100 - (100 / (1 + (avg_gain / avg_loss))).
    """
    if price_col not in df.columns:
        logger.warning(f"{price_col} not in df; skipping RSI.")
        return df

    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    roll_gain = gain.rolling(window=window).mean()
    roll_loss = loss.rolling(window=window).mean()
    roll_loss = roll_loss.replace(0, 1e-6)  # avoid div by zero

    rs = roll_gain / roll_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(0, inplace=True)
    return df

def add_lag_features(df, price_col="close", num_lags=1):
    """
    Adds lagged features for the specified price_col. e.g., close_lag1, close_lag2, ...
    """
    for lag in range(1, num_lags+1):
        df[f"{price_col}_lag{lag}"] = df[price_col].shift(lag)
    # Fill resulting NaNs with 0
    df.fillna(0, inplace=True)
    return df

def add_technical_indicators_inline(df, window=5, bb_window=20, price_col="close"):
    """
    Existing code: rolling means, Bollinger Bands, etc. 
    """
    if price_col not in df.columns:
        logger.warning(f"{price_col} not in df; skipping indicators.")
        return df

    # Rolling mean
    df[f"{price_col}_rolling_mean"] = df[price_col].rolling(window=window).mean()
    # Bollinger
    rolling_mean = df[price_col].rolling(bb_window).mean()
    rolling_std = df[price_col].rolling(bb_window).std()
    rolling_std = rolling_std.replace(0, 1e-6)
    df[f"{price_col}_bb_upper"] = rolling_mean + (rolling_std * 2)
    df[f"{price_col}_bb_lower"] = rolling_mean - (rolling_std * 2)

    df.fillna(0, inplace=True)
    return df

def feature_engineer_data(in_csv, out_csv):
    logger.info(f"Reading data from {in_csv}")
    df = pd.read_csv(in_csv)

    # Existing indicators
    df = add_technical_indicators_inline(df, window=5, bb_window=20, price_col="close")

    # RSI
    df = add_rsi(df, window=14, price_col="close")

    # 1-lag example
    df = add_lag_features(df, price_col="close", num_lags=1)

    logger.info(f"Saving to {out_csv}")
    df.to_csv(out_csv, index=False)
    logger.info("Feature engineering complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True, help="Input CSV file path.")
    parser.add_argument("--out_csv", required=True, help="Output CSV file path.")
    args = parser.parse_args()

    feature_engineer_data(args.in_csv, args.out_csv)