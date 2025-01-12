import pandas as pd

# Load and convert Unix timestamps
df = pd.read_csv("/Users/mchildress/Code/my_crypto_prediction/data/raw/train.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

print("Earliest timestamp:", df["timestamp"].min())
print("Latest timestamp:", df["timestamp"].max())