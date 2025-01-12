import pandas as pd

# Load the datasets
train = pd.read_csv("data/intermediate/train.csv")
val = pd.read_csv("data/intermediate/val.csv")
test = pd.read_csv("data/final/test.csv")

# Convert timestamps to datetime if not already done
train["timestamp"] = pd.to_datetime(train["timestamp"])
val["timestamp"] = pd.to_datetime(val["timestamp"])
test["timestamp"] = pd.to_datetime(test["timestamp"])

# Check for overlaps
train_val_overlap = train["timestamp"].isin(val["timestamp"]).any()
train_test_overlap = train["timestamp"].isin(test["timestamp"]).any()
val_test_overlap = val["timestamp"].isin(test["timestamp"]).any()

print(f"Train-Val Overlap: {train_val_overlap}")
print(f"Train-Test Overlap: {train_test_overlap}")
print(f"Val-Test Overlap: {val_test_overlap}")