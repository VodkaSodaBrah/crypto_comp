import hashlib
import pandas as pd

def hash_row(row):
    """Hashes a row to check for duplicates across datasets."""
    return hashlib.md5(str(row.values).encode()).hexdigest()

# Load the feature-engineered datasets
train_fe = pd.read_csv("data/intermediate/train_fe.csv")
val_fe = pd.read_csv("data/intermediate/val_fe.csv")
test_fe = pd.read_csv("data/final/test_fe.csv")

# Hash each row
train_fe["hash"] = train_fe.apply(hash_row, axis=1)
val_fe["hash"] = val_fe.apply(hash_row, axis=1)
test_fe["hash"] = test_fe.apply(hash_row, axis=1)

# Check for overlaps
train_test_overlap = train_fe["hash"].isin(test_fe["hash"]).any()
val_test_overlap = val_fe["hash"].isin(test_fe["hash"]).any()

print(f"Train-Test Feature Overlap: {train_test_overlap}")
print(f"Val-Test Feature Overlap: {val_test_overlap}")
