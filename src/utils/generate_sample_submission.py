import pandas as pd

# Define the number of rows based on test data
num_rows = 909617

# Create a DataFrame with `row_id` and `target`
sample_submission = pd.DataFrame({
    "row_id": range(num_rows),
    "target": [0] * num_rows  
})

# Save the new `sample_submission.csv`
output_path = "/Users/mchildress/Code/my_crypto_prediction/data/raw/sample_submission.csv"
sample_submission.to_csv(output_path, index=False)

print(f"Sample submission with {num_rows} rows saved to {output_path}")