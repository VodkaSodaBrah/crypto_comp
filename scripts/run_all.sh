#!/usr/bin/env bash

# 1. Split data
python src/data/split_data.py

# 2. Preprocess & feature engineering (optional: or do it during training)
# python src/data/preprocess.py
# python src/features/feature_engineering.py

# 3. Train model
python src/models/train_model.py

# 4. Evaluate on validation set
python src/models/evaluate_model.py --data data/intermediate/val.csv

# 5. Evaluate on test set
python src/models/evaluate_model.py --data data/final/test.csv

echo "Pipeline complete!"