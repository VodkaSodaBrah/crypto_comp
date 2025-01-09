#!/usr/bin/env bash

# Ensure script halts on any error
set -e

echo "Starting the pipeline..."

# 1. Split data
echo "Step 1: Splitting data..."
python src/data/split_data.py

# 2. Preprocess data
echo "Step 2: Preprocessing data..."
python src/data/preprocess.py \
    --in_csv=data/intermediate/train.csv \
    --out_csv=data/intermediate/train_preprocessed.csv
python src/data/preprocess.py \
    --in_csv=data/intermediate/val.csv \
    --out_csv=data/intermediate/val_preprocessed.csv
python src/data/preprocess.py \
    --in_csv=data/final/test.csv \
    --out_csv=data/final/test_preprocessed.csv

# 3. Perform feature engineering
echo "Step 3: Performing feature engineering..."
python src/features/feature_engineering.py \
    --in_csv=data/intermediate/train_preprocessed.csv \
    --out_csv=data/intermediate/train_fe.csv
python src/features/feature_engineering.py \
    --in_csv=data/intermediate/val_preprocessed.csv \
    --out_csv=data/intermediate/val_fe.csv
python src/features/feature_engineering.py \
    --in_csv=data/final/test_preprocessed.csv \
    --out_csv=data/final/test_fe.csv

# 4. Train model
echo "Step 4: Training model..."
python src/models/train_model.py

# 5. Evaluate on validation set
echo "Step 5: Evaluating model on validation set..."
python src/models/evaluate_model.py --data data/intermediate/val_fe.csv

# 6. Evaluate on test set
echo "Step 6: Evaluating model on test set..."
python src/models/evaluate_model.py --data data/final/test_fe.csv

echo "Pipeline complete!"