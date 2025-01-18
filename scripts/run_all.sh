#!/usr/bin/env bash

# Enable bash debugging
set -ex

echo "=============================="
echo "      Starting the Pipeline   "
echo "=============================="

# Path to the configuration file (if needed)
CONFIG_FILE="config.yml"

# Create necessary directories
mkdir -p logs
mkdir -p data/intermediate
mkdir -p data/final
mkdir -p results/models
mkdir -p results/predictions

##############################
# 1. Split Data
##############################
echo "Step 1: Splitting data..."
# This script ensures the test set is exactly 909,617 rows.
python src/data/split_data.py --config $CONFIG_FILE

########################################
# 2. Preprocess Data
########################################
echo "Step 2: Preprocessing data..."

#--- Training Data ---
echo "Preprocessing Training Data..."
python src/data/preprocess.py \
    --in_csv="data/intermediate/train.csv" \
    --out_csv="data/intermediate/train_preprocessed.csv"
    # Note: No --dropna; all rows are kept.

#--- Validation Data ---
echo "Preprocessing Validation Data..."
python src/data/preprocess.py \
    --in_csv="data/intermediate/val.csv" \
    --out_csv="data/intermediate/val_preprocessed.csv"
    # Note: No --dropna; all rows are kept.

#--- Test Data ---
echo "Preprocessing Test Data..."
python src/data/preprocess.py \
    --in_csv="data/final/test.csv" \
    --out_csv="data/final/test_preprocessed.csv"
    # Note: No --dropna; all rows are kept.
    # split_data.py already ensures 909,617 rows for test.

########################################
# 3. Perform Feature Engineering
########################################
echo "Step 3: Performing feature engineering..."

#--- Training Data ---
echo "Feature Engineering on Training Data..."
python src/features/feature_engineering.py \
    --in_csv="data/intermediate/train_preprocessed.csv" \
    --out_csv="data/intermediate/train_fe.csv"

#--- Validation Data ---
echo "Feature Engineering on Validation Data..."
python src/features/feature_engineering.py \
    --in_csv="data/intermediate/val_preprocessed.csv" \
    --out_csv="data/intermediate/val_fe.csv"

#--- Test Data ---
echo "Feature Engineering on Test Data..."
python src/features/feature_engineering.py \
    --in_csv="data/final/test_preprocessed.csv" \
    --out_csv="data/final/test_fe.csv"

########################################
# 4. Train Models (XGB, LSTM, Stacking)
########################################
echo "Step 4: Training models..."
# Your new train_model.py trains XGB+Optuna, LSTM+Optuna, 
# and optionally stacking. It also prints confusion matrix, recall, AUC, etc.
python src/models/train_model.py

########################################
# 5. Generate Submission (Prediction)
########################################
echo "Step 5: Generating predictions..."
python src/models/generate_submission.py --config $CONFIG_FILE

echo "Submission files are ready in results/predictions/"
echo "=============================="
echo "         Pipeline Complete!    "
echo "=============================="