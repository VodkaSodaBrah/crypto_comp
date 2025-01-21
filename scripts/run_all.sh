#!/usr/bin/env bash

# Enable bash debugging
set -ex

echo "=============================="
echo "      Starting the Pipeline   "
echo "=============================="

# Define config file path
CONFIG_FILE="config.yml"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data/intermediate
mkdir -p data/final
mkdir -p results/models
mkdir -p results/predictions
mkdir -p results/logs/tensorboard
mkdir -p results/logs/tensorboard_submission

##############################
# 1. Split Data
##############################
echo "Step 1: Splitting data..."
python src/data/split_data.py --config "$CONFIG_FILE"

########################################
# 2. Preprocess Data
########################################
echo "Step 2: Preprocessing data..."

#--- Training Data ---
echo "Preprocessing Training Data..."
python src/data/preprocess.py \
    --in_csv="data/intermediate/train.csv" \
    --out_csv="data/intermediate/train_preprocessed.csv" \
    --chunksize=100000

#--- Validation Data ---
echo "Preprocessing Validation Data..."
python src/data/preprocess.py \
    --in_csv="data/intermediate/val.csv" \
    --out_csv="data/intermediate/val_preprocessed.csv" \
    --chunksize=100000

#--- Test Data ---
echo "Preprocessing Test Data..."
python src/data/preprocess.py \
    --in_csv="data/final/test.csv" \
    --out_csv="data/final/test_preprocessed.csv" \
    --chunksize=100000

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
python src/models/train_model.py --config "$CONFIG_FILE"

########################################
# 5. Generate Submissions (Predictions)
########################################
echo "Step 5: Generating predictions for submission..."

# Generate XGBoost predictions
python src/models/generate_submission.py \
    --config "$CONFIG_FILE" \
    --model_type xgb \
    --output_path "results/predictions/xgb_submission.csv"

# Generate LSTM predictions
python src/models/generate_submission.py \
    --config "$CONFIG_FILE" \
    --model_type lstm \
    --output_path "results/predictions/lstm_submission.csv"

# Generate Combined predictions
python src/models/generate_submission.py \
    --config "$CONFIG_FILE" \
    --model_type combined \
    --output_path "results/predictions/combined_submission.csv"

echo "=============================="
echo "         Pipeline Complete!   "
echo "=============================="