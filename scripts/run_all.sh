#!/usr/bin/env bash

# Enable bash debugging
set -ex

echo "=============================="
echo "      Starting the Pipeline   "
echo "=============================="

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
# Removed the --config flag as split_data.py does not accept it.
python src/data/split_data.py

########################################
# 2. Preprocess Data
########################################
echo "Step 2: Preprocessing data..."

#--- Training Data ---
echo "Preprocessing Training Data..."
python src/data/preprocess.py \
    --in_csv="data/intermediate/train.csv" \
    --out_csv="data/intermediate/train_preprocessed.csv" \
    --chunksize=100000 \
    --fillna_method=ffill

#--- Validation Data ---
echo "Preprocessing Validation Data..."
python src/data/preprocess.py \
    --in_csv="data/intermediate/val.csv" \
    --out_csv="data/intermediate/val_preprocessed.csv" \
    --chunksize=100000 \
    --fillna_method=ffill

#--- Test Data ---
echo "Preprocessing Test Data..."
python src/data/preprocess.py \
    --in_csv="data/final/test.csv" \
    --out_csv="data/final/test_preprocessed.csv" \
    --chunksize=100000 \
    --fillna_method=ffill

########################################
# 3. Perform Feature Engineering
########################################
echo "Step 3: Performing feature engineering..."

#--- Training Data ---
echo "Feature Engineering on Training Data..."
python src/features/feature_engineering.py \
    --in_csv="data/intermediate/train_preprocessed.csv" \
    --out_csv="data/intermediate/train_fe.csv" \
    --window_size=5 \
    --bb_window=20

#--- Validation Data ---
echo "Feature Engineering on Validation Data..."
python src/features/feature_engineering.py \
    --in_csv="data/intermediate/val_preprocessed.csv" \
    --out_csv="data/intermediate/val_fe.csv" \
    --window_size=5 \
    --bb_window=20

#--- Test Data ---
echo "Feature Engineering on Test Data..."
python src/features/feature_engineering.py \
    --in_csv="data/final/test_preprocessed.csv" \
    --out_csv="data/final/test_fe.csv" \
    --window_size=5 \
    --bb_window=20

########################################
# 4. Train Models (XGB, LSTM, Stacking)
########################################
echo "Step 4: Training models..."
# train_model.py now handles XGBoost+Optuna, LSTM+Optuna, and Stacking Meta-Learner
python src/models/train_model.py \
    --train_file="data/intermediate/train_fe.csv" \
    --val_file="data/intermediate/val_fe.csv" \
    --batch_size=128 \
    --learning_rate=0.001 \
    --epochs=100 \
    --num_trials=100 \
    --model_type="lstm" \
    --dropout=0.3 \
    --use_attention \
    --ensemble_method="performance" \
    --optimize_window=false \
    --device="mps" \
    --verbose

########################################
# 5. Generate Submission (Prediction)
########################################
echo "Step 5: Generating predictions..."
python src/models/generate_submission.py \
    --xgb_model="results/models/xgb_best_model.json" \
    --lstm_model="results/models/lstm_multistep_optuna.pth" \
    --output_weighted="results/predictions/submission_weighted.csv" \
    --output_stacking="results/predictions/submission_stacking.csv"

echo "Submission files are ready in results/predictions/"
echo "=============================="
echo "         Pipeline Complete!    "
echo "=============================="