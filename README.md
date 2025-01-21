# Cryptocurrency Price Prediction Pipeline

This repository contains the code and resources to predict the direction of cryptocurrency price movements using machine learning models. The pipeline is designed to preprocess data, engineer features, train and validate multiple models, and evaluate their performance using the Macro-Averaged F1 Score.

**Note:** This submission adheres to the competition requirements, including public availability of code and reproducibility of results.

---

## Table of Contents
1. [Overview](#overview)
2. [Pipeline Workflow](#pipeline-workflow)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [How to Run the Pipeline](#how-to-run-the-pipeline)
6. [Model Evaluation](#model-evaluation)
7. [Reproducing Results](#reproducing-results)
8. [File Structure](#file-structure)
9. [Contact](#contact)

---

## Overview

The pipeline builds robust predictors for short-term cryptocurrency price movements using the following steps:
1. **Data Preprocessing:** Sequential splitting and normalization to preserve time-series integrity.
2. **Feature Engineering:** Includes technical indicators (e.g., RSI, MACD, Bollinger Bands).
3. **Model Training:**
   - **XGBoost:** Trained with Optuna for hyperparameter optimization.
   - **LSTM:** Leveraging sequential patterns with optional attention mechanisms.
4. **Model Evaluation:** Macro-Averaged F1 Score to ensure balanced performance across classes.
5. **Reproducibility:** Code is designed for easy replication of results.

---

## Pipeline Workflow

The pipeline follows these key stages:
1. **Data Collection:**
   - Load raw cryptocurrency price data.
   - Save to CSV for consistent input.
2. **Feature Engineering:**
   - Apply rolling statistics, exponential moving averages, and technical indicators.
3. **Model Training:**
   - **XGBoost:** Baseline and optimized models using Optuna.
   - **LSTM:** Multi-step LSTM with sliding window, optional attention, and stacked layers.
4. **Evaluation:**
   - Compute Macro-Averaged F1 Score on holdout data.
5. **Model Saving:**
   - Save trained models and metrics to the `results/models/` directory.

---

## Prerequisites

### Tools Required
- **Python:** Version 3.10 or later
- **Conda:** Anaconda or Miniconda installed

---

## Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/VodkaSodaBrah/crypto_prediction_pipeline.git
cd crypto_prediction_pipeline

Step 2: Create the Conda Environment
conda env create -f environment.yml

Step 3: Activate the Environment
conda activate my_crypto_env

How to Run the Pipeline
Run All Steps
bash run_all.sh

Outputs
	1.	Preprocessed data will be saved to the data/processed/ directory.
	2.	Trained models will be saved to the results/models/ directory.
	3.	Logs and evaluation metrics will be available in the results/logs/ directory.
	4.	Predictions will be saved in the results/ directory:
	•	xgb_submission.csv
	•	lstm_submission.csv
	•	combined_submission.csv

Model Evaluation

The pipeline uses the Macro-Averaged F1 Score as the primary evaluation metric to ensure balanced performance across classes.

The F1 Score is calculated as the harmonic mean of Precision and Recall, which balances the trade-off between false positives and false negatives. The formula for F1 Score is:

Two times the product of Precision and Recall, divided by the sum of Precision and Recall.

The Macro-Averaged F1 Score is the mean of F1 Scores computed for each class. It is calculated by summing up the F1 Scores for all classes and dividing by the total number of classes.

Here, “C” represents the total number of classes, which are two in this case (up or not up). “F1_c” refers to the F1 Score for a specific class. This evaluation metric ensures that the model performs well across all classes, maintaining a balance between precision and recall.

Formula

F1 Score:


F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}


Macro-Averaged F1 Score:


F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F1_c


Where:
	•	C: Total number of classes (2: up or not up).
	•	F1_c: F1 score for each class.
Contact

For questions or clarifications, please reach out:
	•	Name: Michael Childress
	•	Email: mchildress@me.com
	•	GitHub: https://github.com/VodkaSodaBrah

	