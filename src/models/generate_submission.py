import argparse
import pandas as pd
import os
import yaml
import logging
import xgboost as xgb

def load_config(config_path):
    """
    Loads configuration from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_submission(config):
    """
    Generates submission predictions using the trained models.
    """
    # Load test data
    try:
        test_df = pd.read_csv('data/final/test_fe.csv')
        logger.info("Test data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Load XGBoost model
    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model(config['model_paths']['xgb'])
        logger.info("XGBoost model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {e}")
        return
    
    # Prepare data for prediction
    try:
        X_test = test_df.drop(columns=['target'])  # Assuming 'target' is not present in test
        dtest = xgb.DMatrix(X_test)
    except Exception as e:
        logger.error(f"Error preparing test data for prediction: {e}")
        return
    
    # Generate predictions
    try:
        preds = xgb_model.predict(dtest)
        preds_binary = [1 if pred > 0.5 else 0 for pred in preds]
        logger.info("Predictions generated successfully.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return
    
    # Save predictions
    try:
        submission = pd.DataFrame({
            'timestamp': test_df['timestamp'],
            'prediction': preds_binary
        })
        submission.to_csv(config['output_paths']['submission_weighted'], index=False)
        logger.info(f"Submission file saved to {config['output_paths']['submission_weighted']}")
    except Exception as e:
        logger.error(f"Error saving submission file: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description="Generate Submission Predictions.")
    parser.add_argument("--config", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    
    generate_submission(config)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/generate_submission.log")
        ]
    )
    logger = logging.getLogger()

    main()