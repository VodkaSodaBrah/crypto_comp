# src/models/infer.py

import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load environment variables from .env file
load_dotenv()

# Constants
API_URL = "https://api-inference.huggingface.co/models/dblOtech/my-crypto-prediction-model"  # Replace with your model's URL
API_TOKEN = os.getenv("HF_API_TOKEN")  # Your Hugging Face API token

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def query(payload):
    """
    Sends a POST request to the Hugging Face Inference API with the given payload.

    Args:
        payload (dict): The input data for the model.

    Returns:
        dict: The JSON response from the API containing predictions.
    """
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")

def prepare_payload(time_series_data):
    """
    Prepares the payload for the API request.

    Args:
        time_series_data (list of lists): Your time series data, where each sublist represents features at a timestep.

    Returns:
        dict: The payload formatted for the API.
    """
    return {
        "inputs": [time_series_data]
    }

def load_and_prepare_data(input_csv, window_size=5):
    """
    Loads data from a CSV file, applies preprocessing, and extracts the latest window.

    Args:
        input_csv (str): Path to the input CSV file containing the time series data.
        window_size (int): Number of timesteps to include in the input window.

    Returns:
        list of lists: The prepared time series data for inference.
    """
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Select relevant features
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]

    # Load the saved scaler
    scaler = joblib.load("results/scalers/scaler.pkl")
    scaled_features = scaler.transform(df[feature_cols])

    # Replace original features with scaled features
    df[feature_cols] = scaled_features

    # Ensure there are enough timesteps
    if len(df) < window_size:
        raise ValueError(f"Not enough data to create a window of size {window_size}.")

    # Extract the last 'window_size' timesteps
    latest_window = df.iloc[-window_size:][feature_cols].values.tolist()

    return latest_window

def main():
    """
    Main function to execute the inference.
    """
    # Path to your input data CSV
    input_csv = "data/intermediate/new_data.csv"  # Update this path as needed

    # Define window size (must match training)
    window_size = 5

    # Load and prepare data
    time_series_data = load_and_prepare_data(input_csv, window_size=window_size)

    payload = prepare_payload(time_series_data)
    
    try:
        output = query(payload)
        print("Prediction Output:")
        print(output)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()