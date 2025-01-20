import numpy as np
import torch

def create_time_features(timestamps, num_features):
    """
    Generate time-based features (hour and day of week) normalized between 0 and 1.
    
    Parameters:
    - timestamps: A pandas Series of datetime objects.
    - num_features: Integer representing the number of time steps in the sequence.
    
    Returns:
    - A PyTorch tensor of time features with shape (batch_size, num_features, 2).
    """
    time_features = np.column_stack((
        timestamps.dt.hour / 23.0,
        timestamps.dt.dayofweek / 6.0
    ))
    time_features = np.repeat(time_features[:, np.newaxis, :], num_features, axis=1)
    return torch.tensor(time_features, dtype=torch.float32)