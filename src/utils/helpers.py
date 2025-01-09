import numpy as np

def custom_log_transform(series):
    return np.log(series + 1e-9)