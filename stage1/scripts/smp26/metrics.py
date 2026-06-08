import numpy as np


def mape(y_true, y_pred):
    y_true = np.maximum(np.abs(y_true), 1e-8)
    return np.mean(np.abs((y_true - y_pred) / y_true))
