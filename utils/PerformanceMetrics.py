# Built-in libraries
import math
import numpy as np
from sklearn import metrics


def smape(A, F):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
        A (numpy.ndarray): Array of true values.
        F (numpy.ndarray): Array of forecasted/predicted values.

    Returns:
        float: The SMAPE score.
    """
    try:
        return (100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))))
    except:
        return np.NaN


def rmse(A, F):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters:
        A (numpy.ndarray): Array of true values.
        F (numpy.ndarray): Array of forecasted/predicted values.

    Returns:
        float: The RMSE score.
    """
    try:
        return math.sqrt(metrics.mean_squared_error(A, F))
    except:
        return np.NaN
    

def hausdorff_distance(A, B):
    """
    Calculate the Hausdorff distance between two sets of one-dimensional points (time series).
    
    Parameters:
        A (numpy.ndarray): Array of shape (n,) representing set A.
        B (numpy.ndarray): Array of shape (m,) representing set B.
        
    Returns:
        float: The Hausdorff distance between sets A and B.
    """
    # Ensure A and B are numpy arrays
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Initialize the maximum of the minimum distances
    max_min_distance = 0
    
    # Compute max min distance from A to B
    for a in A:
        min_distance_to_B = np.min(np.abs(B - a))
        max_min_distance = max(max_min_distance, min_distance_to_B)
    
    # Compute max min distance from B to A
    for b in B:
        min_distance_to_A = np.min(np.abs(A - b))
        max_min_distance = max(max_min_distance, min_distance_to_A)
    
    return max_min_distance


def calculate_ramp_score(y_true, y_pred):
    """
    Calculate the RAMP score (Regression Average Mean Percentage error) for time-series forecasting.

    Parameters:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: The RAMP score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate absolute percentage errors
    abs_percentage_errors = np.abs((y_true - y_pred) / y_true)

    # Replace NaN values with 0 (when y_true is 0)
    abs_percentage_errors[np.isnan(abs_percentage_errors)] = 0

    # Calculate RAMP score
    ramp_score = np.mean(abs_percentage_errors)

    return ramp_score


def RegressionEvaluation(y, pred):
    """
    Evaluate regression model performance using various metrics.

    Parameters:
        y (numpy.ndarray): Array of true values.
        pred (numpy.ndarray): Array of predicted values.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - MAE (float): Mean Absolute Error
            - RMSE (float): Root Mean Squared Error
            - MAPE (float): Mean Absolute Percentage Error
            - SMAPE (float): Symmetric Mean Absolute Percentage Error
            - Hausdorff_score (float): Hausdorff distance score
            - ramp_score (float): Regression Average Mean Percentage Error
            - R2 (float): R-squared score
    """
    # Calculate evaluation metrics
    MAE = metrics.mean_absolute_error(y, pred)
    RMSE = math.sqrt(metrics.mean_squared_error(y, pred))
    SMAPE = smape(y, pred)
    R2 = metrics.r2_score(y, pred)
    
    try:
        y = np.where(y == 0, 1e-6, y)
        MAPE = np.mean(np.abs((y - pred) / y)) * 100.0
    except:
        MAPE = np.NaN
        
    Hausdorff_score = hausdorff_distance(y[:100000], pred[:100000])
    ramp_score = calculate_ramp_score(y, pred)
        
    return MAE, RMSE, MAPE, SMAPE, Hausdorff_score, ramp_score, R2