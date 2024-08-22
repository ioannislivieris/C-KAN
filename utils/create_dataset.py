import numpy as np
from tqdm import tqdm

def create_dataset(df=None, Lag=20, Horizon=4, overlap=1):
    """
    Creates a dataset for time series prediction by splitting the input DataFrame into input sequences and corresponding target sequences.

    Args:
        df (DataFrame, optional): The input DataFrame containing time series data. Default is None.
        Lag (int, optional): The number of time steps to lag behind for input sequences. Default is 1.
        Horizon (int, optional): The number of time steps ahead for target sequences. Default is 1.
        overlap (int, optional): The number of time steps to overlap between consecutive sequences. Default is 1.

    Returns:
        tuple: A tuple containing three numpy arrays:
            - dataX: Input sequences with shape (num_samples, Lag, num_features).
            - dataY: Target sequences with shape (num_samples, Horizon).
            - dataDate: List of lists containing the dates corresponding to each target sequence.
    """
    dataX, dataY, dataDate = [], [], []

    for i in tqdm(range(0, df.shape[0] + 1 - Lag - Horizon, overlap)):

        dataX.append(df.to_numpy()[i:(i + Lag)])
        dataY.append(df.to_numpy()[i + Lag: i + Lag + Horizon])
        dataDate.append(df.index[i + Lag: i + Lag + Horizon].tolist())

    return (np.array(dataX), np.array(dataY), np.array(dataDate))
