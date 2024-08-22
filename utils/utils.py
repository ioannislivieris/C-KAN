import random
from datetime import datetime, timedelta
from torch.utils.data import Dataset

class Data(Dataset):
    """
    A custom dataset class to handle data with features (X) and labels (Y).
    
    Args:
        X (list or numpy.ndarray): Features.
        Y (list or numpy.ndarray): Labels.
    """

    def __init__(self, X=None, Y=None, XTime=None, YTime=None):
        """
        Initializes the Data object.

        Args:
            X (list or numpy.ndarray): Features.
            Y (list or numpy.ndarray): Labels.
        """
        self.X = X
        self.Y = Y
        self.XTime = XTime
        self.YTime = YTime

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.Y)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and the label of the sample.
        """
        if self.XTime is None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx], self.Y[idx], self.XTime[idx], self.YTime[idx]
    
    

def random_date(start_date_str='2000-01-01', end_date_str='2020-01-01'):
    # Convert start and end dates from string to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Calculate the range of days between start and end dates
    delta = end_date - start_date

    # Generate a random number of days within the range
    random_days = random.randint(0, delta.days)

    # Create a random timedelta within the range
    random_timedelta = timedelta(days=random_days)

    # Calculate the random date by adding the random timedelta to the start date
    random_datetime = start_date + random_timedelta

    # # Construct the final random date and time string
    # random_date_str = random_datetime.strftime('%Y-%m-%d %H:%M:%S') 

    return random_datetime