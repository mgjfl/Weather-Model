import numpy as np
import torch
from torch.utils.data import Dataset
from abc import abstractmethod

class WeatherDataset(Dataset):
    
    def __init__(self):
        super(WeatherDataset, self).__init__()
        
    @abstractmethod
    def get_input_shape(self) -> torch.tensor:
        """Returns the shape of the input grid.

        Returns:
            torch.tensor: shape of the input grid
        """
        raise NotImplementedError("Subclass must implement this method.")
    
    @abstractmethod
    def get_in_channels(self) -> int:
        """Returns the number of input channels.

        Returns:
            torch.tensor: number of input channels
        """
        raise NotImplementedError("Subclass must implement this method.")

class PastNDaysForecastDataset(WeatherDataset):
    def __init__(self, observations : np.array, days : int):
        """_summary_

        Args:
            observations (np.array): shape (# days) x (# weather vars) x (# stations)
            n (int): _description_
            device (str): _description_
        """
        self.days = days
        n, w, d = observations.shape
        self.n = n
        self.w = w
        self.d = d
        self.observations = torch.Tensor(observations)
        self.input_shape = torch.tensor([self.days, self.d])
        self.in_channels = w
        
    def to(self, device):
        self.observations = self.observations.to(device)
        self.input_shape = self.input_shape.to(device)
        return self

    def get_input_shape(self):
        return self.input_shape
    
    def get_in_channels(self):
        return self.in_channels

    def __len__(self):
        return len(self.observations) - self.days
    
    def t(self, x):
        return torch.transpose(x, 0, 1)

    def __getitem__(self, idx):
        return (self.t(self.observations[idx : idx + self.days]), 
                self.observations[idx + self.days])

class NextDayForecastDataset(PastNDaysForecastDataset):
    def __init__(self, observations : np.array):
        super().__init__(observations, 1)
    
class Past30DaysForecastDataset(PastNDaysForecastDataset):
    def __init__(self, observations : np.array):
        super().__init__(observations, 30)