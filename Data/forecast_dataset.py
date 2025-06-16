import numpy as np
import torch
from torch.utils.data import Dataset
from abc import abstractmethod
from torch.utils.data import DataLoader, Subset
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import pandas as pd
import os    
    
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IFS_DIR = os.path.join(DATA_DIR, "Real", "IFS", "01-20-2025")
ERA5_DIR = os.path.join(DATA_DIR, "Real", "ERA5")
FIG_DIR = os.path.join(DATA_DIR, "..", "Figures")

class WeatherDataset(Dataset):
    """
    Base class for other datasets containing weather data.
    """
    
    def __init__(self,
                 train_ratio : float,
                 test_ratio : float):
        super(WeatherDataset, self).__init__()

        # Train/test split ratio
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        # Settings for data selection
        self.split_generator = torch.Generator().manual_seed(1)
        self.loader_generator = torch.Generator().manual_seed(2)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        # Only relevant for GP
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def get_train_xy(self):
        self.data_split()

        if self.train_x is None:
            self.train_x = torch.stack([data[0].reshape(-1) for data in self.train_dataset])
            self.train_y = torch.stack([data[1].reshape(-1) for data in self.train_dataset])
        return self.train_x, self.train_y
    
    def get_test_xy(self):
        self.data_split()

        if self.test_x is None:
            self.test_x = torch.stack([data[0].reshape(-1) for data in self.test_dataset])
            self.test_y = torch.stack([data[1].reshape(-1) for data in self.test_dataset])
        
        return self.test_x, self.test_y

    def data_split(self):
        if self.train_dataset is not None: # Already generated
            return

        if self.train_ratio + self.test_ratio == 1.0:
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(
                dataset     = self, 
                lengths     = [self.train_ratio, self.test_ratio],
                generator   = self.split_generator) # For concistency
        else:
            self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(
                dataset     = self, 
                lengths     = [self.train_ratio, 1 - self.train_ratio - self.test_ratio, self.test_ratio],
                generator   = self.split_generator) # For concistency
        
    def get_train_loader(self, batch_size):
        self.data_split()

        if batch_size == 0:
            return [self.get_train_xy()]

        train_dataloader = DataLoader(dataset       = self.train_dataset, 
                                      batch_size    = batch_size,
                                      generator     = self.loader_generator, 
                                      shuffle       = True)
        
        return train_dataloader
    
    def get_test_loader(self, batch_size):
        self.data_split()

        if batch_size == 0:
            test_x, test_y = self.get_test_xy()
            return [(test_x[i], test_y[i]) for i in range(test_x.shape[0])]
        
        test_dataloader = DataLoader(dataset        = self.test_dataset, 
                                      batch_size    = batch_size,
                                      generator     = self.loader_generator, 
                                      shuffle       = True)
        return test_dataloader

    @abstractmethod
    def get_domain_shape(self) -> torch.tensor:
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
    def __init__(self, 
                 observations       : np.array, 
                 training_window    : int,
                 train_ratio        : float,
                 test_ratio         : float,
                 covariances        : np.array = None):
        """
        Dataset class for synthetic data with configurable training window.
        """
        super().__init__(
            train_ratio = train_ratio,
            test_ratio  = test_ratio
            )
        
        # Extract domain shape
        W, V, H, T,  = observations.shape
        self.training_window = training_window

        # Save shape parameters
        self.W = W
        self.T = T
        self.V = V
        self.H = H
        self.observations = torch.Tensor(observations)
        self.domain_shape = torch.tensor([V, H, training_window])
        self.in_channels = W
        if covariances is not None:
            self.covariances = torch.Tensor(covariances)
        else:
            self.covariances = None
        
    def to(self, device):
        self.observations = self.observations.to(device)
        self.domain_shape = self.domain_shape.to(device)
        if self.covariances is not None:
            self.covariances = self.covariances.to(device)
        return self

    def get_domain_shape(self):
        return self.domain_shape
    
    def get_in_channels(self):
        return self.in_channels

    def __len__(self):
        return self.T - self.training_window 

    def __getitem__(self, idx):
        if self.covariances is None:
            return (self.observations[..., idx : idx + self.training_window], 
                self.observations[..., idx + self.training_window])
        else:
            return (self.observations[..., idx : idx + self.training_window], 
                (self.observations[..., idx + self.training_window],
                 self.covariances[..., idx + self.training_window])
                 )
            
    def get_test_xy(self):
        if self.covariances is None:
            return super().get_test_xy()
        
        self.data_split()

        if self.test_x is None:
            self.test_x = torch.stack([data[0].reshape(-1) for data in self.test_dataset])
            self.test_y = (
                torch.stack([data[1][0].reshape(-1) for data in self.test_dataset]),
                torch.stack([data[1][1].reshape(-1) for data in self.test_dataset])
                )
        
        return self.test_x, self.test_y
    
    def get_test_loader(self, batch_size):
        if self.covariances is None:
            return super().get_test_loader(batch_size)
            
        self.data_split()

        if batch_size == 0:
            test_x, test_y = self.get_test_xy()
            return [(test_x[i], (test_y[0][i], test_y[1][i])) for i in range(test_x.shape[0])]
        
        test_dataloader = DataLoader(dataset        = self.test_dataset, 
                                      batch_size    = batch_size,
                                      generator     = self.loader_generator, 
                                      shuffle       = True)
        return test_dataloader
        
    # def get_train_xy(self):
    #     if self.covariances is None:
    #         return  (torch.stack([self.observations[..., idx : idx + self.training_window] 
    #                   for idx in range(self.reserve_days - self.training_window)]),
    #                  torch.stack([self.observations[..., idx + self.training_window]
    #                   for idx in range(self.reserve_days - self.training_window)]))
                
    #     else:
    #         raise ValueError("Parameter prediction does not allow for constructing train_x and train_y.")

class NextDayForecastDataset(PastNDaysForecastDataset):
    def __init__(self, observations : np.array):
        super().__init__(observations, 1)
    
class Past30DaysForecastDataset(PastNDaysForecastDataset):
    def __init__(self, observations : np.array):
        super().__init__(observations, 30)
        
class GridStationDataset(WeatherDataset):
    def __init__(self,
                 grid_ds: xr.Dataset,
                 station_df: pd.DataFrame,
                 sta_df : pd.DataFrame,
                 output_weather_vars : list,
                 training_window: int):
        """
        Dataset class for real-world dataset.
        """
        super().__init__(train_ratio=1.0, test_ratio=0.0)

        # Save parameters
        self.training_window = training_window
        sta_df = sta_df
        self.output_weather_vars = output_weather_vars
        self.output_weather_vars_kis = [self.era5_name_to_kis_name(x) for x in output_weather_vars]


        # Extract and sort time
        self.times = grid_ds.time.values
        self.grid_ds = grid_ds.transpose('time', 'latitude', 'longitude')
        self.lat = grid_ds.latitude.values
        self.lon = grid_ds.longitude.values
        self.weather_vars = list(grid_ds.data_vars)

        # Prepare input tensor of size [N, W, H, V]
        self.X = np.stack([grid_ds[var].values for var in self.weather_vars], axis=1)  
        self.X = torch.tensor(self.X, dtype=torch.float32)

        # Station metadata
        self.station_df = station_df.copy()
        self.station_df['validTime'] = pd.to_datetime(self.station_df['validTime'])
        self.station_df = self.station_df.sort_values('validTime')

        # Train / test selection
        test_start_time = self.station_df['validTime'].min()
        self.test_start_idx = np.where(self.times >= np.datetime64(test_start_time))[0][0]

        self.train_indices = list(range(training_window, self.test_start_idx))
        self.shuffle_train_indices()
        self.test_times = np.unique(station_df["validTime"].values)
        self.test_indices = list(range(self.test_start_idx + training_window, 
                                       self.test_start_idx + len(self.test_times)))

        # Data shapes
        self.in_channels = len(self.weather_vars)
        self.domain_shape = torch.tensor([len(self.lat), len(self.lon), training_window])

        # Output locations
        self.station_numbers = np.sort(self.station_df["station"].unique())
        self.sta_df = sta_df.sort_values('sta_id')
        station_selection = self.sta_df[self.sta_df["sta_id"].isin(self.station_numbers)]
        self.station_coords = {k:v for k,v in zip(self.station_numbers, station_selection[["sta_lat", "sta_lon"]].values)}
        self.interpolation_coords = np.array([self.station_coords[stn] for stn in self.station_numbers])
        # Precompute the interpolated output values
        self.precompute_outputs()
        
        # Define the train and test datasets
        self.train_dataset =  Subset(self, self.train_indices)
        self.test_dataset = Subset(self, self.test_indices)
        
        # Precompute structures for kis
        self.precompute_kis_obs()
        
    def era5_name_to_kis_name(self, name):
        match name:
            case "t2m":
                return "T_DRYB_10"
            case "d2m":
                return "T_DEWP_10"
            case _:
                return "???"
        
        
    def precompute_outputs(self):
        """Precompute interpolated outputs at station locations for all timesteps."""
        self.output_cache = {}

        for t_idx in range(len(self.times)):
            interp_values = []
            for var in self.output_weather_vars:
                values = self.grid_ds[var].isel(time=t_idx).values
                interpolator = RegularGridInterpolator((self.lat, self.lon), values, bounds_error=False)
                result = interpolator(self.interpolation_coords)  # Shape: [#stations]
                interp_values.append(result)
            self.output_cache[t_idx] = torch.tensor(np.stack(interp_values), dtype=torch.float32)  # [M, O]
            
    def interpolate_era5(self, X: torch.Tensor) -> torch.Tensor:
        """
        Interpolates ERA5 data to target coordinates for each batch and weather variable.

        Parameters:
            X (Tensor): Input ERA5 data of shape [B, W, V, H]
        
        Returns:
            Tensor: Interpolated values of shape [B, W_out, M]
        """
        X_np = X.cpu().numpy()  # [B, W, V, H]
        B = X_np.shape[0]
        M = self.interpolation_coords.shape[0]  # number of target stations
        W_out = len(self.output_weather_vars)

        # Output: [B, W_out, M]
        interp_values = np.empty((B, M, W_out), dtype=np.float32)

        for b in range(B):
            for i, wv in enumerate(self.output_weather_vars):
                w_idx = self.weather_vars.index(wv)
                values = X_np[b, w_idx]  # shape [V, H]
                interpolator = RegularGridInterpolator((self.lat, self.lon), values, bounds_error=False)
                interp_values[b, :, i] = interpolator(self.interpolation_coords).T

        return torch.from_numpy(interp_values).to(X.device)  # [B, W_out, M]

    def shuffle_train_indices(self):
        self.permuted_indices = np.array(torch.randperm(len(self.train_indices), generator=self.loader_generator).tolist())
        self.train_indices = np.array(self.train_indices)[self.permuted_indices]

    def __len__(self):
        return len(self.train_indices) + len(self.test_indices)
    
    def get_test_idx(self, idx):
        out = idx - np.max(self.train_indices) - 1 - self.training_window
        
        assert out >= 0, "Test index not available: perhaps between (end train, end_train + training_window)?"
        return out

    def get_train_idx(self, idx):
        return idx - self.training_window

    def __getitem__(self, idx):
        max_train_idx = np.max(self.train_indices)
        is_train = idx <= max_train_idx
        index = self.train_indices[self.get_train_idx(idx)] if is_train else self.test_indices[self.get_test_idx(idx)]

        input_tensor = self.X[index - self.training_window:index].permute(1, 2, 3, 0)  # [W, V, H, T]
        output_tensor = self.output_cache[index].T # [M, O]; M = number of locations, C = number of output weather vars

        if is_train:
            return input_tensor, output_tensor
        else:
            kis_tensor = self.get_kis_obs(idx)
            return input_tensor, (output_tensor, kis_tensor)
    
    def to(self, device):
        self.X = self.X.to(device)
        self.kis_obs_data = torch.tensor(self.kis_obs_data, device = device, dtype=torch.float32)
        for key, value in self.output_cache.items():
            self.output_cache[key] = self.output_cache[key].to(device)
        self.device = device
        return self
    
    def precompute_kis_obs(self):
        self.kis_obs_data = np.empty(shape=(
            len(self.test_indices), 
            len(self.station_numbers),
            len(self.output_weather_vars)))
        for i, idx in enumerate(self.test_indices):
            is_train = idx <= np.max(self.train_indices)
            if is_train:
                raise ValueError(f"Training dataset does not have kis values. Choose idx >= {len(self.train_indices)}")
            
            index = self.test_indices[self.get_test_idx(idx)] - self.test_start_idx
            time = self.test_times[index]
            obs_selection = self.station_df[self.station_df["validTime"] == time]
            obs_ordered = obs_selection.sort_values('station')[[*self.output_weather_vars_kis]]
            obs_t = obs_ordered.values
            self.kis_obs_data[i] = obs_t
            
        pass
    
    def get_kis_obs(self, idx):
        return self.kis_obs_data[idx - min(self.test_indices)]
    
    def get_timestamp(self, idx):
        max_train_idx = np.max(self.train_indices)
        is_train = idx <= max_train_idx
        index = self.train_indices[idx] if is_train else self.test_indices[self.get_test_idx(idx)]
        print(index)
        
        return self.times[index]

    def get_domain_shape(self):
        return self.domain_shape

    def get_in_channels(self):
        return self.in_channels
    
    def get_train_loader(self, batch_size):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size, shuffle = True):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_num_train_data(self):
        return len(self.train_indices)


def create_GridStationDataset(training_window : int):
    # ERA5 data
    era5 = xr.open_dataset(os.path.join(ERA5_DIR, "final_era5_data.nc"), engine="netcdf4", decode_timedelta = True).load()

    # KIS data
    kis = pd.read_pickle(os.path.join(IFS_DIR, "all_data_kiri.pkl")) 
    kis_obs = kis[[c for c in kis.columns if "IFS" not in c]]

    # Station data
    sta_df = pd.read_csv(os.path.join(IFS_DIR, "sta_gp_metadata.csv"))
    
    return GridStationDataset(
        era5,
        kis_obs,
        sta_df,
        ["t2m", "d2m"],
        training_window    
    )