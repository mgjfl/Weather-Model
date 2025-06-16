import torch
import torch.nn as nn
import torch.nn.functional as F
from GP import *
from torch.utils.data import DataLoader
from Data.forecast_dataset import GridStationDataset
from Models.AFNO import AFNONet
from linear_operator.operators import DiagLinearOperator, BlockDiagLinearOperator

class StationVGP(nn.Module):
    """
    A wrapper for a GP model that properly shapes input and output.
    """
    def __init__(self, 
                 mean_model_class       : nn.Module, 
                 gp_model_class         : nn.Module,
                 in_channels            : int, 
                 domain_shape           : torch.tensor, 
                 num_data               : int,
                 dataset                : GridStationDataset,
                 **model_kwargs):
        super(StationVGP, self).__init__()
        

        # The Gaussian Process model
        self.gp_model_class = gp_model_class
        self.mean_model_class = mean_model_class

        # Specify the input
        self.in_channels            = in_channels
        self.domain_shape           = domain_shape
        self.dataset                = dataset

        # Compute the GP parameters
        self.input_dim = in_channels * torch.prod(domain_shape)
        self.output_locations = len(dataset.station_coords)
        self.output_weather_types = len(dataset.output_weather_vars)
        self.output_dim = self.output_locations * self.output_weather_types

        # Model specific arguments
        self.model_kwargs           = model_kwargs

        # Create the GP model
        self.gp_model = self._create_gp_model()
        self.mean_model = self._create_mean_model()

        # First apply a prior computation via 

        # The likelihood and num_data
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim)
        self.likelihood.register_constraint("raw_task_noises", gpytorch.constraints.GreaterThan(1e-3))
        self.num_data = num_data
        
        # The projection layer to the stations
        self.projection = GridProjection(dataset.lat, 
                                         dataset.lon, 
                                         in_channels, 
                                         self.output_weather_types, 
                                         dataset.device)
        
        self.target_coords = torch.tensor(dataset.interpolation_coords).to(dataset.device)

        # Store for loss computation
        self.latest_xm = None
        self.latest_gp_out = None
        
        # For AFNO
        if issubclass(self.mean_model_class, AFNONet):
            V, H, T = self.domain_shape
            
            self.new_h = (H // self.model_kwargs["patch_size"][1]) * self.model_kwargs["patch_size"][1]
            self.new_v = (V // self.model_kwargs["patch_size"][0]) * self.model_kwargs["patch_size"][0]
            
            self.linear_projection = nn.Linear(in_channels * self.new_h * self.new_v, self.output_dim)
            
        
        # Register normalization buffers
        self.initialize_normalization_stats()
        
        
        gpytorch.settings.cholesky_max_tries(5)
        gpytorch.settings.cholesky_jitter(1e-2)
        print("Model Definition Finished.")
        
        
        

    def _create_gp_model(self):
        return self.gp_model_class(
            input_dim     = self.input_dim,
            output_dim    = self.output_dim, 
            out_channels    = self.in_channels, 
            **self.model_kwargs)
    
    def _create_mean_model(self):
        return self.mean_model_class(
            in_channels     = self.in_channels,
            domain_shape    = self.domain_shape, 
            out_channels    = self.in_channels, 
            img_size        = (self.domain_shape[0], self.domain_shape[1]),
            in_chans        = self.in_channels * self.domain_shape[2],
            out_chans       = self.in_channels * self.domain_shape[2],
            **self.model_kwargs)
    
    def forward(self, x):
        # Safeguard: Ensure normalization buffers have been set
        if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in [
            'input_mean', 'input_std', 'output_mean', 'output_std'
        ]):
            raise RuntimeError("Normalization statistics (input_mean, input_std, output_mean, output_std) must be initialized before forward pass.")

        # The baseline: interpolation of last known weather state
        x_base = self.dataset.interpolate_era5(x[:, :, :, :, -1])
        
        # Normalize input weather variables
        x_norm = (x - self.input_mean) / self.input_std
        
        if issubclass(self.mean_model_class, AFNONet):
            x_norm = x_norm.permute(0, 4, 1, 2, 3)
            B, T, W, V, H = x_norm.shape
            x_norm = x_norm.reshape(B, T * W, V, H)
            
        if not issubclass(self.mean_model_class, MGST):
            # Compute mean
            xm = self.mean_model(x_norm)  # [B, W, V, H, T]
            xm = F.dropout(xm, p=0.5, training=self.training)  
        else:
            xm = 0
            
        if issubclass(self.mean_model_class, AFNONet):
            # [B, T * W, V, H]
            # xm = xm.view(B, -1)
            xm = xm.view(B,T, W, self.new_v, self.new_h)
            xm = xm.mean(dim = 1) # [B, W, V, H]
            xm = xm.view(B, -1)
            xm = self.linear_projection(xm)
            xm = xm.reshape(B, self.output_locations, self.output_weather_types) # [B, M, C]
            
            w1_res = xm[:, :, 1]
            w_delta = x_base[:, :, 0] - x_base[:, :, 1]
            w0_res = w1_res - w_delta + (xm[:, :, 0]).abs()
            
            # Full mean
            x_full_mean = x_base + torch.stack([w0_res, w1_res], axis = 2)
            
        elif not issubclass(self.mean_model_class, MGST):         
            xm = self.projection(xm, self.target_coords, is_base = False) # [B, M, C]; M = number of locations, C = number of output weather vars
            xm = xm * self.output_std
        
            # Full mean
            x_full_mean = x_base + xm

        self.latest_x_base = x_base
        self.latest_xm = xm
        

        # Reshape to [n, d] for GP input
        x_flat = x_norm.reshape(x.shape[0], -1)
        
        with gpytorch.settings.cholesky_jitter(1e-2):

            # Get GP distribution (zero mean) with rescaled covariance
            gp_out = self.gp_model(x_flat)  # MultivariateNormal with zero mean
            
            # Scale variance again
            num_outputs = self.output_std.shape[0]
            B, D = gp_out.mean.shape
            locations = D // num_outputs

            scaling_vector = self.output_std.repeat_interleave(locations)  # shape [num_outputs * N]
            scaling_vector = scaling_vector.repeat(B)
            
            # Use this as the full diagonal scaling matrix (S)
            S = DiagLinearOperator(scaling_vector)
            scaled_covar = (S @ gp_out.lazy_covariance_matrix @ S).add_jitter(1e-1)


            self.latest_gp_out = gp_out

            # Add deterministic mean from mean_model
            final_mean = x_full_mean.reshape(x_full_mean.shape[0], -1)
            # print(f"{torch.mean(final_mean)}=")
            final_mvn = gpytorch.distributions.MultitaskMultivariateNormal(
                final_mean, scaled_covar
                )


            return final_mvn
    
    def initialize_normalization_stats(self, batch_size=32):
        """
        Compute and set normalization statistics for input and output in the model from GridStationDataset.

        Parameters:
            model (nn.Module): The model with registered input/output mean/std buffers.
            dataset (GridStationDataset): Dataset instance returning (x, y).
            batch_size (int): Batch size for statistics computation.
        """
        
        print("Computing normalization stats...")
        
        with torch.no_grad():

            loader = self.dataset.get_train_loader(batch_size)
            
            input_sumsq = 0
            input_sum = 0
            input_count = 0

            output_sumsq = 0
            output_sum = 0
            output_count = 0

            for x, y in loader:
                # x : [B, W, V, H, T]
                # y : [B, M, C]

                W = x.shape[1]
                C = y.shape[-1]

                x_ = x.permute(1, 0, 2, 3, 4).reshape(W, -1)  # [W, B*V*H*T]
                input_sum += x_.sum(dim=1)                   # [W]
                input_sumsq += (x_ ** 2).sum(dim=1)          # [W]
                input_count += x_.shape[1]

                y_ = y.reshape(-1, C)                         # [B*M, C]
                output_sum += y_.sum(dim=0)                  # [C]
                output_sumsq += (y_ ** 2).sum(dim=0)         # [C]
                output_count += y_.shape[0]

            # Compute statistics
            input_mean = input_sum / input_count                        # [W]
            input_var = input_sumsq / input_count - input_mean ** 2     # [W]
            input_std = input_var.clamp_min(1e-6).sqrt()                # [W]

            output_mean = output_sum / output_count                     # [C]
            output_var = output_sumsq / output_count - output_mean ** 2 # [C]
            output_std = output_var.clamp_min(1e-6).sqrt()              # [C]

            # Reshape and assign to model           
            self.register_buffer("input_mean", input_mean.view(-1, 1, 1, 1))
            self.register_buffer("input_std", input_std.view(-1, 1, 1, 1))
            self.register_buffer("output_mean", output_mean)
            self.register_buffer("output_std", output_std)
        


class GridProjection(nn.Module):
    def __init__(self, lat_grid: torch.Tensor, lon_grid: torch.Tensor, in_channels: int, out_channels: int, dev):
        """
        GridProjection performs differentiable interpolation from regular grid outputs to fixed coordinates.

        Parameters:
            lat_grid (Tensor): 1D tensor of latitudes, shape [H] (in degrees).
            lon_grid (Tensor): 1D tensor of longitudes, shape [V] (in degrees).
            in_channels (int): Number of input weather variables W.
            out_channels (int): Number of output weather variables (usually 2).
        """
        super().__init__()
        self.H = len(lat_grid)
        self.V = len(lon_grid)
        self.register_buffer('lat_min', torch.tensor(lat_grid.min()).to(dev))
        self.register_buffer('lat_max', torch.tensor(lat_grid.max()).to(dev))
        self.register_buffer('lon_min', torch.tensor(lon_grid.min()).to(dev))
        self.register_buffer('lon_max', torch.tensor(lon_grid.max()).to(dev))

        # Projection block (e.g., reduce W → 2 over channels using 1×1 conv)
        self.projection_base = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
        self.projection_residual = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def normalize_coords(self, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize (lon, lat) coords to [-1, 1] range for grid_sample.

        Parameters:
            target_coords (Tensor): [B, M, 2]; [:,:,0]=lon, [:,:,1]=lat (in degrees)

        Returns:
            norm_coords (Tensor): [B, M, 1, 2] normalized for grid_sample
        """
        lat = target_coords[..., 0]
        lon = target_coords[..., 1]
        

        x = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1  # [M]
        y = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1  # [M]

        grid = torch.stack([x, y], dim=-1).type(torch.float32)  # [M, 2]
        return grid.unsqueeze(1)  # [M, 2]

    def forward(self, x: torch.Tensor, target_coords: torch.Tensor, is_base : bool) -> torch.Tensor:
        """
        Project spatiotemporal input to target coordinates.

        Parameters:
            x (Tensor): Input tensor of shape [B, W, V, H, T]
            target_coords (Tensor): Station coords [B, M, 2]; (lon, lat) in degrees

        Returns:
            interpolated (Tensor): Interpolated output [B, M, out_channels]
        """
        B, W, V, H, T = x.shape
        x = x.permute(0, 1, 4, 3, 2)  # -> [B, W, T, H, V] for Conv3d

        # Apply projection (e.g., channel reduction W -> 2)
        if is_base:
            x_proj = self.projection_base(x)  # [B, out_channels, T, H, V]
        else:
            x_proj = self.projection_residual(x)  # [B, out_channels, T, H, V]
        x_proj = x_proj.permute(0, 2, 1, 3, 4)  # -> [B, T, out_channels, H, V]


        norm_coords = self.normalize_coords(target_coords).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, M, 1, 2]

        # Reshape for batched interpolation over time
        B, T, C_out, H, V = x_proj.shape
        x_flat = x_proj.reshape(B * T, C_out, H, V)  # [B*T, C, H, V]

        coords = norm_coords.repeat_interleave(T, dim=0)  # [B*T, M, 1, 2]

        

        sampled = F.grid_sample(x_flat, coords, mode='bilinear', align_corners=True)  # [B*T, C, M, 1]        
        sampled = sampled.squeeze(-1).transpose(1, 2)  # [B*T, M, C]

        sampled = sampled.view(B, T, -1, C_out)  # [B, T, M, C]
        output = sampled.mean(dim=1)  # [B, M, C]

        return output