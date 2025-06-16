from .neural_models import *
import torch.nn as nn
from Data import sigma_w, sigma_x
from itertools import product 

class UnitCov(nn.Module):
    """
    Returns the identity matrix as covariance matrix, regardless of the input.
    """
    def __init__(self,
                 in_channels   : int, 
                 domain_shape  : torch.tensor):
        
        super().__init__()
        
        # Save parameters
        self.in_channels    = in_channels
        self.domain_shape   = domain_shape
        
        # Extract shape
        self.mean_shape = torch.tensor([in_channels, domain_shape[0], domain_shape[1]])
        self.cov_shape  = torch.tensor([*self.mean_shape, *self.mean_shape])
        
    def forward(self, x):

        unit_matrix = torch.eye(torch.prod(self.mean_shape)).reshape(self.cov_shape)
        unit_matrix = unit_matrix.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        return unit_matrix
        
class ParamCov(nn.Module):
    """
    The parametric covariance structure as defined by the synthetic data.
    """
    def __init__(self,
                 in_channels   : int, 
                 domain_shape  : torch.tensor):
        
        super().__init__()
        
        # Save parameters
        self.device = torch.device("cuda")
        self.in_channels    = in_channels
        self.domain_shape   = domain_shape
        
        # Extract shape
        self.W = in_channels
        self.V = domain_shape[0]
        self.H = domain_shape[1]
        self.training_days = domain_shape[2]
        self.mean_shape = torch.tensor([self.W, self.V, self.H], device=self.device)
        
        self.cov_shape  = torch.tensor([*self.mean_shape, *self.mean_shape], device=self.device)

        # The model parameters
        self.params = torch.nn.Parameter(torch.ones(2)).to(self.device)
        self.eps = torch.tensor(2.5, device=self.device)
        self.h = self.params[0]
        self.c = self.params[1]

        # Fixed computations
        long_range  = np.linspace(-1, 1, self.H)
        lat_range   = np.linspace(-1, 1, self.V)

        # Compute all pairs to construct the covariance matrices
        X_grid = np.meshgrid(long_range, lat_range)
        Z_grid = np.stack(X_grid, axis = 2)
        Z_grid = Z_grid.reshape(-1, Z_grid.shape[-1])

        self.space_pairs = torch.tensor(
            np.array(list(product(Z_grid, repeat=2))), device=self.device
            )
        self.type_pairs  = torch.tensor(
            np.array(list(product(np.arange(self.W), repeat=2))), device=self.device
        )

        # Linear layer on the frequencies to extract the time-dependent part
        self.linear = nn.Linear(self.W * self.V * self.H * (self.training_days // 2 + 1), 1).to(self.device)  # Linear layer mapping frequencies to a single parameter
        
        # Precompute the covariance matrices Sx and Sw for constant pairs
        self.Sx = torch.tensor([sigma_x(*z, self.eps, self.h) for z in self.space_pairs], device=self.device).reshape(self.H * self.V, self.H * self.V)
        self.Sw = torch.tensor([sigma_w(*tp, self.c) for tp in self.type_pairs], device=self.device).reshape(self.W, self.W)

        # Precompute Kronecker product of Sw and Sx
        self.Swx = torch.kron(self.Sw, self.Sx).reshape(self.W, self.V, self.H, self.W, self.V, self.H).to(self.device)


    def forward(self, x):

        
        # Time dependent part
        freq = torch.fft.rfft(x, norm = "forward", dim=-1)  # Get real-valued FFT output
        freq_magnitude = freq.abs()  # Magnitude of frequencies
        freq_magnitude = freq_magnitude.reshape(x.shape[0], -1)

        zeta        = self.linear(freq_magnitude)

        # Reshape for multiplication
        Swx = self.Swx.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1, 1, 1)
        zeta = zeta.view(*zeta.shape, 1, 1, 1, 1, 1)
        zeta = zeta.repeat(1, self.W, self.V, self.H, self.W, self.V, self.H)

        # Full covariance matrix
        S_full      = zeta * Swx
        
        return S_full
    
    def __repr__():
        return "Parametric Covariance"
        
    