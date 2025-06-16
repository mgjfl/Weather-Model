from .neural_models import *
import torch.nn as nn
import torch.nn.functional as F


class MGST(nn.Module):
    """
    Mean for the Gaussian Spatio-Temporal (GST) model. Estimates the mean and covariance based on past observations by assuming no temporal dependence.
    Model is non-parametric and does not change with training.
    """
    def __init__(self,
                 in_channels : int,
                 out_channels : torch.tensor,
                 domain_shape : torch.tensor,
                 **model_kwargs):
        
        super().__init__()
        
        # Save parameters
        self.domain_shape       = domain_shape
        self.in_channels        = in_channels
        self.out_channels       = out_channels

        # Shape parameters
        self.W = self.in_channels
        self.V, self.H, self.training_window = domain_shape
        
    def forward(self, x):
        
        if x.ndim == 4: # shape = W, V, H, training_window
            x = x.unsqueeze(0)
            
        # shape = batch, W, V, H, training_window
        x = x.mean(dim = -1) # Average over the training window
        
        return x
        
class CGST(nn.Module):
    """
    Covariance for the Gaussian Spatio-Temporal (GST) model. Estimates the mean and covariance based on past observations by assuming no temporal dependence.
    Model is non-parametric and does not change with training.
    """
    def __init__(self,
                 in_channels   : int, 
                 domain_shape  : torch.tensor):
        
        super().__init__()
        
        # Save parameters
        self.domain_shape       = domain_shape
        self.in_channels        = in_channels

        # Shape parameters
        self.W = self.in_channels
        self.V, self.H, self.training_window = domain_shape
        self.training_days = domain_shape[2]

        # Extract shape
        self.mean_shape = torch.tensor([in_channels, domain_shape[0], domain_shape[1]])
        self.cov_shape  = torch.tensor([*self.mean_shape, *self.mean_shape])
        
    def forward(self, x):
        out = torch.zeros(x.shape[0], *self.cov_shape, device = x.device)

        # Shape (batch, W, V, H, training_days)
        
        x = x.reshape(x.shape[0], -1, self.training_days)

        for i, xx in enumerate(x):
            out[i, ...] = torch.cov(xx).reshape(*self.cov_shape)

        return out
    