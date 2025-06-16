from .neural_models import *
import torch.nn as nn
import torch.nn.functional as F

# Documentation for NOP: https://neuraloperator.github.io/dev/modules/api.html#module-neuralop.models

class FNN(nn.Module):
    """
    Simple feedforward network designed as component in ACW.
    """
    def __init__(self,
                 in_channels : int,
                 out_channels : torch.tensor,
                 domain_shape : torch.tensor,
                 hidden_channels = 256,
                 n_layers = 4,
                 **model_kwargs):
        
        super().__init__()
        
        # Save parameters
        self.domain_shape       = domain_shape
        self.in_channels        = in_channels
        self.hidden_channels    = hidden_channels
        self.out_channels       = out_channels
        self.n_layers           = n_layers
        self.model_kwargs       = model_kwargs

        # Shape parameters
        self.W = self.in_channels
        self.V, self.H, self.training_window = domain_shape
        
        # Define the non-linearity
        self.non_linearity = F.relu
        
        # Lifting layer
        self.lifting = nn.Linear(in_channels * torch.prod(domain_shape), hidden_channels)
        
        # Middle layers
        self.middle_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.middle_layers.append(nn.Linear(hidden_channels, hidden_channels))
            
        self.dropout_layers = [nn.Dropout(p=0.5) for _ in range(self.n_layers)]
        
        # Projection layer
        self.project = nn.Linear(hidden_channels, out_channels * torch.prod(domain_shape))
        
    def forward(self, x):
        
        if x.ndim == 4: # shape = W, V, H, training_window
            x = x.unsqueeze(0)
            
        # shape = batch, W * V * H * training_window
        x = x.reshape(x.shape[0], -1)
        
        # Lifting
        x = self.lifting(x)
        
        # Middle layers
        for layer, dropout in zip(self.middle_layers, self.dropout_layers):
            x = layer(x)
            x = self.non_linearity(x)
            x = dropout(x)
        
        # Projection
        x = self.project(x)

        # Reshape in correct format
        x = x.reshape(x.shape[0], self.W, self.V, self.H, self.training_window)

        return x
        
        