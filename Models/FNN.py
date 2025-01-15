from .neural_models import *
import torch.nn as nn
import torch.nn.functional as F

# Documentation for NOP: https://neuraloperator.github.io/dev/modules/api.html#module-neuralop.models

class FNN(nn.Module):
    """
    Simple feedforward network designed as component in ACW.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_shape,
                 hidden_channels = 256,
                 n_layers = 4):
        
        super().__init__()
        
        # Save parameters
        self.grid_shape         = grid_shape
        self.in_channels        = in_channels
        self.hidden_channels    = hidden_channels
        self.out_channels       = out_channels
        self.n_layers           = n_layers
        
        # Define the non-linearity
        self.non_linearity = F.relu
        
        # Lifting layer
        self.lifting = nn.Linear(in_channels * torch.prod(grid_shape), hidden_channels)
        
        # Middle layers
        self.middle_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.middle_layers.append(nn.Linear(hidden_channels, hidden_channels))
            
        self.dropout_layers = [nn.Dropout(p=0.5) for _ in range(self.n_layers)]
        
        # Projection layer
        self.project = nn.Linear(hidden_channels, out_channels * torch.prod(grid_shape))
        
    def forward(self, x):
        
        if x.ndim == 3: # shape = w, days, d
            x = x.unsqueeze(0)
            
        # shape = batch, w * days * d
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

        return x
        
        