import numpy as np
from torch import nn
import torch
from typing import Callable
import sys
import os
sys.path.append(os.path.abspath(".."))
from Data.data_models import ProbabilisticModel

class ACW(nn.Module):
    """
    Adjustable Channel Wrapper: a wrapper for neural networks that provides
    for explicit specification of the in- and out-channels.
    """
    
    def __init__(self, 
                 model_class : nn.Module, 
                 in_channels : int, 
                 out_channels : int, 
                 grid_shape : torch.tensor = None, 
                 out_size : int = None, 
                 **model_kwargs):
        super(ACW, self).__init__()
        self.model_class = model_class
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_kwargs = model_kwargs
        self.grid_shape = grid_shape
        self.model = self._create_model()
        self.project = None
        if grid_shape is not None:
            self.in_size = torch.prod(grid_shape) * out_channels
            self.project = nn.Linear(self.in_size, out_size)

    def _create_model(self):
        return self.model_class(
            in_channels = self.in_channels, 
            out_channels = self.out_channels, 
            grid_shape = self.grid_shape,
            **self.model_kwargs)
        
    def use_linear_projection(self, in_shape : torch.tensor, out_size : int):
        
        def f(x : torch.tensor) -> torch.tensor:
            in_size = torch.prod(in_shape)
            
            linear = nn.Linear(in_size, out_size)
            return linear(x)
        
        self._set_project(f)
        
        
    def _set_project(self, f : Callable[[torch.tensor], torch.tensor]):
        self.project = f

    def forward(self, x):
        x = self.model(x)
        # Linear projection layer
        if self.project is not None:
            x = x.reshape((-1, self.in_size))
            x = self.project(x)
            
        return x

class PNN(ACW):
    """
    A probabilistic Neural Network that is a Neural Network (ACW) with output specified by the given probabilistic model. 
    """
    
    def __init__(self, 
                 probabilistic_model : ProbabilisticModel, 
                 model_class : nn.Module, 
                 in_channels : int, 
                 out_channels : int, 
                 grid_shape : torch.tensor = None, 
                 **model_kwargs):
        super().__init__(
            model_class=model_class,
            in_channels=in_channels,
            out_channels=out_channels,
            grid_shape=grid_shape,
            out_size=probabilistic_model.parameter_count,
            **model_kwargs
        )

        self.probabilistic_model = probabilistic_model
        
    def sample(self, x, n):

        # Datastructure for the generated data
        data = np.zeros(shape = (n, self.probabilistic_model.d, self.probabilistic_model.w))

        with torch.eval():
            parameters_array_prob_model = self.forward(x)
            parameters = self.probabilistic_model.array2paramters(parameters_array_prob_model)
            for t in range(n):
                np.random.seed(t)
                data[t] = self.probabilistic_model.sample(parameters).reshape(
                    self.probabilistic_model.d, self.probabilistic_model.w)
                
        return data
    
    def get_prob_model(self):
        return self.probabilistic_model
    
class BNN(ACW):
    """
    A Bayesian Neural Network that is a Neural Network (ACW) with output specified by the given probabilistic model. 
    """
    
    def __init__(self, probabilistic_model : ProbabilisticModel, in_channels : int, model_class : nn.Module, **model_kwargs):
        super().__init__(
            model_class,
            in_channels,
            probabilistic_model.d * probabilistic_model.w,
            **model_kwargs
        )

        self.probabilistic_model = probabilistic_model
        self._transform_to_bayesian()
        
    def _transform_to_bayesian(self):
        """Transforms the Neural Network into a Bayesian neural network."""
        raise NotImplementedError("Not yet implemented.")
        
    def sample(self, x, n):

        # Datastructure for the generated data
        data = np.zeros(shape = (n, self.probabilistic_model.d, self.probabilistic_model.w))

        with torch.eval():
            for t in range(n):
                np.random.seed(t)
                data[t] = self.forward(x).reshape(
                    self.probabilistic_model.d, self.probabilistic_model.w)
                
        return data