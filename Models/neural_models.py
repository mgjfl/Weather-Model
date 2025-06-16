import numpy as np
from torch import nn
import torch
from typing import Callable
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..")
                )
from Data.data_models import ProbabilisticModel

class ACW(nn.Module):
    """
    Adjustable Channel Wrapper: a wrapper for neural networks that provides
    for explicit specification of the in- and out-channels.
    """
    
    def __init__(self, 
                 mean_model_class       : nn.Module, 
                 covariance_model_class : nn.Module, 
                 in_channels            : int, 
                 domain_shape           : torch.tensor, 
                 **model_kwargs):
        super(ACW, self).__init__()

        # Separate branches for mean and covariance computation
        self.mean_model_class       = mean_model_class
        self.covariance_model_class = covariance_model_class

        # Specify the input
        self.in_channels            = in_channels
        self.domain_shape           = domain_shape

        # Specify the output
        self.n_mean_parameters      = in_channels * torch.prod(domain_shape[:-1])
        self.n_cov_parameters       = self.n_mean_parameters ** 2

        # Model specific arguments
        self.model_kwargs           = model_kwargs

        # Initialize the models
        self.mean_model = self._create_mean_model()
        self.cov_model  = self._create_cov_model()

    def _create_mean_model(self):
        return self.mean_model_class(
            in_channels     = self.in_channels,
            domain_shape    = self.domain_shape, 
            out_channels    = self.in_channels, 
            **self.model_kwargs)
    
    def _create_cov_model(self):
        return self.covariance_model_class(
            in_channels         = self.in_channels, 
            domain_shape        = self.domain_shape)

    def forward(self, x):

        # Compute the mean_value
        xm = self.mean_model(x)

        if len(xm.shape) == 5:
            # Shape of mean model output: (batch, W, V, H, training_window)
            xm = xm.mean(dim = -1) # Shape (batch, W, V, H)

        # Compute covariance
        xc = self.cov_model(x) # Shape (batch, W, V, H, W, V, H)
            
        return (xm, xc)

class PNN(ACW):
    """
    A probabilistic Neural Network that is a Neural Network (ACW) with output specified by the given probabilistic model. 
    """
    
    def __init__(self, 
                 probabilistic_model : ProbabilisticModel, 
                 mean_model_class       : nn.Module, 
                 covariance_model_class : nn.Module, 
                 in_channels            : int, 
                 domain_shape           : torch.tensor, 
                 **model_kwargs):
        super().__init__(
            mean_model_class        = mean_model_class,
            covariance_model_class  = covariance_model_class,
            in_channels             = in_channels,
            domain_shape            = domain_shape,
            **model_kwargs
        )

        self.probabilistic_model = probabilistic_model
        
    # def sample(self, x, n):

    #     # Datastructure for the generated data
    #     data = np.zeros(shape = (n, self.probabilistic_model.d, self.probabilistic_model.w))

    #     with torch.eval():
    #         mu, sigma = self.forward(x)
    #         for t in range(n):
    #             np.random.seed(t)
    #             data[t] = self.probabilistic_model.sample(mu, sigma)
                
    #     return data
    
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