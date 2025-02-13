
# Code is strongly based on code from this paper: 
# Kumar, S., Nayek, R., & Chakraborty, S. (2024). Neural Operator induced Gaussian Process framework for probabilistic solution of parametric partial differential equations. arXiv preprint arXiv:2404.15618.

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.means import Mean, MultitaskMean
from pytorch_wavelets import DWT, IDWT
from neural_models import *
import os
import torch
from tqdm.notebook import tqdm
import math
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt

class CustomMean(gpytorch.means.Mean):
    """
    Wrapper for the GPyTorch Mean class with custom mean model.
    """
    def __init__(self, 
                 model_class : nn.Module, 
                 in_channels : int, 
                 out_channels : int, 
                 grid_shape : torch.tensor,
                 out_size : int,
                 **model_kwargs):
        super().__init__()
        self.mean_model = ACW(
            model_class=model_class,
            in_channels=in_channels,
            out_channels=out_channels,
            grid_shape=grid_shape,
            out_size=out_size,
            **model_kwargs)

    def forward(self, x):
        # print(f"Mean module inputs: {x.shape}")
        x = x.reshape(1, x.shape[0], self.w, self.d)
        x = x.permute(0, 2, 1, 3)
        mean_prediction = self.mean_model(x)
        
        return mean_prediction


class CustomMultitaskMean(MultitaskMean):
    def __init__(self, custom_mean, num_tasks):
        super().__init__(base_means=[gpytorch.means.ConstantMean()], num_tasks=num_tasks)
        self.custom_mean = custom_mean
        
    def forward(self, input):
        mean_prediction = self.custom_mean(input)
        return mean_prediction

class GP(gpytorch.models.ExactGP):
    def __init__(self, 
                 train_x,
                 train_y,
                 model_class : nn.Module, 
                 w : int,
                 t : int,
                 d : int,
                 **model_kwargs):        
        super().__init__(
            train_x,
            train_y, 
            gpytorch.likelihoods.MultitaskGaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
                num_tasks=train_y.shape[-1])
            )

        
        num_dims = train_y.shape[-1]
        in_channels  = w
        grid_shape  = torch.tensor([t, d])

        # self.mean_module = CustomMultitaskMean(
        #     CustomMean(
        #          model_class=model_class, 
        #          in_channels=in_channels, 
        #          grid_shape=grid_shape,
        #          out_size=num_dims,
        #          w=w,
        #          d=d,
        #          **model_kwargs), num_tasks=num_dims
        # )
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(train_x.shape[-1]), num_tasks=num_dims
        )


        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()), 
            num_tasks=num_dims, rank=1
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        x = x.reshape(x.shape[0], -1)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std.cpu()) + mean.cpu()
        return x