
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
from gpytorch.kernels import MaternKernel, ScaleKernel, GridInterpolationKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
import math
from gpytorch.models import ApproximateGP, ExactGP, GP
from Models.GST import *
from Models.AFNO import *

class CustomMean(gpytorch.means.Mean):
    """
    Wrapper for the GPyTorch Mean class with custom mean model.
    """
    def __init__(self, 
                 mean_model_class   : nn.Module, 
                 in_channels        : int, 
                 domain_shape       : torch.tensor,
                 **model_kwargs):
        super().__init__()

        # Separate branches for mean and covariance computation
        self.mean_model_class       = mean_model_class

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
    
    def _create_mean_model(self):
        return self.mean_model_class(
            in_channels     = self.in_channels,
            domain_shape    = self.domain_shape, 
            out_channels    = self.in_channels, 
            **self.model_kwargs)

    def forward(self, x):
        mean_prediction = self.mean_model(x)
        
        return mean_prediction

class CustomMultitaskMean(MultitaskMean):
    def __init__(self, custom_mean, num_tasks):
        super().__init__(base_means=[gpytorch.means.ConstantMean()], num_tasks=num_tasks)
        self.custom_mean = custom_mean
        
    def forward(self, input):
        mean_prediction = self.custom_mean(input)
        return mean_prediction

class EGP(gpytorch.models.ExactGP):
    def __init__(self, 
                 train_x,
                 train_y,
                 mean_model_class   : nn.Module, 
                 in_channels        : int,
                 domain_shape       : torch.tensor,
                 **model_kwargs): 
        """
        Implementation of the Exact Gaussian Process architecture.
        """       
        super().__init__(
            train_x,
            train_y, 
            gpytorch.likelihoods.MultitaskGaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
                num_tasks=math.prod(train_y.shape[1:]))
            )

        input_dims = math.prod(train_x.shape[1:])
        output_dims = math.prod(train_y.shape[1:])

        training_days = domain_shape[-1]
        self.in_channels = in_channels
        self.domain_shape = domain_shape
        self.mean_model_class = mean_model_class
        
        self.model_kwargs = model_kwargs

        self.mean_model = CustomMean(
                 mean_model_class   = mean_model_class, 
                 in_channels        = in_channels, 
                 domain_shape       = domain_shape,
                img_size        = (self.domain_shape[0], self.domain_shape[1]),
                in_chans        = self.in_channels * self.domain_shape[2],
                out_chans       = self.in_channels * self.domain_shape[2],
                 **model_kwargs)

        # Linear module that assigns a certain weight to each day for the past training days
        self.day_weighting = nn.Linear(training_days, 1)
        
        # const = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.ConstantKernel(),
        #     num_tasks=output_dims,
        #     rank=1
        # )
        # matern = gpytorch.kernels.MultitaskKernel(
        #         gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=output_dims, has_lengthscale=True), 
        #     num_tasks=output_dims, 
        #     rank=1
        # )
        # sum_kernel = matern + const
        # self.covar_module = gpytorch.kernels.ScaleKernel(sum_kernel)


        self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=output_dims, has_lengthscale=True), 
            num_tasks=output_dims, 
            rank=1
        )
        
        
        # For AFNO
        if issubclass(self.mean_model_class, AFNONet):
            V, H, T = self.domain_shape
            self.linear_projection = nn.Linear(
                                            (V // self.model_kwargs["patch_size"][0]) * self.model_kwargs["patch_size"][0] *
                                            (H // self.model_kwargs["patch_size"][1]) * self.model_kwargs["patch_size"][1] *
                                          in_channels * T,
                                          V * H * in_channels) 
        
    def forward(self, x):
        # Compute mean

        # Return to proper domain format
        x = x.reshape(x.shape[0], self.in_channels, *self.domain_shape)

        if issubclass(self.mean_model_class, AFNONet):
            xm = x.permute(0, 4, 1, 2, 3)
            B, T, W, V, H = xm.shape
            xm = xm.reshape(B, T * W, V, H)

        # Compute mean
        x_model = self.mean_model(xm)
        
        if issubclass(self.mean_model_class, AFNONet):
            # [B, T * W, V, H]
            x_model = x_model.view(B, -1)
            x_model = self.linear_projection(x_model) # [B, M * C]
            x_model = x_model.reshape(B, W, V, H) 


        if len(x_model.shape) == 5:
            # Shape of mean model output: (batch, W, V, H, training_window)
            x_model = x_model.mean(dim = -1) # Shape (batch, W, V, H)

            x_model = x_model.reshape(B, W, V, H) 
            
        mean_x = x_model.reshape(x_model.shape[0], -1)

        # Compute covariance
        x_weighted = self.day_weighting(x)
        x_weighted = x_weighted.reshape(x.shape[0], -1)

        covar_x = self.covar_module(x_weighted)
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
    
class GPWrapper(nn.Module):
    """
    A wrapper for a GP model that properly shapes input and output.
    """
    def __init__(self, 
                 mean_model_class       : nn.Module, 
                 gp_model_class         : nn.Module,
                 in_channels            : int, 
                 domain_shape           : torch.tensor, 
                 num_data               : int,
                 dropout_rate           : float,
                 dataset                ,
                 **model_kwargs):
        super(GPWrapper, self).__init__()

        # The Gaussian Process model
        self.gp_model_class = gp_model_class
        self.mean_model_class = mean_model_class

        # Specify the input
        self.in_channels            = in_channels
        self.domain_shape           = domain_shape
        self.dataset                = dataset

        # Compute the GP parameters
        self.input_dim = in_channels * torch.prod(domain_shape)
        self.output_dim = in_channels * torch.prod(domain_shape[:-1])

        # Model specific arguments
        self.model_kwargs           = model_kwargs

        # Create the GP model
        self.gp_model = self._create_gp_model()
        self.mean_model = self._create_mean_model()

        # First apply a prior computation via 

        # The likelihood and num_data
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim)
        # self.likelihood.register_constraint("raw_task_noises", gpytorch.constraints.GreaterThan(1e-3))

        self.num_data = num_data
        self.dropout = nn.Dropout(p=dropout_rate)

        # Store for loss computation
        self.latest_xm = None
        self.latest_gp_out = None
        
        # Register normalization buffers
        self.initialize_normalization_stats()
        
        # For AFNO
        if issubclass(self.mean_model_class, AFNONet):
            V, H, T = self.domain_shape
            self.linear_projection = nn.Linear(
                                            (V // self.model_kwargs["patch_size"][0]) * self.model_kwargs["patch_size"][0] *
                                            (H // self.model_kwargs["patch_size"][1]) * self.model_kwargs["patch_size"][1] *
                                          in_channels * T,
                                          V * H * in_channels) 

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
        
        # Normalize input weather variables
        x_norm = (x - self.input_mean) / self.input_std
        
        if issubclass(self.mean_model_class, AFNONet):
            x_norm = x_norm.permute(0, 4, 1, 2, 3)
            B, T, W, V, H = x_norm.shape
            x_norm = x_norm.reshape(B, T * W, V, H)

        # Compute mean
        x_model = self.mean_model(x_norm) 
        x_model = self.dropout(x_model)
        
        if issubclass(self.mean_model_class, AFNONet):
            # [B, T * W, V, H]
            x_model = x_model.view(B, -1)
            x_model = self.linear_projection(x_model) # [B, M * C]
            x_model = x_model.reshape(B, W, V, H) 


        if len(x_model.shape) == 5:
            # Shape of mean model output: (batch, W, V, H, training_window)
            x_model = x_model.mean(dim = -1) # Shape (batch, W, V, H)
            
        xm = x_model * self.output_std
        
        if not issubclass(self.mean_model_class, MGST):
            # X has size [B, W, V, H, T]
            x_base = x.mean(dim = -1)
            xm = xm + x_base

        self.latest_xm = xm

        # Reshape to [n, d] for GP input
        x_flat = x_norm.reshape(x.shape[0], -1)
        
        # Get GP distribution (zero mean)
        gp_out = self.gp_model(x_flat)  # MultivariateNormal with zero mean

        self.latest_gp_out = gp_out
        

        # Add deterministic mean from mean_model
        final_mean = xm.reshape(xm.shape[0], -1)
        # print(f"{torch.mean(final_mean)}=")
        final_mvn = gpytorch.distributions.MultitaskMultivariateNormal(
            final_mean, gp_out.lazy_covariance_matrix
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
                # y : [B, W, V, H]

                W = x.shape[1]
                C = y.shape[-1]

                x_ = x.permute(1, 0, 2, 3, 4).reshape(W, -1)  # [W, B*V*H*T]
                input_sum += x_.sum(dim=1)                   # [W]
                input_sumsq += (x_ ** 2).sum(dim=1)          # [W]
                input_count += x_.shape[1]

                y_ = y.permute(1, 0, 2, 3).reshape(W, -1)    # [W, B*V*H]
                output_sum += y_.sum(dim=1)                  # [W]
                output_sumsq += (y_ ** 2).sum(dim=1)         # [W]
                output_count += y_.shape[1]

            # Compute statistics
            input_mean = input_sum / input_count                        # [W]
            input_var = input_sumsq / input_count - input_mean ** 2     # [W]
            input_std = input_var.clamp_min(1e-6).sqrt()                # [W]

            output_mean = output_sum / output_count                     # [W]
            output_var = output_sumsq / output_count - output_mean ** 2 # [W]
            output_std = output_var.clamp_min(1e-6).sqrt()              # [W]

            # Reshape and assign to model           
            self.register_buffer("input_mean", input_mean.view(-1, 1, 1, 1))
            self.register_buffer("input_std", input_std.view(-1, 1, 1, 1))
            self.register_buffer("output_mean", output_mean.view(-1, 1, 1))
            self.register_buffer("output_std", output_std.view(-1, 1, 1))

    

class VariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, 
                 num_latents       : int, 
                 input_dim         : int,
                 output_dim        : int,
                 m_inducing_points : int, 
                 **model_kwargs):
        """
        Implementation of the Variational Gaussian Process architecture.
        """

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, m_inducing_points, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=output_dim,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        
        rbf = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]))
        const = gpytorch.kernels.ConstantKernel(batch_shape=torch.Size([num_latents]))
        
        # Constraints
        rbf.initialize(raw_lengthscale=torch.tensor(1.0).expand(num_latents, 1, 1))
        rbf.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(1e-2, 10.0))
        const.initialize(raw_constant=torch.tensor(1e-2).expand(num_latents))
        const.register_constraint("raw_constant", gpytorch.constraints.Interval(1e-3, 10.0))


        sum_kernel = rbf + const

        self.covar_module = gpytorch.kernels.ScaleKernel(
            sum_kernel,
            batch_shape=torch.Size([num_latents])
        )
        
        
        # self.covar_module.initialize(outputscale=1.0)
        # self.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.Interval(1e-3, 10.0))

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)