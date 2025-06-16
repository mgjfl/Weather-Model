from torch import nn
import torch
import gpytorch
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from  gpytorch.mlls import *
from torch.nn.functional import mse_loss
import torch.nn.functional as F

class GradientSmoothnessLoss(nn.Module):
    def __init__(self, model, grad_regularization=0.001):
        super(GradientSmoothnessLoss, self).__init__()
        self.model = model
        self.regularization_weight = grad_regularization

    def forward(self, x):
        xm = x.mean
        xm = xm.reshape(xm.shape[0], self.model.in_channels, *self.model.domain_shape[:-1])

        # Gradients along the spatial dimensions
        lap_v = torch.gradient(torch.gradient(xm, dim=2)[0], dim=2)[0] # Gradient w.r.t. V (vertical)
        lap_h = torch.gradient(torch.gradient(xm, dim=3)[0], dim=3)[0]  # Gradient w.r.t. H (horizontal)

        # Compute L2 norm of gradients along each direction
        grad_norm = sum(torch.norm(lap, p=2) ** 2 for lap in [lap_v, lap_h])

        # Apply regularization weight
        gradient_penalty = self.regularization_weight * grad_norm

        return gradient_penalty

class NMLLLoss(nn.Module):
    """
    Marginal Negative Log Likelihood for Gaussian Processes.
    """
    def __init__(self,
                 model,
                 regularization_weight,
                 **model_kwargs):
        super().__init__()
        self.model = model
        self.mll = ExactMarginalLogLikelihood(
            likelihood  = model.likelihood, 
            model       = model)
        
        self.regularization_weight = regularization_weight
        
    def forward(self, input, target, isTraining = True):
        target = target.reshape(target.shape[0], -1)
        
        loss = -self.mll(input, target)

        if isTraining:
            l2_mean_model = sum(torch.norm(param, p=2) ** 2 for param in self.model.mean_model.parameters())      
            reg = l2_mean_model * self.regularization_weight

            loss += reg #+ grad_penalty

        return loss
    
class VariationalELBOLoss(nn.Module):
    def __init__(self,
                 model,
                 regularization_weight,
                 grad_regularization):
        super().__init__()
        self.model = model
        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood  = model.likelihood,
            model       = model.gp_model,
            num_data    = model.num_data
        )

        self.regularization_weight = regularization_weight
        # self.gradient_smoothness_loss = GradientSmoothnessLoss(model, grad_regularization)
        
    def forward(self, input, target, isTraining = True):

        B = target.shape[0]

        target_reshaped = target.reshape(B, -1)

        # print(f"{torch.mean(input.mean)=}")
        # print(f"{torch.mean(target)=}")

        # Compute negative log likelihood (data fit term)
        # First map latent variables to distribution over output. Then compute the log probabilities of the target
        nll = -self.model.likelihood(input).log_prob(target_reshaped).mean()  / B

        # Get KL divergence from the GP component
        kl_div = self.model.gp_model.variational_strategy.kl_divergence().mean() / self.model.num_data

        # Combine with proper scaling
        loss = nll + kl_div


        # L2 regularization on FNO parameters
        if isTraining:
            l2_mean_model = sum(torch.norm(param, p=2) ** 2 for param in self.model.mean_model.parameters())      
            reg = l2_mean_model * self.regularization_weight

            # Compute gradient smoothness loss
            # grad_penalty = self.gradient_smoothness_loss(input)
            loss += reg #+ grad_penalty

            huber_loss = F.huber_loss(input.mean, target_reshaped)
            # TODO: make this a parameter
            c_huber = 3
            loss += c_huber * huber_loss
        
        return loss

    
class Mean_MSELoss(nn.Module):
    def __init__(self, **model_kwargs):
        super(Mean_MSELoss, self).__init__()

        self.input_type = model_kwargs["input_type"]
        
    def forward(self, input, target):

        mean, sigma = input

        if self.input_type == "parameters":
            y = target[0]
        elif self.input_type == "data":
            y = target

        # batch_size = mean.shape[0]

        return mean.sub(y).pow(2).mean()
    
class Full_MSELoss(nn.Module):
    def __init__(self, **model_kwargs):
        super(Full_MSELoss, self).__init__()

        
    def forward(self, input, target):

        mean, sigma = input
        mean_target, sigma_target = target

        loss = mean.sub(mean_target).pow(2).mean() + sigma.sub(sigma_target).pow(2).mean()

        return loss