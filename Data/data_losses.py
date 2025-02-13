from torch import nn
import torch
from .data_models import ProbabilisticModel
import gpytorch
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL

class NMLLLoss(nn.Module):
    """
    Marginal Negative Log Likelihood for Gaussian Processes.
    """
    def __init__(self,
                 model,
                 num_data):
        super().__init__()
        self.model = model
        self.mll = DeepApproximateMLL(VariationalELBO(
            model.likelihood, 
            model, 
            num_data))
        # self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        #     likelihood=likelihood,
        #     model=model
        # )
        
    def forward(self, input, target):
        # print(f"loss {input.shape=}")
        target = target.reshape(target.shape[0], -1)
        # print(f"loss {target.shape=}")

        return -self.mll(input, target)
    
class NLLLoss(nn.Module):
    def __init__(self, prob_model : ProbabilisticModel):
        super(NLLLoss, self).__init__()
        self.prob_model = prob_model
        
    def forward(self, input, target):

        total_loss = 0
        batch_size = input.shape[0]
        
        # Put target into correct shape
        target = torch.flatten(target, 1)
        
        for i, t in zip(input, target):
            
            # Convert output layer to parameters
            parameters = self.prob_model.array2parameters(i)

            # Compute the loss
            loss = self.prob_model.nll(parameters, t)

            # Ignore NaN losses
            if torch.isnan(loss):
                continue

            # if torch.isnan(loss):
            #     raise Exception(f"NaN loss with:\nInput: {i}\nTarget: {t}")

            # if torch.isinf(loss):
            #     raise Exception(f"Inf loss with:\nInput: {i}\nTarget: {t}")
            
            total_loss += loss

        mean_loss = total_loss / batch_size

        return mean_loss