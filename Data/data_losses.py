from torch import nn
import torch
from .data_models import ProbabilisticModel
    
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
            
            total_loss += loss

        mean_loss = total_loss / batch_size

        return mean_loss