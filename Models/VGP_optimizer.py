import torch
import gpytorch

class VGPOptimizer:
    """
    Custom optimizer with adjustable hyperparameter optimization and efficient NGD optimization for VGP.
    """
    def __init__(self, 
                 model, 
                 hyperparameter_optimizer, 
                 ngd_lr=0.1, 
                 hyper_lr=0.01):
        self.ngd = gpytorch.optim.NGD(
            model.gp_model.variational_parameters(),
            num_data=model.num_data,
            lr=ngd_lr
        )
        self.hyper = hyperparameter_optimizer([
            {'params': model.gp_model.hyperparameters()},
            {'params': model.mean_model.parameters()},
            {'params': model.likelihood.parameters()}
        ], lr=hyper_lr)

    def step(self):
        self.ngd.step()
        self.hyper.step()

    def zero_grad(self):
        self.ngd.zero_grad()
        self.hyper.zero_grad()

    def state_dict(self):
        return {
            'ngd': self.ngd.state_dict(),
            'hyper': self.hyper.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.ngd.load_state_dict(state_dict['ngd'])
        self.hyper.load_state_dict(state_dict['hyper'])