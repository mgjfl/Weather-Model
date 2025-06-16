
import torch
from abc import abstractmethod, ABCMeta
from typing import Dict
from typing_extensions import override
from numpy import sin, log
from typing import Dict
from torch import Tensor

class ProbabilisticModel(metaclass = ABCMeta):
    
    def __init__(self, in_channels : int, domain_shape : Tensor, parameter_shapes : Dict[str, torch.tensor]):
        """Initializes the parameter sizes

        Args:
            in_channels (int) : number of input channels
            domain_shape (int) : shape of the domain
            parameter_size (dict[str, torch.tensor]): dictionary as param_name : param_shape
        """
        
        # Constants that specify the sample output shape
        self.in_channels    = in_channels
        self.domain_shape   = domain_shape

        # Parameter metrics
        self.parameter_shapes   = parameter_shapes
        self.parameter_sizes    = torch.tensor(list(
            map(lambda x: torch.prod(x), 
                self.parameter_shapes.values())))
        self.parameter_count    = self.parameter_sizes.sum()

    def array2parameters(self, array : torch.tensor) -> Dict[str, torch.tensor]:
        """Creates a map from parameter name to parameter value from a 1D array.

        Args:
            array (torch.tensor): input array

        Returns:
            Parameters: mapping parameter name to value
        """
        
        assert torch.numel(array) == self.parameter_count, f"Array has size {torch.numel(array)}, but should have size {self.parameter_count}."

        cummulative_sizes = torch.cumsum(self.parameter_sizes, dim = 0)

        out = dict()
        for i, (name, shape) in enumerate(self.parameter_shapes.items()):
            if i == 0:
                out[name] = array[:cummulative_sizes[i]].reshape(tuple(shape))
            else:
                out[name] = array[cummulative_sizes[i - 1]:cummulative_sizes[i]].reshape(tuple(shape))

        return out

    def parameters2array(self, param_dict : Dict[str, torch.tensor]) -> torch.tensor:
        """Converts parameters into an 1D array.

        Args:
            param_dict (dict[str, torch.tensor]): mapping from parameter name to value

        Returns:
            torch.tensor: resulting 1D array
        """
        
        return torch.hstack([x.reshape(-1) for x in param_dict.values()])

    @abstractmethod
    def sample(self, params : Dict[str, torch.tensor]) -> torch.tensor:
        """Samples from a random distribution with certain parameters.

        Args:
            params (Parameters) : the parameters
        """
        raise NotImplementedError("Must implement sample method.")
    
    @abstractmethod
    def log_probs(self, params : Dict[str, torch.tensor], x : torch.tensor) -> torch.tensor:
        """Computes the log probability at x.

        Args:
            params (Parameters) : the parameters
            x (torch.tensor) : the input value
        """
        raise NotImplementedError("Must implement pdf method.")
    
    def nll(self, params : Dict[str, torch.tensor], obs : torch.tensor) -> torch.tensor:
        """Computes the Negative Log Likelihood of the distribution given the observation.

        Args:
            params (dict[str, torch.tensor]) : the parameters
            obs (torch.tensor) : the observation
        """
        return - self.log_probs(params, obs)
    
class RandomSampler(ProbabilisticModel):
    
    def generate(self, n : int, seed : int) -> torch.tensor:
        """Generates data for n days with parameters given by parameters_init.

        Args:
            n (int): number of days

        Returns:
            torch.tensor: the output data
        """
        
        # Datastructure for the generated data
        data = torch.zeros(size = (n, self.in_channels, self.domain_shape[0], self.domain_shape[1]))
        
        # Set global seed
        torch.manual_seed(seed)

        # Sample for each day
        for t in range(n):
            parameters = self.parameter_init(t, seed)
            torch.manual_seed(seed + t) # Reset fixed seed from self.paramter_init
            data[t] = self.sample(parameters).reshape(self.in_channels, self.domain_shape[0], self.domain_shape[1])

        return data
    
        
    def parameter_init(self, t : int, seed : int) -> Dict[str, torch.tensor]:
        """
        Generates a random initialization of the parameters that satisfies potential constraints at time t.
        This implementation is time-independent with a fixed seed.
        
        Args:
            t (int) : time

        Returns:
            Parameters: mapping parameter name to value
        """
        
        # All parameters are sampled from U[0,1]
        torch.manual_seed(seed)
        return self.array2parameters(torch.random.uniform(size = (self.parameter_count)))



class GaussianModel(RandomSampler):
    """
    A general class capable of modelling multivariate Gaussian distributions with time-dependent mean and covariance function.
    
    The class can sample new datapoints through:
    1. `parameter_init` which calls `mean_init` and `covariance_init`
    2. `sample`

    The loss is dependent on the log-likelihood of the model, given by `log_probs`
    """
    
    def __init__(self, in_channels : int, domain_shape : Tensor):
        super().__init__(
            in_channels, domain_shape, {
            "mean" : torch.tensor([in_channels * domain_shape[0] * domain_shape[1]]), 
            "lower_tril" : torch.tensor([(in_channels * domain_shape[0] * domain_shape[1] * 
                                          (in_channels * domain_shape[0] * domain_shape[1] + 1)) // 2])
            }
        )
        
        self.covariance_matrix = None
        
    ################################
    ### Parameter initialization ###
    ################################

    @override
    def parameter_init(self, t: int, seed : int) -> Dict[str, torch.tensor]:
        return {
            "mean" : self.mean_init(t, seed),
            "covariance" : self.covariance_init(seed) if self.covariance_matrix is None else self.covariance_matrix
        }
        
    def mean_init(self, t : int, seed : int) -> torch.tensor:

        # Ensure reproducibility
        torch.manual_seed(seed)
        
        return torch.rand(size = (self.parameter_shapes["mean"],))
    
    def covariance_init(self, seed : int) -> torch.Tensor:
        """From https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor"""

        # Reproducibility
        torch.manual_seed(seed)
        
        n = self.w * self.d

        a = 2
        A = torch.stack([torch.randn((n,)) + torch.randn((1,))*a for i in range(n)])
        A = A @ A.T
        D_half = torch.diag(torch.diag(A)**(-0.5))
        C = D_half @ A @ D_half
        
        # Make sure that this is computed in the same form as ML output
        self._minimal_diagonal(C, 1e-3)
        
        return C
    
    ################
    ### Sampling ###
    ################

    def sample(self, params : Dict[str, torch.tensor]) -> torch.Tensor:
        """
        Samples from a multivariate Gaussian distribution with time-dependent parameters.
        """
        
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = params["mean"], 
            covariance_matrix = self._cov_matrix(params))
        
        return dist.sample()
    
    ######################
    ### Log-likelihood ###
    ######################
    
    def log_probs(self, params : Dict[str, torch.tensor], x : torch.Tensor) -> torch.Tensor:
        """
        Computes the log-likelihood of the input `x` based on the given parameters `params`.
        """
        
        mu = params["mean"]
        L = self._array_to_lower_triangular(params["lower_tril"])
        
        if torch.isnan(mu).any() or torch.isnan(L).any():
            raise Exception(f"NaN values")

        y = x.view(-1)
        
        # Step 1: Solve L y = b for y (forward substitution)
        b = (y - mu)
        
        # Log det for Cholesky decomposition of precision matrix: 
        # https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        # https://arxiv.org/pdf/1802.07079
        log_det_cov = - 2 * torch.sum(torch.log(torch.diagonal(L)))
        
        # prec_matrix = self.invert_cholesky(L)
        z = L.T @ b.reshape(-1, 1)

        ## Negative Log Likelihood ##
            
        # Log-determinant
        log_probs = - 1/2 * log_det_cov
        
        # Reconstruction error
        log_probs -= 1/2 * (z.T @ z).item()
        
        ## Regularization terms ##
        
        # Additional penalization independent of the covariance matrix
        log_probs -= torch.linalg.norm(b)
        
        # Regularization for exploding covariance
        log_probs -= torch.linalg.matrix_norm(L)
        
        return log_probs
    
    ########################
    ### Helper functions ###
    ########################

    def _array_to_lower_triangular(self, array : torch.tensor) -> torch.Tensor:
        """
        Transforms a 1D array of size n(n+1)/2 into a n x n lower triangular matrix.

        Args:
            array (torch.Tensor): 1D tensor of size n(n+1)/2 containing the elements of the lower triangular matrix.

        Returns:
            torch.Tensor: n x n lower triangular matrix.
        """
        # Compute the size of the resulting lower triangular matrix
        n = int((-1 + (1 + 8 * array.size(0))**0.5) / 2)
        if n * (n + 1) // 2 != array.size(0):
            raise ValueError("Input size is not valid for a lower triangular matrix.")

        # Create an empty n x n matrix
        lower_triangular = torch.zeros(n, n, dtype=array.dtype, device=array.device)

        # Fill the lower triangular part
        indices = torch.tril_indices(n, n)
        lower_triangular[indices[0], indices[1]] = array
        
        # Make sure the diagonal entries are positive we predict log(l_ii)
        # https://arxiv.org/pdf/1802.07079
        lower_triangular[torch.arange(n), torch.arange(n)] = torch.exp(torch.diag(lower_triangular))
        
        # Make sure L is invertible by ensuring a minimum value
        self._minimal_diagonal(lower_triangular)

        return lower_triangular

    def _minimal_diagonal(self, L: torch.tensor, min_val: float = 1e-6) -> None:
        """
        Modifies input tensor `L` in place to ensure that its diagonal is positive and is at least `min_val`.
        """
        # Extract the diagonal (view) and modify in-place
        L_diag = torch.diagonal(L, 0)
        L_diag.abs_()  # Ensure diagonal elements are positive
        L_diag.clamp_(min=min_val)  # Enforce minimum value

        # No return needed since L is modified in-place
        pass
    
    def _invert_cholesky(self, L : torch.tensor) -> torch.tensor:
        """
        Computes a matrix from its Cholesky decomposition `L`.
        """
        return L @ L.T

    def _cov_matrix(self, params : Dict[str, torch.tensor]) -> torch.tensor:
        """Returns the covariance matrix

        Args:
            params (Parameters): parameters
        """
        if "covariance" in params.keys():
            return params["covariance"]
        else:
            L = self._array_to_lower_triangular(params["lower_tril"])
            prec_matrix = self._invert_cholesky(L)
            return torch.linalg.pinv(prec_matrix)

# class PeriodicGaussianModel(GaussianModel):
#     """
#     Multivariate Gaussian with periodic time-dependent mean.
#     """
    
#     def __init__(self, d : int, w : int):
#         super().__init__(d, w)
        
#     @override
#     def mean_init(self, t : int, seed : int) -> torch.tensor:

#         # Ensure reproducibility
#         torch.manual_seed(seed)

#         # Constructing the mean vector
#         size = self.w * self.d
#         mu_0 = torch.rand(size = (size,))
#         delta_mu = 0.1 * torch.rand(size = (size,))

#         mu = mu_0 + delta_mu * sin(t / 365 * 2 * torch.pi)
        
#         return mu
    
# class LinearGaussian(GaussianData):
#     """
#     Multivariate Gaussian with linear time-dependent mean.
#     """
    
#     def __init__(self, d : int, w : int):
#         super().__init__(d, w)
        
#     @override
#     def mean_init(self, t : int, seed : int) -> torch.tensor:

#         # No time dependence in the base value
#         torch.manual_seed(seed)

#         # Constructing the mean vector
#         size = self.w * self.d
#         mu_0 = torch.rand(size = (size,))
#         # Random on [-0.05, 0.05]
#         delta_mu = 0.5 * (torch.rand(size = (size,)) - 0.5)

#         mu = mu_0 + delta_mu * t
        
#         return mu