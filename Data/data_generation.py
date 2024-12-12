from .data_models import *
from typing_extensions import override
from numpy import sin

class GaussianData(RandomSampler):
    
    def __init__(self, d : int, w : int):
        super().__init__(
            d, w, {
            "mean" : torch.tensor([w * d]), 
            "lower_tril" : torch.tensor([(w * d * (w * d + 1)) // 2])
            }
        )
        
        self.covariance_matrix = self.covariance_init()
    
    def covariance_init(self) -> torch.tensor:
        """From https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor"""

        # Reproducibility
        torch.manual_seed(1)
        
        n = self.w * self.d

        a = 2
        A = torch.stack([torch.randn((n,)) + torch.randn((1,))*a for i in range(n)])
        A = A @ A.T
        D_half = torch.diag(torch.diag(A)**(-0.5))
        C = D_half @ A @ D_half
        
        return C
    
    def array_to_lower_triangular(self, array : torch.tensor):
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
        
        # Make sure the diagonal entries are positive
        idxs = torch.arange(0, n)
        lower_triangular[idxs, idxs] = lower_triangular[idxs, idxs]**2

        return lower_triangular
    
    def array_to_spd(self, array : torch.tensor) -> torch.tensor:
        """
        Transforms array into a lower triangular matrix with positive diagonal and
        subsequently constructs C as a Cholesky decomposition of array.
        """
        L = self.array_to_lower_triangular(array)
        return L @ L.T

    def mean_init(self, t : int) -> torch.tensor:

        # Ensure reproducibility
        torch.manual_seed(1)
        
        return torch.rand(size = (self.parameter_shapes["mean"],))
        
    @override
    def parameter_init(self, t: int) -> dict[str, torch.tensor]:
        return {
            "mean" : self.mean_init(t),
            "covariance" : self.covariance_matrix
        }
        
    def cov_matrix(self, params : dict[str, torch.tensor]) -> torch.tensor:
        """Returns the covariance matrix

        Args:
            params (dict[str, torch.tensor]): parameters
        """
        if "covariance" in params.keys():
            return params["covariance"]
        else:
            return self.array_to_spd(params["lower_tril"])
    
    def sample(self, params):
        """
        Samples from a multivariate Gaussian distribution with time-dependent parameters.
        """
        
        # For reproducibility
        # np.random.seed(1)

        
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = params["mean"], 
            covariance_matrix = self.cov_matrix(params))
        
        return dist.sample()
    
    # def pdf(self, params, x):
        
    #     k = torch.numel(x)
    #     mu = params["mean"]
    #     cov = construct_covariance_matrix(params["covariance"])
    #     icov = torch.linalg.pinv(cov)
    #     y = x.view(-1)
        
    #     c = 1.0 / ((2 * torch.pi) ** (k / 2) * torch.sqrt(torch.det(cov)))
    #     e = torch.exp(- 1 / 2 * (y - mu).T @ icov @ (y - mu))
        
    #     return c * e

    def log_probs(self, params, x):
        
        mu = params["mean"]
        cov = self.cov_matrix(params)
        icov = torch.linalg.pinv(cov)
        y = x.view(-1)
        eps = 1e-10
        
        return - 1/2 * (torch.log(torch.det(cov) + eps) + (y - mu) @ icov @ (y - mu))

class PeriodicGaussianData(GaussianData):
    
    def __init__(self, d : int, w : int):
        super().__init__(w, d)
        
        
    @override
    def mean_init(self, t : int) -> torch.tensor:

        # Ensure reproducibility
        torch.manual_seed(1)

        # Constructing the mean vector
        size = self.w * self.d
        mu_0 = torch.rand(size = (size,))
        delta_mu = 0.1 * torch.rand(size = (size,))

        mu = mu_0 #+ delta_mu * sin(t / 365 * 2 * torch.pi)
        
        return mu
    
