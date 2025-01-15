from .data_models import *
from typing_extensions import override
from numpy import sin, log

class GaussianData(RandomSampler):
    
    def __init__(self, d : int, w : int):
        super().__init__(
            d, w, {
            "mean" : torch.tensor([w * d]), 
            "lower_tril" : torch.tensor([(w * d * (w * d + 1)) // 2])
            }
        )
        
        self.covariance_matrix = None
    
    def covariance_init(self, seed : int) -> torch.tensor:
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
        self.minimal_diagonal(C, 1e-3)
        
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
        
        # Make sure the diagonal entries are positive we predict log(l_ii)
        # https://arxiv.org/pdf/1802.07079
        lower_triangular[torch.arange(n), torch.arange(n)] = torch.exp(torch.diag(lower_triangular))
        
        # Make sure L is invertible by ensuring a minimum value
        self.minimal_diagonal(lower_triangular)

        return lower_triangular

    def minimal_diagonal(self, L: torch.tensor, min_val: float = 1e-6) -> None:
        # Extract the diagonal (view) and modify in-place
        L_diag = torch.diagonal(L, 0)
        L_diag.abs_()  # Ensure diagonal elements are positive
        L_diag.clamp_(min=min_val)  # Enforce minimum value

        # No return needed since L is modified in-place
        pass
    
    def invert_cholesky(self, L : torch.tensor) -> torch.tensor:
        return L @ L.T
    
    # def regularize_cov_matrix(self, A: torch.tensor, epsilon : float = 1e-6) -> None:
    #     """
    #     Regularizes a symmetric matrix A by adding a small term to its diagonal to improve numerical stability.
        
    #     Parameters:
    #     - A (torch.tensor): Symmetric input matrix (modified in-place).
        
    #     Note:
    #     - The function modifies A in-place and does not return a value.
    #     """

    #     # Add regularization to the diagonal in-place
    #     A.diagonal().add_(epsilon)
        
    
    # def array_to_precision_matrix(self, array : torch.tensor, L : torch.tensor = None) -> torch.tensor:
    #     """
    #     Transforms array into a lower triangular matrix with positive diagonal and
    #     subsequently constructs C as a Cholesky decomposition of array.
    #     """
        
    #     if L is None:
    #         L = self.array_to_lower_triangular(array)

    #     # Cholesky decomposition
    #     precision_matrix = self.invert_cholesky(L)

    #     # Add regularization proportional to the largest eigenvalue
    #     # For numerical stability (xT(A+D)x=xTAx+xTDx>0)
    #     self.regularize_matrix(precision_matrix)
        
        
    #     return precision_matrix

    def mean_init(self, t : int, seed : int) -> torch.tensor:

        # Ensure reproducibility
        torch.manual_seed(seed)
        
        return torch.rand(size = (self.parameter_shapes["mean"],))
        
    @override
    def parameter_init(self, t: int, seed : int) -> dict[str, torch.tensor]:
        return {
            "mean" : self.mean_init(t, seed),
            "covariance" : self.covariance_init(seed) if self.covariance_matrix is None else self.covariance_matrix
        }
        
    def cov_matrix(self, params : dict[str, torch.tensor]) -> torch.tensor:
        """Returns the covariance matrix

        Args:
            params (dict[str, torch.tensor]): parameters
        """
        if "covariance" in params.keys():
            return params["covariance"]
        else:
            L = self.array_to_lower_triangular(params["lower_tril"])
            prec_matrix = self.invert_cholesky(L)
            return torch.linalg.pinv(prec_matrix)
    
    def sample(self, params):
        """
        Samples from a multivariate Gaussian distribution with time-dependent parameters.
        """
        
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = params["mean"], 
            covariance_matrix = self.cov_matrix(params))
        
        return dist.sample()
    
    # def stable_lower_tri(self, params = None, L = None):

    #     if L is None:
    #         L = self.array_to_lower_triangular(params["lower_tril"])
    #     cov = self.invert_cholesky(L)
    #     self.minimal_diagonal(cov, 1e-3)
    #     L2, info = torch.linalg.cholesky_ex(cov)
        
    #     return L2, info

    def log_probs(self, params, x):
        
        
        mu = params["mean"]
        L = self.array_to_lower_triangular(params["lower_tril"])
        # L, info = self.stable_lower_tri(params)
        
        # if not info.eq(0):
        #     return torch.tensor(torch.nan, device = L.device)
        
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
            
        # Log-determinant
        log_probs = - 1/2 * log_det_cov
        
        # Reconstruction error
        log_probs -= 1/2 * (z.T @ z).item()
        
        # Additional penalization independent of the covariance matrix
        log_probs -= torch.linalg.norm(b)
        
        # Regularization for exploding covariance
        log_probs -= torch.linalg.matrix_norm(L)
        
        # print(f"{log_probs=}")
        # print(f"{L=}")
        # print(f"{log_det_cov=}")
        # print(f"{torch.linalg.matrix_norm(L)=}")
        # print(f"{torch.linalg.norm(b)=}")
        # print(f"{log_probs=}\n")
        # print(f"{z.T @ z=}")
        
        return log_probs

class PeriodicGaussianData(GaussianData):
    
    def __init__(self, d : int, w : int):
        super().__init__(w, d)
        
        
    @override
    def mean_init(self, t : int, seed : int) -> torch.tensor:

        # Ensure reproducibility
        torch.manual_seed(seed)

        # Constructing the mean vector
        size = self.w * self.d
        mu_0 = torch.rand(size = (size,))
        delta_mu = 0.1 * torch.rand(size = (size,))

        mu = mu_0 + delta_mu * sin(t / 365 * 2 * torch.pi)
        
        return mu
    
class LinearGaussian(GaussianData):
    
    def __init__(self, d : int, w : int):
        super().__init__(w, d)
        
        
    @override
    def mean_init(self, t : int, seed : int) -> torch.tensor:

        # No time dependence in the base value
        torch.manual_seed(seed)

        # Constructing the mean vector
        size = self.w * self.d
        mu_0 = torch.rand(size = (size,))
        # Random on [-0.05, 0.05]
        delta_mu = 0.1 * (torch.rand(size = (size,)) - 0.5)

        mu = mu_0 + delta_mu * t
        
        return mu
    
