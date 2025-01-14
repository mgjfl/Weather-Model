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
        lower_triangular.diagonal().pow_(2)
        
        # Make sure L is invertible by ensuring a minimum value
        self.minimal_diagonal(lower_triangular)

        return lower_triangular

    def minimal_diagonal(self, L: torch.tensor, min_val: float = 1e-3) -> None:
        # Extract the diagonal (view) and modify in-place
        L_diag = torch.diagonal(L, 0)
        L_diag.abs_()  # Ensure diagonal elements are positive
        L_diag.clamp_(min=min_val)  # Enforce minimum value

        # No return needed since L is modified in-place
        pass
    
    def invert_cholesky(self, L : torch.tensor) -> torch.tensor:
        return L @ L.T
    
    def regularize_matrix(self, A: torch.tensor, epsilon : float = 1e-6) -> None:
        """
        Regularizes a symmetric matrix A by adding a small term to its diagonal to improve numerical stability.
        
        Parameters:
        - A (torch.tensor): Symmetric input matrix (modified in-place).
        
        Note:
        - The function modifies A in-place and does not return a value.
        """

        # Compute maximum eigenvalue
        lambda_max = torch.linalg.eigvalsh(A).max()

        # Compute the regularization value
        regularization = epsilon * lambda_max

        # Add regularization to the diagonal in-place
        A.diagonal().add_(regularization)
        
    
    def array_to_spd(self, array : torch.tensor, L : torch.tensor = None) -> torch.tensor:
        """
        Transforms array into a lower triangular matrix with positive diagonal and
        subsequently constructs C as a Cholesky decomposition of array.
        """
        
        if L is None:
            L = self.array_to_lower_triangular(array)

        # Cholesky decomposition
        cov = self.invert_cholesky(L)

        # Add regularization proportional to the largest eigenvalue
        # For numerical stability (xT(A+D)x=xTAx+xTDx>0)
        self.regularize_matrix(cov)
        
        
        return cov

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
        L = self.array_to_lower_triangular(params["lower_tril"])
        n = L.shape[0]
        
        
        y = x.view(-1)
        # Log det for Cholesky decomposition: https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        log_det_cov = 2 * torch.sum(torch.log(torch.diagonal(L)))
        
        # det(A) = prod lambda_i --> regularize with geometric mean
        lambda_avg = torch.exp(log_det_cov / n)
        eps = 1e-6
        
        cov = self.invert_cholesky(L) + eps * lambda_avg * torch.eye(n, device=L.device)
        
        # Step 1: Solve L y = b for y (forward substitution)
        b = (y - mu)
        z = torch.linalg.solve(cov, b.reshape(-1, 1))

        log_probs = - 1/2 * (log_det_cov + b @ z)
        
        if torch.isnan(log_probs).any():
            print(f"{log_probs=}")
            print(f"{log_det_cov=}")
            print(f"{L=}")
            print(f"{L.max()=}")
            print(f"{L.min()=}")
            print(f"{L.diag()=}")
            print(f"{(y-mu)=}")
            print(f"{torch.linalg.pinv(L)=}")
            print(f"{z=}")
            # print(f"{z1=}")
            # print(f"{z2=}")
            # print(f"{icov=}")
            print(f"{torch.linalg.cond(L)=}")
            
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

        mu = mu_0 #+ delta_mu * sin(t / 365 * 2 * torch.pi)
        
        return mu
    
