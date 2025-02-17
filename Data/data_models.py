
import torch
from abc import abstractmethod, ABCMeta
from typing import Dict

class ProbabilisticModel(metaclass = ABCMeta):
    
    def __init__(self, d : int, w : int, parameter_shapes : Dict[str, torch.tensor]):
        """Initializes the parameter sizes

        Args:
            w (int) : number of weather variables
            d (int) : number of locations
            parameter_size (dict[str, torch.tensor]): dictionary as param_name : param_shape
        """
        
        # Constants that specify the sample output shape
        self.w = w
        self.d = d

        # Parameter metrics
        self.parameter_shapes   = parameter_shapes
        self.parameter_sizes    = torch.tensor(list(map(lambda x: torch.prod(x), self.parameter_shapes.values())))
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
        data = torch.zeros(size = (n, self.d, self.w))
        
        # Set global seed
        torch.manual_seed(seed)

        # Sample for each day
        for t in range(n):
            parameters = self.parameter_init(t, seed)
            torch.manual_seed(seed + t) # Reset fixed seed from self.paramter_init
            data[t] = self.sample(parameters).reshape(self.d, self.w)

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
