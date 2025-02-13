
import sys
import os
import pathlib
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)
from neuralop.models import * 
from Data import *
from Training import *
from Models import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from parser import *
from save_results import *
from cProfile import Profile
from pstats import SortKey, Stats
import glob
from collections.abc import Iterable
from torchinfo import summary # Note: summary from torchsummary gives incorrect results
import io


def str_to_class(classname : str):
    """
    Translates class names from configuration files to classes.
    """
    return getattr(sys.modules[__name__], classname)

def create_model(data_config : dict, model_config : dict, dataset : WeatherDataset, device : torch.device) -> ACW:
    """
    Creates the model from the model configurations.

    Args:
        model_config (dict): Model configuration
    """
    
    # The models
    architecture    = model_config["arch"]
    model_type      = str_to_class(model_config["type"])
    
    # Construct the NN
    if architecture == "PNN":
        
        # Probabilistic model
        prob_model = str_to_class(model_config["prob_model"])(dataset.d, dataset.w)
        
        # The neural network
        model = PNN(
            probabilistic_model = prob_model,
            model_class         = model_type,
            in_channels         = dataset.get_in_channels(),
            grid_shape          = dataset.get_input_shape(),
            **model_config["parameters"]
        )
    # elif architecture == "GP":

    #     # The Neural Network
    #     model = GP(
    #         model_class = model_type,
    #         t = dataset.t,
    #         **data_config["parameters"],
    #         **model_config["parameters"]
    #     )
    else:
        raise NotImplementedError(f"Architecture ({architecture}) is not implemented.")

    model.to(device)
    return model

def get_training_components(model : ACW, training_config : dict, data_config : dict):
    """
    Returns the loss function, optimizer and scheduler.

    Args:
        model (nn.Module): The ML model
        training_config (dict): Training configuration
    """
    
     # Construct the loss function
    if issubclass(type(model), PNN):
        loss_fn = str_to_class(training_config["loss_function"])(model.get_prob_model())
    elif issubclass(type(model), GP):
        loss_fn = str_to_class(training_config["loss_function"])(model, data_config["parameters"]["n"])
    else:
        raise NotImplementedError(f"Architecture for {model} not implemented.")
    
    # Optimization settings
    learning_rate = training_config["learning_rate"]
    optimizer = str_to_class(training_config["optimizer"])(model.parameters(), lr = learning_rate)
    scheduler = str_to_class(training_config["scheduler"]["type"])(optimizer, **training_config["scheduler"]["parameters"])
    
    return loss_fn, optimizer, scheduler

def get_dataset(data_config : dict, device : torch.device):
    """
    Gets the dataset in the proper format.

    Args:
        data_config (dict): Dataset configuration
    """
    
    n, d, w = data_config["parameters"]["n"], data_config["parameters"]["d"], data_config["parameters"]["w"]
    
    
    if data_config["type"] == "synthetic":
        # Data generation model
        data_model : RandomSampler = str_to_class(data_config["gen_model"])(d, w)
        
        # Dataset class
        dataset_class = str_to_class(data_config["dataset"])
        
        # Seed for each data generation
        rng = np.random.default_rng(data_config["global_seed"])
        sub_seeds = rng.integers(0, 1e6, size = data_config["n_runs"])
        
        for seed in sub_seeds:
            # First generate the data
            observations                = data_model.generate(n, seed)
            dataset : WeatherDataset    = dataset_class(observations).to(device)
            yield(dataset)
            
    else:
        raise NotImplementedError(f"Dataset for real data not implemented.")
    
    pass

def get_device() -> torch.device:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def get_model_size(model : nn.Module) -> int:
    """
    Returns the number of trainable parameters for a model

    Args:
        model (nn.Module): The PyTorch model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_and_data_by_config(config_name : str, device : torch.device):
     # Load configuration
    config_manager = ConfigManager(config_name)
    config_manager.load_config()
    
    # Access configurations
    data_config     = config_manager.get_data_config()
    model_config    = config_manager.get_model_config()
    
    dataset = get_dataset(data_config, device)

    if isinstance(dataset, Iterable):
        dataset = next(dataset)

    model = create_model(data_config, model_config, dataset, device)
    return model, dataset

def get_model_summary(model : nn.Module, dataset : WeatherDataset, device : torch.device):
    summary(model, input_size = dataset[0][0].unsqueeze(0).shape, device = device, depth = 10, verbose = 1, row_settings=["var_names"])
    pass

    