
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
from  gpytorch.mlls import *
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
    if classname is None:
        return None
    try:
        return getattr(sys.modules[__name__], classname)
    except:
        print(f"Class : {classname} not found.")
        return None

def create_model(data_config : dict, model_config : dict, dataset : WeatherDataset, device : torch.device) -> ACW:
    """
    Creates the model from the model configurations.

    Args:
        model_config (dict): Model configuration
    """
    
    # The models
    architecture        = model_config["arch"]
    mean_model_class    = str_to_class(model_config["mean_class"])
    cov_model_class     = str_to_class(model_config["cov_class"]) 
    dropout_rate        = model_config["dropout_rate"]

    # Additional model parameters
    model_kwargs = model_config["parameters"] if model_config["parameters"] is not None else dict()
    
    # Construct the NN
    if architecture == "PNN":
        
        # Probabilistic model
        prob_model = str_to_class(model_config["prob_model"])(dataset.get_in_channels(), dataset.get_domain_shape())

        
        # The neural network
        model = PNN(
            probabilistic_model     = prob_model,
            mean_model_class        = mean_model_class,
            covariance_model_class  = cov_model_class,
            in_channels             = dataset.get_in_channels(),
            domain_shape            = dataset.get_domain_shape(),
            **model_kwargs
        )
    elif architecture == "EGP":

        train_x, train_y = dataset.get_train_xy()

        # The Neural Network
        model = EGP(
            train_x             = train_x,
            train_y             = train_y,
            mean_model_class    = mean_model_class,
            in_channels         = dataset.get_in_channels(),
            domain_shape        = dataset.get_domain_shape(),
            **model_kwargs
        )
    elif architecture == "VariationalGP":

        train_x, train_y = dataset.get_train_xy()
        num_data = train_y.size(0)
        

        # The Neural Network
        model = GPWrapper(
            gp_model_class      = VariationalGP,
            mean_model_class    = mean_model_class,
            in_channels         = dataset.get_in_channels(),
            domain_shape        = dataset.get_domain_shape(),
            num_data            = num_data,
            dropout_rate        = dropout_rate,
            dataset             = dataset,
            **model_kwargs
        )

    # elif architecture == "VGP":

    #     train_x, train_y = dataset.get_train_xy()
    #     num_data = train_y.size(0)


    #     # The Neural Network
    #     model = VGP(
            # mean_model_class    = mean_model_class,
            # in_channels         = dataset.get_in_channels(),
            # domain_shape        = dataset.get_domain_shape(),
            # num_data            = num_data,
    #         **model_kwargs
    #     )
    elif architecture == "StationVGP":
        
        # train_x, train_y = dataset.get_train_xy()
        # num_data = train_y.size(0)
        
        num_data = dataset.get_num_train_data()

        print("Defining the model...")
        model = StationVGP(
                mean_model_class    = mean_model_class,
                gp_model_class      = VariationalGP,
                in_channels         = dataset.get_in_channels(),
                domain_shape        = dataset.get_domain_shape(),
                num_data            = num_data,
                dataset             = dataset,
                 **model_kwargs
        )
        

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
        loss_fn = str_to_class(training_config["loss_function"])(
            prob_model = model.get_prob_model(),
            input_type = data_config["input_type"]
            )
    elif issubclass(type(model), EGP) or issubclass(type(model), GPWrapper) or issubclass(type(model), StationVGP):
        loss_fn = str_to_class(training_config["loss_function"])(
            model                   = model,
            regularization_weight   = training_config["regularization_weight"],
            grad_regularization     = training_config["grad_regularization"])
    else:
        raise NotImplementedError(f"Architecture for {model} not implemented.")
    
    # Optimization settings
    learning_rate = training_config["learning_rate"]
    try:
        if issubclass(type(model), GPWrapper) or issubclass(type(model), StationVGP):
            hyperparameter_optimizer = str_to_class(training_config["optimizer"])
            optimizer = VGPOptimizer(
                model = model, 
                hyperparameter_optimizer = hyperparameter_optimizer,
                ngd_lr = 0.1,
                hyper_lr = learning_rate)
            scheduler = str_to_class(training_config["scheduler"]["type"])(optimizer.hyper, **training_config["scheduler"]["parameters"])
        else:
            optimizer = str_to_class(training_config["optimizer"])(model.parameters(), lr = learning_rate)
            scheduler = str_to_class(training_config["scheduler"]["type"])(optimizer, **training_config["scheduler"]["parameters"])
    except ValueError as e:
        print(e)
        print("Proceeding with optimizer = None")
        optimizer = None
        scheduler = None
    
    return loss_fn, optimizer, scheduler

def get_dataset(data_config : dict, device : torch.device):
    """
    Gets the dataset in the proper format.

    Args:
        data_config (dict): Dataset configuration
    """

    
    if data_config["type"] == "synthetic":

        # Location where synthetic data is stored
        synthetic_dir = os.path.join(HOME_DIR, "Data", "Synthetic", "Model_1")

        # Data parameters
        setting         = data_config["setting"]
        training_window = data_config["training_window"]
        train_ratio     = data_config["train_ratio"] if "train_ratio" in data_config.keys() else  0.8
        test_ratio     = data_config["test_ratio"] if "test_ratio" in data_config.keys() else  0.2
        # reserve_days    = data_config["reserve_days"] if "reserve_days" in data_config.keys() else 0

        if data_config["input_type"] == "data":
                # Load the pre-generated observations
                observations    = np.load(os.path.join(synthetic_dir, f"data_case_{setting}.npy"))
                dataset         = PastNDaysForecastDataset(
                                    observations    = observations, 
                                    training_window = training_window,
                                    train_ratio     = train_ratio,
                                    test_ratio      = test_ratio).to(device) 
        elif data_config["input_type"] ==  "parameters":
                # Load the pre-generated observations
                mu              = np.load(os.path.join(synthetic_dir, f"mu_case_{setting}.npy"))
                cov             = np.load(os.path.join(synthetic_dir, f"cov_case_{setting}.npy"))
                dataset         = PastNDaysForecastDataset(
                                    observations    = mu, 
                                    covariances     = cov,
                                    training_window = training_window,
                                    train_ratio     = train_ratio,
                                    test_ratio      = test_ratio).to(device)
        elif data_config["input_type"] ==  "evaluation":
                # Load the pre-generated observations
                observations    = np.load(os.path.join(synthetic_dir, f"data_case_{setting}.npy"))
                cov             = np.load(os.path.join(synthetic_dir, f"cov_case_{setting}.npy"))
                dataset         = PastNDaysForecastDataset(
                                    observations    = observations, 
                                    covariances     = cov,
                                    training_window = training_window,
                                    train_ratio     = train_ratio,
                                    test_ratio      = test_ratio).to(device)
        else:
            raise NotImplementedError(f"Data input type {data_config['input_type']} is not implemented.")
            


    elif data_config["type"] == "real":
        dataset =  create_GridStationDataset(
                        training_window = data_config["training_window"]
                    )
            
    else:
        raise NotImplementedError(f"Dataset for {data_config['type']} data not implemented.")
    
    dataset = dataset.to(device)
    return dataset

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

    