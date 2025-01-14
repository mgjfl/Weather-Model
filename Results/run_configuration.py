from parser import *
import sys
import os
import pathlib
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)
from neuralop.models import * 
from Data import *
from Training import *
from Models.neural_models import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from save_results import *
from cProfile import Profile
from pstats import SortKey, Stats
import glob


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def run_configuration(config_name, verbose = False):
    
    # Load configuration
    config_manager = ConfigManager(config_name)
    config_manager.load_config()
    
    # Access configurations
    data_config = config_manager.get_data_config()
    model_config = config_manager.get_model_config()
    training_config = config_manager.get_training_config()
    output_config = config_manager.get_output_config()

    # Identifying the project and run
    print(f"Running configuration:")
    print(f"{'project_name:':<15} {output_config['project_name']}")
    print(f"{'run_name:':<15} {output_config['run_name']}\n")
    
    # Set up device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if verbose:
        print(f"Using {device} device")
        
    n, d, w = data_config["parameters"]["n"], data_config["parameters"]["d"], data_config["parameters"]["w"]
        
    # The models
    architecture    = model_config["arch"]
    model_type      = str_to_class(model_config["type"])
    
    # Optimization settings
    learning_rate = training_config["learning_rate"]
    batch_size = training_config["batch_size"]
    max_epochs = training_config["epochs"]
    
    early_stopping = training_config["early_stopping"]["enabled"]
    patience = training_config["early_stopping"]["patience"] if early_stopping else np.inf
    
        
    if data_config["type"] == "synthetic":
        
        if verbose:
            print(f"\n-- Running configuration with synthetic data --\n\n")
        
        # Data generation model
        data_model = str_to_class(data_config["gen_model"])(d, w)
        
        # Dataset class
        dataset_class = str_to_class(data_config["dataset"])
        
        # Seed for each data generation
        rng = np.random.default_rng(data_config["global_seed"])
        sub_seeds = rng.integers(0, 1e6, size = data_config["n_runs"])

        
        if architecture == "PNN":
            prob_model = str_to_class(model_config["prob_model"])(d, w)
        
        for (run_number, seed) in enumerate(sub_seeds):
            
            if verbose:
                print(f"Starting run {run_number + 1}/{data_config['n_runs']}.")
            
            # First generate the data
            observations    = data_model.generate(n, seed)
            dataset         = dataset_class(observations).to(device)
            
            # Construct the NN
            if architecture == "PNN":
                model = PNN(
                    probabilistic_model= prob_model,
                    model_class = model_type,
                    in_channels=dataset.in_channels,
                    grid_shape = dataset.get_input_shape(),
                    **model_config["parameters"]
                )
                loss_fn = str_to_class(training_config["loss_function"])(prob_model)
                
            # Finalize model and configure optimizer
            model = model.to(device)

            # Optimization parameters
            optimizer = str_to_class(training_config["optimizer"])(model.parameters(), lr = learning_rate)
            scheduler = str_to_class(training_config["scheduler"]["type"])(optimizer, **training_config["scheduler"]["parameters"])

            # Prepare for saving the data
            data_saver = DataSaver(
                config = config_manager.config,
                run_number = run_number
                )

            # Train the model
            train_model(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                loss_fn=loss_fn,
                optimizer=optimizer,
                EPOCHS=max_epochs,
                patience=patience,
                scheduler=scheduler,
                data_saver=data_saver,
                verbose=verbose
            )
            
            # Save run data
            data_saver.end_run()
                
    pass

def profile_configuration(config_name):
    with Profile() as profile:
        run_configuration(config_name=config_name, verbose = False)

    stats = Stats(profile).strip_dirs().sort_stats(SortKey.TIME)
    stats.print_stats()
    return stats

def run_project(project_name : str):
    """Runs all configurations with a certain project name.

    Args:
        project_name (str): Name of the project
    """

    config_dir = os.path.join(HOME_DIR, "Results", "Configurations")

    files           = glob.glob(os.path.join(config_dir, "**", "*.yaml"), recursive=True)
    project_names   = []
    
    for file in files:
        with open(file, "r") as f:
            project_names.append(yaml.safe_load(f)["output"]["project_name"])

    idxs = [x == project_name for x in project_names]
    correct_files = np.array(files)[idxs]

    if correct_files.shape[0] == 0:
        print("No files match that project name.")
        return
    
    for long_config_name in correct_files:
        
        # Put it in the correct format
        config_name = long_config_name.replace(config_dir, "")[1:]
        run_configuration(config_name=config_name)
    
    print("Finished configuration.\n")
    pass

if __name__ == "__main__":
    run_project(sys.argv[1])