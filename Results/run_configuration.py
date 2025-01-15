from parser import *
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
from save_results import *
from cProfile import Profile
from pstats import SortKey, Stats
import glob
from collections.abc import Iterable
from torchinfo import summary # Note: summary from torchsummary gives incorrect results
import io
from extract_from_configuration import *

def run_configuration(config_name, verbose = False):
    
    # Load configuration
    config_manager = ConfigManager(config_name)
    config_manager.load_config()
    
    # Access configurations
    data_config     = config_manager.get_data_config()
    model_config    = config_manager.get_model_config()
    training_config = config_manager.get_training_config()
    output_config   = config_manager.get_output_config()

    # Identifying the project and run
    print(f"Running configuration:")
    print(f"{'project_name:':<15} {output_config['project_name']}")
    print(f"{'run_name:':<15} {output_config['run_name']}\n")
    
    # Set up device
    device = get_device()

    if verbose: print(f"Using {device} device")
        
    # Training parameters
    batch_size      = training_config["batch_size"]
    max_epochs      = training_config["epochs"]
    early_stopping  = training_config["early_stopping"]["enabled"]
    patience        = training_config["early_stopping"]["patience"] if early_stopping else np.inf
        
    if data_config["type"] == "synthetic":
        
        if verbose:
            print(f"\n-- Running configuration with synthetic data --\n\n")
            
        # Generates independent datasets for each run
        dataset_iterator = get_dataset(data_config=data_config, device=device)
        
        # Train the model multiple runs to characterize the randomness
        for (run_number, dataset) in enumerate(dataset_iterator):
            
            if verbose:
                print(f"Starting run {run_number + 1}/{data_config['n_runs']}.")
            
            # Create the model
            model = create_model(model_config=model_config, dataset=dataset, device=device)
            model = model.to(device)
            
            # Define training components
            loss_fn, optimizer, scheduler = get_training_components(model=model, training_config=training_config)

            # Prepare for saving the data
            data_saver = DataSaver(
                config = config_manager.config,
                run_number = run_number
                )

            # Train the model
            trained_model = train_model(
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
            data_saver.end_run(trained_model)
                
    pass

def profile_configuration(config_name):
    with Profile() as profile:
        run_configuration(config_name=config_name, verbose = False)

    s = io.StringIO()
    stats = Stats(profile, stream=s).strip_dirs().sort_stats(SortKey.TIME)
    
    config_manager = ConfigManager(config_name)
    config_manager.load_config()
    output_config   = config_manager.get_output_config()
    
    save_file = os.path.join(
            HOME_DIR,
            "Results",
            "Runs",
            output_config["project_name"], 
            output_config["run_name"],
            "profiler.txt"
            )
    

    stats.print_stats()

    with open(save_file, 'w+') as f:
        f.write(s.getvalue())
        
    print(f"Saved profile at {save_file}.")
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
    
    # Run an entire project (experiment with multiple configs)
    # run_project("TrainingEval")
    
    # Run a single configuration
    run_configuration("config_TFNO.yaml", verbose=False)

    # Profile a configuration
    # profile_configuration("config_FNO.yaml")
    
    # Get a model summary
    # device = get_device()
    # model, dataset = get_model_and_data_by_config("config_TFNO.yaml", device)
    # print(get_model_size(model))
    # get_model_summary(model, dataset, device)