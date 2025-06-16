
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
from extract_from_configuration import *

def run_configuration(config_name, verbose = False, useConfigName = False, device = None, max_batches = None):
    """
    Runs a configuration file.
    """
    
    if useConfigName:
        run_name = config_name.split(os.sep)[-1].split(".")[0]
    else:
        run_name = None
    
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
    if useConfigName:
        print(f"{'run_name:':<15} {run_name}\n")
    else:
        print(f"{'run_name:':<15} {output_config['run_name']}\n")
    
    # Set up device
    if device is None:
        device = get_device()

    if verbose: print(f"Using {device} device")
        
    # Training parameters
    batch_size      = training_config["batch_size"] if model_config["arch"] != "GP" else 1
    max_epochs      = training_config["epochs"]
    early_stopping  = training_config["early_stopping"]["enabled"]
    patience        = training_config["early_stopping"]["patience"] if early_stopping else np.inf
        
    # if data_config["type"] == "synthetic":
        
    if verbose:
        print(f"\n-- Running configuration with {data_config['type']} data --\n\n")
        
    # Generates independent datasets for each run
    dataset = get_dataset(data_config=data_config, device=device)
        
    # Create the model
    model = create_model(data_config=data_config, model_config=model_config, dataset=dataset, device=device)
    model = model.to(device)
    
    # Define training components
    loss_fn, optimizer, scheduler = get_training_components(model=model, training_config=training_config, data_config=data_config)


        
    # Prepare for saving the data
    data_saver = DataSaver(
        config = config_manager.config,
        run_number = 0,
        run_name = run_name
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
        verbose=verbose, 
        max_batches = max_batches
    )
    
    # Save run data
    data_saver.end_run(trained_model)
    # pass

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

def run_folder(folder_name : str, skip_files = [], verbose = False, useConfigName = False, device = None, max_batches = None):
    """
    Runs each configuration within a folder.
    """

    configs = os.listdir(os.path.join(HOME_DIR, "Results", "Configurations", folder_name))


    for config in configs:
        run_config = True
        for model in skip_files:
            if model in config:
                run_config = False

        if run_config:
            run_configuration(os.path.join(folder_name, config), verbose = verbose, useConfigName = useConfigName, device = device, max_batches = max_batches)
        torch.cuda.empty_cache()

    pass
    

if __name__ == "__main__":
    
    # run_folder("InferenceSpeed", 
    #            skip_files= ["_" + str(2**x) + ".yaml" for x in range(0, 10)],
    #            verbose = True,
    #            useConfigName = True,
    #            max_batches = 20)
    
    # run_folder("InferenceSpeed_II_100", 
    #         verbose = False,
    #         skip_files=["_6.", "_12.", "_25.", "_50."],
    #         useConfigName = True,
    #         max_batches = None)
    
    # run_folder("InferenceSpeed_II_EGP_100", 
    #         verbose = True,
    #         useConfigName = True,
    #         max_batches = None)
    
    # run_folder("InferenceSpeedOnly_II_EGP_100", 
    #         verbose = True,
    #         skip_files = ["_" + str(x) + "." for x in [100, 1000, 200, 300, 400, 500, 600, 700, 800]],
    #         useConfigName = True,
    #         max_batches = None)
    
    # run_configuration(os.path.join("InferenceSpeed_II_EGP_100", "II_EGP_AFNO_25.yaml"), verbose = True, useConfigName = True)
    # run_configuration(os.path.join("InferenceSpeed_II_EGP_100", "II_EGP_AFNO_50.yaml"), verbose = True, useConfigName = True)
    run_configuration(os.path.join("InferenceSpeed_II_EGP_100", "II_EGP_AFNO_75.yaml"), verbose = True, useConfigName = True)
    # run_configuration(os.path.join("InferenceSpeed_II_100", "II_VGP_AFNO_12.yaml"), verbose = True, useConfigName = True)
    # run_configuration(os.path.join("InferenceSpeed_II_100", "II_VGP_AFNO_25.yaml"), verbose = True, useConfigName = True)
    # run_configuration(os.path.join("InferenceSpeed_II_100", "II_VGP_AFNO_50.yaml"), verbose = True, useConfigName = True)
    
    # run_configuration(os.path.join("synthetic_vgp_III", "III_VGP_MGST.yaml"), verbose=True)
    
    
    # Run an entire project (experiment with multiple configs)
    # run_project("")
    
    # Run a single configuration
    # run_configuration(os.path.join("synthetic_vgp_III", "III_VGP_MGST.yaml"), verbose=True)
    
    
    # run_configuration(os.path.join("Real_1", "real_VGP_AFNO.yaml"), verbose=False)
    # run_configuration(os.path.join("Real_3", "real_VGP_AFNO.yaml"), verbose=False)
    # run_configuration(os.path.join("Real_7", "real_VGP_AFNO.yaml"), verbose=False)
    # run_configuration(os.path.join("Real_14", "real_VGP_AFNO.yaml"), verbose=False)
    # run_configuration(os.path.join("Real_4", "real_VGP_AFNO_adapted.yaml"), verbose=True)
    # run_configuration(os.path.join("Real_5", "real_VGP_AFNO.yaml"), verbose=False)
    # run_configuration(os.path.join("Real_1", "real_VGP_FNO.yaml"), verbose=False)
    # run_configuration(os.path.join("synthetic_vgp_III", "III_VGP_FNN.yaml"), verbose=False)
    
# run_configuration(os.path.join("Real", "real_VGP_FNO.yaml"), verbose=True)
    # run_configuration(os.path.join("Real", "real_VGP_AFNO.yaml"), verbose=True)
    
    # run_configuration(os.path.join("Real", "real_VGP_FNO_7.yaml"), verbose=False)
    # run_configuration(os.path.join("Real", "real_VGP_AFNO_7.yaml"), verbose=False)

    # Run folder
    # run_folder("synthetic_mean_I")
    # run_folder("synthetic_mean_II")
    # run_folder("synthetic_mean_III")

    # run_folder("synthetic_cov_I")
    # run_folder("synthetic_cov_II")
    # run_folder("synthetic_cov_III")

    # run_folder("synthetic_full_I", ["FNN", "GP_FNO", "MGST"])
    # run_folder("synthetic_full_II")
    # run_folder("synthetic_full_III", ["FNN", "GP_FNO"])

    # run_folder("synthetic_vgp_I")
    # run_folder("synthetic_vgp_II", ["MGST"])
    # run_folder("synthetic_vgp_III", ["MGST"])

    # Profile a configuration
    # profile_configuration("config_FNO.yaml")
    
    # Get a model summary
    # device = get_device()
    # model, dataset = get_model_and_data_by_config("config_TFNO.yaml", device)
    # print(get_model_size(model))
    # get_model_summary(model, dataset, device)