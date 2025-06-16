import numpy as np
import os
import sys
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr
import pandas as pd
DATA_DIR = os.path.dirname(__file__)
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)
from Results.extract_from_configuration import get_dataset, get_device
from Results.results_util import *

def load_obs(setting):
    full_samples  = np.load(os.path.join(DATA_DIR, "Synthetic", "Model_1", f"data_case_{'I' * setting}.npy"))
    return full_samples

def compute_ensemble(setting, model_name, M, model_location):
    
    is_real = "Real" in setting
    
    # Load configuration
    if is_real:
        project_name = setting
    else:
        project_name = f"synthetic_vgp_{'I' * setting}"
    project_data = extract_project_data(project_name)

    # Prepare for model computation
    device = get_device()
    if model_location is None:
        model_location = project_data[model_name]["model_name"][0]
        
    print(model_location)
    model = torch.load(model_location, weights_only=False)
    model.eval();
    model.to(device)

    dataset = get_observations(project_data=project_data, run_name=model_name)
    W = dataset.get_in_channels()
    V, H, T = dataset.get_domain_shape()
    
    if is_real:
        data_loader = dataset.get_test_loader(128, shuffle = False)
        N = len(dataset.test_dataset)
        W_real = len(dataset.output_weather_vars)
        L_real = len(dataset.station_numbers)
        ensemble = np.empty(shape = (N, M, W_real, L_real))
    else:
        data_loader = DataLoader(dataset, batch_size=128)
        N = len(dataset)
        ensemble = np.empty(shape = (N, M, W, V, H))

    # Extract the predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        i = 0
        for x, y in data_loader:
            
            # Batch size
            X_size = x.shape[0]
            
            # The Multivariate Normal
            pred          = model(x)
            
            # Sample from the distribution [B, M, W, V, H]
            samples_over_batches = pred.sample(sample_shape = torch.Size([M]))
            
            if is_real:
                samples_over_batches = samples_over_batches\
                    .permute(1, 0, 2)\
                    .reshape(X_size, M, L_real, W_real)\
                    .permute(0, 1, 3, 2)
            else:
                samples_over_batches = samples_over_batches\
                    .permute(1, 0, 2)\
                    .reshape(X_size, M, W, V, H)
                    
            ensemble[i:(i + X_size)] = samples_over_batches.cpu().numpy()
            
            i += X_size
            
    return ensemble

def create_csv(setting, model_name = "AFNONet_VariationalGP", M = 51, model_location = None):
    
    is_real = "Real" in setting
    
    # Load observations
    if is_real:
        project_name = setting
        project_data = extract_project_data(project_name)

        dataset = get_observations(project_data=project_data, run_name=model_name)
        T = dataset.training_window
        
        test_indices = dataset.test_indices
        full_samples = np.stack([
            dataset.output_cache[t_idx].cpu().detach().numpy()
            for t_idx in test_indices
        ]).transpose(2, 0, 1)
        
        L, N_full, W = full_samples.shape
        
        # Create multi-index for V, H, N
        l_idx, n_idx = np.meshgrid(
            np.arange(L), np.arange(T, N_full + T), indexing='ij'
        )
        
        arr_reshaped = full_samples.reshape(-1, W)

        # Flatten the indices
        df_obs = pd.DataFrame({
            'L': l_idx.flatten(),
            'N': n_idx.flatten(),
            'W0': arr_reshaped[:, 0],
            'W1': arr_reshaped[:, 1]
        })
        
        # Compute the ensemble
        ensemble = compute_ensemble(setting, model_name, M, model_location)
        N = ensemble.shape[0]
        ensemble_reshaped = ensemble.transpose(0, 1, 3, 2).reshape(-1, W) # [N, M, L W]

        # Create multi-index for V, H, N
        n_idx, m_idx, l_idx = np.meshgrid(
            np.arange(N_full - N + T, N_full + T), np.arange(M), np.arange(L), indexing='ij'
        )

        # Flatten the indices
        df_ensemble = pd.DataFrame({
            'N': n_idx.flatten(),
            'M': m_idx.flatten(),
            'L': l_idx.flatten(),
            'Model_W0': ensemble_reshaped[:, 0],
            'Model_W1': ensemble_reshaped[:, 1]
        })

        pivoted = (
            df_ensemble.pivot(index=['N', 'L'], columns='M', values=['Model_W0', 'Model_W1'])
        )

        # Flatten the MultiIndex columns
        pivoted.columns = [f'Model_W{i[-1]}_{m}' for i, m in pivoted.columns]

        # Reset index to bring N, V, H back as columns
        pivoted = pivoted.reset_index()

        # Store the model
        pivoted["model"] = model_name

        # Prepare for saving
        df_combined = pd.merge(df_obs, pivoted, on = ["N", "L"])
        df_combined["station"] = df_combined["L"]
        df_combined["Date"] = dataset.test_times[df_combined["N"]]
        
        df_combined.to_csv(os.path.join(DATA_DIR, "Real", "AI_Ensemble", f"real_model_{model_name}_ens_members_{M}.csv"))
    
    else:
        full_samples = load_obs(setting)
        
        W, V, H, N_full = full_samples.shape
        
        # Put in a dataframe
        arr_reshaped = full_samples.transpose(1, 2, 3, 0).reshape(-1, W)

        # Create multi-index for V, H, N
        v_idx, h_idx, n_idx = np.meshgrid(
            np.arange(V), np.arange(H), np.arange(N_full), indexing='ij'
        )

        # Flatten the indices
        df_obs = pd.DataFrame({
            'V': v_idx.flatten(),
            'H': h_idx.flatten(),
            'N': n_idx.flatten(),
            'W0': arr_reshaped[:, 0],
            'W1': arr_reshaped[:, 1]
        })

        # Compute the ensemble
        ensemble = compute_ensemble(setting, model_name, M)
        N = ensemble.shape[0]
        ensemble_reshaped = ensemble.transpose(0, 1, 3, 4, 2).reshape(-1, W) # [N, M, V, H, W]

        # Create multi-index for V, H, N
        n_idx, m_idx, v_idx, h_idx = np.meshgrid(
            np.arange(N_full - N + 1, N_full + 1), np.arange(M), np.arange(V), np.arange(H), indexing='ij'
        )

        # Flatten the indices
        df_ensemble = pd.DataFrame({
            'N': n_idx.flatten(),
            'M': m_idx.flatten(),
            'V': v_idx.flatten(),
            'H': h_idx.flatten(),
            'Model_W0': ensemble_reshaped[:, 0],
            'Model_W1': ensemble_reshaped[:, 1]
        })

        pivoted = (
            df_ensemble.pivot(index=['N', 'V', 'H'], columns='M', values=['Model_W0', 'Model_W1'])
        )

        # Flatten the MultiIndex columns
        pivoted.columns = [f'Model_W{i[-1]}_{m}' for i, m in pivoted.columns]

        # Reset index to bring N, V, H back as columns
        pivoted = pivoted.reset_index()

        # Store the model
        pivoted["model"] = model_name

        # Prepare for saving
        df_combined = pd.merge(df_obs, pivoted, on = ["N", "V", "H"])
        df_combined["station"] = df_combined["V"] + df_combined["H"] * (df_combined["V"].max() + 1)

        df_selection = df_combined[(df_combined["V"] % 2 == 0) & 
                                (df_combined["H"] % 4 == 0)]

        df_selection.to_csv(os.path.join(DATA_DIR, "Synthetic", "AI_Ensemble", f"toy_s_{setting}_model_{model_name}_ens_members_{M}.csv"))

if __name__ == "__main__":
    
    # Create the datasets
    
    ###############
    ## Setting I ##
    ###############
    
    # create_csv(
    #     setting     = 1, 
    #     model_name  = "AFNONet_VariationalGP", 
    #     M           = 51
    # )
    
    ################
    ## Setting II ##
    ################
        
    # create_csv(
    #     setting     = 2, 
    #     model_name  = "AFNONet_VariationalGP", 
    #     M           = 51
    # )
    
    #################
    ## Setting III ##
    #################
    
    # create_csv(
    #     setting     = 3, 
    #     model_name  = "AFNONet_VariationalGP", 
    #     M           = 51
    # )
    
    #####################
    ## Real-world data ##
    #####################
    
    create_csv(
        setting         = "Real_4", 
        model_name      = "AFNONet_StationVGP-4", 
        M               = 51,
        model_location  = os.path.join(HOME_DIR, "Results", "FineTunedModels", "model_5.pt")
    )