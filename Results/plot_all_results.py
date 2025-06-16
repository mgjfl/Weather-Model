from results_util import *
import os
from itertools import combinations
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
import numpy as np
from Data import plot_contours, get_vmins_and_vmaxs
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_TEMP = 293

def sort_key(label):
    if label.startswith("MGST"):
        return (0, 0)  # MGST first, no need to sort within
    match = re.search(r'(\d+)$', label)
    number = int(match.group(1)) if match else float('inf')
    return (1, number)  # AFNONet sorted by trailing number

def plot_metric(project_name, data, setting, metric='val_loss', scale = "linear", fig_dir = None, from_EPOCH = 1):
    """
    Plot metric for multiple models, runs, and epochs.
    
    Parameters:
        data (dict): A dictionary where data[model][metric] is an n x m numpy array 
                     (n = number of runs, m = number of epochs).
        metric (str): The metric to plot (e.g., 'training_loss').
    """
    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(10, 4), constrained_layout = True)
    
    lowerbounds = np.load(os.path.join(RESULTS_DIR, "..", "Data", "Synthetic", "Model_1", f"lowerbounds.npy"))
    
    if setting is not None:
        minValue = lowerbounds[len(setting) - 1]
    else:
        minValue = 0
    
    for model, metrics in data.items():
        if metric not in metrics:
            print(f"Metric '{metric}' not found for model '{model}'. Skipping...")
            continue
        
        # Extract the metric array (n x m)
        values = metrics[metric][:, from_EPOCH:].flatten()
        
        # Forward fill missing values
        mask = np.isnan(values)
        idx = np.where(~mask,np.arange(mask.shape[0]),0)
        np.maximum.accumulate(idx,out=idx)
        values[mask] = values[idx[mask]]

        epochs = np.arange(from_EPOCH, values.shape[0] + from_EPOCH)  # Epoch indices
        values = values - minValue

        plt.plot(epochs, values, label=f"{model}", linewidth=2)
        
    
    # Customize plot
    plt.title(f"{metric.replace('_', ' ').capitalize()} Across Models", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    nticks = 10
    plt.xticks(epochs[::len(epochs)//nticks])
    plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=14)
    plt.yscale(scale)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda t: sort_key(t[0]))
    labels, handles = zip(*sorted_pairs)
    plt.legend(handles, labels, fontsize=12)
    
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    fig.savefig(os.path.join(RESULTS_DIR, fig_dir, f"metrics_{project_name}_{metric}_{scale}_from_{from_EPOCH}.pdf"), transparent=True)
    # plt.show()

def create_plots(project_name, run_names = None, create_contours = True):
    project_data = extract_project_data(project_name)
    # dataset         = get_observations(project_data=project_data, run_name="MGST_CGST_PNN")
    # print(np.array([y[1].cpu() for x,y in dataset]).shape)
    
    # For Real Data
    setting = None
    
    if "_I" in project_name:
        setting = project_name.split("_")[-1]
        fig_dir = os.path.join(RESULTS_DIR, "Figures", f"Setting_{setting}")
    else:
        fig_dir = os.path.join(RESULTS_DIR, "Figures", project_name)

    if create_contours:
        run_df = dict()

        if run_names is None:
            run_names = project_data.keys()

        for run_name in run_names:
            print(f"Adding {run_name}")
            mu_arr, cov_arr = get_outputs(project_data=project_data, run_name=run_name)
            run_df[run_name] = dict()
            run_df[run_name]["mu_arr"] = mu_arr
            run_df[run_name]["cov_arr"] = cov_arr

        dataset         = get_observations(project_data=project_data, run_name=run_name, input_type = "parameters")
        in_channels = dataset.get_in_channels()
        domain_shape = dataset.get_domain_shape()

    
        # Obtain the Mean and Cov parameters
        true_values     = np.array([y[0].cpu() for x,y in dataset]).transpose(1, 2, 3, 0) + BASE_TEMP
        true_values_cov = np.array([y[1].cpu() for x,y in dataset])
        n = in_channels * torch.prod(domain_shape[:-1])
        true_values_cov = true_values_cov.reshape(true_values_cov.shape[0], n, n)
        idxs = torch.arange(n)
        true_values_cov = true_values_cov[:, idxs, idxs].reshape(true_values_cov.shape[0], in_channels, *domain_shape[:-1])
        true_values_cov = true_values_cov.transpose(1, 2, 3, 0)
        true_values_cov = np.sqrt(true_values_cov)
        
        # Obtain a samples from the true distribution
        dataset                     = get_observations(project_data=project_data, run_name=run_name, input_type = "data")
        true_values_sample          = np.array([y.cpu() for x,y in dataset]).transpose(1, 2, 3, 0) + BASE_TEMP
        vmins, vmaxs                = get_vmins_and_vmaxs(true_values, 0.2)
        vmins_sample, vmaxs_sample  = get_vmins_and_vmaxs(true_values_sample, 0.2)

        if true_values_cov is not None:
            vmins_cov, vmaxs_cov    = get_vmins_and_vmaxs(true_values_cov, 0.2)


        setting = list(project_data.values())[0]["config"]["data"]["setting"]
        fig_dir = os.path.join(RESULTS_DIR, "Figures", f"Setting_{setting}")

        plot_contours(true_values_sample,
                setting = setting, 
                vmins = vmins_sample,
                vmaxs = vmaxs_sample,
                time_offset = dataset.training_window,
                savedir=fig_dir,
                save_name = "true_values_sample",
                model = "Ground Truth (Raw)")
        
        plot_contours(true_values_sample,
                setting = setting, 
                vmins = vmins_sample,
                vmaxs = vmaxs_sample,
                time_offset = dataset.training_window,
                savedir=fig_dir,
                save_name = "true_values_sample",
                model = None,
                saveFull=True,
                num_time_slices=7)

        plot_contours(true_values,
                setting = setting, 
                vmins = vmins,
                vmaxs = vmaxs,
                time_offset = dataset.training_window,
                savedir=fig_dir,
                save_name = "true_values_mean",
                model = "Ground Truth (Mean)")
    
        if true_values_cov is not None:
            plot_contours(true_values_cov,
                setting = setting, 
                vmins = vmins_cov,
                vmaxs = vmaxs_cov,
                time_offset = dataset.training_window,
                savedir=fig_dir,
                save_name = "true_values_cov",
                model = "Ground Truth (Cov)")

        # Mean plots
        show_mean = ("mean" in project_name) or ("full" in project_name) or ("vgp" in project_name)

        if show_mean:
            for run_name in run_names:
                mu_arr = run_df[run_name]["mu_arr"]

                if len(mu_arr.shape) == 2:
                    mu_arr = mu_arr.reshape(mu_arr.shape[0], in_channels, *domain_shape[:-1])
                
                mu_arr = mu_arr.transpose(1, 2, 3, 0) + BASE_TEMP

                # plot_contours(mu_arr,
                #         setting = setting, 
                #         vmins = vmins,
                #         vmaxs = vmaxs,
                #         time_offset = dataset.training_window,
                #         savedir = fig_dir,
                #         save_name = f"model_{run_name}_mean",
                #         model = run_name.split("_")[0])
                
                # plot_contours(np.abs(mu_arr - true_values),
                # setting = setting, 
                # time_offset = dataset.training_window,
                # savedir = fig_dir,
                # save_name = f"model_{run_name}_mean_error",
                # model = run_name.split("_")[0])
                
                plot_contours(np.abs(mu_arr - true_values) / true_values * 100,
                setting = setting, 
                time_offset = dataset.training_window,
                savedir = fig_dir,
                save_name = f"model_{run_name}_mean_rel_error",
                model = run_name.split("_")[0],
                is_rel = True)

        # Variance plots
        show_variance = ("cov" in project_name) or ("full" in project_name) or ("vgp" in project_name)
        if show_variance:
            for run_name in run_names:
                cov_arr = run_df[run_name]["cov_arr"]

                if len(cov_arr.shape) == 7:
                    n = in_channels * torch.prod(domain_shape[:-1])
                    cov_arr = cov_arr.reshape(cov_arr.shape[0], n, n)
                    idxs = torch.arange(n)
                    cov_arr = cov_arr[:, idxs, idxs] 
                    cov_arr = np.sqrt(cov_arr)


                if len(cov_arr.shape) == 2:
                    cov_arr = cov_arr.reshape(cov_arr.shape[0], in_channels, *domain_shape[:-1])
                
                cov_arr = cov_arr.transpose(1, 2, 3, 0)

                # plot_contours(cov_arr,
                #         setting = setting, 
                #         time_offset = dataset.training_window,
                #         savedir = fig_dir,
                #         save_name = f"model_{run_name}_cov",
                #         model = run_name.split("_")[0])
                
                plot_contours(np.abs(cov_arr - true_values_cov),
                setting = setting, 
                time_offset = dataset.training_window,
                savedir = fig_dir,
                save_name = f"model_{run_name}_cov_error",
                model = run_name.split("_")[0])
                
                plot_contours(np.abs(cov_arr - true_values_cov) / true_values_cov * 100,
                setting = setting, 
                time_offset = dataset.training_window,
                savedir = fig_dir,
                save_name = f"model_{run_name}_cov_rel_error",
                model = run_name.split("_")[0],
                is_rel = True)
            
    plot_metric(project_name, project_data, setting, scale = "log", fig_dir = fig_dir)
    plot_metric(project_name, project_data, setting, scale = "log", fig_dir = fig_dir, from_EPOCH = 30)
    plot_metric(project_name, project_data, setting, metric = "train_loss", scale = "log", fig_dir = fig_dir)
    
    # if not create_contours:
    #     plot_metric(project_name, project_data, setting, metric = "kis_mae", fig_dir = fig_dir)
    
if __name__ == "__main__":

        # Setting I
        # create_plots("synthetic_mean_I")
        # create_plots("synthetic_cov_I")
        # create_plots("synthetic_full_I")
        create_plots("synthetic_vgp_I")
        # create_plots("synthetic_vgp_I", create_contours = False)

        # Setting II
        # create_plots("synthetic_mean_II")
        # create_plots("synthetic_cov_II")
        # create_plots("synthetic_full_II")
        create_plots("synthetic_vgp_II")
        # create_plots("synthetic_vgp_II", create_contours = False)

        # Setting III
        # create_plots("synthetic_mean_III")
        # create_plots("synthetic_cov_III")
        # create_plots("synthetic_full_III")
        create_plots("synthetic_vgp_III")
        # create_plots("synthetic_vgp_III", create_contours = False)
        
        # create_plots("Real_1", create_contours = False)
        # create_plots("Real", create_contours = False)
        # create_plots("InferenceSpeed_II_100", create_contours = False)
        # create_plots("InferenceSpeed_II_EGP_100", create_contours = False)



