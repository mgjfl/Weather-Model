from pathlib import Path
import numpy as np
import os
import json
import yaml
import numpy as np
from collections import defaultdict
from extract_from_configuration import get_dataset, get_device, str_to_class
import pandas as pd
import torch
from Models import *
import re
from torch.utils.data import DataLoader
from Models.GP import GP  # Ensure correct import
import re
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_weather_dir():
    return Path(__file__).parent.parent

def extend_patience_value(array, length):
    last_val = array[-1]
    reps = np.tile([last_val], length - len(array))
    return np.hstack([array, reps])

def extract_project_data(project_name):
    
    print(project_name)
    
    if project_name == "Real":
        all_dirs = os.listdir(os.path.join(RESULTS_DIR, "Runs"))
        real_dirs = [dir for dir in all_dirs if "Real" in dir and "Real" != dir]
        data = defaultdict(dict)
        
        for dir in real_dirs:
            data |= extract_project_data(dir)
            
        return data
            
    dir = os.path.join(RESULTS_DIR, "Runs", project_name)
    data = defaultdict(dict)

    for model in os.listdir(dir):
        model_path = os.path.join(dir, model)
        
        if "Real" in dir:
            model += re.sub(r".*Real_", "-", dir)
        runs = [x for x in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, x))]
        
        with open(os.path.join(model_path, "config.yaml")) as f:
            config = yaml.safe_load(f)
            data[model]["config"] = config
            n = config["training"]["epochs"]
            metrics = config["output"]["metrics"]
            
            if "Real" in dir:
                metrics.append("kis_mae")
            
        for metric in metrics:
            data[model][metric] = np.empty(shape = (len(runs), n))
            
        data[model]["model_name"] = list(range(len(runs)))
        
        for (run_id, run) in enumerate(runs):
            with open(os.path.join(model_path, run, "metrics.json"), 'r') as f:
                run_metrics = json.load(f)
                
            for metric in metrics:
                data[model][metric][run_id, :] = extend_patience_value(run_metrics[metric], n)
                
            data[model]["model_name"][run_id] = os.path.join(model_path, run, "trained_model.pt")
            
    return data

def get_observations(project_data, run_name, input_type = None):

    # Extract the dataset
    device = get_device()
    data_config = project_data[run_name]["config"]["data"]

    if input_type is not None:
        data_config["input_type"] = input_type
        
    dataset = get_dataset(data_config=data_config, device=device)

    return dataset
    
def get_predictions(project_data, run_name, dataset):
    
    # Get the trained model
    device = get_device()
    model_name = project_data[run_name]["model_name"][0]
    model = torch.load(model_name, weights_only=False)
    model.eval();
    model.to(device)

    mu_vals     = []
    sigma_vals  = []

    is_GP_model     = ".GP" in str(type(model))
    is_GP_Wrapper   = ".GPWrapper" in str(type(model))
    is_StationVGP   = ".StationVGP" in str(type(model))

    # input_type = project_data[run_name]["config"]["data"]["input_type"]


    if is_GP_model and not is_GP_model:
        model.likelihood.eval()
        data_loader = dataset.get_test_loader(1)
    elif is_GP_Wrapper:
        model.likelihood.eval()
        data_loader = DataLoader(dataset, batch_size=128)
    elif is_StationVGP:
        model.likelihood.eval()
        data_loader = dataset.get_test_loader(batch_size=64)
    else:
        data_loader = DataLoader(dataset, batch_size=128)

    
    # Extract the predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x, y in data_loader:
            if is_GP_model and not is_GP_Wrapper:
                x = x.unsqueeze(0)

            # print(f"{x[0, 0, 0, 0]=}")

            # Model output
            pred          = model(x)

            # print(f"{pred=}")
            # print(f"{pred.mean=}")

            if is_GP_model or is_GP_Wrapper or is_StationVGP:
                mus = pred.mean
                sigmas = pred.variance
            else:
                mus, sigmas = pred

            mus           = mus.detach()
            sigmas        = sigmas.detach()

            # Save values
            mu_vals += [mu.cpu() for mu in mus]
            sigma_vals += [sigma.cpu() for sigma in sigmas]

    return mu_vals, sigma_vals

def postprocess_observations(dataset):
    
    data_x = np.array([np.array(x[0].cpu()) for x in dataset])
    data_y = np.array([np.array(x[1].cpu()) for x in dataset])

    return data_x, data_y

def get_covariance(cov, w1, d1, w2, d2, w):
    id1 = w1 + d1 * w
    id2 = w2 + d2 * w
    return cov[id1, id2]
    

def get_outputs(project_data, run_name):

    # Extract required values
    dataset = get_observations(project_data=project_data, run_name=run_name)
    mu_vals, sigma_vals = get_predictions(project_data=project_data, run_name=run_name, dataset=dataset)

    # Postprocess values
    mu_arr, cov_arr = np.array(mu_vals), np.array(sigma_vals)

    return mu_arr, cov_arr
    
def plot_2d_gaussian_contour(mean, covariance, levels=10, grid_size=100, ax=None):
    """
    Plots a contour for a 2D Gaussian distribution.

    Parameters:
        mean (array-like): The mean of the Gaussian [mu_x, mu_y].
        covariance (array-like): The 2x2 covariance matrix of the Gaussian.
        levels (int): Number of contour levels to display (default: 10).
        grid_size (int): Resolution of the grid for plotting (default: 100).
        ax (matplotlib.axes._axes.Axes, optional): Existing matplotlib Axes to plot on.
    """
    # Create a grid of x and y values
    x = np.linspace(mean[0] - 3 * np.sqrt(covariance[0, 0]), mean[0] + 3 * np.sqrt(covariance[0, 0]), grid_size)
    y = np.linspace(mean[1] - 3 * np.sqrt(covariance[1, 1]), mean[1] + 3 * np.sqrt(covariance[1, 1]), grid_size)
    X, Y = np.meshgrid(x, y)

    # Create a multivariate normal distribution
    rv = multivariate_normal(mean, covariance)

    # Evaluate the probability density function on the grid
    Z = rv.pdf(np.dstack((X, Y)))

    # Plot the contour
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis')
    ax.set_title('2D Gaussian Contour Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.axis('equal')
    ax.grid(True)
    plt.colorbar(contour, ax=ax, label='PDF Value')
    plt.show()

def plot_3d_gaussian_evolution(means, covariances, grid_size=100, time_steps=None, n_std=1):
    """
    Plots a 3D visualization of the evolution of 2D Gaussian contours over time.

    Parameters:
        means (list): List of mean vectors [mu_x, mu_y] for each time step.
        covariances (list): List of 2x2 covariance matrices for each time step.
        grid_size (int): Resolution of the grid for plotting (default: 100).
        time_steps (list, optional): List of time step values. Defaults to range(len(means)).
    """
    # Validate inputs
    if len(means) != len(covariances):
        raise ValueError("Means and covariances must have the same length.")
    
    if time_steps is None:
        time_steps = range(len(means))
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for t in time_steps:
        mean = means[t]
        covariance = covariances[t]
        
        # # Create a grid of x and y values
        # x = np.linspace(mean[0] - 3 * np.sqrt(covariance[0, 0]), mean[0] + 3 * np.sqrt(covariance[0, 0]), grid_size)
        # y = np.linspace(mean[1] - 3 * np.sqrt(covariance[1, 1]), mean[1] + 3 * np.sqrt(covariance[1, 1]), grid_size)
        # X, Y = np.meshgrid(x, y)
        
        # # Create a multivariate normal distribution
        # rv = multivariate_normal(mean, covariance)
        
        # # Evaluate the probability density function on the grid
        # Z = rv.pdf(np.dstack((X, Y)))
        
        # # Normalize Z for better visualization
        # Z /= Z.max()
        
        # # Add the contour at the current time step
        # ax.contour3D(X, Y, Z, cmap='viridis', offset = t, zdir = 'z', alpha=0.7)
        
        # With elipses
                # Calculate the eigenvalues and eigenvectors for the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = eigenvalues.argsort()[::-1]  # Sort eigenvalues descending
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Get the angle of the ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # Width and height of the ellipse (2 * n_std * sqrt(eigenvalue))
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        # Add the ellipse to the plot
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle, alpha=0.5, color='blue'
        )
        ax.add_patch(ellipse)
        ax.plot([mean[0]], [mean[1]], [t], 'ro')  # Plot the mean point at this time
        art3d.pathpatch_2d_to_3d(ellipse, z=t, zdir="z")


    # Extract the mean points for connecting ellipses
    mean_x = [means[t][0] for t in time_steps]
    mean_y = [means[t][1] for t in time_steps]
    mean_t = list(time_steps)

    # Plot the connecting lines between the ellipses
    ax.plot(mean_x, mean_y, mean_t, color='black', linestyle='--', linewidth=3, label='Mean trajectory')


    ax.set_title('3D Evolution of 2D Gaussian Contours Over Time')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Time / PDF Level')
    
    ax.set_zlim(min(time_steps), max(time_steps))
    ax.legend()
    
    return fig, ax

def plot_observations_vs_predictions(df):
    fig_scale = 4
    d = df["Coordinate"].unique().shape[0]
    w = sum(re.search(r"^Var \d$", col) is not None for col in df.columns)
    fig, axes = plt.subplots(d,w,figsize=(w * fig_scale, d * fig_scale))

    font = {'size'   : 18}
    plt.rc('font', **font)

    for i, ax in enumerate(np.ravel(axes)):
        dd = i // w
        ww = i % w
        var_name = f"Var {ww + 1}"
        coord = dd + 1
        
        

        plot_df = df[df["Coordinate"] == coord]
        ax.scatter(x = plot_df["Days"], y = plot_df[var_name], label = "Data")
        ax.plot( plot_df["Days"], plot_df[var_name + " (mean)"], label=f"Prediction (mean)", linewidth=2, c = 'darkorange')
        ax.fill_between(x = plot_df["Days"], 
                        y1 = plot_df[var_name + " (mean)"] - plot_df[var_name + " (std)"], 
                        y2 = plot_df[var_name + " (mean)"] + plot_df[var_name + " (std)"], 
                        alpha=0.7, label=f"Prediction (std)")

        if ww == 0:
            ax.set_ylabel("Value")

        if dd == d - 1:
            ax.set_xlabel("Day")
        # ax.errorbar(, yerr = plot_df["Var 1 (std)"], label = "Prediction", c = "red")
        # ax.legend()
        # break

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Observations versus predictions\n\n\n\n")
    fig.legend(handles, labels, ncol = 5, loc = 'upper center', bbox_to_anchor = (0.5, 0.97))
    fig.tight_layout()
    return fig