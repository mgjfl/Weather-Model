import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch
import sys
sys.path.append(os.getcwd())
from Models import *
from Results.results_util import *
RESULTS_DIR = os.path.join(os.getcwd(), "Results")

device = "cuda"

def get_single_point(project_data, run_name, is_EGP):
    print(f"Running {run_name}")
        
    # Change the dataset such that input = random observations and output = covariance matrix
    project_data[run_name]["config"]["data"]["input_type"] = "evaluation"
    dataset = get_observations(project_data=project_data, run_name=run_name)
    
    # Run identifier
    xval = int(run_name.split("_")[-1])
    
    # Get the trained model
    device = get_device()
    model_name = project_data[run_name]["model_name"][0]
    model = torch.load(model_name, weights_only=False)
    model.eval();
    model.to(device)

    print(str(type(model)))
    is_GP_model     = ".EGP" in str(type(model))
    is_GP_Wrapper   = ".GPWrapper" in str(type(model))
    is_StationVGP   = ".StationVGP" in str(type(model))

    # Get the data
    if is_EGP:
        data_loader = dataset.get_test_loader(0)
    else:
        data_loader = dataset.get_test_loader(1)

    # Compute covariance accuracy
    rmse_norms = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i, (x, y) in enumerate(data_loader):
            print(f"{i + 1:>3} / {len(data_loader)}")
            if is_GP_model and not is_GP_Wrapper:
                x = x.unsqueeze(0)

            # Model output
            pred        = model(x)
            
            pred_cov    = pred.covariance_matrix
            true_cov    = y[1].reshape(pred_cov.shape)
            
            rmse_norm   = torch.sqrt((pred_cov - true_cov).pow(2).mean())
            rmse_norms.append(rmse_norm)
            
    print()
            
    # Metric of interest
    avg_rmse_norm = torch.tensor(rmse_norms).mean()
    
    torch.cuda.empty_cache()
    
    return (xval, avg_rmse_norm)

def get_data(is_EGP = True):

    # Obtain project specific information
    if is_EGP:
        project_name = "InferenceSpeed_II_EGP_100"
    else:
        project_name = "InferenceSpeed_II_100"
        
    project_data = extract_project_data(project_name)

    # To store the data
    xvals           = []
    avg_rmse_norms  = []

    for run_name in project_data.keys():
        xval, avg_rmse_norm = get_single_point(project_data, run_name, is_EGP)
        xvals.append(xval)
        avg_rmse_norms.append(avg_rmse_norm)
        
        torch.cuda.empty_cache()
    
    return (xvals, avg_rmse_norms)


if __name__ == "__main__":
    
    #########
    ## VGP ##
    #########
    
    xvals_vgp, avg_rmse_norms_vgp = get_data(False)
    xvals_vgp, avg_rmse_norms_vgp = torch.tensor(xvals_vgp), torch.tensor(avg_rmse_norms_vgp)
    print(xvals_vgp)
    print()
    print(avg_rmse_norms_vgp)
    
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(xvals_vgp, avg_rmse_norms_vgp, 'o')
    ax.set_xlabel("Inducing points")
    ax.set_ylabel("RMSE Covariance Matrix")
    
    fig.savefig(os.path.join(RESULTS_DIR, "Figures", "vgp_plot.png"))
    
    #########
    ## EGP ##
    #########
    
    xvals_egp, avg_rmse_norms_egp = get_data(True)
    xvals_egp, avg_rmse_norms_egp = torch.tensor(xvals_egp), torch.tensor(avg_rmse_norms_egp)
    print(xvals_egp)
    print()
    print(avg_rmse_norms_egp)
    
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(xvals_egp  * 0.005, avg_rmse_norms_egp, 'o')
    ax.set_xlabel("% of training set")
    ax.set_ylabel("RMSE Covariance Matrix")
    
    fig.savefig(os.path.join(RESULTS_DIR, "Figures", "egp_plot.png"))
    
    ##############
    ## Combined ##
    ##############
    
    fig, ax = plt.subplots(figsize = (6, 3), constrained_layout = True)

    # Plot EGP on the main axes
    ax.plot(xvals_vgp, avg_rmse_norms_vgp, 'o', label="VGP", color="C0")
    ax.set_xlabel("Inducing points", color="C0")
    ax.set_ylabel("RMSE Covariance Matrix")
    ax.tick_params(axis='x', colors="C0")

    # Create a second X axis sharing the same Y axis
    ax2 = ax.twiny()
    ax2.plot(xvals_egp  * 0.005, avg_rmse_norms_egp, 'o', label="EGP", color="C1")
    ax2.set_xlabel("% of training set", color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax2.tick_params(axis='x', colors="C1")

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc="center left", bbox_to_anchor=(0.9, 0.5))


    fig.savefig(os.path.join(RESULTS_DIR, "Figures", "combined_plot.png"))
