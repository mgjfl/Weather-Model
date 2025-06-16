import os
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy import special
from itertools import product 
from numpy.random import multivariate_normal
from IPython.display import display, Latex
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import torch
from math import sqrt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

def integrand(tau, t, alpha, x_vec):
    """
    Defines the integrand of the temperature solution.
    """
    if (tau == t): 

        if (tau == 0 and np.linalg.norm(x_vec) == 0.0):
            # Lim sin(x)/x -> 1 as x -> 0
            return 1.0 / (4 * np.pi * alpha)
        return 0.0
    
    term_1 = (1.0 / (4 * np.pi * alpha * (t - tau))) 
    if (np.linalg.norm(x_vec) == 0.0):
        term_2 = 1.0
    else:
        term_2 = np.exp(- np.linalg.norm(x_vec) ** 2 / (4 * alpha * (t - tau)))
    term_3 = np.abs(np.sin(tau))
    return term_1 * term_2 * term_3

def temp_p(x_vec, t, alpha, beta):
    """
    Computes the particular temperature solution that is caused by the forcing term beta * sin(t) * delta(x). 
    """
    
    result = beta * quad(integrand, 0, t, args = (t, alpha, x_vec))[0]      

    return result

def temp_i(x_vec, gamma, delta):
    """ 
    Provides the initial temperature profile.
    """
    x_norm_2 = np.linalg.norm(x_vec) ** 2
    return gamma * np.exp(- x_norm_2 / (delta ** 2))


def temp_h(x_vec, t, alpha, gamma, delta):
    """
    Computes the solution to the homogeneous heat equation given the Gaussian initial temperature profile.
    """
    term_1 = (delta ** 2 / (4 * alpha * t + delta ** 2))
    term_2 = np.exp(- np.linalg.norm(x_vec) ** 2 / (4 * alpha * t + delta ** 2))
    return gamma * term_1 * term_2

def temp(z, alpha, beta, gamma, delta):
    """
    Computes the solution to the inhomogeneous forced heat equation with Gaussian IC.
    """
    x_vec   = z[:2]
    t       = z[2]
    return temp_p(x_vec, t, alpha, beta) + temp_h(x_vec, t, alpha, gamma, delta)

def sigma_w(d1, d2, c):
    """
    The weather type contribution to the covariance matrix.
    """
    if (d1 == d2):
        return 1.0
    else:
        return c
    
def sigma_t(t):
    """
    The time contribution to the covariance matrix.
    """
    return 1 + np.abs(np.sin(t))
    # return np.exp(-np.abs(t1 - t2))

def sigma_x(x1, x2, eps, h):
    """
    The spatial contribution to the covariance matrix.
    """
    if  torch.linalg.norm(x1 - x2) == 0.0:
        return 1.0
    term1   = 1.0 / (2 ** (eps - 1) * torch.exp(torch.lgamma(eps)))
    norm    = torch.linalg.norm(x1 - x2) / h
    arg     = torch.sqrt(2 * eps) * norm
    term2   = arg ** eps

    if (eps == 2.5):
        term3 = (1 + sqrt(5) * norm + 5 / 3 * (norm ** 2)) * torch.exp(-sqrt(5) * norm)
    else:
        term3   = special.kv(eps, arg)

    return term1 * term2 * term3

def get_vmins_and_vmaxs(data, p):

    W, V, H, T  = data.shape


    # Determine the global min and max for color normalization
    vmins   = data.reshape(W, -1).min(axis = 1)
    vmaxs   = data.reshape(W, -1).max(axis = 1)

    # Extend by p % of the range on both sides
    diff = (vmaxs - vmins) * p

    return vmins - diff, vmaxs + diff

def plot_contours(data, 
                  num_time_slices = 3, 
                  ncycles = 10, 
                  vmins = None,
                  vmaxs = None,
                  savedir = None, 
                  setting = None, 
                  save_name = None, 
                  time_offset = 0,
                  model = None,
                  closeOnSave = True,
                  is_rel = False,
                  saveFull = False):
    """
    General-purpose plotting function to plot synthetic data samples at various timesteps with several configurable settings.
    """
    plt.rc('text', usetex=True)\
    
    if saveFull:
        plt.rcParams.update({'font.size': 35})
    else:
        plt.rcParams.update({'font.size': 40})
        

    W, V, H, T  = data.shape

    time_range  = np.linspace(0, 2 * np.pi * ncycles, T)
    
    if saveFull:
        time_indices = np.linspace(0, T-1, num_time_slices, dtype=int)
    else:
        time_indices = np.linspace(500, T-1, num_time_slices, dtype=int)

    # Determine the global min and max for color normalization
    if vmins is None:
        vmins   = data[..., time_indices].reshape(W, -1).min(axis = 1)
    if vmaxs is None:
        vmaxs   = data[..., time_indices].reshape(W, -1).max(axis = 1)
    norms   = [Normalize(vmin=vmin, vmax=vmax) for vmin, vmax in zip(vmins, vmaxs)]
    cmaps   = [plt.colormaps['coolwarm'] for _ in range(W)]
    ims     = [cm.ScalarMappable(norm=norms[w], cmap = cmaps[w]) for w in range(W)]

    # Create a single figure with W rows
    fig, axes = plt.subplots(W, num_time_slices, figsize=(5 + 2.5 * num_time_slices, (3 * W if saveFull else 5 * W)), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.05)
    
    if saveFull and model is not None:
        suptitle = model
        fig.suptitle(suptitle, fontsize=40)

    # Create contour plots
    for w in range(W):

        # Store the first plot object for a shared colorbar
        # contour_plots = []
        
        # Plot all subplots with shared normalization
        for i, t in enumerate(time_indices):
            ax = ax = axes[w, i]
            c = ax.contourf(data[w, :, :, t], levels=100, cmap = cmaps[w], norm=norms[w])  # Use global norm
            c.set_edgecolor("face") # For better pdf output
            # ax.set_title(rf'Time = {t}\newline{{\small\centering($\phi$ = {time_range[t] * 180 / np.pi % 360:.2f}$^\circ$)}}')
            if saveFull:
                ax.set_title(rf'\begin{{tabular}}{{c}}\fontsize{{23}}{{25}}\selectfont{{$(t, \phi)$ = ({t}, {time_range[t] * 180 / np.pi % 360:.0f}$^\circ$)}}\end{{tabular}}')
            else:
                ax.set_title(rf'\begin{{tabular}}{{c}}Time = {t + time_offset} \\\fontsize{{35}}{{40}}\selectfont{{$\phi$ = {time_range[t] * 180 / np.pi % 360:.2f}$^\circ$}}\end{{tabular}}')
            # ax.set_xlabel('H')
            # ax.set_ylabel('V')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add a single colorbar shared across all subplots
        cbar = fig.colorbar(ims[w], ax=axes[w,:].ravel().tolist())
        # Ensure at least 3 ticks
        vmin, vmax = norms[w].vmin, norms[w].vmax
        nticks = 3 #max(3, len(cbar.get_ticks()))  # enforce at least 3
        ticks = np.linspace(vmin, vmax, nticks)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
        
        if is_rel:
            label = r"Error (\%)"
        else:
            label = 'Data Values'
        cbar.set_label(label)

    if savedir is not None:
        fig.savefig(os.path.join(savedir, f"contour_{setting}" + 
                                    (f"_{save_name}" if save_name is not None else "") +
                                    (f"_full" if saveFull else "")+ ".pdf"))
    
    if closeOnSave:
        plt.close()
        
        # plt.show()

def compute_covariance(W, V, H, T, zeta, eps, h, c, ncycles = 10):
    """
    Computes the full covariance matrix at various timesteps for the synthetic data.
    """

    long_range  = np.linspace(-1, 1, H)
    lat_range   = np.linspace(-1, 1, V)
    time_range  = np.linspace(0, 2 * np.pi * ncycles, T)

    # Compute all pairs to construct the covariance matrices
    X_grid = np.meshgrid(long_range, lat_range)
    Z_grid = np.stack(X_grid, axis = 2)
    Z_grid = Z_grid.reshape(-1, Z_grid.shape[-1])

    space_pairs = list(product(Z_grid, repeat=2))
    # time_pairs  = list(product(time_range, repeat=2))
    type_pairs  = list(product(np.arange(W), repeat=2))

    # Compute covariance matrices
    St          = np.array([sigma_t(t) for t in time_range]) # np.array([sigma_t(*tp) for tp in time_pairs]).reshape(T, T)
    Sx          = np.array([sigma_x(*z, eps, h) for z in space_pairs]).reshape(H * V, H * V)
    Sw          = np.array([sigma_w(*tp, c) for tp in type_pairs]).reshape(W, W)
    S_full      = zeta * np.kron(np.kron(Sw, Sx), St).reshape(W, V, H, W, V, H, T)

    return S_full

def compute_mean(alpha, beta, gamma, delta, RH, X, Z):
    """
    Computes the mean at various timesteps for the synthetic data.
    """

    # Function that computes the solution
    Tf = lambda alpha, beta, gamma, delta: np.array([temp(z, alpha, beta, gamma, delta) for z in Z]).reshape(X[0].shape)

    Ta = Tf(alpha, beta, gamma, delta)
    Td = Ta - (100 - RH) / 5
    mu = np.stack([Ta, Td])

    return mu

def compute_output(mu, S_full):
    """
    Computes samples at various timesteps for the synthetic data based on the means and covariance matrices.
    """
    W, V, H, T  = mu.shape
    N           = W * V * H 
    output = np.zeros(shape = (W, V, H, T))

    mu_norm_max = 0.0
    mu_norm_sd = mu.std()
    sigma_norm = 0.0

    avg_mu_sd = 0.0
    avg_sigma = 0.0

    for t in range(T):

        # The distribution parameters
        mu_val      = mu[..., t].reshape(N)
        sigma_val   = S_full[..., t].reshape(N, N)

        # Keep track of max norms
        mu_max      = np.abs(mu_val).max()
        mu_sd       = mu_val.std()
        avg_mu_sd   += mu_sd
        sigma_max   = np.abs(sigma_val).max()
        avg_sigma   += np.sqrt(sigma_val).mean()

        if (mu_max > mu_norm_max):
            mu_norm_max = mu_max

        if (mu_sd < mu_norm_sd):
            mu_norm_sd = mu_sd

        if (sigma_max > sigma_norm):
            sigma_norm = sigma_max

        # A random sample
        output[..., t] = multivariate_normal(mu_val, sigma_val).reshape(W, V, H)

    avg_mu_sd /= T
    avg_sigma /= T

    display(Latex(rf"$||\mu||_\infty \approx $ {mu_norm_max:.2f} $\quad \min_T(\sigma(\mu)) \approx $ {mu_norm_sd:.2f} $\quad \sqrt{{||\Sigma||_\infty}} \approx $ {np.sqrt(sigma_norm):.2f}"))
    display(Latex(rf"$\text{{avg}}_t(\sigma(\mu(x, t))) \approx $ {avg_mu_sd:.2f} $\quad \text{{avg}}_{{x, t}}\sqrt{{\Sigma(x, t)}} \approx $ {avg_sigma:.2f}"))
    return output