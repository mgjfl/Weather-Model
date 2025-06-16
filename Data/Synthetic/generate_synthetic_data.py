from synthetic_data_util import *
import os
import numpy as np

def compute_and_sample_values():
    """
    Computes parameters mu and Sigma for the synthetic model at three settings. 
    Saves these parameters and samples from the probabilistic distribution.
    """
    DATA_DIR = os.getcwd()

    # Define the grid
    V       = 12                # Latitude  : Must be even
    H       = 18                # Longitude : Must be even
    ncycles = 10                # Number of years
    T       = ncycles * 100     # Time steps
    W       = 2                 # Weather variables
    N       = W * V * H 
    RH      = 40

    # Locations to compute solution
    long_range  = np.linspace(-1, 1, H)
    lat_range   = np.linspace(-1, 1, V)
    time_range  = np.linspace(0, 2 * np.pi * ncycles, T)
    X = np.meshgrid(long_range, lat_range, time_range)
    Z = np.stack(X, axis = 3)
    Z = Z.reshape(-1, Z.shape[-1])

    # Function that computes the mean
    mf = lambda alpha, beta, gamma, delta: compute_mean(alpha, beta, gamma, delta, RH, X, Z)
        
    # Function that computes the covariance
    Sf = lambda zeta, eps, h, c: compute_covariance(long_range, lat_range, time_range, W, V, H, T, zeta, eps, h, c)

    # Parameterizations
    param_cases = {
        "case_I" : {
            "alpha"   : 1,
            "beta"    : 5,
            "gamma"   : 1,
            "delta"   : 1,
            "eps"     : 2.5,
            "h"       : 1,
            "c"       : 0.5,
            "zeta"    : 1e-2,
        },
        "case_II" : {
            "alpha"   : 1,
            "beta"    : 5,
            "gamma"   : 1,
            "delta"   : 1,
            "eps"     : 2.5,
            "h"       : 1,
            "c"       : 0.5,
            "zeta"    : 1,
        },
        "case_III" : {
            "alpha"   : 1,
            "beta"    : 5,
            "gamma"   : 1,
            "delta"   : 1,
            "eps"     : 2.5,
            "h"       : 1,
            "c"       : 0.5,
            "zeta"    : 1e2,
        },
    }

    for setting in ["case_I", "case_II", "case_III"]:
        # Mean
        mu = mf(param_cases[setting]["alpha"],
                param_cases[setting]["beta"],
                param_cases[setting]["gamma"],
                param_cases[setting]["delta"])
        
        # Covariance
        S_full = Sf(param_cases[setting]["zeta"],
                    param_cases[setting]["eps"],
                    param_cases[setting]["h"],
                    param_cases[setting]["c"])
        
        # Output
        output = compute_output(mu, S_full)   
        
        # Save the results
        np.save(os.path.join(DATA_DIR, "Model_1", f"mu_{setting}"), mu)
        np.save(os.path.join(DATA_DIR, "Model_1", f"cov_{setting}"), S_full)
        np.save(os.path.join(DATA_DIR, "Model_1", f"data_{setting}"), output)

if __name__ == "__main__":
    compute_and_sample_values()


