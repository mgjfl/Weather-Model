# Hybrid Approaches to Weather Forecasting: Integrating Statistical Models and Neural Operators

This repository provides a pipeline for combining machine learning-based forecasting with statistical postprocessing techniques. It integrates deep learning models with multivariate statistical correction using copulas, EMOS, and ensemble verification methods. The repository is organized into functional components structured across synthetic experiments and real-world ensemble forecast data. Further documentation for the python modules can be found [here](https://mgjfl.github.io/Weather-Model/).



## Data
This directory contains both raw and processed data, model output, and postprocessing utilities. Key subdirectories:

- **`Synthetic/`**: Tools and scripts for generating and analyzing synthetic datasets using Python and R. Includes:
  - Dataset containing AI-generated ensemble forecasts (`AI_Ensemble/`)
  - Dataset containing toy model samples (`Model_1/`)
  - Data generation utilities (`generate_synthetic_data.py`, `synthetic_data_util.py`)

- **`Real/`**: Observational and model-based data from multiple sources:
  - `ERA5/`, `IFS/`, `LAEF/` contain station-level meteorological data
  - `AI_Ensemble/` includes AI-generated ensemble forecasts

- **`MVPP/`, `UVPP/`, `RVineMatrix/`, `SimilarityMatrix/`, `TestStatistic/`**: 
  - RData files storing intermediate and final outputs (transformed data, score lists, copula matrices, similarity metrics, and DM test statistics) used in verification and comparison.

### `Postprocessing/`

- `MVPP/`: MultiVariate PostPostprocessing (MVPP)
  - `ensfc.R`: Ensemble forecasting utilities for multivariate settings.
  - `mvpp_methods.R`: Core implementation of MVPP  methods.

- `mvpp_central.R`, `mvpp_central_AI_ensemble.R`, `mvpp_central_parallel.R`: Scripts to apply MVPP and scoring, with variants for centralized settings, parallel computation, or specific AI ensembles.


#### Preprocessing and Data Management

- `load_data.R`: Loads and structures ensemble or observational data for further analysis.

- `uvpp_util.R`: Utilities for Univariate Verification of Probabilistic Predictions (UVPP), complementing MVPP.

- `sim_util.R`: Utility functions for creating similarity matrices.

- `recompute_all.R`: Central computation function, doing UVPP, formatting of data, constructing similarity matrices and computing DM scores.

#### Evaluation and Scoring

- `compute_DM_scores.R`: Computes Diebold-Mariano (DM) scores to statistically compare forecasting models.

- `compute_mcs.R`: Computes the Model Confidence Set (MCS), a statistical method for selecting superior models.

- `scores.R`: Computes general forecast scores.


## Models
Collection of Python modules implementing AI-based forecasting models and related utilities.

- **Core architectures**:
  - `AFNO.py`: Adaptive Fourier Neural Operator
  - `FNN.py`: Feedforward Neural Network
  - `GP.py`: Gaussian Process model
  - `GST.py`: Gaussian SpatioTemporal Model
  - `VGP_optimizer.py`: Variational Gaussian Process optimizer

- **Wrapper modules**:
  - `neural_models.py`: Wrapper for applying models to synthetic data.
  - `real_data_model.py`: Wrapper for applying models to real-world data.

## Training
Utilities related to training AI models.

- `train_utils.py`: Core functions for model training, checkpointing, and evaluation

## Results

This directory contains all experiment outputs, configurations, figures, models, and postprocessing utilities.

- `Configurations/`: YAML configuration files and folders for different experiment setups, including real-world and synthetic data scenarios.
- `cov_accuracy.py`, `parameter_tuning.py`, `plot_all_results.py`: Python scripts for accuracy analysis, hyperparameter tuning, and aggregated plotting.
- `extract_from_configuration.py`, `run_configuration.py`, `save_results.py`: Core utilities to run and log experiments based on configurations.
- `results_util.py`, `parser.py`: Supporting modules for experiment execution and results parsing.

### `Postprocessing/`

#### Plotting
- `create_all_plots.R`: Generate full sets of plots
- `dm_plots.R`, `threefold_dm_plot.R`: Visualize Diebold-Mariano test results.
- `mae_plots.R`, `score_boxplots.R`: MAE and score distribution plots.
- `similarity_dm_plots.R`, `stat_plots.R`: Structural and statistical summaries.
- `univariate_pit.R`: Marginal PIT analysis.
- `multivariate_pit.R`, `multivariate_pit_util.R`: Joint PIT analysis and utilities.

#### Scoring
- `create_score_tables.R`: Generate tables of forecast scores.
- `skill_scores.R`: Compute skill scores relative to baselines.
---
