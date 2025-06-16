import optuna
import torch
from torch.utils.data import DataLoader 
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

def objective(trial):
    # Suggest hyperparameters
    num_latents = trial.suggest_categorical("num_latents", [4, 6])
    m_inducing_points = trial.suggest_categorical("m_inducing_points", [8, 16, 32])
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32])
    n_modes = trial.suggest_categorical("n_modes", [4, 6, 12])
    n_layers = trial.suggest_categorical("n_layers", [4, 6, 8])

    device = get_device()
    EPOCHS = 10
    BATCH_SIZE = 64
 
    data_config = {
        "type" : "real",
        "training_window" : 3
    }
    model_config = {
        "arch" : "StationVGP",
        "dropout_rate" : None,
        "parameters" : {
            "num_latents" : num_latents,
            "m_inducing_points" : m_inducing_points,
            "hidden_channels" : hidden_channels,
            "n_modes" : [n_modes, n_modes, n_modes],
            "n_layers" : n_layers
        },
        "mean_class" : "FNO",
        "cov_class" : None
    }
    
    training_config = {
        "batch_size": BATCH_SIZE,
        "early_stopping": {
            "enabled": "true",
            "patience": 100
        },
        "epochs": EPOCHS,
        "learning_rate": 0.001,
        "loss_function": "VariationalELBOLoss",
        "regularization_weight": 0.001,
        "grad_regularization": 0,
        "optimizer": "Adam",
        "scheduler": {
            "parameters": {
                "gamma": 0.5,
                "step_size": 20
            },
            "type": "StepLR"
        }
    }

        
    # Generates independent datasets for each run
    dataset = get_dataset(data_config=data_config, device=device)
        
    # Create the model
    model = create_model(data_config=data_config, model_config=model_config, dataset=dataset, device=device)
    model = model.to(device)
    
    # Define training components
    loss_fn, optimizer, scheduler = get_training_components(model=model, training_config=training_config, data_config=data_config)

        # Train and get validation loss
    total_val_loss = train_and_evaluate_model(
        model,
        dataset,
        batch_size=BATCH_SIZE,
        loss_fn=loss_fn,
        optimizer=optimizer,
        EPOCHS=EPOCHS,
        patience=5,
        scheduler=scheduler,
        data_saver=None,
        verbose=False
    )
    
    return total_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "optuna_results.txt"), "w") as f:
        def logging_callback(study, trial):
            f.write(f"Trial {trial.number}: val_loss={trial.value}, params={trial.params}\n")

        study.optimize(objective, n_trials=10, n_jobs=1, callbacks=[logging_callback], gc_after_trial=True)
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "optuna_results_best.txt")
              , "w") as f:
        f.write(f"Best hyperparameters:\n{study.best_params}\n")
        f.write(f"Best value: {study.best_value}\n")

