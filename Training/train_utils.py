import torch
from torch.utils.data import DataLoader
import numpy as np
from timeit import default_timer as timer
import sys
import os
import pathlib
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)
from Models import *
import gpytorch
from sklearn.metrics import mean_absolute_error

class EarlyStopper:
    """
    Class implementing early stopping of model training.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(
    dataloader, 
    model, 
    loss_fn, 
    optimizer, 
    batch_size, 
    verbose = False,
    max_batches = None):
    """
    Trains the model for one epoch.
    """

    running_loss = 0.0
    
    # To obtain a stacktrace when anomalies are detected
    # NOTE: Huge increase in computation time.
    # torch.autograd.set_detect_anomaly(True)
    
    size = 0

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        
        if max_batches is not None and batch >= max_batches:
            break

        if issubclass(type(model), GP):
            X = X.squeeze(0)
            x_size = 1
        else:
            x_size = X.size(0)
            
        size += x_size
        #     pred = model(X)
        # else:
        #     loss = model.compute_loss(loss_fn, pred, y)
        
        
        pred = model(X)
        loss = loss_fn(pred, y, isTraining = True)
        
        if verbose:
            print(f"Batch {batch:>3} : loss = {loss.item():>7f}")

        # Backpropagation
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        if optimizer is not None:
            optimizer.step()

        # Update loss
        running_loss += loss.item() * x_size
            
    # Compute the training loss
    train_loss = running_loss / size

    return train_loss


def val_one_epoch(dataloader, model, loss_fn, verbose = False, max_batches = None):
    """
    Evaluates the model on the validation set.
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    if issubclass(type(model), GP) or issubclass(type(model), StationVGP):
        model.likelihood.eval()

    size = 0
        
    if issubclass(type(model), StationVGP):
        interp_mae = np.longdouble(0.0)
        kis_mae = np.longdouble(0.0)
        kis_mae0 = np.longdouble(0.0)
        kis_mae1 = np.longdouble(0.0)

    test_loss = np.longdouble(0.0)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i, (X, y) in enumerate(dataloader):  
            if max_batches is not None and i >= max_batches:
                break
            if issubclass(type(model), GP):
                # X = X[0]
                # y = y[0]
                # X = X.reshape(-1)
                X = X.unsqueeze(0)
                x_size = 1
            else:
                x_size = X.size(0)
                
            size += x_size
           

            #     test_loss += np.longdouble(model.compute_loss(loss_fn, pred, y).item() * X.size(0))
            # else:
            pred = model(X)
            
            
            if issubclass(type(model), StationVGP):
                interp_mae += mean_absolute_error(
                    pred.mean.cpu().flatten(), 
                    y[0].cpu().flatten()) * x_size
                kis_mae += mean_absolute_error(
                    pred.mean.cpu().flatten(), 
                    y[1].cpu().flatten()) * x_size
                kis_mae0 += mean_absolute_error(
                    pred.mean.reshape(x_size, -1, 2)[:, :, 0].cpu().flatten(), 
                    y[1][:, :, 0].cpu().flatten()) * x_size
                kis_mae1 += mean_absolute_error(
                    pred.mean.reshape(x_size, -1, 2)[:, :, 1].cpu().flatten(), 
                    y[1][:, :, 1].cpu().flatten()) * x_size
                test_loss += np.longdouble(loss_fn(pred, y[0], isTraining = False).item() * x_size)
            else:
                test_loss += np.longdouble(loss_fn(pred, y, isTraining = False).item() * x_size)

    test_loss /= size
    
    if verbose:
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        
    if issubclass(type(model), StationVGP):
        interp_mae /= size
        kis_mae /= size
        kis_mae0 /= size
        kis_mae1 /= size
        
        if verbose:
            print(f"Interpolation MAE: {interp_mae:>8f}")
            print(f"KIS MAE: {kis_mae:>8f}")
            print(f"KIS MAE Air: {kis_mae0:>8f}")
            print(f"KIS MAE Dew: {kis_mae1:>8f}\n")
        return (test_loss, kis_mae)
    
    return test_loss
    
    
def train_model(
    model,
    dataset,
    batch_size,
    loss_fn,
    optimizer,
    EPOCHS,
    patience,
    scheduler,
    data_saver,
    verbose = False,
    max_batches = None
):
    """
    Trains, evaluates and saves neural models.
    """

    # Train / test split
    train_dataloader = dataset.get_train_loader(batch_size)
    test_dataloader = dataset.get_test_loader(batch_size)    

    # Early stopping
    early_stopper = EarlyStopper(patience = patience, min_delta = 10)

    for t in range(EPOCHS):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        
        start = timer()
        train_loss  = train_one_epoch(train_dataloader, model, loss_fn, optimizer, batch_size, verbose, max_batches)
        middle = timer()
        val_loss    = val_one_epoch(test_dataloader, model, loss_fn, verbose, max_batches)
        end = timer()
        
        if issubclass(type(model), StationVGP):
            data_saver.log_metrics({
                "train_loss"    : float(train_loss),
                "train_time"    : middle - start,
                "val_loss"      : float(val_loss[0]),
                "kis_mae"       : float(val_loss[1]),
                "val_time"      : end - middle,
                "lr"            : scheduler.get_last_lr()[0] if scheduler is not None else 0.0
            })
            val_loss = val_loss[0]
        else:
            data_saver.log_metrics({
                "train_loss"    : float(train_loss),
                "train_time"    : middle - start,
                "val_loss"      : float(val_loss),
                "val_time"      : end - middle,
                "lr"            : scheduler.get_last_lr()[0] if scheduler is not None else 0.0
            })
        
        if early_stopper.early_stop(val_loss):
            print("Stoping early!")
            return model
        
        if scheduler is not None:
            scheduler.step()

    if verbose:
        print("Done!")
        
    return model

def train_and_evaluate_model(
    model,
    dataset,
    batch_size,
    loss_fn,
    optimizer,
    EPOCHS,
    patience,
    scheduler,
    data_saver,
    verbose = False
):
    """
    Implementation of model training that allows for `optuna` optimization.
    """

    # Train / test split
    train_dataloader = dataset.get_train_loader(batch_size)
    test_dataloader = dataset.get_test_loader(batch_size)

    
    total_val_loss = 0

    # Early stopping
    early_stopper = EarlyStopper(patience = patience, min_delta = 10)

    for t in range(EPOCHS):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        
        train_loss  = train_one_epoch(train_dataloader, model, loss_fn, optimizer, batch_size, verbose)
        val_loss    = val_one_epoch(test_dataloader, model, loss_fn, verbose)
        
        kis_loss = val_loss[1]
        
        total_val_loss += kis_loss
        
        if early_stopper.early_stop(kis_loss):
            break
        
        if scheduler is not None:
            scheduler.step()
        
    return total_val_loss