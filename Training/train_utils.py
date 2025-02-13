import torch
from torch.utils.data import DataLoader
import numpy as np
from timeit import default_timer as timer
import sys
import os
import pathlib
HOME_DIR = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(HOME_DIR)
from Models import GP

class EarlyStopper:
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
    verbose = False):

    running_loss = 0.0
    
    # To obtain a stacktrace when anomalies are detected
    # NOTE: Huge increase in computation time.
    # torch.autograd.set_detect_anomaly(True)
    
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # if issubclass(type(model), GP):
        #     pred = model(X)
        #     loss = model.compute_loss(loss_fn, pred, y)
        # else:
        pred = model(X)
        loss = loss_fn(pred, y)
        
        if verbose:
            print(f"Batch {batch:>3} : loss = {loss.item():>7f}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()

        # Update loss
        running_loss += loss.item() * X.size(0)
            
    # Compute the training loss
    train_loss = running_loss / len(dataloader.dataset)

    return train_loss


def val_one_epoch(dataloader, model, loss_fn, verbose = False):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss = np.longdouble(0.0)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            # if issubclass(type(model), GP):
            #     pred = model(X)
            #     test_loss += np.longdouble(model.compute_loss(loss_fn, pred, y).item() * X.size(0))
            # else:
            pred = model(X)
            test_loss += np.longdouble(loss_fn(pred, y).item() * X.size(0))

    test_loss /= len(dataloader.dataset)
    
    if verbose:
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
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
    verbose = False
):

    # Train / test split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Early stopping
    early_stopper = EarlyStopper(patience = patience, min_delta = 10)

    for t in range(EPOCHS):
        if verbose:
            print(f"Epoch {t+1}\n-------------------------------")
        
        start = timer()
        train_loss  = train_one_epoch(train_dataloader, model, loss_fn, optimizer, batch_size, verbose)
        middle = timer()
        val_loss    = val_one_epoch(test_dataloader, model, loss_fn, verbose)
        end = timer()
        
        data_saver.log_metrics({
            "train_loss"    : float(train_loss),
            "train_time"    : middle - start,
            "val_loss"      : float(val_loss),
            "val_time"      : end - middle,
            "lr"            : scheduler.get_last_lr()[0]
        })
        
        if early_stopper.early_stop(val_loss):
            break
        
        scheduler.step()

    if verbose:
        print("Done!")
        
    return model