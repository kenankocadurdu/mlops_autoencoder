import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from fastai.vision.all import *
import mlflow
import mlflow.pyfunc

class BaseModule(nn.Module):
    """
    Base class for model modules with common training and validation steps.
    """
    def training_step(self, batch: tuple, criterion: str, model_generator):
        images = batch
        out, mu, log_var, sparsity_loss = self.forward_pass(images, criterion, model_generator)

        # Calculate loss
        loss = self.calculate_loss(out, images, criterion, mu, log_var, model_generator, sparsity_loss)
        return loss, 0  # acc placeholder
    
    def validation_step(self, batch: tuple, criterion: str, model_generator):
        images = batch
        out, mu, log_var, sparsity_loss = self.forward_pass(images, criterion, model_generator)

        # Calculate loss
        loss = self.calculate_loss(out, images, criterion, mu, log_var, model_generator, sparsity_loss)
        return {'val_loss': loss.detach()}

    def forward_pass(self, images, criterion, model_generator):
        """
        Handles forward passes based on the model and criterion.
        """
        mu, log_var, sparsity_loss = None, None, None
        if criterion == "loss_VAE":
            out, mu, log_var = self(images)
        elif model_generator.name == "SparseAutoencoder":
            out, sparsity_loss = self(images)
        else:
            out = self(images)
        return out, mu, log_var, sparsity_loss

    def calculate_loss(self, out, images, criterion, mu, log_var, model_generator, sparsity_loss):
        """
        Calculate the loss based on the criterion and model type.
        """
        if criterion == "MSELoss":
            loss = nn.MSELoss()(out, images)
            if model_generator.name == "SparseAutoencoder":
                encoded = self.encoder(images)
                sparsity_loss = self.sparsity(encoded.mean(dim=0), torch.zeros_like(encoded.mean(dim=0)))
                loss += sparsity_loss

        elif criterion == "SmoothL1Loss":
            loss = nn.SmoothL1Loss()(out, images)

        elif criterion == "BCELoss":
            loss = nn.BCELoss()(torch.sigmoid(out), images.float())

        elif criterion == "loss_VAE":
            loss = loss_VAE(out, images, mu, log_var)

        elif criterion == "KLDivLoss":
            loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(images, dim=1), torch.softmax(out, dim=1))
        
        return loss

    def validation_epoch_end(self, outputs: list):
        """
        Compute the mean validation loss at the end of the epoch.
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        logging.info("Epoch [{}], Train Loss: {:.10f}, Val Loss: {:.10f}".format(epoch, result['train_loss'], result['val_loss']))

@torch.no_grad()
def evaluate(model, val_loader, criterion: str, model_generator):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    outputs = [model.validation_step(batch, criterion, model_generator) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def loss_VAE(reconstructed_x, x, mu, log_var):
    """
    Compute VAE loss, combining reconstruction loss and KL divergence.
    """
    reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

def get_default_device():
    """
    Get the default device (CUDA if available, otherwise CPU).
    """
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    """
    Move tensor(s) to the given device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """
    Wraps a DataLoader to move data to a device.
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss stops improving.
    """
    def __init__(self, patience=5, verbose=False, delta=1e-7, pth_name="checkpoint.pth", lr=0.0001):
        self.patience = patience
        self.counter = 0
        self.lr = lr
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.getcwd() + "/models/" + pth_name

    def __call__(self, val_loss: float, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model):
        """
        Save the model if validation loss has decreased.
        """
        logging.info(f"Validation loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}). Saving model...")
        torch.save(model.state_dict(), self.path + ".pth")
        self.val_loss_min = val_loss

def fit(epochs: int, lr: float, model_generator, train_loader, val_loader, opt_func: str, patience: int, criterion: str, 
        pth_name: str, ml_flow: bool, log_desc: str, save_path: str, path_pth: str, batch_size: int, load_w: bool):
    """
    Train and validate the model with early stopping and optional MLflow logging.
    """
    device = get_default_device()
    model = model_generator.model
    if load_w:
        model.load_state_dict(torch.load(path_pth))
    model = to_device(model, device)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    early_stopping = EarlyStopping(patience=patience, pth_name=pth_name, lr=lr)

    optimizer = get_optimizer(opt_func, model.parameters(), lr)

    history = []
    for epoch in range(epochs):
        logging.info(f"Epoch [{epoch + 1}/{epochs}] started with lr = {early_stopping.lr:.10f}")
        
        # Training Phase
        train_losses = train_one_epoch(model, train_loader, optimizer, criterion, model_generator, ml_flow, epoch)

        # Validation Phase
        result = evaluate(model, val_loader, criterion, model_generator)
        result['train_loss'] = torch.stack(train_losses).mean().item()

        model.epoch_end(epoch, result)
        history.append(result)

        early_stopping(result['val_loss'], model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    plot_training_progress(history, save_path, pth_name)
    return history, model

def get_optimizer(opt_func, parameters, lr):
    """
    Return the optimizer based on the string name.
    """
    if opt_func == 'RMSprop':
        return torch.optim.RMSprop(parameters, lr, alpha=0.9)
    elif opt_func == 'SGD':
        return torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    elif opt_func == 'Adam':
        return torch.optim.Adam(parameters, lr)
    # Add more optimizers if needed
    else:
        raise ValueError(f"Unknown optimizer function: {opt_func}")

def train_one_epoch(model, train_loader, optimizer, criterion, model_generator, ml_flow, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    train_losses = []
    for _indx, batch in enumerate(tqdm(train_loader, desc='Epoch')):
        optimizer.zero_grad()
        loss, _ = model.training_step(batch, criterion, model_generator)
        loss.backward()
        optimizer.step()
        train_losses.append(loss)

        if ml_flow and _indx % 10 == 0:
            mlflow.log_metric(f"batch_loss_epoch_{epoch}", loss.item())
        
        torch.cuda.empty_cache()

    return train_losses

def plot_training_progress(history, save_path, pth_name):
    """
    Plot the training and validation losses across epochs.
    """
    epochs = range(1, len(history) + 1)
    val_loss = [entry['val_loss'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]

    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.plot(epochs, train_loss, 'g-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{pth_name}_loss_plot.png'))
