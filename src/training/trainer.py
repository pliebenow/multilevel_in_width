"""
Multilevel training implementation with MLflow integration.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

from src.models.mlp import MultilevelMLP


class MultilevelTrainer:
    """
    Trainer class for multilevel neural network training.
    
    Args:
        model (MultilevelMLP): The multilevel model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (nn.Module): Loss function
        device (torch.device): Device to train on
        mlflow_experiment (str): Name of MLflow experiment
    """
    
    def __init__(
        self,
        model: MultilevelMLP,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        mlflow_experiment: str = "multilevel_training"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.mlflow_experiment = mlflow_experiment
        
        # Initialize MLflow
        mlflow.set_experiment(mlflow_experiment)
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        level: str = 'fine'
    ) -> Dict[str, float]:
        """
        Perform one training step at specified level.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input and target tensors
            level (str): Training level ('fine' or 'coarse')
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        outputs = self.model(x, level=level)
        loss = self.criterion(outputs, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        level: str = 'fine'
    ) -> Dict[str, float]:
        """
        Train for one epoch at specified level.
        
        Args:
            train_loader (DataLoader): Training data loader
            level (str): Training level ('fine' or 'coarse')
            
        Returns:
            Dict[str, float]: Average metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = len(train_loader)
        
        for batch in tqdm(train_loader, desc=f"Training {level} level"):
            metrics = self.train_step(batch, level)
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        level: str = 'fine'
    ) -> Dict[str, float]:
        """
        Validate the model at specified level.
        
        Args:
            val_loader (DataLoader): Validation data loader
            level (str): Validation level ('fine' or 'coarse')
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating {level} level"):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x, level=level)
                loss = self.criterion(outputs, y)
                
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y).sum().item()
                accuracy = correct / y.size(0)
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        fine_steps: int = 3,
        coarse_steps: int = 2
    ) -> Dict[str, List[float]]:
        """
        Train the model using multilevel approach.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            fine_steps (int): Number of fine level steps per cycle
            coarse_steps (int): Number of coarse level steps per cycle
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Fine level training
            for _ in range(fine_steps):
                train_metrics = self.train_epoch(train_loader, level='fine')
                val_metrics = self.validate(val_loader, level='fine')
                
                # Log metrics
                with mlflow.start_run(nested=True):
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy']
                    })
                
                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['train_accuracy'].append(train_metrics['accuracy'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Coarse level training
            for _ in range(coarse_steps):
                train_metrics = self.train_epoch(train_loader, level='coarse')
                val_metrics = self.validate(val_loader, level='coarse')
                
                # Log metrics
                with mlflow.start_run(nested=True):
                    mlflow.log_metrics({
                        'coarse_train_loss': train_metrics['loss'],
                        'coarse_train_accuracy': train_metrics['accuracy'],
                        'coarse_val_loss': val_metrics['loss'],
                        'coarse_val_accuracy': val_metrics['accuracy']
                    })
        
        return history 