"""
Training script for Multilevel MLP on MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
from pathlib import Path

from src.models.mlp import MultilevelMLP
from src.training.trainer import MultilevelTrainer


def load_mnist(batch_size: int = 64):
    """
    Load MNIST dataset with preprocessing.
    
    Args:
        batch_size (int): Batch size for data loaders
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data
    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = load_mnist()
    
    # Initialize model
    model = MultilevelMLP(
        input_size=784,  # 28x28 flattened
        hidden_sizes=[512, 256],
        output_size=10,  # 10 classes
        dropout_rate=0.1,
        coarsening_factor=0.5
    )
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize trainer
    trainer = MultilevelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        mlflow_experiment="mnist_multilevel"
    )
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        fine_steps=3,
        coarse_steps=2
    )
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pt')
    
    # Log final metrics
    with mlflow.start_run():
        mlflow.log_metrics({
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1]
        })
        mlflow.pytorch.log_model(model, "model")


if __name__ == '__main__':
    main() 