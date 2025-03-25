"""
Multilevel MLP implementation with support for hierarchical training.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.hem import HeavyEdgeMatching
from src.utils.dual_hem import DualHeavyEdgeMatching


class BaseMLP(nn.Module):
    """
    Base MLP architecture without multilevel components.
    
    Args:
        input_size (int): Size of input features
        hidden_sizes (List[int]): List of hidden layer sizes
        output_size (int): Size of output layer
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Output layer without activation
        x = self.layers[-1](x)
        return x


class MultilevelMLP(nn.Module):
    """
    Multilevel MLP implementation with hierarchical training support.
    
    Args:
        input_size (int): Size of input features
        hidden_sizes (List[int]): List of hidden layer sizes
        output_size (int): Size of output layer
        dropout_rate (float): Dropout rate for regularization
        coarsening_factor (float): Factor for layer size reduction in coarse levels
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.1,
        coarsening_factor: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.coarsening_factor = coarsening_factor
        
        # Create fine level network
        self.fine_network = BaseMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=dropout_rate
        )
        
        # Initialize HEM and dual-HEM
        self.hem = HeavyEdgeMatching(coarsening_factor=coarsening_factor)
        self.dual_hem = DualHeavyEdgeMatching(
            coarsening_factor=coarsening_factor
        )
        
        # Create coarse level network and operators
        self.coarse_network, self.restriction_ops, self.prolongation_ops = (
            self._create_coarse_network()
        )
        
    def _create_coarse_network(
        self
    ) -> Tuple[BaseMLP, nn.ModuleList, nn.ModuleList]:
        """
        Create coarse network using HEM and dual-HEM.
        
        Returns:
            Tuple[BaseMLP, nn.ModuleList, nn.ModuleList]: 
                Coarse network, restriction operators, prolongation operators
        """
        coarse_layers = nn.ModuleList()
        restriction_ops = nn.ModuleList()
        prolongation_ops = nn.ModuleList()
        
        # Coarsen each layer
        for layer in self.fine_network.layers:
            # Apply HEM for restriction
            coarse_layer, R, P = self.hem.coarsen_layer(layer)
            
            # Apply dual-HEM for prolongation
            dual_layer, P_dual, R_dual = self.dual_hem.create_dual_layer(
                coarse_layer,
                self.hem.find_matches(layer.weight, layer.out_features)
            )
            
            coarse_layers.append(dual_layer)
            restriction_ops.append(R)
            prolongation_ops.append(P_dual)
        
        # Create coarse network
        coarse_network = BaseMLP(
            input_size=self.input_size,
            hidden_sizes=[layer.out_features for layer in coarse_layers[:-1]],
            output_size=self.output_size,
            dropout_rate=self.fine_network.dropout.p
        )
        
        # Copy coarsened layers
        coarse_network.layers = coarse_layers
        
        return coarse_network, restriction_ops, prolongation_ops
    
    def restrict(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """
        Apply restriction operator to reduce dimensionality.
        
        Args:
            x (torch.Tensor): Input tensor
            level (int): Layer level to apply restriction
            
        Returns:
            torch.Tensor: Restricted tensor
        """
        return torch.mm(x, self.restriction_ops[level].t())
    
    def prolong(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """
        Apply prolongation operator to increase dimensionality.
        
        Args:
            x (torch.Tensor): Input tensor
            level (int): Layer level to apply prolongation
            
        Returns:
            torch.Tensor: Prolongated tensor
        """
        return torch.mm(x, self.prolongation_ops[level].t())
    
    def forward(self, x: torch.Tensor, level: str = 'fine') -> torch.Tensor:
        """
        Forward pass through the network at specified level.
        
        Args:
            x (torch.Tensor): Input tensor
            level (str): Level to use ('fine' or 'coarse')
            
        Returns:
            torch.Tensor: Output tensor
        """
        if level == 'fine':
            return self.fine_network(x)
        else:
            return self.coarse_network(x) 