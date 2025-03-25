"""
Heavy Edge Matching (HEM) implementation for neural network coarsening.
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class HeavyEdgeMatching:
    """
    Implements heavy edge matching for neural network coarsening.
    
    Args:
        coarsening_factor (float): Factor by which to reduce the network size
    """
    
    def __init__(self, coarsening_factor: float = 0.5):
        self.coarsening_factor = coarsening_factor
    
    def compute_similarity_matrix(
        self,
        weights: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute similarity matrix between neurons based on weights and biases.
        
        Args:
            weights (torch.Tensor): Weight matrix of shape (out_features, in_features)
            bias (Optional[torch.Tensor]): Bias vector of shape (out_features,)
            
        Returns:
            torch.Tensor: Similarity matrix of shape (out_features, out_features)
        """
        # Normalize weights
        weights_norm = torch.norm(weights, dim=1)
        weights_normalized = weights / (weights_norm.unsqueeze(1) + 1e-8)
        
        # Compute similarity
        similarity = torch.mm(weights_normalized, weights_normalized.t())
        
        # Add bias contribution if available
        if bias is not None:
            bias_norm = torch.norm(bias)
            bias_normalized = bias / (bias_norm + 1e-8)
            bias_similarity = torch.outer(bias_normalized, bias_normalized)
            similarity = 0.7 * similarity + 0.3 * bias_similarity
        
        return similarity
    
    def find_matches(
        self,
        similarity: torch.Tensor,
        num_neurons: int
    ) -> List[Tuple[int, int]]:
        """
        Find neuron pairs to match using maximum weight matching.
        
        Args:
            similarity (torch.Tensor): Similarity matrix
            num_neurons (int): Number of neurons to match
            
        Returns:
            List[Tuple[int, int]]: List of matched neuron pairs
        """
        # Convert to numpy for scipy operations, detaching from computation graph
        similarity_np = similarity.detach().cpu().numpy()
        
        # Ensure square matrix for minimum spanning tree
        if similarity_np.shape[0] != similarity_np.shape[1]:
            # Take only the first num_neurons rows and columns
            similarity_np = similarity_np[:num_neurons, :num_neurons]
        
        # Create adjacency matrix for minimum weight matching
        # Use negative similarity for minimum spanning tree (equivalent to maximum matching)
        adj_matrix = csr_matrix(-similarity_np)
        
        # Find maximum spanning tree (equivalent to minimum weight matching)
        mst = minimum_spanning_tree(adj_matrix)
        
        # Convert to list of edges
        edges = []
        for i in range(mst.shape[0]):
            for j in range(i + 1, mst.shape[1]):
                if mst[i, j] != 0:
                    edges.append((i, j))
        
        # Sort edges by weight (using positive similarity)
        edges.sort(key=lambda x: similarity_np[x[0], x[1]], reverse=True)
        
        # Select top edges based on coarsening factor
        num_matches = int(num_neurons * (1 - self.coarsening_factor))
        matches = edges[:num_matches]
        
        return matches
    
    def create_coarsening_operators(
        self,
        matches: List[Tuple[int, int]],
        num_neurons: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create restriction and prolongation operators.
        
        Args:
            matches (List[Tuple[int, int]]): List of matched neuron pairs
            num_neurons (int): Total number of neurons
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Restriction and prolongation operators
        """
        num_coarse = num_neurons - len(matches)
        
        # Create prolongation operator
        P = torch.zeros((num_neurons, num_coarse))
        col_idx = 0
        
        # Handle matched neurons
        matched_neurons = set()
        for i, j in matches:
            matched_neurons.add(i)
            matched_neurons.add(j)
            P[i, col_idx] = 0.5
            P[j, col_idx] = 0.5
            col_idx += 1
        
        # Handle unmatched neurons
        for i in range(num_neurons):
            if i not in matched_neurons:
                P[i, col_idx] = 1.0
                col_idx += 1
        
        # Create restriction operator (transpose of prolongation)
        R = P.t()
        
        return R, P
    
    def coarsen_layer(
        self,
        layer: nn.Linear
    ) -> Tuple[nn.Linear, torch.Tensor, torch.Tensor]:
        """
        Coarsen a linear layer using HEM.
        
        Args:
            layer (nn.Linear): Linear layer to coarsen
            
        Returns:
            Tuple[nn.Linear, torch.Tensor, torch.Tensor]: 
                Coarsened layer, restriction operator, prolongation operator
        """
        # Compute similarity matrix
        similarity = self.compute_similarity_matrix(
            layer.weight,
            layer.bias
        )
        
        # Find matches
        matches = self.find_matches(similarity, layer.out_features)
        
        # Create operators
        R, P = self.create_coarsening_operators(
            matches,
            layer.out_features
        )
        
        # Create coarsened layer
        coarse_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=R.shape[0],
            bias=layer.bias is not None
        )
        
        # Initialize coarsened weights and biases
        if layer.bias is not None:
            coarse_layer.bias.data = torch.mm(
                R,
                layer.bias.unsqueeze(1)
            ).squeeze()
        coarse_layer.weight.data = torch.mm(R, layer.weight)
        
        return coarse_layer, R, P 