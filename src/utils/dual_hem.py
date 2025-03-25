"""
Dual Heavy Edge Matching (Dual-HEM) implementation for neural network prolongation.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


class DualHeavyEdgeMatching:
    """
    Implements dual heavy edge matching for neural network prolongation.
    
    Args:
        coarsening_factor (float): Factor by which to reduce the network size
        interpolation_weight (float): Weight for interpolation between levels
    """
    
    def __init__(
        self,
        coarsening_factor: float = 0.5,
        interpolation_weight: float = 0.5
    ):
        self.coarsening_factor = coarsening_factor
        self.interpolation_weight = interpolation_weight
    
    def compute_dual_matrix(
        self,
        weights: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute dual matrix for prolongation based on weights and biases.
        
        Args:
            weights (torch.Tensor): Weight matrix of shape (out_features, in_features)
            bias (Optional[torch.Tensor]): Bias vector of shape (out_features,)
            
        Returns:
            torch.Tensor: Dual matrix of shape (out_features, out_features)
        """
        # Normalize weights
        weights_norm = torch.norm(weights, dim=1)
        weights_normalized = weights / (weights_norm.unsqueeze(1) + 1e-8)
        
        # Compute dual similarity
        dual_similarity = torch.mm(weights_normalized, weights_normalized.t())
        
        # Add bias contribution if available
        if bias is not None:
            bias_norm = torch.norm(bias)
            bias_normalized = bias / (bias_norm + 1e-8)
            bias_dual = torch.outer(bias_normalized, bias_normalized)
            dual_similarity = 0.7 * dual_similarity + 0.3 * bias_dual
        
        return dual_similarity
    
    def find_dual_matches(
        self,
        dual_matrix: torch.Tensor,
        num_neurons: int
    ) -> List[Tuple[int, int]]:
        """
        Find dual neuron pairs for prolongation.
        
        Args:
            dual_matrix (torch.Tensor): Dual similarity matrix
            num_neurons (int): Number of neurons to match
            
        Returns:
            List[Tuple[int, int]]: List of dual matched neuron pairs
        """
        # Convert to numpy for scipy operations, detaching from computation graph
        dual_np = dual_matrix.detach().cpu().numpy()
        
        # Ensure square matrix for minimum spanning tree
        if dual_np.shape[0] != dual_np.shape[1]:
            # Take only the first num_neurons rows and columns
            dual_np = dual_np[:num_neurons, :num_neurons]
        
        # Create adjacency matrix for minimum weight matching
        # Use negative similarity for minimum spanning tree (equivalent to maximum matching)
        adj_matrix = csr_matrix(-dual_np)
        
        # Find maximum spanning tree (equivalent to minimum weight matching)
        mst = minimum_spanning_tree(adj_matrix)
        
        # Convert to list of edges
        edges = []
        for i in range(mst.shape[0]):
            for j in range(i + 1, mst.shape[1]):
                if mst[i, j] != 0:
                    edges.append((i, j))
        
        # Sort edges by weight (using positive similarity)
        edges.sort(key=lambda x: dual_np[x[0], x[1]], reverse=True)
        
        # Select top edges based on coarsening factor
        num_matches = int(num_neurons * (1 - self.coarsening_factor))
        matches = edges[:num_matches]
        
        return matches
    
    def create_dual_operators(
        self,
        matches: List[Tuple[int, int]],
        num_neurons: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create dual prolongation and restriction operators.
        
        Args:
            matches (List[Tuple[int, int]]): List of dual matched neuron pairs
            num_neurons (int): Total number of neurons
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Dual prolongation and restriction operators
        """
        num_coarse = num_neurons - len(matches)
        
        # Create dual prolongation operator
        P_dual = torch.zeros((num_neurons, num_coarse))
        col_idx = 0
        
        # Handle matched neurons
        matched_neurons = set()
        for i, j in matches:
            matched_neurons.add(i)
            matched_neurons.add(j)
            P_dual[i, col_idx] = self.interpolation_weight
            P_dual[j, col_idx] = 1 - self.interpolation_weight
            col_idx += 1
        
        # Handle unmatched neurons
        for i in range(num_neurons):
            if i not in matched_neurons:
                P_dual[i, col_idx] = 1.0
                col_idx += 1
        
        # Create dual restriction operator (transpose of prolongation)
        R_dual = P_dual.t()
        
        return P_dual, R_dual
    
    def create_dual_layer(
        self,
        layer: nn.Linear,
        matches: List[Tuple[int, int]]
    ) -> Tuple[nn.Linear, torch.Tensor, torch.Tensor]:
        """
        Create dual layer for prolongation.
        
        Args:
            layer (nn.Linear): Original linear layer
            matches (List[Tuple[int, int]]): List of dual matched neuron pairs
            
        Returns:
            Tuple[nn.Linear, torch.Tensor, torch.Tensor]: 
                Dual layer, dual prolongation operator, dual restriction operator
        """
        # Compute dual matrix
        dual_matrix = self.compute_dual_matrix(
            layer.weight,
            layer.bias
        )
        
        # Create dual operators
        P_dual, R_dual = self.create_dual_operators(
            matches,
            layer.out_features
        )
        
        # Create dual layer
        dual_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=P_dual.shape[0],
            bias=layer.bias is not None
        )
        
        # Initialize dual weights and biases
        if layer.bias is not None:
            dual_layer.bias.data = torch.mm(
                P_dual,
                layer.bias.unsqueeze(1)
            ).squeeze()
        dual_layer.weight.data = torch.mm(P_dual, layer.weight)
        
        return dual_layer, P_dual, R_dual 