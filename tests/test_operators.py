"""
Tests for HEM and Dual-HEM operators on fixed networks.
"""

import pytest
import torch
import torch.nn as nn
from src.utils.hem import HeavyEdgeMatching
from src.utils.dual_hem import DualHeavyEdgeMatching


@pytest.fixture
def fixed_network():
    """Create a fixed network with predefined weights."""
    network = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    )
    
    # Set fixed weights and biases
    network[0].weight.data = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6],
        [1.7, 1.8, 1.9, 2.0],
        [2.1, 2.2, 2.3, 2.4]
    ])
    network[0].bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    network[2].weight.data = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
    ])
    network[2].bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    network[4].weight.data = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
    network[4].bias.data = torch.tensor([0.1, 0.2])
    
    return network


def test_hem_operator(fixed_network):
    """Test HEM operator on fixed network."""
    hem = HeavyEdgeMatching(coarsening_factor=0.5)
    
    # Test first layer
    layer = fixed_network[0]
    coarse_layer, R, P = hem.coarsen_layer(layer)
    
    # Check dimensions
    assert coarse_layer.weight.shape[0] == 3  # 6 -> 3 neurons
    assert coarse_layer.weight.shape[1] == 4  # input size unchanged
    assert R.shape == (3, 6)  # restriction operator
    assert P.shape == (6, 3)  # prolongation operator
    
    # Check operator properties
    assert torch.allclose(torch.mm(R, P), torch.eye(3), atol=1e-6)
    
    # Test second layer
    layer = fixed_network[2]
    coarse_layer, R, P = hem.coarsen_layer(layer)
    
    # Check dimensions
    assert coarse_layer.weight.shape[0] == 2  # 4 -> 2 neurons
    assert coarse_layer.weight.shape[1] == 6  # input size unchanged
    assert R.shape == (2, 4)  # restriction operator
    assert P.shape == (4, 2)  # prolongation operator
    
    # Check operator properties
    assert torch.allclose(torch.mm(R, P), torch.eye(2), atol=1e-6)


def test_dual_hem_operator(fixed_network):
    """Test Dual-HEM operator on fixed network."""
    hem = HeavyEdgeMatching(coarsening_factor=0.5)
    dual_hem = DualHeavyEdgeMatching(coarsening_factor=0.5)
    
    # Test first layer
    layer = fixed_network[0]
    matches = hem.find_matches(layer.weight, layer.out_features)
    dual_layer, P_dual, R_dual = dual_hem.create_dual_layer(layer, matches)
    
    # Check dimensions
    assert dual_layer.weight.shape[0] == 6  # same as original
    assert dual_layer.weight.shape[1] == 4  # input size unchanged
    assert P_dual.shape == (6, 3)  # prolongation operator
    assert R_dual.shape == (3, 6)  # restriction operator
    
    # Check operator properties
    assert torch.allclose(torch.mm(R_dual, P_dual), torch.eye(3), atol=1e-6)
    
    # Test second layer
    layer = fixed_network[2]
    matches = hem.find_matches(layer.weight, layer.out_features)
    dual_layer, P_dual, R_dual = dual_hem.create_dual_layer(layer, matches)
    
    # Check dimensions
    assert dual_layer.weight.shape[0] == 4  # same as original
    assert dual_layer.weight.shape[1] == 6  # input size unchanged
    assert P_dual.shape == (4, 2)  # prolongation operator
    assert R_dual.shape == (2, 4)  # restriction operator
    
    # Check operator properties
    assert torch.allclose(torch.mm(R_dual, P_dual), torch.eye(2), atol=1e-6)


def test_combined_operators(fixed_network):
    """Test combined HEM and Dual-HEM operators."""
    hem = HeavyEdgeMatching(coarsening_factor=0.5)
    dual_hem = DualHeavyEdgeMatching(coarsening_factor=0.5)
    
    # Test first layer
    layer = fixed_network[0]
    
    # Apply HEM
    coarse_layer, R, P = hem.coarsen_layer(layer)
    matches = hem.find_matches(layer.weight, layer.out_features)
    
    # Apply Dual-HEM
    dual_layer, P_dual, R_dual = dual_hem.create_dual_layer(coarse_layer, matches)
    
    # Check consistency between operators
    assert torch.allclose(torch.mm(R, P_dual), torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.mm(R_dual, P), torch.eye(3), atol=1e-6)
    
    # Test forward pass consistency
    x = torch.randn(4)  # random input
    y1 = layer(x)
    y2 = torch.mm(P_dual, torch.mm(R, y1))
    assert torch.allclose(y1, y2, atol=1e-6)


def test_operator_properties(fixed_network):
    """Test mathematical properties of operators."""
    hem = HeavyEdgeMatching(coarsening_factor=0.5)
    dual_hem = DualHeavyEdgeMatching(coarsening_factor=0.5)
    
    layer = fixed_network[0]
    matches = hem.find_matches(layer.weight, layer.out_features)
    
    # Get operators
    _, R, P = hem.coarsen_layer(layer)
    _, P_dual, R_dual = dual_hem.create_dual_layer(layer, matches)
    
    # Test idempotency
    assert torch.allclose(torch.mm(R, P), torch.mm(R, P) @ torch.mm(R, P), atol=1e-6)
    assert torch.allclose(torch.mm(R_dual, P_dual), 
                         torch.mm(R_dual, P_dual) @ torch.mm(R_dual, P_dual), 
                         atol=1e-6)
    
    # Test orthogonality
    assert torch.allclose(torch.mm(R, R.t()), torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.mm(P_dual.t(), P_dual), torch.eye(3), atol=1e-6)


def test_operator_scaling(fixed_network):
    """Test operator behavior with different scaling factors."""
    coarsening_factors = [0.25, 0.5, 0.75]
    
    for cf in coarsening_factors:
        hem = HeavyEdgeMatching(coarsening_factor=cf)
        dual_hem = DualHeavyEdgeMatching(coarsening_factor=cf)
        
        layer = fixed_network[0]
        matches = hem.find_matches(layer.weight, layer.out_features)
        
        # Get operators
        _, R, P = hem.coarsen_layer(layer)
        _, P_dual, R_dual = dual_hem.create_dual_layer(layer, matches)
        
        # Check dimensions scale correctly
        expected_size = int(layer.out_features * cf)
        assert R.shape[0] == expected_size
        assert P.shape[1] == expected_size
        assert P_dual.shape[1] == expected_size
        assert R_dual.shape[0] == expected_size 