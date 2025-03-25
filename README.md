# Multilevel-in-Width Training for Deep Neural Networks

A PyTorch implementation of multilevel training algorithms for deep neural networks, featuring Heavy Edge Matching (HEM) and Dual Heavy Edge Matching (Dual-HEM) operators for network coarsening.

## Features

- **Multilevel Training**: Implements hierarchical training strategies for deep neural networks
- **HEM Coarsening**: Heavy Edge Matching algorithm for network dimensionality reduction
- **Dual-HEM Prolongation**: Complementary operator for network prolongation
- **MLflow Integration**: Comprehensive experiment tracking and monitoring
- **Comprehensive Testing**: Extensive test suite for operators and network behavior

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multilevel_in_width.git
cd multilevel_in_width
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
multilevel_in_width/
├── src/
│   ├── __init__.py      # Package initialization
│   ├── models/          # Network architectures
│   │   └── mlp.py      # MLP implementation
│   ├── training/        # Training logic
│   │   └── trainer.py  # Multilevel trainer
│   ├── utils/          # Helper functions
│   │   ├── hem.py     # HEM implementation
│   │   └── dual_hem.py # Dual-HEM implementation
│   └── config/         # Configuration files
├── tests/
│   ├── __init__.py     # Test package initialization
│   ├── conftest.py     # Pytest configuration
│   └── test_operators.py # Operator tests
├── docs/               # Documentation
├── examples/           # Usage examples
└── requirements.txt    # Dependencies
```

## Usage

### Basic MLP Training

```python
from src.models.mlp import MultilevelMLP
from src.training.trainer import MultilevelTrainer

# Initialize model
model = MultilevelMLP(
    input_size=784,
    hidden_sizes=[512, 256],
    output_size=10,
    coarsening_factor=0.5
)

# Initialize trainer
trainer = MultilevelTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=torch.nn.CrossEntropyLoss()
)

# Train model
trainer.train(train_loader, val_loader, num_epochs=10)
```

### Running Tests

1. Make sure you're in the project root directory:
```bash
cd multilevel_in_width
```

2. Run tests:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_operators.py

# Run with coverage report
pytest --cov=src tests/

# Run with verbose output
pytest -v

# Run with print statements
pytest -s
```

## Implementation Details

### Heavy Edge Matching (HEM)

HEM is used for network coarsening, reducing the dimensionality of the network while preserving its structure. The algorithm:

1. Computes similarity matrix between neurons
2. Finds maximum weight matching
3. Creates restriction and prolongation operators

### Dual Heavy Edge Matching (Dual-HEM)

Dual-HEM provides a complementary approach to network prolongation, ensuring:

1. Consistent operator relationships
2. Preserved network properties
3. Improved multilevel training stability

### Testing

The test suite includes:

- Fixed network tests with predefined weights
- Mathematical property verification
- Operator consistency checks
- Scaling behavior tests
- Forward pass preservation tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research in multilevel methods for neural networks
- Inspired by algebraic multigrid techniques
- Built with PyTorch and MLflow 