# Multilevel-in-Width Training for Deep Neural Networks

A PyTorch implementation of a Full Approximation Scheme (FAS)-based multilevel training algorithm for deep neural networks, designed to improve generalization performance through hierarchical representations.

## Overview

This project implements a novel approach to training deep neural networks using multilevel techniques adapted from algebraic multigrid methods. The algorithm constructs a hierarchy of neural networks where each level contains a coarsened version of the original network, leading to improved training efficiency and generalization.

### Key Features

- **Hierarchical Training**: Implements a two-level V-cycle approach for neural network training
- **Heavy Edge Matching (HEM)**: Efficient coarsening strategy for network layers
- **Momentum Transfer**: Smooth parameter updates between hierarchy levels
- **MLflow Integration**: Comprehensive experiment tracking and monitoring
- **Multiple Architecture Support**: 
  - Multilayer Perceptrons (MLP)
  - Convolutional Neural Networks (CNN)
  - Transformer Models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multilevel_in_width.git
cd multilevel_in_width

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
multilevel_in_width/
├── src/
│   ├── models/
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── optimizer.py
│   ├── utils/
│   │   ├── mlflow_utils.py
│   │   └── visualization.py
│   └── config/
│       └── default_config.yaml
├── tests/
├── docs/
│   ├── idea.md
│   ├── technical.md
│   ├── tasks.md
│   └── status.md
├── examples/
├── requirements.txt
└── README.md
```

## Usage

### Basic Example

```python
from src.models.mlp import MultilevelMLP
from src.training.trainer import MultilevelTrainer

# Initialize model
model = MultilevelMLP(input_size=784, hidden_sizes=[512, 256], output_size=10)

# Initialize trainer
trainer = MultilevelTrainer(model=model)

# Train the model
trainer.train(train_loader, val_loader, num_epochs=10)
```

### MLflow Integration

```python
import mlflow

# Start MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("multilevel_training")

# Your training code here
```

## Datasets

The project supports multiple datasets for different architectures:

1. **MNIST** (MLP)
   - 60,000 training images
   - 10,000 test images
   - 28x28 grayscale images

2. **CIFAR-10** (CNN)
   - 50,000 training images
   - 10,000 test images
   - 32x32 RGB images

3. **IMDB** (Transformer)
   - 25,000 training reviews
   - 25,000 test reviews
   - Text classification task

## Documentation

- [Project Overview](docs/idea.md)
- [Technical Details](docs/technical.md)
- [Implementation Tasks](docs/tasks.md)
- [Project Status](docs/status.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multilevel_in_width2024,
  author = {Your Name},
  title = {Multilevel-in-Width Training for Deep Neural Networks},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/multilevel_in_width}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- MLflow team for the comprehensive ML lifecycle management
- Contributors and maintainers of the datasets used in this project 