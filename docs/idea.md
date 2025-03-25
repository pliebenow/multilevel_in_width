# Multilevel-in-Width Training for Deep Neural Network Regression

## Algorithmic Overview

The paper presents a **Full Approximation Scheme (FAS)**-based multilevel training algorithm for deep neural networks (DNNs), designed to improve generalization performance by leveraging hierarchical representations. The key idea is to construct a hierarchy of neural networks where each level contains a coarsened version of the original network, leading to a multilevel training procedure.

## DNN FAS Algorithm

The algorithm follows a **two-level V-cycle** approach, which can be extended recursively:

1. **Perform Stochastic Gradient Descent (SGD)** on the original network to update parameters.
2. **Compute the stochastic tau correction (τ)** to account for the difference between fine and coarse levels.
3. **Apply a restriction operation** to transfer parameters to the coarser neural network representation.
4. **Perform SGD on the coarser level** using the corrected auxiliary problem.
5. **Apply the prolongation operation** to transfer the updated parameters back to the fine level and update weights.
6. **Perform final SGD iterations** on the fine level to refine the solution.
7. If using more than two levels, recursively apply the process at step 4.

## Algorithmic Enhancements

### 1. **Minibatch Selection**

- Minibatches are selected to match traditional one-level training strategies.
- The tau correction is computed over a set of minibatches and reused at the coarse level.
- The same minibatches are used for coarse-level SGD to maintain consistency.

### 2. **Momentum Transfer**

- Momentum vectors are computed and transferred between hierarchy levels.
- Ensures smooth updates and prevents information loss when transitioning between coarse and fine levels.

### 3. **Heavy Edge Matching (HEM) Coarsening**

- Coarsening is performed using the **HEM algorithm**, which groups neurons based on similarity.
- A greedy matching algorithm identifies pairs of neurons with high correlation, reducing dimensionality while preserving structure.

### 4. **Stability Considerations**

- The method maintains numerical stability by selecting appropriate step sizes.
- Uses regularization techniques to prevent excessive overfitting at the coarse level.

## Pseudocode

Below is the pseudocode for the **DNN FAS Algorithm**:

```python
Algorithm DNN_FAS_Training(x, g, levels):
    while not converged do:
        # Apply smoothing at the fine level
        x ← SGD(x, g)
        
        # Construct hierarchy operators
        P, R ← Heavy_Edge_Matching(x)
        xc ← R * x
        mc ← R * m  # Restrict momentum
        
        # Compute tau correction
        tau ← Compute_Tau(g, xc, x, R)
        
        # Solve on coarse level
        gc ← Define_Coarse_Objective(g, xc, tau)
        xc ← SGD(xc, gc)
        
        # Prolongation step
        x ← x + P * (xc - R * x)
        m ← m + P * (mc - R * m)
        
        # Final smoothing
        x ← SGD(x, g)
    return x
```

## Mathematical Formulation

### 1. **Smoothing Operator**

Smoothing is performed using **Stochastic Gradient Descent (SGD)**:

$$
x^{(t+1)} = x^{(t)} - \eta \nabla g_s(x^{(t)})
$$

where:

- \(x^{(t)}\) represents the current set of neural network parameters.
- \(\eta\) is the learning rate.
- \(\nabla g_s(x)\) is the stochastic gradient computed over a minibatch.

### 2. **Correction Term (Tau Correction)**

The tau correction is computed as:

$$
\tau = \nabla g_c(\Pi x) - R \nabla g(x)
$$

where:

- \(\Pi x\) is the restricted (coarse) version of the parameter vector.
- \(R\) is the restriction operator.
- \(g_c(x)\) is the coarse-level loss function.

This correction helps align the coarse gradient approximation with the fine-level optimization process.

### 3. **Restriction and Prolongation Operators**

The restriction operator \(R\) and prolongation operator \(P\) are defined as:

$$
R x = \Pi x
$$

$$
P x_c = P x_c
$$

where:

- \(\Pi\) is a projection operator that maps fine-level parameters to a coarser representation.
- \(P\) is an interpolation operator that maps coarse-level parameters back to the fine level.
- The operators satisfy the property \(\Pi P = I\), ensuring consistency between hierarchy levels.

For layer-wise restriction and prolongation, we use:

$$
W_c = \Pi W P, \quad b_c = \Pi b
$$

$$
W = P W_c \Pi, \quad b = P b_c
$$

where \(W\) and \(b\) represent the weight matrices and bias vectors of the neural network.

### 4. **Restriction and Prolongation in Heavy Edge Matching (HEM) for MLPs and CNNs**

#### **For Multi-Layer Perceptrons (MLPs)**

$$
P_k = \begin{bmatrix}
1 & 1 & 0 & 0 & \dots \\
0 & 0 & 1 & 1 & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix},
\quad \Pi_k = \frac{1}{2} P_k^T
$$

#### **For Convolutional Neural Networks (CNNs)**

In the case of CNNs, HEM is applied across feature channels, resulting in:

$$
P_k = \begin{bmatrix}
I & I & 0 & 0 & \dots \\
0 & 0 & I & I & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix},
\quad \Pi_k = \frac{1}{2} P_k^T
$$

where:
- **Restriction (\(\Pi\))** averages feature maps of matched channels to create a coarser representation.
- **Prolongation (\(P\))** interpolates coarse feature maps back to the fine level.

This preserves spatial structure while reducing channel count, improving computational efficiency in CNNs.

## Summary

The **DNN FAS algorithm** provides a structured way to train deep neural networks using multilevel techniques adapted from algebraic multigrid methods. By applying coarse-to-fine corrections and utilizing hierarchical updates, the method achieves improved generalization and efficiency compared to standard SGD training.


