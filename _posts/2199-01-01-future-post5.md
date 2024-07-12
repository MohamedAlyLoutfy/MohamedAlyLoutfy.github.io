---
title: 'The architecture '
date: 2024-05-15
tags:
  - cool posts
  - category1
  - category2
---
The architecture 
======
Choosing an architecture to represent the machine learning (ML) model is crucial once the problem is identified, and data is collected and curated. A typical ML model is a function that maps inputs to outputs, often represented within a specified family of functions parameterized by θ. 

Function Representation:
    ML models map inputs (x) to outputs (y) through a function f(x;θ).
    Linear regression models outputs as a linear function of inputs, with θ parameterizing this linear map.
Neural Networks:
    Powerful and Flexible: Neural networks can approximate complex functions given sufficient data and depth.
    Variety of Architectures: The most common is the feedforward network, where data passes through an input layer, multiple computational layers, and an output layer.
    Layer Composition: Each layer consists of nodes where data is processed through a weighted sum and a nonlinear activation function.
    Parameters (θ): Determine the network weights and how data is passed between layers.
    Network Topology: The architecture (number of layers, size, activation functions) is specified by the designer or through meta-optimization, determining the family of functions the network can approximate.
Optimization:
    Network weights are optimized over the data to minimize a given loss function.
Alternative Architectures:
    Not all ML architectures are neural networks. Other prominent models include:
        Random Forests: A leading architecture for supervised learning.
        Support Vector Machines: Another top architecture for supervised learning.
        Bayesian Methods: Widely used, especially for dynamical systems.
        Genetic Programming: Used for learning human-interpretable models and control.
        Standard Linear and Generalized Linear Regression: Common for modeling time-series data.
        Dynamic Mode Decomposition (DMD): Uses linear regression with a low-rank constraint to find dominant spatiotemporal structures.
        Sparse Identification of Nonlinear Dynamics (SINDy): Uses generalized linear regression with sparsity-promoting loss functions or sparse optimization algorithms to identify differential equation models with minimal terms necessary to fit the data.
Overall, the architecture choice, from neural networks to alternative models, is essential for accurately representing the ML model and leveraging the data effectively.

Embedding physics in the architecture
======
Choosing a machine learning (ML) architecture provides a significant opportunity to embed physical knowledge into the learning process. Key methods include:
    Convolutional Networks: Suitable for translationally invariant systems.
    Recurrent Networks: Such as Long-Short Term Memory (LSTM) networks and reservoir computing, which are effective for systems evolving over time.
    Equivariant Networks: Designed to encode various symmetries, improving accuracy and reducing data requirements.
    Autoencoder Networks: Impose an information bottleneck to uncover low-dimensional structure in high-dimensional data.
    Physics-Embedded Architectures: Incorporate structures like Hamiltonian or Lagrangian frameworks.
    Physics-Informed Neural Networks (PINNs): Solve supervised learning problems while being constrained by governing physical laws.
    Graph Neural Networks: Learn generalizable physics across various domains.
    Deep Operator Networks: Learn continuous operators, such as governing partial differential equations, from limited training data.

2.3.2 Examples in Fluid Mechanics
======
Custom neural network architectures are used extensively to enforce physical solutions in fluid mechanics applications:
    Galilean Invariance: Ling et al. designed a neural network layer enforcing Galilean invariance in Reynolds stress tensors.
    Reynolds Stress Models: Developed using the SINDy sparse modeling approach.
    Hybrid Models: Combining linear system identification with nonlinear neural networks for complex aeroelastic systems.
    Hidden Fluid Mechanics (HFM): A physics-informed neural network strategy encoding Navier–Stokes equations, flexible to boundary conditions and geometry, for accurate flow field estimations from limited data.
    Sparse Sensing: Used to recover pressure distributions around airfoils.
    Fourier Neural Operator: Performs super-resolution upscaling and simulation modeling tasks.
    Equivariant Convolutional Networks: Enforce symmetries in high-dimensional complex systems from fluid dynamics.
    Subgrid-Scale Scalar Flux Modeling: Incorporate physical invariances into neural networks.
    Deep Convolutional Autoencoder Networks: Integrated into reduced-order modeling frameworks for superior dimensionality reduction capabilities.


