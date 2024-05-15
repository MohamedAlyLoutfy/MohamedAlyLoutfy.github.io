---
title: 'Mapping to Machine learning'
date: 2024-05-15
tags:
  - cool posts
  - category1
  - category2
---
Inputs
======
The inputs to the machine learning models in fluid mechanics include diverse data forms like turbulence models, CFD solvers, modal decompositions, and high-dimensional flow fields.
Inputs may also be derived from varied sources including numerical simulations, laboratory experiments, and in-flight tests, which provide multifidelity data crucial for training.


Outputs
======
Outputs generally represent physical quantities of interest such as Reynolds stresses, flow fields, or other relevant parameters that the model predicts.
For example, outputs could include predictions of aerodynamic forces, turbulence behaviors, or simulations of fluid dynamics under various conditions.


Loss Function
======
The loss function in fluid mechanics-related machine learning typically includes terms like the L2 error to measure the difference between model predictions and actual outcomes.
Regularization terms (e.g., L1 or L2 norms of parameters) are added to the loss function to promote model simplicity and prevent overfitting.
Physics-informed loss functions may incorporate physical laws directly, promoting physically plausible solutions and enhancing the model's generalizability.


