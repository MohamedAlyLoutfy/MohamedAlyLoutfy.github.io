---
title: 'The Data'
date: 2024-05-15
---
The Data
======
Data is fundamental to machine learning (ML), determining the effectiveness of the models built. Key points include:
Interconnectedness with Problem Choice: Selecting data to inform a model is intrinsically linked to choosing the problem to model. This means that data selection cannot be separated from problem definition.

Impact on Model Quality: The quality and quantity of data directly impact the resulting ML model. Diverse and extensive training data are crucial, especially for architectures like deep neural networks, which function as sophisticated interpolation engines.
Example of Deep Learning Success: The rise of modern deep learning is attributed to the pairing of vast labeled datasets with novel deep learning architectures. A notable example is the success of deep convolutional neural networks in image classification, highlighted by their performance on the ImageNet database, which contains over 14 million labeled images across more than 20,000 categories. This large and rich dataset was instrumental in achieving high classification accuracy and marked the beginning of the modern era of deep learning.
Overall, the availability and diversity of data are critical for building effective and generalizable ML models.

Embedding physics in the Data
======
Training data can embed prior physical knowledge in several ways:

Symmetries: If a system exhibits symmetries like translational or rotational invariance, training data can be enriched with shifted or rotated examples.

![Machine Learning Data](../assets/images/Untitled7.png)

Physical Intuition: Crafting new features using physical intuition, such as applying coordinate transformations to simplify representation or training.

Multifidelity Data: Combining data from multiple sources of different fidelity (e.g., simulations, experiments) is crucial, especially in fields like flight testing and unsteady aerodynamics. Recent advances include using physics-informed neural networks with multifidelity data to approximate partial differential equations (PDEs).

Examples in Fluid Mechanics
======
Fluids data is typically vast and high-dimensional, often requiring millions of degrees of freedom to characterize flow fields. These fields evolve over time, creating time-series data. Key points include:

Sparse Data: Despite the large spatial and temporal dimensions, data can be sparse in parameter space due to the high cost of investigating various geometries and Reynolds numbers.

Algorithm Design: There are algorithms designed for both rich and sparse data. In some cases, limited data may come from specific measurements, such as time-series data from a few pressure measurements on an airfoil or force recordings on an experimental turbine.

Transients: Observing system evolution during transients, when the system is away from its natural state, provides valuable insights.

![Machine Learning Data](../assets/images/Untitled6.png)

Overall, embedding physical knowledge into training data and addressing the challenges of vast and high-dimensional fluids data are essential for effective machine learning applications in fluid mechanics.


