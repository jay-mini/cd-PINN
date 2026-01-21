# cd-PINN: Continuous Dependence Physics-Informed Neural Networks

## Overview

Physics-informed neural networks (PINNs) have emerged as a powerful framework for solving partial differential equations (PDEs) by embedding physical laws directly into neural network training. They have shown particular promise for high-dimensional PDEs and problems with complex or irregular geometries. However, in practical scientific and engineering applications, classical PINNs often suffer from poor generalization performance, especially when extrapolating to unseen parameters, initial conditions, or boundary conditions. Motivated by rigorous mathematical results on the well-posedness of PDEs, we propose cd-PINN (continuous-dependence–informed PINN), a novel extension of PINNs that explicitly incorporates the continuous dependence of PDE solutions on parameters and initial/boundary data. In well-posed PDEs, small perturbations in inputs lead to controlled changes in the solution; cd-PINN embeds this structural property directly into the learning objective.Extensive numerical experiments demonstrate that, under limited labeled data, cd-PINN consistently achieves 1–3 orders of magnitude lower test MSE compared to popular operator-learning baselines such as DeepONet and Fourier Neural Operator (FNO). These results indicate that enforcing continuous dependence provides a principled and effective pathway to extend PINNs from single-instance solvers to reliable operator learners.

## Key Features

- **Enhanced Generalization**: cd-PINN leverages the continuous dependence property inherent to well-posed PDEs, ensuring that the learned solution operator responds smoothly and stably to variations in parameters and initial/boundary conditions. This significantly improves generalization beyond the training distribution.
- **Data Efficiency**: By encoding structural information from PDE theory, cd-PINN reduces reliance on large labeled datasets. Numerical results show that cd-PINN outperforms DeepONet and FNO by 1–3 orders of magnitude in test MSE when only limited supervised data are available.
- **Compatibility with Standard PINNs**: cd-PINN is not a replacement but a minimal and principled extension of standard PINNs. It can be seamlessly integrated into existing PINN frameworks without altering the underlying PDE residual formulation or network architecture.
- **Operator Learning from a PDE Perspective** Unlike purely data-driven operator learning approaches, cd-PINN provides a mathematically grounded bridge between classical PDE well-posedness theory and modern neural operator learning, making the learned operators more interpretable and reliable.
- **Broad Applicability** The methodology is general and applies to a wide class of PDEs, including time-dependent, parameterized, and nonlinear systems, as long as a notion of continuous dependence is available or can be approximated.

## Applications

cd-PINN is particularly suitable for scenarios where robust generalization across varying inputs is essential:
- **Parametric PDEs**: Rapid evaluation of solutions under varying physical parameters (e.g., diffusion coefficients, reaction rates, material properties).
- **Uncertainty Quantification**: Stable propagation of uncertainty in initial conditions, boundary conditions, or coefficients to solution fields.
- **Scientific Operator Learning**: Learning solution operators for families of PDEs with strong physical guarantees, serving as an alternative to DeepONet and FNO when data are scarce.
- **Engineering and Physics Simulations**: Fast surrogate modeling for heat transfer, diffusion–reaction systems, fluid dynamics, and related PDE-governed processes.

By explicitly embedding the continuous dependence structure of PDEs into neural network training, cd-PINN offers a simple yet powerful framework for elevating PINNs from equation solvers to reliable, data-efficient operator learners.

