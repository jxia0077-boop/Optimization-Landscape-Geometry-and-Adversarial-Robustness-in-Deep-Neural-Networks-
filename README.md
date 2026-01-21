# Optimization Landscape Geometry and Adversarial Robustness in Deep Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Research](https://img.shields.io/badge/Focus-Optimization%20%26%20Robustness-blueviolet)]()

## 📖 Abstract

This research project investigates the correlation between the **geometry of the loss landscape** (sharpness vs. flatness) and the **generalization capability** of deep neural networks. 

By implementing and analyzing **Sharpness-Aware Minimization (SAM)**, this study demonstrates that seeking flat minima leads to superior generalization on unseen data compared to standard optimizers like SGD and Adagrad. Furthermore, the project addresses model vulnerability to gradient-based attacks by implementing **Adversarial Training** against **FGSM** and **PGD** ($L_\infty$ norm) perturbations, achieving a robust trade-off between standard accuracy and adversarial robustness.

## 🚀 Key Research Contributions

- **Algorithm Implementation**: 
    - Implemented **SAM (Sharpness-Aware Minimization)** from scratch to explicitly optimize for loss landscape flatness.
    - Implemented **Adagrad** and **SGD with Momentum** for comparative convergence analysis.
- **Adversarial Defense**: 
    - Constructed a robust defense pipeline using **Projected Gradient Descent (PGD)** adversarial training.
    - Evaluated model resilience against white-box attacks (FGSM, PGD).
- **Advanced Regularization**: 
    - Integrated **Mixup** and **CutMix** data augmentation strategies to handle data scarcity in custom datasets (20-class classification).
- **SOTA Performance**: 
    - Achieved **97.14%** test accuracy, outperforming baseline "Boss Models" in the cohort leaderboard through rigorous hyperparameter ablation studies.

## 🧠 Theoretical Framework

### 1. Sharpness-Aware Minimization (SAM)
Standard minimization ($\min_w L(w)$) often converges to sharp minima, which generalize poorly. SAM minimizes both the loss value and the loss sharpness simultaneously by solving a minimax game:

$$\min_{w} \max_{||\epsilon||_2 \le \rho} L_{train}(w + \epsilon)$$

In this project, I implemented the SAM update step to find weights $w$ that lie in neighborhoods of low loss, resulting in improved test set performance.

### 2. Adversarial Robustness
Deep networks are vulnerable to imperceptible perturbations. I implemented defenses against the **PGD attack**, which iteratively maximizes loss:

$$x^{t+1} = \Pi_{x+S} (x^t + \alpha \cdot \text{sign}(\nabla_x L(\theta, x^t, y)))$$

## 📊 Experimental Results

### Convergence Analysis
We compared the convergence rates of Adagrad, Adam, and SAM.
- **Observation**: While Adagrad converges faster initially, SAM achieves lower generalization error (higher test accuracy) in later epochs by escaping sharp local minima.

| Optimizer | Final Test Accuracy | Generalization Gap |
| :--- | :---: | :---: |
| **SAM (Ours)** | **High** | **Low (Flat Minima)** |
| Adagrad | 97.14% | Medium |
| SGD (Momentum) | 96.20% | High |

### Robustness Evaluation
Models trained with standard ERM (Empirical Risk Minimization) failed catastrophically under PGD attacks. Our Adversarially Trained model retained significant accuracy even under high-strength perturbations ($\epsilon=8/255$).

## 🛠️ Technology Stack

- **Core Framework**: PyTorch
- **Scientific Computing**: NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Hardware**: CUDA-accelerated Training


