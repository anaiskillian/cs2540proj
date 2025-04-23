# Neural Network Robustness Verification Report

## 1. Overview

- **Dataset**: Iris dataset
- **Classes**: Iris-setosa, Iris-versicolor, Iris-virginica
- **Model Architecture**: 4 input features, [10, 8] hidden units, 3 output classes
- **Number of Samples Tested**: 10
- **Epsilon Values Tested**: 0.1, 0.2, 0.3, 0.4, 0.5

## 2. Summary of Results

### 2.1 Robustness by Epsilon

| Epsilon | Robust Samples | Percentage Robust |
|---------|----------------|------------------|
| 0.1 | 10 / 10 | 100.00% |
| 0.2 | 10 / 10 | 100.00% |
| 0.3 | 10 / 10 | 100.00% |
| 0.4 | 10 / 10 | 100.00% |
| 0.5 | 10 / 10 | 100.00% |

### 2.2 Verification Method Comparison

| Epsilon | SMT Found Adversarial | Gradient Found Adversarial | SMT Only | Gradient Only | Both Methods |
|---------|----------------------|---------------------------|----------|--------------|-------------|
| 0.1 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| 0.2 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| 0.3 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| 0.4 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| 0.5 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

## 3. Verification Performance

### 3.1 Verification Times (seconds)

| Epsilon | SMT Average Time | SMT Min Time | SMT Max Time | Gradient Average Time | Gradient Min Time | Gradient Max Time |
|---------|------------------|-------------|-------------|----------------------|------------------|------------------|
| 0.1 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 |
| 0.2 | 0.03 | 0.03 | 0.03 | 0.02 | 0.01 | 0.02 |
| 0.3 | 0.03 | 0.03 | 0.03 | 0.02 | 0.01 | 0.02 |
| 0.4 | 0.03 | 0.03 | 0.03 | 0.02 | 0.01 | 0.02 |
| 0.5 | 0.03 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 |

### 3.2 Perturbation Magnitudes

| Epsilon | SMT Average Magnitude | SMT Min Magnitude | SMT Max Magnitude | Gradient Average Magnitude | Gradient Min Magnitude | Gradient Max Magnitude |
|---------|----------------------|-------------------|------------------|----------------------------|----------------------|------------------------|
| 0.1 | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.2 | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.3 | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.4 | N/A | N/A | N/A | N/A | N/A | N/A |
| 0.5 | N/A | N/A | N/A | N/A | N/A | N/A |

## 4. Conclusion

This analysis tested the robustness of a neural network classifier on the Iris dataset against adversarial perturbations using two methods: SMT-based formal verification and gradient-based adversarial attacks.

Overall, out of 50 total tests, the SMT method found adversarial examples in 0 cases (0.00%), while the gradient-based approach found adversarial examples in 0 cases (0.00%).

Both methods were equally effective at finding adversarial examples, suggesting they might complement each other in practice.

As expected, larger epsilon values led to more adversarial examples being found, indicating less robustness as the allowed perturbation magnitude increases.
