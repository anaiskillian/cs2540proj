import numpy as np
import torch
import torch.nn as nn
from z3 import *
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns

class SimpleNN(nn.Module):
    """A simple neural network with configurable architecture."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()

        self.layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def get_weights_and_biases(self):
        """Extract weights and biases from the model."""
        params = []
        for layer in self.layers:
            weights = layer.weight.data.numpy()
            biases = layer.bias.data.numpy()
            params.append({'weights': weights, 'biases': biases})
        return params

def train_simple_model(input_size, hidden_sizes, output_size, X_train, y_train, epochs=1000, lr=0.01):
    """Train a simple neural network."""
    model = SimpleNN(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 1 == 0:  # Print every epoch since we're only doing 10
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model, loss_history

def encode_nn_in_z3(model, input_vars):
    """Encode a neural network as Z3 constraints."""
    params = model.get_weights_and_biases()
    layer_outputs = []
    current_vars = input_vars
    constraints = []

    for i, param_dict in enumerate(params):
        weights = param_dict['weights']
        biases = param_dict['biases']
        layer_out = []

        for j in range(len(biases)):
            # Compute weighted sum + bias
            weighted_sum = biases[j]
            for k in range(len(current_vars)):
                weighted_sum += weights[j, k] * current_vars[k]

            # Apply ReLU for all but the last layer
            if i < len(params) - 1:
                relu_out = Real(f'relu_{i}_{j}')
                # ReLU(x) = max(0, x)
                layer_out.append(relu_out)
                # Add constraint: relu_out >= 0
                constraints.append(relu_out >= 0)
                # Add constraint: relu_out >= weighted_sum
                constraints.append(relu_out >= weighted_sum)
                # Add constraint: (relu_out == 0) OR (relu_out == weighted_sum)
                constraints.append(relu_out == If(weighted_sum > 0, weighted_sum, 0))
            else:
                # For the output layer, no ReLU
                output_var = Real(f'output_{j}')
                layer_out.append(output_var)
                constraints.append(output_var == weighted_sum)

        layer_outputs.append(layer_out)
        current_vars = layer_out

    return constraints, layer_outputs[-1]  # Return both constraints and the final layer output variables

def verify_robustness(model, x_sample, epsilon, expected_class, num_classes):
    """
    Verify whether the neural network is robust to perturbations within epsilon
    for a given input sample.

    Returns:
    - is_robust: Boolean indicating whether the model is robust for this sample
    - counterexample: If not robust, a perturbed input that causes misclassification
    - perturbed_class: The class predicted for the perturbed input
    """
    start_time = time.time()

    # Initialize Z3 solver
    solver = Solver()

    # Create variables for the input and perturbed input
    input_dim = x_sample.shape[0]
    x_vars = [Real(f'x_{i}') for i in range(input_dim)]
    x_perturbed = [Real(f'x_perturbed_{i}') for i in range(input_dim)]

    # Add constraints for perturbation
    for i in range(input_dim):
        # Original input value
        solver.add(x_vars[i] == x_sample[i])

        # Perturbation bounds
        solver.add(x_perturbed[i] >= x_sample[i] - epsilon)
        solver.add(x_perturbed[i] <= x_sample[i] + epsilon)

    # Encode the neural network for both original and perturbed inputs
    original_constraints, original_output_vars = encode_nn_in_z3(model, x_vars)
    perturbed_constraints, perturbed_output_vars = encode_nn_in_z3(model, x_perturbed)

    # Add all constraints to the solver
    for constraint in original_constraints:
        solver.add(constraint)

    for constraint in perturbed_constraints:
        solver.add(constraint)

    # Add constraint that the perturbed input leads to a different class
    for i in range(num_classes):
        if i != expected_class:
            # The output for another class is greater than the expected class
            solver.add(perturbed_output_vars[i] > perturbed_output_vars[expected_class])
            break  # Only need to find one misclassification

    # Check satisfiability
    result = solver.check()

    end_time = time.time()
    print(f"Verification completed in {end_time - start_time:.2f} seconds")

    if result == sat:
        # Found a counterexample (adversarial example)
        model_solution = solver.model()

        # Extract the perturbed input values
        counterexample = np.array([float(model_solution.eval(var).as_decimal(10).replace('?', ''))
                                for var in x_perturbed])

        # Find predicted class for perturbed input
        perturbed_outputs = []

        for j in range(num_classes):
            # Get the output variable's value from the model
            value = float(model_solution.eval(perturbed_output_vars[j]).as_decimal(10).replace('?', ''))
            perturbed_outputs.append(value)

        perturbed_class = np.argmax(perturbed_outputs)

        print(f"Found adversarial example: Original class: {expected_class}, Perturbed class: {perturbed_class}")
        print(f"Perturbation magnitude: {np.linalg.norm(counterexample - x_sample)}")

        return False, counterexample, perturbed_class
    else:
        print(f"No adversarial example found within epsilon = {epsilon}")
        return True, None, None

def adversarial_attack(model, x_sample, epsilon, expected_class, steps=100):
    """
    Perform a simple gradient-based adversarial attack (FGSM-like)
    to compare with SMT results.
    """
    # Convert to PyTorch tensor
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(0)
    x_tensor.requires_grad = True

    # Target is the expected class
    target = torch.LongTensor([expected_class])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Maximize loss instead of minimizing
    for i in range(steps):
        # Forward pass
        output = model(x_tensor)
        loss = -criterion(output, target)  # Negative because we want to maximize

        # Backward pass
        loss.backward()

        # Get gradient sign
        grad_sign = x_tensor.grad.sign().detach().numpy().squeeze()

        # Update tensor
        step_size = epsilon / steps
        x_tensor_np = x_tensor.detach().numpy().squeeze()
        x_tensor_np += step_size * grad_sign

        # Clip to ensure we're within epsilon
        x_tensor_np = np.clip(x_tensor_np, x_sample - epsilon, x_sample + epsilon)

        # Create new tensor
        x_tensor = torch.FloatTensor(x_tensor_np).unsqueeze(0)
        x_tensor.requires_grad = True

        # Get predictions
        pred = model(x_tensor).detach().numpy().squeeze()
        pred_class = np.argmax(pred)

        # Check if we've successfully created an adversarial example
        if pred_class != expected_class:
            print(f"Found adversarial example at step {i}")
            return x_tensor_np, pred_class

    # Get final prediction
    pred = model(x_tensor).detach().numpy().squeeze()
    pred_class = np.argmax(pred)

    return x_tensor.detach().numpy().squeeze(), pred_class

def evaluate_robustness_methods(model, X_test, y_test, epsilons, max_samples=50):
    """
    Evaluate and compare SMT-based and gradient-based robustness verification.

    Args:
        model: The neural network model
        X_test: Test data
        y_test: Test labels
        epsilons: List of epsilon values to test
        max_samples: Maximum number of samples to test

    Returns:
        results_df: DataFrame with metrics for each epsilon
    """
    # Get model predictions on test set
    with torch.no_grad():
        predicted = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()

    # Filter correctly classified samples
    correct_indices = np.where(predicted == y_test)[0]
    print(f"Total correctly classified samples: {len(correct_indices)}/{len(X_test)}")

    # Select a subset of correctly classified samples
    num_samples = min(max_samples, len(correct_indices))
    selected_indices = np.random.choice(correct_indices, size=num_samples, replace=False)

    results = []

    for epsilon in epsilons:
        print(f"\n=== Testing epsilon = {epsilon} ===")

        # Initialize counters for confusion matrix
        smt_vulnerable = []  # SMT says vulnerable
        grad_vulnerable = []  # Gradient attack says vulnerable

        for i, idx in enumerate(selected_indices):
            x_sample = X_test[idx]
            expected_class = predicted[idx]

            print(f"\nSample {i+1}/{num_samples}")

            # SMT-based verification
            print("Running SMT verification...")
            is_robust_smt, counterexample_smt, perturbed_class_smt = verify_robustness(
                model, x_sample, epsilon, expected_class, model.layers[-1].out_features
            )

            # Gradient-based attack
            print("Running gradient-based attack...")
            x_perturbed_grad, perturbed_class_grad = adversarial_attack(
                model, x_sample, epsilon, expected_class
            )

            # Record results
            smt_vulnerable.append(not is_robust_smt)
            grad_vulnerable.append(perturbed_class_grad != expected_class)

            # Print comparison
            print("\n--- Comparison ---")
            print(f"SMT: {'Vulnerable' if not is_robust_smt else 'Robust'}")
            print(f"Gradient Attack: {'Vulnerable' if perturbed_class_grad != expected_class else 'Robust'}")

        # Calculate metrics (using gradient attack as ground truth)
        # This is an assumption - in reality, we might want to use the SMT solver as ground truth
        # since it's theoretically more complete

        # Convert to numpy arrays for easier computation
        smt_vulnerable = np.array(smt_vulnerable)
        grad_vulnerable = np.array(grad_vulnerable)

        # True positives: Both methods find vulnerability
        tp = np.sum(smt_vulnerable & grad_vulnerable)

        # False positives: SMT finds vulnerability, gradient doesn't
        fp = np.sum(smt_vulnerable & ~grad_vulnerable)

        # False negatives: SMT doesn't find vulnerability, gradient does
        fn = np.sum(~smt_vulnerable & grad_vulnerable)

        # True negatives: Both methods find no vulnerability
        tn = np.sum(~smt_vulnerable & ~grad_vulnerable)

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Store results
        results.append({
            'epsilon': epsilon,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'smt_vulnerable_count': np.sum(smt_vulnerable),
            'grad_vulnerable_count': np.sum(grad_vulnerable),
            'total_samples': num_samples
        })

        print(f"\n--- Metrics for epsilon = {epsilon} ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Negatives: {tn}")
        print(f"SMT Vulnerable Count: {np.sum(smt_vulnerable)}/{num_samples}")
        print(f"Gradient Vulnerable Count: {np.sum(grad_vulnerable)}/{num_samples}")

        # Create and display confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Robust (Grad)', 'Vulnerable (Grad)'],
                    yticklabels=['Robust (SMT)', 'Vulnerable (SMT)'])
        plt.title(f'Confusion Matrix (ε = {epsilon})')
        plt.ylabel('SMT Solver')
        plt.xlabel('Gradient Attack')
        plt.show()

    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Plot metrics vs epsilon
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['epsilon'], results_df['precision'], 'o-', label='Precision')
    plt.plot(results_df['epsilon'], results_df['recall'], 's-', label='Recall')
    plt.plot(results_df['epsilon'], results_df['f1'], '^-', label='F1 Score')
    plt.xlabel('Epsilon')
    plt.ylabel('Score')
    plt.title('SMT Solver Performance vs Epsilon (Compared to Gradient Attack)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot vulnerability rate vs epsilon
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['epsilon'], results_df['smt_vulnerable_count'] / results_df['total_samples'], 'o-',
             label='SMT Vulnerability Rate')
    plt.plot(results_df['epsilon'], results_df['grad_vulnerable_count'] / results_df['total_samples'], 's-',
             label='Gradient Vulnerability Rate')
    plt.xlabel('Epsilon')
    plt.ylabel('Vulnerability Rate')
    plt.title('Vulnerability Rate vs Epsilon')
    plt.legend()
    plt.grid(True)
    plt.show()

    return results_df

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot the decision boundary of the model."""
    h = 0.02  # Step size in the mesh

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions on the mesh grid
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            # Convert to tensor
            x_tensor = torch.FloatTensor([xx[i, j], yy[i, j]])

            # Get prediction
            with torch.no_grad():
                output = model(x_tensor)
                Z[i, j] = torch.argmax(output).item()

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()

def create_complex_dataset(n_samples=500, n_classes=5, n_features=2, random_state=42):
    """
    Create a complex classification dataset with multiple classes and different shapes.
    """
    # Create blobs with different cluster sizes
    np.random.seed(random_state)

    # Create blobs in a circular pattern
    centers = []
    radius = 5
    for i in range(n_classes):
        angle = 2 * np.pi * i / n_classes
        center = [radius * np.cos(angle), radius * np.sin(angle)]
        centers.append(center)

    centers = np.array(centers)

    # Different cluster standard deviations to create different shapes
    cluster_std = [0.8, 1.2, 0.5, 1.5, 0.7][:n_classes]

    # Generate the blobs
    X, y = make_blobs(n_samples=n_samples, centers=centers,
                      n_features=n_features, cluster_std=cluster_std,
                      random_state=random_state)

    # Add some non-linearity by applying transformations to some points
    indices = np.random.choice(len(X), size=int(0.3*len(X)), replace=False)
    X[indices] = X[indices] * np.random.normal(1, 0.2, size=(len(indices), n_features))

    # Add some outliers
    n_outliers = int(0.05 * n_samples)
    outlier_indices = np.random.choice(len(X), size=n_outliers, replace=False)
    X[outlier_indices] = np.random.uniform(-10, 10, size=(n_outliers, n_features))

    return X, y

def flip_labels(y, flip_percentage=0.35, random_state=42):
    """
    Flip a percentage of the labels to introduce noise.
    """
    np.random.seed(random_state)
    n_samples = len(y)
    n_classes = len(np.unique(y))

    # Number of labels to flip
    n_flip = int(n_samples * flip_percentage)

    # Indices to flip
    flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)

    # New random labels for the flipped indices
    y_flipped = y.copy()
    for idx in flip_indices:
        current_label = y[idx]
        # Choose a random label different from the current one
        possible_labels = [l for l in range(n_classes) if l != current_label]
        new_label = np.random.choice(possible_labels)
        y_flipped[idx] = new_label

    return y_flipped

def run_modified_experiment(max_samples=20, seed=42):
    """
    Run a complete robustness evaluation experiment with the modified dataset.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create more complex dataset with 5 classes
    print("Creating complex dataset...")
    X, y = create_complex_dataset(n_samples=500, n_classes=5, random_state=seed)

    # Flip 35% of labels to introduce noise
    print("Flipping 35% of labels...")
    y = flip_labels(y, flip_percentage=0.35, random_state=seed)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model with only 10 epochs
    input_size = 2
    output_size = 5  # 5-class classification
    hidden_sizes = [20, 15, 10]  # More complex architecture

    print("Training neural network for only 10 epochs...")
    model, loss_history = train_simple_model(input_size, hidden_sizes, output_size,
                                          X_train, y_train, epochs=10, lr=0.01)

    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss (Complex Dataset, 10 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot decision boundary
    plot_decision_boundary(model, X_train, y_train, title="Training Data Decision Boundary (Complex Dataset)")
    plot_decision_boundary(model, X_test, y_test, title="Test Data Decision Boundary (Complex Dataset)")

    # Evaluate the model on the test set
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Define epsilons to test - using smaller values since the model is less robust
    epsilons = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    # Evaluate robustness methods
    results_df = evaluate_robustness_methods(model, X_test, y_test, epsilons, max_samples=max_samples)

    # Display results table
    print("\n=== Results Summary ===")
    print(results_df[['epsilon', 'precision', 'recall', 'f1',
                     'smt_vulnerable_count', 'grad_vulnerable_count', 'total_samples']])

    return results_df, model

# Run the modified experiment
run_modified_experiment(max_samples=20)