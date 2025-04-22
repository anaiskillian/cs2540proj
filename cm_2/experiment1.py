# Builds off of code in circles_moons_classification

def run_terrible_experiment(dataset_name='moons', max_samples=20, seed=42):
    """
    Run a robustness evaluation experiment with a deliberately terrible neural network.

    This function creates a neural network that is highly susceptible to adversarial examples.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create dataset with excessive noise
    print(f"Creating noisy {dataset_name} dataset...")

    if dataset_name == 'circles':
        # Excessive noise in the dataset
        X, y = make_circles(n_samples=100, noise=0.4, factor=0.3, random_state=seed)
        # Extremely narrow architecture - information bottleneck
        hidden_sizes = [2]
        epochs = 2  # Insufficient training
    elif dataset_name == 'moons':
        # Very noisy moons dataset
        X, y = make_moons(n_samples=100, noise=0.5, random_state=seed)
        # Overly simple architecture
        hidden_sizes = [3]
        epochs = 2  # Under-trained
    elif dataset_name == 'xor':
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        # Single neuron can't solve XOR
        hidden_sizes = [1]
        epochs = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # If not XOR, split with uneven distribution
    if dataset_name != 'xor':
        # Extremely unbalanced train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

        # Add outliers to training data
        num_outliers = max(5, int(len(X_train) * 0.1))
        outlier_indices = np.random.choice(len(X_train), num_outliers, replace=False)

        # Flip labels for some training samples to create inconsistency
        for idx in outlier_indices:
            y_train[idx] = 1 - y_train[idx]  # Flip the binary label

        # No scaling - important features will dominate
        # This is bad because the model will be very sensitive to the scale of features
        X_train_scaled = X_train
        X_test_scaled = X_test
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train model with poor settings
    input_size = 2
    output_size = 2  # Binary classification

    class TerribleNN(nn.Module):
        """A neural network designed to be non-robust."""
        def __init__(self, input_size, hidden_sizes, output_size):
            super(TerribleNN, self).__init__()

            self.layers = nn.ModuleList()
            prev_size = input_size

            for hidden_size in hidden_sizes:
                # Initialize with large weights - makes the network sensitive to small changes
                layer = nn.Linear(prev_size, hidden_size)
                nn.init.normal_(layer.weight, mean=0.0, std=2.0)  # Much larger standard deviation
                self.layers.append(layer)
                prev_size = hidden_size

            output_layer = nn.Linear(prev_size, output_size)
            nn.init.normal_(output_layer.weight, mean=0.0, std=2.0)
            self.layers.append(output_layer)

        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                # Use leaky ReLU with a high negative slope - makes boundaries less smooth
                x = torch.nn.functional.leaky_relu(layer(x), negative_slope=0.5)
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

    # Create the terrible model
    model = TerribleNN(input_size, hidden_sizes, output_size)

    # Use a poor optimizer with high learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # High learning rate with basic SGD
    criterion = nn.CrossEntropyLoss()

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)

    print("Training the terrible neural network...")
    loss_history = []

    # Train very briefly with poor early stopping
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # Early stopping that's harmful - stop as soon as we see any minor improvement
        if epoch > 5 and loss_history[-1] < loss_history[-2]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(f'Training Loss for Terrible Model ({dataset_name.capitalize()})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot decision boundary
    plot_decision_boundary(model, X_train_scaled, y_train,
                          title=f"Training Data Decision Boundary - Terrible Model ({dataset_name.capitalize()})")
    plot_decision_boundary(model, X_test_scaled, y_test,
                          title=f"Test Data Decision Boundary - Terrible Model ({dataset_name.capitalize()})")

    # Evaluate the model on the test set
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

    print(f"Test Accuracy of Terrible Model: {accuracy * 100:.2f}%")

    # Define epsilons to test - smaller values to show even minor perturbations cause issues
    epsilons = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # Evaluate robustness methods
    results_df = evaluate_robustness_methods(model, X_test_scaled, y_test, epsilons, max_samples=max_samples)

    # Display results table
    print("\n=== Results Summary for Terrible Model ===")
    print(results_df[['epsilon', 'precision', 'recall', 'f1',
                     'smt_vulnerable_count', 'grad_vulnerable_count', 'total_samples']])

    return results_df, model

# Run the experiment with the terrible model
run_terrible_experiment('moons', max_samples=20)