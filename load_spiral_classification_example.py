import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from neural_network.utils import one_hot_encode, plot_training_history

# Load the trained model
model = NeuralNetwork.load('spiral_model.json')

# Generate some test data (same as training)
def generate_spiral_data(n_samples=200, n_classes=3, noise=0.1):
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)
    
    for class_id in range(n_classes):
        r = np.linspace(0, 1, n_samples)
        t = np.linspace(class_id * 4, (class_id + 1) * 4, n_samples) + np.random.randn(n_samples) * noise
        
        ix = range(n_samples * class_id, n_samples * (class_id + 1))
        X[ix] = np.column_stack([r * np.sin(t * 2.5), r * np.cos(t * 2.5)])
        y[ix] = class_id
    
    return X, one_hot_encode(y, n_classes)

# Generate test data
X_test, y_test = generate_spiral_data(200, 4, 0.2)

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot the decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    y_classes = np.argmax(y, axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=y_classes, edgecolors='k', marker='o', cmap=plt.cm.Spectral)
    plt.title(f'Decision Boundary (Accuracy: {accuracy:.2%})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

plot_decision_boundary(model, X_test, y_test)

# Plot training history from the loaded model
plot_training_history(model.history)

# Print some examples
print("\nSample predictions:")
for i in range(5):
    print(f"Input: {X_test[i]}, True: {true_classes[i]}, Predicted: {predicted_classes[i]}, Confidence: {np.max(predictions[i]):.4f}")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")