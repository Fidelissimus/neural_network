import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, Dense, BatchNorm, Dropout
from neural_network.activations import LeakyReLU
from neural_network.utils import train_test_split, one_hot_encode, plot_training_history

# Generate spirals with multiple classes
def generate_spiral_data(n_samples=1000, n_classes=3, noise=0.1):
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)
    
    for class_id in range(n_classes):
        # Spiral parameters
        r = np.linspace(0, 1, n_samples)
        t = np.linspace(class_id * 4, (class_id + 1) * 4, n_samples) + np.random.randn(n_samples) * noise
        
        # Create spiral
        ix = range(n_samples * class_id, n_samples * (class_id + 1))
        X[ix] = np.column_stack([r * np.sin(t * 2.5), r * np.cos(t * 2.5)])
        y[ix] = class_id
    
    return X, one_hot_encode(y, n_classes)

# Generate data
X, y = generate_spiral_data(1500, 4, 0.2)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network, input of (x, y) and outout of N_classes
layers = [
    Dense(2, 128, LeakyReLU(alpha=0.1)),
    BatchNorm(128),
    Dropout(0.2),
    Dense(128, 256, LeakyReLU(alpha=0.1)),
    BatchNorm(256),
    Dropout(0.2),
    Dense(256, 256, LeakyReLU(alpha=0.1)),
    Dense(256, 128, LeakyReLU(alpha=0.1)),
    BatchNorm(128),
    Dropout(0.2),
    Dense(128, 64, LeakyReLU(alpha=0.1)),
    Dense(64, 4, 'softmax')  # Output layer for N_classes classes
]

model = NeuralNetwork(layers)
model.compile(loss='categoricalcrossentropy', optimizer='adam', learning_rate=0.001)

# Print model summary
model.summary()

# Train the model
history = model.train(X_train, y_train, epochs=1000, batch_size=64, 
                     validation_data=(X_test, y_test), verbose=True)

# Save the model
model.save('spiral_model.json')

# Plot training history
plot_training_history(history)

# Create a decision boundary plot
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
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

plot_decision_boundary(model, X_test, y_test)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")