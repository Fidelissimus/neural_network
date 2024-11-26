from neural_network import *

####

# Create a neural network with one input layer, 20 hidden layers, and one output layer
layers = [
    LayerDense(1, 6, relu),    # First hidden layer with ReLU
    LayerDense(6, 8, relu),   # Hidden layer with ReLU
    LayerDense(8, 6, relu),   # Hidden layer with ReLU
    LayerDense(6, 2, tanh)   # Output layer with Sigmoid
]
neural_net = NeuralNetwork(layers)

# Generate dataset with cartion features and targets based on a function
num_samples = 20000
inputs = np.random.rand(num_samples, 1) * 2 * np.pi  # random inputs in the range [0, 2*pi]
#targets = np.sin(inputs.sum(axis=1))[:, None]  # Target is the sine of the sum of the inputs
targets = np.hstack([np.sin(inputs), np.cos(inputs)]) # target is sine and cosine of the input

# Train the neural network
learning_rate = 0.001  # Increased learning rate
epochs = 500
batch_size = 32  # Added batch size, this shit so we train slowly batch by batch (not sure if that even needed but this is how i made it finally learn and work)
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, num_samples, batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]

        neural_net.forward(batch_inputs)
        predictions = neural_net.get_output()
        loss = np.mean((predictions - batch_targets) ** 2)  # Mean squared error loss
        epoch_loss += loss

        # Backpropagation
        errors = predictions - batch_targets
        neural_net.backward(errors)
        neural_net.update(learning_rate)

    epoch_loss /= (num_samples // batch_size)
    losses.append(epoch_loss)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {epoch_loss}')


# Save the trained neural network to a JSON file
#neural_net.save('neural_network.json')


# Plot the loss over epochs
plt.plot(losses)
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Test the neural network on some sample inputs
test_inputs = np.random.rand(10, 1) * 2 * np.pi
#expected_outputs = np.sin(test_inputs.sum(axis=1))[:, None]
expected_outputs = np.hstack([np.sin(test_inputs), np.cos(test_inputs)])
predicted_outputs = neural_net.forward(test_inputs)

print("\nExpected Outputs:")
print(expected_outputs)
print("\nPredicted Outputs:")
print(predicted_outputs)

# Plot the expected vs predicted outputs
plt.plot(expected_outputs, label='Expected Outputs')
plt.plot(predicted_outputs, label='Predicted Outputs', linestyle='--')
plt.title('Expected vs Predicted Outputs')
plt.xlabel('Sample')
plt.ylabel('Output')
plt.legend()
plt.show()

