from neural_network import *

# Load the trained neural network from a JSON file
loaded_neural_net = NeuralNetwork.load('sine_nn_example.json', activation_functions)

# Test the loaded neural network on some sample inputs
test_inputs = np.random.rand(10, 1) * 2 * np.pi
expected_outputs = np.sin(test_inputs.sum(axis=1))[:, None]
predicted_outputs = loaded_neural_net.forward(test_inputs)

print("\nExpected Outputs (Loaded Model):")
print(expected_outputs)
print("\nPredicted Outputs (Loaded Model):")
print(predicted_outputs)

# Plot the expected vs predicted outputs for the loaded model
plt.plot(expected_outputs, label='Expected Outputs')
plt.plot(predicted_outputs.flatten(), label='Predicted Outputs (Loaded Model)', linestyle='dashed')
plt.title('Expected vs Predicted Outputs (Loaded Model)')
plt.xlabel('Sample')
plt.ylabel('Output')
plt.legend()
plt.show()
