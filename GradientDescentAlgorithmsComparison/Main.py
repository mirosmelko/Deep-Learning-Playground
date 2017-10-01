from tensorflow.examples.tutorials.mnist import input_data
from GradientDescentAlgorithmsComparison.NeuralNetwork import NeuralNetwork

# Read input data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Create neural network
nn = NeuralNetwork()

# Train neural network
nn.train(mnist.train.images, mnist.train.labels)

# Evaluate trained network on test set
e = nn.evaluate(mnist.test.images, mnist.test.labels)

# Print accuracy for now
print("Testing Accuracy:", e['accuracy'])
