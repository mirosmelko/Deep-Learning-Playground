import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from GradientDescentAlgorithmsComparison.HyperParameters import HyperParameters
from GradientDescentAlgorithmsComparison.NeuralNetwork import NeuralNetwork

# Read input data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Create parameters for neural network
hp = HyperParameters()
hp.num_steps = 1000  # less parameters than default = who would want to wait that long :)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate)

# Create neural network
nn = NeuralNetwork(optimizer, hp)

# Train neural network
nn.train(mnist.train.images, mnist.train.labels)

# Evaluate trained network on test set
e = nn.evaluate(mnist.test.images, mnist.test.labels)

# Print accuracy for now
print("Testing Accuracy:", e['accuracy'])
