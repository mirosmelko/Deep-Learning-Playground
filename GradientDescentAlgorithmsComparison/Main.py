import shutil

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

from GradientDescentAlgorithmsComparison.HyperParameters import HyperParameters
from GradientDescentAlgorithmsComparison.NeuralNetwork import NeuralNetwork

# Read input data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Create parameters for neural network
hp = HyperParameters()
#hp.num_steps = 10000  # less parameters than default = who would want to wait that long :)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate)

# Prepare log dir
path = os.path.abspath("./logs/log")
if os.path.exists(path):
    shutil.rmtree(path)  # os.remove(path)

# Create neural network
nn = NeuralNetwork(optimizer, hp, path)

# Train neural network
nn.train(mnist.train.images, mnist.train.labels)

# Evaluate trained network on test set
e = nn.evaluate(mnist.test.images, mnist.test.labels)

# Print accuracy for now
print("Testing Accuracy:", e['accuracy'])
