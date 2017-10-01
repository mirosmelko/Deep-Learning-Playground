from tensorflow.examples.tutorials.mnist import input_data
from GradientDescentAlgorithmsComparison.NeuralNetwork import NeuralNetwork

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
nn = NeuralNetwork()

e = nn.evaluate(mnist.test.images, mnist.test.labels)
print("Testing Accuracy:", e['accuracy'])
