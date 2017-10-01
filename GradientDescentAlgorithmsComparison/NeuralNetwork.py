"""
Trains neural network using tensorflow
"""

import tensorflow as tf


class NeuralNetwork:

    hp = None
    model = None
    optimizer = None

    def __init__(self, optimizer, hyper_parameters):
        self.model = tf.estimator.Estimator(self.model_fn)
        self.optimizer = optimizer
        self.hp = hyper_parameters

    """
    Builds neural network
    """
    def model_fn(self, features, labels, mode):
        # we just need the images from the dictionary
        x = features['images']

        # Hidden fully connected layer
        layer_1 = tf.layers.dense(x, self.hp.n_hidden_1)

        # Hidden fully connected layer
        layer_2 = tf.layers.dense(layer_1, self.hp.n_hidden_2)

        # Output fully connected layer with a neuron for each class
        logits = tf.layers.dense(layer_2, self.hp.num_classes)

        # Predictions
        y_ = tf.argmax(logits, axis=1)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

        # Create train operation using given optimizer
        train_op = self.optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=y_)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=y_,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    """
    Trains neural network
    """
    def train(self, x, y):
        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': x}, y=y,
            batch_size=self.hp.batch_size, num_epochs=None, shuffle=True)

        # Train the Model
        self.model.train(input_fn, steps=self.hp.num_steps)

    """
    Evaluates the trained model against training set
    """
    def evaluate(self, x, y):
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': x}, y=y,
            batch_size=self.hp.batch_size, shuffle=False)

        # Use the Estimator 'evaluate' method
        e = self.model.evaluate(input_fn)

        return e
