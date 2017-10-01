"""
Trains neural network using tensorflow
"""

import tensorflow as tf
from GradientDescentAlgorithmsComparison.HyperParameters import HyperParameters


class NeuralNetwork:
    hp = HyperParameters()

    # Define the model function (following TF Estimator Template)
    def model_fn(self, features, labels, mode):
        # Build the neural network

        # TF Estimator input is a dict, in case of multiple inputs
        x = features['images']
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.layers.dense(x, self.hp.n_hidden_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.layers.dense(layer_1, self.hp.n_hidden_2)
        # Output fully connected layer with a neuron for each class
        logits = tf.layers.dense(layer_2, self.hp.num_classes)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.hp.learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    def evaluate(self, x, y):
        # Build the Estimator
        model = tf.estimator.Estimator(self.model_fn)

        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': x}, y=y,
            batch_size=self.hp.batch_size, num_epochs=None, shuffle=True)
        # Train the Model
        model.train(input_fn, steps=self.hp.num_steps)

        # Evaluate the Model
        # Define the input function for evaluating
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': x}, y=y,
            batch_size=self.hp.batch_size, shuffle=False)
        # Use the Estimator 'evaluate' method
        e = model.evaluate(input_fn)

        return e
