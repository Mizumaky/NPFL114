#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from typing import List, Tuple
from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Model(tf.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        # Create trainable variables (init to vectors with random values according to normal distribution / zeros)
        #   From input to first layer
        self._W1 = tf.Variable(
            tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed),
            trainable=True
        )
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)
        #   From first layer to output
        self._W2 = tf.Variable(
            tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed),
            trainable=True
        )
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs: tf.Tensor) -> tf.Tensor:
        # Define the computation of the network with one hidden layer
        # - start by reshaping the inputs to shape [inputs.shape[0], -1].
        #   The -1 is a wildcard which is computed so that the number of elements before and after the reshape fits
        #   This does flatten the vectors into self._args.batch_size x MNIST.W * MNIST.H * MNIST.C
        #   (it's a matrix since multiple input vectors per batch)
        shaped_inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        # - multiply the inputs by wights `self._W1` and add bias `self._b1`
        hidden_layer_in = tf.linalg.matmul(shaped_inputs, self._W1)
        hidden_layer_in += self._b1
        # - apply activation `tf.nn.tanh`
        hidden_layer_out = tf.nn.tanh(hidden_layer_in)
        # - multiply the result by `self._W2` and then add `self._b2`
        output_layer_in = tf.linalg.matmul(hidden_layer_out, self._W2)
        output_layer_in += self._b2
        # - finally apply `tf.nn.softmax` and return the result
        output_layer_out = tf.nn.softmax(output_layer_in)
        # Great demo of softmax https://www.youtube.com/watch?v=ytbYRIN0N4g ✨
        return output_layer_out

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            # The tf.GradientTape is used to record all operations inside the with block.
            with tf.GradientTape() as tape:
                # Compute the predicted probabilities of the batch images using `self.predict`
                probabilities = self.predict(tf.convert_to_tensor(batch["images"]))
                #            Labels:    car0 dog1 cat2
                # Example:   picture0 [ 0.2  0.3  0.5
                #            picture1   0.4  0.2  0.4
                #            picture2   0.1  0.1  0.8 ]
                # Get actual labels
                labels = tf.convert_to_tensor(batch["labels"])
                # Example: [ dog1, dog1, cat2 ]
                # Prepare labels for computation
                # labels = tf.one_hot(labels)
                #                      car0 dog1 cat2
                # Example:   picture0 [ 0    0    1
                #            picture1   1    0    0
                #            picture2   0    0    1 ]

                # Manually compute the loss:
                # - For every batch example, the loss is the categorical crossentropy of the
                #   predicted probabilities and the gold label. To compute the crossentropy, you can
                #   - either use `tf.one_hot` to obtain one-hot encoded gold labels,
                #   - or use `tf.gather` with `batch_dims=1` to "index" the predicted probabilities.
                # - Finally, compute the average across the batch examples.
                # Additional reference for manual computation:
                # https://stackoverflow.com/questions/58159154/how-to-calculate-categorical-cross-entropy-by-hand
                scce = tf.keras.losses.SparseCategoricalCrossentropy()
                loss = scce(labels, probabilities)

            # We create a list of all variables. Note that a `tf.Module` automatically
            # tracks owned variables, so we could also used `self.trainable_variables`
            # (or even `self.variables`, which is useful for loading/saving).
            # variables = [self._W1, self._b1, self._W2, self._b2]
            variables: Tuple[tf.Variable] = self.trainable_variables

            # Compute the gradient of the loss with respect to variables using
            # backpropagation algorithm via `tape.gradient`
            gradients = tape.gradient(loss, variables)

            for variable, gradient in zip(variables, gradients):
                # Perform the SGD update with learning rate `self._args.learning_rate`
                # for the variable and computed gradient. You can modify
                # variable value with `variable.assign` or in this case the more
                # efficient `variable.assign_sub`.
                variable.assign_sub(self._args.learning_rate * gradient)

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        # Compute the accuracy of the model prediction
        correct: int = 0
        for batch in dataset.batches(self._args.batch_size):
            # Compute the probabilities of the batch images
            probabilities = self.predict(tf.convert_to_tensor(batch["images"]))

            # Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            # Find max probability for each sample, compare it with sample's label, sum correct samples
            correct += tf.math.count_nonzero(tf.math.equal(tf.math.argmax(probabilities, axis=1), batch["labels"]))

        return correct / dataset.size


def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(mnist.train)
        # Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        # calculate test loss
        probabilities = model.predict(tf.convert_to_tensor(mnist.dev.data["images"]))
        labels = tf.convert_to_tensor(mnist.dev.data["labels"])
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = scce(labels, probabilities)
        print("Dev loss after epoch {} is {:.2f}".format(epoch + 1, loss), flush=True)

        with writer.as_default(step=epoch + 1):
            tf.summary.scalar("dev/accuracy", 100 * accuracy)
            tf.summary.scalar("dev/loss", loss)



    # Evaluate the test data using `evaluate` on `mnist.test` dataset
    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default(step=epoch + 1):
        tf.summary.scalar("test/accuracy", 100 * accuracy)

    # Return test accuracy for ReCodEx to validate
    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
