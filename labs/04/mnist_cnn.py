#!/usr/bin/env python3
# 7797f596-9326-11ec-986f-f39926f24a9c
# 449dba85-9adb-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
import re
import typing
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k
# fix for intellisense as in here: https://github.com/tensorflow/tensorflow/issues/53144

from mnist import MNIST

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.3/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/extras/CUPTI/lib64")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The neural network model
class Model(k.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = k.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.

        # Change [ and ] residual layer markers to specific start and end "layers" (..., R, layers, RE, ...)
        arguments = args.cnn.replace("-[", ",").replace("]", ",RE")
        # Split by commas
        layer_specifications = arguments.split(",")
        # Produce the results in the variable `x`.
        x = inputs  # We will pass this one changing variable between layers
        # Init helper variables for residual blocks
        creating_residual = False
        res_in = []
        # Iterate over comma separated specs
        for spec in layer_specifications:
            # Split arguments by dash
            arg = spec.split("-")
            print(arg)
            # Convolutional layer
            if arg[0] == "C" or arg[0] == "CB":
                filters = int(arg[1])
                kernel_size = int(arg[2])
                strides = int(arg[3])
                padding = arg[4]
                # With batch normalization
                if arg[0] == "CB":
                    activation = None
                    use_bias = False
                else:
                    activation = "relu"
                    use_bias = True
                x = k.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation, use_bias=use_bias)(x)
                print(f"Conv2D\n\tfilters = {filters}\n\tkernel_size = {kernel_size}\n\tstrides = {strides}\n\tpadding = {padding}\n\tactivation = {activation}\n\tuse_bias = {use_bias}")
                # Batch normalization
                if arg[0] == "CB":
                    x = k.layers.BatchNormalization()(x)
                    print("BatchNormalization")
                    x = k.layers.Activation('relu')(x)  # alternatively ReLu()
                    print("Activation - relu")
            # Max pooling layer
            elif arg[0] == "M":
                pool_size = eval(arg[1])
                strides = eval(arg[2])
                padding = "valid"
                x = k.layers.MaxPooling2D(pool_size, strides, padding=padding)(x)
                print(f"MaxPooling2D")
            # Flatten layer
            elif arg[0] == "F":
                x = k.layers.Flatten()(x)
                print("Flatten")
            # Dense layer
            elif arg[0] == "H":
                units = int(arg[1])
                x = k.layers.Dense(units, activation="relu")(x)
                print(f"Dense\n\tunits = {units}")
            # Dropout layer
            elif arg[0] == "D":
                rate = float(arg[1])
                x = k.layers.Dropout(rate)(x)
                print(f"Dropout\n\tunits = {rate}")
            # Start residual block
            elif arg[0] == "R":
                creating_residual = True
                res_in = x  # Save the current input to be copied
                print("Residual start")
            # End residual block
            elif arg[0] == "RE":
                # Mix the saved input with current layer
                x = k.layers.Add()([res_in, x])
                creating_residual = False
                print("Residual end")
            else:
                print(f"Unknown arg '{arg[0]}'")

        # Add the final output layer
        outputs = k.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax, name="output_layer")(x)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = k.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    k.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)
    model.summary()

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
