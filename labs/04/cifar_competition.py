#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import typing
from typing import Dict
# Team: 7797f596-9326-11ec-986f-f39926f24a9c, 449dba85-9adb-11ec-986f-f39926f24a9c

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.add_dll_directory("C:/Program Files Custom/zlib123dllx64/dll_x64")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/extras/CUPTI/lib64")

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k
# fix for intellisense as in here: https://github.com/tensorflow/tensorflow/issues/53144

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
parser.add_argument("--cnn",
                    default="models/model_def_01.txt",
                    type=str, help="Name of file containing CNN architecture definition.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


# The neural network model
class Model(k.Model):
    def __init__(self, args: argparse.Namespace) -> None:

        inputs = k.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        with open(args.cnn, 'r') as file:
            # Change [ and ] residual layer markers to specific start and end "layers" (..., R, layers, RE, ...)
            arguments = file.read().replace("-[", ",").replace("]", ",RE").replace("\n", "")
        # Split by commas
        layer_specifications = arguments.split(",")
        # Produce the results in the variable `x`.
        x = inputs  # We will pass this one changing variable between layers
        # Init helper variables for residual blocks
        creating_residual = False
        add_relu = False
        res_in = []
        # Iterate over comma separated specs
        for spec in layer_specifications:
            # Split arguments by dash
            arg = spec.split("-")
            # print(arg)
            if add_relu is True and arg[0] != "RE":
                x = k.layers.Activation('relu')(x)  # alternatively ReLu()
                add_relu = False
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
                    activation = None
                    use_bias = True
                    add_relu = True

                x = k.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation, use_bias=use_bias, kernel_initializer="he_uniform")(x)
                print(
                    f"Conv2D\n\tfilters = {filters}\n\tkernel_size = {kernel_size}\n\tstrides = {strides}\n\tpadding ="
                    f" {padding}\n\tactivation = {activation}\n\tuse_bias = {use_bias}")
                # Batch normalization
                if arg[0] == "CB":
                    x = k.layers.BatchNormalization()(x)
                    print("BatchNormalization")
                    # x = k.layers.Activation('relu')(x)  # alternatively ReLu()
                    add_relu = True
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
                res_in = tf.identity(x)  # Save the current input to be copied
                print("Residual start")
            # End residual block
            elif arg[0] == "RE":
                # Mix the saved input with current layer
                x = k.layers.Add()([res_in, x])
                if add_relu is True:
                    x = k.layers.Activation('relu')(x)  # alternatively ReLu()
                    add_relu = False
                creating_residual = False
                print("Residual end")
            else:
                print(f"Unknown arg '{arg[0]}'")

        # Add the final output layer
        outputs = k.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax, name="output_layer")(x)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = k.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), val) for key, val in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # TODO: Create the model and train it
    model = Model(args)

    train_generator = k.preprocessing.image.ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                                                               height_shift_range=0.1, zoom_range=0.2,
                                                               horizontal_flip=True)

    logs = model.fit(
        train_generator.flow(x=cifar.train.data["images"], y=cifar.train.data["labels"], batch_size=args.batch_size,
                             seed=args.seed),
        shuffle=False, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)

    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
