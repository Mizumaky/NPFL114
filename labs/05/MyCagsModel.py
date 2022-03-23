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
from cags_dataset import CAGS

# The neural network model
class MyCagsModel(k.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Example arguments
        inputs = k.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])

        # Prepare regularizer?
        L2_reg = tf.keras.regularizers.L1L2(
            l1=0.0, l2=0.01
        )
        L1L2_reg = tf.keras.regularizers.L1L2(
            l1=0.01, l2=0.01
        )
        L1_reg = tf.keras.regularizers.L1L2(
            l1=0.01, l2=0.0
        )

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
        outputs = k.layers.Dense(CAGS.LABELS, activation=tf.nn.softmax, name="output_layer")(x)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=k.optimizers.Adam(learning_rate=k.optimizers.schedules.CosineDecay()),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = k.callbacks.TensorBoard(args.logdir)