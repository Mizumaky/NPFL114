import argparse
import datetime
import os
import re
from typing import Dict
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import typing
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k
# fix for intellisense as in here: https://github.com/tensorflow/tensorflow/issues/53144

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> None:
    dataset = tf.data.Dataset.from_tensor_slices((["1", "2", "3", "4", "5", "6", "7", "8"], [1, 2, 3, 4, 5, 6, 7, 8]))
    dataset = dataset.batch(2)
    print(dataset.cardinality())
    for entry in dataset:
        print(entry)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)