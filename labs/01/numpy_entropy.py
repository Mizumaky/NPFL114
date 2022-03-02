#!/usr/bin/env python3
import argparse
from typing import List, Dict, Tuple
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> Tuple[float, float, float]:
    data_dist: Dict[str, int] = {}  # mapping of symbols to their count
    n: int = 0  # size of input data
    # Load data distribution
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            if line in data_dist:
                data_dist[line] += 1
            else:
                data_dist[line] = 1
            n += 1
    k: int = len(data_dist)  # number of different input symbols
    # Create NumPy array containing the data distribution.
    np_data = np.empty(k, dtype=float)  # np array for symbol probabilities (unordered)
    # Fill np array with data
    for i, val in enumerate(data_dist.values()):
        np_data[i] = val/n

    # Load model distribution, each line `string \t probability`.
    model_dist: Dict[str, float] = {}
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, val = line.split()
            model_dist[key] = float(val)

    # NumPy array containing the model distribution
    # np_model = np.fromiter(model_dist.values(), dtype=float)  # np array for symbol probabilities (unordered - PROBLEM)
    np_model = np.empty(k, dtype=float)  # np array for symbol probabilities (ordered to correspond with np_data)
    missing_symbol = False
    for i, key in enumerate(data_dist.keys()):
        if key in model_dist:
            np_model[i] = model_dist[key]
        else:
            # we have a data symbol, which is not in model symbols, so we cannot calculate crossentropy
            missing_symbol = True
            break

    # The reason np.array(Samples.values()) doesn't give what you expect in Python 3 is that in Python 3,
    # the values() method of a dict returns an iterable view, whereas in Python 2,
    # it returns an actual list of the keys. (so one cannot np.array(list(Samples.values())))

    # Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -(np_data * np.log(np_data)).sum()  # bonus - computational time reference https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python

    # Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    if not missing_symbol:
        crossentropy = -(np_data * np.log(np_model)).sum()
    else:
        crossentropy = np.inf

    # Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
