#!/usr/bin/env python3
# 7797f596-9326-11ec-986f-f39926f24a9c
# 449dba85-9adb-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
import re
import typing
from typing import Dict, Tuple
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
parser.add_argument("--epochs", default=16, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.0005, type=int, help="Learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=int, help="Final learning rate.")
parser.add_argument("--fine_tuning", default=True, type=bool, help="If fine tuning.")
parser.add_argument("--name", default="seg_3", type=str, help="Model name")

# REFERENCE MATERIALS:
# https://catalog.ngc.nvidia.com/orgs/nvidia/resources/efficientnet_for_tensorflow2
# https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    fine_tuning = args.fine_tuning
    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    if not fine_tuning:
        print("Loading effnet...")
        # Load the EfficientNet-B0 model
        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
        efficientnet_b0.trainable = False

        # TODO: Create the model and train it
        kernel_sizes = np.array([[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]])
        filters = np.array([32, 24, 40, 112, 192, 1280])

        def conv_block(x, kernel_size, filterr):
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Conv2D(filterr, kernel_size, padding='same', activation=None)(x)
            # x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            # x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Conv2D(filterr, kernel_size, padding='same', activation=None)(x)
            # x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            return x

        # def skip_block(x, kernel_size, filterr):
        #     x = keras.layers.Conv2D(filterr, kernel_size, padding='same', activation = None)(x)
        #     x = keras.layers.BatchNormalization()(x)
        #     x = keras.layers.Activation('relu')(x)
        #     x = keras.layers.Dropout(0.5)(x)
        #     return x

        # input
        inputs = keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
        # efficient net
        efnt = efficientnet_b0(inputs)
        # fifth level
        fifth = keras.layers.Conv2D(filters[5], kernel_sizes[5], padding='same', activation='relu')(efnt[1])
        # fifth = conv_block(efnt[1], kernel_sizes[5], filters[5])  # top convolutional block - same as near top at efnt
        fifth = keras.layers.Conv2DTranspose(filters[4], [2, 2], strides=(2, 2))(fifth)  # upsampling
        # fourth level
        # skip4 = skip_block(efnt[2], kernel_sizes[4], filters[4]) #skip connection
        skip4 = efnt[2]
        fourth = keras.layers.Concatenate(axis=-1)([skip4, fifth])  # concatenation
        fourth = conv_block(fourth, kernel_sizes[4], filters[4])  # convblock
        fourth = keras.layers.Conv2DTranspose(filters[3], [2, 2], strides=(2, 2))(fourth)  # upsampling
        # third level
        # skip3 = skip_block(efnt[3], kernel_sizes[3], filters[3]) #skip connection
        skip3 = efnt[3]
        third = keras.layers.Concatenate(axis=-1)([skip3, fourth])  # concatenation
        third = conv_block(third, kernel_sizes[2], filters[2])  # convblock
        third = keras.layers.Conv2DTranspose(filters[2], [2, 2], strides=(2, 2))(third)  # upsampling
        # second level
        # skip2 = skip_block(efnt[4], kernel_sizes[2], filters[2]) #skip connection
        skip2 = efnt[4]
        second = keras.layers.Concatenate(axis=-1)([skip2, third])  # concatenation
        second = conv_block(second, kernel_sizes[1], filters[1])  # convblock
        second = keras.layers.Conv2DTranspose(filters[1], [2, 2], strides=(2, 2))(second)  # upsampling
        # first level
        # skip1 = skip_block(efnt[5], kernel_sizes[1], filters[1]) #skip connection
        skip1 = efnt[5]
        first = keras.layers.Concatenate(axis=-1)([skip1, second])  # concatenation
        first = conv_block(first, kernel_sizes[0], filters[0])  # convblock
        first = keras.layers.Conv2DTranspose(filters[0], [2, 2], strides=(2, 2))(first)  # upsampling
        # output layer
        outputs = keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(first)

        # sch = tf.optimizers.schedules.CosineDecay(initial_learning_rate=args.learning_rate,
        #                                                   decay_steps=n/args.batch_size*args.epochs)

        model = keras.Model(inputs=inputs, outputs=outputs)
    else:
        # use the model from previous training
        print("Loading model...")
        model = keras.models.load_model("./models/" + args.name, {"MaskIoUMetric": cags.MaskIoUMetric})
        model.layers[1].trainable = True  # unfreeze effnet
        # for layer in efficientnet_b0.layers:
        #     if isinstance(layer, keras.layers.BatchNormalization):
        #         layer.trainable = False
    model.summary()

    print("Compiling model...")
    lr = args.learning_rate_final if fine_tuning else args.learning_rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=cags.MaskIoUMetric(name="iou"),
    )
    tb_callback = keras.callbacks.TensorBoard(args.logdir)

    # Define early stopping and checkpoint creation
    checkpoint_filepath = './tmp/checkpoint'
    es_callback = keras.callbacks.EarlyStopping(monitor='val_iou', mode='max', min_delta=0.002, verbose=1, patience=25)
    mc_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                  save_weights_only=True, monitor='val_iou', mode='max',
                                                  verbose=1, save_best_only=True)

    # Training augmentation
    with tf.device("/cpu:0"):
        generator = tf.random.Generator.from_seed(args.seed)

    def train_augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 64, CAGS.W + 64)
        label = tf.image.resize_with_crop_or_pad(label, CAGS.H + 64, CAGS.W + 64)
        a = generator.uniform([], CAGS.H, CAGS.H + 64 + 1, dtype=tf.int32)
        b = generator.uniform([], CAGS.W, CAGS.W + 64 + 1, dtype=tf.int32)
        image = tf.image.resize(image, [a, b])
        label = tf.image.resize(label, [a, b])
        r = generator.uniform([], -0.1, 0.1, dtype=tf.float32)
        image = tfa.image.rotate(image, r, interpolation='bilinear')
        label = tfa.image.rotate(label, r, interpolation='bilinear')
        image = tf.image.random_brightness(image, 0.1)
        c = generator.uniform([], maxval=tf.shape(image)[0] - CAGS.H + 1, dtype=tf.int32)
        d = generator.uniform([], maxval=tf.shape(image)[1] - CAGS.W + 1, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(
            image, target_height=CAGS.H, target_width=CAGS.W, offset_height=c,
            offset_width=d)
        label = tf.image.crop_to_bounding_box(
            label, target_height=CAGS.H, target_width=CAGS.W,
            offset_height=c,
            offset_width=d)
        return image, label

    print("Preparing data...")
    n = tf.data.experimental.cardinality(cags.train).numpy()
    train = cags.train.map(lambda example: (example["image"], example["mask"]))
    train = train.shuffle(n, seed=args.seed)
    train = train.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.batch(args.batch_size)
    # visualize train data
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        print("showing plot")
        for i in range(min(25, args.batch_size)):
            ax = plt.subplot(5, 5, i + 1)
            img = images[i].numpy()
            # print(img)
            plt.imshow(img)
            # plt.title(CAGS.LABELS[labels[i]])
            plt.axis("off")
    plt.show()

    dev = cags.dev.map(lambda example: (example["image"], example["mask"]))
    dev = dev.batch(args.batch_size)

    test = cags.test.map(lambda example: (example["image"]))
    test = test.batch(args.batch_size)

    print("Start fitting...")
    logs = model.fit(train,
                     epochs=args.epochs,
                     validation_data=dev,
                     callbacks=[tb_callback, es_callback, mc_callback])

    # Load best checkpoint weights
    print("load best weights...")
    model.load_weights(checkpoint_filepath)

    # Save the model
    print("Saving model weights...")
    model.save("./models/" + args.name)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        test_masks = model.predict(test)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
