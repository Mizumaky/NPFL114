#!/usr/bin/env python3
import argparse
import datetime
import os
# fix for intellisense as in here: https://github.com/tensorflow/tensorflow/issues/53144
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # Report only TF errors by default
import re
import typing
import numpy as np
import tensorflow as tf
from tensorflow import keras as k
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k
from cags_dataset import CAGS
import efficient_net


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=10, type=int, help="Maximum number of threads to use.")
parser.add_argument("--model_name", default="v2", type=str, help="Name of the model to train/test")
parser.add_argument("--training", default=True, type=bool, help="Whether to train the model")

# Goal: practice transfer learning - pretrenovat efficientnet
# idealne si chci:
#   1) vzit layery modelu, freeznout, za jeho predposledni layer pridat trenovatelne nove layery
#      vysledny model pak pouzit ke klasifikaci mych pozadovanych veci
#   1b) alternativou je jen si udelat maly novy model, a vzit ten vystup z puvodniho, coz se nazyva
#       feature extraction, avsak to nam neumoznuje delat hezky data augmentation apod
# anebo nasledne lepe:
#   2) FineTuning
#      unfreeznu i puvodni layeru modelu finetunuju
#      mala learning rate, bacha na batch normy a tak
# Puvodni efficient net b0 tridi do 1000 trid ruznych veci
# Ja ho chci predelat na nas cags dataset o 34 tridach
def main(args: argparse.Namespace) -> None:
    # Set model name and if training
    model_name = args.model_name
    training = args.training

    # Fix random seeds and threads
    # with tf.device("cpu"):
    #     generator = tf.random.Generator.from_seed(args.seed)
    k.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), v) for key, v in sorted(vars(args).items())))
    ))

    # Load the data + data augmentation
    print("Loading data...")
    cags = CAGS()
    if training:
        # Pouziti train, dev a test jakozto tf.Datasety, co obsahuji "image", "label" a "mask"
        # cags.train
        train = cags.train.map(lambda example: (example["image"], example["label"]))  # nutnost zahodit mask
        print(f"train size: {len(train)}")  # podle toho vybrat shuffle buffer size
        train = train.shuffle(2000, seed=args.seed)
        train = train.batch(args.batch_size)
        train = train.prefetch(tf.data.AUTOTUNE)  # allows the pipeline to run in parallel with the training process
        # dynamically adjusting the number of threads to fully saturate the training process
    # cags.dev
    dev = cags.dev.map(lambda example: (example["image"], example["label"]))
    print(f"dev size: {len(dev)}")
    dev = dev.batch(args.batch_size)
    # cags.test
    test = cags.test.map(lambda example: (example["image"], example["label"]))
    print(f"test size: {len(test)}")
    test = test.batch(args.batch_size)
    # train_generator = k.preprocessing.image.ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
    #                                                            height_shift_range=0.1, zoom_range=0.2,
    #                                                            horizontal_flip=True)
    # + in fit:
    # train_generator.flow(x=cifar.train.data["images"], y=cifar.train.data["labels"], batch_size=args.batch_size,
    #                              seed=args.seed),

    # Create and train / get model
    if training:
        print("Creating model...")
        # Load the EfficientNet-B0 model
        # include_top - means include resulting probabilities or not (else, return last feature layer)
        efficientnet_b0: k.Model = efficient_net.pretrained_efficientnet_b0(include_top=False, dynamic_input_shape=False)
        efficientnet_b0.trainable = False  # freeze the model
        # Create the model
        inputs = k.Input(shape=(CAGS.W, CAGS.H, CAGS.C))
        # We make sure that the base model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning.
        x = efficientnet_b0(inputs, training=False)[0]  # Get only last layer from the outputs (first since reversed)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        # x = k.layers.GlobalAveragePooling2D()(x) <- dont need this since it's done in efficient_net
        # A Dense classifier with softmax for categorical classification
        outputs = k.layers.Dense(len(CAGS.LABELS), activation=k.activations.softmax, name="vyhodnoceni_cags")(x)
        model = k.Model(inputs, outputs)
        model.summary()

        # Prepare model for training
        print("Compiling model...")
        model.compile(optimizer=k.optimizers.Adam(),
                      loss=k.losses.SparseCategoricalCrossentropy(),
                      metrics=[k.metrics.SparseCategoricalAccuracy("accuracy")])

        # Prepare TensorBoard callback
        tb_callback = k.callbacks.TensorBoard(args.logdir, histogram_freq=1)
        tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.

        # Train the model
        print("Starting training...")
        logs = model.fit(
            train,
            epochs=args.epochs,
            callbacks=[tb_callback],
            validation_data=dev
        )
        # Save the model
        print("Saving model...")
        model.save("/models/" + model_name)
    else:
        print("Loading model...")
        model = k.models.load_model("/models/" + model_name)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    # # Evaluate on dev data
    # print("Evaluating model on dev...")
    # dev_logs = model.evaluate(dev, return_dict=True)
    # results = {"dev_" + metric: value for metric, value in dev_logs.items()}
    # print(results)
    # if training:
    #     tb_callback.on_epoch_end(args.epochs, results)
    # Save test probabilities to file
    print("Opening file...")
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict on test data
        print("Predicting probabilities for test...")
        # Predict the probabilities on the test set
        test_probabilities = model.predict(test)
        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)
    print("End")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
