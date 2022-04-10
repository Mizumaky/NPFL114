#!/usr/bin/env python3
import argparse
import datetime
import os

# Team:
# 7797f596-9326-11ec-986f-f39926f24a9c
# 449dba85-9adb-11ec-986f-f39926f24a9c
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
parser.add_argument("--model_name", default="v10_best_plus_label", type=str, help="Name of the model to train/test")
parser.add_argument("--batch_size", default=35, type=int, help="Batch size.")  # TODO
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
parser.add_argument("--training", default=True, type=bool, help="Whether to train the model")
parser.add_argument("--fine_tuning", default=True, type=bool, help="Whether to fine_tune the model")
parser.add_argument("--fine_tuning_top", default=False, type=bool, help="Whether to fine_tune top of the model")

# To achieve best results, first run it training True with 100 batch_size, lr=0.0003
# Then run it again with fine_tuning also True (ran with only 35 batch_size since OOM errors on my machine). lr=0.0001

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
    fine_tuning = args.fine_tuning
    fine_tuning_top = args.fine_tuning_top  # currently not in use

    # Fix random seeds and threads
    # with tf.device("cpu"):
    #     generator = tf.random.Generator.from_seed(args.seed)
    k.utils.set_random_seed(args.seed)
    # tf.config.threading.set_inter_op_parallelism_threads(args.threads)  # DONT FORGET TO UNCOMMENT
    # tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # debugging
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), v) for key, v in sorted(vars(args).items())))
    ))

    # Load the data + data augmentation
    print("Loading data...")
    cags = CAGS()
    # Prepare data augmentation
    train_generator = k.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2, rotation_range=15,  # shear_range=0.05,
        horizontal_flip=True, width_shift_range=0.15, height_shift_range=0.15,
        brightness_range=[0.88, 1.12],
        fill_mode='nearest',
    )

    def augment(image, label):
        ret = tf.ensure_shape(
            tf.numpy_function(train_generator.random_transform, [image], tf.float32),
            image.shape
        )
        # fix
        ret = tf.math.scalar_mul(1. / 255, ret)
        return ret, label

    # Training data
    if training:
        # Pouziti train, dev a test jakozto tf.Datasety, co obsahuji "image", "label" a "mask"
        # cags.train
        train = cags.train.map(
            lambda example: (example["image"], tf.one_hot(example["label"], len(CAGS.LABELS))))  # nutnost zahodit mask
        # shuffle
        train = train.shuffle(2200, seed=args.seed)  # TODO: possibly remove seed
        # possibly repeat - so that an epoch contains copies of augmented image
        # train = train.repeat(2)
        # create augmented copies of each element
        train = train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        # example = train.take(1)
        # for img in example:
        #     print(img)
        print(f"train size: {len(train)}")
        # batch
        train = train.batch(args.batch_size)
        # allow the pipeline to run in parallel with the training process
        # dynamically adjusting the number of threads to fully saturate the training process
        train = train.prefetch(tf.data.AUTOTUNE)
        # visually demonstrate training set
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

    # cags.dev
    dev = cags.dev.map(lambda example: (example["image"], tf.one_hot(example["label"], len(CAGS.LABELS))))
    # dev = cags.dev.map(lambda example: (example["image"], example["label"]))
    # dev_orig = cags.dev.map(lambda example: (example["image"], example["label"]))
    # dev_aug = cags.dev.map(lambda example: (example["image"], example["label"]))
    # dev_aug = dev_aug.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    # dev = dev_orig.concatenate(dev_aug)
    print(f"dev size: {len(dev)}")
    dev = dev.batch(args.batch_size)

    # cags.test
    test = cags.test.map(lambda example: (example["image"], tf.one_hot(example["label"], len(CAGS.LABELS))))
    print(f"test size: {len(test)}")
    test = test.batch(args.batch_size)

    # Create and train / get model
    if training:
        if not (fine_tuning or fine_tuning_top):
            print("Creating model...")
            # Load the EfficientNet-B0 model
            # include_top - means include resulting probabilities or not (else, return last feature layer)
            efficientnet_b0: k.Model = efficient_net.pretrained_efficientnet_b0(include_top=False,
                                                                                dynamic_input_shape=False)
            efficientnet_b0.trainable = False  # freeze the model
            # Create the model
            inputs = k.Input(shape=(CAGS.W, CAGS.H, CAGS.C))
            # We make sure that the base model is running in inference mode here,
            # by passing `training=False`. This is important for fine-tuning.
            x = efficientnet_b0(inputs, training=False)[0]  # Get only last layer from the outputs (first since reverse)
            # We get a 1280 neuron layer
            # Add another hidden layer with dropout
            x = k.layers.Dropout(rate=0.2)(x)
            x = k.layers.Dense(1280, activation=k.activations.swish,
                               # kernel_regularizer=k.regularizers.l2(0.0001),
                               # bias_regularizer=k.regularizers.l2(0.0001),
                               name="pridany_layer")(x)
            x = k.layers.Dropout(rate=0.5)(x)
            # A Dense classifier with softmax for categorical classification
            outputs = k.layers.Dense(len(CAGS.LABELS), activation=k.activations.softmax, name="vyhodnoceni_cags")(x)
            model = k.Model(inputs, outputs)
            learning_rate = 0.0003
        else:
            print("Loading model...")
            model = k.models.load_model("./models/" + model_name)
            model.trainable = True
            # if fine_tuning_top:
            #     to_lock = 220
            print("Locking batch norms...")
            for layer in model.layers[-5].layers:
                # tf.print(layer.output_shape)
                if isinstance(layer, k.layers.BatchNormalization):
                    layer.trainable = False
                    # tf.print("batch")
                    # elif to_lock > 0:
                    #     layer.trainable = False
                    #     tf.print("locked")
                    #     to_lock -= 1
            learning_rate = 0.0001
        model.summary()

        # Custom metrics for checkpoint creation (validation data is small, so accuracy is not that accurate)
        def accuracy_loss_mix(y_true, y_pred):
            acc = k.metrics.categorical_accuracy(y_true, y_pred)
            loss = k.metrics.categorical_crossentropy(y_true, y_pred)  # label_smoothing=0.05
            return acc + (1 - loss)

        # Prepare model for training
        print("Compiling model...")
        model.compile(optimizer=k.optimizers.Adam(learning_rate=learning_rate),
                      loss=k.losses.CategoricalCrossentropy(label_smoothing=0.05),
                      metrics=[k.metrics.CategoricalAccuracy("accuracy"), accuracy_loss_mix])

        # Prepare TensorBoard callback
        tb_callback = k.callbacks.TensorBoard(args.logdir, histogram_freq=1)
        tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.

        # Define early stopping and checkpoint creation
        checkpoint_filepath = './tmp/checkpoint'
        es_callback = k.callbacks.EarlyStopping(monitor='val_accuracy_loss_mix', mode='max', verbose=1, patience=25)
        mc_callback = k.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                  save_weights_only=True, monitor='val_accuracy_loss_mix', mode='max',
                                                  verbose=1, save_best_only=True)

        # Train the model
        print("Starting training...")
        logs = model.fit(
            train,
            epochs=args.epochs,
            callbacks=[tb_callback, es_callback, mc_callback],
            validation_data=dev
        )

        # Load best checkpoint weights
        model.load_weights(checkpoint_filepath)

        # Save the model
        print("Saving model...")
        if fine_tuning:
            model.save("./models/" + model_name + "_fine_tuned_2")
        elif fine_tuning_top:
            model.save("./models/" + model_name + "_fine_tuned_top")
        else:
            model.save("./models/" + model_name)
    else:
        print("Loading model...")
        model = k.models.load_model("./models/" + model_name)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    result = model.evaluate(dev, return_dict=True)
    for metric, value in result.items():
        print(f"{metric}: {value}")
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


# some resources on dataset + imagedatagenerator
# https://stackoverflow.com/questions/59648804/how-can-i-combine-imagedatagenerator-with-tensorflow-datasets-in-tf2
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
