#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

import keras
from tensorflow.keras import layers, Model
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import RMSprop

from cags_dataset import CAGS
import efficient_net

def comp_iou(y_true, y_pred):
    return tf.keras.metrics.MeanIoU(2)(y_true*1, tf.greater_equal(y_pred, 0.5))


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    train = cags.train.map(CAGS.parse)
    dev = cags.dev.map(CAGS.parse)

    x_train, y_train = [], []
    x_dev, y_dev = [], []
    for x in train:
        x_train.append((x['image'] * x['mask']).numpy())
        y_train.append(tf.keras.utils.to_categorical(x['label'].numpy(), num_classes=len(CAGS.LABELS)))

    for x in dev:
        x_dev.append(x['image'].numpy())
        y_dev.append(tf.keras.utils.to_categorical(x['label'].numpy(), num_classes=len(CAGS.LABELS)))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

    datagen.fit(x_train)


    # TODO: Create the model and train it
    efficientnet_b0.trainable = False

    outputs = layers.Flatten()(efficientnet_b0.outputs[-1])
    outputs = layers.Dense(len(CAGS.LABELS), activation='softmax')(outputs)

    model = Model(inputs=efficientnet_b0.inputs, outputs=outputs)

    opt = RMSprop(lr=0.001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=[tf.keras.metrics.MeanIoU(num_classes=len(CAGS.LABELS))])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=x_train.shape[0] // args.batch_size,
                        epochs=125,
                        verbose = 1, validation_data=(x_dev, y_dev), callbacks=[LearningRateScheduler(lr_schedule)])

    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

    test = cags.test.map(CAGS.parse)
    x_test = []
    for x in test:
        x_test.append(x['image'])

    x_test = np.array(x_test)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(x_test, batch_size=args.batch_size)
        for probs in test_probabilities:
            print(np.argmax(probs), file=out_file)
