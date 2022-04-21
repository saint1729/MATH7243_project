import os.path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from bayes_opt import BayesianOptimization


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


def get_training_and_validation_data(batch_size, file_dir, seed):
    training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        file_dir, image_size=(512, 512), batch_size=batch_size, validation_split=0.2,
        subset="training", seed=seed, label_mode="binary")
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        file_dir, image_size=(512, 512), batch_size=batch_size, validation_split=0.2,
        subset="validation", seed=seed, label_mode="binary")
    return training_dataset, validation_dataset


def create_model(class_names):
    n_classes = len(class_names)
    n_classes = 1 if n_classes == 2 else n_classes
    base_model = tf.keras.applications.xception.Xception(weights=None, include_top=False)
    # for layer in base_model.layers:
    #     layer.trainable = False
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(n_classes, activation="sigmoid")(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model


def train_and_validate(class_weight, model, training_dataset, validation_dataset, num_epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.AUC(),
                                                                            tf.keras.metrics.TruePositives(),
                                                                            tf.keras.metrics.TrueNegatives(),
                                                                            tf.keras.metrics.FalsePositives(),
                                                                            tf.keras.metrics.FalseNegatives(),
                                                                            tf.keras.metrics.Precision(),
                                                                            tf.keras.metrics.Recall()])
    history = model.fit(training_dataset, epochs=num_epochs, validation_data=validation_dataset,
                        class_weight=class_weight)
    return history


def do_it(file_dir):
    seed = 42
    batch_size = 64

    training_dataset, validation_dataset = get_training_and_validation_data(batch_size, file_dir, seed)

    class_names = training_dataset.class_names

    p, n = 0, 0
    for x, y in training_dataset:
        p += np.sum(y == 1)
        n += np.sum(y == 0)

    print(n + p)
    weight_for_0 = (1 / n) * ((n + p) / 2.0)
    weight_for_1 = (1 / p) * ((n + p) / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(class_weight)

    training_dataset = training_dataset.map(preprocess)

    if os.path.exists("saved_model/xception_model"):
        model = tf.keras.models.load_model("saved_model/xception_model")
    else:
        model = create_model(class_names)

    # history = train_and_validate(class_weight, model, training_dataset, validation_dataset, 20)
    history = train_and_validate(class_weight, model, training_dataset, validation_dataset, 5)
    model.save("saved_model/xception_model")

    plt.figure(figsize=(10, 8))
    plt.plot(history.history['val_false_positives'])
    plt.plot(history.history['val_false_negatives'])
    plt.show()


if __name__ == "__main__":
    directory = "/Users/sakshisuman12/Desktop/MATH7243_project/unzipped/XN_project/renders_small/"

    do_it(directory)
