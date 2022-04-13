import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


if __name__ == "__main__":
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    file_dir = "/Users/sakshisuman12/Desktop/MATH7243_project/unzipped/XN_project/renders_small/"

    training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        file_dir, image_size=(512, 512), batch_size=64, validation_split=0.2,
        subset="training", seed=123, label_mode="binary",
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        file_dir, image_size=(512, 512), batch_size=64, validation_split=0.2,
        subset="validation", seed=123, label_mode="binary",
    )

    class_names = training_dataset.class_names

    class_weight = {0: 1 / 10000, 1: 1 / 1600}

    batch_size = 64
    training_dataset = training_dataset.map(preprocess)

    n_classes = len(class_names)
    n_classes = 1 if n_classes == 2 else n_classes

    base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(n_classes, activation="sigmoid")(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.summary()

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[tf.keras.metrics.AUC(),
                                                                            tf.keras.metrics.FalsePositives(),
                                                                            tf.keras.metrics.FalseNegatives(),
                                                                            tf.keras.metrics.Precision(),
                                                                            tf.keras.metrics.Recall()])
    history = model.fit(training_dataset, epochs=5, validation_data=validation_dataset, class_weight=class_weight)

    plt.figure(figsize=(10, 8))
    plt.plot(history.history['val_false_positives'])
    plt.plot(history.history['val_false_negatives'])
    plt.show()
