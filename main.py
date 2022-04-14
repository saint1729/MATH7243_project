import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# importing os module
import os


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


if __name__ == "__main__":
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    # User Specific Data Location
    if (os.getlogin() == 'ramse'):
        file_dir = "C:/Users/ramse/Documents/School/Northeastern/Machine Learning/Final Project/max_contrast_renders/"
    else:
        file_dir = "/Users/sakshisuman12/Desktop/MATH7243_project/unzipped/XN_project/renders_small/"

    # All folders (must be in alphabetical order for keras loading function to work correctly)
    # QUESTION: How are we sorting/classifying multi? Should we manually move those files into 
    # other folders based on labels in the csv?
    label_types = ['epidural', 'intraparenchymal', 'intraventricular', 'multi', 'normal', 'subarachnoid', 'subdural']
    
    # Which class we are isolating with this binary classifier
    binary_classifier_type = 'subdural'

    labelList = []
    for ii in range(len(label_types)):
        num_samples = 0
        for r, d, files in os.walk(file_dir + label_types[ii]):
            num_samples = len(files) 
        if label_types[ii]==binary_classifier_type:
            binaryLabel = 1
        else:
            binaryLabel = 0
        listToAdd = [binaryLabel] * num_samples
        labelList = labelList + listToAdd
    print(len(labelList))

    # https://stackoverflow.com/questions/52887227/understanding-follow-links-argument-in-kerass-imagedatagenerator
    training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        file_dir, labels = labelList, image_size=(512, 512), batch_size=64, validation_split=0.2,
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
