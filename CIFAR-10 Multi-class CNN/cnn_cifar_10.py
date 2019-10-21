# -*- coding: utf-8 -*-
"""
CNN Implementation on CIFAR-10 [ Source: Kaggle (https://www.cs.toronto.edu/~kriz/cifar.html) ] dataset using TensorFlow 2.0.0 Sequential API on CPU.
Training dataset contains 50,000 image/records & Testing dataset contains additional 10,000 records. Dataset has 10 different label/classes of 32*32*3 4-dimensional color images for Classification (Object Recognition).
Model attains around 84.13% accuracy on Training dataset, whereas succumbs to 73.32% accuracy on Testing dataset.
"""

# Importing external dependencies:
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


# Loading Dataset:
class_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']  #Setting class/label names to be predicted.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  #'y_train' & 'y_test' have labels.


# Pre-processing training data (Normalizing to [0,1] by dividing by max pixels[255] + Converting to Vector format by reshaping) for faster processing:
X_train = X_train/255.0
X_test = X_test/255.0

#plt.imshow(X_test[10])  #Visualizing image


# Compiling architechture of fully-connected CNN model:
cnn = tf.keras.models.Sequential()  #Initializing our model.

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32,32,3]))  #First Layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))  #Second layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))  #Second layer image size reduction retaining ONLY critical features, hence 'valid' padding. Alternate is 'AveragePool2D'.

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))  #Third layer

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))  #Fourth layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))  #Second layer image size reduction retaining features.

cnn.add(tf.keras.layers.Flatten())  #Flattening Layer

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))  #Adding first fully-connected layer.

cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))  #Adding second dense layer (Output layer). For Binary classification, 'units=1'

cnn.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])  #Suitable for 2+ label/classes, else loss='binary_crossentropy' & metrics=['accuracy'].


# Overview of model architecture:
print(cnn.summary())


# Training & Evaluation of our model:
cnn.fit(X_train, y_train, epochs=50, batch_size=64)
test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
print(f"Test data Accuracy: {test_accuracy}")


# Saving our Model architecture & network weights:
with open('cifar_10_model.json', 'w') as f:
    f.write(cnn.to_json())

cnn.save_weights('cifar_10_model_weights.h5')
