# -*- coding: utf-8 -*-
"""
ANN Implementation on Fashion MNIST [ Source: Kaggle (https://www.kaggle.com/zalando-research/fashionmnist) ] using TensorFlow 2.0.0 on CPU.
Training dataset contains 60,000 image/records & Testing dataset contains additional 10,000 records. Dataset has 10 different label/classes of 28*28 images for Classification.
Model attains around 94.13% accuracy on Training dataset, whereas succumbs to 89.87% accuracy on Testing dataset.
"""

# Importing external dependencies:
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


# Loading Dataset:
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# Pre-processing training data (Normalizing to [0,1] by dividing by max pixels[255] + Converting to Vector format by reshaping) for faster processing:
X_train = X_train/255.0
X_test = X_test/255.0

X_train = X_train.reshape(-1, 28*28)  #Images in our data are in 28*28 shape
X_test = X_test.reshape(-1, 28*28)


# Compiling fully-connected ANN Model:
ann = tf.keras.models.Sequential()  #Initializing our model.

ann.add(tf.keras.layers.Dense(units=256, activation='relu', input_shape=(784,)))  #First Layer
ann.add(tf.keras.layers.Dropout(0.25))  #First layer regularization to avoid overfitting during backpropagation

ann.add(tf.keras.layers.Dense(units=128, activation='relu'))  #Second layer
ann.add(tf.keras.layers.Dropout(0.20))  #Second layer regularization

ann.add(tf.keras.layers.Dense(units=64, activation='relu'))  #Third layer

ann.add(tf.keras.layers.Dense(units=10, activation='softmax'))  #Final layer with units representing our num of label/classes to be predicted

ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])  #Suitable for 2+ classes


# Overview of model architecture:
print(ann.summary())


# Training & Evaluation of our model:
ann.fit(X_train, y_train, epochs=80)
test_loss, test_accuracy = ann.evaluate(X_test, y_test)
print(f"Test data Accuracy: {test_accuracy}")


# Saving our Model architecture & network weights:
with open('fashion_mnist_ann.json', 'w') as json_file:
    json_file.write(ann.to_json())

ann.save_weights('fashion_mnist_ann_weights.h5')
