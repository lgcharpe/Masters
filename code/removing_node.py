from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import copy
import tqdm
import IProgress
from hfunc import models
from hfunc import metrics

fashion_mnist = tf.keras.datasets.fashion_mnist
class_accuracy = metrics.ClassAccuracy()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Converting interger values to floats (0 to 1)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

tester_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
    
tester_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

model.fit(x_train, y_train, epochs=5)

