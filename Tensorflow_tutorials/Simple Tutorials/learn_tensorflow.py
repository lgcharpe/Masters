from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
# Install TensorFlow

import tensorflow as tf

# Load MNIST Dataset from the tensorflow datasets
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Converting interger values to floats (0 to 1)

# Building the NN model (in this case a simple ANN)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Setting up the optimizer and loss

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model

model.fit(x_train, y_train, epochs=5)

# Evaluating the model

model.evaluate(x_test, y_test, verbose=2)
