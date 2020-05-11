from __future__ import absolute_import, division, print_function

from datetime import datetime

import tensorflow as tf
from tensorflow import keras

# Define model
i = keras.layers.Input(shape=(28, 28))
x = keras.layers.Flatten()(i)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(i, x)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

(train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0

# Define the Keras TensorBoard callback.
logdir = ".\\logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model.
model.fit(
    train_images,
    train_labels,
    batch_size=64,
    epochs=5,
    callbacks=[tensorboard_callback])

# Graph from tf.Graph

# The function to be traced.
@tf.function
def my_func(x, y):
    # A simple hand-rolled layer.
    return tf.nn.relu(tf.matmul(x, y))


# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f'.\\logs\\func\\{stamp}'
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = my_func(x, y)
with writer.as_default():
    tf.summary.trace_export(
        name='my_func_trace',
        step=0,
        profiler_outdir=logdir
    )
