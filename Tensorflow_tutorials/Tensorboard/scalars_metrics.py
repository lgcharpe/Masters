from __future__ import absolute_import, division, print_function

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

import numpy as np

print("TensorFlow version:", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This code requires TensorFlow 2.0 or above"

data_size = 1000
# 80% of data is for training
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the ouput data
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size, ))

# Split data into train and test
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Specify log directory
log_dir = ".\\logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Create Tensorboard callback

tb_call = keras.callbacks.TensorBoard(log_dir=log_dir,
                                      histogram_freq=1)

# Build model
# i = keras.layers.Input(shape=(0,))
# x = keras.layers.Dense(16, activation='relu', input_dim=1)(x)
# x = keras.layers.Dense(1, activation='linear')(x)

# model = keras.Model(i, x)

model = keras.models.Sequential([
    keras.layers.Dense(16, activation='linear', input_dim=1),
    keras.layers.Dense(1, activation='linear')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.SGD(lr=0.2),
    loss='mse'
)

# Fit model
print("Training ... With default parameters, this should take less than " +
      "10 seconds")
r = model.fit(
    x_train,
    y_train,
    batch_size=train_size,
    verbose=0,
    validation_data=[x_test, y_test],
    epochs=100,
    callbacks=[tb_call]
)

print("Average test loss:", np.average(r.history['loss']))

print(model.predict([60, 25, 2]))

# Logging custom scalars
# For example dynamic learning rate

log_dir = ".\\logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "\\metrics")
file_writer.set_as_default()


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.2
    if epoch > 10:
        learning_rate = 0.02
    if epoch > 20:
        learning_rate = 0.01
    if epoch > 50:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
tb_call = keras.callbacks.TensorBoard(log_dir=log_dir,
                                      histogram_freq=1)

model = keras.models.Sequential([
    keras.layers.Dense(16, activation='linear', input_dim=1),
    keras.layers.Dense(1, activation='linear')
])

model.compile(
    loss='mse',
    optimizer=keras.optimizers.SGD()
)

r = model.fit(
    x_train,
    y_train,
    batch_size=train_size,
    verbose=0,
    validation_data=[x_test, y_test],
    epochs=100,
    callbacks=[tb_call, lr_callback]
)

print(model.predict([60, 25, 2]))
