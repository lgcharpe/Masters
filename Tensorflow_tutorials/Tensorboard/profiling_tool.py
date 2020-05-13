from __future__ import absolute_import, division, print_function

from datetime import datetime

import os

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

print('TensorFlow version:', tf.__version__)

device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print(f'Found GPU at: {device_name}')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


def normalize_img(image, label):
    """
    Normalizes images: `uint8` -> `float32`.
    """

    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(128)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Create a Tensorboard callback
logs = '.\\logs\\profile\\' + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs,
    histogram_freq=1,
    profile_batch='500,520'
)

with tf.device("/device:GPU:0"):
    model.fit(
        ds_train,
        epochs=2,
        validation_data=ds_test,
        callbacks=[tboard_callback]
    )
