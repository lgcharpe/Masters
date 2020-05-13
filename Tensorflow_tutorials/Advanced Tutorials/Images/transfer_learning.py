import os

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()
# tf.compat.v1.disable_eager_execution()
# tf.keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False

IMG_SIZE = 160  # All images will be resized to 160x160

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

logs = '.\\logs\\profile\\' + datetime.now().strftime("%Y%m%d-%H%M%S")

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str

# for image, label in raw_train.take(2):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))

# Format the data

# IMG_SIZE = 160  # All images will be resized to 160x160


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# for image_batch, label_batch in train_batches.take(1):
#     pass

# image_batch.shape

# IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# # Create the base model from the pre-trained model MobileNet V2
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=IMG_SHAPE,
#     include_top=False,
#     weights='imagenet'
# )

# feature_batch = base_model(image_batch)
# print(feature_batch.shape)

# This will prevent the weights from changing during training
base_model.trainable = False

# Let us take a look as the base model architecture
print(base_model.summary())

# Add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs,
    histogram_freq=1,
    profile_batch='1200,3000'
)

# Compile model
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(amsgrad=True, lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print(model.summary())

initial_epochs = 10
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

with tf.device("/device:GPU:0"):
    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches,
                        callbacks=[tboard_callback])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')


plt.show()
