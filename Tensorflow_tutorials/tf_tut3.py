# Classifying text with tensorflow/keras.

from __future__ import absolute_import, division, print_function,\
    unicode_literals

import tensorflow as tf

from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

print(tf.__version__)

# Dataset has already been preprocessed, i.e. reviews are converted
# to sequences of integers.

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a
    # dictionary).
    as_supervised=True,
    # Also return the `info`structure.
    with_info=True
)

# Trying the encoder

encoder = info.features['text'].encoder

print(f'Vocabulary size: {encoder.vocab_size}')

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print(f'Encoded string is {encoded_string}')

original_string = encoder.decode(encoded_string)
print(f'The original string: {original_string}')

assert original_string == sample_string

for ts in encoded_string:
    print(f'{ts} ----> {encoder.decode([ts])}')

# Explore the data

for train_example, train_label in train_data.take(1):
    print('Encoded text:', train_example[:10].numpy())
    print('Label:', train_label.numpy())

print(encoder.decode(train_example))

# Prepare the data for training

BUFFER_SIZE = 1000

train_batches = (
    train_data.shuffle(BUFFER_SIZE)
    .padded_batch(32, train_data.output_shapes)
)

test_batches = (
    test_data.padded_batch(32, train_data.output_shapes)
)

for example_batch, label_batch in train_batches.take(2):
    print("Batch shape:", example_batch.shape)
    print("label shape:", label_batch.shape)

# Build the model

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Loss function and optimizer

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches,
    validation_steps=30
)

# Evaluate the model

loss, accuracy = model.evaluate(test_batches)

print("Loss:", loss)
print("Accuracy:", accuracy)

# Create a graph of accuracy and loss over time

history_dict = history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Train loss')
