from __future__ import absolute_import, division, print_function

from datetime import datetime
import io
import itertools
from packaging import version
from six.moves import range

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# Load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('Shape:', train_images[0].shape)
print("Label:", train_labels[0], '->', class_names[train_labels[0]])

# Reshape the image for summary API
img = np.reshape(train_images[0], (-1, 28, 28, 1))

# Specify directory
log_dir = ".\\logs\\train_data\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Create a file writer for the log directory
file_writer = tf.summary.create_file_writer(log_dir)

# Using the file writer, log the reshaped image.
with file_writer.as_default():
    tf.summary.image('Training data', img, step=0)

with file_writer.as_default():
    # Do not forget to reshape
    images = np.reshape(train_images[0:25], (-1, 28, 28, 1))
    tf.summary.image('25 training data examples', images, max_outputs=25,
                     step=0)

# Logging arbitrary image data
log_dir = ".\\logs\\plots\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)


def plot_to_image(figure):
    """
    Converts the matplot lib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed inside
    # the notebook
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid():
    """
    Return a 5x5 grid of the MNIST images as a matplotlib figure.
    """
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot
        plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

    return figure


# Prepare the plot
figure = image_grid()
# Convert to image and log
with file_writer.as_default():
    tf.summary.image("Trainig data", plot_to_image(figure), step=0)

# Building an image classifier
i = keras.layers.Input(shape=(28, 28))
x = keras.layers.Flatten()(i)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(i, x)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'
    """

    figure = plt.figure(figsize=(8, 8))

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                       decimals=2)
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color='white' if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


log_dir = ".\\logs\\plots\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb_call = keras.callbacks.TensorBoard(log_dir=log_dir,
                                      histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(log_dir + '\\cm')


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(test_images)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, classes=class_names, normalize=True)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# Define the per-epoch callback
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# Train the classifier.
model.fit(
    train_images,
    train_labels,
    epochs=10,
    verbose=0,
    callbacks=[tb_call, cm_callback],
    validation_data=(test_images, test_labels)
)
