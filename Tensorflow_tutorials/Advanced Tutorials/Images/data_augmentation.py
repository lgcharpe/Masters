import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import matplotlib as mpl

AUTOTUNE = tf.data.experimental.AUTOTUNE
mpl.rcParams['figure.figsize'] = (12, 5)

image_path = tf.keras.utils.get_file(
    "cat.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg"
)

image_string = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_string, channels=3)


def visualize(original, augmented):
    _ = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)


# Flipping the image
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)

# Grayscale the image
grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled))
plt.colorbar()

# Saturate the image
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)

# Change image brightness
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)

# Rotate the image
rotated = tf.image.rot90(image)
visualize(image, rotated)

# Center crop the image
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image, cropped)

# Augment a dataset and train a model with it

dataset, info = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

num_train_examples = info.splits['train'].num_examples


def convert(image, label):
    # Cast and normalize the image to [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def augment(image, label):
    image, label = convert(image, label)
    # Cast and normalize the image to [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Add 6 pixels of padding
    image = tf.image.resize_with_crop_or_pad(image, 34, 34)
    # Random crop back to 28x28
    image = tf.image.random_crop(image, size=[28, 28, 1])
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.5)

    return image, label


BATCH_SIZE = 64
# Only use a subset of the data so it is easier to overfit, for this tutorial
NUM_EXAMPLES = 4096

augmented_train_batches = (
    train_dataset
    # Only train on a subset, so you can quickly see the effect.
    .take(NUM_EXAMPLES)
    .cache()
    .shuffle(num_train_examples//4)
    # The augmentation is added here.
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

non_augmented_train_batches = (
    train_dataset
    # Only train on a subset, so you can quickly see the effect.
    .take(NUM_EXAMPLES)
    .cache()
    .shuffle(num_train_examples//4)
    # No augmentation.
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

validation_batches = (
    test_dataset
    .map(convert, num_parallel_calls=AUTOTUNE)
    .batch(2 * BATCH_SIZE)
)


def make_model():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


model_without_aug = make_model()

no_aug_history = model_without_aug.fit(
    non_augmented_train_batches,
    epochs=50,
    validation_data=validation_batches
)

model_with_aug = make_model()

aug_history = model_with_aug.fit(
    augmented_train_batches,
    epochs=50,
    validation_data=validation_batches
)

plt.figure()
plotter = tfdocs.plots.HistoryPlotter()
plotter.plot(
    {"Augmented": aug_history, "Non-Augmented": no_aug_history},
    metric='accuracy')
plt.title("Accuracy")
plt.ylim([0.75, 1])

plt.figure()
plotter = tfdocs.plots.HistoryPlotter()
plotter.plot(
    {"Augmented": aug_history, "Non-Augmented": no_aug_history},
    metric='loss')
plt.title("Loss")
plt.ylim([0, 1])

plt.show()
