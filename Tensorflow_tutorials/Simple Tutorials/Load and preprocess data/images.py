import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__)

# Setup

data_dir = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname='flower_photos',
    untar=True
)

data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

CLASS_NAMES = np.array([
    item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)

roses = list(data_dir.glob('roses/*'))

for image_path in roses[:3]:
    display.display(Image.open(str(image_path)))

# Load using keras.preprocessing

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(
    directory=str(data_dir),
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(CLASS_NAMES)
)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        _ = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')


image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

# Load using tf.data

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

for f in list_ds.take(5):
    print(f.numpy)


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # Convert the compressed images to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Basic methods for training, this lets the data be:
# Well shuffled
# Batched
# Quick of access

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

del list_ds

for image, label in labeled_ds.take(1):
    print("Image shape:", image.numpy().shape)
    print('Label:', label.numpy())

del image, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    """
    This is a small dataset, only load it once, and keep it in memory.
    Use `.cache(filename)`to cache preprocessing work for datasets that do not
    fir in memory.
    """

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the data fetch batches in the background while
    # the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

show_batch(image_batch.numpy(), label_batch.numpy())

# Tricks to improve Performance

default_timeit_steps = 1000


def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        _ = next(it)
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print(f'{steps} batches: {duration} s')
    print(f'{BATCH_SIZE * steps / duration:0.5f} Images/s')


# `keras.preprocessing`
timeit(train_data_gen)
# 1000 batches: 57.89s
# 553 Images/s

# `tf.data`
timeit(train_ds)
# 1000 batches: 8.60s
# 3719 Images/s

uncached_ds = prepare_for_training(labeled_ds, cache=False)
timeit(uncached_ds)
# 1000 batches: 35.84s
# 893 Images/s

filecache_ds = prepare_for_training(labeled_ds, cache='./flowers.tfcache')
timeit(filecache_ds)
# 1000 batches: 31.52s
# 1015 Images/s


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
del list_ds
train_ds = prepare_for_training(labeled_ds)
# filecache_ds = prepare_for_training(labeled_ds, cache='./flowers.tfcache')
# uncached_ds = prepare_for_training(labeled_ds, cache=False)
del labeled_ds
timeit(train_ds)
# timeit(filecache_ds)
# timeit(uncached_ds)
