import tensorflow as tf
import numpy as np
import os


def train_val_test_split(dataset, train_frac=0, val_frac=0,
                         test_frac=0):
    """
    This function takes a TensorFlow dataset and splits it into train,
    validation and test sets. If only a train_frac is specified, the
    function will return a train set and test set. A train set will
    always be returned unless the fractions of the validation and test
    sets sum up to 1.

    Args:

        dataset (tf.data.Dataset): A TensorFlow Dataset object.

        train_frac (double): fraction of the dataset that is for training.

        val_frac (double): fraction of the dataset that is for validation.

        test_frac (double): fraction of the dataset that is for testing.

    Returns:

        list: A list containing the split datasets, the order is train,
              val, test. It will only return non-empty datasets.

    """

    DATASET_LENGTH = len(list(dataset.as_numpy_iterator()))

    assert(train_frac or test_frac or val_frac),\
        "specify at least one of the fractions"
    assert(train_frac + test_frac + val_frac <= 1),\
        "The fractions cannot sum-up to more than one"

    if train_frac:
        if test_frac:
            if not val_frac:
                val_frac = 1 - (test_frac + train_frac)
        elif val_frac:
            test_frac = 1 - (val_frac + train_frac)
        else:
            test_frac = 1 - train_frac
    elif test_frac:
        if val_frac:
            train_frac = 1 - (test_frac + val_frac)
        else:
            train_frac = 1 - test_frac
    else:
        train_frac = 1 - val_frac

    train_size = int(train_frac * DATASET_LENGTH)
    test_size = int(test_frac * DATASET_LENGTH)
    val_size = int(val_frac * DATASET_LENGTH)

    datasets = []

    if train_size:
        train = dataset.take(train_size)
        datasets.append(train)
    if val_size:
        val = dataset.skip(train_size).take(val_size)
        datasets.append(val)
    if test_size:
        test = dataset.skip(train_size + val_size)
        datasets.append(test)

    return datasets


def get_label(file_path, CLASS_NAMES):
    """
    This function generates the one-hot encoded label of the image.

    Arguments:

        file_path (str): Where the path is located.

        CLASS_NAMES (list): list containing the names of the classes.

    Returns:

        list: One-hot encoded label vector

    """

    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


def decode_img(img, im_height, im_width, channels=3):
    """
    Converts a jpeg image to a 3D float32 tensor with values between
    0 and 1. It then resizes it to the desired size.

    Arguments:

        img (jpeg image object): The jpeg image to convert.

        im_height (int): Desired image height.

        im_width (int): Desired image width.

    Returns:

        tf.Tensor: A tensor object representing the image

    """

    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [im_height, im_width])


def process_path(file_path, CLASS_NAMES, im_height, im_width, channels=3):
    """
    Generates a image tensor and label for the image found at the specified
    path.

    Arguments:

        file_path (str): Where the path is located.

        CLASS_NAMES (list): list containing the names of the classes.

        im_height (int): Desired image height.

        im_width (int): Desired image width.

    Returns:

        tf.Tensor: A tensor object representing the image

        list: One-hot encoded label vector

    """
    label = get_label(file_path, CLASS_NAMES)
    img = tf.io.read_file(file_path)
    img = decode_img(img, im_height, im_width, channels)
    return img, label


def load_dataset_images(path):
    """
    This function loads image data into a tf.data.Dataset object, where
    the images have been rescaled to have pixel values between 0 and 1.

    Args:

        path (str): String containing the location of the data

    Returns:

        dataset (tf.data.Dataset): A Dataset object containing the data
                                   specified by the path

    """

    CLASS_NAMES = np.array([
        item.name for item in path.glob('*') if item.name != 'LICENSE.txt'
    ])

    NUM_CLASSES = len(CLASS_NAMES)

    list_ds = tf.data.Dataset.list_files(str(path/'*/*'))

    labeled_ds = list_ds.interleave(
        process_path,
        cycle_length=NUM_CLASSES,
        num_parllel_calls=tf.data.experimental.AUTOTUNE
    )

    del list_ds

    return labeled_ds


def prepare_for_model_use(
    dataset,
    cache=True,
    shuffle=True,
    shuffle_buffer_size=1000,
    batch_size=32,
    prefetch=True,
    repeat=True,
):
    """
    Makes the dataset ready for use by a model by possibly caching and
    shuffling it. It will also batch the dataset. Finally, it will also
    activate prefetching to make the reading of data more efficient.

    Arguments:
        dataset (tf.data.Dataset): the dataset to prepare for training

        cache (str/Boolean): Whether or not to cache the data, if a str is
        passed, this will indicate where to store the cache, this is useful
        when the dataset is too large to store in memory. Default: True

        shuffle (Boolean): Whether to shuffle the data of not. Default: True

        shuffle_buffer_size (int): The number of elements to randomly select
        from. Default: 1000

        batch_size (int): The size of the batches. If 0 or None is passed the
        dataset will not be batched. Default: 32

        prefetch (Boolean): Whether or not to have prefetching active for the
        dataset. Default: True

        repeat (Boolean): Whether or not to repeat the dataset indefinitely.
        If true make sure to sure fit_generator and not fit when fitting the
        model with keras. Default: True

    Returns:

        tf.data.Dataset: the dataset inputed but now ready to be efficiently
        used for training.

    """

    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    if repeat:
        dataset = dataset.repeat()

    if batch_size:
        dataset = dataset.batch(batch_size)

    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
