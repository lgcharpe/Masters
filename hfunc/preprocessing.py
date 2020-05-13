import tensorflow as tf
import numpy as np


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
