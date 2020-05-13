import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%]'],
    with_info=True,
    as_supervised=True
)

print(raw_train)
print(raw_validation)
print(raw_test)