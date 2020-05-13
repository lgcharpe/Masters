from hfunc import preprocessing
import tensorflow as tf
import pathlib

DIR_PATH = pathlib.Path(
    'C:\\Users\\lucas\\Documents\\Masters\\data\\kvasir-dataset-v2'
)

ds = preprocessing.load_dataset_images(DIR_PATH, 128, 128)

for image, label in ds.take(8):
    print("Image shape:", image.numpy().shape)
    print('Label:', label.numpy())

del image, label

train_ds, val_ds, test_ds = preprocessing.train_val_test_split(
    ds,
    0.7,
    0.15
)

cachefile = "C:\\Users\\lucas\\Documents\\Masters\\cache\\kvasir"

train_ds = preprocessing.prepare_for_model_use(
    train_ds,
    cache=cachefile,
    prefetch=True
)
val_ds = preprocessing.prepare_for_model_use(
    val_ds,
    cache=cachefile,
    shuffle=False,
    repeat=False
)
test_ds = preprocessing.prepare_for_model_use(
    test_ds,
    cache=False,
    shuffle=False,
    prefetch=False,
    repeat=False
)
