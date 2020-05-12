import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins import projector

(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True,
)
encoder = info.features['text'].encoder

# shuffle and pad the data.
train_batches = train_data.shuffle(1000).padded_batch(
    10, padded_shapes=((None,), ())
)
test_batches = test_data.shuffle(1000).padded_batch(
    10, padded_shapes=((None,), ())
)
train_batch, train_labels = next(iter(train_batches))

# Create an embedding layer
embedding_dim = 16
embedding = tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim)
# Train this embedding as part of a keras model
model = tf.keras.models.Sequential([
    embedding,  # The embedding layer should be the first layer in a model
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train model
r = model.fit(
    train_batches,
    epochs=1,
    validation_data=test_batches,
    validation_steps=20
)

# Set up a logs directory, so Tensorboard knows where to look for files
log_dir = '.\\logs\\imdb_examples\\'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = 'embedding'
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

# Save Labels seperately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), 'w', encoding='utf-8') as f:
    for subwords in encoder.subwords:
        f.write(f'{subwords}\n')
    # Fill in the rest of the labels with 'unknown'
    for unknown in range(1, encoder.vocab_size - len(encoder.subwords)):
        f.write(f'unknown #{unknown}\n')

# Save the weights we want to analyse as a variable. Note that the first value
# represents any unknown word, which is not in the metadat, so we sill remove
# that value.
# weights = tf. Variable(model.layers[0].get_weights()[0][1:], name='embedding')
# # Create a checkpoint from embedding, the filename and key are
# # name of the tensor.
# checkpoint = tf.train.Checkpoint(embedding=weights)
# checkpoint.save('.\\logs\\imdb_examples\\embedding.ckpt')
LOG_DIR = '.\\logs\\imdb_examples'
EMBEDDINGS_TENSOR_NAME = 'embedding'
EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + '.ckpt')
tensor_embeddings = tf.Variable(model.layers[0].get_weights()[0][1:],
                                name=EMBEDDINGS_TENSOR_NAME)
saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
saver.save(sess=None, global_step=0, save_path=EMBEDDINGS_FPATH)

print("Done")
