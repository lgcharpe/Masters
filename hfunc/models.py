import tensorflow as tf


def create_1_layer_ANN_classification(
    input_shape,
    num_hidden_nodes,
    num_classes,
    activation_function='relu'
):
    """
    Creates a 1 layer classification ANN with the keras backend.

    Arguments:

        input_shape (int tuple): The shape of the input excluding batch.

        num_hidden_nodes (int): Number of hidden nodes in the hidden layers.

        num_classes (int): Number of classes the data contains.

        activation_function (str/tf.keras.activations): Activation function to
        use for the ANN. Default: ReLU.

    Returns:

        tf.keras.model.Sequential: A model of the ANN created.

    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(
            num_hidden_nodes,
            activation=activation_function),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_basic_ANN(
    x_train,
    y_train,
    num_hidden_nodes,
    validation_data=None,
    epochs=10,
    callbacks=None,
    verbose=0
):

    in_shape = x_train.shape[1:]
    K = 0
    if len(y_train.shape) > 1:
        K = y_train.shape[1]
    elif len(y_train.shape) == 1:
        K = len(set(y_train))

    assert(K), "Your y_train has no data"

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=in_shape),
        tf.keras.layers.Dense(num_hidden_nodes, activation='relu'),
        tf.keras.layers.Dense(K, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    return model, history
