import tensorflow as tf


def get_tiplet_model(input_shape, embedding_size):
    batch_size = None

    inputs = tf.keras.Input(input_shape, name="input_rep", batch_size=batch_size)
    x = tf.reshape(inputs, tf.concat([[-1], input_shape[1:]], 0))
    x = process_input_patch(x)
    x = model_head(x, embedding_size, "individual_embeddings")
    x = tf.math.l2_normalize(x, axis=-1)
    embeddings = tf.reshape(x, [-1, 3, embedding_size], name="embeddings")

    model = tf.keras.Model(inputs=inputs, outputs=embeddings)
    return model


def get_forwarding_model(input_shape, embedding_size):
    batch_size = None

    inputs = tf.keras.Input(input_shape, name="input_rep", batch_size=batch_size)
    x = process_input_patch(inputs)
    x = model_head(x, embedding_size, "individual_embeddings")
    embeddings = tf.math.l2_normalize(x, axis=-1, name="embeddings")

    model = tf.keras.Model(inputs=inputs, outputs=embeddings)
    return model


def model_head(x, output_size, output_name, cnn_activation_function=tf.nn.leaky_relu, use_batchnorm=True, final_dropout_dense=True):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation=cnn_activation_function, use_bias=use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=cnn_activation_function, use_bias=use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    if final_dropout_dense:
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(output_size, use_bias=True, name=output_name)(x)
    return x


def get_classification_model(input_shape, num_classes):
    batch_size = None

    inputs = tf.keras.Input(input_shape, name="input_rep", batch_size=batch_size)
    x = process_input_patch(inputs)
    logits = model_head(x, num_classes, "logits")

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def get_classification_forwarding_model(input_shape, num_classes):
    batch_size = None

    inputs = tf.keras.Input(input_shape, name="input_rep", batch_size=batch_size)
    x = process_input_patch(inputs)
    x = model_head(x, num_classes, "pre_embeddings", final_dropout_dense=False)
    embeddings = tf.math.l2_normalize(x, axis=-1, name="embeddings")

    model = tf.keras.Model(inputs=inputs, outputs=embeddings)
    return model


def process_input_patch(inputs, cnn_activation_function=tf.nn.leaky_relu, use_batchnorm=True):
    x = inputs

    # Prefiltering stage with large kernel
    x = tf.keras.layers.Conv2D(64, kernel_size=(15, 15), strides=(1, 1), padding="SAME", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    # Aggregate 3 bins per semitone into one
    x = tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=(1, 3), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    # Conv-conv-pool
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding="VALID")(x)

    # Conv-conv-pool
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding="VALID")(x)

    # Conv-conv-pool
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(3, 3), padding="VALID")(x)

    # Aggregate time-axis
    x = tf.keras.layers.Conv2D(512, kernel_size=(5, 1), strides=(1, 1), padding="VALID", activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    return x


def get_probing_model(embedding_size, num_output_classes, cnn_activation_function=tf.nn.leaky_relu, use_batchnorm=False):
    inputs = tf.keras.Input(embedding_size, name="input_rep", batch_size=None)

    x = tf.keras.layers.Dense(128, activation=cnn_activation_function, use_bias=not use_batchnorm)(inputs)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dense(64, activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dense(32, activation=cnn_activation_function, use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    # x = inputs
    logits = tf.keras.layers.Dense(num_output_classes, use_bias=True, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model