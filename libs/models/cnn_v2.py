"""
    Simple vanilla CNN model but with more parameters than the v1 model

"""
import datetime
import typing

import tensorflow as tf


def my_model_builder(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        verbose=True):
    """
        Builder function for the second cnn model

        This model is a bigger model than cnn_v1 (about 1.4m parameters)


    :param stations:
    :param target_time_offsets:
    :param config:
    :param verbose:
    :return:
    """

    def my_cnn_encoder():
        """
            This function return the CNN encoder module, needed to extract features map.

        :return: Keras model containing the CNN encoder module
        """
        encoder_input = tf.keras.Input(shape=(5, 5, 50, 50), name='original_img')
        x = tf.keras.layers.Reshape(target_shape=(5 * 5, 50, 50))(encoder_input)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5),
                                   data_format='channels_first',
                                   activation=tf.keras.activations.relu,
                                   padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                   data_format='channels_first',
                                   activation=tf.keras.activations.relu,
                                   padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                   data_format='channels_first',
                                   activation=tf.keras.activations.relu,
                                   padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                   data_format='channels_first',
                                   activation=tf.keras.activations.relu,
                                   padding='same')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                   data_format='channels_first',
                                   activation=tf.keras.activations.relu,
                                   padding='same')(x)

        x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
        encoder_output = tf.keras.layers.Flatten()(x)

        return tf.keras.Model(encoder_input, encoder_output, name='encoder')

    def my_head(input_size, dropout):
        """
            This function return the classification head module.
        :param input_size:
        :param dropout:
        :return: Keras model containing the classification head module
        """
        clf_input = tf.keras.Input(shape=input_size, name='feature_map')

        x = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(clf_input)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(4, activation=None)(x)

        return tf.keras.Model(clf_input, x, name='head')

    def my_cnn_model(my_cnn_encoder, my_head):
        """
            This function aggregates the all modules for the model.
        :param my_cnn_encoder: Encoder which will extract features map.
        :param my_head: Classification head
        :return: Consolidation Keras model
        """
        # noinspection PyProtectedMember
        img_input = tf.keras.Input(shape=my_cnn_encoder.layers[0]._batch_input_shape[1:], name='img')
        metadata_input = tf.keras.Input(shape=(8,), name='metadata')

        x = my_cnn_encoder(img_input)
        all_inputs = tf.keras.layers.Concatenate()([x, metadata_input])
        x = my_head(all_inputs)

        return tf.keras.Model([img_input, metadata_input], x, name='cnn_v2')

    model_hparams = config["model"]["hyper_params"]

    nb_metadata = model_hparams["nb_metadata"]

    my_cnn_encoder = my_cnn_encoder()
    if verbose:
        print("")
        my_cnn_encoder.summary()
        print("")

    my_head = my_head(input_size=my_cnn_encoder.layers[-1].output_shape[1] + nb_metadata,
                                  dropout=model_hparams["dropout"])
    if verbose:
        print("")
        my_head.summary()
        print("")

    my_cnn_model = my_cnn_model(my_cnn_encoder, my_head)
    if verbose:
        print("")
        my_cnn_model.summary()
        print("")

    return my_cnn_model
