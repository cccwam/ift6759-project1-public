"""
    ConvLSTM model inspired by https://keras.io/examples/conv_lstm/

    This model is too small to achieve any performance.

"""
import datetime
import typing

import tensorflow as tf


def my_conv_lstm_model_builder(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        verbose=True):
    """
        Builder function for the first convlstm model

        This model is too small to achieve any performance.

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
        encoder_input = tf.keras.Input(shape=(None, 5, 50, 50), name='original_img')

        x = tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                                       data_format='channels_last',
                                       padding='same', return_sequences=True)(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                                       data_format='channels_last',
                                       padding='same', return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        encoder_output = tf.keras.layers.Flatten()(x)

        return tf.keras.Model(encoder_input, encoder_output, name='encoder')

    def my_head(input_size):
        """
            This function return the head module.
        :param input_size:
        :return: Keras model containing the head module
        """
        clf_input = tf.keras.Input(shape=input_size, name='feature_map')

        x = tf.keras.layers.Dense(4, activation=None)(clf_input)

        return tf.keras.Model(clf_input, x, name='head')

    def my_convlstm_model(my_cnn_encoder, my_head):
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

        return tf.keras.Model([img_input, metadata_input], x, name='convLSTMModel')

    my_cnn_encoder = my_cnn_encoder()
    if verbose:
        print("")
        my_cnn_encoder.summary()
        print("")

    my_head = my_head(input_size=my_cnn_encoder.layers[-1].output_shape[1] + 8)
    if verbose:
        print("")
        my_head.summary()
        print("")

    my_convlstm_model = my_convlstm_model(my_cnn_encoder, my_head)
    if verbose:
        print("")
        my_convlstm_model.summary()
        print("")

    return my_convlstm_model
