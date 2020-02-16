"""
    Dummy ConvLSTM model inspired by https://keras.io/examples/conv_lstm/

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
        Builder function for the first cnn model

        This model is a vanilla CNN similar to the ConvLSTM_v2 in term of layers.
        It has got much lower number of parameters (about 280k instead of 7m)
        Compared to ConvLSTM_V2, vanilla CNN are much faster to train and the performance


    :param stations:
    :param target_time_offsets:
    :param config:
    :param verbose:
    :return:
    """

    def my_classifier(input_size):
        """
            This function return the classification head module.
        :param input_size:
        :return: Keras model containing the classification head module
        """
        clf_input = tf.keras.Input(shape=input_size, name='feature_map')

        x = tf.keras.layers.Lambda(lambda x: x[:, 8:12], output_shape=(4,), name="extract_clearsky")(clf_input)

        return tf.keras.Model(clf_input, x, name='classifier')

    def my_clearsky_model(my_classifier):
        """
            This function aggregates the all modules for the model.
        :param my_classifier: Classification head
        :return: Consolidation Keras model
        """
        # noinspection PyProtectedMember
        img_input = tf.keras.Input(shape=(5, 5, 50, 50), name='original_img')
        metadata_input = tf.keras.Input(shape=my_classifier.layers[0]._batch_input_shape[1:], name='metadata')

        x = my_classifier(metadata_input)

        return tf.keras.Model([img_input, metadata_input], x, name='convLSTMModel')

    model_hparams = config["model"]["hyper_params"]

    my_classifier = my_classifier(input_size=model_hparams["nb_metadata"])
    if verbose:
        print("")
        my_classifier.summary()
        print("")

    my_clearsky_model = my_clearsky_model(my_classifier)
    if verbose:
        print("")
        my_clearsky_model.summary()
        print("")

    return my_clearsky_model
