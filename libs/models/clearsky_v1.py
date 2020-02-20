"""
    Benchmark model which forward the clearsky predictions as inputs to output

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
        Builder function for the clearsky model.

        Note: No parameter is learned in this model.


    :param stations:
    :param target_time_offsets:
    :param config:
    :param verbose:
    :return:
    """

    def my_head(input_size):
        """
            This function return the head module.

        :param input_size:
        :return: Keras model containing the classification head module
        """
        clf_input = tf.keras.Input(shape=input_size, name='feature_map')

        x = tf.keras.layers.Lambda(lambda x: x[:, 8:12], output_shape=(4,), name="extract_clearsky")(clf_input)

        return tf.keras.Model(clf_input, x, name='head')

    def my_clearsky_model(my_head):
        """
            This function aggregates the all modules for the model.

        :param my_head:  head
        :return: Consolidation Keras model
        """
        # noinspection PyProtectedMember
        img_input = tf.keras.Input(shape=(5, 5, 50, 50), name='original_img')
        metadata_input = tf.keras.Input(shape=my_head.layers[0]._batch_input_shape[1:], name='metadata')

        x = my_head(metadata_input)

        return tf.keras.Model([img_input, metadata_input], x, name='clearsky')

    model_hparams = config["model"]["hyper_params"]

    my_head = my_head(input_size=model_hparams["nb_metadata"])
    if verbose:
        print("")
        my_head.summary()
        print("")

    my_clearsky_model = my_clearsky_model(my_head=my_head)
    if verbose:
        print("")
        my_clearsky_model.summary()
        print("")

    return my_clearsky_model
