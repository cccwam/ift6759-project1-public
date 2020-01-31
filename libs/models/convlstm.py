"""
    Dummy ConvLSTM model inspired by https://keras.io/examples/conv_lstm/

"""
import tensorflow as tf


def my_conv_lstm_model_builder(verbose=False):
    """
        Builder function
    :param verbose:
    :return:
    """

    def my_cnn_encoder():
        """
            This function return the CNN encoder module, needed to extract features map.
        :return: Keras model containing the CNN encoder module
        """
        encoder_input = tf.keras.Input(shape=(None, 40, 40, 1), name='original_img')

        x = tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                                       padding='same', return_sequences=True)(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                                       padding='same', return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        encoder_output = tf.keras.layers.Flatten()(x)

        return tf.keras.Model(encoder_input, encoder_output, name='encoder')

    def my_classifier(my_cnn_encoder):
        """
            This function return the classification head module.
        :param my_cnn_encoder: Encoder which will extract features map. Used to get the output size.
        :return: Keras model containing the classification head module
        """
        clf_input = tf.keras.Input(shape=my_cnn_encoder.layers[-1].output_shape[1:], name='feature_map')

        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(clf_input)

        return tf.keras.Model(clf_input, x, name='classifier')

    def my_convlstm_model(my_cnn_encoder, my_classifier):
        """
            This function aggregates the all modules for the model.
        :param my_cnn_encoder: Encoder which will extract features map.
        :param my_classifier: Classification head
        :return: Consolidation Keras model
        """
        # noinspection PyProtectedMember
        model_input = tf.keras.Input(shape=my_cnn_encoder.layers[0]._batch_input_shape[1:], name='original_img')

        x = my_cnn_encoder(model_input)
        x = my_classifier(x)

        return tf.keras.Model(model_input, x, name='convLSTMModel')

    my_cnn_encoder = my_cnn_encoder()
    if verbose:
        print("")
        my_cnn_encoder.summary()
        print("")

    my_classifier = my_classifier(my_cnn_encoder)
    if verbose:
        print("")
        my_classifier.summary()
        print("")

    my_convlstm_model = my_convlstm_model(my_cnn_encoder, my_classifier)
    if verbose:
        print("")
        my_convlstm_model.summary()
        print("")

    return my_convlstm_model
