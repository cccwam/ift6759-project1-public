"""
    Dummy ConvLSTM model inspired by https://keras.io/examples/conv_lstm/

"""
import tensorflow as tf


class ConvLSTM(tf.keras.Model):
    def __init__(self):
        super(ConvLSTM, self).__init__(name='Convolutional LSTM')
        self.encoder = CNNEncoder()
        self.classifier = Classifier()

    def call(self, inputs):
        y = self.encoder(inputs)
        return self.classifier(y)


class CNNEncoder(tf.keras.Model):
    """
    The CNN encoder module, needed to extract features map.

    """
    def __init__(self, **kwargs):
        super(CNNEncoder, self).__init__(name='Convolutional Neural Network Encoder', **kwargs)
        self.seq = tf.keras.Sequential()
        self.seq.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
        self.seq.add(tf.keras.layers.BatchNormalization())
        self.seq.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
        self.seq.add(tf.keras.layers.BatchNormalization())
        self.seq.add(tf.keras.layers.GlobalAveragePooling3D())
        self.seq.add(tf.keras.layers.Flatten())

    def call(self, inputs):
        return self.seq(inputs)


class Classifier(tf.keras.Model):
    """
    The Classifier

    """
    def __init__(self, **kwargs):
        super(Classifier, self).__init__(name='Classifier', **kwargs)
        self.seq = tf.keras.Sequential()
        self.seq.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))

    def call(self, inputs):
        return self.seq(inputs)
