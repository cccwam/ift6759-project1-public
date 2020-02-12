import datetime
import typing

import numpy as np
import tensorflow as tf


def test_dataloader(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
):
    """This function should be modified in order to prepare & return your own prediction model.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A model
    """

    class TestDataLoader(tf.keras.Model):

        def __init__(self, target_time_offsets):
            super(TestDataLoader, self).__init__()
            self.verbose = True

        def call(self, inputs):
            if self.verbose:
                if not isinstance(inputs, tuple):
                    inputs_tuple = (inputs, )
                else:
                    inputs_tuple = inputs
                for i, one_input in enumerate(inputs_tuple):
                    print(f"Input {i} type: {type(one_input)}")
                    print(f"Input {i} shape: {one_input.shape}")
                    print(f"Input {i} min: {one_input.numpy().min()}")
                    print(f"Input {i} mean: {one_input.numpy().mean()}")
                    print(f"Input {i} max: {one_input.numpy().max()}")
                    print(f"Input {i} number of nan values: {tf.math.is_nan(one_input).numpy().sum()}")
                self.verbose = False
            preds = np.zeros([inputs.shape[0], 4])
            return preds

    model = TestDataLoader(target_time_offsets)

    return model


def test_mlp(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
):
    """This function should be modified in order to prepare & return your own prediction model.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A model
    """

    encoder_input = tf.keras.Input(batch_size=256, shape=(5, 5, 50, 50), name='original_img')
    x = tf.keras.layers.Flatten()(encoder_input)
    x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
    encoder_output = tf.keras.layers.Dense(len(target_time_offsets))(x)

    return tf.keras.Model(encoder_input, encoder_output, name='encoder')

    # class TestMLP(tf.keras.Model):
    #
    #     def __init__(self, target_time_offsets):
    #         super(TestMLP, self).__init__()
    #         self.flatten = tf.keras.layers.Flatten()
    #         self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
    #         self.dense2 = tf.keras.layers.Dense(len(target_time_offsets))
    #         self.loss_object = tf.keras.losses.MeanSquaredError()
    #         self.optimizer = tf.keras.optimizers.Adam()
    #         self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    #         self.train_accuracy = tf.keras.metrics.RootMeanSquaredError(
    #             name='train_accuracy')
    #         self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    #         self.test_accuracy = tf.keras.metrics.RootMeanSquaredError(
    #             name='test_accuracy')
    #
    #     @tf.function
    #     def call(self, inputs, training=False):
    #         x = self.dense1(self.flatten(inputs))
    #         return self.dense2(x)
    #
    #     def train_step(self, images, labels):
    #         with tf.GradientTape() as tape:
    #             # training=True is only needed if there are layers with different
    #             # behavior during training versus inference (e.g. Dropout).
    #             predictions = model(images, training=True)
    #             loss = self.loss_object(labels, predictions)
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         self.optimizer.apply_gradients(
    #             zip(gradients, model.trainable_variables))
    #
    #         self.train_loss(loss)
    #         self.train_accuracy(labels, predictions)
    #
    #     @tf.function
    #     def test_step(self, images, labels):
    #         # training=False is only needed if there are layers with different
    #         # behavior during training versus inference (e.g. Dropout).
    #         predictions = model(images, training=False)
    #         t_loss = self.loss_object(labels, predictions)
    #
    #         self.test_loss(t_loss)
    #         self.test_accuracy(labels, predictions)
    #
    #     def fit(self, train_ds, validation_data=None, epochs=1, callbacks=None):
    #         for epoch in range(1):
    #             # Reset the metrics at the start of the next epoch
    #             self.train_loss.reset_states()
    #             self.train_accuracy.reset_states()
    #             self.test_loss.reset_states()
    #             self.test_accuracy.reset_states()
    #
    #             for images, labels in train_ds:
    #                 self.train_step(images, labels)
    #
    #             if validation_data:
    #                 for test_images, test_labels in validation_data:
    #                     self.test_step(test_images, test_labels)
    #
    #             template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #             print(template.format(epoch + 1,
    #                                   self.train_loss.result(),
    #                                   self.train_accuracy.result(),
    #                                   self.test_loss.result(),
    #                                   self.test_accuracy.result()))
    #
    # model = TestMLP(target_time_offsets)
    #
    # return model
