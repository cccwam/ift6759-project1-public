import datetime
import pickle
import typing

import tensorflow as tf

import numpy as np


def global_mean(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any]
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

    class GlobalMean(tf.keras.Model):

        def __init__(self, target_time_offsets):
            super(GlobalMean, self).__init__()
            self.global_mean = None
            self.train()

        def train(self):
            station_names = ['BND', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']
            with open(config['model']['hyper_params']['dataframe_path'], 'rb') as f:
                df = pickle.load(f)
                for station_name in station_names:
                    setattr(self, station_name,
                            df[f"{station_name}_GHI"].mean())
                self.global_mean = (self.BND + self.DRA + self.FPK +
                                    self.GWN + self.PSU + self.SXF +
                                    self.TBL) / 7.

        def call(self, inputs):

            x = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(np.repeat(self.global_mean, 4)),
                                       output_shape=(4,), name="global_mean")(inputs)

            return x

    model = GlobalMean(target_time_offsets)

    return model
