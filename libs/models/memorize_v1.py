import datetime
import typing

import tensorflow as tf


# TODO check the logic with Blaise
# Is it to use the target as input and to scale it with a factor ?
# If so, then a new dataloader is needed (to give the true as input)
def memorize(
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

    class MemorizeData(tf.keras.Model):
        # TODO remove this
        pass

        # def __init__(self, target_time_offsets):
        #     super(MemorizeData, self).__init__()
        #     self.datetimes = None
        #     self.train()
        #
        # def train(self):
        #     station_names = ['BND', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']
        #     with open(config['model']['hyper_params']['dataframe_path'], 'rb') as f:
        #         df = pickle.load(f)
        #         self.datetimes = df.index
        #         for station_name in station_names:
        #             setattr(self, station_name, df[f"{station_name}_GHI"])
        #
        # def call(self, inputs):
        #     if len(list(stations.keys())) > 1:
        #         raise NotImplementedError()
        #     station_name = list(stations.keys())[0]
        #     preds = np.zeros([inputs.shape[0], 4])
        #     for i in range(inputs.shape[0]):
        #         mydate = datetime.datetime(
        #             int(np.round(inputs[i, 0])), int(np.round(inputs[i, 1])),
        #             int(np.round(inputs[i, 2])), int(np.round(inputs[i, 3])),
        #             int(np.round(inputs[i, 4])))
        #         for m in range(4):
        #             k = self.datetimes.get_loc(mydate + target_time_offsets[m])
        #             ghi = getattr(self, station_name)[k]
        #             if np.isnan(ghi):
        #                 ghi = 0
        #             preds[i, m] = ghi * config['rmse_test_scale_factor']
        #
        #     x = tf.keras.layers.Lambda(lambda x: tf.repeat(tf.convert_to_tensor(self.global_mean), 4),
        #                                output_shape=(4,), name="global_mean")(inputs)
        #
        #     return preds

    raise NotImplementedError("")
    model = MemorizeData(target_time_offsets)

    return model
