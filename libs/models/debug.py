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
