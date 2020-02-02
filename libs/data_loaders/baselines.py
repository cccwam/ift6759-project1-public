import datetime
import typing

import numpy as np
import pandas as pd
import tensorflow as tf


def data_loader_v1(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """Baseline data loader.

    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
        target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
            The ordering of this list is important, as each element corresponds to a sequence of GHI values
            to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
            which are added to each timestamp (T=0) in this datetimes list.
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """

    def baseline_generator():
        """Baseline generator with time/lat/lon/elev."""

        batch_size = 256
        output_seq_len = 4
        if len(list(stations.keys())) > 1:
            raise NotImplementedError()
        station_name = list(stations.keys())[0]

        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i:i + batch_size]
            samples_np = np.zeros([len(batch_of_datetimes), 8])
            targets_np = np.zeros([len(batch_of_datetimes), output_seq_len])

            for j, dt in enumerate(batch_of_datetimes):
                samples_np[j, 0] = dt.year
                samples_np[j, 1] = dt.month
                samples_np[j, 2] = dt.day
                samples_np[j, 3] = dt.hour
                samples_np[j, 4] = dt.minute
                samples_np[j, 5] = stations[station_name][0]
                samples_np[j, 6] = stations[station_name][1]
                samples_np[j, 7] = stations[station_name][2]
                for m in range(output_seq_len):
                    k = dataframe.index.get_loc(dt + target_time_offsets[m])
                    targets_np[j, m] = dataframe[f"{station_name}_GHI"][k]

            samples = tf.convert_to_tensor(samples_np, dtype=tf.int32)
            targets = tf.convert_to_tensor(targets_np)

            yield samples, targets

    return tf.data.Dataset.from_generator(
        baseline_generator, (tf.float32, tf.float32)
    )
