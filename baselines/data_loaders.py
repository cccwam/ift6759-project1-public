import datetime
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

def baselines(
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

    def dummy_data_generator():
        """
        Generate dummy data for the model, only for example purposes.
        """
        batch_size = 32
        output_seq_len = 4
        if len(list(stations.keys())) > 1:
            raise NotImplementedError()
        station_name = list(stations.keys())[0]


        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i:i + batch_size]
            samples_np = np.zeros([len(batch_of_datetimes), 8], dtype=np.int32)
            targets_np = np.zeros([len(batch_of_datetimes), output_seq_len])

            for j, datetime in enumerate(batch_of_datetimes):
                samples_np[j, 0] = datetime.year
                samples_np[j, 1] = datetime.month
                samples_np[j, 2] = datetime.day
                samples_np[j, 3] = datetime.hour
                samples_np[j, 4] = datetime.minute
                samples_np[j, 5] = stations[station_name][0]*1000
                samples_np[j, 6] = stations[station_name][1]*1000
                samples_np[j, 7] = stations[station_name][2]
                for l in range(output_seq_len):
                    k = dataframe.index.get_loc(datetime + target_time_offsets[l])
                    targets_np[j, l] = dataframe[f"{station_name}_GHI"][k]

            samples = tf.convert_to_tensor(samples_np, dtype=tf.int32)
            targets = tf.convert_to_tensor(targets_np)


            yield samples, targets

    data_loader = tf.data.Dataset.from_generator(
        dummy_data_generator, (tf.float32, tf.float32)
    )

    return data_loader
