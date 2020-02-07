import os
import datetime
import typing

import numpy as np
import pandas as pd
import netCDF4
import tensorflow as tf


def data_loader_images(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """Satellite images data loader.

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

    def image_generator():
        """Baseline generator with time/lat/lon/elev."""

        batch_size = 256
        output_seq_len = 4
        # Currently only support one station at a time.
        if len(list(stations.keys())) > 1:
            raise NotImplementedError()

        station_name = list(stations.keys())[0]
        data_file = f"preloader_{config['data_loader']['params']['admin_name']}_{station_name}.nc"

        nc = netCDF4.Dataset(
            os.path.join('/project/cq-training-1/project1/teams/team03/data', data_file), 'r')
        nc_var = nc.variables['data']
        # Loading all data in memory greatly speeds up things, but if we move to much larger
        # sample size and crop size this might need to be done differently.
        nc_var_data = nc_var[:, :, :, :, :]
        nc_time = nc.variables['time']

        # match target datenums with indices in the netcdf file, we need to allow for
        # small mismatch in seconds due to the nature of num2date and date2num.
        target_datenums = netCDF4.date2num(target_datetimes, nc_time.units, nc_time.calendar)
        nc_time_data = nc_time[:]
        indices_in_nc = np.zeros(len(target_datenums), dtype='i8')
        for i, target_datenum in enumerate(target_datenums):
            indices_in_nc[i] = np.where(np.isclose(nc_time_data, target_datenum, atol=0.001))[0][0]

        # Generate batch
        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i:i + batch_size]
            batch_of_nc_indices = indices_in_nc[i:i + batch_size]
            # Extract ground truth GHI from dataframe
            targets_np = np.zeros([len(batch_of_datetimes), output_seq_len])
            for j, dt in enumerate(batch_of_datetimes):
                for m in range(output_seq_len):
                    k = dataframe.index.get_loc(dt + target_time_offsets[m])
                    targets_np[j, m] = dataframe[f"{station_name}_GHI"][k]

            samples = tf.convert_to_tensor(nc_var_data[batch_of_nc_indices, :, :, :, :])
            targets = tf.convert_to_tensor(targets_np)

            yield samples, targets

    return tf.data.Dataset.from_generator(
        image_generator, (tf.float32, tf.float32)
    )
