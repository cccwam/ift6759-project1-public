import datetime
import os
import typing

import netCDF4
import numpy as np
import pandas as pd
import tensorflow as tf


# The idea is to parallize as much as possible all data loading processes
def data_loader_images_multimodal(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        preprocessed_data=None
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
        preprocessed_data: A path to the folder containing the preprocessed data or an in-memory data structure

    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """

    def image_generator():
        """Baseline generator with time/lat/lon/elev."""

        output_seq_len = 4

        for station_name in stations.keys():
            # Generate batch
            i_load_min = 0
            i_load_max = 5000

            if not isinstance(preprocessed_data, str):
                nc_var_data, nc_time_data, time_units = preprocessed_data[station_name]
            else:
                data_file = f"{station_name}.nc"
                nc = netCDF4.Dataset(os.path.join(preprocessed_data, data_file), 'r')
                nc_var = nc.variables['data']
                nc_time = nc.variables['time']
                time_units = nc_time.units
                nc_time_data = nc_time[:]
                nc_var_data = nc_var[i_load_min:i_load_max, :, :, :, :]

            # match target datenums with indices in the netcdf file, we need to allow for
            # small mismatch in seconds due to the nature of num2date and date2num.
            target_datenums = netCDF4.date2num(target_datetimes, time_units)
            indices_in_nc = np.zeros(len(target_datenums), dtype='i8')
            for i, target_datenum in enumerate(target_datenums):
                indices_in_nc[i] = \
                    np.where(np.isclose(nc_time_data, target_datenum, atol=0.001))[0][0]

            nc_var_data = nc_var[i_load_min:i_load_max, :, :, :, :]
            for i in range(0, len(target_datetimes)):
                metadata = np.zeros([8], dtype=np.float32)
                metadata[0] = target_datetimes[i].year
                metadata[0] /= 2020
                metadata[1] = target_datetimes[i].month
                metadata[1] /= 12
                metadata[2] = target_datetimes[i].day
                metadata[2] /= 31
                metadata[3] = target_datetimes[i].hour
                metadata[3] /= 24
                metadata[4] = target_datetimes[i].minute
                metadata[4] /= 60
                metadata[5] = stations[station_name][0] / 180.
                metadata[6] = stations[station_name][1] / 360.
                metadata[7] = stations[station_name][2] / 10000.
                # Extract ground truth GHI from dataframe
                targets_np = np.zeros([output_seq_len], dtype=np.float32)
                for m in range(output_seq_len):
                    k = dataframe.index.get_loc(target_datetimes[i] + target_time_offsets[m])
                    targets_np[m] = dataframe[f"{station_name}_GHI"][k]

                # Loading all data in memory greatly speeds up things, but if we move to much larger
                # sample size and crop size this might need to be done differently.
                # Let's hack something to save memory
                if indices_in_nc[i] >= i_load_max:
                    i_load_min += 5000
                    i_load_max += 5000
                    nc_var_data = nc_var[i_load_min:i_load_max, :, :, :, :]
                tmp_data = nc_var_data[indices_in_nc[i] - i_load_min, :, :, :, :]
                # tmp_data = tmp_data.reshape([tmp_data.shape[0], 25, 50, 50])
                assert not np.any(np.isnan(tmp_data))
                targets_np = np.nan_to_num(targets_np)
                assert not np.any(np.isnan(targets_np))
                assert not np.any(np.isnan(metadata))
                images = tf.convert_to_tensor(
                    tmp_data / 400.0)
                metadata = tf.convert_to_tensor(metadata)
                targets = tf.convert_to_tensor(targets_np)

                assert images.dtype == tf.float32
                assert metadata.dtype == tf.float32
                assert targets.dtype == tf.float32

                yield (images, metadata), targets

    return tf.data.Dataset.from_generator(
        image_generator, ((tf.float32, tf.float32), tf.float32),
        output_shapes=(
            ((tf.TensorShape([None, 5, 50, 50]), tf.TensorShape([8])),
             tf.TensorShape([4])))
    )
