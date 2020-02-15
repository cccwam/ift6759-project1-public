import datetime
import os
import typing

import netCDF4
import numpy as np
import pandas as pd
import tensorflow as tf

import libs.helpers


# TODO to remove this old dataloader (see with Blaise)
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

        # batch_size = 1
        output_seq_len = 4
        # Currently only support one station at a time.
        #        if len(list(stations.keys())) > 1: # TODO fix this
        #            raise NotImplementedError()

        station_name = list(stations.keys())[0]
        data_file = f"preloader_{config['data_loader']['hyper_params']['admin_name']}_{station_name}.nc"

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

        # Remove night values and nan NCDF paths
        dataframe_preprocessed = libs.helpers.remove_night_values(dataframe)
        dataframe_preprocessed = libs.helpers.remove_null_path(dataframe_preprocessed)

        # TODO take care of nan GHI values
        # dataframe_preprocessed = libs.helpers.fill_ghi(dataframe_preprocessed)

        # Generate batch
        # TODO what are the implications of this change on performance
        # now there is a lot more i/o on the netcdf files...
        # I change this to yield only one sample at a time
        for i in range(0, len(target_datetimes)):
            # Extract ground truth GHI from dataframe
            targets_np = np.zeros([output_seq_len], dtype='float32')
            for m in range(output_seq_len):
                k = dataframe_preprocessed.index.get_loc(target_datetimes[i] + target_time_offsets[m])
                targets_np[m] = dataframe_preprocessed[f"{station_name}_GHI"][k]

            samples = tf.convert_to_tensor(nc_var_data[indices_in_nc[i], :, :, :, :] / 100.)
            targets = tf.convert_to_tensor(targets_np)

            assert samples.dtype == tf.float32
            assert targets.dtype == tf.float32

            samples = tf.clip_by_value(samples, clip_value_min=0, clip_value_max=1)
            targets = tf.clip_by_value(targets, clip_value_min=-70, clip_value_max=450)

            yield samples, targets

    return tf.data.Dataset.from_generator(
        image_generator, (tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([5, 5, 50, 50]), tf.TensorShape([4]))
    ).batch(256)


def data_loader_images_multimodal(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        data_mode='train',
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

        batch_size = 200
        output_seq_len = 4

        for station_name in stations.keys():
            data_file = f"preloader_{config['data_loader']['hyper_params']['admin_name']}_{station_name}.nc"
            if data_mode == 'validation':
                data_file = data_file.replace('_train_', '_validation_')

            nc = netCDF4.Dataset(
                os.path.join('/project/cq-training-1/project1/teams/team03/data',
                             data_file), 'r')
            nc_var = nc.variables['data']
            # Loading all data in memory greatly speeds up things, but if we move to much larger
            # sample size and crop size this might need to be done differently.
            nc_var_data = nc_var[:, :, :, :, :]
            nc_time = nc.variables['time']

            # match target datenums with indices in the netcdf file, we need to allow for
            # small mismatch in seconds due to the nature of num2date and date2num.
            target_datenums = netCDF4.date2num(target_datetimes, nc_time.units,
                                               nc_time.calendar)
            nc_time_data = nc_time[:]
            indices_in_nc = np.zeros(len(target_datenums), dtype='i8')
            for i, target_datenum in enumerate(target_datenums):
                indices_in_nc[i] = \
                    np.where(np.isclose(nc_time_data, target_datenum, atol=0.001))[0][0]

            # Generate batch
            for i in range(0, len(target_datetimes), batch_size):
                batch_of_datetimes = target_datetimes[i:i + batch_size]
                # ToDo: how to deal with the last batch with different size
                # if len(batch_of_datetimes) != batch_size:
                #     continue
                batch_of_nc_indices = indices_in_nc[i:i + batch_size]
                # Datetime metadata
                metadata = np.zeros([len(batch_of_datetimes), 8],
                                    dtype=np.float32)
                metadata[:, 0] = [od.year for od in batch_of_datetimes]
                metadata[:, 0] /= 2020
                metadata[:, 1] = [od.month for od in batch_of_datetimes]
                metadata[:, 1] /= 12
                metadata[:, 2] = [od.day for od in batch_of_datetimes]
                metadata[:, 2] /= 31
                metadata[:, 3] = [od.hour for od in batch_of_datetimes]
                metadata[:, 3] /= 24
                metadata[:, 4] = [od.minute for od in batch_of_datetimes]
                metadata[:, 4] /= 60
                metadata[:, 5] = stations[station_name][0] / 180.
                metadata[:, 5] = stations[station_name][1] / 360.
                metadata[:, 5] = stations[station_name][2] / 10000.
                # Extract ground truth GHI from dataframe
                targets_np = np.zeros([len(batch_of_datetimes), output_seq_len],
                                      dtype=np.float32)
                for j, dt in enumerate(batch_of_datetimes):
                    for m in range(output_seq_len):
                        k = dataframe.index.get_loc(dt + target_time_offsets[m])
                        targets_np[j, m] = dataframe[f"{station_name}_GHI"][k]

                tmp_data = nc_var_data[batch_of_nc_indices, :, :, :, :]
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
            ((tf.TensorShape([None, None, 5, 50, 50]), tf.TensorShape([None, 8])),
             tf.TensorShape([None, 4])))
    )


# TODO discuss with BLaise
#  this version is faster using the batch feature of tf.dataset when using multiple GPU
# The idea is to parallize as much as possible all data loading processes
def data_loader_images_multimodal_fm(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        data_mode='train',
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

        output_seq_len = 4

        for station_name in stations.keys():
            data_file = f"preloader_{config['data_loader']['hyper_params']['admin_name']}_{station_name}.nc"
            if data_mode == 'validation':
                data_file = data_file.replace('_train_', '_validation_')

            nc = netCDF4.Dataset(
                os.path.join('/project/cq-training-1/project1/teams/team03/data',
                             data_file), 'r')
            nc_var = nc.variables['data']
            # Loading all data in memory greatly speeds up things, but if we move to much larger
            # sample size and crop size this might need to be done differently.
            # TODO to not load everything in memory
            nc_var_data = nc_var[:, :, :, :, :]
            nc_time = nc.variables['time']

            # match target datenums with indices in the netcdf file, we need to allow for
            # small mismatch in seconds due to the nature of num2date and date2num.
            target_datenums = netCDF4.date2num(target_datetimes, nc_time.units,
                                               nc_time.calendar)
            nc_time_data = nc_time[:]
            indices_in_nc = np.zeros(len(target_datenums), dtype='i8')
            for i, target_datenum in enumerate(target_datenums):
                indices_in_nc[i] = \
                    np.where(np.isclose(nc_time_data, target_datenum, atol=0.001))[0][0]

            # Generate batch
            for i in range(0, len(target_datetimes)):

                metadata = np.zeros([8],
                                    dtype=np.float32)
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
                targets_np = np.zeros([output_seq_len],
                                      dtype=np.float32)
                for m in range(output_seq_len):
                    k = dataframe.index.get_loc(target_datetimes[i] + target_time_offsets[m])
                    targets_np[m] = dataframe[f"{station_name}_GHI"][k]

                tmp_data = nc_var_data[indices_in_nc[i], :, :, :]
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
