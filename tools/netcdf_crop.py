import os
import json
import pickle
import datetime
import argparse

import numpy as np
import numpy.ma as ma
import netCDF4
import tqdm

# Script for preprocessing crop around stations
# Usage: python netcdf_crop.py cfg_file.json [crop_size] [path_output]


def netcdf_preloader(cfg_file, crop_size=50, path_output='.',
                     mem_size=1000000000):
    # hard coded values for now
    n_channels = 5
    n_timestep = 5
    batch_size = 140  # multiple of 7 (nb of training stations)
    n_stations = 7

    # Calculating optimal variable dimensions
    sample_size = (crop_size ** 2) * n_channels * n_timestep * (32 / 8)
    max_mem_batch_size = int(mem_size / sample_size)
    mem_batch_size = max_mem_batch_size - (max_mem_batch_size % batch_size)
    if mem_batch_size == 0:
        mem_batch_size = batch_size

    dc = crop_size // 2
    delta_t = datetime.timedelta(minutes=15)
    delta_1h = datetime.timedelta(hours=1)

    cfg_name = os.path.basename(cfg_file).split('.')[0]
    with open(cfg_file, 'r') as cfg_file_handler:
        cfg = json.loads(cfg_file_handler.read())

    with open(cfg['dataframe_path'], 'rb') as df_file_handler:
        df = pickle.load(df_file_handler)

    # Generate all datetimes including prior timesteps from targets
    all_datetimes = [datetime.datetime.fromisoformat(x) for x in cfg['target_datetimes']]

    # Loop through all timesteps and load images
    flag_initialize_tmp_arrays = True
    flag_first_nc = True
    file_id = 0
    n_samples = 0
    for t_id, sample_datetime in enumerate(tqdm.tqdm(all_datetimes)):
        if flag_initialize_tmp_arrays:
            tmp_array_data = ma.masked_all(
                (mem_batch_size, n_timestep, n_channels, crop_size, crop_size))
            tmp_array_time = ma.masked_all((mem_batch_size,))
            tmp_stations_crop_coord = {}
            tmp_station_lat = ma.masked_all((mem_batch_size,))
            tmp_station_lon = ma.masked_all((mem_batch_size,))
            tmp_metadata = ma.masked_all((mem_batch_size, 8))
            tmp_ghi_targets = ma.masked_all((mem_batch_size, 4))
            flag_initialize_tmp_arrays = False
        for ts_id, steps_in_the_past in enumerate(range(n_timestep - 1, 0, -1)):
            target_datetime = sample_datetime - steps_in_the_past * delta_t
            try:
                k = df.index.get_loc(target_datetime)
                nc_path = df['ncdf_path'][k]
                nc_loop = netCDF4.Dataset(nc_path, 'r')
            except (KeyError, OSError):
                continue
            if flag_first_nc:
                lat_loop = nc_loop['lat'][:]
                lon_loop = nc_loop['lon'][:]
                for station, coord in cfg['stations'].items():
                    lat_diff = np.abs(lat_loop - coord[0])
                    i = np.where(lat_diff == lat_diff.min())[0][0]
                    lon_diff = np.abs(lon_loop - coord[1])
                    j = np.where(lon_diff == lon_diff.min())[0][0]
                    tmp_stations_crop_coord[station] = (
                        (i, j), lat_loop[i - dc:i + dc], lon_loop[j - dc:j + dc]
                    )
                flag_first_nc = False

            for d, c in enumerate([1, 2, 3, 4, 6]):
                channel_data = nc_loop.variables[f'ch{c}'][0, :, :]
                for station_num, (station, coord) in enumerate(cfg['stations'].items()):
                    i, j = tmp_stations_crop_coord[station][0]
                    if c == 1:
                        tmp_array_data[n_samples + station_num, ts_id, d, :, :] = \
                            channel_data[i - dc:i + dc, j - dc:j + dc]
                    else:
                        tmp_array_data[n_samples + station_num, ts_id, d, :, :] = \
                            channel_data[i - dc:i + dc, j - dc:j + dc] / 400.
                    tmp_station_lat[n_samples + station_num] = coord[0]
                    tmp_station_lon[n_samples + station_num] = coord[1]
                    tmp_metadata[n_samples + station_num, 5] = coord[0] / 180.
                    tmp_metadata[n_samples + station_num, 6] = coord[1] / 360.
                    tmp_metadata[n_samples + station_num, 7] = coord[2] / 10000.
                    if d == 0:
                        for step_id, hour_step in enumerate([0, 1, 3, 6]):
                            target_datetime = sample_datetime + hour_step * delta_1h
                            try:
                                k = df.index.get_loc(target_datetime)
                                tmp_ghi_targets[n_samples + station_num, step_id] = \
                                    df[f"{station}_GHI"][k]
                            except KeyError:
                                continue

        tmp_array_time[n_samples:n_samples + n_stations] = \
            netCDF4.date2num(sample_datetime, 'days since 1970-01-01 00:00:00')
        tmp_metadata[n_samples:n_samples + n_stations, 0] = sample_datetime.year / 2020.
        tmp_metadata[n_samples:n_samples + n_stations, 1] = sample_datetime.month / 12.
        tmp_metadata[n_samples:n_samples + n_stations, 2] = sample_datetime.day / 31.
        tmp_metadata[n_samples:n_samples + n_stations, 3] = sample_datetime.hour / 24.
        tmp_metadata[n_samples:n_samples + n_stations, 4] = sample_datetime.minute / 60.

        n_samples += n_stations

        if (n_samples == mem_batch_size) or (t_id == len(all_datetimes) - 1):
            nc_out = netCDF4.Dataset(
                os.path.join(path_output, f'preloader_{cfg_name}_{file_id.zfill(3)}.nc'), 'w')
            nc_out.createDimension('time', n_samples)
            nc_out.createDimension('lat', crop_size)
            nc_out.createDimension('lon', crop_size)
            nc_out.createDimension('channel', n_channels)
            nc_out.createDimension('timestep', n_timestep)
            nc_out.createDimension('metadata', 8)
            nc_out.createDimension('ghi', 4)
            time = nc_out.createVariable('time', 'f8', ('time',))
            time.calendar = 'standard'
            time.units = 'days since 1970-01-01 00:00:00'
            nc_out.createVariable('lat', 'f4', ('lat',))
            nc_out.createVariable('lon', 'f4', ('lon',))
            station_lat = nc_out.createVariable('station_lat', 'f4', ('time',))
            station_lon = nc_out.createVariable('station_lon', 'f4', ('time',))
            data_out = nc_out.createVariable(
                f'data', 'f4', ('time', 'timestep', 'channel', 'lat', 'lon'), zlib=True)
            data_out[:, :, :, :, :] = ma.filled(tmp_array_data[0:n_samples, :, :, :, :], 0)
            time[:] = tmp_array_time[0:n_samples]
            station_lat[:] = tmp_station_lat[0:n_samples]
            station_lon[:] = tmp_station_lon[0:n_samples]
            for station, coord in cfg['stations'].items():
                lat_out = nc_out.createVariable(f'lat_{station}', 'f4', ('lat',))
                lat_out[:] = tmp_stations_crop_coord[station][1]
                lon_out = nc_out.createVariable(f'lon_{station}', 'f4', ('lon',))
                lon_out[:] = tmp_stations_crop_coord[station][2]
            metadata = nc_out.createVariable(f'metadata', 'f4', ('time', 'metadata'))
            metadata[:, :] = tmp_metadata[0:n_samples, :]
            ghi_targets = nc_out.createVariable(f'ghi_targets', 'f4', ('time', 'ghi'))
            ghi_targets[:, :] = tmp_ghi_targets[0:n_samples, :]
            nc_out.close()
            file_id += 1
            n_samples = 0
            flag_first_nc = True
            flag_initialize_tmp_arrays = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='admin cfg file')
    parser.add_argument('-c', '--crop_size', type=int, default=50,
                        help='crop size')
    parser.add_argument('-o', '--output_path', type=str, default='.',
                        help='output path')
    parser.add_argument('-m', '--mem_size', type=int, default=1000000000,
                        help='memory size allowed for arrays')
    args = parser.parse_args()
    netcdf_preloader(args.cfg_file, args.crop_size, args.output_path,
                     args.mem_size)
