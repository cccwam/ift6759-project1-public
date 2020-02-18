import argparse
import datetime
import json
import os
import pickle
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import numpy.ma as ma
import tqdm

from libs import helpers


# Script for preprocessing crop around stations
# Usage: python netcdf_crop.py cfg_file.json [crop_size] [path_output]


def netcdf_preloader(
        crop_size=50,
        path_output='.',
        tmp_array_size=200,
        cfg_file=None,
        dataframe=None,
        target_datetimes=None,
        stations=None
):
    if cfg_file:
        cfg_dict = helpers.load_dict(cfg_file)
        dataframe_path = cfg_dict['dataframe_path']
        with open(dataframe_path, 'rb') as df_file_handler:
            dataframe = pickle.load(df_file_handler)
        target_datetimes = [datetime.strptime(s, '%Y-%m-%dT%H:%M:%S') for s in cfg_dict['target_datetimes']]
        stations = cfg_dict['stations']

    # hard coded values for now
    n_channels = 5
    n_timestep = 5

    dc = crop_size // 2
    ddt = datetime.timedelta(minutes=15)

    cfg_name = os.path.basename(cfg_file).split('.')[0]

    # number of sample points
    n_sample = len(target_datetimes)

    # Generate all datetimes including prior timesteps from targets
    all_dt = []
    for dt_str in target_datetimes:
        dt0 = datetime.datetime.fromisoformat(dt_str)
        for i in range(4, 0, -1):
            all_dt.append(dt0 - i * ddt)
        all_dt.append(dt0)

    chunksizes = 256
    if len(all_dt) < 256:
        chunksizes = len(all_dt)

    # Initialize output netcdf files (one for each station)
    nc_outs = {}
    for station, coord in stations.items():
        nc_outs[station] = netCDF4.Dataset(
            os.path.join(path_output, f'preloader_{cfg_name}_{station}.nc'),
            'w')
        nc_outs[station].createDimension('time', n_sample)
        nc_outs[station].createDimension('lat', crop_size)
        nc_outs[station].createDimension('lon', crop_size)
        nc_outs[station].createDimension('channel', n_channels)
        nc_outs[station].createDimension('timestep', n_timestep)
        time = nc_outs[station].createVariable('time', 'f8', ('time',))
        time.calendar = 'standard'
        time.units = 'days since 1970-01-01 00:00:00'
        nc_outs[station].createVariable('lat', 'f4', ('lat',))
        nc_outs[station].createVariable('lon', 'f4', ('lon',))
        nc_outs[station].createVariable(
            f'data', 'f4', ('time', 'timestep', 'channel', 'lat', 'lon'), zlib=True,
            chunksizes=(chunksizes, n_timestep, n_channels, crop_size, crop_size))

    # Initialize temporary arrays to store data to limit constantly writing to disk
    init = True
    tmp_arrays = {}
    coord_ij = {}
    for station, coord in stations.items():
        tmp_arrays[f"{station}"] = ma.masked_all((tmp_array_size, n_timestep, n_channels, crop_size, crop_size))
    tmp_arrays["time"] = ma.masked_all((tmp_array_size,))

    # Loop through all timesteps and load images
    for t, dt in enumerate(tqdm.tqdm(all_dt)):
        at_t0 = not ((t + 1) % n_timestep)
        t_sample = t // n_timestep
        t_sample_tmp = t_sample % tmp_array_size
        timestep_id = t % n_timestep
        k = dataframe.index.get_loc(dt)
        nc_path = dataframe['ncdf_path'][k]
        try:
            nc_loop = netCDF4.Dataset(nc_path, 'r')
        except OSError:
            # What to do when the netCDF4 file is not available.
            # Currently filling with 0 later in the code...
            if at_t0:
                tmp_arrays["time"][t_sample_tmp] = netCDF4.date2num(dt, time.units, time.calendar)
        else:
            if init:
                lat_loop = nc_loop['lat'][:]
                lon_loop = nc_loop['lon'][:]
                for station, coord in stations.items():
                    lat_diff = np.abs(lat_loop - coord[0])
                    i = np.where(lat_diff == lat_diff.min())[0][0]
                    lon_diff = np.abs(lon_loop - coord[1])
                    j = np.where(lon_diff == lon_diff.min())[0][0]
                    nc_outs[station].variables['lat'][:] = lat_loop[i - dc:i + dc]
                    nc_outs[station].variables['lon'][:] = lon_loop[j - dc:j + dc]
                    coord_ij[station] = (i, j)
                init = False

            for d, c in enumerate([1, 2, 3, 4, 6]):
                channel_data = nc_loop.variables[f'ch{c}'][0, :, :]
                for station, coord in stations.items():
                    i, j = coord_ij[station]
                    tmp_arrays[f"{station}"][t_sample_tmp, timestep_id, d, :, :] = \
                        channel_data[i - dc:i + dc, j - dc:j + dc]

            if at_t0:
                tmp_arrays["time"][t_sample_tmp] = nc_loop.variables['time'][0]

            nc_loop.close()

        if ((t_sample_tmp == (tmp_array_size - 1)) and (timestep_id == n_timestep - 1)) or (t == (len(all_dt) - 1)):
            t0 = t_sample - t_sample_tmp
            for station, coord in stations.items():
                # Here we fill missing values with 0
                nc_outs[station]['data'][t0:t_sample + 1, :, :, :, :] = \
                    ma.filled(tmp_arrays[f"{station}"][:t_sample_tmp + 1, :, :, :, :], 0)
                tmp_arrays[f"{station}"] = \
                    ma.masked_all((tmp_array_size, n_timestep, n_channels, crop_size, crop_size))
                nc_outs[station]['time'][t0:t_sample + 1] = \
                    tmp_arrays['time'][:t_sample_tmp + 1]
            tmp_arrays["time"] = ma.masked_all((tmp_array_size,))

    for station, coord in stations.items():
        nc_outs[station].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='admin cfg file')
    parser.add_argument('-c', '--crop_size', type=int, default=50,
                        help='crop size')
    parser.add_argument('-o', '--output_path', type=str, default='.',
                        help='output path')
    args = parser.parse_args()
    netcdf_preloader(args.cfg_file, args.crop_size, args.output_path)
