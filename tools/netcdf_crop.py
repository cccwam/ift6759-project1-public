import os
import sys
import json
import pickle
import datetime
import multiprocessing as mp

import numpy as np
import numpy.ma as ma
import netCDF4
import tqdm

# Script for preprocessing crop around stations
# Usage: python netcdf_crop.py cfg_file.json [crop_size] [path_output]


def write_netcdf(path_output, cfg_name, station, all_dt, crop_size, chunksizes,
                 df, debug_mnt_path, coord, dc):
    nc = netCDF4.Dataset(
        os.path.join(path_output, f'preloader_{cfg_name}_{station}.nc'), 'w')
    nc.createDimension('time', len(all_dt))
    nc.createDimension('lat', crop_size)
    nc.createDimension('lon', crop_size)
    time = nc.createVariable('time', 'f8', ('time',))
    time.calendar = 'standard'
    time.units = 'days since 1970-01-01 00:00:00'
    lat = nc.createVariable('lat', 'f4', ('lat',))
    lon = nc.createVariable('lon', 'f4', ('lon',))
    channels = []
    for c in [1, 2, 3, 4, 6]:
        channels.append(
            nc.createVariable(
                f'ch{c}', 'f4', ('time', 'lat', 'lon'), zlib=True,
                chunksizes=(chunksizes, crop_size, crop_size)))

    init = True
    for t, dt in enumerate(tqdm.tqdm(all_dt)):
        k = df.index.get_loc(dt)
        nc_path = df['ncdf_path'][k]
        if debug_mnt_path:
            nc_path = os.path.join(debug_mnt_path, nc_path.lstrip('/'))
        try:
            nc_loop = netCDF4.Dataset(nc_path, 'r')
        except OSError:
            # What to do when the netCDF4 file is not available.
            # Currently set everything to 0
            for d, c in enumerate([1, 2, 3, 4, 6]):
                channels[d][t, :, :] = np.zeros(channels[d].shape)
            time[t] = netCDF4.date2num(dt, time.units, time.calendar)
            continue
        if init:
            lat_loop = nc_loop['lat'][:]
            lon_loop = nc_loop['lon'][:]
            lat_diff = np.abs(lat_loop - coord[0])
            i = np.where(lat_diff == lat_diff.min())[0][0]
            lon_diff = np.abs(lon_loop - coord[1])
            j = np.where(lon_diff == lon_diff.min())[0][0]
            lat[:] = lat_loop[i - dc:i + dc]
            lon[:] = lon_loop[j - dc:j + dc]
            init = False
        for d, c in enumerate([1, 2, 3, 4, 6]):
            ncvar = nc_loop.variables[f'ch{c}']
            # Here we fill missing values with 0
            channels[d][t, :, :] = ma.filled(
                ncvar[0, i - dc:i + dc, j - dc:j + dc], 0)
            time[t] = nc_loop.variables['time'][0]
        nc_loop.close()

    nc.close()


def netcdf_preloader(cfg_file, crop_size=50, path_output='.',
                     debug_mnt_path=None):
    dc = crop_size // 2
    cfg_name = os.path.basename(cfg_file).split('.')[0]
    with open(cfg_file, 'r') as cfg_file_handler:
        cfg = json.loads(cfg_file_handler.read())

    with open(cfg['dataframe_path'], 'rb') as df_file_handler:
        df = pickle.load(df_file_handler)

    ddt = datetime.timedelta(minutes=15)

    all_dt = []
    for dt_str in cfg['target_datetimes']:
        dt0 = datetime.datetime.fromisoformat(dt_str)
        for i in range(4, 0, -1):
            all_dt.append(dt0 - i * ddt)
        all_dt.append(dt0)

    chunksizes = 480
    if len(all_dt) < 480:
        chunksizes = len(all_dt)

    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(write_netcdf, args=(
        path_output, cfg_name, station, all_dt, crop_size, chunksizes,
        df, debug_mnt_path, coord, dc)) for station, coord in cfg['stations'].items()]
    for result in results:
        result.wait()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        crop = int(sys.argv[2])
    else:
        crop = 50
    if len(sys.argv) > 3:
        path_out = sys.argv[3]
    else:
        path_out = '.'
    if len(sys.argv) > 4:
        mnt_path = sys.argv[4]
    else:
        mnt_path = None
    netcdf_preloader(sys.argv[1], crop, path_out, mnt_path)
