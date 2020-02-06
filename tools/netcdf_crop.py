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


def netcdf_preloader(cfg_file, crop_size=50, path_output='.',
                     debug_mnt_path=None, tmp_array_size=1000):
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

    nc_outs = {}
    for station, coord in cfg['stations'].items():
        nc_outs[station] = netCDF4.Dataset(
            os.path.join(path_output, f'preloader_{cfg_name}_{station}.nc'),
            'w')
        nc_outs[station].createDimension('time', len(all_dt))
        nc_outs[station].createDimension('lat', crop_size)
        nc_outs[station].createDimension('lon', crop_size)
        time = nc_outs[station].createVariable('time', 'f8', ('time',))
        time.calendar = 'standard'
        time.units = 'days since 1970-01-01 00:00:00'
        lat = nc_outs[station].createVariable('lat', 'f4', ('lat',))
        lon = nc_outs[station].createVariable('lon', 'f4', ('lon',))
        channels = []
        for c in [1, 2, 3, 4, 6]:
            channels.append(
                nc_outs[station].createVariable(
                    f'ch{c}', 'f4', ('time', 'lat', 'lon'), zlib=True,
                    chunksizes=(chunksizes, crop_size, crop_size)))

    init = True
    tmp_arrays = {}
    coord_ij = {}
    for station, coord in cfg['stations'].items():
        for d, c in enumerate([1, 2, 3, 4, 6]):
            tmp_arrays[f"{station}_ch{str(c)}"] = ma.masked_all((tmp_array_size, crop_size, crop_size))
    tmp_arrays["time"] = ma.masked_all((tmp_array_size,))
    for t, dt in enumerate(tqdm.tqdm(all_dt)):
        nt_tmp = t % tmp_array_size
        k = df.index.get_loc(dt)
        nc_path = df['ncdf_path'][k]
        if debug_mnt_path:
            nc_path = os.path.join(debug_mnt_path, nc_path.lstrip('/'))
        try:
            nc_loop = netCDF4.Dataset(nc_path, 'r')
        except OSError:
            # What to do when the netCDF4 file is not available.
            # Currently filling with 0 below...
            tmp_arrays["time"][nt_tmp] = netCDF4.date2num(dt, time.units, time.calendar)
        else:
            if init:
                lat_loop = nc_loop['lat'][:]
                lon_loop = nc_loop['lon'][:]
                for station, coord in cfg['stations'].items():
                    lat_diff = np.abs(lat_loop - coord[0])
                    i = np.where(lat_diff == lat_diff.min())[0][0]
                    lon_diff = np.abs(lon_loop - coord[1])
                    j = np.where(lon_diff == lon_diff.min())[0][0]
                    nc_outs[station].variables['lat'][:] = lat_loop[i - dc:i + dc]
                    nc_outs[station].variables['lon'][:] = lon_loop[j - dc:j + dc]
                    coord_ij[station] = (i, j)
                init = False
            for d, c in enumerate([1, 2, 3, 4, 6]):
                channel_data = nc_loop.variables[f'ch{c}'][0,:,:]
                for station, coord in cfg['stations'].items():
                    i, j = coord_ij[station]
                    tmp_arrays[f"{station}_ch{str(c)}"][nt_tmp, :, :] = channel_data[i - dc:i + dc, j - dc:j + dc]
            tmp_arrays["time"][nt_tmp] = nc_loop.variables['time'][0]
            nc_loop.close()
        if (nt_tmp == (tmp_array_size - 1)) or (t == (len(all_dt) - 1)):
            t0 = t - nt_tmp
            for station, coord in cfg['stations'].items():
                for d, c in enumerate([1, 2, 3, 4, 6]):
                    # Here we fill missing values with 0
                    nc_outs[station][f'ch{c}'][t0:t+1, :, :] = ma.filled(tmp_arrays[f"{station}_ch{str(c)}"][:nt_tmp+1,:,:], 0)
                    tmp_arrays[f"{station}_ch{str(c)}"] = ma.masked_all(
                        (tmp_array_size, crop_size, crop_size))
                nc_outs[station]['time'][t0:t+1] = tmp_arrays['time'][:nt_tmp+1]
            tmp_arrays["time"] = ma.masked_all((tmp_array_size,))

    for station, coord in cfg['stations'].items():
        nc_outs[station].close()


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
