import os
import sys
import json
import pickle
import datetime

import numpy as np
import netCDF4
import tqdm

# Script for preprocessing crop around stations
# Usage: python netcdf_crop.py cfg_file.json [crop_size] [path_output]


def netcdf_preloader(cfg_file, crop_size=50, path_output='.'):
    dc = crop_size // 2
    with open(cfg_file, 'r') as f:
        cfg = json.loads(f.read())

    with open(cfg['dataframe_path'], 'rb') as f:
        df = pickle.load(f)

    dt0 = datetime.datetime(*list(map(int, cfg['start_bound'].split('-'))))
    ddt = datetime.timedelta(minutes=15)
    dtf = datetime.datetime(*list(map(int, cfg['end_bound'].split('-'))))

    all_dt = []
    next_dt = dt0
    while next_dt < dtf:
        all_dt.append(next_dt)
        next_dt = all_dt[-1] + ddt

    for station, coord in cfg['stations'].items():
        nc = netCDF4.Dataset(
            os.path.join(path_output, f'preloader_{station}.nc'), 'w')
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
                    chunksizes=(480, crop_size, crop_size)))

        init = True
        for t, dt in enumerate(tqdm.tqdm(all_dt)):
            k = df.index.get_loc(dt)
            try:
                nc_loop = netCDF4.Dataset(df['ncdf_path'][k], 'r')
            except OSError:
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
                channels[d][t, :, :] = ncvar[0, i - dc:i + dc, j - dc:j + dc]
            nc_loop.close()

        nc.close()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        crop = int(sys.argv[2])
    else:
        crop = 50
    if len(sys.argv) > 3:
        path_out = sys.argv[3]
    else:
        path_out = '.'
    netcdf_preloader(sys.argv[1], crop, path_out)
