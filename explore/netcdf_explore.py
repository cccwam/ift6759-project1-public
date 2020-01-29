import json
import numpy as np
import netCDF4

with open('project1_cfg_validation.json', 'r') as fsf:
    dj1 = json.loads(fsf.read())

nc = netCDF4.Dataset('../data/netcdf/GridSat-CONUS.goes13.2015.06.01.0015.v01.nc', 'r')

lat = nc.variables['lat']
lon = nc.variables['lon']
ch1 = nc.variables['ch1']
ch2 = nc.variables['ch2']
ch3 = nc.variables['ch3']
ch4 = nc.variables['ch4']
ch5 = nc.variables['ch5']
ch6 = nc.variables['ch6']

# dj1.stations

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)


ax.pcolormesh(lon[:], lat[:], ch1[0,:,:])
#ax.coastlines()
#ax.set_global()
plt.savefig('out.png')
