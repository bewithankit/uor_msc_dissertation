import numpy as np
import scipy as sp
import sys
import os
from netCDF4 import Dataset
import netCDF4 as nc
%matplotlib inline
import matplotlib
import xarray as xr
import matplotlib.pyplot as plt
import warnings

import matplotlib.pyplot as plt
import matplotlib.path as mpath
# Quick plot to show the results
from cartopy import config
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import scipy.ndimage as ndimage
#from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.basemap import addcyclic
from cartopy.util import add_cyclic_point
import matplotlib.pylab as pl
from scipy import stats
import glob

def preprocess(ds):
    print(ds.encoding['source'])
    return(ds)

### observation

temp_min = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['tmn']
temp_max = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmx.dat.nc')['tmx']
lat_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['lat']
lon_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['lon']


weights_obs = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_obs.name = "weights_obs"



###

temp_min_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')
temp_max_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmx.dat.nc')
lat_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['lat']
lon_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['lon']
weights2 = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights2.name = "weights2"

global_max_obs=temp_max_obs["tmx"]

global_min_obs=temp_min_obs["tmn"]

global_average_tmax_obs=global_max_obs.weighted(weights2).mean(("lon", "lat"))
annual_tmax_gm_obs=global_average_tmax_obs.groupby('time.year').mean('time')

global_average_tmin_obs=global_min_obs.weighted(weights2).mean(("lon", "lat"))
annual_tmin_gm_obs=global_average_tmin_obs.groupby('time.year').mean('time')
dtr_obs = annual_tmax_gm_obs - annual_tmin_gm_obs


# colors = plt.cm.rainbow(np.linspace(0, 1, 5))
# for i in scalings:
#     plt.plot(x, i*scalings[i], color=colors[i])
# #plt.xlim(4, 6)
# plt.show()

# evenly_spaced_interval = np.linspace(0, 1, len(scalings))
# colors = [plt.cm.rainbow(x) for x in evenly_spaced_interval]
# for i, color in enumerate(colors):
#     plt.plot(dtr_0p2.year,scalings[i], color = color, alpha =0.5)


fig = plt.figure()

fig, ax = plt.subplots()

ax.plot(dtr_obs.year[(dtr_obs.year > 1900) & (dtr_obs.year < 2015)], dtr_obs[(dtr_obs.year > 1900) & (dtr_obs.year < 2015)], linestyle='solid', color='black', linewidth=0.8)

ax.set(title='CRU TS v4 Global Mean Diurnal Temperature Range (DTR)\n variation (1901-2014)',
     xlabel='Years', ylabel='$^\circ$C') 

plt.savefig('/Users/ankit/Documents/UoR/Dissertation/obsdtr.png', format='png', dpi=1200)

plt.show()


