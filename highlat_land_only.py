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

lat = xr.open_dataset("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc")['latitude']
lon = xr.open_dataset("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc")['longitude']

maxfiles_0p2 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")
minfiles_0p2 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")

maxfiles_0p4 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p4/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")
minfiles_0p4 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p4/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")

maxfiles_0p7 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p7/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")
minfiles_0p7 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p7/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")

maxfiles_1p0 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
minfiles_1p0 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

maxfiles_1p5 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p5/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")
minfiles_1p5 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")

preprocess=preprocess

temp_max_0p2 = xr.open_mfdataset(maxfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_0p2 = xr.open_mfdataset(minfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_0p4 = xr.open_mfdataset(maxfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_0p4 = xr.open_mfdataset(minfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_0p7 = xr.open_mfdataset(maxfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_0p7 = xr.open_mfdataset(minfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_1p0 = xr.open_mfdataset(maxfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_1p0 = xr.open_mfdataset(minfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_1p5 = xr.open_mfdataset(maxfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_1p5 = xr.open_mfdataset(minfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15


tmax_rol_0p2 = temp_max_0p2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
tmin_rol_0p2 = temp_min_0p2.rolling(min_periods=3, center=True, time=3).mean().mean('members')

tmax_rol_0p4 = temp_max_0p4.rolling(min_periods=3, center=True, time=3).mean().mean('members')
tmin_rol_0p4 = temp_min_0p4.rolling(min_periods=3, center=True, time=3).mean().mean('members')

tmax_rol_0p7 = temp_max_0p7.rolling(min_periods=3, center=True, time=3).mean().mean('members')
tmin_rol_0p7 = temp_min_0p7.rolling(min_periods=3, center=True, time=3).mean().mean('members')

tmax_rol_1p0 = temp_max_1p0.rolling(min_periods=3, center=True, time=3).mean().mean('members')
tmin_rol_1p0 = temp_min_1p0.rolling(min_periods=3, center=True, time=3).mean().mean('members')

tmax_rol_1p5 = temp_max_1p5.rolling(min_periods=3, center=True, time=3).mean().mean('members')
tmin_rol_1p5 = temp_min_1p5.rolling(min_periods=3, center=True, time=3).mean().mean('members')

## 0p2
weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"

annual_tmax_0p2 = tmax_rol_0p2[9::12].groupby('time.year').mean('time')
region_tmax_0p2=annual_tmax_0p2.latitude>70.
weights.name = "weights"
high_lats_tmax_0p2=annual_tmax_0p2[:,region_tmax_0p2,:].weighted(weights).mean(("longitude", "latitude"))


annual_tmin_0p2 = tmin_rol_0p2[9::12].groupby('time.year').mean('time')
region_tmin_0p2=annual_tmin_0p2.latitude>70.
weights.name = "weights"
high_lats_tmin_0p2=annual_tmin_0p2[:,region_tmin_0p2,:].weighted(weights).mean(("longitude", "latitude"))

dtr_0p2 = high_lats_tmax_0p2 - high_lats_tmin_0p2

## 0p4

weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"

annual_tmax_0p4 = tmax_rol_0p4[9::12].groupby('time.year').mean('time')
region_tmax_0p4=annual_tmax_0p4.latitude>70.
weights.name = "weights"
high_lats_tmax_0p4=annual_tmax_0p4[:,region_tmax_0p4,:].weighted(weights).mean(("longitude", "latitude"))


annual_tmin_0p4 = tmin_rol_0p4[9::12].groupby('time.year').mean('time')
region_tmin_0p4=annual_tmin_0p4.latitude>70.
weights.name = "weights"
high_lats_tmin_0p4=annual_tmin_0p4[:,region_tmin_0p4,:].weighted(weights).mean(("longitude", "latitude"))


dtr_0p4 = high_lats_tmax_0p4 - high_lats_tmin_0p4

## 0p7

weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"

annual_tmax_0p7 = tmax_rol_0p7[9::12].groupby('time.year').mean('time')
region_tmax_0p7=annual_tmax_0p7.latitude>70.
weights.name = "weights"
high_lats_tmax_0p7=annual_tmax_0p7[:,region_tmax_0p7,:].weighted(weights).mean(("longitude", "latitude"))


annual_tmin_0p7 = tmin_rol_0p7[9::12].groupby('time.year').mean('time')
region_tmin_0p7=annual_tmin_0p7.latitude>70.
weights.name = "weights"
high_lats_tmin_0p7=annual_tmin_0p7[:,region_tmin_0p7,:].weighted(weights).mean(("longitude", "latitude"))


dtr_0p7 = high_lats_tmax_0p7 - high_lats_tmin_0p7

## 1p0

weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"

annual_tmax_1p0 = tmax_rol_1p0[9::12].groupby('time.year').mean('time')
region_tmax_1p0=annual_tmax_1p0.latitude>70.
weights.name = "weights"
high_lats_tmax_1p0=annual_tmax_1p0[:,region_tmax_1p0,:].weighted(weights).mean(("longitude", "latitude"))


annual_tmin_1p0 = tmin_rol_1p0[9::12].groupby('time.year').mean('time')
region_tmin_1p0=annual_tmin_1p0.latitude>70.
weights.name = "weights"
high_lats_tmin_1p0=annual_tmin_1p0[:,region_tmin_1p0,:].weighted(weights).mean(("longitude", "latitude"))


dtr_1p0 = high_lats_tmax_1p0 - high_lats_tmin_1p0

## 1p5

weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
 
annual_tmax_1p5 = tmax_rol_1p5[9::12].groupby('time.year').mean('time')
region_tmax_1p5=annual_tmax_1p5.latitude>70.
weights.name = "weights"
high_lats_tmax_1p5=annual_tmax_1p5[:,region_tmax_1p5,:].weighted(weights).mean(("longitude", "latitude"))


annual_tmin_1p5 = tmin_rol_1p5[9::12].groupby('time.year').mean('time')
region_tmin_1p5=annual_tmin_1p5.latitude>70.
weights.name = "weights"
high_lats_tmin_1p5=annual_tmin_1p5[:,region_tmin_1p5,:].weighted(weights).mean(("longitude", "latitude"))


dtr_1p5 = high_lats_tmax_1p5 - high_lats_tmin_1p5



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

ax.plot(dtr_0p2.year[(dtr_0p2.year > 1900) & (dtr_0p2.year < 2015)], dtr_0p2[(dtr_0p2.year > 1900) & (dtr_0p2.year < 2015)], linestyle='solid', color='red', label='0.2x forcing', linewidth=0.8)

ax.plot(dtr_0p4.year[(dtr_0p4.year > 1900) & (dtr_0p4.year < 2015)], dtr_0p4[(dtr_0p4.year > 1900) & (dtr_0p4.year < 2015)], linestyle='solid', color='orange', label='0.4x forcing', linewidth=0.8)

ax.plot(dtr_0p7.year[(dtr_0p7.year > 1900) & (dtr_0p7.year < 2015)], dtr_0p7[(dtr_0p7.year > 1900) & (dtr_0p7.year < 2015)], linestyle='solid', color='grey', label='0.7x forcing', linewidth=0.8)

ax.plot(dtr_1p0.year[(dtr_1p0.year > 1900) & (dtr_1p0.year < 2015)], dtr_1p0[(dtr_1p0.year > 1900) & (dtr_1p0.year < 2015)], linestyle='solid', color='dodgerblue', label='1.0x forcing', linewidth=0.8)

ax.plot(dtr_1p5.year[(dtr_1p5.year > 1900) & (dtr_1p5.year < 2015)], dtr_1p5[(dtr_1p5.year > 1900) & (dtr_1p5.year < 2015)], linestyle='solid', color='purple', label='1.5x forcing', linewidth=0.8)




###


#ax.plot(dtr_obs.year[(dtr_obs.year > 1900) & (dtr_obs.year < 2015)], dtr_obs[(dtr_obs.year > 1900) & (dtr_obs.year < 2015)], linestyle='solid', color='black', label='CRU TS v4')
# ax.tick_params('y', colors='blue')     # 'y' because we want to change the y axis
# ax2.tick_params('y', colors='red')

#ax.set_ylim(3,12)
#fig.legend(bbox_to_anchor=[0.15, 0.30], loc='upper left')
fig.legend(bbox_to_anchor=[0.13, 0.43], loc='upper left')

ax.set(title='Mean SON Diurnal Temperature Range (DTR)\n variation for high latitudes (>70$^\circ$N) (1901-2014)',
     xlabel='Years', ylabel='$^\circ$C') 

plt.savefig('/Users/ankit/Documents/UoR/Dissertation/model_son_dtr_high_lats.png', format='png', dpi=1200)

plt.show()


