#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:30:37 2022

@author: ankit
"""

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




preprocess=preprocess


### observation

temp_min = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['tmn']
temp_max = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmx.dat.nc')['tmx']
lat_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['lat']
lon_obs = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.nc')['lon']


weights_obs = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_obs.name = "weights_obs"


temp_min_1901_1950 = temp_min.sel(time=slice('1901-01-16', '1950-12-16'))
temp_min_1901_2014 = temp_min.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_min = np.mean(temp_min_1901_1950.groupby("time.year").mean("time").weighted(weights_obs).mean(("lon", "lat")))
anomalies_min = temp_min_1901_2014.groupby("time.year").mean("time").weighted(weights_obs).mean(("lon", "lat")) - climatology_min

temp_max_1901_1950 = temp_max.sel(time=slice('1901-01-16', '1950-12-16'))
temp_max_1901_2014 = temp_max.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_max = np.mean(temp_max_1901_1950.groupby("time.year").mean("time").weighted(weights_obs).mean(("lon", "lat")))
anomalies_max = temp_max_1901_2014.groupby("time.year").mean("time").weighted(weights_obs).mean(("lon", "lat")) - climatology_max





lat = xr.open_dataset("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc")['latitude']
lon = xr.open_dataset("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc")['longitude']

weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"


ds1 = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/sftlf_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc')
# lat = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/sftlf_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc')['lat']
# lon = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/sftlf_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc')['lon']


landfrac=ds1['sftlf']/100.


maxfiles_0p2 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")
minfiles_0p2 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p2/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")

maxfiles_1p5 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p5/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")
minfiles_1p5 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")

maxfiles_0p4 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p4/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")
minfiles_0p4 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p4/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")

maxfiles_0p7 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p7/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")
minfiles_0p7 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/0p7/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")

maxfiles_1p0 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
minfiles_1p0 = glob.glob("/Users/ankit/Documents/UoR/Dissertation/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

temp_max_0p4 = xr.open_mfdataset(maxfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_0p4 = xr.open_mfdataset(minfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_0p7 = xr.open_mfdataset(maxfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_0p7 = xr.open_mfdataset(minfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_1p0 = xr.open_mfdataset(maxfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_1p0 = xr.open_mfdataset(minfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_0p2 = xr.open_mfdataset(maxfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_0p2 = xr.open_mfdataset(minfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15

temp_max_1p5 = xr.open_mfdataset(maxfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)["tasmax"] - 273.15
temp_min_1p5 = xr.open_mfdataset(minfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)["tasmin"] - 273.15


### all


temp_min_1901_1950_0p4 = temp_min_0p4.sel(time=slice('1901-01-16', '1950-12-16'))
temp_min_1901_2014_0p4 = temp_min_0p4.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_min_0p4 = np.mean(temp_min_1901_1950_0p4.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_min_0p4 = temp_min_1901_2014_0p4.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_min_0p4
tas_land_min_0p4=(anomalies_min_0p4*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))



temp_max_1901_1950_0p4 = temp_max_0p4.sel(time=slice('1901-01-16', '1950-12-16'))
temp_max_1901_2014_0p4 = temp_max_0p4.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_max_0p4 = np.mean(temp_max_1901_1950_0p4.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_max_0p4 = temp_max_1901_2014_0p4.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_max_0p4
tas_land_max_0p4=(anomalies_max_0p4*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))



# 0p7

temp_min_1901_1950_0p7 = temp_min_0p7.sel(time=slice('1901-01-16', '1950-12-16'))
temp_min_1901_2014_0p7 = temp_min_0p7.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_min_0p7 = np.mean(temp_min_1901_1950_0p7.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_min_0p7 = temp_min_1901_2014_0p7.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_min_0p7
tas_land_min_0p7=(anomalies_min_0p7*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))



temp_max_1901_1950_0p7 = temp_max_0p7.sel(time=slice('1901-01-16', '1950-12-16'))
temp_max_1901_2014_0p7 = temp_max_0p7.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_max_0p7 = np.mean(temp_max_1901_1950_0p7.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_max_0p7 = temp_max_1901_2014_0p7.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_max_0p7
tas_land_max_0p7=(anomalies_max_0p7*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))


# 1p0

temp_min_1901_1950_1p0 = temp_min_1p0.sel(time=slice('1901-01-16', '1950-12-16'))
temp_min_1901_2014_1p0 = temp_min_1p0.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_min_1p0 = np.mean(temp_min_1901_1950_1p0.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_min_1p0 = temp_min_1901_2014_1p0.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_min_1p0
tas_land_min_1p0=(anomalies_min_1p0*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))


temp_max_1901_1950_1p0 = temp_max_1p0.sel(time=slice('1901-01-16', '1950-12-16'))
temp_max_1901_2014_1p0 = temp_max_1p0.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_max_1p0 = np.mean(temp_max_1901_1950_1p0.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_max_1p0 = temp_max_1901_2014_1p0.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_max_1p0
tas_land_max_1p0=(anomalies_max_1p0*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))


# 0p2

temp_min_1901_1950_0p2 = temp_min_0p2.sel(time=slice('1901-01-16', '1950-12-16'))
temp_min_1901_2014_0p2 = temp_min_0p2.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_min_0p2 = np.mean(temp_min_1901_1950_0p2.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_min_0p2 = temp_min_1901_2014_0p2.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_min_0p2
tas_land_min_0p2=(anomalies_min_0p2*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))



temp_max_1901_1950_0p2 = temp_max_0p2.sel(time=slice('1901-01-16', '1950-12-16'))
temp_max_1901_2014_0p2 = temp_max_0p2.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_max_0p2 = np.mean(temp_max_1901_1950_0p2.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_max_0p2 = temp_max_1901_2014_0p2.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_max_0p2
tas_land_max_0p2=(anomalies_max_0p2*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))


# 1p5

temp_min_1901_1950_1p5 = temp_min_1p5.sel(time=slice('1901-01-16', '1950-12-16'))
temp_min_1901_2014_1p5 = temp_min_1p5.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_min_1p5 = np.mean(temp_min_1901_1950_1p5.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_min_1p5 = temp_min_1901_2014_1p5.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_min_1p5
tas_land_min_1p5=(anomalies_min_1p5*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))



temp_max_1901_1950_1p5 = temp_max_1p5.sel(time=slice('1901-01-16', '1950-12-16'))
temp_max_1901_2014_1p5 = temp_max_1p5.sel(time=slice('1901-01-16', '2014-12-16'))
climatology_max_1p5 = np.mean(temp_max_1901_1950_1p5.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")))
anomalies_max_1p5 = temp_max_1901_2014_1p5.groupby("time.year").mean("time").mean('members').weighted(weights).mean(("longitude", "latitude")) - climatology_max_1p5
tas_land_max_1p5=(anomalies_max_1p5*landfrac).weighted(weights*landfrac).mean(("lat", "lon"))


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=[0.01, 0.99], loc='upper left')


fig = plt.figure()

fig, ax = plt.subplots()

ax.plot(tas_land_min_0p2.year, tas_land_min_0p2, linestyle='solid', color='red', label='0.2x forcing', linewidth=0.8)

ax.plot(tas_land_min_0p4.year, tas_land_min_0p4, linestyle='solid', color='orange', label='0.4x forcing', linewidth=0.8)

ax.plot(tas_land_min_0p7.year, tas_land_min_0p7, linestyle='solid', color='grey', label='0.7x forcing', linewidth=0.8)

ax.plot(tas_land_min_1p0.year, tas_land_min_1p0, linestyle='solid', color='dodgerblue', label='1.0x forcing', linewidth=0.8)

ax.plot(tas_land_min_1p5.year, tas_land_min_1p5, linestyle='solid', color='purple', label='1.5x forcing', linewidth=0.8)

ax.plot(anomalies_min.year, anomalies_min, linestyle='solid', color='black', label='CRU TS v4', linewidth=0.8)

legend_without_duplicate_labels(ax)

ax.set(title='Landmasked Global Average Minimum Temperature Anomaly\n(1901-2014)',
      xlabel='Years', ylabel='Difference from 1901-1950 average ($^\circ$C)') 

plt.savefig('/Users/ankit/Documents/UoR/Dissertation/Land_Temp_Anomaly_min_all_forcings.png', format='png', dpi=1200)

plt.show()

# fig.legend(bbox_to_anchor=[0.15, 0.85], loc='upper left')

# ax.set(title='Landmasked Global Average Minimum Temperature Anomaly (1901-2014)',
#      xlabel='Years', ylabel='Difference from 1901-1950 average ($^\circ$C)') 

# plt.savefig('/Users/ankit/Documents/UoR/Dissertation/Land_Temp_Anomaly_min_all_forcings.png', format='png', dpi=1200)

# plt.show()

### max

fig = plt.figure()

fig, ax = plt.subplots()

ax.plot(tas_land_max_0p2.year, tas_land_max_0p2, linestyle='solid', color='red', label='0.2x forcing', linewidth=0.8)

ax.plot(tas_land_max_0p4.year, tas_land_max_0p4, linestyle='solid', color='orange', label='0.4x forcing', linewidth=0.8)

ax.plot(tas_land_max_0p7.year, tas_land_max_0p7, linestyle='solid', color='grey', label='0.7x forcing', linewidth=0.8)

ax.plot(tas_land_max_1p0.year, tas_land_max_1p0, linestyle='solid', color='dodgerblue', label='1.0x forcing', linewidth=0.8)

ax.plot(tas_land_max_1p5.year, tas_land_max_1p5, linestyle='solid', color='purple', label='1.5x forcing', linewidth=0.8)

ax.plot(anomalies_max.year, anomalies_max, linestyle='solid', color='black', label='CRU TS v4', linewidth=0.8)

legend_without_duplicate_labels(ax)

ax.set(title='Landmasked Global Average Maximum Temperature Anomaly\n(1901-2014)',
     xlabel='Years', ylabel='Difference from 1901-1950 average ($^\circ$C)') 

plt.savefig('/Users/ankit/Documents/UoR/Dissertation/Land_Temp_Anomaly_max_all_forcings.png', format='png', dpi=1200)

plt.show()


#### dtr


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=[0.01, 0.48], loc='upper left')

fig = plt.figure()

fig, ax = plt.subplots()

ax.plot(tas_land_max_0p2.year, tas_land_max_0p2 - tas_land_min_0p2 , linestyle='solid', color='red', label='0.2x forcing', linewidth=0.8)

ax.plot(tas_land_max_0p4.year, tas_land_max_0p4 - tas_land_min_0p4 , linestyle='solid', color='orange', label='0.4x forcing', linewidth=0.8)

ax.plot(tas_land_max_0p7.year, tas_land_max_0p7 - tas_land_min_0p7, linestyle='solid', color='grey', label='0.7x forcing', linewidth=0.8)

ax.plot(tas_land_max_1p0.year, tas_land_max_1p0 - tas_land_min_1p0, linestyle='solid', color='dodgerblue', label='1.0x forcing', linewidth=0.8)

ax.plot(tas_land_max_1p5.year, tas_land_max_1p5 - tas_land_min_1p5, linestyle='solid', color='purple', label='1.5x forcing', linewidth=0.8)

ax.plot(anomalies_max.year, anomalies_max - anomalies_min , linestyle='solid', color='black', label='CRU TS v4', linewidth=0.8)

legend_without_duplicate_labels(ax)

ax.set(title='Landmasked Global mean Diurnal Temperature Range (DTR) Anomaly\n(1901-2014)',
     xlabel='Years', ylabel='Difference from 1901-1950 average ($^\circ$C)') 

plt.savefig('/Users/ankit/Documents/UoR/Dissertation/Land_DTR_Temp_Anomaly_all_forcings.png', format='png', dpi=1200)

plt.show()