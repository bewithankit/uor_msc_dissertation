#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 00:22:03 2022

@author: ty824157
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point

def preprocess(ds):
    print(ds.encoding['source'])
    return(ds)

lat = xr.open_dataset("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_r5i1p1f2_gn_18500101-20141230.nc")['latitude']
lon = xr.open_dataset("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc")['longitude']

temp_min_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmn.dat.lowres.nc')['tmn']
temp_max_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['tmx']

lat_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['latitude']
lon_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['longitude']

# maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
# minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

# maxfiles_0p2 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")
# minfiles_0p2 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")

# maxfiles_0p4 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")
# minfiles_0p4 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")

# maxfiles_0p7 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")
# minfiles_0p7 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")

maxfiles_1p0 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
minfiles_1p0 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

# maxfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")
# minfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")

preprocess=preprocess

# temp_max_0p2 = xr.open_mfdataset(maxfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)
# temp_min_0p2 = xr.open_mfdataset(minfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)

# temp_max_0p4 = xr.open_mfdataset(maxfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)
# temp_min_0p4 = xr.open_mfdataset(minfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)

# temp_max_0p7 = xr.open_mfdataset(maxfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)
# temp_min_0p7 = xr.open_mfdataset(minfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)

temp_max_1p0 = xr.open_mfdataset(maxfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)['tasmax']
temp_min_1p0 = xr.open_mfdataset(minfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)['tasmin']

# temp_max_1p5 = xr.open_mfdataset(maxfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)
# temp_min_1p5 = xr.open_mfdataset(minfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)

global_average_tmax_1p0 = temp_max_1p0.groupby('time.year').mean('time').mean('members')
global_average_tmax_1p0_1901_2014 = global_average_tmax_1p0.sel(year=slice('1901', '2014'))
annual_tmax_gm_1p0_1901_2014=global_average_tmax_1p0_1901_2014 - 273.15
annual_tmax_gm_1p0_1901_2014_new,lon2 = add_cyclic_point(annual_tmax_gm_1p0_1901_2014,lon,axis=2)

global_average_tmin_1p0 = temp_min_1p0.groupby('time.year').mean('time').mean('members')
global_average_tmin_1p0_1901_2014 = global_average_tmin_1p0.sel(year=slice('1901', '2014'))
annual_tmin_gm_1p0_1901_2014=global_average_tmin_1p0_1901_2014 - 273.15
annual_tmin_gm_1p0_1901_2014_new,lon2 = add_cyclic_point(annual_tmin_gm_1p0_1901_2014,lon,axis=2)

dtr_1p0_1901_2014 = annual_tmax_gm_1p0_1901_2014_new - annual_tmin_gm_1p0_1901_2014_new

#
global_average_tmax_1p0_1951_1980 = global_average_tmax_1p0.sel(year=slice('1951', '1980'))
annual_tmax_gm_1p0_1951_1980=global_average_tmax_1p0_1951_1980 - 273.15
annual_tmax_gm_1p0_1951_1980_new,lon2 = add_cyclic_point(annual_tmax_gm_1p0_1951_1980,lon,axis=2)

global_average_tmin_1p0_1951_1980 = global_average_tmin_1p0.sel(year=slice('1951', '1980'))
annual_tmin_gm_1p0_1951_1980=global_average_tmin_1p0_1951_1980 - 273.15
annual_tmin_gm_1p0_1951_1980_new,lon2 = add_cyclic_point(annual_tmin_gm_1p0_1951_1980,lon,axis=2)

dtr_1p0_1951_1980 = annual_tmax_gm_1p0_1951_1980_new - annual_tmin_gm_1p0_1951_1980_new

#
global_average_tmax_1p0_1981_2014 = global_average_tmax_1p0.sel(year=slice('1981', '2014'))
annual_tmax_gm_1p0_1981_2014=global_average_tmax_1p0_1981_2014 - 273.15
annual_tmax_gm_1p0_1981_2014_new,lon2 = add_cyclic_point(annual_tmax_gm_1p0_1981_2014,lon,axis=2)

global_average_tmin_1p0_1981_2014 = global_average_tmin_1p0.sel(year=slice('1981', '2014'))
annual_tmin_gm_1p0_1981_2014=global_average_tmin_1p0_1981_2014 - 273.15
annual_tmin_gm_1p0_1981_2014_new,lon2 = add_cyclic_point(annual_tmin_gm_1p0_1981_2014,lon,axis=2)

dtr_1p0_1981_2014 = annual_tmax_gm_1p0_1981_2014_new - annual_tmin_gm_1p0_1981_2014_new

# observation

global_average_tmax_obs=temp_max_obs.groupby('time.year').mean('time')
global_average_tmax_obs_1901_2014 = global_average_tmax_obs.sel(year=slice('1901', '2014'))
global_average_tmax_obs_1901_2014_new,lon_obs2 = add_cyclic_point(global_average_tmax_obs_1901_2014,lon_obs,axis=2)

global_average_tmin_obs=temp_min_obs.groupby('time.year').mean('time')
global_average_tmin_obs_1901_2014 = global_average_tmin_obs.sel(year=slice('1901', '2014'))
global_average_tmin_obs_1901_2014_new,lon_obs2 = add_cyclic_point(global_average_tmin_obs_1901_2014,lon_obs,axis=2)

dtr_obs_1901_2014 = global_average_tmax_obs_1901_2014_new - global_average_tmin_obs_1901_2014_new

#
global_average_tmax_obs_1951_1980 = global_average_tmax_obs.sel(year=slice('1951', '1980'))
global_average_tmax_obs_1951_1980_new,lon_obs2 = add_cyclic_point(global_average_tmax_obs_1951_1980,lon_obs,axis=2)


global_average_tmin_obs_1951_1980 = global_average_tmin_obs.sel(year=slice('1951', '1980'))
global_average_tmin_obs_1951_1980_new,lon_obs2 = add_cyclic_point(global_average_tmin_obs_1951_1980,lon_obs,axis=2)

dtr_obs_1951_1980 = global_average_tmax_obs_1951_1980_new - global_average_tmin_obs_1951_1980_new
###

global_average_tmax_obs_1981_2014 = global_average_tmax_obs.sel(year=slice('1981', '2014'))
global_average_tmax_obs_1981_2014_new,lon_obs2 = add_cyclic_point(global_average_tmax_obs_1981_2014,lon_obs,axis=2)

global_average_tmin_obs_1981_2014 = global_average_tmin_obs.sel(year=slice('1981', '2014'))
global_average_tmin_obs_1981_2014_new,lon_obs2 = add_cyclic_point(global_average_tmin_obs_1981_2014,lon_obs,axis=2)

dtr_obs_1981_2014 = global_average_tmax_obs_1981_2014_new - global_average_tmin_obs_1981_2014_new

##
f=plt.figure()
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(17, 13), constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)
    
cf1 = axlist[0].contourf(lon2,lat,dtr_1p0_1901_2014[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('1.0x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(lon_obs2,lat_obs,global_average_tmax_obs_1901_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\nCRU TS v4\n',fontsize=16)
cf3 = axlist[2].contourf(lon2,lat,(annual_tmax_gm_1p0_1901_2014_new[0,:,:] - global_average_tmax_obs_1901_2014_new[0,:,:]),np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('Difference between 1.0x ensemble\nmember mean and CRU TS v4\n',fontsize=16)

cf4 = axlist[3].contourf(lon2,lat,annual_tmax_gm_1p0_1951_1980_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(lon_obs2,lat_obs,global_average_tmax_obs_1951_1980_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(lon2,lat,(annual_tmax_gm_1p0_1951_1980_new[0,:,:] - global_average_tmax_obs_1951_1980_new[0,:,:]),np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)

cf7 = axlist[6].contourf(lon2,lat,annual_tmax_gm_1p0_1981_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(lon_obs2,lat_obs,global_average_tmax_obs_1981_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(lon2,lat,(annual_tmax_gm_1p0_1981_2014_new[0,:,:] - global_average_tmax_obs_1981_2014_new[0,:,:]),np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('\n',fontsize=16)

fig.suptitle('SMURPHS Model and CRU TS v4 Observation Bias for Maximum Temperature\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/model_bias_max.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### min

f=plt.figure()
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(17, 13), constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)
    
cf1 = axlist[0].contourf(lon2,lat,annual_tmin_gm_1p0_1901_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('1.0x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(lon_obs2,lat_obs,global_average_tmin_obs_1901_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\nCRU TS v4\n',fontsize=16)
cf3 = axlist[2].contourf(lon2,lat,(annual_tmin_gm_1p0_1901_2014_new[0,:,:] - global_average_tmin_obs_1901_2014_new[0,:,:]),np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('Difference between 1.0x ensemble\nmember mean and CRU TS v4\n',fontsize=16)

cf4 = axlist[3].contourf(lon2,lat,annual_tmin_gm_1p0_1951_1980_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(lon_obs2,lat_obs,global_average_tmin_obs_1951_1980_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(lon2,lat,(annual_tmin_gm_1p0_1951_1980_new[0,:,:] - global_average_tmin_obs_1951_1980_new[0,:,:]),np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)

cf7 = axlist[6].contourf(lon2,lat,annual_tmin_gm_1p0_1981_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(lon_obs2,lat_obs,global_average_tmin_obs_1981_2014_new[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(lon2,lat,(annual_tmin_gm_1p0_1981_2014_new[0,:,:] - global_average_tmin_obs_1981_2014_new[0,:,:]),np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('\n',fontsize=16)

fig.suptitle('SMURPHS Model and CRU TS v4 Observation Bias for Minimum Temperature\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/model_bias_min.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()


#### dtr

f=plt.figure()
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(17, 13), constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)
    
cf1 = axlist[0].contourf(lon2,lat,dtr_1p0_1901_2014[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('1.0x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(lon_obs2,lat_obs,dtr_obs_1901_2014[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\nCRU TS v4\n',fontsize=16)
cf3 = axlist[2].contourf(lon2,lat,(dtr_1p0_1901_2014[0,:,:] - dtr_obs_1901_2014[0,:,:]),np.linspace(-30,30,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('Difference between 1.0x ensemble\nmember mean and CRU TS v4\n',fontsize=16)

cf4 = axlist[3].contourf(lon2,lat,dtr_1p0_1951_1980[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(lon_obs2,lat_obs,dtr_obs_1951_1980[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(lon2,lat,(dtr_1p0_1951_1980[0,:,:] - dtr_obs_1951_1980[0,:,:]),np.linspace(-30,30,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)

cf7 = axlist[6].contourf(lon2,lat,dtr_1p0_1981_2014[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(lon_obs2,lat_obs,dtr_obs_1981_2014[0,:,:],np.linspace(-40,40,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(lon2,lat,(dtr_1p0_1981_2014[0,:,:] - dtr_obs_1981_2014[0,:,:]),np.linspace(-30,30,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('\n',fontsize=16)

fig.suptitle('SMURPHS Model and CRU TS v4 Observation Bias for DTR\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/model_bias_dtr_n.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()