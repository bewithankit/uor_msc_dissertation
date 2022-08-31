#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:20:19 2022

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

#members_tmax = xr.open_mfdataset('/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/*.nc', concat_dim='time', combine='nested')
#members_tmin = xr.open_mfdataset('/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/*.nc', concat_dim='time', combine='nested')


# members_tempmax = xr.open_mfdataset(['/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc', 
#                                      '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_r1i1p1f2_gn_18500101-20141230.nc',
#                                      '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_r1i1p1f2_gn_18500101-20141230.nc',
#                                      '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_r1i1p1f2_gn_18500101-20141230.nc',
#                                      '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_r1i1p1f2_gn_18500101-20141230.nc'], concat_dim='time', combine='nested')


#temp_min = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/member_r1_min.nc')
#temp_max = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/member_r1_max.nc')
lat = xr.open_dataset("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_r5i1p1f2_gn_18500101-20141230.nc")['latitude']
lon = xr.open_dataset("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc")['longitude']


# members = [xr.open_mfdataset(single_member, combine="nested", concat_dim="time") for single_member in file_list]
# xr.concat(members, dim="M")

# members = xr.open_mfdataset(['/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc', 
#                             '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_r1i1p1f2_gn_18500101-20141230.nc',
#                             '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_r1i1p1f2_gn_18500101-20141230.nc',
#                             '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_r1i1p1f2_gn_18500101-20141230.nc',
#                             '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_r1i1p1f2_gn_18500101-20141230.nc'], concat_dim='time', combine='nested')

# dat = xr.concat(members, dim='m')

# nfiles = [
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc', 
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_r1i1p1f2_gn_18500101-20141230.nc',
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_r1i1p1f2_gn_18500101-20141230.nc',
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_r1i1p1f2_gn_18500101-20141230.nc',
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_r1i1p1f2_gn_18500101-20141230.nc'
# ]


maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")



# temp_max = xr.open_mfdataset(nfiles,
#                         concat_dim='member',
#                         combine='nested')

preprocess=preprocess

#temp_max = xr.open_mfdataset(maxfiles,concat_dim='members',combine='nested',preprocess=preprocess)

# minfiles = [
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_r1i1p1f2_gn_18500101-20141230.nc', 
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_r1i1p1f2_gn_18500101-20141230.nc',
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_r1i1p1f2_gn_18500101-20141230.nc',
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_r1i1p1f2_gn_18500101-20141230.nc',
#     '/Users/ankit/Documents/UoR/Dissertation/SMURPHS/r1/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_r1i1p1f2_gn_18500101-20141230.nc'
# ]

# temp_min = xr.open_mfdataset(minfiles,
#                         concat_dim='member',
#                         combine='nested')

## maximum

temp_max = xr.open_mfdataset(maxfiles,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min = xr.open_mfdataset(minfiles,concat_dim='members',combine='nested',preprocess=preprocess)


weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
global_max=temp_max["tasmax"] - 273.15

#tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean('cases')
tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja=tmax_rol[6::12].groupby('time.year').mean('time')

mask0=(jja.year > 1900)
print(jja.year[mask0])

trend_ds=jja[mask0].polyfit(dim='year',deg=1,skipna=True)
trend=trend_ds['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask=((jja.year > 1950) & (jja.year < 1981))
print(jja.year[mask])
trend_ds_sub=jja[mask].polyfit(dim='year',deg=1,skipna=True)
trend_sub=trend_ds_sub['polyfit_coefficients'][0]

mask2=(jja.year > 1980)
print(jja.year[mask2])
trend_ds_sub2=jja[mask2].polyfit(dim='year',deg=1,skipna=True)
trend_sub2=trend_ds_sub2['polyfit_coefficients'][0]


## minimum

weights_min = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_min.name = "weights2"
global_min=temp_min["tasmin"] - 273.15

tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_min=tmin_rol[6::12].groupby('time.year').mean('time')

mask0_min=(jja_min.year > 1900)
print(jja_min.year[mask0_min])

trend_ds_min=jja_min.polyfit(dim='year',deg=1,skipna=True)
trend_min=trend_ds_min['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_min=((jja_min.year > 1950) & (jja_min.year < 1981))
print(jja_min.year[mask_min])
trend_ds_sub_min=jja_min[mask_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub_min=trend_ds_sub_min['polyfit_coefficients'][0]

mask2_min=(jja_min.year > 1980)
print(jja_min.year[mask2_min])
trend_ds_sub2_min=jja_min[mask2_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_min=trend_ds_sub2_min['polyfit_coefficients'][0]

## dtr

weights_dtr = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_dtr.name = "weights3"
global_dtr = global_max - global_min

dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_dtr=dtr_rol[6::12].groupby('time.year').mean('time')

trend_ds_dtr=jja_dtr.polyfit(dim='year',deg=1,skipna=True)
trend_dtr=trend_ds_dtr['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_dtr=((jja_dtr.year > 1950) & (jja_dtr.year < 1981))
print(jja_dtr.year[mask_dtr])
trend_ds_sub_dtr=jja_dtr[mask_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub_dtr=trend_ds_sub_dtr['polyfit_coefficients'][0]

mask2_dtr=(jja_dtr.year > 1980)
print(jja_dtr.year[mask2_dtr])
trend_ds_sub2_dtr=jja_dtr[mask2_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_dtr=trend_ds_sub2_dtr['polyfit_coefficients'][0]

## plot

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(20, 13), constrained_layout=True, subplot_kw={'projection': crs})
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf2 = axlist[0].contourf(lon,lat,trend*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('\nJJA Maximum Temperature Trend\n1.0x ensemble member mean (1901-2014)',fontsize=16)
cf2 = axlist[1].contourf(lon,lat,trend_sub*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('JJA Maximum Temperature Trend\n1.0x ensemble member mean (1951-1980)',fontsize=16)
cf3 = axlist[2].contourf(lon,lat,trend_sub2*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('JJA Maximum Temperature Trend\n1.0x ensemble member mean (1981-2014)',fontsize=16)
cbar = fig.colorbar(cf3,ax=axlist[1:2],orientation='horizontal')
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[3].contourf(lon,lat,trend_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\nJJA Minimum Temperature Trend\n1.0x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[4].contourf(lon,lat,trend_sub_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('JJA Minimum Temperature Trend\n1.0x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[5].contourf(lon,lat,trend_sub2_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('JJA Minimum Temperature Trend\n1.0x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[4:5],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[6].contourf(lon,lat,trend_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\nJJA DTR Trend\n1.0x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[7].contourf(lon,lat,trend_sub_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('JJA DTR Trend\n1.0x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[8].contourf(lon,lat,trend_sub2_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('JJA DTR Trend\n1.0x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[7:8],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_JJA_trend_1p0.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

## 1.5

maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")
minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")


preprocess=preprocess

## maximum

temp_max = xr.open_mfdataset(maxfiles,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min = xr.open_mfdataset(minfiles,concat_dim='members',combine='nested',preprocess=preprocess)


weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
global_max=temp_max["tasmax"] - 273.15

#tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean('cases')
tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja=tmax_rol[6::12].groupby('time.year').mean('time')

mask0=(jja.year > 1900)
print(jja.year[mask0])

trend_ds=jja[mask0].polyfit(dim='year',deg=1,skipna=True)
trend=trend_ds['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask=((jja.year > 1950) & (jja.year < 1981))
print(jja.year[mask])
trend_ds_sub=jja[mask].polyfit(dim='year',deg=1,skipna=True)
trend_sub=trend_ds_sub['polyfit_coefficients'][0]

mask2=(jja.year > 1980)
print(jja.year[mask2])
trend_ds_sub2=jja[mask2].polyfit(dim='year',deg=1,skipna=True)
trend_sub2=trend_ds_sub2['polyfit_coefficients'][0]


## minimum

weights_min = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_min.name = "weights2"
global_min=temp_min["tasmin"] - 273.15

tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_min=tmin_rol[6::12].groupby('time.year').mean('time')

mask0_min=(jja_min.year > 1900)
print(jja_min.year[mask0_min])

trend_ds_min=jja_min.polyfit(dim='year',deg=1,skipna=True)
trend_min=trend_ds_min['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_min=((jja_min.year > 1950) & (jja_min.year < 1981))
print(jja_min.year[mask_min])
trend_ds_sub_min=jja_min[mask_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub_min=trend_ds_sub_min['polyfit_coefficients'][0]

mask2_min=(jja_min.year > 1980)
print(jja_min.year[mask2_min])
trend_ds_sub2_min=jja_min[mask2_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_min=trend_ds_sub2_min['polyfit_coefficients'][0]

## dtr

weights_dtr = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_dtr.name = "weights3"
global_dtr = global_max - global_min

dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_dtr=dtr_rol[6::12].groupby('time.year').mean('time')

trend_ds_dtr=jja_dtr.polyfit(dim='year',deg=1,skipna=True)
trend_dtr=trend_ds_dtr['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_dtr=((jja_dtr.year > 1950) & (jja_dtr.year < 1981))
print(jja_dtr.year[mask_dtr])
trend_ds_sub_dtr=jja_dtr[mask_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub_dtr=trend_ds_sub_dtr['polyfit_coefficients'][0]

mask2_dtr=(jja_dtr.year > 1980)
print(jja_dtr.year[mask2_dtr])
trend_ds_sub2_dtr=jja_dtr[mask2_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_dtr=trend_ds_sub2_dtr['polyfit_coefficients'][0]

## plot

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(20, 13), constrained_layout=True, subplot_kw={'projection': crs})
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf2 = axlist[0].contourf(lon,lat,trend*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('\nJJA Maximum Temperature Trend\n1.5x ensemble member mean (1901-2014)',fontsize=16)
cf2 = axlist[1].contourf(lon,lat,trend_sub*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('JJA Maximum Temperature Trend\n1.5x ensemble member mean (1951-1980)',fontsize=16)
cf3 = axlist[2].contourf(lon,lat,trend_sub2*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('JJA Maximum Temperature Trend\n1.5x ensemble member mean (1981-2014)',fontsize=16)
cbar = fig.colorbar(cf3,ax=axlist[1:2],orientation='horizontal')
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[3].contourf(lon,lat,trend_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\nJJA Minimum Temperature Trend\n1.5x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[4].contourf(lon,lat,trend_sub_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('JJA Minimum Temperature Trend\n1.5x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[5].contourf(lon,lat,trend_sub2_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('JJA Minimum Temperature Trend\n1.5x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[4:5],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[6].contourf(lon,lat,trend_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\nJJA DTR Trend\n1.5x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[7].contourf(lon,lat,trend_sub_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('JJA DTR Trend\n1.5x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[8].contourf(lon,lat,trend_sub2_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('JJA DTR Trend\n1.5x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[7:8],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_JJA_trend_1p5.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

## o.7

maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")
minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")


preprocess=preprocess

## maximum

temp_max = xr.open_mfdataset(maxfiles,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min = xr.open_mfdataset(minfiles,concat_dim='members',combine='nested',preprocess=preprocess)


weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
global_max=temp_max["tasmax"] - 273.15

#tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean('cases')
tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja=tmax_rol[6::12].groupby('time.year').mean('time')

mask0=(jja.year > 1900)
print(jja.year[mask0])

trend_ds=jja[mask0].polyfit(dim='year',deg=1,skipna=True)
trend=trend_ds['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask=((jja.year > 1950) & (jja.year < 1981))
print(jja.year[mask])
trend_ds_sub=jja[mask].polyfit(dim='year',deg=1,skipna=True)
trend_sub=trend_ds_sub['polyfit_coefficients'][0]

mask2=(jja.year > 1980)
print(jja.year[mask2])
trend_ds_sub2=jja[mask2].polyfit(dim='year',deg=1,skipna=True)
trend_sub2=trend_ds_sub2['polyfit_coefficients'][0]


## minimum

weights_min = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_min.name = "weights2"
global_min=temp_min["tasmin"] - 273.15

tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_min=tmin_rol[6::12].groupby('time.year').mean('time')

mask0_min=(jja_min.year > 1900)
print(jja_min.year[mask0_min])

trend_ds_min=jja_min.polyfit(dim='year',deg=1,skipna=True)
trend_min=trend_ds_min['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_min=((jja_min.year > 1950) & (jja_min.year < 1981))
print(jja_min.year[mask_min])
trend_ds_sub_min=jja_min[mask_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub_min=trend_ds_sub_min['polyfit_coefficients'][0]

mask2_min=(jja_min.year > 1980)
print(jja_min.year[mask2_min])
trend_ds_sub2_min=jja_min[mask2_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_min=trend_ds_sub2_min['polyfit_coefficients'][0]

## dtr

weights_dtr = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_dtr.name = "weights3"
global_dtr = global_max - global_min

dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_dtr=dtr_rol[6::12].groupby('time.year').mean('time')

trend_ds_dtr=jja_dtr.polyfit(dim='year',deg=1,skipna=True)
trend_dtr=trend_ds_dtr['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_dtr=((jja_dtr.year > 1950) & (jja_dtr.year < 1981))
print(jja_dtr.year[mask_dtr])
trend_ds_sub_dtr=jja_dtr[mask_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub_dtr=trend_ds_sub_dtr['polyfit_coefficients'][0]

mask2_dtr=(jja_dtr.year > 1980)
print(jja_dtr.year[mask2_dtr])
trend_ds_sub2_dtr=jja_dtr[mask2_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_dtr=trend_ds_sub2_dtr['polyfit_coefficients'][0]

## plot

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(20, 13), constrained_layout=True, subplot_kw={'projection': crs})
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf2 = axlist[0].contourf(lon,lat,trend*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('\nJJA Maximum Temperature Trend\n0.7x ensemble member mean (1901-2014)',fontsize=16)
cf2 = axlist[1].contourf(lon,lat,trend_sub*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('JJA Maximum Temperature Trend\n0.7x ensemble member mean (1951-1980)',fontsize=16)
cf3 = axlist[2].contourf(lon,lat,trend_sub2*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('JJA Maximum Temperature Trend\n0.7x ensemble member mean (1981-2014)',fontsize=16)
cbar = fig.colorbar(cf3,ax=axlist[1:2],orientation='horizontal')
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[3].contourf(lon,lat,trend_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\nJJA Minimum Temperature Trend\n0.7x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[4].contourf(lon,lat,trend_sub_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('JJA Minimum Temperature Trend\n0.7x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[5].contourf(lon,lat,trend_sub2_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('JJA Minimum Temperature Trend\n0.7x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[4:5],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[6].contourf(lon,lat,trend_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\nJJA DTR Trend\n0.7x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[7].contourf(lon,lat,trend_sub_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('JJA DTR Trend\n0.7x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[8].contourf(lon,lat,trend_sub2_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('JJA DTR Trend\n0.7x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[7:8],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_JJA_trend_0p7.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

##### 0.4

maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")
minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")


preprocess=preprocess

## maximum

temp_max = xr.open_mfdataset(maxfiles,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min = xr.open_mfdataset(minfiles,concat_dim='members',combine='nested',preprocess=preprocess)


weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
global_max=temp_max["tasmax"] - 273.15

#tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean('cases')
tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja=tmax_rol[6::12].groupby('time.year').mean('time')

mask0=(jja.year > 1900)
print(jja.year[mask0])

trend_ds=jja[mask0].polyfit(dim='year',deg=1,skipna=True)
trend=trend_ds['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask=((jja.year > 1950) & (jja.year < 1981))
print(jja.year[mask])
trend_ds_sub=jja[mask].polyfit(dim='year',deg=1,skipna=True)
trend_sub=trend_ds_sub['polyfit_coefficients'][0]

mask2=(jja.year > 1980)
print(jja.year[mask2])
trend_ds_sub2=jja[mask2].polyfit(dim='year',deg=1,skipna=True)
trend_sub2=trend_ds_sub2['polyfit_coefficients'][0]


## minimum

weights_min = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_min.name = "weights2"
global_min=temp_min["tasmin"] - 273.15

tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_min=tmin_rol[6::12].groupby('time.year').mean('time')

mask0_min=(jja_min.year > 1900)
print(jja_min.year[mask0_min])

trend_ds_min=jja_min.polyfit(dim='year',deg=1,skipna=True)
trend_min=trend_ds_min['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_min=((jja_min.year > 1950) & (jja_min.year < 1981))
print(jja_min.year[mask_min])
trend_ds_sub_min=jja_min[mask_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub_min=trend_ds_sub_min['polyfit_coefficients'][0]

mask2_min=(jja_min.year > 1980)
print(jja_min.year[mask2_min])
trend_ds_sub2_min=jja_min[mask2_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_min=trend_ds_sub2_min['polyfit_coefficients'][0]

## dtr

weights_dtr = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_dtr.name = "weights3"
global_dtr = global_max - global_min

dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_dtr=dtr_rol[6::12].groupby('time.year').mean('time')

trend_ds_dtr=jja_dtr.polyfit(dim='year',deg=1,skipna=True)
trend_dtr=trend_ds_dtr['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_dtr=((jja_dtr.year > 1950) & (jja_dtr.year < 1981))
print(jja_dtr.year[mask_dtr])
trend_ds_sub_dtr=jja_dtr[mask_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub_dtr=trend_ds_sub_dtr['polyfit_coefficients'][0]

mask2_dtr=(jja_dtr.year > 1980)
print(jja_dtr.year[mask2_dtr])
trend_ds_sub2_dtr=jja_dtr[mask2_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_dtr=trend_ds_sub2_dtr['polyfit_coefficients'][0]

## plot

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(20, 13), constrained_layout=True, subplot_kw={'projection': crs})
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf2 = axlist[0].contourf(lon,lat,trend*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('\nJJA Maximum Temperature Trend\n0.4x ensemble member mean (1901-2014)',fontsize=16)
cf2 = axlist[1].contourf(lon,lat,trend_sub*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('JJA Maximum Temperature Trend\n0.4x ensemble member mean (1951-1980)',fontsize=16)
cf3 = axlist[2].contourf(lon,lat,trend_sub2*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('JJA Maximum Temperature Trend\n0.4x ensemble member mean (1981-2014)',fontsize=16)
cbar = fig.colorbar(cf3,ax=axlist[1:2],orientation='horizontal')
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[3].contourf(lon,lat,trend_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\nJJA Minimum Temperature Trend\n0.4x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[4].contourf(lon,lat,trend_sub_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('JJA Minimum Temperature Trend\n0.4x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[5].contourf(lon,lat,trend_sub2_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('JJA Minimum Temperature Trend\n0.4x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[4:5],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[6].contourf(lon,lat,trend_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\nJJA DTR Trend\n0.4x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[7].contourf(lon,lat,trend_sub_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('JJA DTR Trend\n0.4x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[8].contourf(lon,lat,trend_sub2_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('JJA DTR Trend\n0.4x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[7:8],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_JJA_trend_0p4.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()


####### 0.2

maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")
minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")


preprocess=preprocess

## maximum

temp_max = xr.open_mfdataset(maxfiles,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min = xr.open_mfdataset(minfiles,concat_dim='members',combine='nested',preprocess=preprocess)


weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
global_max=temp_max["tasmax"] - 273.15

#tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean('cases')
tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja=tmax_rol[6::12].groupby('time.year').mean('time')

mask0=(jja.year > 1900)
print(jja.year[mask0])

trend_ds=jja[mask0].polyfit(dim='year',deg=1,skipna=True)
trend=trend_ds['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask=((jja.year > 1950) & (jja.year < 1981))
print(jja.year[mask])
trend_ds_sub=jja[mask].polyfit(dim='year',deg=1,skipna=True)
trend_sub=trend_ds_sub['polyfit_coefficients'][0]

mask2=(jja.year > 1980)
print(jja.year[mask2])
trend_ds_sub2=jja[mask2].polyfit(dim='year',deg=1,skipna=True)
trend_sub2=trend_ds_sub2['polyfit_coefficients'][0]


## minimum

weights_min = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_min.name = "weights2"
global_min=temp_min["tasmin"] - 273.15

tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_min=tmin_rol[6::12].groupby('time.year').mean('time')

mask0_min=(jja_min.year > 1900)
print(jja_min.year[mask0_min])

trend_ds_min=jja_min.polyfit(dim='year',deg=1,skipna=True)
trend_min=trend_ds_min['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_min=((jja_min.year > 1950) & (jja_min.year < 1981))
print(jja_min.year[mask_min])
trend_ds_sub_min=jja_min[mask_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub_min=trend_ds_sub_min['polyfit_coefficients'][0]

mask2_min=(jja_min.year > 1980)
print(jja_min.year[mask2_min])
trend_ds_sub2_min=jja_min[mask2_min].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_min=trend_ds_sub2_min['polyfit_coefficients'][0]

## dtr

weights_dtr = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_dtr.name = "weights3"
global_dtr = global_max - global_min

dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
jja_dtr=dtr_rol[6::12].groupby('time.year').mean('time')

trend_ds_dtr=jja_dtr.polyfit(dim='year',deg=1,skipna=True)
trend_dtr=trend_ds_dtr['polyfit_coefficients'][0]

# time mask, calculate trend over subset of data 1951-1980

mask_dtr=((jja_dtr.year > 1950) & (jja_dtr.year < 1981))
print(jja_dtr.year[mask_dtr])
trend_ds_sub_dtr=jja_dtr[mask_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub_dtr=trend_ds_sub_dtr['polyfit_coefficients'][0]

mask2_dtr=(jja_dtr.year > 1980)
print(jja_dtr.year[mask2_dtr])
trend_ds_sub2_dtr=jja_dtr[mask2_dtr].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_dtr=trend_ds_sub2_dtr['polyfit_coefficients'][0]

## plot

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(20, 13), constrained_layout=True, subplot_kw={'projection': crs})
axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf2 = axlist[0].contourf(lon,lat,trend*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('\nJJA Maximum Temperature Trend\n0.2x ensemble member mean (1901-2014)',fontsize=16)
cf2 = axlist[1].contourf(lon,lat,trend_sub*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('JJA Maximum Temperature Trend\n0.2x ensemble member mean (1951-1980)',fontsize=16)
cf3 = axlist[2].contourf(lon,lat,trend_sub2*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('JJA Maximum Temperature Trend\n0.2x ensemble member mean (1981-2014)',fontsize=16)
cbar = fig.colorbar(cf3,ax=axlist[1:2],orientation='horizontal')
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[3].contourf(lon,lat,trend_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\nJJA Minimum Temperature Trend\n0.2x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[4].contourf(lon,lat,trend_sub_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('JJA Minimum Temperature Trend\n0.2x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[5].contourf(lon,lat,trend_sub2_min*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('JJA Minimum Temperature Trend\n0.2x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[4:5],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

cf4 = axlist[6].contourf(lon,lat,trend_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\nJJA DTR Trend\n0.2x ensemble member mean (1901-2014)',fontsize=16)
cf5 = axlist[7].contourf(lon,lat,trend_sub_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('JJA DTR Trend\n0.2x ensemble member mean (1951-1980)',fontsize=16)
cf6 = axlist[8].contourf(lon,lat,trend_sub2_dtr*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('JJA DTR Trend\n0.2x ensemble member mean (1981-2014)',fontsize=16)
cbar2 = fig.colorbar(cf4,ax=axlist[7:8],orientation='horizontal')
cbar2.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_JJA_trend_0p2.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()