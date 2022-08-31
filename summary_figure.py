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

maxfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

maxfiles_0p2 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")
minfiles_0p2 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")

maxfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")
minfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")

preprocess=preprocess

temp_max_0p2 = xr.open_mfdataset(maxfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_0p2 = xr.open_mfdataset(minfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)

temp_max_1p5 = xr.open_mfdataset(maxfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_1p5 = xr.open_mfdataset(minfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)

def calc_tmax_trend_jja(temp_max, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_max=temp_max["tasmax"] - 273.15
    tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja=tmax_rol[0::12].groupby('time.year').mean('time')
    mask=((jja.year > year1) & (jja.year < year2))
    trend_ds=jja[mask].polyfit(dim='year',deg=1,skipna=True)
    trend = trend_ds['polyfit_coefficients'][0]
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_max_0p2_1901_2014 = calc_tmax_trend_jja(temp_max_0p2, 1900, 2015)
trend_max_0p2_1951_1980 = calc_tmax_trend_jja(temp_max_0p2, 1950, 1981)
trend_max_0p2_1981_2014 = calc_tmax_trend_jja(temp_max_0p2, 1980, 2015)

trend_max_1p5_1901_2014 = calc_tmax_trend_jja(temp_max_1p5, 1900, 2015)
trend_max_1p5_1951_1980 = calc_tmax_trend_jja(temp_max_1p5, 1950, 1981)
trend_max_1p5_1981_2014 = calc_tmax_trend_jja(temp_max_1p5, 1980, 2015)

def calc_tmin_trend_jja(temp_min, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min=temp_min["tasmin"] - 273.15
    tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja=tmin_rol[0::12].groupby('time.year').mean('time')
    mask=((jja.year > year1) & (jja.year < year2))
    trend_ds=jja[mask].polyfit(dim='year',deg=1,skipna=True)
    trend = trend_ds['polyfit_coefficients'][0]
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_min_0p2_1901_2014 = calc_tmin_trend_jja(temp_min_0p2, 1900, 2015)
trend_min_0p2_1951_1980 = calc_tmin_trend_jja(temp_min_0p2, 1950, 1981)
trend_min_0p2_1981_2014 = calc_tmin_trend_jja(temp_min_0p2, 1980, 2015)

trend_min_1p5_1901_2014 = calc_tmin_trend_jja(temp_min_1p5, 1900, 2015)
trend_min_1p5_1951_1980 = calc_tmin_trend_jja(temp_min_1p5, 1950, 1981)
trend_min_1p5_1981_2014 = calc_tmin_trend_jja(temp_min_1p5, 1980, 2015)

def calc_dtr_trend_jja(temp_max, temp_min, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_dtr=((temp_max["tasmax"] - 273.15) - (temp_min["tasmin"] - 273.15))
    dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja=dtr_rol[0::12].groupby('time.year').mean('time')
    mask=((jja.year > year1) & (jja.year < year2))
    trend_ds=jja[mask].polyfit(dim='year',deg=1,skipna=True)
    trend = trend_ds['polyfit_coefficients'][0]
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_dtr_0p2_1901_2014 = calc_dtr_trend_jja(temp_max_0p2, temp_min_0p2, 1900, 2015)
trend_dtr_0p2_1951_1980 = calc_dtr_trend_jja(temp_max_0p2, temp_min_0p2, 1950, 1981)
trend_dtr_0p2_1981_2014 = calc_dtr_trend_jja(temp_max_0p2, temp_min_0p2, 1980, 2015)

trend_dtr_1p5_1901_2014 = calc_dtr_trend_jja(temp_max_1p5, temp_min_1p5, 1900, 2015)
trend_dtr_1p5_1951_1980 = calc_dtr_trend_jja(temp_max_1p5, temp_min_1p5, 1950, 1981)
trend_dtr_1p5_1981_2014 = calc_dtr_trend_jja(temp_max_1p5, temp_min_1p5, 1980, 2015)

###### summary

def calc_summary_trend(temp_max_1, temp_max_2, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_max_1=temp_max_1["tasmax"] - 273.15
    tmax_rol_1 = global_max_1.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmax_1=tmax_rol_1[0::12].groupby('time.year').mean('time')
    mask_tmax_1=((jja_tmax_1.year > year1) & (jja_tmax_1.year < year2))
    trend_ds_tmax_1=jja_tmax_1[mask_tmax_1].polyfit(dim='year',deg=1,skipna=True)
    trend_tmax_1 = trend_ds_tmax_1['polyfit_coefficients'][0]
    global_max_2=temp_max_2["tasmax"] - 273.15
    tmax_rol_2 = global_max_2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmax_2=tmax_rol_2[0::12].groupby('time.year').mean('time')
    mask_tmax_2=((jja_tmax_2.year > year1) & (jja_tmax_2.year < year2))
    trend_ds_tmax_2=jja_tmax_2[mask_tmax_2].polyfit(dim='year',deg=1,skipna=True)
    trend_tmax_2 = trend_ds_tmax_2['polyfit_coefficients'][0]
    trend = trend_tmax_2 - trend_tmax_1
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_summary_1901_2014 = calc_summary_trend(temp_max_0p2, temp_max_1p5, 1900, 2015)
trend_summary_1951_1980 = calc_summary_trend(temp_max_0p2, temp_max_1p5, 1950, 1981)
trend_summary_1981_2014 = calc_summary_trend(temp_max_0p2, temp_max_1p5, 1980, 2015)

###
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
    
cf1 = axlist[0].contourf(trend_max_0p2_1901_2014[1],lat,trend_max_0p2_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_max_1p5_1901_2014[1],lat,trend_max_1p5_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\n1.5x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014[1],lat,trend_summary_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('1.5x and 0.2x ensemble member mean difference\n',fontsize=16)

cf4 = axlist[3].contourf(trend_max_0p2_1951_1980[1],lat,trend_max_0p2_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(trend_max_1p5_1951_1980[1],lat,trend_max_1p5_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(trend_summary_1951_1980[1],lat,trend_summary_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)

cf7 = axlist[6].contourf(trend_max_0p2_1981_2014[1],lat,trend_max_0p2_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(trend_max_1p5_1981_2014[1],lat,trend_max_1p5_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1981_2014[1],lat,trend_summary_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")


fig.suptitle('DJF Maximum Temperature Trend Summary\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/summary_0p2_1p5_DJF_max_trend_new.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### minimum

def calc_summary_trend_min(temp_min_1, temp_min_2, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min_1=temp_min_1["tasmin"] - 273.15
    tmin_rol_1 = global_min_1.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmin_1=tmin_rol_1[0::12].groupby('time.year').mean('time')
    mask_tmin_1=((jja_tmin_1.year > year1) & (jja_tmin_1.year < year2))
    trend_ds_tmin_1=jja_tmin_1[mask_tmin_1].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_1 = trend_ds_tmin_1['polyfit_coefficients'][0]
    global_min_2=temp_min_2["tasmin"] - 273.15
    tmin_rol_2 = global_min_2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmin_2=tmin_rol_2[0::12].groupby('time.year').mean('time')
    mask_tmin_2=((jja_tmin_2.year > year1) & (jja_tmin_2.year < year2))
    trend_ds_tmin_2=jja_tmin_2[mask_tmin_2].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_2 = trend_ds_tmin_2['polyfit_coefficients'][0]
    trend = trend_tmin_2 - trend_tmin_1
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_summary_1901_2014_min = calc_summary_trend_min(temp_min_0p2, temp_min_1p5, 1900, 2015)
trend_summary_1951_1980_min = calc_summary_trend_min(temp_min_0p2, temp_min_1p5, 1950, 1981)
trend_summary_1981_2014_min = calc_summary_trend_min(temp_min_0p2, temp_min_1p5, 1980, 2015)


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
    
cf1 = axlist[0].contourf(trend_min_0p2_1901_2014[1],lat,trend_min_0p2_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_min_1p5_1901_2014[1],lat,trend_min_1p5_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\n1.5x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014_min[1],lat,trend_summary_1901_2014_min[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('1.5x and 0.2x ensemble member mean difference\n',fontsize=16)

cf4 = axlist[3].contourf(trend_min_0p2_1951_1980[1],lat,trend_min_0p2_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(trend_min_1p5_1951_1980[1],lat,trend_min_1p5_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(trend_summary_1951_1980_min[1],lat,trend_summary_1951_1980_min[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)

cf7 = axlist[6].contourf(trend_min_0p2_1981_2014[1],lat,trend_min_0p2_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(trend_min_1p5_1981_2014[1],lat,trend_min_1p5_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1981_2014_min[1],lat,trend_summary_1981_2014_min[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('\n',fontsize=16)

fig.suptitle('DJF Minimum Temperature Trend Summary\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/summary_0p2_1p5_DJF_min_trend_new.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### dtr

def calc_summary_trend_dtr(temp_max_1, temp_min_1, temp_max_2, temp_min_2, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    
    global_dtr_1=((temp_max_1["tasmax"] - 273.15) - (temp_min_1["tasmin"] - 273.15))
    dtr_rol_1 = global_dtr_1.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_1=dtr_rol_1[0::12].groupby('time.year').mean('time')
    mask_1=((jja_1.year > year1) & (jja_1.year < year2))
    trend_ds_1=jja_1[mask_1].polyfit(dim='year',deg=1,skipna=True)
    trend_1 = trend_ds_1['polyfit_coefficients'][0]
    
    global_dtr_2=((temp_max_2["tasmax"] - 273.15) - (temp_min_2["tasmin"] - 273.15))
    dtr_rol_2 = global_dtr_2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_2=dtr_rol_2[0::12].groupby('time.year').mean('time')
    mask_2=((jja_2.year > year1) & (jja_2.year < year2))
    trend_ds_2=jja_2[mask_2].polyfit(dim='year',deg=1,skipna=True)
    trend_2 = trend_ds_2['polyfit_coefficients'][0]
    
    trend = trend_2 - trend_1
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_summary_1901_2014_dtr = calc_summary_trend_dtr(temp_max_0p2, temp_min_0p2, temp_max_1p5, temp_min_1p5, 1900, 2015)
trend_summary_1951_1980_dtr = calc_summary_trend_dtr(temp_max_0p2, temp_min_0p2, temp_max_1p5, temp_min_1p5, 1950, 1981)
trend_summary_1981_2014_dtr = calc_summary_trend_dtr(temp_max_0p2, temp_min_0p2, temp_max_1p5, temp_min_1p5, 1980, 2015)

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
    
cf1 = axlist[0].contourf(trend_dtr_0p2_1901_2014[1],lat,trend_dtr_0p2_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_dtr_1p5_1901_2014[1],lat,trend_dtr_1p5_1901_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\n1.5x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014_dtr[1],lat,trend_summary_1901_2014_dtr[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('1.5x and 0.2x ensemble member mean difference\n',fontsize=16)

cf4 = axlist[3].contourf(trend_dtr_0p2_1951_1980[1],lat,trend_dtr_0p2_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(trend_dtr_1p5_1951_1980[1],lat,trend_dtr_1p5_1951_1980[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(trend_summary_1951_1980_dtr[1],lat,trend_summary_1951_1980_dtr[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)

cf7 = axlist[6].contourf(trend_dtr_0p2_1981_2014[1],lat,trend_dtr_0p2_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(trend_dtr_1p5_1981_2014[1],lat,trend_dtr_1p5_1981_2014[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1981_2014_dtr[1],lat,trend_summary_1981_2014_dtr[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('\n',fontsize=16)

fig.suptitle('DJF Diurnal Temperature Range Trend Summary\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/summary_0p2_1p5_DJF_dtr_trend_new.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()
