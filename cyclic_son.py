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

maxfiles_0p4 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")
minfiles_0p4 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")

maxfiles_0p7 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/max/tasmax_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")
minfiles_0p7 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")

maxfiles_1p0 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")
minfiles_1p0 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

maxfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/max/tasmax_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")
minfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")

preprocess=preprocess

temp_max_0p2 = xr.open_mfdataset(maxfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_0p2 = xr.open_mfdataset(minfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)

temp_max_0p4 = xr.open_mfdataset(maxfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_0p4 = xr.open_mfdataset(minfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)

temp_max_0p7 = xr.open_mfdataset(maxfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_0p7 = xr.open_mfdataset(minfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)

temp_max_1p0 = xr.open_mfdataset(maxfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_1p0 = xr.open_mfdataset(minfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)

temp_max_1p5 = xr.open_mfdataset(maxfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)
temp_min_1p5 = xr.open_mfdataset(minfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)

def calc_tmax_trend_jja(temp_max, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_max=temp_max["tasmax"] - 273.15
    tmax_rol = global_max.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja=tmax_rol[9::12].groupby('time.year').mean('time')
    mask=((jja.year > year1) & (jja.year < year2))
    trend_ds=jja[mask].polyfit(dim='year',deg=1,skipna=True)
    trend = trend_ds['polyfit_coefficients'][0]
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_max_0p2_1901_2014 = calc_tmax_trend_jja(temp_max_0p2, 1900, 2015)
trend_max_0p2_1951_1980 = calc_tmax_trend_jja(temp_max_0p2, 1950, 1981)
trend_max_0p2_1981_2014 = calc_tmax_trend_jja(temp_max_0p2, 1980, 2015)

trend_max_0p4_1901_2014 = calc_tmax_trend_jja(temp_max_0p4, 1900, 2015)
trend_max_0p4_1951_1980 = calc_tmax_trend_jja(temp_max_0p4, 1950, 1981)
trend_max_0p4_1981_2014 = calc_tmax_trend_jja(temp_max_0p4, 1980, 2015)

trend_max_0p7_1901_2014 = calc_tmax_trend_jja(temp_max_0p7, 1900, 2015)
trend_max_0p7_1951_1980 = calc_tmax_trend_jja(temp_max_0p7, 1950, 1981)
trend_max_0p7_1981_2014 = calc_tmax_trend_jja(temp_max_0p7, 1980, 2015)

trend_max_1p0_1901_2014 = calc_tmax_trend_jja(temp_max_1p0, 1900, 2015)
trend_max_1p0_1951_1980 = calc_tmax_trend_jja(temp_max_1p0, 1950, 1981)
trend_max_1p0_1981_2014 = calc_tmax_trend_jja(temp_max_1p0, 1980, 2015)

trend_max_1p5_1901_2014 = calc_tmax_trend_jja(temp_max_1p5, 1900, 2015)
trend_max_1p5_1951_1980 = calc_tmax_trend_jja(temp_max_1p5, 1950, 1981)
trend_max_1p5_1981_2014 = calc_tmax_trend_jja(temp_max_1p5, 1980, 2015)


def calc_tmin_trend_jja(temp_min, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min=temp_min["tasmin"] - 273.15
    tmin_rol = global_min.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja=tmin_rol[9::12].groupby('time.year').mean('time')
    mask=((jja.year > year1) & (jja.year < year2))
    trend_ds=jja[mask].polyfit(dim='year',deg=1,skipna=True)
    trend = trend_ds['polyfit_coefficients'][0]
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_min_0p2_1901_2014 = calc_tmin_trend_jja(temp_min_0p2, 1900, 2015)
trend_min_0p2_1951_1980 = calc_tmin_trend_jja(temp_min_0p2, 1950, 1981)
trend_min_0p2_1981_2014 = calc_tmin_trend_jja(temp_min_0p2, 1980, 2015)

trend_min_0p4_1901_2014 = calc_tmin_trend_jja(temp_min_0p4, 1900, 2015)
trend_min_0p4_1951_1980 = calc_tmin_trend_jja(temp_min_0p4, 1950, 1981)
trend_min_0p4_1981_2014 = calc_tmin_trend_jja(temp_min_0p4, 1980, 2015)

trend_min_0p7_1901_2014 = calc_tmin_trend_jja(temp_min_0p7, 1900, 2015)
trend_min_0p7_1951_1980 = calc_tmin_trend_jja(temp_min_0p7, 1950, 1981)
trend_min_0p7_1981_2014 = calc_tmin_trend_jja(temp_min_0p7, 1980, 2015)

trend_min_1p0_1901_2014 = calc_tmin_trend_jja(temp_min_1p0, 1900, 2015)
trend_min_1p0_1951_1980 = calc_tmin_trend_jja(temp_min_1p0, 1950, 1981)
trend_min_1p0_1981_2014 = calc_tmin_trend_jja(temp_min_1p0, 1980, 2015)

trend_min_1p5_1901_2014 = calc_tmin_trend_jja(temp_min_1p5, 1900, 2015)
trend_min_1p5_1951_1980 = calc_tmin_trend_jja(temp_min_1p5, 1950, 1981)
trend_min_1p5_1981_2014 = calc_tmin_trend_jja(temp_min_1p5, 1980, 2015)

def calc_dtr_trend_jja(temp_max, temp_min, year1, year2):
    weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_dtr=((temp_max["tasmax"] - 273.15) - (temp_min["tasmin"] - 273.15))
    dtr_rol = global_dtr.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja=dtr_rol[9::12].groupby('time.year').mean('time')
    mask=((jja.year > year1) & (jja.year < year2))
    trend_ds=jja[mask].polyfit(dim='year',deg=1,skipna=True)
    trend = trend_ds['polyfit_coefficients'][0]
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_dtr_0p2_1901_2014 = calc_dtr_trend_jja(temp_max_0p2, temp_min_0p2, 1900, 2015)
trend_dtr_0p2_1951_1980 = calc_dtr_trend_jja(temp_max_0p2, temp_min_0p2, 1950, 1981)
trend_dtr_0p2_1981_2014 = calc_dtr_trend_jja(temp_max_0p2, temp_min_0p2, 1980, 2015)

trend_dtr_0p4_1901_2014 = calc_dtr_trend_jja(temp_max_0p4, temp_min_0p4, 1900, 2015)
trend_dtr_0p4_1951_1980 = calc_dtr_trend_jja(temp_max_0p4, temp_min_0p4, 1950, 1981)
trend_dtr_0p4_1981_2014 = calc_dtr_trend_jja(temp_max_0p4, temp_min_0p4, 1980, 2015)

trend_dtr_0p7_1901_2014 = calc_dtr_trend_jja(temp_max_0p7, temp_min_0p7, 1900, 2015)
trend_dtr_0p7_1951_1980 = calc_dtr_trend_jja(temp_max_0p7, temp_min_0p7, 1950, 1981)
trend_dtr_0p7_1981_2014 = calc_dtr_trend_jja(temp_max_0p7, temp_min_0p7, 1980, 2015)

trend_dtr_1p0_1901_2014 = calc_dtr_trend_jja(temp_max_1p0, temp_min_1p0, 1900, 2015)
trend_dtr_1p0_1951_1980 = calc_dtr_trend_jja(temp_max_1p0, temp_min_1p0, 1950, 1981)
trend_dtr_1p0_1981_2014 = calc_dtr_trend_jja(temp_max_1p0, temp_min_1p0, 1980, 2015)

trend_dtr_1p5_1901_2014 = calc_dtr_trend_jja(temp_max_1p5, temp_min_1p5, 1900, 2015)
trend_dtr_1p5_1951_1980 = calc_dtr_trend_jja(temp_max_1p5, temp_min_1p5, 1950, 1981)
trend_dtr_1p5_1981_2014 = calc_dtr_trend_jja(temp_max_1p5, temp_min_1p5, 1980, 2015)


###
f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=5, figsize=(25, 13), constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf1 = axlist[0].contourf(trend_max_0p2_1901_2014[1],lat,trend_max_0p2_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_max_0p4_1901_2014[1],lat,trend_max_0p4_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_max_0p7_1901_2014[1],lat,trend_max_0p7_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_max_1p0_1901_2014[1],lat,trend_max_1p0_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_max_1p5_1901_2014[1],lat,trend_max_1p5_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_max_0p2_1951_1980[1],lat,trend_max_0p2_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_max_0p4_1951_1980[1],lat,trend_max_0p4_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_max_0p7_1951_1980[1],lat,trend_max_0p7_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_max_1p0_1951_1980[1],lat,trend_max_1p0_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_max_1p5_1951_1980[1],lat,trend_max_1p5_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_max_0p2_1981_2014[1],lat,trend_max_0p2_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_max_0p4_1981_2014[1],lat,trend_max_0p4_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_max_0p7_1981_2014[1],lat,trend_max_0p7_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_max_1p0_1981_2014[1],lat,trend_max_1p0_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_max_1p5_1981_2014[1],lat,trend_max_1p5_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('SON Maximum Temperature Trend', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
#plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_SON_max_trend_cyclic.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### minimum

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=5, figsize=(25, 13), constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf1 = axlist[0].contourf(trend_min_0p2_1901_2014[1],lat,trend_min_0p2_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_min_0p4_1901_2014[1],lat,trend_min_0p4_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_min_0p7_1901_2014[1],lat,trend_min_0p7_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_min_1p0_1901_2014[1],lat,trend_min_1p0_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_min_1p5_1901_2014[1],lat,trend_min_1p5_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_min_0p2_1951_1980[1],lat,trend_min_0p2_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_min_0p4_1951_1980[1],lat,trend_min_0p4_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_min_0p7_1951_1980[1],lat,trend_min_0p7_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_min_1p0_1951_1980[1],lat,trend_min_1p0_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_min_1p5_1951_1980[1],lat,trend_min_1p5_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_min_0p2_1981_2014[1],lat,trend_min_0p2_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_min_0p4_1981_2014[1],lat,trend_min_0p4_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_min_0p7_1981_2014[1],lat,trend_min_0p7_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_min_1p0_1981_2014[1],lat,trend_min_1p0_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_min_1p5_1981_2014[1],lat,trend_min_1p5_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('SON Minimum Temperature Trend', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_SON_min_trend_cyclic.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### dtr

f=plt.figure() 
crs = ccrs.Robinson(central_longitude=0., globe=None)

def plot_background(ax):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.set_anchor('S')
    return ax

fig, axarr = plt.subplots(nrows=3, ncols=5, figsize=(25, 13), constrained_layout=True, subplot_kw={'projection': crs})

axlist = axarr.flatten()
for ax in axlist:
    plot_background(ax)

cf1 = axlist[0].contourf(trend_dtr_0p2_1901_2014[1],lat,trend_dtr_0p2_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_dtr_0p4_1901_2014[1],lat,trend_dtr_0p4_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_dtr_0p7_1901_2014[1],lat,trend_dtr_0p7_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_dtr_1p0_1901_2014[1],lat,trend_dtr_1p0_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_dtr_1p5_1901_2014[1],lat,trend_dtr_1p5_1901_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_dtr_0p2_1951_1980[1],lat,trend_dtr_0p2_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_dtr_0p4_1951_1980[1],lat,trend_dtr_0p4_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_dtr_0p7_1951_1980[1],lat,trend_dtr_0p7_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_dtr_1p0_1951_1980[1],lat,trend_dtr_1p0_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_dtr_1p5_1951_1980[1],lat,trend_dtr_1p5_1951_1980[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_dtr_0p2_1981_2014[1],lat,trend_dtr_0p2_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_dtr_0p4_1981_2014[1],lat,trend_dtr_0p4_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_dtr_0p7_1981_2014[1],lat,trend_dtr_0p7_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_dtr_1p0_1981_2014[1],lat,trend_dtr_1p0_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_dtr_1p5_1981_2014[1],lat,trend_dtr_1p5_1981_2014[0]*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('SON Diurnal Temperature Range Trend', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/ensemble_SON_dtr_trend_cyclic.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()