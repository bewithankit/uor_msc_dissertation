#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 22:01:47 2022

@author: ankit
"""

#import cf, cfplot as cfp
import numpy as np
import scipy as sp
import sys
import os
from netCDF4 import Dataset
import netCDF4 as nc
%matplotlib inline
import matplotlib
import xarray as xr
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import matplotlib.path as mpath
# Quick plot to show the results
from cartopy import config
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import scipy.ndimage as ndimage
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import addcyclic
from cartopy.util import add_cyclic_point
import matplotlib.pylab as pl
from scipy import stats
from cartopy.util import add_cyclic_point
import seaborn as sns
import missingno as msno

with np.printoptions(threshold=np.inf):
    print(np.count_nonzero(np.isnan(jja[mask0]),axis=0))


temp_min = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmn.dat.lowres.nc')
temp_max = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')
lat = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['latitude']
lon = xr.open_dataset('/Users/ankit/Documents/UoR/Dissertation/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['longitude']

## maximum

weights = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights.name = "weights"
global_max=temp_max["tmx"]

#tmax_rol = global_max.weighted(weights).mean(("longitude", "latitude")
jja = global_max.groupby('time.year').mean('time')

mask0=((jja.year > 1900) & (jja.year < 2015))
print(jja.year[mask0])
trend_ds=jja[mask0,:,:].polyfit(dim='year',deg=1,skipna=True)
trend=trend_ds['polyfit_coefficients'][0,:,:]


percent1 = 0.75*len(jja.year[mask0])

trend_mask=np.where(np.count_nonzero(~np.isnan(jja[mask0]),axis=0) >= percent1, trend, np.nan)
trend_maskc,lonc = add_cyclic_point(trend_mask,lon,axis=1)

# with np.printoptions(threshold=np.inf):
#    print(np.count_nonzero(~np.isnan(jja[mask0]),axis=0))

# time mask, calculate trend over subset of data 1951-1980

mask=((jja.year > 1950) & (jja.year < 1981))
percent2 = 0.75*len(jja.year[mask])
print(jja.year[mask])
trend_ds_sub=jja[mask,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_sub=trend_ds_sub['polyfit_coefficients'][0,:,:]

trend_sub_mask=np.where(np.count_nonzero(~np.isnan(jja[mask]),axis=0) >= percent2, trend_sub, np.nan)
trend_sub_maskc,lonc = add_cyclic_point(trend_sub_mask,lon,axis=1)

mask2=((jja.year > 1980) & (jja.year < 2015))
percent3 = 0.75*len(jja.year[mask2])
print(jja.year[mask2])
trend_ds_sub2=jja[mask2,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_sub2=trend_ds_sub2['polyfit_coefficients'][0,:,:]

trend_sub2_mask=np.where(np.count_nonzero(~np.isnan(jja[mask2]),axis=0) >= percent3, trend_sub2, np.nan)
trend_sub2_maskc,lonc = add_cyclic_point(trend_sub2_mask,lon,axis=1)

## minimum

weights_min = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_min.name = "weights2"
global_min=temp_min["tmn"]

#tmin_rol = global_min.weighted(weights).mean(("longitude", "latitude")
jja_min = global_min.groupby('time.year').mean('time')


mask0_min=((jja_min.year > 1900) & (jja_min.year < 2015))
print(jja_min.year[mask0_min])
trend_ds_min=jja_min[mask0_min,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_min=trend_ds_min['polyfit_coefficients'][0,:,:]

trend_min_mask=np.where(np.count_nonzero(~np.isnan(jja_min[mask0_min]),axis=0) >= percent1, trend_min, np.nan)
trend_min_maskc,lonc = add_cyclic_point(trend_min_mask,lon,axis=1)

# time mask, calculate trend over subset of data 1951-1980

mask_min=((jja_min.year > 1950) & (jja_min.year < 1981))
print(jja_min.year[mask_min])
trend_ds_sub_min=jja_min[mask_min,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_sub_min=trend_ds_sub_min['polyfit_coefficients'][0,:,:]

trend_sub_min_mask=np.where(np.count_nonzero(~np.isnan(jja_min[mask_min]),axis=0) >= percent2, trend_sub_min, np.nan)
trend_sub_min_maskc,lonc = add_cyclic_point(trend_sub_min_mask,lon,axis=1)

mask2_min=((jja_min.year > 1980) & (jja_min.year < 2015))
print(jja_min.year[mask2_min])
trend_ds_sub2_min=jja_min[mask2_min,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_min=trend_ds_sub2_min['polyfit_coefficients'][0,:,:]

trend_sub2_min_mask=np.where(np.count_nonzero(~np.isnan(jja_min[mask2_min]),axis=0) >= percent3, trend_sub2_min, np.nan)
trend_sub2_min_maskc,lonc = add_cyclic_point(trend_sub2_min_mask,lon,axis=1)

## dtr

weights_dtr = np.cos(np.deg2rad(lat))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
weights_dtr.name = "weights3"
global_dtr = temp_max["tmx"] - temp_min["tmn"]

#dtr_rol = global_dtr.weighted(weights).mean(("longitude", "latitude")
jja_dtr = global_dtr.groupby('time.year').mean('time')


mask0_dtr=((jja_dtr.year > 1900) & (jja_dtr.year < 2015))
print(jja_dtr.year[mask0_dtr])
trend_ds_dtr=jja_dtr[mask0_dtr,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_dtr=trend_ds_dtr['polyfit_coefficients'][0,:,:]

trend_dtr_mask=np.where(np.count_nonzero(~np.isnan(jja_dtr[mask0_dtr]),axis=0) >= percent1, trend_dtr, np.nan)
trend_dtr_maskc,lonc = add_cyclic_point(trend_dtr_mask,lon,axis=1)


mask_dtr=((jja_dtr.year > 1950) & (jja_dtr.year < 1981))
print(jja_dtr.year[mask_dtr])
trend_ds_sub_dtr=jja_dtr[mask_dtr,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_sub_dtr=trend_ds_sub_dtr['polyfit_coefficients'][0,:,:]

trend_sub_dtr_mask=np.where(np.count_nonzero(~np.isnan(jja_dtr[mask_dtr]),axis=0) >= percent2, trend_sub_dtr, np.nan)
trend_sub_dtr_maskc,lonc = add_cyclic_point(trend_sub_dtr_mask,lon,axis=1)

mask2_dtr=((jja_dtr.year > 1980) & (jja_dtr.year < 2015))
print(jja_dtr.year[mask2_dtr])
trend_ds_sub2_dtr=jja_dtr[mask2_dtr,:,:].polyfit(dim='year',deg=1,skipna=True)
trend_sub2_dtr=trend_ds_sub2_dtr['polyfit_coefficients'][0,:,:]

trend_sub2_dtr_mask=np.where(np.count_nonzero(~np.isnan(jja_dtr[mask2_dtr]),axis=0) >= percent3, trend_sub2_dtr, np.nan)
trend_sub2_dtr_maskc,lonc = add_cyclic_point(trend_sub2_dtr_mask,lon,axis=1)

## plot

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

cf1 = axlist[0].contourf(lonc,lat,trend_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('Maximum Temperature Trend\n',fontsize=16)
cf2 = axlist[1].contourf(lonc,lat,trend_min_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('\n(1901-2014)\n\nMinimum Temperature Trend\n',fontsize=16)
cf3 = axlist[2].contourf(lonc,lat,trend_dtr_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('Diurnal Temperature Range (DTR) Trend\n',fontsize=16)


cf4 = axlist[3].contourf(lonc,lat,trend_sub_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('\n',fontsize=16)
cf5 = axlist[4].contourf(lonc,lat,trend_sub_min_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('\n\n(1951-1980)\n',fontsize=16)
cf6 = axlist[5].contourf(lonc,lat,trend_sub_dtr_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[5].set_title('\n',fontsize=16)


cf7 = axlist[6].contourf(lonc,lat,trend_sub2_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[6].set_title('\n',fontsize=16)
cf8 = axlist[7].contourf(lonc,lat,trend_sub2_min_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n\n(1981-2014)\n',fontsize=16)
cf9 = axlist[8].contourf(lonc,lat,trend_sub2_dtr_maskc*10,np.linspace(-1,1,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[8].set_title('\n',fontsize=16)

fig.suptitle('CRU TS v4 Annual Temperature Trends\n', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[6:9],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/Users/ankit/Documents/UoR/Dissertation/annualtrends_testt.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()