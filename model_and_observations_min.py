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

temp_min_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmn.dat.lowres.nc')

lat_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['latitude']
lon_obs = xr.open_dataset('/storage/silver/metstudent/msc/users_2022/ty824157/cru_ts4.05.1901.2020.tmx.dat.lowres.nc')['longitude']

minfiles = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

minfiles_0p2 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p2/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p2_*_gn_18500101-20141230.nc")

minfiles_0p4 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p4/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p4_*_gn_18500101-20141230.nc")

minfiles_0p7 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/0p7/min/tasmin_monthly_HadGEM3-GC3-1_hist-0p7_*_gn_18500101-20141230.nc")

minfiles_1p0 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p0/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p0_*_gn_18500101-20141230.nc")

minfiles_1p5 = glob.glob("/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/1p5/min/tasmin_monthly_HadGEM3-GC3-1_hist-1p5_*_gn_18500101-20141230.nc")

preprocess=preprocess

temp_min_0p2 = xr.open_mfdataset(minfiles_0p2,concat_dim='members',combine='nested',preprocess=preprocess)

temp_min_0p4 = xr.open_mfdataset(minfiles_0p4,concat_dim='members',combine='nested',preprocess=preprocess)

temp_min_0p7 = xr.open_mfdataset(minfiles_0p7,concat_dim='members',combine='nested',preprocess=preprocess)

temp_min_1p0 = xr.open_mfdataset(minfiles_1p0,concat_dim='members',combine='nested',preprocess=preprocess)

temp_min_1p5 = xr.open_mfdataset(minfiles_1p5,concat_dim='members',combine='nested',preprocess=preprocess)


def calc_summary_trend_min(temp_min_1, temp_min_2, year1, year2):
    weights = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min_1=temp_min_1["tmn"] 
    tmin_rol_1 = global_min_1.rolling(min_periods=3, center=True, time=3).mean()
    jja_tmin_1=tmin_rol_1[3::12].groupby('time.year').mean('time')
    mask_tmin_1=((jja_tmin_1.year > year1) & (jja_tmin_1.year < year2))
    trend_ds_tmin_1=jja_tmin_1[mask_tmin_1].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_1 = trend_ds_tmin_1['polyfit_coefficients'][0]
    global_min_2=temp_min_2["tasmin"] - 273.15
    tmin_rol_2 = global_min_2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmin_2=tmin_rol_2[3::12].groupby('time.year').mean('time')
    mask_tmin_2=((jja_tmin_2.year > year1) & (jja_tmin_2.year < year2))
    trend_ds_tmin_2=jja_tmin_2[mask_tmin_2].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_2 = trend_ds_tmin_2['polyfit_coefficients'][0]
    trend = trend_tmin_2 - trend_tmin_1
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_summary_1901_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1900, 2015)
trend_summary_1901_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1900, 2015)
trend_summary_1901_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1900, 2015)
trend_summary_1901_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1900, 2015)
trend_summary_1901_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1900, 2015)

trend_summary_1951_1980_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1950, 1981)
trend_summary_1951_1980_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1950, 1981)
trend_summary_1951_1980_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1950, 1981)
trend_summary_1951_1980_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1950, 1981)
trend_summary_1951_1980_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1950, 1981)

trend_summary_1981_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1980, 2015)
trend_summary_1981_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1980, 2015)
trend_summary_1981_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1980, 2015)
trend_summary_1981_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1980, 2015)
trend_summary_1981_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1980, 2015)

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

cf1 = axlist[0].contourf(trend_summary_1901_2014_min_0p2[1],lat,trend_summary_1901_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_summary_1901_2014_min_0p4[1],lat,trend_summary_1901_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014_min_0p7[1],lat,trend_summary_1901_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_summary_1901_2014_min_1p0[1],lat,trend_summary_1901_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_summary_1901_2014_min_1p5[1],lat,trend_summary_1901_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_summary_1951_1980_min_0p2[1],lat,trend_summary_1951_1980_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_summary_1951_1980_min_0p4[1],lat,trend_summary_1951_1980_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_summary_1951_1980_min_0p7[1],lat,trend_summary_1951_1980_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1951_1980_min_1p0[1],lat,trend_summary_1951_1980_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_summary_1951_1980_min_1p5[1],lat,trend_summary_1951_1980_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_summary_1981_2014_min_0p2[1],lat,trend_summary_1981_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_summary_1981_2014_min_0p4[1],lat,trend_summary_1981_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_summary_1981_2014_min_0p7[1],lat,trend_summary_1981_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_summary_1981_2014_min_1p0[1],lat,trend_summary_1981_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_summary_1981_2014_min_1p5[1],lat,trend_summary_1981_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('MAM Minimum Temperature Trend: Model and Observation difference', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/MAM_model-obs_min.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### jja

def calc_summary_trend_min(temp_min_1, temp_min_2, year1, year2):
    weights = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min_1=temp_min_1["tmn"] 
    tmin_rol_1 = global_min_1.rolling(min_periods=3, center=True, time=3).mean()
    jja_tmin_1=tmin_rol_1[6::12].groupby('time.year').mean('time')
    mask_tmin_1=((jja_tmin_1.year > year1) & (jja_tmin_1.year < year2))
    trend_ds_tmin_1=jja_tmin_1[mask_tmin_1].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_1 = trend_ds_tmin_1['polyfit_coefficients'][0]
    global_min_2=temp_min_2["tasmin"] - 273.15
    tmin_rol_2 = global_min_2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmin_2=tmin_rol_2[6::12].groupby('time.year').mean('time')
    mask_tmin_2=((jja_tmin_2.year > year1) & (jja_tmin_2.year < year2))
    trend_ds_tmin_2=jja_tmin_2[mask_tmin_2].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_2 = trend_ds_tmin_2['polyfit_coefficients'][0]
    trend = trend_tmin_2 - trend_tmin_1
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_summary_1901_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1900, 2015)
trend_summary_1901_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1900, 2015)
trend_summary_1901_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1900, 2015)
trend_summary_1901_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1900, 2015)
trend_summary_1901_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1900, 2015)

trend_summary_1951_1980_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1950, 1981)
trend_summary_1951_1980_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1950, 1981)
trend_summary_1951_1980_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1950, 1981)
trend_summary_1951_1980_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1950, 1981)
trend_summary_1951_1980_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1950, 1981)

trend_summary_1981_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1980, 2015)
trend_summary_1981_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1980, 2015)
trend_summary_1981_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1980, 2015)
trend_summary_1981_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1980, 2015)
trend_summary_1981_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1980, 2015)

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

cf1 = axlist[0].contourf(trend_summary_1901_2014_min_0p2[1],lat,trend_summary_1901_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_summary_1901_2014_min_0p4[1],lat,trend_summary_1901_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014_min_0p7[1],lat,trend_summary_1901_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_summary_1901_2014_min_1p0[1],lat,trend_summary_1901_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_summary_1901_2014_min_1p5[1],lat,trend_summary_1901_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_summary_1951_1980_min_0p2[1],lat,trend_summary_1951_1980_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_summary_1951_1980_min_0p4[1],lat,trend_summary_1951_1980_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_summary_1951_1980_min_0p7[1],lat,trend_summary_1951_1980_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1951_1980_min_1p0[1],lat,trend_summary_1951_1980_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_summary_1951_1980_min_1p5[1],lat,trend_summary_1951_1980_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_summary_1981_2014_min_0p2[1],lat,trend_summary_1981_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_summary_1981_2014_min_0p4[1],lat,trend_summary_1981_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_summary_1981_2014_min_0p7[1],lat,trend_summary_1981_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_summary_1981_2014_min_1p0[1],lat,trend_summary_1981_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_summary_1981_2014_min_1p5[1],lat,trend_summary_1981_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('JJA Minimum Temperature Trend: Model and Observation difference', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/JJA_model-obs_min.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

##### SON

def calc_summary_trend_min(temp_min_1, temp_min_2, year1, year2):
    weights = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min_1=temp_min_1["tmn"] 
    tmin_rol_1 = global_min_1.rolling(min_periods=3, center=True, time=3).mean()
    jja_tmin_1=tmin_rol_1[9::12].groupby('time.year').mean('time')
    mask_tmin_1=((jja_tmin_1.year > year1) & (jja_tmin_1.year < year2))
    trend_ds_tmin_1=jja_tmin_1[mask_tmin_1].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_1 = trend_ds_tmin_1['polyfit_coefficients'][0]
    global_min_2=temp_min_2["tasmin"] - 273.15
    tmin_rol_2 = global_min_2.rolling(min_periods=3, center=True, time=3).mean().mean('members')
    jja_tmin_2=tmin_rol_2[9::12].groupby('time.year').mean('time')
    mask_tmin_2=((jja_tmin_2.year > year1) & (jja_tmin_2.year < year2))
    trend_ds_tmin_2=jja_tmin_2[mask_tmin_2].polyfit(dim='year',deg=1,skipna=True)
    trend_tmin_2 = trend_ds_tmin_2['polyfit_coefficients'][0]
    trend = trend_tmin_2 - trend_tmin_1
    trend2,lon2 = add_cyclic_point(trend,lon,axis=1)
    return trend2, lon2

trend_summary_1901_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1900, 2015)
trend_summary_1901_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1900, 2015)
trend_summary_1901_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1900, 2015)
trend_summary_1901_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1900, 2015)
trend_summary_1901_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1900, 2015)

trend_summary_1951_1980_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1950, 1981)
trend_summary_1951_1980_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1950, 1981)
trend_summary_1951_1980_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1950, 1981)
trend_summary_1951_1980_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1950, 1981)
trend_summary_1951_1980_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1950, 1981)

trend_summary_1981_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1980, 2015)
trend_summary_1981_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1980, 2015)
trend_summary_1981_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1980, 2015)
trend_summary_1981_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1980, 2015)
trend_summary_1981_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1980, 2015)

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

cf1 = axlist[0].contourf(trend_summary_1901_2014_min_0p2[1],lat,trend_summary_1901_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_summary_1901_2014_min_0p4[1],lat,trend_summary_1901_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014_min_0p7[1],lat,trend_summary_1901_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_summary_1901_2014_min_1p0[1],lat,trend_summary_1901_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_summary_1901_2014_min_1p5[1],lat,trend_summary_1901_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_summary_1951_1980_min_0p2[1],lat,trend_summary_1951_1980_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_summary_1951_1980_min_0p4[1],lat,trend_summary_1951_1980_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_summary_1951_1980_min_0p7[1],lat,trend_summary_1951_1980_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1951_1980_min_1p0[1],lat,trend_summary_1951_1980_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_summary_1951_1980_min_1p5[1],lat,trend_summary_1951_1980_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_summary_1981_2014_min_0p2[1],lat,trend_summary_1981_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_summary_1981_2014_min_0p4[1],lat,trend_summary_1981_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_summary_1981_2014_min_0p7[1],lat,trend_summary_1981_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_summary_1981_2014_min_1p0[1],lat,trend_summary_1981_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_summary_1981_2014_min_1p5[1],lat,trend_summary_1981_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('SON Minimum Temperature Trend: Model and Observation difference', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/SON_model-obs_min.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()

### DJF

def calc_summary_trend_min(temp_min_1, temp_min_2, year1, year2):
    weights = np.cos(np.deg2rad(lat_obs))  #global means need to be weighted with the cosine of the latitude to account for the changing grid cell area with latitude (smallest at the poles)
    weights.name = "weights"
    global_min_1=temp_min_1["tmn"] 
    tmin_rol_1 = global_min_1.rolling(min_periods=3, center=True, time=3).mean()
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

trend_summary_1901_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1900, 2015)
trend_summary_1901_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1900, 2015)
trend_summary_1901_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1900, 2015)
trend_summary_1901_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1900, 2015)
trend_summary_1901_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1900, 2015)

trend_summary_1951_1980_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1950, 1981)
trend_summary_1951_1980_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1950, 1981)
trend_summary_1951_1980_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1950, 1981)
trend_summary_1951_1980_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1950, 1981)
trend_summary_1951_1980_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1950, 1981)

trend_summary_1981_2014_min_0p2 = calc_summary_trend_min(temp_min_obs, temp_min_0p2, 1980, 2015)
trend_summary_1981_2014_min_0p4 = calc_summary_trend_min(temp_min_obs, temp_min_0p4, 1980, 2015)
trend_summary_1981_2014_min_0p7 = calc_summary_trend_min(temp_min_obs, temp_min_0p7, 1980, 2015)
trend_summary_1981_2014_min_1p0 = calc_summary_trend_min(temp_min_obs, temp_min_1p0, 1980, 2015)
trend_summary_1981_2014_min_1p5 = calc_summary_trend_min(temp_min_obs, temp_min_1p5, 1980, 2015)

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

cf1 = axlist[0].contourf(trend_summary_1901_2014_min_0p2[1],lat,trend_summary_1901_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[0].set_title('0.2x ensemble member mean\n',fontsize=16)
cf2 = axlist[1].contourf(trend_summary_1901_2014_min_0p4[1],lat,trend_summary_1901_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[1].set_title('0.4x ensemble member mean\n',fontsize=16)
cf3 = axlist[2].contourf(trend_summary_1901_2014_min_0p7[1],lat,trend_summary_1901_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[2].set_title('\n(1901-2014)\n\n0.7x ensemble member mean\n',fontsize=16)
cf4 = axlist[3].contourf(trend_summary_1901_2014_min_1p0[1],lat,trend_summary_1901_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[3].set_title('1.0x ensemble member mean\n',fontsize=16)
cf5 = axlist[4].contourf(trend_summary_1901_2014_min_1p5[1],lat,trend_summary_1901_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[4].set_title('1.5x ensemble member mean\n',fontsize=16)

cf6 = axlist[5].contourf(trend_summary_1951_1980_min_0p2[1],lat,trend_summary_1951_1980_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf7 = axlist[6].contourf(trend_summary_1951_1980_min_0p4[1],lat,trend_summary_1951_1980_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf8 = axlist[7].contourf(trend_summary_1951_1980_min_0p7[1],lat,trend_summary_1951_1980_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[7].set_title('\n(1951-1980)\n',fontsize=16)
cf9 = axlist[8].contourf(trend_summary_1951_1980_min_1p0[1],lat,trend_summary_1951_1980_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf10 = axlist[9].contourf(trend_summary_1951_1980_min_1p5[1],lat,trend_summary_1951_1980_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

cf11 = axlist[10].contourf(trend_summary_1981_2014_min_0p2[1],lat,trend_summary_1981_2014_min_0p2[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf12 = axlist[11].contourf(trend_summary_1981_2014_min_0p4[1],lat,trend_summary_1981_2014_min_0p4[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf13 = axlist[12].contourf(trend_summary_1981_2014_min_0p7[1],lat,trend_summary_1981_2014_min_0p7[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
axlist[12].set_title('\n(1981-2014)\n',fontsize=16)
cf14 = axlist[13].contourf(trend_summary_1981_2014_min_1p0[1],lat,trend_summary_1981_2014_min_1p0[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")
cf15 = axlist[14].contourf(trend_summary_1981_2014_min_1p5[1],lat,trend_summary_1981_2014_min_1p5[0]*10,np.linspace(-2,2,41),transform=ccrs.PlateCarree(),cmap=plt.cm.RdBu_r,extend="both")

fig.suptitle('DJF Minimum Temperature Trend: Model and Observation difference', fontsize=30)

cbar = fig.colorbar(cf1,ax=axlist[11:14],orientation='horizontal', pad=0.2)
cbar.ax.set_xlabel('$^\circ$C/decade',fontsize=12)

for ax in axlist:
    plot_background(ax)
    
plt.savefig('/storage/silver/metstudent/msc/users_2022/ty824157/SMURPHS/DJF_model-obs_min.png', format='png', dpi=1200, bbox_inches='tight')
plt.show()