import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from tqdm.autonotebook import tqdm
import rioxarray
# import hydroeval as he
import xesmf as xe
from xclim.sdba.adjustment import EmpiricalQuantileMapping, DetrendedQuantileMapping
import dask
import warnings
warnings.filterwarnings("ignore")


# get CMIP6 model table
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

df_ro = df[(df.table_id == 'day') & (df.variable_id == 'mrro')]

def load_data(df, source_id, expt_id, grid_label):
    """
    Load 3hr runoff data for given source and expt ids
    """
    uri = df[(df.source_id == source_id) &
                         (df.experiment_id == expt_id) & (df.grid_label==grid_label)].zstore.values[0]

    ds = xr.open_zarr(fsspec.get_mapper(uri), consolidated=True).convert_calendar('standard', missing=0)
    return ds

df_ro =  df_ro[(df_ro.experiment_id.str.contains('historical')) & (df_ro.member_id.str.contains('r1i1p1f1'))]

def swap_western_hemisphere(array):
    """Set longitude values in range -180, 180.
    Western hemisphere longitudes should be negative.
    """

    # Set longitude values in range -180, 180.
    array['lon'] = (array['lon'] + 180) % 360 - 180

    # Re-index data along longitude values
    west = array.where(array.lon < 0, drop=True)
    east = array.where(array.lon >= 0, drop=True)
    return west.combine_first(east)

def regrid_to_1deg(array):
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(89.5, -89.5, -1)),
                     'lon': (['lon'], np.arange(-179.5, 179.5, 1))})
    regridder = xe.Regridder(array, ds_out, 'bilinear')
    # regridder.clean_weight_file()
    out= regridder(array) 
    return out


def download_data(df_ro, source_id, scenario, grid):
    ds= load_data(df_ro, source_id, 'historical', grid).sel(time=slice('1950-01-01', '2015-01-01'))['mrro'].convert_calendar('standard',
                                                            missing=0, use_cftime=True).to_dataset()
    ds= swap_western_hemisphere(ds)
    ds= regrid_to_1deg(ds)
    (ds*86400).assign_attrs(units='mm/day').compute().to_netcdf('%s_hist_%s_1x1deg.nc'%(source_id.replace('-','_'), grid),
                        encoding = {"mrro": {'zlib': True, 'dtype':'float32', '_FillValue':-9999}})
    
for name, subdf in tqdm(df_ro.iterrows()):
    source_id= str(subdf.source_id)
    experiment_id= str(subdf.experiment_id)
    grid_label= str(subdf.grid_label)
    download_data(df_ro, source_id, experiment_id, grid_label)
