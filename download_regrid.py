import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
from tqdm.autonotebook import tqdm
import rioxarray
import xesmf as xe
from xclim.sdba.adjustment import EmpiricalQuantileMapping, DetrendedQuantileMapping
import dask
import warnings
warnings.filterwarnings("ignore")


SCENARIO='ssp585' #historical, ssp119-4, ssp126-11, ssp434, ssp245-10, ssp460, ssp370-11, ssp585-12
MEM= 'r2i1p1f1'

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

df_ro =  df_ro[(df_ro.experiment_id.str.contains(SCENARIO)) & (df_ro.member_id.str.contains(MEM))]
print('Found %d available datasets!'%(len(df_ro)))

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
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(89.5, -90.5, -1)),
                     'lon': (['lon'], np.arange(-179.5, 180.5, 1))})
    regridder = xe.Regridder(array, ds_out, 'bilinear')
    # regridder.clean_weight_file()
    out= regridder(array) 
    return out


def download_data(df_ro, source_id, scenario, grid, mem):
    try:
        if scenario=='historical':
            ds= load_data(df_ro, source_id, scenario, grid).sel(time=slice('1950-01-01', '2015-01-01'))['mrro'].convert_calendar('standard', align_on="year",
                                                                    missing=0, use_cftime=True).to_dataset()
        else:
            ds= load_data(df_ro, source_id, scenario, grid)['mrro'].convert_calendar('standard', align_on="year",
                                                                    missing=0, use_cftime=True).to_dataset()
        ds= swap_western_hemisphere(ds)
        ds= regrid_to_1deg(ds)
        ds= (ds*86400).assign_attrs(units='mm/day')
        new_dir= '%s-%s-%s-%s'%(source_id, grid, mem, scenario.replace('historical', 'hist'))
        if not os.path.exists(new_dir):
            os.system('mkdir %s'%(new_dir))
        years, datasets= zip(*ds.groupby("time.year"))
        paths= ['%s/%s_%s_%s_1x1deg_%d.nc'%(new_dir, source_id.replace('-','_'), scenario.replace('historical', 'hist'), grid, year) for year in years]
        xr.save_mfdataset(datasets, paths, encoding = {"mrro": {'zlib': True, 'dtype':'float32', '_FillValue':-9999}})
    # for year in range(1950, 2015, 1):
    #     ds.sel(time=slice('%d-01-01'%year,'%d-12-31'%year)).to_netcdf('%s/%s_hist_%s_1x1deg_%d.nc'%(new_dir, source_id.replace('-','_'), grid, year),
    #                         encoding = {"mrro": {'zlib': True, 'dtype':'float32', '_FillValue':-9999}})
    except Exception:
        print('dataset %s-%s failed to export'%(source_id, grid))

    
for name, subdf in tqdm(df_ro.iterrows()):

    source_id= str(subdf.source_id)
    experiment_id= str(subdf.experiment_id)
    grid_label= str(subdf.grid_label)
    if not os.path.exists('%s-%s-%s-%s'%(source_id, grid_label, MEM, SCENARIO)):        
        download_data(df_ro, source_id, experiment_id, grid_label, MEM)
    else:
        print('%s-%s-%s-%s exists!'%(source_id, grid_label, MEM, SCENARIO))
