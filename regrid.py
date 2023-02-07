import os
import re
from glob import glob
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

scenario='ssp460'
folders= os.listdir('ssp460')
for name in folders:
	print('processing %s'%name)
	source_id= name.split('-g')[0]
	grid_label=re.search(r'g\w\d?', name).group(0)
	mem= re.search(r'r\di1p1f1', name).group(0)
	files= glob(os.path.join(scenario, name,'*.nc'))
	ds= xr.open_mfdataset(files, parallel=True)
	ds= swap_western_hemisphere(ds)
	ds= regrid_to_1deg(ds)
	ds= (ds*86400).assign_attrs(units='mm/day')
	years, datasets= zip(*ds.groupby("time.year"))
	paths= ['%s/%s/%s_%s_%s_1x1deg_%d.nc'%(scenario,name,source_id.replace('-','_'), scenario, grid_label, year) for year in years]
	xr.save_mfdataset(datasets, paths, encoding = {"mrro": {'zlib': True, 'dtype':'float32', '_FillValue':-9999}})
