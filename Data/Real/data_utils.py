import os
import xarray as xr
import pandas as pd
import re
import gzip
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature

def lon_to_360(dlon: float) -> float:
  return ((360 + (dlon % 360)) % 360)

def load_dataset(data_dir, file_name = "combined_raw_2t_2023.nc"):
    #Code to change coordinates of the xarray dataset.
    #Run hours 00 and 12 are undistinguishable in the original dataset.
    #For each dataset, you should do:

    # Example: path to the NetCDF file of the Raw forecasts:
    data_path   = os.path.join(data_dir, file_name)

    # Open the NetCDF file
    ds = xr.open_dataset(data_path)

    #To change runtime coord
    # Extract the current runtime and validTime
    runtime = ds['runtime']
    validTime = ds['validTime']

    #The validtime at step0 is your run hour.
    validTime2= validTime.sel(step=0)
    step = ds['step']

    # Create a new runtime coordinate
    new_runtime = validTime2.copy()  
    new_runtime=new_runtime.values
    new_ds = ds.assign_coords(runtime=new_runtime)
    
    return new_ds

def extract_header(filename):
    with gzip.open(filename, "rt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                header = line 
            else:
                break  # Stop reading once data starts
    header_names = header.lstrip("#").strip().split()  # Remove "#" and extra spaces
    
    positions = [match.start() for match in re.finditer(r'\S+', header.lstrip("# "))]
    widths = [j - i for i, j in zip(positions, positions[1:])] + [len(header) - positions[-1]]
    widths[0] += 2
    
    return header_names, widths

def load_observations(data_dir, file_name = "kis_tot_202101.gz"):

    file_path = os.path.join(data_dir, file_name)
    col_names, col_widths = extract_header(file_path)

    # Create a DataFrame using `read_fwf`
    df_kis = pd.read_fwf(
        file_path,
        compression="gzip",     # File is compressed with gzip
        names=col_names,        # Use the extracted column names
        widths=col_widths,      # Specify fixed column widths
        comment="#",            # Skip the metadata and header rows
        na_values=[" "],        # Treat spaces as NaN
        parse_dates=["DTG"]     # Parse DTG as datetime
    )
    
    return df_kis

def data_array_to_mesh_and_values(z):
    # Extract latitude, longitude, and data values
    values = z.values.flatten()

    # Create meshgrid of latitude and longitude
    lon, lat = z.longitude.values, z.latitude.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten meshgrid
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    
    return lon_flat, lat_flat, values

def bounding_box_ds(ds, long_low, long_high, lat_low, lat_high):
    return ds.where(
        (ds.longitude > lon_to_360(long_low)) & (ds.latitude > lat_low) &
        (ds.longitude < lon_to_360(long_high)) & (ds.latitude < lat_high),
        drop=True
    )

def select_nl(ds):
    
    long_low, lat_low, long_high, lat_high = 3.31497114423, 50.803721015 - 0.25, 7.09205325687 + 0.25, 53.5104033474 # https://gist.github.com/graydon/11198540
    return bounding_box_ds(ds,
                           long_low=long_low,
                           long_high=long_high,
                           lat_low=lat_low,
                           lat_high=lat_high)
    
def get_cartopy_ax():
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent([lat_low, lat_high, lon_to_360(long_low), lon_to_360(long_high)])

    ax.gridlines()
    resol = '10m'  # use data at this scale
    bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
        name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    land = cartopy.feature.NaturalEarthFeature('physical', 'land', \
        scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
        scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', \
        scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
    rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
        scale=resol, edgecolor='b', facecolor='none')

    ax.add_feature(land, facecolor='beige')
    ax.add_feature(ocean, linewidth=0.2 )
    ax.add_feature(lakes)
    ax.add_feature(rivers, linewidth=0.5)
    ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

    return ax