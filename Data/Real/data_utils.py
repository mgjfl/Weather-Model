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
from scipy.interpolate import griddata

stations_covering = [
    260, # De Bilt 
    280, # KNMI Weerstation Eelde (Groningen)
    344, # Rotterdam
    370, # Eindhoven airport
    249, # Berkhout (Noord-Holland)
]

stations_kiri = [
    235, # De Kooy
    240, # Schiphol
    260, # De Bilt
    280, # Eelde
    290, # Twenthe
    310, # Vlissingen
    380  # Maastricht
    ]

stations_kiri_subset = [235, 310, 260] # 2 coast and one in-land locations

DATA_DIR = os.path.join(os.getcwd(), "..")
IFS_DIR = os.path.join(os.getcwd(), "IFS", "01-20-2025")
FIG_DIR = os.path.join(os.getcwd(), "..", "Figures")

class BoundingBox:
    def __init__(self, long_low, lat_low, long_high, lat_high):
        self.long_low   = long_low
        self.lat_low    = lat_low
        self.long_high  = long_high
        self.lat_high   = lat_high

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

def adjust_bbox_to_grid(bbox):
    import numpy as np
    def floor_to_grid(x): return np.floor(x * 4) / 4  # 0.25° grid
    def ceil_to_grid(x): return np.ceil(x * 4) / 4

    return BoundingBox(
        long_low=floor_to_grid(bbox.long_low),
        long_high=ceil_to_grid(bbox.long_high),
        lat_low=floor_to_grid(bbox.lat_low),
        lat_high=ceil_to_grid(bbox.lat_high),
    )

def bounding_box_ds(ds, bbox : BoundingBox, fromCDS = False):
    
    bbox = adjust_bbox_to_grid(bbox)
    
    if not fromCDS:
        return ds.sel(
            latitude=slice(bbox.lat_high, bbox.lat_low)
        ).where(
            (ds.longitude > lon_to_360(bbox.long_low)) &
            (ds.longitude < lon_to_360(bbox.long_high))
        )
    else:
        return ds.sel(
            latitude=slice(bbox.lat_high, bbox.lat_low)
        ).where(
            (ds.longitude > bbox.long_low) &
            (ds.longitude < bbox.long_high),
            drop = True
        )
    
def bounding_box_nl():
    # Returns long_low, lat_low, long_high, lat_high
    return BoundingBox(3.31497114423, 50.803721015 - 0.1, 7.09205325687 + 0.1, 53.5104033474) # https://gist.github.com/graydon/11198540 

def select_nl(ds):
    
    return bounding_box_ds(ds, bounding_box_nl())
    
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

def construct_obs_combined(stations, savename):
    obs_files = [x for x in os.listdir(IFS_DIR) if "kis_tot" in x]
    df_list = []

    for obs_file in obs_files:
        
        #########################
        # Step 1. Load the data #
        #########################
        obs = load_observations(IFS_DIR, obs_file) 

        ###############################################################
        # Step 2. Select the proper locations and columns of interest #
        ###############################################################
        obs_selected = obs[[int(x.split("_")[0]) in stations for x in obs["LOCATION"]]][["LOCATION", "DTG", 'T_DEWP_10', "T_DRYB_10"]]
        
        ##################################
        # Step 3. Select the 12pm values #
        ##################################

        # Extend search for missing values to 09:00:00 - 15:00:00
        expanded_search = obs_selected[
            (obs_selected["DTG"].dt.time >= pd.to_datetime("09:00:00").time()) & 
            (obs_selected["DTG"].dt.time <= pd.to_datetime("15:00:00").time())
        ].copy()

        # Compute absolute time difference from 12:00:00 on the same day
        expanded_search["time_diff"] = (
            (expanded_search["DTG"] - expanded_search["DTG"].dt.normalize()) - pd.Timedelta(hours=12)
        ).abs()
        
        has_nan = expanded_search.isna().any(axis = 1)
        expanded_search.loc[has_nan, "time_diff"] = pd.Timedelta(hours=23)

        # Select the closest available time per day
        obs_closest = expanded_search.loc[expanded_search.groupby(["LOCATION", expanded_search["DTG"].dt.date])["time_diff"].idxmin()]
        
        ##########################
        # Step 4. Store the data #
        ##########################
        df_list.append(obs_closest)

    # Construct the combined df
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # Delete station if contains nan
    has_nan = df_combined[["LOCATION", "T_DEWP_10", "T_DRYB_10"]].groupby('LOCATION').apply(lambda g: g.isna().any(), include_groups=False).any(axis = 1)
    df_combined = df_combined[df_combined["LOCATION"].apply(lambda x : not has_nan[x])]

    # Save the df in pickle for python and csv for R
    df_combined.to_pickle(os.path.join(IFS_DIR, savename + ".pk"))
    df_combined.to_csv(os.path.join(IFS_DIR, savename + ".csv"))
    
    return df_combined, df_list


def construct_ifs_combined(weather_var, gp_df):
    # weather_var is 'd' for dew point and 't' for air temperature  

    ifs_files = [x for x in os.listdir(IFS_DIR) if f"combined_raw_2{weather_var}" in x]
    df_list = []

    for ifs_file in ifs_files:
        
        print(f"Processing {ifs_file}")
        
        # Load dataset (xarray)
        ds_2x = load_dataset(IFS_DIR, ifs_file)
        
        # Select 12pm values with 24h forecast and transform to dataframe
        df_2x = (
            ds_2x
            .where(ds_2x.runtime.dt.hour == 12, drop = True) # Only select 12pm values
            .sel(step = 4) # Select 24 hours forecast time
            .to_dataframe()
            )
        
        # Store the data
        df_list.append(df_2x)
        
    # Construct the combined df
    df_2x_combined = pd.concat(df_list)
        
    # Add station locations
    df_2x_combined = (
        df_2x_combined
        .reset_index()
        .merge(gp_df[["gridpoint", "gp_lat", "gp_lon"]], on="gridpoint", how="left")
        .set_index(df_2x_combined.index.names)
        )

    # Sort index for better performance
    df_2x_combined = df_2x_combined.sort_index()
    
    return df_2x_combined

def construct_ifs_station_df(df_2x, stations, sta_df, savename):
    
    if "x2d" in df_2x.columns:
        colname = "x2d"
    elif "x2t" in df_2x.columns:
        colname = "x2t"
    else:
        raise NotImplementedError()
    
    # Get station locations only once
    sel_stations_df = sta_df.set_index("sta_id").loc[stations]
    xy_data_sta = np.column_stack((sel_stations_df["gp_lon"], sel_stations_df["gp_lat"]))

    # Function to apply interpolation to each group
    def interpolate_group(df_test):
        xy_data = np.column_stack((df_test["gp_lon"], df_test["gp_lat"]))
        z_data = df_test[colname].values
        interpolated_values = griddata(xy_data, z_data, xy_data_sta, method="cubic")

        # Create a DataFrame with stations as index
        return pd.DataFrame({colname: interpolated_values}, 
                            index=pd.Index(stations, name = "station"))

    # Apply interpolation to each (runtime, member) group and reconstruct index
    df_out = (
        df_2x.groupby(["runtime", "member"])[[colname, "gp_lat", "gp_lon"]]
        .apply(interpolate_group)
    )

    # Extract unique validTime for each runtime from df_2d
    valid_time_map = df_2x['validTime'].groupby('runtime').first()

    # Map validTime to df_out based on the runtime index
    df_out['validTime'] = df_out.index.get_level_values('runtime').map(valid_time_map)
    
    # Save the df in pickle for python and csv for R
    df_out.to_pickle(os.path.join(IFS_DIR, f"ifs_{colname}_{savename}.pk"))
    df_out.to_csv(os.path.join(IFS_DIR, f"ifs_{colname}_{savename}.csv"))
    
    return df_out

def pivot_ifs(df_2x):
    
    if "x2d" in df_2x.columns:
        colname = "x2d"
    elif "x2t" in df_2x.columns:
        colname = "x2t"
    else:
        raise NotImplementedError()
    
    
    # Pivot the 'member' index to columns
    df_unstacked = df_2x[[colname]].unstack(level='member')

    # Reset index to make 'runtime' and 'station' columns
    df_unstacked = df_unstacked.reset_index()

    # Rename columns
    ifs_columns = [f'IFS_{colname}_{col[1]}' for col in df_unstacked.columns[2:]]
    df_unstacked.columns = ['runtime', 'station'] + ifs_columns

    # Set the index
    df_unstacked = df_unstacked.set_index(["runtime", "station"])

    # Extract unique validTime for each runtime from df_2x
    valid_time_map = df_2x['validTime'].groupby('runtime').first()

    # Map validTime to df_out based on the runtime index
    df_unstacked['validTime'] = df_unstacked.index.get_level_values('runtime').map(valid_time_map)

    # Reorder the columns
    df_unstacked = df_unstacked.reset_index()
    df_unstacked = df_unstacked[['validTime', 'station', 'runtime'] + ifs_columns]

    return df_unstacked

def check_missing_days(datetime_list):
    """Check if there are missing days between the min and max date in a list of datetime values."""
    dates = pd.to_datetime(datetime_list).normalize()  # Ensure all times are at midnight
    all_days = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
    missing_days = all_days.difference(dates)
    print(f"{missing_days.shape[0]} days out of {all_days.shape[0]} were missing.")
    return missing_days

def get_avg_dist(ds, direction, include_pressure = False):
    # Surface
    avg_speed = np.abs(ds[f"{direction}10"]).mean().item()

    if include_pressure:
        # Pressure
        levels = [50, 500, 850, 1000]

        for level in levels:
            val             = np.abs(ds[f"{direction}_{level}"]).mean().item()
            avg_speed   += val
        
        avg_speed /= len(levels) + 1

        
    avg_dist = avg_speed * 60 * 60 * 24 / 1000 
    return avg_dist

def ds_diff(ds1, ds2):
    lats  = ds1.latitude
    longs = ds1.longitude

    lat_mask  = ~ds2.latitude.isin(lats)
    long_mask = ~ds2.longitude.isin(longs)

    ds_diff   = ds2.sel(latitude = ds2.latitude[lat_mask],
                        longitude = ds2.longitude[long_mask])
    
    return ds_diff

def custom_interpolate(lon_flat, lat_flat, values, resolution = 100, method = "cubic"):
    """Interpolates values on a grid lon_flat x lat_flat to a finer grid.

    Method is one of:
    - nearest
    - linear
    - cubic
    """

    grid_x, grid_y = np.meshgrid(np.linspace(lon_flat.min(), lon_flat.max(), resolution),
                                np.linspace(lat_flat.min(), lat_flat.max(), resolution), indexing='ij')
    
    interpolated_values = griddata(list(zip(lon_flat, lat_flat)), values, (grid_x, grid_y), method = method)
    
    return ((grid_x, grid_y), interpolated_values)