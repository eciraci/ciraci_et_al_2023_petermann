"""
Enrico Ciraci' 12/2021
Set of utility functions used to calculate Ice Elevation Changes and Ice Shelves
Basal Melt rate by using TanDEM-X DEMs.
"""
# - python dependencies
from __future__ import print_function
import os
import datetime
import numpy as np
from scipy import spatial
from scipy.interpolate import griddata
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from utility_functions_rio import load_dem_tiff


def create_dir(abs_path: str, dir_name: str) -> str:
    """
    Create directory
    :param abs_path: absolute path to the output directory
    :param dir_name: new directory name
    :return: absolute path to the new directory
    """
    dir_to_create = os.path.join(abs_path, dir_name)
    if not os.path.exists(dir_to_create):
        os.mkdir(dir_to_create)
    return dir_to_create


def do_kdtree(combined_x_y_arrays, points):
    """
    Use scipy kdtree function to locate the nearest grid cell to
    the considered points
    :param combined_x_y_arrays: 2-D array containing the flattened
           coordinates of the search domain
    :param points: list of points considered in the search
    :return:
    """
    kd_tree = spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = kd_tree.query(points)

    return indexes


def p_griddata(x_coords: np.ndarray, y_coords: np.ndarray, g_data: np.ndarray,
               x_grid: np.ndarray, y_grid: np.ndarray, method: str = "scipy",
               sc_gd_method: str = "nearest") -> np.ndarray:
    """
    Creates regular grid from the scattered data.
    Similar to Gdal_Grid
    :param x_coords: 1-D array containing points x-coordinates
    :param y_coords: 1-D array containing points y-coordinates
    :param g_data: 1-D array containing points data values
    :param x_grid: 2-D array containing output grid x-coordinates
    :param y_grid: 2-D array containing output grid y-coordinates
    :param method: Gridding Approach [scipy/kdtree]
    :param sc_gd_method:scipy.griddata interpolation method
    :return:
    """
    s_points = np.zeros([len(x_coords), 2])
    s_points[:, 0] = x_coords
    s_points[:, 1] = y_coords

    if method == "scipy":
        gridded_data = griddata(s_points, g_data.ravel(),
                                (x_grid, y_grid), method=sc_gd_method)
        return gridded_data
    elif method == "kdtree":
        # - Create Output Data Grid.
        gridded_data = np.full(x_grid.shape, np.nan)
        # - Crate Indices matrix
        ref_grid_index = np.indices(x_grid.shape)
        x_index = ref_grid_index[0]
        y_index = ref_grid_index[1]
        x_index_flat = x_index.ravel()
        y_index_flat = y_index.ravel()
        g_data_flat = g_data.ravel()
        # - For each of the considered particles,
        # - find the closest grid cell on the velocity domain.
        combined_x_y_arrays = np.dstack([x_grid.ravel(), y_grid.ravel()])[0]
        indexes = do_kdtree(combined_x_y_arrays, s_points)
        # -
        x_index_selected = x_index_flat[indexes]
        y_index_selected = y_index_flat[indexes]
        # -
        ind_shape = x_index.shape
        for i in range(ind_shape[0]):
            for j in range(ind_shape[1]):
                f_inf = np.where((x_index_selected == i)
                                 & (y_index_selected == j))[0]
                if len(f_inf):
                    gridded_data[i, j] = np.median(g_data_flat[f_inf])

        return gridded_data

    else:
        raise ValueError("Unknown method selected.")


def ibe_from_era5(data_dir, year, month, day, hour, lat, lon,
                  median=True, m_r_year=1992):
    """
    Calculate Inverse Barometer Effect (IBE) correction from ERA5
    :param data_dir: absolute path to project data directory
    :param year: date year
    :param month: date month
    :param day: date day
    :param hour: date hour
    :param lat: sample point latitude
    :param lon: sample point longitude
    :param median: if True, use Long-term median of Meas Sea Level Pressure;
                   if False, use Standard Atmospheric Pressure;
    :param m_r_year: median calculation reference year.
    :return: IBE
    """
    # - Calculate Inverse Barometer Effect  [m]
    rho_sea = 1028          # - seawater density  kg/m3
    gravity_acc = 9.8       # - gravity acceleration in m/sec^2 [N/kg]
    std_atm = 101325        # - standard atmosphere in Pa

    # - path to ERA5 data
    era5_path = os.path.join(data_dir, "Reanalysis", "ERA5",
                             "reanalysis-era5-single-levels",
                             "mean_sea_level_pressure_utc_petermann"
                             "_2010_2021.nc")
    # - load hourly mean sea level pressure data
    d_input = xr.open_dataset(era5_path)
    # - mean sea level pressure in Pascal
    msl_input = np.nanmean(np.array(d_input["msl"].values), axis=1)
    lat_ax = np.array(d_input["latitude"].values)
    lon_ax = np.array(d_input["longitude"].values)
    time = np.array(d_input["time"].values)  # - time axis
    time_ax = pd.to_datetime(list(time))

    ds = xr.Dataset(data_vars=dict(msl=(["time", "lat", "lon"], msl_input)),
                    coords=dict(time=(["time"], time_ax),
                                lat=(["lat"], lat_ax),
                                lon=(["lon"], lon_ax))
                    )
    # - - extract MSL at the selected UTC time
    msl_point_s = ds.where(((ds["time.year"] == year)
                            & (ds["time.month"] == month)
                            & (ds["time.day"] == day)
                            & (ds["time.hour"] == hour)),
                           drop=True)["msl"].values
    msl_sample = np.squeeze(msl_point_s)

    # - find sample point grid coordinates
    dist_lat = np.abs(lat_ax - lat)
    ind_lat = np.where(dist_lat == np.min(dist_lat))
    dist_lon = np.abs(lon_ax - lon)
    ind_lon = np.where(dist_lon == np.min(dist_lon))
    print(f"# - Closest point to the selected coordinates "
          f"within the input data domain -> "
          f"Lat: {lat_ax[ind_lat][0]}, Lon: {lon_ax[ind_lon][0]}")

    # - Extract MSL at the selected location for the considered date
    msl_pt = np.squeeze(msl_sample[ind_lat, ind_lon])
    print("# - Sea Level Pressure:")
    print(msl_pt)
    # - IBE in meters
    if median:
        # - Long-term median of Meas Sea Level Pressure at the selected
        # - location.
        msl_ts = np.squeeze(msl_input[:, ind_lat, ind_lon])
        ind_median = np.where(ds["time.year"].values >= m_r_year)[0]
        mls_lt_median = np.median(msl_ts[ind_median])
        ibe = (msl_pt - mls_lt_median) / (rho_sea * gravity_acc)
        print("# - Using Long-Term MSLP to evaluate pressure anomaly.")
        print(f"# - mls_lt_median = {mls_lt_median}")
        print(f"# - DeltaP: {msl_pt - mls_lt_median}")
    else:
        # - standard Atmosphere.
        print("# - Using Standard Atmosphere to evaluate pressure anomaly.")
        print(f"# - std_atm = {std_atm}")
        print(f"# - DeltaP: {msl_pt - std_atm}")
        ibe = (msl_pt - std_atm) / (rho_sea * gravity_acc)

    return ibe


def add_colorbar(fig, ax: plt.Axes,
                 im: plt.pcolormesh) -> plt.colorbar:
    """
    Add colorbar to the selected plt.Axes.
    :param fig: plt.figure object
    :param ax: plt.Axes object.
    :param im: plt.pcolormesh object.
    :return: plt.colorbar
    """
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.6, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    return cb


def calculate_n_month(start_date: datetime.datetime,
                      end_date: datetime.datetime,
                      reference_day=15) -> dict:
    """
    Calculate number of months separating two acquisitions
    :param start_date: first date of the considered period of time
    :param end_date: last date of the considered period of time
    :param reference_day: monthly reference day
    :return: dict()
    """
    # - Calculate number of months separating the two acquisitions
    n_months = (((end_date.year - start_date.year) * 12)
                + (end_date.month - start_date.month))
    # - Compute Delta Time
    delta_time = end_date - start_date

    # - Generate sequence of monthly dates
    dates_list = [datetime.datetime(year=start_date.year,
                                    month=start_date.month,
                                    day=reference_day)]
    for _ in range(1, n_months):
        month = dates_list[-1].month
        year = dates_list[-1].year
        if month != 12:
            dates_list.append(datetime.datetime(year=year, month=month+1,
                                                day=reference_day))
        else:
            dates_list.append(datetime.datetime(year=year+1, month=1,
                                                day=reference_day))

    return {"n_months": n_months, "dates_list": dates_list,
            "delta_time": delta_time}


def divergence(x_coords: np.ndarray, y_coords: np.ndarray,
               v_x_field: np.ndarray, v_y_field: np.ndarray) -> dict:
    """
    Evaluate Divergence of the provided bi-dimensional field
    :param x_coords: X direction coordinate axis
    :param y_coords: Y direction coordinate axis
    :param v_x_field: Vector Field - X
    :param v_y_field: Vector Field - Y
    :param v_y_field: Vector Field - Y
    :return: Divergence Field
    """
    # - Evaluate if the X and Y coordinates are increasing or
    # - decreasing along the raster/figure borders.
    slope_x = 1
    if x_coords[-1] < x_coords[0]:
        slope_x = -1
    slope_y = 1
    if y_coords[-1] < y_coords[0]:
        slope_y = -1
    # - Compute raster resolution
    x_res = np.abs(x_coords[1] - x_coords[0])
    y_res = np.abs(y_coords[1] - y_coords[0])
    # - compute velocity partial derivatives
    dvx_dx = np.gradient(v_x_field, 1, axis=1) * slope_x / x_res
    dvy_dy = np.gradient(v_y_field, 1, axis=0) * slope_y / y_res
    # - compute velocity divergence
    div_v = dvx_dx + dvy_dy

    return {"dvx_dx": dvx_dx, "dvy_dy": dvy_dy, "div_v": div_v}


def f_gradient(x_coords: np.ndarray, y_coords: np.ndarray,
               bd_field: np.ndarray) -> dict:
    """
    Evaluate First-Order Gradient of the provided bi-dimensional field
    :param x_coords: X direction coordinate axis
    :param y_coords: Y direction coordinate axis
    :param bd_field: bi-dimensional field
    :return: Gradient X and Y components
    """
    # - Evaluate if the X and Y coordinates are increasing or
    # - decreasing along the raster/figure borders.
    slope_x = 1
    if x_coords[-1] < x_coords[0]:
        slope_x = -1
    slope_y = 1
    if y_coords[-1] < y_coords[0]:
        slope_y = -1
    # - Compute raster resolution
    x_res = np.abs(x_coords[1] - x_coords[0])
    y_res = np.abs(y_coords[1] - y_coords[0])
    # - compute velocity partial derivatives
    dbd_dx = np.gradient(bd_field, 1, axis=1) * slope_x / x_res
    dbd_dy = np.gradient(bd_field, 1, axis=0) * slope_y / y_res

    return{"dbd_dx": dbd_dx, "dbd_dy": dbd_dy}


def f_gradient_baseline(x_coords: np.ndarray, y_coords: np.ndarray,
                        bd_field: np.ndarray,
                        baseline_x: float, baseline_y: float) -> dict:
    """
    Evaluate First-Order Gradient of the provided bi-dimensional field
    :param x_coords: X direction coordinate axis
    :param y_coords: Y direction coordinate axis
    :param bd_field: bi-dimensional field
    :param baseline_x: derivative calculation baseline X axis [same unit of X]
    :param baseline_y: derivative calculation baseline Y axis [same unit of Y]
    :return: Gradient X and Y components
    """
    # - Compute Pixel Spacing
    res_x = np.abs(x_coords[1] - x_coords[0])
    res_y = np.abs(y_coords[1] - y_coords[0])
    # - Compute Derivative Step size
    step_size_x = int(baseline_x / res_x)
    step_size_y = int(baseline_y / res_y)

    # - Step Size must, be an odd number.
    if step_size_x % 2 == 0:
        step_size_x += 1
    if step_size_y % 2 == 0:
        step_size_y += 1

    # - From kernel size to central difference step.
    d_hx = int(np.floor(step_size_x / 2))
    d_hy = int(np.floor(step_size_y / 2))

    # - Consider if step_size_x, step_size_y == 1 d_hx, d_hy = 1
    if d_hx == 0:
        d_hx += 1
    if d_hy == 0:
        d_hy += 1
    # - Evaluate if the X and Y coordinates are increasing or
    # - decreasing along the raster borders.
    slope_x = 1
    if x_coords[-1] < x_coords[0]:
        slope_x = -1

    slope_y = 1
    if y_coords[-1] < y_coords[0]:
        slope_y = -1

    # - Alternative Gradient Calculation Approach
    dbd_dx = np.zeros(bd_field.shape)
    dbd_dy = np.zeros(bd_field.shape)

    # - x direction
    for c in range(d_hx, len(x_coords) - d_hx):
        dbd_dx[:, c] \
            = (bd_field[:, c + d_hx] - bd_field[:, c - d_hx]) / (2 * d_hx)
    # - evaluate gradient at the domain edges
    for cc in range(d_hx):
        dbd_dx[:, cc] = (bd_field[:, cc + 1] - bd_field[:, 0]) / (cc + 1)
        dbd_dx[:, -1 - cc] = (bd_field[:, -2 - cc] - bd_field[:, -1]) / (cc + 1)

    # - y direction
    for r in range(d_hy, len(y_coords) - d_hy):
        dbd_dy[r, :] = (bd_field[r + d_hy, :] - bd_field[r - d_hy, :]) / (
                    2 * d_hy)

    for rr in range(d_hy):
        dbd_dy[rr, :] = (bd_field[rr + 1, :] - bd_field[0, :]) / (rr + 1)
        dbd_dy[-1 - rr, :] = (bd_field[-2 - rr, :] - bd_field[-1, :]) / (rr + 1)

    # - divide the Velocity Gradient Field to pass from the pixel domain
    # - to the spatial domain.
    dbd_dx /= res_x
    dbd_dy /= res_y
    # - Include Axis direction in the calculation
    dbd_dx *= slope_x
    dbd_dy *= slope_y

    return{"dbd_dx": dbd_dx, "dbd_dy": dbd_dy}
