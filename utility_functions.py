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
from velocity_map_time_interpolation import load_velocity_map, \
    load_velocity_map_nearest


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


def grounding_line_path(data_dir: str, ref_year: int = 2020,
                        sensor: str = "CSK") -> dict:
    """
    Return Path to Petermann Glacier Estimated Grounding Line Position (GL)
    based on the selected reference year. Note: GL estimates are not available
    at a yearly basis
    :param data_dir: absolute path to project data directory
    :param ref_year: reference year
    :param ref_year: reference year
    :param sensor: selected sensor [ERS, CSK, Sentinel]
    :return: dictionary containing Geopandas Dataframe containing the selected
           grounding line.
    """
    if sensor in ["ERS", "ers", "Ers"]:
        if ref_year <= 2012:
            # - ERS 2011
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_GL_Shivani",
                                      "Ddiff_ERS2_2011",
                                      "coco83763-83720-84279-84236.shp")
            gnd_label = "ERS-2011"
        else:
            print(" # - ERS Data Not available for the selected period.")
            gnd_ln_shp = ""
            gnd_label = ""

    elif sensor in ["CSK", "csk", "Csk"]:
        if ref_year == 2013:
            # - CSK - 2013
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_GL_Shivani",
                                      "Ddiff_CSK_2013",
                                      "coco131124-131125-131125-131128.shp")
            gnd_label = "CSK-11/2013"
        elif ref_year == 2020:
            # - Petermann Grounding Line - 2020/2021
            gnd_ln_shp = os.path.join(data_dir,
                                      "coco_petermann_grnd_lines_2020-2021",
                                      "grnd_lines_shp_to_share",
                                      "coco20200501_20200502-20200517_20200518",
                                      "coco20200501_20200502-20200517_20200518"
                                      "_grnd_line.shp")
            gnd_label = "CSK-05/2020"
        elif ref_year == 2021:
            # - Petermann Grounding Line - 2020/2021
            gnd_ln_shp = os.path.join(data_dir,
                                      "coco_petermann_grnd_lines_2020-2021",
                                      "grnd_lines_shp_to_share",
                                      "coco20210213_20210214-20210317_20210318",
                                      "coco20210213_20210214-20210317_20210318"
                                      "_grnd_line.shp")
            gnd_label = "CSK-02/2021"
        else:
            print(" # - CSK Data Not available for the selected period.")
            gnd_ln_shp = ""
            gnd_label = ""

    elif sensor in ["Sentinel", "Sentinel", "SENTINEL"]:
        if ref_year == 2016:
            # - Sentinel 2016
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_Merged_Millan", "single_gl",
                                      "161211.shp")
            gnd_label = "Sentinel-12/2016"
        elif ref_year == 2017:
            # - Sentinel 2017
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_Merged_Millan", "single_gl",
                                      "171230.shp")
            gnd_label = "Sentinel-12/2017"
        elif ref_year == 2018:
            # - Sentinel 2018
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_Merged_Millan", "single_gl",
                                      "180423.shp")
            gnd_label = "Sentinel-04/2018"
        elif ref_year == 2019:
            # - Sentinel 2019
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_Merged_Millan", "single_gl",
                                      "190414.shp")
            gnd_label = "Sentinel-04/2019"
        elif ref_year == 2020:
            # - Sentinel 2020
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_Merged_Millan", "single_gl",
                                      "201110.shp")
            gnd_label = "Sentinel-11/2020"

        elif ref_year == 2021:
            # - Sentinel 2021
            gnd_ln_shp = os.path.join(data_dir, "GIS_Data",
                                      "Petermann_Merged_Millan", "single_gl",
                                      "210222.shp")
            gnd_label = "Sentinel-02/2021"
        else:
            print(" # - Sentinel Data Not available for the selected period.")
            gnd_ln_shp = ""
            gnd_label = ""
    else:
        # - Data Not Found
        print(" # - Sentinel Data Not available for the selected period.")
        gnd_ln_shp = ""
        gnd_label = ""

    # - Petermann Grounding Zone - 2011/2021
    gnd_zn_shp = os.path.join(data_dir, "GIS_Data",
                              "Petermann_features_extraction",
                              "Petermann_grounding_line_migration_"
                              "range_epsg3413.shp")

    return {"gnd_ln_shp": gnd_ln_shp, "gnd_zn_shp": gnd_zn_shp,
            "gnd_label": gnd_label}


def load_grounding_line_mask(data_dir: str, ref_crs: int = 3413,
                             ref_year: int = 2020, sensor: str = "CSK",
                             grnd_zn_buffer: int = 1500,
                             sampling_prof: str
                             = "longitudinal_profile_hr") -> dict:
    """
    Load Grounding Line Mask
    :param data_dir: absolute path to data directory
    :param ref_crs: Reference CRS
    :param ref_year: Reference Year [Grounding Line]
    :param sensor: Selected Sensor [CSK, ERS, Sentinel]
    :param grnd_zn_buffer: Grounding Zone Buffer in meters
    :param sampling_prof: Longitudinal Sampling profile file name
    :return Dictionary containing grounding line mask + info
    """
    # - Petermann Grounding Line
    gnd_path = grounding_line_path(data_dir, ref_year=ref_year, sensor=sensor)
    gnd_ln_shp = gnd_path["gnd_ln_shp"]
    gnd_ln_label = gnd_path["gnd_label"]

    # - Petermann Grounding Zone - Migration 2011/2021
    gnd_zn_shp = gnd_path["gnd_zn_shp"]

    # - Import Grounding Line infor as GeodataFrame
    gnd_ln_df = gpd.read_file(gnd_ln_shp).to_crs(epsg=ref_crs)

    # - features extraction shapefiles path
    feat_ex_path = os.path.join(data_dir, "GIS_Data",
                                "Petermann_features_extraction")
    # - Longitudinal Profile
    long_prof_path = os.path.join(feat_ex_path, sampling_prof + ".shp")
    long_prof_df = gpd.read_file(long_prof_path).to_crs(epsg=ref_crs)

    if grnd_zn_buffer:
        # - Clip Grounding Zone Mask
        clip_shp_mask_path \
            = os.path.join(data_dir, "GIS_Data",
                           "Petermann_features_extraction",
                           "Petermann_iceshelf_clipped_epsg3413.shp")
        clip_mask = gpd.read_file(clip_shp_mask_path).to_crs(epsg=ref_crs)

        gnd_zn_shp_buff = os.path.join(data_dir, "GIS_Data",
                                       "Petermann_features_extraction",
                                       "Petermann_grounding_line_migration_"
                                       f"range_buff{grnd_zn_buffer}"
                                       f"_epsg3413.shp")
        if not os.path.isfile(gnd_zn_shp_buff):
            gnd_zn_to_bf = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)
            gnd_zn_to_bf["geometry"] = gnd_zn_to_bf.geometry \
                .buffer(grnd_zn_buffer)
            # - clip the obtained buffered mask with the
            # - ice shelf perimeter mask.
            gnd_zn_to_bf = gpd.overlay(gnd_zn_to_bf, clip_mask,
                                       how="intersection")
            # - save buffered mask to file
            gnd_zn_to_bf.to_file(gnd_zn_shp_buff)

        # -
        gnd_zn_shp = gnd_zn_shp_buff

    # - Compute the point coordinates of the intersection between the
    # - longitudinal sampling profile and the grounding zone.
    gdz_df = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)

    # - Find intersection between Grounding Line and Longitudinal profile
    gnd_ln_inter = long_prof_df.intersection(gnd_ln_df)
    xsp_ln, ysp_ln = gnd_ln_inter.geometry[0].coords.xy

    # - Find intersection between Grounding Zone mask and Longitudinal profile
    gdz_df_inter = long_prof_df.intersection(gdz_df)
    xsp_gz, ysp_gz = gdz_df_inter.geometry[0].coords.xy

    return {"xsp_ln": xsp_ln, "ysp_ln": ysp_ln,
            "ysp_gz": ysp_gz, "xsp_gz": xsp_gz,
            "gnd_ln_label": gnd_ln_label}


def move_gpt_along_flow(dt_1: datetime.datetime, dt_2: datetime.datetime,
                        data_dir: str, verbose: bool = False) -> dict:
    """
    Compute the travel history of an ice particle selected at the intersection
    between the Petermann's longitudinal sampling profile and the grounding
    line. The grounding line used in the calculation is selected based on the
    first reference data.
    parm dt_1: initial reference date
    parm dt_2: final reference date
    parm data_dir: absolute path to project data directory
    parm verbose: print output intermediate results on standard output
    """
    # - Compute temporal distance between the considered dates
    delta_time = calculate_n_month(dt_1, dt_2)
    n_months = delta_time["n_months"]
    dates_list = delta_time["dates_list"]

    # - Parameters
    ref_crs = 3413

    # - Velocity Domain
    v_domain = "Petermann_Domain_Velocity_Stereo_tdx"
    v_smooth = False        # - smooth velocity maps
    v_smooth_mode = "ave"   # - velocity smoothing filter (average/median)
    v_smooth_size = 11      # - velocity filter smoothing filter size
    # - Velocity Interpolation Method interpolation - Time domain
    v_t_interp = "bilinear"

    # - features extraction shapefiles path
    feat_ex_path = os.path.join(data_dir, "GIS_Data",
                                "Petermann_features_extraction")
    # - Longitudinal Profile
    long_prof_path = os.path.join(feat_ex_path, "longitudinal_profile_hr.shp")
    long_prof_df = gpd.read_file(long_prof_path).to_crs(epsg=ref_crs)

    # - Import Grounding Line Dataset
    gl_millan_path = os.path.join(data_dir, "GIS_Data",
                                  "Petermann_Merged_Millan",
                                  "gl_merged_petermann_allyear_ps_"
                                  "intersect_long_prof.shp")
    gl_millan_df = gpd.read_file(gl_millan_path)[["Date1", "geometry"]]
    # - Remove Lines with not valid geometry
    gl_millan_df = gl_millan_df.mask(gl_millan_df.eq(None)).dropna()
    gl_millan_df = gl_millan_df.mask(gl_millan_df.eq(None)).dropna()
    # - Add date time axis to geo-dataframe
    time_ax = []
    ind_no_date = []
    for index, row in gl_millan_df.iterrows():
        i_date = row["Date1"]

        if len(i_date) == 6:
            i_year = 2000 + int(i_date[:2])
            i_month = int(i_date[2:4])
            i_day = int(i_date[4:])
            time_ax.append(datetime.datetime(year=i_year, month=i_month,
                                             day=i_day))
        else:
            ind_no_date.append(index)
    gl_millan_df = gl_millan_df.drop(ind_no_date)
    gl_millan_df["time"] = time_ax
    gl_millan_df = gl_millan_df.drop(columns=["Date1"])
    gl_millan_df = gl_millan_df.drop_duplicates()

    # - Load Grounding Line from CSK 2013
    gnd_path = grounding_line_path(data_dir, ref_year=2013, sensor="CSK")
    gnd_ln_shp = gnd_path["gnd_ln_shp"]
    csk_2013 = gpd.read_file(gnd_ln_shp)
    csk_date = datetime.datetime(year=2013, month=11, day=24)
    csk_2013["time"] = [csk_date]
    gl_millan_df = pd.concat([gl_millan_df, csk_2013])

    # - sort dataframe according to time axis
    gl_millan_df = gl_millan_df.set_index("time").sort_index()

    idx_gl = gl_millan_df.index.get_indexer([dt_1], method="nearest")

    if verbose:
        print("# - Selected Grounding Line:")
        print(gl_millan_df.iloc[idx_gl])
    gl_time = pd.to_datetime(str(gl_millan_df.iloc[idx_gl].index.values[0]))
    gl_time_str = gl_time.strftime("%Y/%m/%d")

    # - find intersection between selected grounding line and the
    # - longitudinal sampling profile.
    gnd_ln_inter \
        = long_prof_df.intersection(gl_millan_df.iloc[idx_gl].geometry[0])
    xsp_ln, ysp_ln = gnd_ln_inter.geometry[0].coords.xy
    gl_sp = [xsp_ln[0], ysp_ln[0]]
    m_sp = []

    # - Compute the ice particle travel history by updating its location
    # - at a monthly time step. Used temporally interpolated yearly velocity
    # - maps from the MEaSUREs project to update the particles coordinates.
    for mnth in range(n_months):
        r_date = dates_list[mnth]
        year = r_date.year
        month = r_date.month
        day = r_date.day

        if year > 2013:
            # - Load Interpolate velocity Map
            # - If selected, apply smoothing filter to the interpolated maps.
            if v_t_interp == "bilinear":
                v_map = load_velocity_map(year, month, day, data_dir,
                                          domain=v_domain,
                                          smooth=v_smooth,
                                          smooth_mode=v_smooth_mode,
                                          smooth_w_size=v_smooth_size,
                                          verbose=False)
            else:
                v_map = load_velocity_map_nearest(year, month, day,
                                                  data_dir,
                                                  domain=v_domain,
                                                  smooth=v_smooth,
                                                  smooth_mode=v_smooth_mode,
                                                  smooth_w_size=v_smooth_size,
                                                  verbose=False)
        else:
            # - Velocity maps for years before 2014 are characterized
            # - by high noise level and discontinuities. Do not use these maps.
            # - Use velocity from 2014 based on the assumption that ice velocity
            # - does not change significantly over time.
            v_map = load_velocity_map_nearest(2014, 6, 1,
                                              data_dir,
                                              domain=v_domain,
                                              smooth=v_smooth,
                                              smooth_mode=v_smooth_mode,
                                              smooth_w_size=v_smooth_size,
                                              verbose=False)
        # -
        grid_vx = v_map["vx_out"]
        grid_vy = v_map["vy_out"]
        v_map_x_mm = v_map["m_xx"]
        v_map_y_mm = v_map["m_yy"]
        # - Convert yearly velocity [m/yr] to monthly velocity [m/month].
        grid_vx_m = grid_vx / 12.
        grid_vy_m = grid_vy / 12.

        # - Assign glacier elevation values to a specific grid cell
        combined_x_y_arrays = np.dstack([v_map_x_mm.ravel(),
                                         v_map_y_mm.ravel()])[0]
        # - For each of the considered particles,
        # - find the closest grid cell on the velocity domain.
        indexes = do_kdtree(combined_x_y_arrays, gl_sp) - 1
        # - Update Particle coordinates using monthly interpolated velocities
        gl_sp[0] += grid_vx_m.ravel()[indexes]
        gl_sp[1] += grid_vy_m.ravel()[indexes]

        if mnth == int(n_months/2.):
            m_sp = [gl_sp[0], gl_sp[1]]

    return{"s_loc": [xsp_ln[0], ysp_ln[0]], "e_loc": gl_sp, "m_loc": m_sp,
           "gl_time_str": gl_time_str}


def load_closest_gl(dt_1: datetime.datetime, crs: int = 3413) -> dict:
    """
    Load the reference grounding line for the selected date.
    parm dt_1: reference date
    parm crs: Coordinate Reference system
    """
    # - Project Data Directory
    data_dir = os.path.join("/", "Volumes", "Extreme Pro")
    # - Import Grounding Line Dataset
    gl_unified_path = os.path.join(data_dir, "GIS_Data",
                                   "Petermann_GL_Unified",
                                   "Petermann_GL_Unified.shp")
    gl_unified_df \
        = gpd.read_file(gl_unified_path).to_crs(crs)[["time", "geometry"]]

    gl_unified_df.drop_duplicates(subset=["time"], inplace=True)
    # - sort dataframe according to time axis
    time_ax_t = [pd.to_datetime(x) for x in list(gl_unified_df["time"].values)]
    time_ax = [datetime.datetime(x.year, x.month, x.day) for x in time_ax_t]
    gl_unified_df["time"] = time_ax
    # - Use time as DetaFrame Index
    gl_unified_df = gl_unified_df.set_index("time").sort_index()

    # - Find the closest GL
    idx_gl = gl_unified_df.index.get_indexer([dt_1], method="nearest")
    gl_sel = gl_unified_df.iloc[idx_gl].geometry.values[0]
    gl_time = gl_unified_df.iloc[idx_gl].index.values[0]

    return{"gl_time": gl_time, "geometry": gl_sel}


def sample_raster_petermann(data_dir: str, raster_path: str,
                            out_path: str, ref_crs: int,
                            land_color: str = "black",
                            ice_color: str = "grey",
                            grnd_ln_color: str = "k",
                            grnd_zn_color: str = "g",
                            grnd_zn_buffer: float = 0,
                            tt_color: str = "#33ff5e",
                            tl_color: str = "red",
                            title: str = "", annotate: str = "",
                            fig_format: str = "jpeg",
                            vmin: int = 0, vmax: int = 80,
                            ymin: int = -50, ymax: int = 50,
                            save_sample: bool = False,
                            ref_year: int = 2020,
                            units: str = "[m/yr]",
                            cmap: plt = plt.get_cmap("plasma")) -> dict:
    """
    Sample the Input Raster covering the Petermann Glacier"s ice shelf area
    along a longitudinal and a transverse profile
    :param data_dir: absolute path to project data directory
    :param raster_path: absolute path to basal melt rate in GeoTiff format
    :param out_path: absolute path to output figure.=
    :param ref_crs: reference CRS
    :param land_color: land edges color
    :param ice_color: ice edges color
    :param grnd_ln_color: grounding line color
    :param grnd_zn_color: grounding zone color
    :param grnd_zn_buffer: Grounding Zone Buffer
    :param tt_color: Transverse Profile Color
    :param tl_color: Longitudinal Profile Color
    :param title: figure title
    :param annotate: add annotate object
    :param fig_format: figure format [jpeg]
    :param vmin: pcolormesh vmin
    :param vmax: pcolormesh vmax
    :param ymin: sample plot ymin
    :param ymax: sample plot ymax
    :param save_sample: save values along longitudinal and transverse profiles
    :param ref_year: reference year for grounding line selection
    :param units: raster measurement unit
    :param cmap: pcolormesh color map
    :return:
    """
    # - text size
    txt_size = 14       # - main text size
    leg_size = 11       # - legend text size
    label_size = 12     # - label text size
    gl_y_lim = (ymin, ymax)     # - sample plot vertical limits

    # - features extraction shapefiles path
    feat_ex_path = os.path.join(data_dir, "GIS_Data",
                                "Petermann_features_extraction")

    # - Longitudinal Profile - Central Sector
    long_prof_path = os.path.join(feat_ex_path, "longitudinal_profile_hr.shp")
    long_prof_df = gpd.read_file(long_prof_path).to_crs(epsg=ref_crs)
    # - sample basal melt along longitudinal profile
    xl, yl = long_prof_df["geometry"].geometry[0].xy
    long_sample_vect = [(xl[i], yl[i]) for i in range(len(xl))]

    # - Transverse Profile along transverse profile
    trans_prof_path = os.path.join(feat_ex_path, "transverse_profile_hr.shp")
    trans_prof_df = gpd.read_file(trans_prof_path).to_crs(epsg=ref_crs)
    xt, yt = trans_prof_df["geometry"].geometry[0].xy
    trans_sample_vect = [(xt[i], yt[i]) for i in range(len(xt))]

    # - Sample input raster
    with rasterio.open(raster_path, mode="r+") as src_bm:
        long_bm = np.array(list(src_bm.sample(long_sample_vect))).squeeze()
        trans_bm = np.array(list(src_bm.sample(trans_sample_vect))).squeeze()

    # - Plot Basal Melt
    bm_plot = load_dem_tiff(raster_path)
    bm_fig = bm_plot["data"]
    bm_fig[bm_fig == bm_plot["nodata"]] = np.nan
    x_coords = bm_plot["x_centroids"]
    y_coords = bm_plot["y_centroids"]
    xx, yy = np.meshgrid(x_coords, y_coords)

    # - Map Extent
    map_extent = [-61.1, -59.9, 80.4, 81.2]

    # - Path to Ice and Land Masks
    ics_shp = os.path.join("..", "esri_shp", "GIMP",
                           "Petermann_Domain_glaciers_wgs84.shp")
    land_shp = os.path.join("..", "esri_shp", "GIMP",
                            "GSHHS_i_L1_Petermann_clip.shp")
    # - Petermann Grounding Line
    gnd_path = grounding_line_path(data_dir, ref_year=ref_year)
    gnd_ln_shp = gnd_path["gnd_ln_shp"]
    gnd_ln_label = gnd_path["gnd_label"]
    # - Import Grounding Line infor as GeodataFrame
    gnd_ln_df = gpd.read_file(gnd_ln_shp).to_crs(epsg=ref_crs)
    # - grounding line coordinates
    xg, yg = gnd_ln_df["geometry"].geometry[0].xy

    # - Petermann Grounding Zone - Migration 2011/2021
    gnd_zn_shp = gnd_path["gnd_zn_shp"]

    if grnd_zn_buffer:
        # - Clip Output Raster
        clip_shp_mask_path \
            = os.path.join(data_dir, "GIS_Data",
                           "Petermann_features_extraction",
                           "Petermann_iceshelf_clipped_epsg3413.shp")
        clip_mask = gpd.read_file(clip_shp_mask_path).to_crs(epsg=ref_crs)

        gnd_zn_shp_buff = os.path.join(data_dir, "GIS_Data",
                                       "Petermann_features_extraction",
                                       "Petermann_grounding_line_migration_"
                                       f"range_buff{grnd_zn_buffer}"
                                       f"_epsg3413.shp")
        if not os.path.isfile(gnd_zn_shp_buff):
            gnd_zn_to_bf = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)
            gnd_zn_to_bf["geometry"] = gnd_zn_to_bf.geometry\
                .buffer(grnd_zn_buffer)
            # - clip the obtained buffered mask with the
            # - ice shelf perimeter mask.
            gnd_zn_to_bf = gpd.overlay(gnd_zn_to_bf, clip_mask,
                                       how="intersection")
            # - save buffered mask to file
            gnd_zn_to_bf.to_file(gnd_zn_shp_buff)

        # -
        gnd_zn_shp = gnd_zn_shp_buff

    # - Compute the point coordinates of the intersection between the
    # - longitudinal sampling profile and the grounding zone.
    gdz_df = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)

    # - Find intersection between Grounding Line and Longitudinal profile
    gnd_ln_inter = long_prof_df.intersection(gnd_ln_df)
    ysp_ln = gnd_ln_inter.geometry[0].coords.xy[1]

    # - Find intersection between Grounding Zone mask and Longitudinal profile
    gdz_df_inter = long_prof_df.intersection(gdz_df)
    ysp_gz = gdz_df_inter.geometry[0].coords.xy[1]

    # - set Coordinate Reference System
    ref_crs = ccrs.NorthPolarStereo(central_longitude=-45,
                                    true_scale_latitude=70)

    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig)
    # - initialize legend labels
    leg_label_list = []

    # - Plot Melt Rate Map
    ax = fig.add_subplot(gs[:, 0], projection=ref_crs)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Plot Coastlines
    shape_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=land_color)
    # - Plot Glaciers Mask
    shape_feature = ShapelyFeature(Reader(ics_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=ice_color)
    # - Plot Grounding Line 2020/2021
    l1, = ax.plot(xg, yg, color=grnd_ln_color, lw=2, zorder=10, ls="-.")
    leg_label_list.append("Grounding Line - " + gnd_ln_label)
    # - Plot Grounding Zone 2011/2021
    shape_feature = ShapelyFeature(Reader(gnd_zn_shp).geometries(),
                                   ref_crs)
    ax.add_feature(shape_feature, facecolor="None",
                   edgecolor=grnd_zn_color, linestyle="--",
                   linewidth=2)
    leg_label_list.append("Grounding Zone 2011-2020")
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor=grnd_zn_color, facecolor="none",
                            linestyle="--")
    # - Plot Transverse Profile
    l3, = ax.plot(xt, yt, color=tt_color, lw=2, ls="--", zorder=10)
    leg_label_list.append("Transverse Profile")
    # - Plot Longitudinal Profile
    l4, = ax.plot(xl, yl, color=tl_color, lw=2, ls="--", zorder=10)
    leg_label_list.append("Longitudinal Profile")

    # - Set Map Grid
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False, color="k", linestyle="dotted",
                      alpha=0.3)
    gl.top_labels = True
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[0]) - 3.5,
                                         np.floor(map_extent[1]) + 3, 1))
    gl.ylocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[2]) - 5,
                                         np.floor(map_extent[3]) + 5, 0.2))
    gl.xlabel_style = {"rotation": 0, "weight": "bold", "size": 11}
    gl.ylabel_style = {"rotation": 0, "weight": "bold", "size": 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # - Figure title
    ax.set_title(title, weight="bold", loc="left", size=txt_size)
    # - Add Figure Annotation
    ax.annotate(annotate, xy=(0.03, 0.03), xycoords="axes fraction",
                size=label_size, zorder=100,
                bbox=dict(boxstyle="square", fc="w", alpha=0.8))

    # - Plot Melt Rate map
    im = ax.pcolormesh(xx, yy, bm_fig, cmap=cmap,
                       zorder=0, vmin=vmin, vmax=vmax, rasterized=True)

    # add an axes above the main axes.
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("bottom", size="7%", pad="2%",
                                 axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.set_xlabel(units, labelpad=2, size=12, weight="bold")
    cax.xaxis.set_ticks_position("bottom")

    # - Add Legend to Melt Rate Maps
    ax.legend([l1, l2, l3, l4], leg_label_list, loc="upper right",
              fontsize=leg_size, framealpha=0.8,
              facecolor="w", edgecolor="k")

    # - Add ScaleBar
    ax.add_artist(ScaleBar(1, units="m", location="lower right",
                           border_pad=1, pad=0.5, box_color="w",
                           frameon=True))

    # - Plot Longitudinal Profile
    ax = fig.add_subplot(gs[0, 1:])
    title_str = "Longitudinal Profile"
    ax.set_title(title_str, weight="bold", loc="left", size=13)
    ax.set_ylabel(units, weight="bold", size=13)
    ax.set_xlabel("Northing", weight="bold", size=13)
    ax.set_ylim(gl_y_lim)
    ax.grid(color="k", linestyle="dotted", alpha=0.3)
    plt.plot(yl, long_bm, lw=2, color=tl_color, zorder=5)
    # - Add axvspan to mark the spatial range spanned by the Grounding Zone
    ax.axvspan(np.min(ysp_gz), np.max(ysp_gz), facecolor="grey",
               zorder=0, alpha=0.3)
    l_axv = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                               facecolor="grey", edgecolor=grnd_zn_color)

    # - Plot Chosen Grounding line location
    l5 = ax.vlines(ysp_ln, gl_y_lim[0], gl_y_lim[1], color="k",
                   ls="--", lw=2, zorder=1)
    # - Plot Grounding Zone Migration Borders
    ax.vlines(np.min(ysp_gz), gl_y_lim[0], gl_y_lim[1], color=grnd_zn_color,
              ls="--", lw=2, zorder=1)
    ax.vlines(np.max(ysp_gz), gl_y_lim[0], gl_y_lim[1], color=grnd_zn_color,
              ls="--", lw=2, zorder=1)

    # - Add Legend to DhDt Maps
    ax.legend([l_axv, l5], ["Grounding Zone",
                            "Grounding Line - " + gnd_ln_label],
              loc="upper right", fontsize=leg_size, framealpha=0.8,
              facecolor="w", edgecolor="k")

    # - Plot Transverse Profile
    ax = fig.add_subplot(gs[1, 1:])
    title_str = "Transverse Profile"
    ax.set_title(title_str, weight="bold", loc="left", size=13)
    ax.set_ylabel(units, weight="bold", size=13)
    ax.set_xlabel("Easting", weight="bold", size=13)
    ax.set_ylim(gl_y_lim)
    ax.grid(color="k", linestyle="dotted", alpha=0.3)
    plt.plot(xt, trans_bm, lw=2, color=tt_color)

    # - save output figure
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()

    if save_sample:
        out_path_smp_t\
            = out_path.replace("."+fig_format, "_transverse_profile.txt")
        # - Save Raster Values Sampled along the
        # - longitudinal and transverse profiles.
        with open(out_path_smp_t, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xt)
            for s in range(n_samples):
                print(str(xt[s]).ljust(25) + str(yt[s]).ljust(25)
                      + str(trans_bm[s]).ljust(25), file=w_fid)
        out_path_smp_l\
            = out_path_smp_t.replace("transverse", "longitudinal")
        with open(out_path_smp_l, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xl)
            for s in range(n_samples):
                print(str(xl[s]).ljust(25) + str(yl[s]).ljust(25)
                      + str(long_bm[s]).ljust(25), file=w_fid)

    return{"xl": xl, "yl": yl, "long_bm": long_bm,
           "xt": xt, "yt": yt, "trans_bm": trans_bm}


def sample_raster_petermann_no_fig(data_dir: str, raster_path: str,
                                   ref_crs: int, write: bool = True) -> dict:
    """
    Sample the Input Raster covering the Petermann Glacier's ice shelf
    along a longitudinal and a transverse profile - Save sampled data
    in textual format inside the same directory containing the input data
    :param data_dir: absolute path to project data directory
    :param raster_path: absolute path to basal melt rate in GeoTiff format
    :param ref_crs: reference CRS
    :param write: save the obtained samples in ascii format
    :return:
    """
    # - features extraction shapefiles path
    feat_ex_path = os.path.join(data_dir, "GIS_Data",
                                "Petermann_features_extraction")

    # - Longitudinal Profile
    long_prof_path = os.path.join(feat_ex_path, "longitudinal_profile_hr.shp")
    long_prof_df = gpd.read_file(long_prof_path).to_crs(epsg=ref_crs)
    # -  Sample input raster along longitudinal profile
    xl, yl = long_prof_df["geometry"].geometry[0].xy
    long_sample_vect = [(xl[i], yl[i]) for i in range(len(xl))]

    # - Transverse Profile
    trans_prof_path = os.path.join(feat_ex_path, "transverse_profile_hr.shp")
    trans_prof_df = gpd.read_file(trans_prof_path).to_crs(epsg=ref_crs)
    # - Sample input raster along transverse profile
    xt, yt = trans_prof_df["geometry"].geometry[0].xy
    trans_sample_vect = [(xt[i], yt[i]) for i in range(len(xt))]

    # - Longitudinal Profile - Eastern Sector
    long_prof_east_path = os.path.join(feat_ex_path,
                                       "longitudinal_profile_east_hr.shp")
    long_prof_east_df = gpd.read_file(long_prof_east_path).to_crs(epsg=ref_crs)
    # -  Sample input raster along longitudinal profile east
    xle, yle = long_prof_east_df["geometry"].geometry[0].xy
    long_east_sample_vect = [(xle[i], yle[i]) for i in range(len(xle))]

    with rasterio.open(raster_path, mode="r+") as src_bm:
        long_bm = np.array(list(src_bm.sample(long_sample_vect))).squeeze()
        trans_bm = np.array(list(src_bm.sample(trans_sample_vect))).squeeze()
        long_east_bm \
            = np.array(list(src_bm.sample(long_east_sample_vect))).squeeze()

    if write:
        out_path_smp_t\
            = raster_path.replace(".tiff", "_transverse_profile.txt")
        # - Save Raster Values Sampled along the
        # - longitudinal and transverse profiles.
        with open(out_path_smp_t, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xt)
            for s in range(n_samples):
                print(str(xt[s]).ljust(25) + str(yt[s]).ljust(25)
                      + str(trans_bm[s]).ljust(25), file=w_fid)
        out_path_smp_l\
            = out_path_smp_t.replace("transverse", "longitudinal")
        with open(out_path_smp_l, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xl)
            for s in range(n_samples):
                print(str(xl[s]).ljust(25) + str(yl[s]).ljust(25)
                      + str(long_bm[s]).ljust(25), file=w_fid)
        # - Longitudinal Profile East
        out_path_smp_le \
            = out_path_smp_t.replace("transverse_profile",
                                     "longitudinal_profile_east_hr")
        with open(out_path_smp_le, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xle)
            for s in range(n_samples):
                print(str(xle[s]).ljust(25) + str(yle[s]).ljust(25)
                      + str(long_east_bm[s]).ljust(25), file=w_fid)

    return{"xl": xl, "yl": yl, "long_bm": long_bm,
           "xt": xt, "yt": yt, "trans_bm": trans_bm,
           "xle": xle, "yle": yle, "long_east_bm": long_east_bm}


def plot_dhdt_map(data_dir: str, img: np.array, xx: np.array, yy: np.array,
                  out_path: str, ref_crs: int, extent: int = 1,
                  land_color: str = "black", ice_color: str = "grey",
                  grnd_ln_color: str = "k", grnd_zn_color: str = "g",
                  grnd_zn_buffer: float = 0, title: str = "",
                  annotate: str = "", cmap=plt.get_cmap("bwr_r"),
                  vmin: int = -10, vmax: int = 10,
                  fig_format: str = "jpeg") -> None:
    """
    Plot Elevation Change DhDt Map - Wide Domain
    :param data_dir: Project Data Directory
    :param img: elevation change [numpy array]
    :param xx: xx m-grid - centroids [numpy array]
    :param yy: yy m-grid - centroids [numpy array]
    :param out_path: absolute path to output file
    :param ref_crs: coordinates reference system
    :param extent: figure extent [1,2,3]
    :param land_color: land edges color
    :param ice_color: ice edges color
    :param grnd_ln_color: Grounding Line Color
    :param grnd_zn_color: Grounding Zone Color
    :param grnd_zn_buffer: Grounding Zone Buffer
    :param title: figure title
    :param annotate: add annotate object
    :param fig_format: figure format [jpeg]
    :param cmap: imshow/pcolormesh color map
    :param vmin: imshow/pcolormesh vmin
    :param vmax: imshow/pcolormesh vmax
    :return: None
    """
    # - text size
    txt_size = 14  # - main text size
    leg_size = 13  # - legend text size
    label_size = 12  # - label text size

    if extent == 1:
        # - Map Extent 1 - wide
        map_extent = [-68, -55, 80, 82]
        figsize = (9, 9)
    elif extent == 2:
        # - Map Extent 2 - zoom 1
        map_extent = [-61.5, -57, 79.5, 81.5]
        figsize = (5, 8)
    elif extent == 3:
        # - Map Extent 3 - zoom2
        map_extent = [-61.5, -57.6, 80.2, 81.2]
        figsize = (9, 9)
    else:
        # - Map Extent 4 - zoom3
        map_extent = [-61.1, -59.9, 80.4, 81.2]
        figsize = (5, 8)
        txt_size = 12  # - main text size
        leg_size = 10   # - legend text size
        label_size = 10  # - label text size

    # - Path to Ice and Land Masks
    ics_shp = os.path.join("..", "esri_shp", "GIMP",
                           "Petermann_Domain_glaciers_wgs84.shp")
    land_shp = os.path.join("..", "esri_shp", "GIMP",
                            "GSHHS_i_L1_Petermann_clip.shp")

    # - Petermann Grounding Line - 2021/2021
    gnd_ln_shp = os.path.join(data_dir, "coco_petermann_grnd_lines_2020-2021",
                              "grnd_lines_shp_to_share",
                              "coco20200501_20200502-20200517_20200518",
                              "coco20200501_20200502-20200517_20200518"
                              "_grnd_line.shp")
    gnd_ln_df = gpd.read_file(gnd_ln_shp).to_crs(epsg=ref_crs)
    # -
    xg, yg = gnd_ln_df["geometry"].geometry[0].xy

    # - Petermann Grounding Zone - 2011/2021
    gnd_zn_shp = os.path.join(data_dir, "GIS_Data",
                              "Petermann_features_extraction",
                              "Petermann_grounding_line_migration_"
                              "range_epsg3413.shp")
    if grnd_zn_buffer:
        # - Clip Output Raster
        clip_shp_mask_path \
            = os.path.join(data_dir, "GIS_Data",
                           "Petermann_features_extraction",
                           "Petermann_iceshelf_clipped_epsg3413.shp")
        clip_mask = gpd.read_file(clip_shp_mask_path).to_crs(epsg=ref_crs)

        gnd_zn_shp_buff = os.path.join(data_dir, "GIS_Data",
                                       "Petermann_features_extraction",
                                       "Petermann_grounding_line_migration_"
                                       f"range_buff{grnd_zn_buffer}"
                                       f"_epsg3413.shp")
        if not os.path.isfile(gnd_zn_shp_buff):
            gnd_zn_to_bf = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)
            gnd_zn_to_bf["geometry"] = gnd_zn_to_bf.geometry\
                .buffer(grnd_zn_buffer)
            # - clip the obtained buffered mask with the
            # - ice shelf perimeter mask.
            gnd_zn_to_bf = gpd.overlay(gnd_zn_to_bf, clip_mask,
                                       how="intersection")
            # - save buffered mask to file
            gnd_zn_to_bf.to_file(gnd_zn_shp_buff)

        # -
        gnd_zn_shp = gnd_zn_shp_buff

    # - set Coordinate Reference System
    ref_crs = ccrs.NorthPolarStereo(central_longitude=-45,
                                    true_scale_latitude=70)
    # - initialize matplotlib figure object
    fig = plt.figure(figsize=figsize)
    # - initialize legend labels
    leg_label_list = []
    ax = fig.add_subplot(1, 1, 1, projection=ref_crs)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Plot Coastlines
    shape_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=land_color)
    # - Plot Glaciers Mask
    shape_feature = ShapelyFeature(Reader(ics_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=ice_color)

    # - Plot Grounding Line 2020/2021
    l1, = ax.plot(xg, yg, color=grnd_ln_color, lw=2, zorder=10, ls="-.")
    leg_label_list.append("Grounding Line 2020")
    # - Plot Grounding Zone 2011/2021
    shape_feature = ShapelyFeature(Reader(gnd_zn_shp).geometries(), ref_crs)
    ax.add_feature(shape_feature, facecolor="None",
                   edgecolor=grnd_zn_color, linestyle="--",
                   linewidth=2)
    leg_label_list.append("Grounding Zone 2011-2020")
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor=grnd_zn_color, facecolor="none",
                            linestyle="--")

    # - Set Map Grid
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False)
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(map_extent[0] - 5,
                                                 map_extent[1] + 5, 2))
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(map_extent[2]) - 5,
                  np.floor(map_extent[3]) + 5,
                  0.5))
    gl.xlabel_style = {"rotation": 0, "weight": "bold", "size": label_size}
    gl.ylabel_style = {"rotation": 0, "weight": "bold", "size": label_size}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # - Figure title
    ax.set_title(title, weight="bold", loc="left", size=txt_size)
    # - Add Figure Annotation
    ax.annotate(annotate, xy=(0.03, 0.03), xycoords="axes fraction",
                size=label_size, zorder=100,
                bbox=dict(boxstyle="square", fc="w", alpha=0.8))

    # - Plot DH/DT map
    im = ax.pcolormesh(xx, yy, img, cmap=cmap,
                       zorder=0, vmin=vmin, vmax=vmax)

    # - colorbar - solution specific for cartopy figures
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label(label="[m/year]", weight="bold", size=label_size)
    cb.ax.tick_params(labelsize=leg_size)

    # - Add Legend to DhDt Maps
    ax.legend([l1, l2], leg_label_list, loc="upper right",
              fontsize=leg_size, framealpha=0.8,
              facecolor="w", edgecolor="k")

    # - Add ScaleBar
    ax.add_artist(ScaleBar(1, units="m", location="lower right",
                           border_pad=1, pad=0.5, box_color="w",
                           frameon=True))

    # - save output figure
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()


def plot_dhdt_map_zoom(data_dir: str, dhdt_path: str, out_path: str,
                       ref_crs: int, extent: int = 1, land_color: str = "black",
                       ice_color: str = "grey", grnd_ln_color: str = "k",
                       grnd_zn_color: str = "g", grnd_zn_buffer: float = 0,
                       title: str = "", annotate: str = "",
                       fig_format: str = "jpeg",
                       cmap=plt.get_cmap("bwr_r"),
                       vmin: int = -10, vmax: int = 10
                       ) -> None:
    """
    Plot Ice Elevation Change DhDt Map - Zoom Domain
    :param data_dir: absolute path to project data directory
    :param dhdt_path: absolute path to dhdt in GeoTiff format
    :param out_path: absolute path to output figure
    :param ref_crs: reference CRS
    :param extent: map extent
    :param land_color: land edges color
    :param ice_color: ice edges color
    :param grnd_ln_color: Grounding Line Color
    :param grnd_zn_color: Grounding Zone Color
    :param grnd_zn_buffer: Grounding Zone Buffer
    :param title: figure title
    :param annotate: add annotate object
    :param fig_format: figure format [jpeg]
    :param cmap: imshow/pcolormesh color map
    :param vmin: imshow/pcolormesh vmin
    :param vmax: imshow/pcolormesh vmax
    :return:
    """
    # - text size
    txt_size = 14       # - main text size
    leg_size = 13       # - legend text size
    label_size = 12     # - label text siz

    # - Plot DhDt Map
    dhdt_plot = load_dem_tiff(dhdt_path)
    dhdt_fig = dhdt_plot["data"]
    dhdt_fig[dhdt_fig == dhdt_plot["nodata"]] = np.nan
    x_coords = dhdt_plot["x_centroids"]
    y_coords = dhdt_plot["y_centroids"]
    xx, yy = np.meshgrid(x_coords, y_coords)

    # - Map Extent
    if extent == 1:
        map_extent = [-61.1, -59.9, 80.4, 81.2]
        figsize = (6, 9)
    else:
        map_extent = [-60.8, -59.1, 80.4, 80.7]
        figsize = (9, 9)

    # - Path to Ice and Land Masks
    ics_shp = os.path.join("..", "esri_shp", "GIMP",
                           "Petermann_Domain_glaciers_wgs84.shp")
    land_shp = os.path.join("..", "esri_shp", "GIMP",
                            "GSHHS_i_L1_Petermann_clip.shp")
    # - Petermann Grounding Line - 2021/2021
    gnd_path = grounding_line_path(data_dir)
    gnd_ln_shp = gnd_path["gnd_ln_shp"]
    gnd_ln_df = gpd.read_file(gnd_ln_shp).to_crs(epsg=ref_crs)
    # -
    xg, yg = gnd_ln_df["geometry"].geometry[0].xy

    # - Petermann Grounding Zone - 2011/2021
    gnd_zn_shp = gnd_path["gnd_zn_shp"]

    if grnd_zn_buffer:
        # - Clip Output Raster
        clip_shp_mask_path \
            = os.path.join(data_dir, "GIS_Data",
                           "Petermann_features_extraction",
                           "Petermann_iceshelf_clipped_epsg3413.shp")
        clip_mask = gpd.read_file(clip_shp_mask_path).to_crs(epsg=ref_crs)

        gnd_zn_shp_buff = os.path.join(data_dir, "GIS_Data",
                                       "Petermann_features_extraction",
                                       "Petermann_grounding_line_migration_"
                                       f"range_buff{grnd_zn_buffer}"
                                       f"_epsg3413.shp")
        if not os.path.isfile(gnd_zn_shp_buff):
            gnd_zn_to_bf = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)
            gnd_zn_to_bf["geometry"] = gnd_zn_to_bf.geometry\
                .buffer(grnd_zn_buffer)
            # - clip the obtained buffered mask with the
            # - ice shelf perimeter mask.
            gnd_zn_to_bf = gpd.overlay(gnd_zn_to_bf, clip_mask,
                                       how="intersection")
            # - save buffered mask to file
            gnd_zn_to_bf.to_file(gnd_zn_shp_buff)

        # -
        gnd_zn_shp = gnd_zn_shp_buff

    # - set Coordinate Reference System
    ref_crs = ccrs.NorthPolarStereo(central_longitude=-45,
                                    true_scale_latitude=70)
    fig = plt.figure(figsize=figsize)
    # - initialize legend labels
    leg_label_list = []

    # - Plot DhDt Map
    ax = fig.add_subplot(1, 1, 1, projection=ref_crs)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Plot Coastlines
    shape_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=land_color)

    # - Plot Glaciers Mask
    shape_feature = ShapelyFeature(Reader(ics_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=ice_color)

    # - Plot Grounding Line 2020/2021
    l1, = ax.plot(xg, yg, color=grnd_ln_color, lw=2, zorder=10, ls="-.")
    leg_label_list.append("Grounding Line 2020")
    # - Plot Grounding Zone 2011/2021
    shape_feature = ShapelyFeature(Reader(gnd_zn_shp).geometries(), ref_crs)
    ax.add_feature(shape_feature, facecolor="None",
                   edgecolor=grnd_zn_color, linestyle="--",
                   linewidth=2)
    leg_label_list.append("Grounding Zone 2011-2020")
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor=grnd_zn_color, facecolor="none",
                            linestyle="--")

    # - Set Map Grid
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False, color="k", linestyle="dotted",
                      alpha=0.3)
    gl.top_labels = False
    gl.bottom_labels = True
    gl.right_labels = False
    gl.xlocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[0]) - 3.5,
                                         np.floor(map_extent[1]) + 3, 1))
    gl.ylocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[2]) - 5,
                                         np.floor(map_extent[3]) + 5, 0.2))
    gl.xlabel_style = {"rotation": 0, "weight": "bold", "size": label_size}
    gl.ylabel_style = {"rotation": 0, "weight": "bold", "size": label_size}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # - Plot DH/DT map
    im = ax.pcolormesh(xx, yy, dhdt_fig, cmap=cmap,
                       zorder=0, vmin=vmin, vmax=vmax, rasterized=True)

    # add an axes above the main axes.
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label(label="[m/year]", weight="bold", size=label_size)
    cb.ax.tick_params(labelsize="medium")

    # - Figure title
    ax.set_title(title, weight="bold", loc="left", size=txt_size)
    # - Add Figure Annotation
    ax.annotate(annotate, xy=(0.03, 0.03), xycoords="axes fraction",
                size=label_size, zorder=100,
                bbox=dict(boxstyle="square", fc="w", alpha=0.8))

    # - Add Legend to DhDt Maps
    ax.legend([l1, l2], leg_label_list, loc="upper right",
              fontsize=leg_size, framealpha=0.8,
              facecolor="w", edgecolor="k")

    # - Add ScaleBar
    ax.add_artist(ScaleBar(1, units="m", location="lower right",
                           border_pad=1, pad=0.5, box_color="w",
                           frameon=True))

    # - save output figure
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()


def sample_melt_rate_petermann(data_dir: str, raster_path: str,
                               dt_1: datetime.datetime,
                               dt_2: datetime.datetime,
                               out_path: str, ref_crs: int,
                               land_color: str = "black",
                               ice_color: str = "grey",
                               grnd_ln_color: str = "b",
                               grnd_zn_color: str = "g",
                               tt_color: str = "#33ff5e",
                               tl_color: str = "red",
                               title: str = "", annotate: str = "",
                               fig_format: str = "jpeg",
                               vmin: int = 0, vmax: int = 80,
                               ymin: int = -50, ymax: int = 50,
                               save_sample: bool = False,
                               ref_year: int = 2020,
                               units: str = "[m/yr]",
                               cmap: plt = plt.get_cmap("plasma")) -> dict:
    """
    Sample the Input Raster covering the Petermann Glacier"s ice shelf area
    along a longitudinal and a transverse profile
    :param data_dir: absolute path to project data directory
    :param raster_path: absolute path to basal melt rate in GeoTiff format
    :param dt_1: initial reference date
    :param dt_2: final reference date
    :param out_path: absolute path to output figure.=
    :param ref_crs: reference CRS
    :param land_color: land edges color
    :param ice_color: ice edges color
    :param grnd_ln_color: grounding line color
    :param grnd_zn_color: grounding zone color
    :param tt_color: Transverse Profile Color
    :param tl_color: Longitudinal Profile Color
    :param title: figure title
    :param annotate: add annotate object
    :param fig_format: figure format [jpeg]
    :param vmin: pcolormesh vmin
    :param vmax: pcolormesh vmax
    :param ymin: sample plot ymin
    :param ymax: sample plot ymax
    :param save_sample: save values along longitudinal and transverse profiles
    :param ref_year: reference year for grounding line selection
    :param units: raster measurement unit
    :param cmap: pcolormesh color map
    :return:
    """
    # - text size
    txt_size = 14       # - main text size
    leg_size = 11       # - legend text size
    label_size = 12     # - label text size
    gl_y_lim = (ymin, ymax)     # - sample plot vertical limits
    ca_color = grnd_ln_color    # - Grounding Line Propagation color
    ca_color_txt = "#2072f2"    # - Grounding Line Propagation color

    # - features extraction shapefiles path
    feat_ex_path = os.path.join(data_dir, "GIS_Data",
                                "Petermann_features_extraction")

    # - Longitudinal Profile
    long_prof_path = os.path.join(feat_ex_path, "longitudinal_profile_hr.shp")
    long_prof_df = gpd.read_file(long_prof_path).to_crs(epsg=ref_crs)
    # - sample basal melt along longitudinal profile
    xl, yl = long_prof_df["geometry"].geometry[0].xy
    long_sample_vect = [(xl[i], yl[i]) for i in range(len(xl))]

    # - Transverse Profile along transverse profile
    trans_prof_path = os.path.join(feat_ex_path, "transverse_profile_hr.shp")
    trans_prof_df = gpd.read_file(trans_prof_path).to_crs(epsg=ref_crs)

    # - Sample input raster
    xt, yt = trans_prof_df["geometry"].geometry[0].xy
    trans_sample_vect = [(xt[i], yt[i]) for i in range(len(xt))]
    with rasterio.open(raster_path, mode="r+") as src_bm:
        long_bm = np.array(list(src_bm.sample(long_sample_vect))).squeeze()
        trans_bm = np.array(list(src_bm.sample(trans_sample_vect))).squeeze()

    # - Compute intersection between Longitudinal and Transverse Profile
    profile_inter = long_prof_df.intersection(trans_prof_df)
    _, ysp_ln = profile_inter.geometry[0].coords.xy

    # - Plot Basal Melt
    bm_plot = load_dem_tiff(raster_path)
    bm_fig = bm_plot["data"]
    bm_fig[bm_fig == bm_plot["nodata"]] = np.nan
    x_coords = bm_plot["x_centroids"]
    y_coords = bm_plot["y_centroids"]
    xx, yy = np.meshgrid(x_coords, y_coords)

    # - Map Extent
    map_extent = [-61.1, -59.9, 80.4, 81.2]

    # - Path to Ice and Land Masks
    ics_shp = os.path.join("..", "esri_shp", "GIMP",
                           "Petermann_Domain_glaciers_wgs84.shp")
    land_shp = os.path.join("..", "esri_shp", "GIMP",
                            "GSHHS_i_L1_Petermann_clip.shp")

    # - Petermann Grounding Line - Grounding Zone info
    gnd_path = grounding_line_path(data_dir, ref_year=ref_year)

    # - Import the closest Grounding Line to the first datetime object
    gnd_ln_close = load_closest_gl(dt_1)
    # - grounding line coordinates
    xg, yg = gnd_ln_close["geometry"].xy

    # - Petermann Grounding Zone - Migration 2011/2021
    gnd_zn_shp = gnd_path["gnd_zn_shp"]

    gpt_travel = move_gpt_along_flow(dt_1, dt_2, data_dir,
                                     verbose=False)

    # - Compute the point coordinates of the intersection between the
    # - longitudinal sampling profile and the grounding zone.
    gdz_df = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)

    # - Find intersection between Grounding Zone mask and Longitudinal profile
    gdz_df_inter = long_prof_df.intersection(gdz_df)
    ysp_gz = gdz_df_inter.geometry[0].coords.xy[1]

    # - set Coordinate Reference System
    ref_crs = ccrs.NorthPolarStereo(central_longitude=-45,
                                    true_scale_latitude=70)

    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig)
    # - initialize legend labels
    leg_label_list = []

    # - Plot Melt Rate Map
    ax = fig.add_subplot(gs[:, 0], projection=ref_crs)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Plot Coastlines
    shape_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=land_color)
    # - Plot Glaciers Mask
    shape_feature = ShapelyFeature(Reader(ics_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor="None", edgecolor=ice_color)
    # - Plot Grounding Line 2020/2021
    l1, = ax.plot(xg, yg, color=grnd_ln_color, lw=2, zorder=10, ls="-.")
    leg_label_list.append("Grounding Line - " + gpt_travel["gl_time_str"])
    # - Plot Grounding Zone 2011/2021
    shape_feature = ShapelyFeature(Reader(gnd_zn_shp).geometries(),
                                   ref_crs)
    ax.add_feature(shape_feature, facecolor="None",
                   edgecolor=grnd_zn_color, linestyle="--",
                   linewidth=2)
    leg_label_list.append("Grounding Zone 2011-2020")
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor=grnd_zn_color, facecolor="none",
                            linestyle="--")
    # - Plot Transverse Profile
    l3, = ax.plot(xt, yt, color=tt_color, lw=2, ls="--", zorder=10)
    leg_label_list.append("Transverse Profile")
    # - Plot Longitudinal Profile
    l4, = ax.plot(xl, yl, color=tl_color, lw=2, ls="--", zorder=10)
    leg_label_list.append("Longitudinal Profile")

    # - Set Map Grid
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False, color="k", linestyle="dotted",
                      alpha=0.3)
    gl.top_labels = True
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[0]) - 3.5,
                                         np.floor(map_extent[1]) + 3, 1))
    gl.ylocator \
        = mticker.FixedLocator(np.arange(np.floor(map_extent[2]) - 5,
                                         np.floor(map_extent[3]) + 5, 0.2))
    gl.xlabel_style = {"rotation": 0, "weight": "bold", "size": 11}
    gl.ylabel_style = {"rotation": 0, "weight": "bold", "size": 11}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # - Figure title
    ax.set_title(title, weight="bold", loc="left", size=txt_size)
    # - Add Figure Annotation
    ax.annotate(annotate, xy=(0.03, 0.03), xycoords="axes fraction",
                size=label_size, zorder=100,
                bbox=dict(boxstyle="square", fc="w", alpha=0.8))

    # - Plot Melt Rate map
    im = ax.pcolormesh(xx, yy, bm_fig, cmap=cmap,
                       zorder=0, vmin=vmin, vmax=vmax, rasterized=True)

    # add an axes above the main axes.
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("bottom", size="7%", pad="2%",
                                 axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.set_xlabel(units, labelpad=2, size=12, weight="bold")
    cax.xaxis.set_ticks_position("bottom")

    # - Add Legend to Melt Rate Maps
    ax.legend([l1, l2, l3, l4], leg_label_list, loc="upper right",
              fontsize=leg_size, framealpha=0.8,
              facecolor="w", edgecolor="k")

    # - Add ScaleBar
    ax.add_artist(ScaleBar(1, units="m", location="lower right",
                           border_pad=1, pad=0.5, box_color="w",
                           frameon=True))

    # - Plot Longitudinal Profile
    ax = fig.add_subplot(gs[0, 1:])
    title_str = "Longitudinal Profile"
    ax.set_title(title_str, weight="bold", loc="left", size=13)
    ax.set_ylabel(units, weight="bold", size=13)
    ax.set_xlabel("Northing", weight="bold", size=13)
    ax.set_ylim(gl_y_lim)
    ax.grid(color="k", linestyle="dotted", alpha=0.3)
    plt.plot(yl, long_bm, lw=2, color=tl_color, zorder=5)
    # - Add axvspan to mark the spatial range spanned by the Grounding Zone
    ax.axvspan(np.min(ysp_gz), np.max(ysp_gz), facecolor="grey",
               zorder=0, alpha=0.3)
    l_axv = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                               facecolor="grey", edgecolor=grnd_zn_color)

    # - Plot Chosen Grounding line location
    # - Plot Grounded Ice Propagation Limits
    ax.axvspan(gpt_travel["s_loc"][1], gpt_travel["m_loc"][1],
               facecolor=ca_color, zorder=0, alpha=0.3)
    l_vln_gl = ax.axvline(gpt_travel["s_loc"][1],
                          color=ca_color_txt, ls="--", lw=2, zorder=2)
    l_gl = mpatches.Rectangle((0, 0), 200, 200, linewidth=2,
                              facecolor=ca_color, edgecolor=ca_color_txt,
                              ls="--", alpha=0.3)

    # - Add Figure Annotation - Starting Grounding Line
    ax.text(gpt_travel["s_loc"][1]-1200, -1,
            gpt_travel["gl_time_str"],
            weight="bold", size=16, rotation="vertical",
            color=ca_color_txt, zorder=100)

    # - Plot Grounding Zone Migration Borders
    ax.vlines(np.min(ysp_gz), gl_y_lim[0], gl_y_lim[1], color=grnd_zn_color,
              ls="--", lw=2, zorder=1)
    ax.vlines(np.max(ysp_gz), gl_y_lim[0], gl_y_lim[1], color=grnd_zn_color,
              ls="--", lw=2, zorder=1)

    # - Plot Transverse Profile
    tt_p = ax.vlines(ysp_ln, gl_y_lim[0], gl_y_lim[1], color=tt_color,
                     ls="--", lw=2, zorder=2)

    # - Add Legend to DhDt Maps
    ax.legend([l_axv, l_vln_gl, l_gl, tt_p],
              ["Grounding Zone",
               "Grounding Line - " + gpt_travel["gl_time_str"],
               "Melt Rate Validity Limit", "Transverse Profile"],
              loc="upper right", fontsize=leg_size, framealpha=0.8,
              facecolor="w", edgecolor="k")

    # - Plot Transverse Profile
    ax = fig.add_subplot(gs[1, 1:])
    title_str = "Transverse Profile"
    ax.set_title(title_str, weight="bold", loc="left", size=13)
    ax.set_ylabel(units, weight="bold", size=13)
    ax.set_xlabel("Easting", weight="bold", size=13)
    ax.set_ylim(gl_y_lim)
    ax.grid(color="k", linestyle="dotted", alpha=0.3)
    plt.plot(xt, trans_bm, lw=2, color=tt_color)

    # - save output figure
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()

    if save_sample:
        out_path_smp_t\
            = out_path.replace("."+fig_format, "_transverse_profile.txt")
        # - Save Raster Values Sampled along the
        # - longitudinal and transverse profiles.
        with open(out_path_smp_t, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xt)
            for s in range(n_samples):
                print(str(xt[s]).ljust(25) + str(yt[s]).ljust(25)
                      + str(trans_bm[s]).ljust(25), file=w_fid)
        out_path_smp_l\
            = out_path_smp_t.replace("transverse", "longitudinal")
        with open(out_path_smp_l, "w", encoding="utf8") as w_fid:
            print("x".ljust(25) + "y".ljust(25) + "bm".ljust(25), file=w_fid)
            n_samples = len(xl)
            for s in range(n_samples):
                print(str(xl[s]).ljust(25) + str(yl[s]).ljust(25)
                      + str(long_bm[s]).ljust(25), file=w_fid)

    return{"xl": xl, "yl": yl, "long_bm": long_bm,
           "xt": xt, "yt": yt, "trans_bm": trans_bm}
