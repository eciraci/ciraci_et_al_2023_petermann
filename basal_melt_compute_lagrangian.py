#!/usr/bin/env python
u"""
basal_melt_compute_lagrangian_ts.py
Written by Enrico Ciraci' (02/2022)

Compute list Ice Shelf basal melt rate from a stack of digital elevation models
from the TanDEM-X mission in a Lagrangian framework.

NOTE: Basal melt-rate is calculated following the methodology described by
      Shean, D.E., Joughin, I.R., Dutrieux, P., Smith, B.E. and Berthier,
      E., 2019. Ice shelf basal melt rates from a high-resolution digital
      elevation model (DEM) record for Pine Island Glacier, Antarctica.
      The Cryosphere, 13(10), pp.2633-2656.
      https://tc.copernicus.org/articles/13/2633/2019/
                tc-13-2633-2019-discussion.html

usage: basal_melt_compute_lagrangian_ts.py [-h] [--directory DIRECTORY]
      [--outdir OUTDIR] [--res RES] [--f_year F_YEAR] [--l_year L_YEAR]
      [--delta_time DELTA_TIME] [--smooth] [--w_size W_SIZE]
      [--method {bilinear,nearest}] [--v_smooth]
      [--bm_loc {init_pixel,along_flow,midpoint}]
      [--sample_bm] [--grid_method {bilinear,nearest}] [--np NP]


optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -D DIRECTORY
                        Project data directory.
  --outdir OUTDIR, -O OUTDIR
                        Output directory.
  --res RES, -R RES     Input raster resolution.
  --f_year F_YEAR, -F F_YEAR
                        First Year to consider.
  --l_year L_YEAR, -L L_YEAR
                        Last Year to consider.
  --delta_time DELTA_TIME, -T DELTA_TIME
                        Distance in years between DEMs to compare.
  --smooth              Smooth Melt Rate estimates using a moving average filter
  --w_size W_SIZE, -W W_SIZE
                        Smoothing Window Size [in meters]
  --method {bilinear,nearest}, -M {bilinear,nearest}
                        Velocity TEMPORAL Interpolation Method.
  --v_smooth, -V        Smooth Input velocity Maps.
  --bm_loc {init_pixel,along_flow,midpoint}, -B {init_pixel,along_flow,midpoint}
                        Basal melt redistribution strategy.
  --sample_bm           Sample basal melt rate estimate.
  --sample_only         Sample previously calculated basal melt rate estimate.
  --grid_method {bilinear,nearest}, -G {bilinear,nearest}
                        Griddata interpolation Method.
  --np NP, -N NP        Number of Parallel Processes.

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    scipy: Fundamental algorithms for scientific computing in Python
          https://scipy.org
    matplotlib: Library for creating static, animated, and interactive
           visualizations in Python.
           https://matplotlib.org
    pandas: Python Data Analysis Library
           https://pandas.pydata.org
    geopandas: Python tools for geographic data
           https://pandas.pydata.org
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.io
    fiona: Fiona reads and writes geographic data files.
           https://fiona.readthedocs.io
    shapely: Manipulation and analysis of geometric objects in the Cartesian
           plane.
           https://shapely.readthedocs.io/en/stable

UPDATE HISTORY:

"""
# - Python Dependencies
from __future__ import print_function
import os
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
from scipy.interpolate import griddata
from scipy import signal
import pandas as pd
import datetime
from pyproj import CRS
from pyproj import Transformer
# - Program Dependencies
from utility_functions_rio import load_dem_tiff, virtual_warp_rio, save_raster,\
    clip_raster
from utility_functions import calculate_n_month, create_dir, divergence,\
    do_kdtree, sample_raster_petermann_no_fig
from utility_functions_tdx import dem_2_skip
from velocity_map_time_interpolation import load_velocity_map,\
    load_velocity_map_nearest
from RACMO23p2Loader import RACMO23p2Loader


def load_smb_data(dem_1_date: datetime.datetime, dem_2_date: datetime.datetime,
                  project_dir: str, ref_crs: int = 3413,
                  verbose: bool = False) -> dict:
    """
    Load RACMO data using RACMO23p2Loader
    :param dem_1_date: DEM1 acquisition time
    :param dem_2_date: DEM2 acquisition time
    :param project_dir: absolute path to project data directory
    :param ref_crs: smb data crs
    :param verbose: if True print on screen
    :return: Python dictionary
    """
    if verbose:
        print('# - Loading SMB data from RACMO2.3p2.')
    smb_domain_name = 'Petermann_Domain_Velocity_Stereo'
    lat_pt = 80.5
    lon_pt = -60.
    rho_ice = 0.9167  # - density of ice g/cm3
    # - Calculate the number of months separating the two DEMs
    time_ax = calculate_n_month(dem_1_date, dem_2_date)
    delta_time_yr = time_ax['delta_time'].days/365.25

    # - Define Output Projection transformation - to calculate lat/lon
    # - coordinates of each grid point.
    transformer = Transformer.from_crs(CRS.from_epsg(4326),
                                       CRS.from_epsg(ref_crs))
    # - transform input sample coordinates provided in Lat/Lon into
    # - north polar stereographic projection
    x_sp, y_sp = transformer.transform(lat_pt, lon_pt)
    # - remove reference mean before calculating Cum. SMB.
    rm_mean = True
    smb_data_path = os.path.join(project_dir, 'SMB', 'RACMO2.3p2',
                                 'output.dir', smb_domain_name, 'smb')
    racmo_data = RACMO23p2Loader(smb_data_path)

    # - Extract SMB time series at the selected sample location.
    ds_cum_pt_ts = racmo_data.smb_pt_t_series(x_sp, y_sp, rm_mean=rm_mean)
    # - Convert SMB time series from mm. w.e. to meter of ice equivalent
    ds_cum_pt_ts['cum_smb_pt'] = ds_cum_pt_ts['cum_smb_pt'] / 1e3 * rho_ice
    ds_cum_pt_ts['smb_pt'] = ds_cum_pt_ts['smb_pt'] / 1e3 * rho_ice

    # - Calculate smb time series over the entire SMB data domain
    ds_cum_ts = racmo_data.smb_t_series(rm_mean=rm_mean)
    # - Convert SMB time series from mm. w.e. to meter of ice equivalent
    ds_cum_ts['cum_smb_grid'] = ds_cum_ts['cum_smb_grid'] / 1e3 * rho_ice
    # - Extract dataset portion relative to the period of interest
    ds_cum_ts_sel = ds_cum_ts.sel(time=slice(dem_1_date,
                                             dem_2_date), drop=True)
    # - Plot Cumulative SMB Map
    cum_smb_grid_sel = ds_cum_ts_sel['cum_smb_grid'].values
    # - Convert the ice elevation change into average annual elevation change
    delta_h_grid = (cum_smb_grid_sel[-1] - cum_smb_grid_sel[0])/delta_time_yr

    # -
    return {'delta_h_grid': delta_h_grid, 'x_coords': ds_cum_ts_sel['x'].values,
            'y_coords': ds_cum_ts_sel['y'].values}


def load_ice_mask(data_dir: str, res: int = 150, crs: int = 3413,
                  resampling_alg: str = 'average') -> dict:
    """
    Load ICE Mask From BedMachine v.4.0
    :param data_dir: absolute path to project data directory
    :param res: mask resolution
    :param crs: map coordinate reference system
    :param resampling_alg: gdal warp resampling method.
    :return: python dictionary
    """
    # - Load ICE Mask From BedMachine v.4.0
    bed_machine_path = os.path.join(data_dir, 'GIS_Data',
                                    'BedMachineGreenland',
                                    'BedMachineGreenlandIceMaskv4_'
                                    'Petermann_Stereo.tiff')
    bed_machine_path_interp \
        = bed_machine_path.replace('.tiff', f'_res{res}m.tiff')
    if not os.path.isfile(bed_machine_path_interp):
        virtual_warp_rio(bed_machine_path, bed_machine_path_interp,
                         res=res, crs=crs, method=resampling_alg,
                         dtype='uint8')
    ice_mask_input = load_dem_tiff(bed_machine_path_interp)
    ice_mask = ice_mask_input['data']
    ice_x_coords = ice_mask_input['x_coords']
    ice_y_coords = ice_mask_input['y_coords']
    ice_x_centroids = ice_mask_input['x_centroids']
    ice_y_centroids = ice_mask_input['y_centroids']

    return {'ice_mask': ice_mask, 'x_coords': ice_x_coords,
            'y_coords': ice_y_coords, 'x_centroids': ice_x_centroids,
            'y_centroids': ice_y_centroids}


def compute_melt_rate(data_dir: str, dem_1_p: str, dem_2_p: str, out_dir: str,
                      smooth: bool = False, w_size: int = 0, res: int = 150,
                      grid_method: str = 'nearest', bm_loc: str = 'midpoint',
                      area_thresh: int = 2e9, sample_bm=False) -> None:
    # - Parameters and Constants
    ref_crs = 3413  # - coordinate reference system
    rho_ice = 0.9167  # - density of ice g/cm3
    rho_h20 = 1.028  # - density of saltwater g/cm3

    # - Default Resampling Algorithm
    resampling_alg = 'average'

    # - Velocity Domain
    v_domain = 'Petermann_Domain_Velocity_Stereo_tdx'
    v_smooth = False        # - smooth velocity maps
    v_domain_buffer = 5e3   # - velocity crop buffer in meters
    v_smooth_mode = 'ave'   # - velocity smoothing filter (average/median)
    v_smooth_size = 11      # - velocity filter smoothing filter size
    # - Velocity Interpolation Method interpolation - Time domain
    v_t_interp = 'bilinear'

    # - Clip Output Raster
    clip_shp_mask_path = os.path.join(data_dir, 'GIS_Data',
                                      'Petermann_features_extraction',
                                      'Petermann_iceshelf_clipped_epsg3413.shp')

    # - Calculate Basal Melt Smoothing filter Kernel size
    if smooth:
        # - if a smoothing option has been selected but
        # - no radius value has been indicated, use a default
        # - value of 1000 meters.
        if w_size == 0:
            w_size = 1000

    smth_kernel_size = int(w_size/res)
    if smth_kernel_size % 2 == 0:
        # - Kernel Size must be an odd number
        smth_kernel_size += 1
    # - DEM1 - First DEM in temporal order
    dem_1_name = dem_1_p.split('/')[-1]
    # - DEM2 - First DEM in temporal order.
    dem_2_name = dem_2_p.split('/')[-1]

    # - Starting and Ending Date of the considered period of time
    f_date = dem_1_name.replace('_tdemx_mosaic.tiff', '')
    l_date = dem_2_name.replace('_tdemx_mosaic.tiff', '')
    t_start = [int(x) for x in f_date.split('-')]
    t_end = [int(x) for x in l_date.split('-')]
    # -
    t00 = datetime.datetime(*t_start)
    t11 = datetime.datetime(*t_end)

    # - Calculate delta time between two dates
    delta_time = calculate_n_month(t00, t11)
    delta_time_yr = delta_time['delta_time'].days / 365.25
    n_months = delta_time['n_months']
    dates_list = delta_time['dates_list']
    # - output file name
    out_file_name = f_date + '-' + l_date

    # - Import SMB estimates from RACMO2.3
    smb_data = load_smb_data(t00, t11, data_dir, ref_crs=ref_crs)
    smb_dhdt = smb_data['delta_h_grid']
    smb_x_coords = smb_data['x_coords']
    smb_y_coords = smb_data['y_coords']

    # - SMB Domain Mesh-grid
    smb_m_xx, smb_m_yy = np.meshgrid(smb_x_coords, smb_y_coords)

    # - Load Ice Mask from Bedmachine 4.0
    ice_mask_input = load_ice_mask(data_dir, res=res, crs=ref_crs,
                                   resampling_alg=resampling_alg)
    ice_mask = ice_mask_input['ice_mask']
    ice_x_coords = ice_mask_input['x_coords']
    ice_y_coords = ice_mask_input['y_coords']

    # - Load Digital Elevation Models -> DEM1(Date) < DEM2(Date)
    # - DEM1
    dem_1_in = load_dem_tiff(dem_1_p)
    dem_1 = dem_1_in['data']
    dem_1[dem_1 == dem_1_in['nodata']] = np.nan

    # - raster upper - left and lower - right corners
    ul_corner_1 = dem_1_in['ul_corner']
    lr_corner_1 = dem_1_in['lr_corner']
    dem_1_minx = ul_corner_1[0]
    dem_1_miny = lr_corner_1[1]
    dem_1_maxx = lr_corner_1[0]
    dem_1_maxy = ul_corner_1[1]

    # - DEM2
    dem_2_in = load_dem_tiff(dem_2_p)
    dem_2 = dem_2_in['data']
    dem_2[dem_2 == dem_2_in['nodata']] = np.nan

    # - raster upper - left and lower - right corners
    ul_corner_2 = dem_2_in['ul_corner']
    lr_corner_2 = dem_2_in['lr_corner']
    dem_2_minx = ul_corner_2[0]
    dem_2_miny = lr_corner_2[1]
    dem_2_maxx = lr_corner_2[0]
    dem_2_maxy = ul_corner_2[1]

    # - Create an X/Y grid containing both DEMs
    minx = np.min([dem_1_minx, dem_2_minx])
    miny = np.min([dem_1_miny, dem_2_miny])
    maxx = np.max([dem_1_maxx, dem_2_maxx])
    maxy = np.max([dem_1_maxy, dem_2_maxy])

    # - Difference domain coordinate axes
    d_diff_vect_x = np.arange(minx, maxx + 1, res)
    d_diff_vect_y = np.arange(miny, maxy + 1, res)

    # - Create the difference domain coordinates grids
    m_xx, m_yy = np.meshgrid(d_diff_vect_x, d_diff_vect_y)
    d_diff_shape = m_xx.shape

    # - crop Ice Mask
    ice_ind_x = np.where((ice_x_coords >= d_diff_vect_x[0])
                         & (ice_x_coords <= d_diff_vect_x[-1]))[0]
    ice_ind_y = np.where((ice_y_coords >= d_diff_vect_y[0])
                         & (ice_y_coords <= d_diff_vect_y[-1]))[0]
    ice_ind_xx, ice_ind_yy = np.meshgrid(ice_ind_x, ice_ind_y)
    ice_mask_crop = ice_mask[ice_ind_yy, ice_ind_xx]

    # - Add DEM1 to difference domain grid [Older DEM1]
    d_dem_1 = np.full(d_diff_shape, np.nan, dtype=np.float32)
    ind_x_1 = np.where((d_diff_vect_x >= dem_1_minx)
                       & (d_diff_vect_x < dem_1_maxx))[0]
    ind_y_1 = np.where((d_diff_vect_y >= dem_1_miny)
                       & (d_diff_vect_y < dem_1_maxy))[0]
    ind_xx, ind_yy = np.meshgrid(ind_x_1, ind_y_1)

    d_dem_1[ind_yy, ind_xx] = dem_1  # - no need to rotate
    # - Find grid point with no valid elevation data
    ind_nan = np.isnan(d_dem_1)

    # - Add DEM2 to difference domain grid [Later DEM]
    d_dem_2 = np.full(d_diff_shape, np.nan, dtype=np.float32)
    ind_x_2 = np.where((d_diff_vect_x >= dem_2_minx)
                       & (d_diff_vect_x < dem_2_maxx))[0]
    ind_y_2 = np.where((d_diff_vect_y >= dem_2_miny)
                       & (d_diff_vect_y < dem_2_maxy))[0]
    ind_xx, ind_yy = np.meshgrid(ind_x_2, ind_y_2)
    d_dem_2[ind_yy, ind_xx] = dem_2  # - no need to rotate

    # - Find DEM1 finite values on the difference grid
    ind_finite = np.isfinite(d_dem_2)

    # - Compute overlapping area among the two DEMs
    d_dem_1_mask = np.full(d_diff_shape, np.nan, dtype=np.float32)
    d_dem_2_mask = np.full(d_diff_shape, np.nan, dtype=np.float32)
    d_dem_1_mask[np.isfinite(d_dem_1)] = 1
    d_dem_2_mask[np.isfinite(d_dem_2)] = 1
    common_grid = d_dem_1_mask * d_dem_2_mask
    common_area = np.nansum(common_grid) * (res * res)

    if common_area < area_thresh:
        # - if overlapping ares < area_thresh, skip this comparison.
        # - Use this threshold to try to consider only overlapping DEMs
        # - that cover most of the ice shelf.
        return

    # - Flatten DEM2 to calculate temporal shift for every grid point
    # - NOTE: Particles are propagated back in time, therefore, the coordinates
    # -       selected below, represent the particles final locations.
    d_dem_2_flat = d_dem_2[ind_finite].ravel()
    m_xx_flat = m_xx[ind_finite].ravel()
    m_yy_flat = m_yy[ind_finite].ravel()

    # - Initialize sample points array
    s_points = None
    # - Initialize sample points history array
    s_points_hist = None
    # - Initialize ice elevation history
    ice_elev_hist = None
    # - Initialize velocity divergence history
    vel_div_hist = None

    # - Start Velocity-based DEMs co-registration
    for mnth in range(n_months):
        r_date = dates_list[mnth]
        year = r_date.year
        month = r_date.month
        day = r_date.day

        if year > 2013:
            # - Load Interpolate velocity Map
            # - If selected, apply smoothing filter to the interpolated maps.
            if v_t_interp == 'bilinear':
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
        grid_vx = v_map['vx_out']
        grid_vy = v_map['vy_out']
        v_x_axis = v_map['x']
        v_y_axis = v_map['y']
        v_map_x_mm = v_map['m_xx']
        v_map_y_mm = v_map['m_yy']

        # - Convert yearly velocity [m/yr] to monthly velocity [m/month].
        grid_vx_m = grid_vx / 12.
        grid_vy_m = grid_vy / 12.

        # - Multiply velocity components per -1 to propagate
        # - ice particles location back in time
        grid_vx_m *= -1
        grid_vy_m *= -1

        # - calculate loaded velocity divergence - with velocity in m/year
        vel_div = divergence(v_x_axis, v_y_axis, grid_vx, grid_vy)['div_v']

        # - Crop velocity field around the DEM difference domain
        try:
            ind_x_v = np.where((v_x_axis >= minx - v_domain_buffer)
                               & (v_x_axis < maxx + v_domain_buffer))[0]
            ind_y_v = np.where((v_y_axis >= miny - v_domain_buffer)
                               & (v_y_axis < maxy + v_domain_buffer))[0]

            ind_xx_v, ind_yy_v = np.meshgrid(ind_x_v, ind_y_v)

            # -
            v_x_ice_m_flat = grid_vx_m[ind_yy_v, ind_xx_v].ravel()
            v_y_ice_m_flat = grid_vy_m[ind_yy_v, ind_xx_v].ravel()
            v_map_x_mm_flat = v_map_x_mm[ind_yy_v, ind_xx_v].ravel()
            v_map_y_mm_flat = v_map_y_mm[ind_yy_v, ind_xx_v].ravel()
            vel_div_flat = vel_div[ind_yy_v, ind_xx_v].ravel()

        except IndexError:
            # - In the case the default buffer value exceeds the velocity
            # - map domain, use a buffer value equal to 0.
            v_domain_buffer = 0
            ind_x_v = np.where((v_x_axis >= minx - v_domain_buffer)
                               & (v_x_axis < maxx + v_domain_buffer))[0]
            ind_y_v = np.where((v_y_axis >= miny - v_domain_buffer)
                               & (v_y_axis < maxy + v_domain_buffer))[0]
            ind_xx_v, ind_yy_v = np.meshgrid(ind_x_v, ind_y_v)
            # -
            v_x_ice_m_flat = grid_vx_m[ind_yy_v, ind_xx_v].ravel()
            v_y_ice_m_flat = grid_vy_m[ind_yy_v, ind_xx_v].ravel()
            v_map_x_mm_flat = v_map_x_mm[ind_yy_v, ind_xx_v].ravel()
            v_map_y_mm_flat = v_map_y_mm[ind_yy_v, ind_xx_v].ravel()
            vel_div_flat = vel_div[ind_yy_v, ind_xx_v].ravel()

        # - Assign glacier elevation values to a specific grid cell
        combined_x_y_arrays = np.dstack([v_map_x_mm_flat,
                                         v_map_y_mm_flat])[0]

        if mnth == 0:
            # - Particles location array
            s_points = np.zeros([len(m_xx_flat), 2])
            s_points[:, 0] = m_xx_flat.squeeze()
            s_points[:, 1] = m_yy_flat.squeeze()

            # - Particles location history array
            s_points_hist = np.zeros([n_months + 1, len(m_xx_flat), 2])
            # - Particles Elevation history array
            ice_elev_hist = np.zeros([n_months + 1, len(m_xx_flat)])
            # - Ice Velocity Divergence history array
            vel_div_hist = np.zeros([n_months + 1, len(m_xx_flat)])

        # - For each of the considered particles,
        # - find the closest grid cell on the velocity domain.
        indexes = do_kdtree(combined_x_y_arrays, s_points) - 1

        # - Save Updated Particle Coordinates.
        s_points_hist[mnth, :, 0] = s_points[:, 0]
        s_points_hist[mnth, :, 1] = s_points[:, 1]
        # - Save Ice Elevation for the considered month
        ice_elev_hist[mnth, :] = d_dem_2_flat
        # - Compute ice velocity divergence a
        vel_div_hist[mnth, :] = vel_div_flat[indexes]

        # - Update Particle coordinates using monthly interpolated velocities
        s_points[:, 0] += v_x_ice_m_flat[indexes]
        s_points[:, 1] += v_y_ice_m_flat[indexes]

    # - Extract intermediate coordinates for each ice particle
    s_mid_points = np.zeros([len(m_xx_flat), 2])
    s_mid_points[:, 0] = s_points_hist[int(n_months/2), :, 0]
    s_mid_points[:, 1] = s_points_hist[int(n_months/2), :, 1]

    # - Interpolate Shifted Elevation Data on the Dem Difference Domain
    # - observations. See above.
    grid_z0 = griddata(s_points, d_dem_2_flat, (m_xx, m_yy),
                       method=grid_method)
    grid_z0[ind_nan] = np.nan

    # - 1) Calculate difference between DEMs
    # - 2) Convert Dh/Dt in meters/year by dividing by
    #      time separating the two DEMs expressed in years.
    dem_diff_lagrangian = (grid_z0 - d_dem_1) / delta_time_yr
    dem_diff_lagrangian[ice_mask_crop == 0] = np.nan

    # - Sample Elevation and Elevation change at their first location
    # - before applying the backward propagation. Find the closest pixel on
    # - the later DEM (DEM2) grid.

    # - Flatten difference domain coordinates to apply kd-tree
    combined_x_y_arrays = np.dstack([m_xx.ravel(), m_yy.ravel()])[0]
    # - s_points - last coordinates of the ice particles.
    indexes = do_kdtree(combined_x_y_arrays, s_points)

    # - convert the sampled dhdt from meters/year to meters/month
    dhdt_lagr = dem_diff_lagrangian.ravel()[indexes] / 12.
    # - Generate particles elevation change ramp.
    dhdt_ramp = (np.tile(np.arange(n_months + 1),
                         (len(dhdt_lagr), 1)).T * dhdt_lagr)
    # - Sum temporal elevation changes estimated above to the
    # - particle initial elevation values (extracted from DEM2).
    ice_elev_hist -= dhdt_ramp

    # - Compute integrated ice divergence in meters/year
    # - see Shean et., al 2019
    i_delta_v = np.sum(ice_elev_hist * vel_div_hist, axis=0) / n_months

    # - Interpolate the final dhdt and ice divergence values on the
    # - dem difference grid.
    ice_div = griddata(s_points, i_delta_v, (m_xx, m_yy), method=grid_method)
    ice_div[ind_nan] = np.nan

    # - Sample SMB average change in meters of ice equivalent at the
    # - intermediate location of each ice particles.
    combined_x_y_arrays = np.dstack([smb_m_xx.ravel(), smb_m_yy.ravel()])[0]
    # - s_mid_points - intermediate coordinates of the ice particles.
    indexes_smb = do_kdtree(combined_x_y_arrays, s_mid_points)
    smb_sample = smb_dhdt.ravel()[indexes_smb]
    # - Interpolate SMB dhdt - No need to use bi-linear interpolation in this
    # - case given the coarse resolution os SMB data [1 km]
    smb_f_grid = griddata(s_mid_points, smb_sample, (m_xx, m_yy),
                          method='nearest')
    smb_f_grid[ind_nan] = np.nan
    # - Calculate Melt-rate - see Shean et al. 2019 for the
    # - formula derivation.
    melt_rate = (-1 * (rho_h20/(rho_h20 - rho_ice))
                 * (dem_diff_lagrangian + ice_div)) + smb_f_grid

    if bm_loc == 'along_flow':
        # - Along FLow Redistribution Method
        # - Sample Melt rate at the final location of each ice particle
        combined_x_y_arrays = np.dstack([m_xx.ravel(), m_yy.ravel()])[0]
        # - use kd-tree to find the closest location to each ice particle
        indexes_mlt = do_kdtree(combined_x_y_arrays, s_points)
        mlt_sample = np.flipud(melt_rate).ravel()[indexes_mlt]

        mlt_sample_tra_grid_med \
            = np.zeros([n_months, len(d_diff_vect_y), len(d_diff_vect_x)])

        for t in range(n_months):
            # - Interpolate Melt Rate
            s_points_temp = np.zeros(s_points.shape)
            s_points_temp[:, 0] = s_points_hist[t, :, 0]
            s_points_temp[:, 1] = s_points_hist[t, :, 1]
            mlt_sample_tra_grid = griddata(s_points_temp, mlt_sample,
                                           (m_xx, m_yy),
                                           method=grid_method)
            mlt_sample_tra_grid[ind_nan] = np.nan
            mlt_sample_tra_grid_med[t, :, :] = np.flipud(mlt_sample_tra_grid)

        # - Final Melt Rate Estimate
        melt_rate = np.median(mlt_sample_tra_grid_med, axis=0)

    elif bm_loc == 'midpoint':
        # - Midpoint Redistribution Method
        # - Sample Melt rate at the final location of each ice particle
        combined_x_y_arrays = np.dstack([m_xx.ravel(), m_yy.ravel()])[0]
        # - use kd-tree to find the closest location to each ice particle
        indexes_mlt = do_kdtree(combined_x_y_arrays, s_points)
        mlt_sample = melt_rate.ravel()[indexes_mlt]

        s_points_temp = np.zeros(s_mid_points.shape)
        s_points_temp[:, 0] = s_mid_points[:, 0]
        s_points_temp[:, 1] = s_mid_points[:, 1]
        mlt_sample_tra_grid = griddata(s_points_temp, mlt_sample,
                                       (m_xx, m_yy), method=grid_method)
        mlt_sample_tra_grid[ind_nan] = np.nan
        # - Final Melt Rate Estimate
        melt_rate = mlt_sample_tra_grid

    # - Mask grid points not covered by ice
    melt_rate[np.isnan(ice_mask_crop)] = np.nan

    if smooth:
        # - Smooth Estimated Melt Rate using a moving average filter
        # - with the selected kernel size.
        ave_filter = np.ones((smth_kernel_size, smth_kernel_size))
        melt_rate\
            = signal.convolve2d(melt_rate, ave_filter,
                                mode='same') / np.sum(ave_filter)

    # - save estimated basal melt rate in GeoTiff format
    out_path_bme_temp = os.path.join(out_dir, out_file_name + '_temp.tiff')
    save_raster(np.float32(melt_rate).copy(), res,
                d_diff_vect_x.copy(), d_diff_vect_y.copy(),
                out_path_bme_temp, ref_crs)

    out_path_bme = os.path.join(out_dir, out_file_name + '.tiff')
    clip_raster(out_path_bme_temp, clip_shp_mask_path, out_path_bme)
    # - Delete Temporary Output
    os.remove(out_path_bme_temp)

    # - sample the obtained melt rate estimate along the defined longitudinal
    # - and transverse profile.
    if sample_bm:
        try:
            sample_raster_petermann_no_fig(data_dir, out_path_bme,
                                           ref_crs=ref_crs)
        except TypeError:
            return


def main():
    parser = argparse.ArgumentParser(
        description="""Compute Ice Shelf basal melt rate from a stack
        of digital elevation models from the TanDEM-X mission in a
        Lagrangian framework."""
    )

    # - Absolute Path to directory containing input data.
    default_dir = os.path.join('/', 'Volumes', 'Extreme Pro')
    parser.add_argument('--directory', '-D',
                        type=lambda p: os.path.abspath(os.path.expanduser(p)),
                        default=default_dir,
                        help='Project data directory.')

    # - Output directory - Absolute Path
    parser.add_argument('--outdir', '-O',
                        type=str,
                        default=default_dir,
                        help='Output directory.')

    # - Input Data Resolution
    parser.add_argument('--res', '-R', type=int,
                        default=150,
                        help='Input raster resolution.')

    # - First Year of Data to Consider
    parser.add_argument('--f_year', '-F', type=int,
                        default=2011,
                        help='First Year to consider.')

    # - Last Year of Data to Consider
    parser.add_argument('--l_year', '-L', type=int,
                        default=2021,
                        help='Last Year to consider.')

    # - Delta Time - Time separation in years between considered DEM pairs.
    parser.add_argument('--delta_time', '-T', type=int,
                        default=2,
                        help='Distance in years between DEMs to compare.')

    # - Apply Spatial Smoothing to Melt Rate Map
    parser.add_argument('--smooth', action='store_true',
                        help='Smooth Melt Rate estimates using a moving average'
                             'filter')

    # - Spatial Smoothing: Kernel Size
    parser.add_argument('--w_size', '-W', type=int, default=0,
                        help='Smoothing Window Size [in meters]')

    # - Velocity TEMPORAL Interpolation Method
    parser.add_argument('--method', '-M', type=str, default='bilinear',
                        choices=['bilinear', 'nearest'],
                        help='Velocity TEMPORAL Interpolation Method.')

    # - Smooth Input velocity Map - Spatial Domain - Not Needed.
    parser.add_argument('--v_smooth', '-V', action='store_true',
                        help='Smooth Input velocity Maps.')

    # - Melt Rate redistribution method - USE midpoint.
    parser.add_argument('--bm_loc', '-B', default='midpoint',
                        choices=['init_pixel', 'along_flow', 'midpoint'],
                        help='Basal melt redistribution strategy.')

    # - Sample Melt Rate
    parser.add_argument('--sample_bm', action='store_true',
                        help='Sample basal melt rate estimate.')

    # - Sample Only Existing Estimates
    parser.add_argument('--sample_only', action='store_true',
                        help='Sample previously calculated basal'
                             ' melt rate estimate.')

    # - Griddata interpolation Method - Used when shifting DEMs
    parser.add_argument('--grid_method', '-G', default='nearest',
                        choices=['bilinear', 'nearest'],
                        help='Griddata interpolation Method.')

    # - Number of simultaneous processes
    parser.add_argument('--np', '-N',
                        type=int, default=os.cpu_count()-2,
                        help='Number of Parallel Processes.')

    args = parser.parse_args()

    # - Parameters and Constants
    ref_crs = 3413              # - coordinate reference system
    resampling_alg = 'average'  # - Raster resampling algorithm
    gdal_binding = 'rio'        # - Python GDAL Binding
    # - Reference Temporal Interval
    t00 = datetime.datetime(args.f_year, 1, 1)
    t11 = datetime.datetime(args.l_year, 12, 31)
    if t00 >= t11:
        raise ValueError('f_year must be smaller than l_year')

    # - Create Output directory
    out_dir = create_dir(os.path.join(args.outdir, 'TanDEM-X'),
                         'Basal_Melt_Rate_Lagrangian_Framework')
    # - Basal melt redistribution approach
    out_dir = create_dir(out_dir, args.bm_loc)
    # - create a subdirectory for final estimate with smoothing filter applied
    if not args.smooth:
        out_dir \
            = create_dir(out_dir, f'bm_nosmooth_delta_time={args.delta_time}')
    else:
        out_dir \
            = create_dir(out_dir, f'bm_average_{args.w_size}'
                                  f'_delta_time={args.delta_time}')

    if args.sample_only:
        # - Sample previously calculated basal melt estimates
        bm_list = [os.path.join(out_dir, x) for x in os.listdir(out_dir)
                   if x.endswith('.tiff') and not x.startswith('.')]
        if bm_list:
            for bm_raster in bm_list:
                sample_raster_petermann_no_fig(args.directory, bm_raster,
                                               ref_crs=ref_crs)
        else:
            print('# - No melt rate estimates found.')

    else:
        # - Absolute Path to directory containing TanDEM-X digital models.
        # - NOTE: Elevation data must have been converted from elevation with
        # -      respect to the standard Ellipsoid WGS84 to elevation with
        # -      respect to the mean sea level(give a selected reference geoid).
        # -      The corrected elevation values must also include
        # -      the effects of:
        # -       - Inverse Barometer Effect;
        # -       - Ocean Mean Dynamic Topography;
        # -       - Ice Shelf Shift due to Tidal cycles;

        t_dem_dir = os.path.join(args.directory, 'TanDEM-X',
                                 'Petermann_Glacier_out', 'Mosaics',
                                 f'Petermann_Glacier_Mosaics_EPSG-{ref_crs}_'
                                 f'res-{args.res}_ralg-{resampling_alg}'
                                 f'_{gdal_binding}_amsl_corrected_poly0')

        # - Load list of DEMs that should be excluded from the evaluation
        dem_2_skip_list = dem_2_skip()

        # - list input directory content
        input_dir_list = sorted([os.path.join(t_dem_dir, x)
                                 for x in os.listdir(t_dem_dir)
                                 if not x.startswith('.')
                                 and x.endswith('tiff')])
        # - Extract time-tag from each daily mosaic file name
        dem_name_list = []
        dem_path_list = []
        date_list = []
        day_list = []
        month_list = []
        year_list = []

        for dt in input_dir_list:
            f_name = dt.split('/')[-1].replace('_tdemx_mosaic.tiff', '')
            if f_name in dem_2_skip_list:
                continue
            date_str = f_name.split('-')
            dem_name_list.append(f_name)
            dem_path_list.append(dt)
            day_list.append(int(date_str[-2]))
            month_list.append(int(date_str[1]))
            year_list.append(int(date_str[0]))
            date_list.append(datetime.datetime(int(date_str[0]),
                                               int(date_str[1]),
                                               int(date_str[2])))
        # - Create a Pandas Dataframe to index the available DEMs
        data_frame_cont = {'name': dem_name_list, 'path': dem_path_list,
                           'year': year_list, 'month': month_list,
                           'day': day_list, 'time': date_list}
        dem_df = pd.DataFrame(data_frame_cont)

        # - Use only Mosaics available during the selected reference period
        # - of time.
        dem_df = dem_df.loc[(dem_df['time'] > t00) & (dem_df['time'] <= t11)]
        print(f'# - Selected Reference Period: {args.f_year} - {args.l_year}')
        print(f'# - Available DEMs: {dem_df.shape[0]}')

        # - Iterate through the pandas index to determine the candidate DEM
        # - pairs that will be used in the melt rate calculation based on the
        # - selected criteria.
        # - By default DEMS acquired during the same month at two years
        # - distance are included in the comparison.
        n_pairs = 0
        ave_comb = []
        # - create new Dataframe containing a row for each DEM pair
        dem_1 = []
        dem_2 = []
        date_1 = []
        date_2 = []
        for _, row in dem_df.iterrows():
            r_year = row['year']
            r_month = row['month']
            sample_pair = dem_df.query(f'year == {r_year+args.delta_time} '
                                       f'& month == {r_month}')
            n_pairs += sample_pair.shape[0]
            ave_comb.append(sample_pair.shape[0])

            for _, s_row in sample_pair.iterrows():
                dem_1.append(row['path'])
                dem_2.append(s_row['path'])
                date_1.append(row['time'])
                date_2.append(s_row['time'])

        print(f'# - Number of DEMs combinations based on the '
              f'considered selection criteria: {n_pairs}')
        print(f'# - Average Number of Comparisons per DEM: '
              f'{np.average(ave_comb)}')

        # - Initialize DEM comparison dataframe
        dem_pair_dict = {'dem_1': dem_1, 'dem_2': dem_2,
                         'date_1': date_1, 'date_2': date_2}
        dem_pair_df = pd.DataFrame(dem_pair_dict)
        # -
        dem_1_list = []
        dem_2_list = []
        for _, d_row in dem_pair_df.iterrows():
            dem_1_list.append(d_row['dem_1'])
            dem_2_list.append(d_row['dem_2'])

        # - Project data directory
        data_dir_list = [args.directory] * len(dem_1_list)
        # - output directory
        out_dir_list = [out_dir] * len(dem_1_list)

        # - define input data batch
        p_proc_input = zip(data_dir_list[:], dem_1_list[:],
                           dem_2_list[:], out_dir_list[:])

        print('# - Computing Basal Melt Rate.')
        from tqdm import tqdm
        with Pool(args.np) as p:
            kwargs = {'bm_loc': args.bm_loc, 'grid_method': args.grid_method,
                      'res': args.res, 'smooth': args.smooth,
                      'w_size': args.w_size,
                      'sample_bm': args.sample_bm
                      }
            map_func = partial(compute_melt_rate, **kwargs)
            r = list(tqdm(p.starmap(map_func, p_proc_input), total=30))


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f'# - Computation Time: {end_time - start_time}')
