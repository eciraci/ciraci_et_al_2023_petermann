#!/usr/bin/env python
u"""
velocity_map_time_interpolation.py
Written by Enrico Ciraci' (11/2021)

Load Annual Velocity Maps from Rignot et al. 2009.

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    pandas: Python open source data analysis and manipulation tool
           https://pandas.pydata.org
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.ioo
    datetime: Basic date and time types
           https://docs.python.org/3/library/datetime.html#module-datetime
    matplotlib: Visualization with Python
           https://matplotlib.org

UPDATE HISTORY:

"""
# - Python Dependencies
from __future__ import print_function
import os
import sys
from datetime import datetime
import numpy as np
from scipy import signal
import pandas as pd
import rasterio


def load_velocity_map(year: int, month: int, day: int,
                      args_path: str,
                      domain: str = 'ROI_Velocity_Stereo',
                      res: int = 150,
                      smooth: bool = False,
                      smooth_mode: str = 'ave',
                      smooth_w_size: int = 11,
                      verbose: bool = True) -> dict:
    """
    Load Ice Velocity Reference Map based on the provided input date.
    - > Extrapolate the ice velocity at the selected date linearly interpolating
        two consecutive yearly maps
    :param year: input date - year
    :param month: input date - month
    :param day: input date - day
    :param args_path: absolute path to directory containing annual velocity maps
    :param domain: velocity map domain for the region of interest [ROI]
    :param res: velocity map resolution
    :param smooth: smooth interpolate velocity field
    :param smooth_mode: smoothing filter strategy: average(ave)/median
    :param smooth_w_size: smoothing filter size in pixels
    :param verbose: print function outputs on screen.
    :return: dict()
    """
    date_ref = datetime(year=year, month=month, day=day)
    if verbose:
        print(f'# - Reference Date: {date_ref}')
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args_path, domain)

    # - Create Velocity Maps Index
    # - List Input directory content
    dir_list = sorted([os.path.join(data_dir, x)
                       for x in os.listdir(data_dir)
                       if x.endswith('.nc') and not x.startswith('.')])
    # - initialize DEMs index dataframe.
    data_index = []
    for file_path in dir_list:
        # - Read Yearly Velocity Map
        velocity_f_name = str(file_path.split('/')[-1].replace('.nc', ''))
        # - extract DEM reference dates
        year_1 = int(velocity_f_name[4:8])
        month_1 = int(velocity_f_name[9:11])
        day_1 = int(velocity_f_name[12:14])

        year_2 = int(velocity_f_name[15:19])
        month_2 = int(velocity_f_name[20:22])
        day_2 = int(velocity_f_name[23:25])
        # - Add DEM info to dataframe index
        data_index.append([velocity_f_name, year_1, month_1, day_1,
                           year_2, month_2, day_2])
    # Create the pandas DataFrame
    v_maps_index \
        = pd.DataFrame(data_index, columns=['Name', 'Year_1',
                                            'Month_1', 'Day_1',
                                            'Year_2', 'Month_2',
                                            'Day_2'])

    # - calculate time-delta between input date and the date reference date
    # - of each of the DEMs listed by the DEMs index dataframe.
    delta_days = []
    for _, row in v_maps_index.iterrows():
        # - Ice velocity maps from Rignot et al. 2009 cover a one-year
        # - time frame going from July of year n and June of year n+1.
        # - Consider January 1st of year n+1 as the reference date.
        row_date = datetime(year=row['Year_2'], month=1, day=1)
        delta_days.append(np.abs((date_ref - row_date).days))
    delta_days_sorted = sorted(delta_days)
    w1_index = delta_days.index(delta_days_sorted[0])
    w2_index = delta_days.index(delta_days_sorted[1])
    w1_f_name = v_maps_index.iloc[w1_index, :]['Name']
    w2_f_name = v_maps_index.iloc[w2_index, :]['Name']
    if verbose:
        print('# - Files Selected for the interpolation:')
        print('# - ' + w1_f_name)
        print('# - ' + w2_f_name)
    # - velocity map_name
    v_interp_name = w1_f_name+'\n'+w2_f_name

    # - Load Velocity Maps
    # - V-MAP1 - velocity map
    f_name_1 = os.path.join(data_dir, f'interp_vmaps_res{res}',
                            w1_f_name, w1_f_name
                            + f'-rio_EPSG-3413_res-{res}_average.tiff')
    with rasterio.open(f_name_1, mode="r+") as src:
        vx_1 = src.read(1).astype(src.dtypes[0])  # - read band #1 - Vx
        vy_1 = src.read(2).astype(src.dtypes[0])  # - read band #2 - Vy
        if src.transform.e < 0:
            vx_1 = np.flipud(vx_1)
            vy_1 = np.flipud(vy_1)

    # - V-MAP2
    f_name_2 = os.path.join(data_dir, f'interp_vmaps_res{res}',
                            w2_f_name, w2_f_name
                            + f'-rio_EPSG-3413_res-{res}_average.tiff')
    with rasterio.open(f_name_2, mode="r+") as src:
        vx_2 = src.read(1).astype(src.dtypes[0])  # - read band #1 - Vx
        vy_2 = src.read(2).astype(src.dtypes[0])  # - read band #2 - Vy
        if src.transform.e < 0:
            vx_2 = np.flipud(vx_2)
            vy_2 = np.flipud(vy_2)
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res
        # - compute x- and y-axis coordinates
        x_2 = np.arange(ul_corner[0], lr_corner[0], grid_res[0])
        y_2 = np.arange(lr_corner[1], ul_corner[1], grid_res[1])
        src_2_minx = np.min(x_2)
        src_2_miny = np.min(y_2)
        src_2_maxx = np.max(x_2)
        src_2_maxy = np.max(y_2)

    v_vect_x = np.arange(src_2_minx, src_2_maxx + 1, res)
    v_vect_y = np.arange(src_2_miny, src_2_maxy + 1, res)

    # - create difference domain coordinates grids
    m_xx, m_yy = np.meshgrid(v_vect_x, v_vect_y)
    # - extrapolate velocity at the selected date as the inverse
    # - distance weighted interpolation of the DEMs
    w_1 = 1 - (delta_days_sorted[0]
               / (delta_days_sorted[0] + delta_days_sorted[1]))
    w_2 = 1 - (delta_days_sorted[1]
               / (delta_days_sorted[0] + delta_days_sorted[1]))
    vx_out = (vx_1 * w_1) + (vx_2 * w_2)
    vy_out = (vy_1 * w_1) + (vy_2 * w_2)
    vx_out[np.isnan(vx_out)] = 0.
    vy_out[np.isnan(vy_out)] = 0.

    if smooth:
        # - if selected, smooth the interpolated velocity field.
        w_size = smooth_w_size
        if smooth_mode in ['average', 'ave']:
            # - Use w_size*w_size Average filter
            ave_filter = np.ones((w_size, w_size))
            vx_out = signal.convolve2d(vx_out, ave_filter,
                                       mode='same')/np.sum(ave_filter)
            vy_out = signal.convolve2d(vy_out, ave_filter,
                                       mode='same')/np.sum(ave_filter)
        elif smooth_mode in ['median', 'med']:
            # - Use w_size*w_size Median filter
            vx_out = signal.medfilt2d(vx_out, kernel_size=w_size)
            vy_out = signal.medfilt2d(vy_out, kernel_size=w_size)
        else:
            print(f'# - Unknown Smoothing Strategy selected: {smooth_mode}')
            sys.exit()

    return{'vx_out': vx_out, 'vy_out': vy_out,
           'x': v_vect_x, 'y': v_vect_y, 'm_xx': m_xx, 'm_yy': m_yy,
           'v_interp_name': v_interp_name}


def load_velocity_map_nearest(year: int, month: int, day: int,
                              args_path: str,
                              domain: str = 'ROI_Velocity_Stereo',
                              res: int = 150,
                              smooth: bool = False,
                              smooth_mode: str = 'ave',
                              smooth_w_size: int = 11,
                              verbose: bool = True) -> dict:
    """
    Load Ice Velocity Reference Map based on the provided input date.
    - > Nearest-Neighbor approach.
    - > Input velocity Maps must have been previously interpolated at the
        selected resolution
    :param year: input date - year
    :param month: input date - month
    :param day: input date - day
    :param args_path: absolute path to directory containing annual velocity maps
    :param domain: velocity map domain
    :param res: velocity map resolution
    :param smooth: smooth interpolate velocity field
    :param smooth_mode: smoothing filter strategy: average(ave)/median
    :param smooth_w_size: smoothing filter size in pixels
    :param verbose: print function outputs on screen
    :return: Python Dictionary containing the following keys:
    - vx_out: velocity map - x-component
    - vy_out: velocity map - y-component
    - x: velocity map - x-axis coordinates
    - y: velocity map - y-axis coordinates
    - m_xx: velocity map - x-axis coordinates grid
    - m_yy: velocity map - y-axis coordinates grid
    - v_interp_name: interpolated velocity
    """
    # - Velocity Maps resolution
    args_res = 150
    # -
    date_ref = datetime(year=year, month=month, day=day)
    if verbose:
        print(f'# - Reference Date: {date_ref}')
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args_path, domain)

    # - Create Velocity Maps Index
    # - List Input directory content
    dir_list = sorted([os.path.join(data_dir, x)
                       for x in os.listdir(data_dir)
                       if x.endswith('.nc') and not x.startswith('.')])
    # - initialize DEMs index dataframe.
    data_index = []
    for file_path in dir_list:
        # - Read Yearly Velocity Map
        velocity_f_name = str(file_path.split('/')[-1].replace('.nc', ''))
        # - extract first year reference dates
        year_1 = int(velocity_f_name[4:8])
        month_1 = int(velocity_f_name[9:11])
        day_1 = int(velocity_f_name[12:14])
        # - extract second year reference dates
        year_2 = int(velocity_f_name[15:19])
        month_2 = int(velocity_f_name[20:22])
        day_2 = int(velocity_f_name[23:25])
        # - Add DEM info to dataframe index
        data_index.append([velocity_f_name, year_1, month_1, day_1,
                           year_2, month_2, day_2])
    # Create the pandas DataFrame
    v_maps_index \
        = pd.DataFrame(data_index, columns=['Name', 'Year_1',
                                            'Month_1', 'Day_1',
                                            'Year_2', 'Month_2',
                                            'Day_2'])
    # - calculate time-delta between input date and the date reference date
    # - of each of the DEMs listed by the DEMs index dataframe.
    delta_days = []
    for _, row in v_maps_index.iterrows():
        row_date = datetime(year=row['Year_2'], month=1, day=1)
        delta_days.append(np.abs((date_ref - row_date).days))

    delta_days_sorted = sorted(delta_days)
    w1_index = delta_days.index(delta_days_sorted[0])
    w1_f_name = v_maps_index.iloc[w1_index, :]['Name']
    if verbose:
        print('# - Files Selected for the interpolation:')
        print('# - ' + w1_f_name)

    # - Load Velocity
    f_name = os.path.join(data_dir, f'interp_vmaps_res{res}',
                          w1_f_name, w1_f_name
                          + f'-rio_EPSG-3413_res-{res}_average.tiff')
    with rasterio.open(f_name, mode="r+") as src:
        vx_out = src.read(1).astype(src.dtypes[0])  # - read band #1 - Vx
        vy_out = src.read(2).astype(src.dtypes[0])  # - read band #2 - Vy
        if src.transform.e < 0:
            vx_out = np.flipud(vx_out)
            vy_out = np.flipud(vy_out)
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res
        # - compute x- and y-axis coordinates
        x_1 = np.arange(ul_corner[0], lr_corner[0], grid_res[0])
        y_1 = np.arange(lr_corner[1], ul_corner[1], grid_res[1])
        src_1_minx = np.min(x_1)
        src_1_miny = np.min(y_1)
        src_1_maxx = np.max(x_1)
        src_1_maxy = np.max(y_1)

    # - difference domain coordinate axes
    v_vect_x = np.arange(src_1_minx, src_1_maxx + 1, args_res)
    v_vect_y = np.arange(src_1_miny, src_1_maxy + 1, args_res)
    # - create difference domain coordinates grids
    m_xx, m_yy = np.meshgrid(v_vect_x, v_vect_y)
    vx_out[np.isnan(vx_out)] = 0.
    vy_out[np.isnan(vy_out)] = 0.

    if smooth:
        # - if selected, smooth the interpolated velocity field.
        w_size = smooth_w_size
        if smooth_mode in ['average', 'ave']:
            # - Use w_size*w_size Average filter
            ave_filter = np.ones((w_size, w_size))
            vx_out = signal.convolve2d(vx_out, ave_filter,
                                       mode='same')/np.sum(ave_filter)
            vy_out = signal.convolve2d(vy_out, ave_filter,
                                       mode='same')/np.sum(ave_filter)
        elif smooth_mode in ['median', 'med']:
            # - Use w_size*w_size Median filter
            vx_out = signal.medfilt2d(vx_out, kernel_size=w_size)
            vy_out = signal.medfilt2d(vy_out, kernel_size=w_size)
        else:
            print(f'# - Unknown Smoothing Strategy selected: {smooth_mode}')
            sys.exit()

    return{'vx_out': vx_out, 'vy_out': vy_out,
           'x': v_vect_x, 'y': v_vect_y, 'm_xx': m_xx, 'm_yy': m_yy,
           'v_interp_name': w1_f_name}
