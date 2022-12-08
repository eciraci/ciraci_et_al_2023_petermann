#!/usr/bin/env python
u"""
velocity_map_time_interpolation.py
Written by Enrico Ciraci' (11/2021)

TEST: Evaluate ICE VELOCITY MAPS at the selected date by using an inverse
distance interpolation of velocity maps from the MEASuREs Project

COMMAND LINE OPTIONS:
usage: velocity_map_time_interpolation.py [-h] [--directory DIRECTORY]
                                          [--res RES] [--method METHOD]

TEST: Evaluate ICE VELOCITY MAPS at the selected data by using an inverse
distance interpolation of velocity maps from the MEASuREs Project

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -D DIRECTORY
                        Project data directory.
  --res RES, -R RES     Output raster resolution.
  --method METHOD, -M METHOD
                        Method of interpolation.

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
01/11/2022 - Updated:
     - load_interp_velocity_map/load_interp_velocity_map_nearest
        -> added - process interpolated maps.
     - load_velocity_map/load_velocity_map_nearest
        -> updated - process velocity maps at native resolution.

"""
# - Python Dependencies
from __future__ import print_function
import os
import sys
import argparse
import numpy as np
from scipy import signal
import pandas as pd
import rasterio
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-deep')


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
    cax = divider.new_vertical(size='5%', pad=0.6, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    return cb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""Evaluate ICE VELOCITY at a selected location and date
        by employing annual velocity maps from Rignot et al. 2009. Velocity
        at the selected date is computed by using an inverse distance 
        interpolation"""
    )

    # - Absolute Path to directory containing input data.
    default_dir = os.environ['PYTHONDATA']
    parser.add_argument('--directory', '-D',
                        type=lambda p: os.path.abspath(
                            os.path.expanduser(p)),
                        default=default_dir,
                        help='Project data directory.')
    # - Output X/Y grid spacing.
    parser.add_argument('--res', '-R', type=int,
                        default=50,
                        help='Output raster resolution.')

    # - Method of interpolation.
    parser.add_argument('--method', '-M', type=str,
                        default='bilinear',
                        help='Method of interpolation.')

    args = parser.parse_args()

    # - reference data
    year_ref = 2017
    month_ref = 11
    day_ref = 16
    date_ref = datetime(year=year_ref, month=month_ref, day=day_ref)
    print('# - Reference Date: {}'.format(date_ref))
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args.directory, 'Greenland_Ice_Velocity_MEaSUREs',
                            'Petermann_Domain_Velocity_Stereo',
                            'interp_vmaps_res{}'.format(args.res))

    # - load velocity maps index
    v_maps_index = pd.read_csv(os.path.join(data_dir, 'dem_index.csv'))

    print('# - Available DEMs')
    print(v_maps_index.head())

    # - calculate time-delta between input date and the date reference date
    # - of each of the DEMs listed by the DEMs index dataframe.
    delta_days = list()
    for index, row in v_maps_index.iterrows():
        # - Ice velocity maps from MEaSUREs cover a one year  time frame
        # - going from July of year n and June of year n+1.
        # - Consider January 1st of year n+1 as the reference date.
        row_date = datetime(year=row['Year_2'], month=1, day=1)
        delta_days.append(np.abs((date_ref-row_date).days))

    delta_days_sorted = sorted(delta_days)
    w1_index = delta_days.index(delta_days_sorted[0])
    w2_index = delta_days.index(delta_days_sorted[1])
    w1_f_name = v_maps_index.iloc[w1_index, :]['Name']
    w2_f_name = v_maps_index.iloc[w2_index, :]['Name']
    print('# - Files Selected for the interpolation:')
    print('# - ' + w1_f_name)
    print('# - ' + w2_f_name)

    # - Load DEM1 and DEM2
    # - DEM1
    f_name_1 = os.path.join(data_dir, w1_f_name,
                            w1_f_name
                            + '-rio_EPSG-3413_res-{}_{}.tiff'
                            .format(args.res, args.method))
    with rasterio.open(f_name_1, mode="r+") as src_1:
        vx_1 = src_1.read(1)
        vy_1 = src_1.read(2)
        transform = src_1.transform
        crs = src_1.crs

    # DEM2
    f_name_2 = os.path.join(data_dir, w2_f_name,
                            w2_f_name
                            + '-rio_EPSG-3413_res-{}_{}.tiff'
                            .format(args.res, args.method))
    with rasterio.open(f_name_2, mode="r+") as src_2:
        vx_2 = src_2.read(1)
        vy_2 = src_2.read(2)

    # - extrapolate velocity at the selected date as the inverse
    # - distance weighted interpolation of the DEMs
    vx_interp = (((vx_1*delta_days_sorted[0]) + (vx_2*delta_days_sorted[1]))
                 / (delta_days_sorted[0]+delta_days_sorted[1]))
    vy_interp = (((vy_1*delta_days_sorted[0]) + (vy_2*delta_days_sorted[1]))
                 / (delta_days_sorted[0]+delta_days_sorted[1]))
    vx_interp = np.flipud(vx_interp)
    vy_interp = np.flipud(vy_interp)

    # - Save Velocity Field Components in GeoTiff format.
    # - Vx - band 0
    # - Vy - band 1
    out_path = os.path.join(os.path.expanduser("~"), 'Desktop',
                            'time_interp_test_{}_{}_{}.tiff'
                            .format(year_ref, month_ref, day_ref))
    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=vx_interp.shape[0],
                       width=vx_interp.shape[1], count=2,
                       dtype=vx_interp.dtype, crs=crs,
                       transform=transform,
                       nodata=-9999) as dst:
        dst.write(np.flipud(vx_interp), 1)
        dst.write(np.flipud(vy_interp), 2)

    # - Plot Difference Between Fields used for the Interpolation
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(vx_1-vx_2, vmin=-100, vmax=100, cmap=plt.cm.get_cmap('bwr'))
    add_colorbar(fig, ax, im)
    ax.grid(color='k', linestyle='dotted', alpha=0.3)
    ax.set_title('Vx', weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('y')

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(vy_1-vy_2, vmin=-100, vmax=100, cmap=plt.cm.get_cmap('bwr'))
    add_colorbar(fig, ax, im)
    ax.grid(color='k', linestyle='dotted', alpha=0.3)
    ax.set_title('Vy', weight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('y')

    plt.show()


def load_velocity_map(year: int, month: int, day: int,
                      args_path: str,
                      domain: str = 'Petermann_Domain_Velocity_Stereo',
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
    :param domain: velocity map domain
    :param res: velocity map resolution
    :param smooth: smooth interpolate velocity field
    :param smooth_mode: smoothing filter strategy: average(ave)/median
    :param smooth_w_size: smoothing filter size in pixels
    :param verbose: print function outputs on screen.
    :return: dict()
    """
    date_ref = datetime(year=year, month=month, day=day)
    if verbose:
        print('# - Reference Date: {}'.format(date_ref))
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args_path, 'Greenland_Ice_Velocity_MEaSUREs',
                            domain)

    # - Create Velocity Maps Index
    # - List Input directory content
    dir_list = sorted([os.path.join(data_dir, x)
                       for x in os.listdir(data_dir)
                       if x.endswith('.nc') and not x.startswith('.')])
    # - initialize DEMs index dataframe.
    data_index = list()
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
    delta_days = list()
    for index, row in v_maps_index.iterrows():
        # - Ice velocity maps from MEaSUREs cover a one-year  time frame
        # - going from July of year n and June of year n+1.
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
    # - V-MAP1
    f_name_1 = os.path.join(data_dir, 'interp_vmaps_res{}'.format(res),
                            w1_f_name, w1_f_name
                            + '-rio_EPSG-3413_res-{}_average.tiff'.format(res))
    with rasterio.open(f_name_1, mode="r+") as src:
        vx_1 = src.read(1).astype(src.dtypes[0])  # - read band #1 - Vx
        vy_1 = src.read(2).astype(src.dtypes[0])  # - read band #2 - Vy
        if src.transform.e < 0:
            vx_1 = np.flipud(vx_1)
            vy_1 = np.flipud(vy_1)

    # - V-MAP2
    f_name_2 = os.path.join(data_dir, 'interp_vmaps_res{}'.format(res),
                            w2_f_name, w2_f_name
                            + '-rio_EPSG-3413_res-{}_average.tiff'.format(res))
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
            print('# - Unknown Smoothing Strategy selected: {}'
                  .format(smooth_mode))
            sys.exit()

    return{'vx_out': vx_out, 'vy_out': vy_out,
           'x': v_vect_x, 'y': v_vect_y, 'm_xx': m_xx, 'm_yy': m_yy,
           'v_interp_name': v_interp_name}


def load_velocity_map_nearest(year: int, month: int, day: int,
                              args_path: str,
                              domain: str = 'Petermann_Domain_Velocity_Stereo',
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
    :return: dict()
    """
    # - Velocity Maps resolution
    args_res = 150
    # -
    date_ref = datetime(year=year, month=month, day=day)
    if verbose:
        print('# - Reference Date: {}'.format(date_ref))
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args_path, 'Greenland_Ice_Velocity_MEaSUREs',
                            domain)

    # - Create Velocity Maps Index
    # - List Input directory content
    dir_list = sorted([os.path.join(data_dir, x)
                       for x in os.listdir(data_dir)
                       if x.endswith('.nc') and not x.startswith('.')])
    # - initialize DEMs index dataframe.
    data_index = list()
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
    delta_days = list()
    for index, row in v_maps_index.iterrows():
        row_date = datetime(year=row['Year_2'], month=1, day=1)
        delta_days.append(np.abs((date_ref - row_date).days))

    delta_days_sorted = sorted(delta_days)
    w1_index = delta_days.index(delta_days_sorted[0])
    w1_f_name = v_maps_index.iloc[w1_index, :]['Name']
    if verbose:
        print('# - Files Selected for the interpolation:')
        print('# - ' + w1_f_name)

    # - Load Velocity
    f_name = os.path.join(data_dir, 'interp_vmaps_res{}'.format(res),
                          w1_f_name, w1_f_name
                          + '-rio_EPSG-3413_res-{}_average.tiff'.format(res))
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
            print('# - Unknown Smoothing Strategy selected: {}'
                  .format(smooth_mode))
            sys.exit()

    return{'vx_out': vx_out, 'vy_out': vy_out,
           'x': v_vect_x, 'y': v_vect_y, 'm_xx': m_xx, 'm_yy': m_yy,
           'v_interp_name': w1_f_name}


def load_interp_velocity_map(year: int, month: int, day: int,
                             args_path: str, args_res=50,
                             domain: str = 'Petermann_Domain_Velocity_Stereo',
                             args_method: str = 'bilinear',
                             smooth: bool = False,
                             smooth_mode: str = 'ave',
                             smooth_w_size: int = 11,
                             verbose: bool = True) -> dict:
    """
    Load Ice Velocity Reference Map based on the provided input date.
    - > Input velocity Maps must have been previously interpolated at the
        selected resolution.
    - > Extrapolate the ice velocity at the selected date linearly interpolating
        two consecutive yearly maps
    :param year: input date - year
    :param month: input date - month
    :param day: input date - day
    :param args_path: absolute path to directory containing annual velocity maps
    :param args_res: velocity map resolution
    :param domain: velocity map domain
    :param args_method: interpolation method used to extrapolate ice velocity st
               the selected resolution
    :param smooth: smooth interpolate velocity field
    :param smooth_mode: smoothing filter strategy: average(ave)/median
    :param smooth_w_size: smoothing filter size in pixels
    :param verbose: print function outputs on screen.
    :return: dict()
    """
    date_ref = datetime(year=year, month=month, day=day)
    if verbose:
        print('# - Reference Date: {}'.format(date_ref))
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args_path, 'Greenland_Ice_Velocity_MEaSUREs',
                            domain, 'interp_vmaps_res{}'.format(args_res))

    # - load velocity maps index
    v_maps_index = pd.read_csv(os.path.join(data_dir, 'dem_index.csv'))
    # - calculate time-delta between input date and the date reference date
    # - of each of the DEMs listed by the DEMs index dataframe.
    delta_days = list()
    for index, row in v_maps_index.iterrows():
        # - Ice velocity maps from MEaSUREs cover a one year  time frame
        # - going from July of year n and June of year n+1.
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
    # - interpolated velocity map_name
    v_interp_name = w1_f_name+'\n'+w2_f_name
    # - Load Velocity Maps
    # - V-MAP1
    f_name_1 = os.path.join(data_dir, w1_f_name,
                            w1_f_name
                            + '-rio_EPSG-3413_res-{}_{}.tiff'
                            .format(args_res, args_method))
    with rasterio.open(f_name_1, mode="r+") as src_1:
        vx_1 = src_1.read(1)
        vy_1 = src_1.read(2)

    # - V-MAP2
    f_name_2 = os.path.join(data_dir, w2_f_name,
                            w2_f_name
                            + '-rio_EPSG-3413_res-{}_{}.tiff'
                            .format(args_res, args_method))
    with rasterio.open(f_name_2, mode="r+") as src_2:
        vx_2 = src_2.read(1)
        vy_2 = src_2.read(2)
        # - raster upper - left and lower - right corners
        ul_corner_2 = src_2.transform * (0, 0)
        lr_corner_2 = src_2.transform * (src_2.width, src_2.height)
        src_2_minx = ul_corner_2[0]
        src_2_miny = lr_corner_2[1]
        src_2_maxx = lr_corner_2[0]
        src_2_maxy = ul_corner_2[1]

    # - difference domain coordinate axes
    v_vect_x = np.arange(src_2_minx, src_2_maxx + 1, args_res)
    v_vect_y = np.arange(src_2_miny, src_2_maxy + 1, args_res)

    # - create difference domain coordinates grids
    m_xx, m_yy = np.meshgrid(v_vect_x, v_vect_y)
    # - extrapolate velocity at the selected date as the inverse
    # - distance weighted interpolation of the DEMs
    w_1 = 1 - (delta_days_sorted[0]
               / (delta_days_sorted[0] + delta_days_sorted[1]))
    w_2 = 1 - (delta_days_sorted[1]
               / (delta_days_sorted[0] + delta_days_sorted[1]))
    vx_interp = (vx_1 * w_1) + (vx_2 * w_2)
    vy_interp = (vy_1 * w_1) + (vy_2 * w_2)
    vx_interp = np.flipud(vx_interp)
    vy_interp = np.flipud(vy_interp)

    if smooth:
        # - if selected, smooth the interpolated velocity field.
        w_size = smooth_w_size
        if smooth_mode in ['average', 'ave']:
            # - Use w_size*w_size Average filter
            ave_filter = np.ones((w_size, w_size))
            vx_interp = signal.convolve2d(vx_interp, ave_filter,
                                          mode='same')/np.sum(ave_filter)
            vy_interp = signal.convolve2d(vy_interp, ave_filter,
                                          mode='same')/np.sum(ave_filter)
        elif smooth_mode in ['median', 'med']:
            # - Use w_size*w_size Median filter
            vx_interp = signal.medfilt2d(vx_interp, kernel_size=w_size)
            vy_interp = signal.medfilt2d(vy_interp, kernel_size=w_size)
        else:
            print('# - Unknown Smoothing Strategy selected: {}'
                  .format(smooth_mode))
            sys.exit()

    return{'vx_interp': vx_interp, 'vy_interp': vy_interp,
           'x': v_vect_x, 'y': v_vect_y, 'm_xx': m_xx, 'm_yy': m_yy,
           'v_interp_name': v_interp_name}


def load_interp_velocity_map_nearest(year: int, month: int, day: int,
                                     args_path: str, args_res=50,
                                     domain: str = 'Petermann_Domain_'
                                                   'Velocity_Stereo',
                                     args_method='bilinear',
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
    :param args_res: velocity map resolution
    :param domain: velocity map domain
    :param args_method: interpolation method used to extrapolate ice velocity st
               the selected resolution
    :param smooth: smooth interpolate velocity field
    :param smooth_mode: smoothing filter strategy: average(ave)/median
    :param smooth_w_size: smoothing filter size in pixels
    :param verbose: print function outputs on screen
    :return: dict()
    """
    date_ref = datetime(year=year, month=month, day=day)
    if verbose:
        print('# - Reference Date: {}'.format(date_ref))
    # - path to directory containing velocity data at the selected spatial
    # - resolution.
    data_dir = os.path.join(args_path, 'Greenland_Ice_Velocity_MEaSUREs',
                            domain, 'interp_vmaps_res{}'.format(args_res))

    # - load velocity maps index
    v_maps_index = pd.read_csv(os.path.join(data_dir, 'dem_index.csv'))
    # - calculate time-delta between input date and the date reference date
    # - of each of the DEMs listed by the DEMs index dataframe.
    delta_days = list()
    for index, row in v_maps_index.iterrows():
        row_date = datetime(year=row['Year_2'], month=1, day=1)
        delta_days.append(np.abs((date_ref - row_date).days))

    delta_days_sorted = sorted(delta_days)
    w1_index = delta_days.index(delta_days_sorted[0])
    w1_f_name = v_maps_index.iloc[w1_index, :]['Name']
    if verbose:
        print('# - Files Selected for the interpolation:')
        print('# - ' + w1_f_name)

    # - Load Velocity Maps
    f_name_1 = os.path.join(data_dir, w1_f_name,
                            w1_f_name
                            + '-rio_EPSG-3413_res-{}_{}.tiff'
                            .format(args_res, args_method))
    with rasterio.open(f_name_1, mode="r+") as src_1:
        vx_1 = src_1.read(1)
        vy_1 = src_1.read(2)
        # - raster upper - left and lower - right corners
        ul_corner_1 = src_1.transform * (0, 0)
        lr_corner_1 = src_1.transform * (src_1.width, src_1.height)
        src_1_minx = ul_corner_1[0]
        src_1_miny = lr_corner_1[1]
        src_1_maxx = lr_corner_1[0]
        src_1_maxy = ul_corner_1[1]

        # - difference domain coordinate axes
    v_vect_x = np.arange(src_1_minx, src_1_maxx + 1, args_res)
    v_vect_y = np.arange(src_1_miny, src_1_maxy + 1, args_res)
    # - create difference domain coordinates grids
    m_xx, m_yy = np.meshgrid(v_vect_x, v_vect_y)
    vx_interp = np.flipud(vx_1)
    vy_interp = np.flipud(vy_1)

    if smooth:
        # - if selected, smooth the interpolated velocity field.
        w_size = smooth_w_size
        if smooth_mode in ['average', 'ave']:
            # - Use w_size*w_size Average filter
            ave_filter = np.ones((w_size, w_size))
            vx_interp = signal.convolve2d(vx_interp, ave_filter,
                                          mode='same')/np.sum(ave_filter)
            vy_interp = signal.convolve2d(vy_interp, ave_filter,
                                          mode='same')/np.sum(ave_filter)
        elif smooth_mode in ['median', 'med']:
            # - Use w_size*w_size Median filter
            vx_interp = signal.medfilt2d(vx_interp, kernel_size=w_size)
            vy_interp = signal.medfilt2d(vy_interp, kernel_size=w_size)
        else:
            print('# - Unknown Smoothing Strategy selected: {}'
                  .format(smooth_mode))
            sys.exit()

    return{'vx_interp': vx_interp, 'vy_interp': vy_interp,
           'x': v_vect_x, 'y': v_vect_y, 'm_xx': m_xx, 'm_yy': m_yy,
           'v_interp_name': w1_f_name}


# -- run main program
if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("# - Computation Time: {}".format(end_time - start_time))
