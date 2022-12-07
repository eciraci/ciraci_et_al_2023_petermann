#!/usr/bin/env python
u"""
mosaicing_dem_rasterio.py
Written by Enrico Ciraci' (08/2021)

Create a Daily Mosaics of TanDEM-X dems over the regions of interest.
Before  creating the mosaic, the raster covering the largest area is chosen as
reference. The other (secondary) rasters are afterward aligned to the reference
one by employing on the estimation strategies reported below.

COMMAND LINE OPTIONS:
    --directory X, -D X: Project data directory.
    --outdir X, -O X: Output Directory.
    --crs CRS, -C CRS: Input Data Coordinate Reference System - def.
                       EPSG:3413
    --res X, -R X: Input raster Resolution. - def. 50 meters.
    --poly {-1,0,1,2}, -P {-1,0,1,2} Mean Bias Estimator - Polynomial Order
            -> -1: no mean bias correction applied.
            ->  0: mean bias correction evaluated as mean of the difference
                   of the two rasters over the overlapping region.
            ->  1: use polynomial surface of order one to estimate/correct
                   the mean bias between the two raster.
            ->  2: use polynomial surface of order two to estimate/correct
                   the mean bias between the two raster.

Note: This preliminary version of the script has been developed to process
      TanDEM-X data available between 2011 and 2020 for the area surrounding
      Petermann Glacier (Northwest Greenland).

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    pandas: Python Data Analysis Library
           https://pandas.pydata.org/
    geopandas: Python tools for geographic data
           https://pandas.pydata.org/
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.io
    tqdm: A Fast, Extensible Progress Bar for Python and CLI
           https://tqdm.github.io
   datetime: Basic date and time types
           https://docs.python.org/3/library/datetime.html#module-datetime

UPDATE HISTORY:

11/11/2021: Save output Mosaic using the standard GeoTIFF pixel order:
             - top-left raster corner = Pixel(0, 0) -> Xmin, Ymax,
             - bottom-right corner = Pixel(-1, -1) -> Xmax, Ymin

04/13/2022: Minor Changes.

11/08/2022: Save Mosaics index in ESRI shapefile format.

"""
# - Python Dependencies
from __future__ import print_function
import os
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import fiona
import rasterio
from rasterio.transform import Affine
from rasterio import features
import geopandas as gpd

# - Project Utility Functions
from utility_functions import create_dir


def polynomial_mean_bias_estimator(x_v_reg: np.ndarray, y_v_reg: np.ndarray,
                                   elev_d_v_reg: np.ndarray,
                                   poly_order: int = 1) -> np.ndarray:
    """
    Apply Linear Regression to calculate mean bias among
    partially overlapping DEMs
    :param x_v_reg: flattened easting domain coordinates
    :param y_v_reg: flattened northing domain coordinates
    :param elev_d_v_reg: flattened elevation values
    :param poly_order: polynomial order
    :return: regression coefficients
    """
    # - Apply Linear Regression to calculate mean bias.
    npts_reg = len(x_v_reg)
    if poly_order == 1:
        d_mat = np.vstack([x_v_reg, y_v_reg,
                           np.ones(npts_reg)]).T
        reg = np.linalg.lstsq(d_mat, elev_d_v_reg,
                              rcond=None)[0]
    elif poly_order == 2:
        d_mat = np.vstack([x_v_reg, x_v_reg ** 2,
                           y_v_reg, y_v_reg ** 2,
                           x_v_reg * y_v_reg,
                           np.ones(npts_reg)]).T

        reg = np.linalg.lstsq(d_mat, elev_d_v_reg,
                              rcond=None)[0]
    else:
        d_mat = np.vstack([x_v_reg, y_v_reg,
                           np.ones(npts_reg)]).T
        reg = np.linalg.lstsq(d_mat, elev_d_v_reg,
                              rcond=None)[0]

    return reg


def main() -> None:
    """
    Create a Daily Mosaics of TanDEM-X dems
    """
    # - Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Create Mosaic of TanDEM-X DEMs using Rasterio."""
    )
    # - Project data directory.
    default_dir = os.path.join(os.getenv('PYTHONDATA'), 'TanDEM-X')
    parser.add_argument('--directory', '-D',
                        type=lambda p: os.path.abspath(os.path.expanduser(p)),
                        default=default_dir,
                        help='Project data directory.')
    # - Output directory
    parser.add_argument('--outdir', '-O',
                        type=str,
                        default=default_dir,
                        help='Output directory.')
    # - Destination Coordinate Reference System
    # - Default: WGS 84 / NSIDC Sea Ice Polar Stereographic North
    parser.add_argument('--crs', '-C',
                        type=int, default=3413,
                        help='Input Data Coordinate Reference System - def. '
                             'EPSG:3413')
    # - Output Resolution
    parser.add_argument('--res', '-R',
                        type=int, default=50,
                        help='Output raster Resolution. - def. 50 meters.')
    # - GDAL Resampling Algorithm used in the reprojection step.
    parser.add_argument('--resampling_alg',
                        type=str,
                        default='bilinear',
                        help='Warp Resampling Algorithm. - def. bilinear')
    # - Mean Bias Estimator - Polynomial Order
    parser.add_argument('--poly', '-P',
                        type=int, default=0, choices=[-1, 0, 1, 2],
                        help=' Mean Bias Estimator - Polynomial Order')

    args = parser.parse_args()

    # - Processing Parameters
    # - Temporary Parameters
    poly_order = args.poly

    # - Input directory
    input_data_path = os.path.join(args.directory, 'Petermann_Glacier_out',
                                   f'TanDEM-X_EPSG-{args.crs}_res-{args.res}'
                                   f'_ralg-{args.resampling_alg}_rio')
    # - Create Output directory
    output_data_path \
        = create_dir(os.path.join(args.directory, 'Petermann_Glacier_out'),
                     'Mosaics')
    # -
    output_data_path = create_dir(output_data_path,
                                  f'Petermann_Glacier_Mosaics_EPSG-{args.crs}'
                                  f'_res-{args.res}_ralg-{args.resampling_alg}'
                                  f'_rio_poly{args.poly}')
    # - DEM Index File
    indx_file = os.path.join(args.directory, 'Petermann_Glacier_out',
                             'petermann_tandemx_dem_index.shp')

    # - Read DEM index
    print('# - Load TanDEM-X DEMs Index.')
    dem_df = gpd.read_file(indx_file)

    # - The TanDEM-X index files reports the DEMs bounds polygons in
    dem_df['datetime'] = pd.DatetimeIndex(dem_df['time'])
    dem_df['ntime'] = dem_df['datetime']
    dem_df.drop(['time'], axis=1)
    dem_df = dem_df.set_index('datetime')
    dem_df = dem_df.sort_index()

    # - create year, month, and day axis
    dem_df['year'] = \
        dem_df['ntime'].apply(lambda x: x.year)
    dem_df['month'] = \
        dem_df['ntime'].apply(lambda x: x.month)
    dem_df['day'] = \
        dem_df['ntime'].apply(lambda x: x.day)
    # - create range of date range
    date_range = pd.date_range(start=dem_df['ntime'].iloc[0],
                               end=dem_df['ntime'].iloc[-1])
    # -
    for s_date in tqdm(date_range, ncols=70,
                       desc='# - Create Daily TanDEM-X Mosaics: '):
        day_df = dem_df.query(f'year == {s_date.year} & month =='
                              f' {s_date.month} & day == {s_date.day}')
        # - if DEMs are found for the considered date,
        # - calculate their mosaic.
        if not day_df.empty:
            # - do not consider the [HH[:MM[:SS[.mmm[uuu]]]]]
            # - portion of the isoformat value of the considered date
            date_isoformat\
                = datetime.datetime(s_date.year, s_date.month,
                                    s_date.day).isoformat().split('T')[0]
            # - output mosaic file name
            output_file_name = date_isoformat + '_tdemx_mosaic.tiff'

            # - list files available for the selected date
            input_file_list = list(day_df['Name'])

            minx, miny, maxx, maxy = 0, 0, 0, 0
            crs_mosaic = None
            mosaic_dict = {}
            for cnt, dem_name in enumerate(input_file_list):
                # - Import DEM data
                # - List input data directory content
                f_name = [os.path.join(input_data_path, x)
                          for x in os.listdir(input_data_path)
                          if dem_name in x][0]
                mosaic_dict[dem_name] = {}

                with rasterio.open(f_name, mode='r+') as src:
                    # - read band #1 - elevation in meters
                    mosaic_dict[dem_name]['elev'] = src.read(1)
                    # - set no-data grid points to NaN
                    mosaic_dict[dem_name]['elev'][mosaic_dict[dem_name]['elev']
                                                  == src.nodata] = np.nan
                    # - evaluate raster size
                    mosaic_dict[dem_name]['npts'] = src.width * src.height
                    # - raster upper-left and lower-right corners
                    mosaic_dict[dem_name]['ul_corner'] = src.transform * (0, 0)
                    mosaic_dict[dem_name]['lr_corner'] \
                        = src.transform * (src.width, src.height)

                # - Define Patch Bounds
                ptch_minx = mosaic_dict[dem_name]['ul_corner'][0]
                ptch_miny = mosaic_dict[dem_name]['lr_corner'][1]
                ptch_maxx = mosaic_dict[dem_name]['lr_corner'][0]
                ptch_maxy = mosaic_dict[dem_name]['ul_corner'][1]
                mosaic_dict[dem_name]['bounds'] = [ptch_minx, ptch_miny,
                                                   ptch_maxx, ptch_maxy]

                # - Find Mosaic Domain Bounds
                if cnt == 0:
                    minx = ptch_minx
                    miny = ptch_miny
                    maxx = ptch_maxx
                    maxy = ptch_maxy
                    crs_mosaic = src.crs
                else:
                    if ptch_minx < minx:
                        minx = ptch_minx
                    if ptch_miny < miny:
                        miny = ptch_miny
                    if ptch_maxx > maxx:
                        maxx = ptch_maxx
                    if ptch_maxy > maxy:
                        maxy = ptch_maxy

            # - Round Bounding-box values to define mosaic coordinate axes grids
            # - Note that these axes define a raster coordinate system
            # - where the origin is the lower left corner of the lower left
            # - pixel. The output  raster will have to be referenced
            # - is a coordinates' system where the origin is the upper left
            # - pixel of the raster.
            mosaic_vect_x = np.arange(minx, maxx+1, args.res)
            mosaic_vect_y = np.arange(miny, maxy+1, args.res)

            # - Define Reference Raster
            # - -> Use as reference raster with the on with  highest
            # -    #pixels [i.e. approximately the largest covered area]
            reference_raster = None
            n_pts_max = 0
            for dem_name in mosaic_dict.keys():
                if mosaic_dict[dem_name]['npts'] > n_pts_max:
                    n_pts_max = mosaic_dict[dem_name]['npts']
                    reference_raster = dem_name

            # - create mosaic domain coordinates grids
            m_xx, m_yy = np.meshgrid(mosaic_vect_x, mosaic_vect_y)
            mosaic_stack_shape = (len(mosaic_dict), m_xx.shape[0],
                                  m_xx.shape[1])

            # - Mosaic Stack to be used to save temporary aligned DEMs
            mosaic_stack = np.full(mosaic_stack_shape, np.nan, dtype=np.float32)
            mosaic_reference = np.full((m_xx.shape[0],
                                        m_xx.shape[1]), np.nan,
                                       dtype=np.float32)
            # - Add Reference Raster to Mosaic Stack
            dem_info_d = mosaic_dict[reference_raster]
            ind_x = np.where((mosaic_vect_x >=
                              mosaic_dict[reference_raster]['bounds'][0])
                             & (mosaic_vect_x <
                                mosaic_dict[reference_raster]['bounds'][2]))[0]
            ind_y = np.where((mosaic_vect_y >=
                              mosaic_dict[reference_raster]['bounds'][1])
                             & (mosaic_vect_y <
                                mosaic_dict[reference_raster]['bounds'][3]))[0]
            ind_xx, ind_yy = np.meshgrid(ind_x, ind_y)
            mosaic_reference[ind_yy, ind_xx] = np.flipud(dem_info_d['elev'])

            # - Analyze each of the DEMs available for the considered
            # - date.
            for cnt, dem_name in enumerate(mosaic_dict.keys()):
                dem_info_d = mosaic_dict[dem_name]
                ind_x = np.where((mosaic_vect_x
                                  >= mosaic_dict[dem_name]['bounds'][0])
                                 & (mosaic_vect_x
                                    < mosaic_dict[dem_name]['bounds'][2]))[0]
                ind_y = np.where((mosaic_vect_y
                                  >= mosaic_dict[dem_name]['bounds'][1])
                                 & (mosaic_vect_y
                                    < mosaic_dict[dem_name]['bounds'][3]))[0]
                ind_xx, ind_yy = np.meshgrid(ind_x, ind_y)

                # - NOTE flip-ud is necessary here because of the ordering
                # - of the y-axis in GeoTiff rasters. The grid point with
                # - coordinates (0, 0) represents the upper-left corner of
                # - the dataset (i.e. the grid-point associated with
                # - the maximum y value).
                if dem_name == reference_raster:
                    # - reference raster
                    mosaic_stack[cnt, :, :] = mosaic_reference
                else:
                    # - secondary raster
                    mosaic_stack_dif = \
                        np.full((m_xx.shape[0], m_xx.shape[1]), np.nan)
                    mosaic_stack_dif[ind_yy, ind_xx] = \
                        np.flipud(dem_info_d['elev'])
                    # - Calculate difference between the Reference and
                    # - the secondary dem over the overlapping areas.
                    diff_temp = mosaic_stack_dif - mosaic_reference
                    # - find difference raster's finite values
                    ind_finite = np.isfinite(diff_temp)
                    # - calculate mean bias function b(x, y)
                    # - over the considered domain
                    elev_d_v_reg = diff_temp[ind_finite].flatten()
                    x_v_reg = m_xx[ind_finite].flatten()
                    y_v_reg = m_yy[ind_finite].flatten()

                    if poly_order in [1, 2]:
                        # - Apply Linear Regression to calculate mean bias.
                        reg = \
                            polynomial_mean_bias_estimator(x_v_reg, y_v_reg,
                                                           elev_d_v_reg,
                                                           poly_order=
                                                           poly_order)
                        if poly_order == 1:
                            # - mean_bias(x, y) - polynomial order 1
                            mean_bias = ((m_xx*reg[0]) + (m_yy*reg[1])
                                         + (np.ones(m_yy.shape)*reg[2]))
                        else:
                            # - mean_bias(x, y) - polynomial order 2
                            mean_bias = ((m_xx*reg[0]) + ((m_xx**2)*reg[1])
                                         + (m_yy*reg[2]) + ((m_yy**2)*reg[3])
                                         + (m_xx*m_yy*reg[4])
                                         + (np.ones(m_yy.shape)*reg[5]))

                    elif poly_order in [0]:
                        # - Calculate the mean bias as the mean of the
                        # - difference between the two rasters considered
                        # - over the overlapping areas.
                        with warnings.catch_warnings():
                            # - Ignore RuntimeWarning: Mean of empty slice
                            warnings.simplefilter('ignore',
                                                  category=RuntimeWarning)
                            mean_bias = np.nanmean(diff_temp)
                    else:
                        # - do not apply the bias correction
                        mean_bias = np.zeros(diff_temp.shape)

                    # - Apply mean bias correction.
                    mosaic_stack[cnt, :, :] = mosaic_stack_dif - mean_bias

            # - Calculate the value of the final mosaic as the
            # - average of the corrected raster.
            # - A significant portion of the raster domain is
            # - composed by NaN values.
            with warnings.catch_warnings():
                # - Ignore RuntimeWarning: Mean of empty slice
                warnings.simplefilter('ignore', category=RuntimeWarning)
                mosaic = np.nanmean(mosaic_stack, axis=0)

            # - save the obtained mosaic
            out_path = os.path.join(output_data_path,
                                    output_file_name)
            # - Calculate Affine Transformation of the output raster
            res = args.res
            x_mos = mosaic_vect_x
            y_mos = mosaic_vect_y
            # - Rotate Mosaic Y-Axis
            mosaic = np.flipud(mosaic)
            y_mos = np.flipud(y_mos)
            # - shift the y-axis by 1-pixel in order to use the
            # - upper left corner of the pixel as reference.
            y_mos += res
            transform = (Affine.translation(x_mos[0], y_mos[0])
                         * Affine.scale(res, -res))

            out_meta = {'driver': 'GTiff', 'height': mosaic.shape[0],
                        'width': mosaic.shape[1], 'crs': crs_mosaic,
                        'count': 1, 'nodata': -9999, 'dtype': mosaic.dtype,
                        'compress': 'lzw', 'transform': transform}
            with rasterio.open(out_path, 'w', **out_meta) as dst:
                dst.write(mosaic, 1)

    # - Create Mosaics Index Shapefile
    # - List input data directory content
    input_data_dir = [os.path.join(output_data_path, x)
                      for x in os.listdir(output_data_path) if not
                      x.startswith('.') and x.endswith('.tiff')]
    print('# - Create Shapefile Index of the time-tagged Digital Elevation '
          'Models MOSAICS from the DLR/TanDEM-X mission.')
    # -
    print(f'# - Number of DEMs Found: {len(input_data_dir)}')

    # - Define output index shapefile schema
    crs = f"EPSG:{args.crs}"       # - default coordinate reference system
    schema = {
        'geometry': 'Polygon',
        'properties': [('Name', 'str'), ('time', 'str'),
                       ('Width', 'int'), ('Height', 'int'),
                       ('Units', 'str'), ('npts', 'str')]
    }
    out_f_name = os.path.join(output_data_path,
                              'petermann_tandemx_dem_mosaics_index.shp')
    with fiona.open(out_f_name, mode='w', driver='ESRI Shapefile',
                    schema=schema, crs=crs) as poly_shp:
        for dem in tqdm(sorted(input_data_dir), ncols=70,
                        desc="# - Indexing TanDEM-X DEM: "):
            # - Extract time-tag from each of the considered DEM
            f_name = dem.split('/')[-1].split('_')[0].split('-')
            year_d = int(f_name[0])
            month_d = int(f_name[1])
            day_d = int(f_name[2])
            # - convert dat string to datetime
            dem_date = datetime.datetime(year=year_d, month=month_d, day=day_d)
            # - DEM name
            d_name = dem.split('/')[-1].split('_')[0]
            # - Import DEM data
            with rasterio.open(dem, mode="r+") as src:
                # - read band #1
                m_arr = src.read(1, masked=True)

                # - generate valid data binary mask
                # - valid data - msk = 255
                # - not valid data - msk = 0
                msk = np.full(m_arr.shape, 255).astype('float32')
                msk[np.isnan(m_arr)] = 0

                # - Use rasterio.features.shapes to get valid data region
                # - boundaries. For more details:
                # - https://rasterio.readthedocs.io/en/latest/api/
                # -         rasterio.features.html
                b_shapes = list(features.shapes(msk,
                                                transform=src.transform))
                # - In several cases, features.shapes returns multiple
                # - polygons. Use only the polygon with the maximum number
                # - of points to delineate the area covered by valid elevation
                # - data.
                poly_vect_len = list()
                for shp_bound_tmp in b_shapes:
                    poly_vect_len.append(len(shp_bound_tmp[0]
                                             ['coordinates'][0]))
                max_index = poly_vect_len.index(max(poly_vect_len))
                shp_bound = b_shapes[max_index]
                # -
                row_dict = {
                    # - Geometry [Polygon]
                    'geometry': {'type': 'Polygon',
                                 'coordinates': shp_bound[0]['coordinates']},
                    # - Properties [based on the schema defined above]
                    'properties': {'Name': d_name,
                                   'time': dem_date.isoformat(),
                                   'Width': src.width, 'Height': src.height,
                                   'Units': src.units[0],
                                   'npts': len(shp_bound[0]['coordinates'][0])},
                }
                poly_shp.write(row_dict)


# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f'# - Computation Time: {end_time - start_time}')
