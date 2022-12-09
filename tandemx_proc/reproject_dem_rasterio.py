#!/usr/bin/env python
u"""
reproject_dem_rasterio.py
Written by Enrico Ciraci' (08/2021)

Reproject TanDEM-X DEMs from their native projection [EPSG:4326] to the selected
new coordinate reference system.
Use GDAL (Geospatial Data Abstraction Library) Python bindings provided by the
Rasterio project to apply the reprojection/interpolation.
Rasterio provides more Pythonic ways to access GDAL API if compared to the
standard GDAL Python Binding.

Find more details on the project website:
    https://rasterio.readthedocs.io

usage: reproject_dem_rasterio.py [-h] [--directory DIRECTORY]
  [--outdir OUTDIR] [--crs CRS] [--res RES] [--resampling_alg RESAMPLING_ALG]

Reproject TanDEM-X from Geographic Coordinates to the selected
coordinate reference system.

options:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -D DIRECTORY
                        Project data directory.
  --outdir OUTDIR, -O OUTDIR
                        Output directory.
  --crs CRS, -C CRS     Destination Coordinate Reference System - def. EPSG:3413
  --res RES, -R RES     Output raster Resolution. - def. 50 meters.
  --resampling_alg RESAMPLING_ALG
                        Warp Resampling Algorithm. - def. bi-linear

The complete list of the available warp resampling algorithms can be found here:
https://rasterio.readthedocs.io/en/
      latest/api/rasterio.enums.html#rasterio.enums.Resampling

Note: This script was developed to process TanDEM-X elevation data  available
    between  2011 and 2021 for the area surrounding  Petermann Glacier
    (Northwest Greenland)

PYTHON DEPENDENCIES:
    argparse: Parser for command-line options, arguments and sub-commands
           https://docs.python.org/3/library/argparse.html
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.io
    geopandas: extends the datatypes used by pandas to allow spatial operations
           on geometric types/ geospatial data.
           https://geopandas.org
    tqdm: A Fast, Extensible Progress Bar for Python and CLI
           https://tqdm.github.io

UPDATE HISTORY:
04/13/2022: Minor Changes.
"""
# - Python Dependencies
from __future__ import print_function
import os
import argparse
import datetime
from tqdm import tqdm
import affine
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
import geopandas as gpd

# - Project Utility Functions
from utility_functions import create_dir


def vrt_param(crs: int, res: int, bounds: list,
              resampling_alg: str) -> dict:
    """
    Virtual Warp Parameters
    :param crs: destination coordinate reference system
    :param res: output x/y-resolution
    :param bounds: Interpolation Domain Boundaries
    :param resampling_alg: Interpolation Algorithm
    :return: dictionary containing vrt options.
    """
    # - Reprojection Parameters
    dst_crs = CRS.from_epsg(crs)  # - Destination CRS
    # - Output image transform
    xres = yres = res
    left, bottom, right, top = bounds
    dst_width = (right - left) / xres
    dst_height = (top - bottom) / yres
    # - Affine transformation matrix
    dst_transform = affine.Affine(xres, 0.0, left,
                                  0.0, -yres, top)
    # - Virtual Warping Options
    vrt_options = {
        'resampling': Resampling[resampling_alg],
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
        'src_nodata': -9999,
        'nodata': -9999,
    }
    return vrt_options


def main() -> None:
    """Reproject TanDEM-X DEMs"""
    # - Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Reproject TanDEM-X from Geographic Coordinates to the
        selected coordinate reference system.
        """
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
                        help='Destination Coordinate Reference System - def. '
                             'EPSG:3413')

    # - Output Resolution
    parser.add_argument('--res', '-R',
                        type=int, default=50,
                        help='Output raster Resolution. - def. 50 meters.')

    # - coordinates.
    parser.add_argument('--resampling_alg',
                        type=str,
                        default='bilinear',
                        help='Warp Resampling Algorithm. - def. bilinear')

    args = parser.parse_args()

    # - Input directory - directory containing the TanDEM-X DEMs
    input_data_path = os.path.join(args.directory, 'RAW_DEMs')
    # - DEM Index File
    indx_file = os.path.join(args.directory, 'Processed_DEMs',
                             'roi_tandemx_dem_index.shp')
    # - Create Output Directory
    out_dir = create_dir(os.path.join(args.directory, 'Processed_DEMs'),
                         f'TanDEM-X_EPSG-{args.crs}_res-{args.res}_ralg'
                         f'-{args.resampling_alg}_rio')

    # - Read DEM index
    print('# - Load TanDEM-X DEMs Index.')
    dem_df = gpd.read_file(indx_file)

    # - DEM index uses EPSG:4326 as crs
    # - Reproject polygons coordinates to the selected CRS.
    reproj_dem_df = dem_df.to_crs(args.crs)
    del dem_df  # - remove original dataframe from memory

    print('# - Reproject TanDEM-X DEMs. ')
    print(f'# - Destination CRS: EPSG-{args.crs}')
    print(f'# - Output Resolution: {args.res}')

    # - list input directory content
    input_data_dir = [os.path.join(input_data_path, x)
                      for x in os.listdir(input_data_path) if not
                      x.startswith('.')]
    print(f'# - Number of DEMs Found: {len(input_data_dir)}')

    for dem in tqdm(sorted(input_data_dir), ncols=70,
                    desc="# - Reproject TanDEM-X DEM: "):
        # - extract DEM name
        # - Extract time-tag from each of the considered DEM
        f_name = dem.split('/')[-1].split('_')
        # - DEM name
        d_name = f_name[-3]+'_'+f_name[-2]+'_'+f_name[-1][:-5]
        # - load DEM valid data bounding box
        dem_info = reproj_dem_df.query(f"Name=='{d_name}'")
        dem_bbox = dem_info['geometry'].bounds

        # - Output raster transform boundaries defined on a regular grid
        # - with step equal to the selected resolution
        minx = int((dem_bbox['minx']//args.res)*args.res)-args.res
        miny = int((dem_bbox['miny']//args.res)*args.res)-args.res
        maxx = int((dem_bbox['maxx']//args.res)*args.res)+args.res
        maxy = int((dem_bbox['maxy']//args.res)*args.res)+args.res
        output_bounds = [minx, miny, maxx, maxy]

        with rasterio.open(dem) as src:
            # - virtual Warp Parameters
            vrt_options = vrt_param(args.crs, args.res,
                                    output_bounds, args.resampling_alg)
            with WarpedVRT(src, **vrt_options) as vrt:
                # Read all data into memory.
                data = vrt.read()
                # - Process the dataset in chunks.
                # - See Rasterio Documentation for more details.
                # - https://rasterio.readthedocs.io/en/latest
                # - /topics/virtual-warping.html
                for _, window in vrt.block_windows():
                    data = vrt.read(window=window)

                # - Save Reprojected Data
                i_directory, name = os.path.split(dem)
                out_name = name.replace('.tiff', '')
                outfile = os.path.join(out_dir,
                                       out_name
                                       + f'-rio_EPSG-{args.crs}_res'
                                         f'-{args.res}.tiff')
                rio_shutil.copy(vrt, outfile, driver='GTiff')


# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f'# - Computation Time: {end_time - start_time}')
