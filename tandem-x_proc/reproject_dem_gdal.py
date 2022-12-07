#!/usr/bin/env python
u"""
reproject_dem_rasterio.py
Written by Enrico Ciraci' (08/2021)

Reproject TanDEM-X DEMs from their native projection [EPSG:4326] to the selected
new coordinate reference system.
Use GDAL (Geospatial Data Abstraction Library) Python bindings package to apply
the reprojection/interpolation. (https://gdal.org/api/python.html)

COMMAND LINE OPTIONS:
    --directory X, -D X: Project data directory.
    --outdir X, -O X: Output Directory.
    --crs X, -C X: Destination Coordinate Reference System - def.EPSG:3413
    --res X, -R X: Output raster Resolution. - def. 50 meters.
    --resampling_alg X: Warp Resampling Algorithm. - def. bilinear

The complete list of warp resampling algorithms can be found here:
https://gdal.org/programs/gdalwarp.html
https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r

Note: This preliminary version of the script has been developed to process
      TanDEM-X data available between 2011 and 2020 for the area surrounding
      Petermann Glacier (Northwest Greenland).

PYTHON DEPENDENCIES:
    argparse: Parser for command-line options, arguments and sub-commands
           https://docs.python.org/3/library/argparse.html
    gdal: GDAL python binding.
           https://gdal.org/api/python.html
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
from osgeo import gdal, osr
import geopandas as gpd

# - Project Utility Functions
from utility_functions import create_dir


def main() -> None:
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

    # - GDAL Resampling Algorithm
    parser.add_argument('--resampling_alg',
                        type=str,
                        default='bilinear',
                        help='Warp Resampling Algorithm. - def. bilinear')

    args = parser.parse_args()

    # - Input directory
    input_data_path = os.path.join(args.directory, 'Petermann_Glacier')
    # - DEM Index File
    indx_file = os.path.join(args.directory, 'Petermann_Glacier_out',
                             'petermann_tandemx_dem_index.shp')

    # - Create Output Directory
    out_dir = create_dir(os.path.join(args.directory, 'Petermann_Glacier_out'),
                         f'TanDEM-X_EPSG-{args.crs}_res-{args.res}_ralg'
                         f'-{args.resampling_alg}_gdal')
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

        # - Output image transform
        xres = yres = args.res

        # - Extract source coordinate reference system EPSG code
        ds = gdal.Open(dem)
        prj = ds.GetProjection()
        epsg_code = osr.SpatialReference(wkt=prj).GetAuthorityCode('GEOGCS')
        del ds
        # - Set GDAL Warp options
        options = gdal.WarpOptions(format="GTiff",
                                   outputBounds=output_bounds,
                                   srcSRS=f'EPSG:{epsg_code}',
                                   dstSRS=f'EPSG:{args.crs}',
                                   xRes=xres, yRes=yres,
                                   srcNodata=-9999, dstNodata=-9999,
                                   resampleAlg=args.resampling_alg)
        # - Save Reprojected Data
        i_directory, name = os.path.split(dem)
        out_name = name.replace('.tiff', '')
        outfile = os.path.join(out_dir,
                               out_name
                               + f'-gdal_EPSG-{args.crs}_res-{args.res}.tiff')
        gdal.Warp(outfile, dem, options=options)


# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f'# - Computation Time: {end_time - start_time}')
