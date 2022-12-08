#!/usr/bin/env python
u"""
dem_tdx_iceshelf_corrections_compute.py
Written by Enrico Ciraci' (12/2021)

Convert TanDEM-X data from elevation above the WGS84 Standard Ellipsoid  into
elevation above the mean sea level. This conversion is required to apply the
Lagrangian Approach which assumes an ice shelf in hydrostatic equilibrium.
Following Shean et al. 2019, the corrected ice surface elevation above sea
level is calculated as follows:

H = He − Hg − αlpha (MDT + Htide + hIBE )
Where:
. He    -> DEM elevation above the WGS84 ellipsoid;
. Hg    -> the EEIGEN-6C4 geoid offset;
. MDT   -> Ocean Mean Dynamic Topography;
. Htide -> Tide elevations above the average sea level;
. hIBE  -> Inverse Barometer Effect on sea level height.
. alpha -> coefficient αlpha that increase linearly with distance l downstream
           of the grounding line (see Shean et al. 2019).
           Note that this last component is actually not considered in this
           implementation of the algorithm.

NOTE (1): at least for now, we subtract the EGM2008 geoid offset from each
    pixel of the input DEM. The other corrections are computed at the entry
    of the glacier's fjord and applied uniformly over the entire DEM's area.
    The effect of the alpha coefficient is, for this reason, neglected.

The EEIGEN-6C4 geoid offset is available with NSIDC BedMachine v5.
https://nsidc.org/data/idbmg4/versions/5

The Ocean Mean Dynamic Topography is distributed by AVISO and available here:
https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/mdt.html
See ./mdt_cnes/crop_mdt_cnes.py for more details.

Tide Elevation at the time-tag associated with each TanDEM-X dem is estimates
by employing outputs from the Arctic Ocean Tidal Inverse Model, 5km (AOTIM-5).
See the project website for more details:
https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/aotim-5

The model solutions are extracted by employing the PyTMD module by Tyler
Sutterley. Below, the link to the project home page on GitHUB:
https://github.com/tsutterley/pyTMD

The Inverse Barometer Correction is calculated by employing hourly estimates
of Mean Sea Level Pressure provided by the ECMWF ERA5 Reanalysis.
See Era5Loader.py for more details.

Form more information on how to create the TanDEM-X index file, see:
https://github.com/eciraci/DHDT-TanDEM-X_DEM/blob/main/index_tandemx_dem.py


COMMAND LINE OPTIONS:
usage: dem_tdx_iceshelf_corrections_compute.py [-h] [--directory DIRECTORY]
                                               [--outdir OUTDIR] [--lat LAT]
                                               [--lon LON] [--crs CRS]
                                               [--res RES] [--median]

Compute Height corrections for time-tagged TanDEM-X DEMs to express elevation
values in meters above sea level. Corrections considered:
    1) Conversion to elevation data to Geoid Height;
    2) IBE from ERA5;
    3) Tidal Elevation from AOTIM-5;
    4) Ocean Mean Dynamic Topography (MDT) - AVISO;

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -D DIRECTORY
                        Project data directory.
  --outdir OUTDIR, -O OUTDIR
                        Output directory.
  --lat LAT             Sample Point Latitude in degrees north
  --lon LON             Sample Point Longitude in degrees north.
  --crs CRS, -T CRS     Coordinate Reference System - def. EPSG:3413
  --res RES, -R RES     Input raster resolution.
  --geoid {BedMachine,EGM2008}, -G {BedMachine,EGM2008}
                        Reference Geoid.
  --median              User long-term Median MSL as reference in the Pressure
                        anomaly calculation.

Note: This preliminary version of the script has been developed to process
      TanDEM-X data available between 2011 and 2020 for the area surrounding
      Petermann Glacier (Northwest Greenland).

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    pandas: Python Data Analysis Library
           https://pandas.pydata.org
    geopandas: Python tools for geographic data
           https://pandas.pydata.org
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.io
    datetime: Basic date and time types
           https://docs.python.org/3/library/datetime.html#module-datetime
    xarray: xarray: N-D labeled arrays and datasets in Python
           https://xarray.pydata.org/en/stable

UPDATE HISTORY:
01/04/2022: --geoid, -G: option added.
09/29/2022: If the IBE correction is not available, use the median value
    of the hourly climatology extract for the entire period of data
    availability.
"""
# - python dependencies
from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import xarray as xr
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

# - pyTMD
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tide import predict_tide
from pyTMD.time import convert_calendar_dates
# - Program Dependencies
from Era5Loader import Era5Loader
from utility_functions import create_dir, do_kdtree


def load_geoid(in_path: str, geoid: str = 'EGM2008', n_res: str = '1',
               epsg: int = 3413,  res: int = 50) -> dict:
    """
    Load Reference Geoid Model
    :param in_path: absolute path to Geoid data directory
    :param geoid: reference Geoid [EGM2008, BedMachine]
    :param n_res: geoid native resolution - used only for EGM2008
    :param epsg: input geoid CRS.
    :param res: interpolated Geoid resolution.
    :return:
    """
    if geoid == 'EGM2008':
        g_f_name = 'Petermann_Domain_Velocity_Stereo_egm2008-{}f_EPSG' \
                   '-{}_res{}_bilinear.tiff'.format(n_res, epsg, res)
    elif geoid == 'BedMachine':
        g_f_name = 'Petermann_Domain_Velocity_Stereo_EIGEN-EC4_height_' \
                   'EPSG-{}_res{}_bilinear.tiff'.format(epsg, res)
    else:
        print('# - Unknown Reference Geoid Model Selected.')
        import sys
        sys.exit()

    # - extract geoid data
    with rasterio.open(os.path.join(in_path, g_f_name), mode="r+") as src:
        # - read band #1 - Geoid elevation in meters
        geoid_input = src.read(1)
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res
        # -
        x_coords = np.arange(ul_corner[0], lr_corner[0], grid_res[0])
        y_coords = np.arange(lr_corner[1], ul_corner[1], grid_res[1])
        if src.transform.e < 0:
            geoid_input = np.flipud(geoid_input)
        # - Compute New Affine Transform
        transform = (Affine.translation(x_coords[0], y_coords[0])
                     * Affine.scale(src.res[0], src.res[1]))
        return{'geoid': geoid_input, 'x_coords': x_coords,
               'y_coords': y_coords, 'res': grid_res,
               'transform': transform, 'src_transform': src.transform,
               'nodata': src.nodata}


def load_dem_tiff(in_path: str) -> dict:
    # - Load TanDEM-X DEM raster saved in GeoTiff format.
    with rasterio.open(in_path, mode="r+") as src:
        # - read band #1 - DEM elevation in meters
        dem_input = src.read(1)
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res
        # -
        x_coords = np.arange(ul_corner[0], lr_corner[0], grid_res[0])
        y_coords = np.arange(lr_corner[1], ul_corner[1], grid_res[1])
        if src.transform.e < 0:
            dem_input = np.flipud(dem_input)
        # - Compute New Affine Transform
        transform = (Affine.translation(x_coords[0], y_coords[0])
                     * Affine.scale(src.res[0], src.res[1]))
        return{'dem': dem_input, 'x_coords': x_coords,
               'y_coords': y_coords, 'res': grid_res,
               'transform': transform,
               'src_transform': src.transform, 'nodata': src.nodata}


class MdtCnes:
    """Load Ocean Mean Dynamic Topography."""
    def __init__(self, d_path):
        # - class attributes
        self.lat_ax = None
        self.lon_ax = None
        self.mdt_input = None
        # - Input file name and variable to consider
        self.f_name = 'mdt_cnes_cls2018_global'
        self.var_mdt = 'mdt'
        v_input = xr.open_dataset(os.path.join(d_path, 'AVISO', 'mdt',
                                               'input.dir', self.f_name,
                                               self.f_name + '.nc'),
                                  mask_and_scale=True)
        self.lon_ax = v_input.coords['longitude'].values - 180.
        self.lat_ax = v_input.coords['latitude'].values

        # - extract MDT data
        mdt_input = np.squeeze(v_input[self.var_mdt].values)
        mdt_input_t = np.zeros(mdt_input.shape)
        mdt_input_t[:, :int(mdt_input.shape[1] / 2)] \
            = mdt_input[:, int(mdt_input.shape[1] / 2):]
        mdt_input_t[:, int(mdt_input.shape[1] / 2):] \
            = mdt_input[:, :int(mdt_input.shape[1] / 2)]
        self.mdt_input = mdt_input_t

    def sample_mdt_pt_coords(self, pt_lat: float, pt_lon: float,
                             verbose: bool = False) -> dict:
        # - Create MLS data domain grid coordinates
        x_coords_mm, y_coords_mm = np.meshgrid(self.lon_ax, self.lat_ax)
        # - Flatten Domain Coordinates
        combined_x_y_arrays = np.dstack([x_coords_mm.ravel(),
                                         y_coords_mm.ravel()])[0]
        s_points = [pt_lon, pt_lat]
        # - Use kd-tree to find the index of the closest location
        index = do_kdtree(combined_x_y_arrays, s_points)
        if verbose:
            print('# - Closest point to the selected coordinates '
                  'within the input data domain -> Lat: {}, Lon: {}'
                  .format(y_coords_mm.ravel()[index],
                          x_coords_mm.ravel()[index]))

        return {'msl_sample': self.mdt_input.ravel()[index],
                'index': index}


def main():
    parser = argparse.ArgumentParser(
        description="""Compute Height corrections for time-tagged
            TanDEM-X dems to express elevation values in meters above sea 
            level. Corrections considered:
            1) Conversion to elevation data to Geoid Height.
            2) IBE from ERA5;
            3) Tidal Elevation from AOTIM-5;
            4) Ocean Mean Dynamic Topography (MDT) - AVISO;
            """
    )

    # - Absolute Path to directory containing input data.
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

    # - Point Latitude in degrees north
    parser.add_argument('--lat',
                        type=float,
                        default=81.543,
                        help='Sample Point Latitude in degrees north')

    # - Point Longitude in degrees north
    parser.add_argument('--lon',
                        type=float,
                        default=-62.181,
                        help='Sample Point Longitude in degrees north.')

    # - Input DEM CRS
    parser.add_argument('--crs', '-T',
                        type=int, default=3413,
                        help='Coordinate Reference System - def. EPSG:3413')

    # - Input DEM Resolution
    parser.add_argument('--res', '-R', type=int,
                        default=150,
                        help='Input raster resolution.')

    # - Reference Geoid
    parser.add_argument('--geoid', '-G', type=str,
                        default='BedMachine',
                        choices=['BedMachine', 'EGM2008'],
                        help='Reference Geoid.')

    # - User Long Term Median MSL as reference in the Pressure anomaly
    # - calculation.
    parser.add_argument('--median',
                        action='store_true',
                        help='User long-term Median MSL as reference in the '
                             'Pressure anomaly calculation.')

    args = parser.parse_args()

    print('# - Apply Height Corrections to time-tagged TanDEM-X DEMs.')

    # - GDAL Binding [Rasterio (rio) or GDAL (gdal)]
    gdal_binding = 'rio'
    # - TanDEM-X DEMs projection algorithm
    resampling_alg = 'average'

    # - Load Corrections
    # - Load Geoid Elevation - EGM2008
    if args.geoid == 'EGM2008':
        geoid_path = os.path.join(args.directory, 'Geoid_Height', 'output.dir')
    elif args.geoid == 'BedMachine':
        geoid_path = os.path.join(args.directory, 'BedMachine', 'output.dir',
                                  'BedMachineGreenland-version_05_Geoid_Height')
    else:
        print('# - Unknown Geoid Model Selected.')
        sys.exit()

    geoid_in = load_geoid(geoid_path, args.geoid, n_res='1', epsg=args.crs,
                          res=args.res)
    x_coords_geoid = geoid_in['x_coords']
    y_coords_geoid = geoid_in['y_coords']
    geoid_ref = geoid_in['geoid']

    # - Load ERA5 data
    print('# - Load Hourly Mean Sea Level Pressure Estimates from ERA5.')
    era_d_input = Era5Loader(args.directory)
    print('# - Reanalysis data loaded.')

    # - Load Ocean Mean Dynamic Topography from AVISO
    print('# - Load Ocean Mean Dynamic Topography from AVISO.')
    mdt_in = MdtCnes(args.directory)
    mdt_corr = mdt_in.sample_mdt_pt_coords(args.lat, args.lon)['msl_sample']

    # - AOTIM-5 Tidal Model data
    print('# - Load AOTIM-5 Tidal Model.')
    tide_dir = os.path.join(args.directory, 'aotim5_tmd', 'data')
    grid_file = os.path.join(tide_dir, 'grid_Arc5km2018')
    model_file = os.path.join(tide_dir, 'h_Arc5km2018')
    # - pyTMD parameters
    model_format = 'OTIS'
    epsg_code = 'PSNorth'
    var_type = 'z'
    # -- read tidal constants and interpolate to grid points
    amp, ph, d, c = extract_tidal_constants(args.lon, args.lat, grid_file,
                                            model_file, epsg_code,
                                            TYPE=var_type,
                                            METHOD='spline',
                                            GRID=model_format)
    # -- calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0
    # -- calculate constituent oscillation
    hc = amp * np.exp(cph)

    # - TanDEM-X DEMS input path
    dem_input_path = os.path.join(args.directory, 'TanDEM-X',
                                  'Petermann_Glacier_out',
                                  'TanDEM-X_EPSG-{}_res-{}_ralg-{}_{}'
                                  .format(args.crs, args.res,
                                          resampling_alg, gdal_binding))
    # - Create Output Directory
    out_dir = create_dir(os.path.join(args.directory, 'TanDEM-X',
                                      'Petermann_Glacier_out'),
                         'TanDEM-X_EPSG-{}_res-{}_ralg-{}_{}_amsl_corrected'
                         .format(args.crs, args.res,
                                 resampling_alg, gdal_binding)
                         )

    # - Load TanDEM-X index shapefile.
    index_file = os.path.join(args.directory, 'TanDEM-X',
                              'Petermann_Glacier_out',
                              'petermann_tandemx_dem_index.shp')

    # - Read DEM index
    print('# - Load TanDEM-X DEMs Index.')
    dem_df = gpd.read_file(index_file).to_crs(epsg=args.crs)
    print('# - Total Number of DEMs available: {}'.format(len(dem_df.index)))

    # - The TanDEM-X index files reports the DEMs bounds polygons in
    dem_df['datetime'] = pd.DatetimeIndex(dem_df['time'])

    # - IBE Corr List - save the IBE correction for each DEM
    ibe_corr_list = []

    # - Initialize a Corrections Report File
    with open(os.path.join(out_dir, '00_corrections_report.csv'), 'w') as w_fid:
        print('DEM_Name,Tide_Corr,IBE_Corr,MDT,TOTAL', file=w_fid)

    # - TanDEM-X File name format
    n_form_1 = 'DEM_TAXI_TDM1_SAR__COS_MONO_SM_S_SRA_'
    n_form_2 = 'DEM_TAXI_TDM1_SAR__COS_BIST_SM_S_SRA_'

    for index, row in tqdm(dem_df.iterrows(), ascii=True,
                           desc="# - Applying Corrections: ",
                           total=dem_df.shape[0]):
        dem_file_in = os.path.join(dem_input_path, n_form_1
                                   + row['Name']+'-{}_EPSG-{}_res-{}.tiff'
                                   .format(gdal_binding, args.crs, args.res))
        if not os.path.isfile(dem_file_in):
            dem_file_in = os.path.join(dem_input_path, n_form_2
                                       + row['Name'] + '-{}_EPSG-{}_res-{}.tiff'
                                       .format(gdal_binding, args.crs,
                                               args.res))

        # - Load Elevation Data
        dem_in = load_dem_tiff(dem_file_in)
        dem_elev = dem_in['dem']
        dem_elev[dem_elev == dem_in['nodata']] = np.nan
        x_coords_dem = dem_in['x_coords']
        y_coords_dem = dem_in['y_coords']

        # - Crop Geoid Area overlapping with the considered DEM
        ind_x = np.where((x_coords_geoid >= x_coords_dem[0])
                         & (x_coords_geoid <= x_coords_dem[-1]))
        ind_y = np.where((y_coords_geoid >= y_coords_dem[0])
                         & (y_coords_geoid <= y_coords_dem[-1]))
        ind_xx, ind_yy = np.meshgrid(ind_x, ind_y)

        # - Express DEM elevation values as height above the EGM2008 Geoid.
        dem_elev -= geoid_ref[ind_yy, ind_xx]

        # - Calculate Inverse Barometer Effect Correction at the considered
        # - location and time.
        try:
            ibe_corr \
                = era_d_input.compute_ibe_correction(args.lat, args.lon,
                                                     row['datetime'].year,
                                                     row['datetime'].month,
                                                     row['datetime'].day,
                                                     row['datetime'].hour,
                                                     median=args.median)
        except IndexError:
            # - If the IBE correction is not available, use the median value
            # - of the hourly climatology extract for the entire period of
            # - data availability.
            # - NOTE: This is a temporary solution introduced to process
            # -       the TanDEM-X DEMs during the CDS Service disruption
            # -       period in September 2022.
            ibe_clim \
                = era_d_input.compute_ibe_climatology(args.lat, args.lon)
            ibe_corr = ibe_clim["ibe"].values[row['datetime'].hour]

        # - Calculate Number rof days relative to Jan 1, 1992 (48622 MJD)
        delta_time_t = convert_calendar_dates(row['datetime'].year,
                                              row['datetime'].month,
                                              row['datetime'].day,
                                              hour=row['datetime'].hour,
                                              minute=row['datetime'].minute,
                                              second=row['datetime'].second)
        # - predict tidal elevations at time and infer minor corrections
        tide_p = predict_tide(delta_time_t, hc, c, DELTAT=0,
                              CORRECTIONS=model_format)
        minor_p = infer_minor_corrections(delta_time_t, hc, c,
                                          DELTAT=0,
                                          CORRECTIONS=model_format)
        # - Tide Correction in Meters
        tide_corr = tide_p + minor_p[0]

        # - Evaluate the sum of all the considered corrections:
        total_correction = mdt_corr + tide_corr + ibe_corr
        # - Apply Corrections
        dem_elev -= total_correction

        # - Save Corrected DEM
        out_file = os.path.join(out_dir, 'DEM_TAXI_TDM1_SAR__COS_BIST_SM_S_SRA_'
                                + row['Name']+'-{}_EPSG-{}_res-{}.tiff'
                                .format(gdal_binding, args.crs, args.res))
        if dem_in['src_transform'].e < 0:
            # - Save the final corrected DEM employing the same Affine
            # - Transformation of the Input one.
            dem_elev = np.flipud(dem_elev)
        with rasterio.open(out_file, 'w', driver='GTiff',
                           height=dem_elev.shape[0],
                           width=dem_elev.shape[1], count=1,
                           dtype=dem_elev.dtype, crs=args.crs,
                           transform=dem_in['src_transform'],
                           nodata=-9999) as dst:
            dst.write(dem_elev, 1)

        # - Save corrections magnitude inside the chosen report file
        with open(os.path.join(out_dir, '00_corrections_report.csv'),
                  'a') as w_fid:
            print('{},{},{},{},{}'
                  .format(row['Name'], tide_corr, ibe_corr, mdt_corr,
                          total_correction),
                  file=w_fid)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print("# - Computation Time: {}".format(end_time - start_time))
