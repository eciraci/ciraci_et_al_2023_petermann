#!/usr/bin/env python
u"""
dhdt_trend_map.py
Written by Enrico Ciraci' (01/2022)

TEST: Evaluate TanDEM-X data coverage over the selected region of interest and
during the selected reference temporal interval.

The Region of Interest boundaries can be provided either in the form of:
- Esri polygon shapefile.
- Domain boundaries bounding box including Lat Min,Lat Max,Lon Min,Lon Max.

For more details on how to create the index file see:
https://github.com/eciraci/DHDT-TanDEM-X_DEM/blob/main/index_tandemx_dem.py

Returns a figure containing:
- Bar plot showing the number of DEMs available per sampling period.


COMMAND LINE OPTIONS:
usage: dem_tdx_temporal_coverage.py [-h] [--directory DIRECTORY]
                                    [--outdir OUTDIR]
                                    [--boundaries BOUNDARIES]
                                    [--shapefile SHAPEFILE] [--crs CRS]
                                    [--res RES] [--f_date F_DATE]
                                    [--l_date L_DATE]

TEST: Evaluate TanDEM-X temporal coverage of the considered region of
interest.

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -D DIRECTORY
                        Project data directory.
  --outdir OUTDIR, -O OUTDIR
                        Output directory.
  --boundaries BOUNDARIES, -B BOUNDARIES
                        Domain BBOX (WGS84) - Lat Min,Lat Max,Lon Min,Lon Max
  --shapefile SHAPEFILE, -S SHAPEFILE
                        Absolute path to the shapefile containing the
                        coordinates of the locations to consider.
  --crs CRS, -T CRS     Coordinate Reference System - def. EPSG:3413
  --res RES, -R RES     Input raster resolution.
  --f_date F_DATE, -F F_DATE
                        Starting Date of the Considered Period of time.Passed
                        as year,month,day or reference year only.
  --l_date L_DATE, -L L_DATE
                        Ending Date of the Considered Period of time.Passed as
                        year,month,day or reference year only.
  --freq FREQ, -Q FREQ  Sampling Frequency.

PYTHON DEPENDENCIES:
    matplotlib: Library for creating static, animated, and interactive
           visualizations in Python.
           https://matplotlib.org
    pandas: Python Data Analysis Library
           https://pandas.pydata.org
    geopandas: Python tools for geographic data
           https://pandas.pydata.org
    shapely: Manipulation and analysis of geometric objects in the Cartesian
           plane.
           https://shapely.readthedocs.io/en/stable
    datetime: Basic date and time types
           https://docs.python.org/3/library/datetime.html#module-datetime
    xarray: xarray: N-D labeled arrays and datasets in Python
           https://xarray.pydata.org/en/stable

UPDATE HISTORY:

"""
# - Python Dependencies
from __future__ import print_function
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import dates
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from datetime import datetime
# - change matplotlib default setting
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-deep')


def dem_tdx_temporal_coverage(output_path: str, dem_df: pd.DataFrame,
                              n_month: int, freq: str = '1M', var: str = 'area',
                              title: str = 'TanDEM-X Data Availability',
                              xlabel: str = '[time]',
                              ylabel: str = '[Count]',
                              fig_format: str = 'jpeg') -> None:
    """
    Plot input dataframe temporal coverage at the selected sampling frequency
    :param output_path: absolute path to output figure
    :param dem_df: reference index Dataframe containing dataset list
    :param n_month: number of months covered by the considered reference period
    :param freq: data coverage sampling period
    :param var: reference variable for frequency calculation
    :param title: output figure title
    :param xlabel: output figure xlabel
    :param ylabel: output figure ylabel
    :param fig_format: output figure format
    :return: None
    """
    # - Number of DEMs available for the selected sampling period
    dem_df_area_cnt = dem_df[var] \
        .groupby(pd.Grouper(freq=freq)).count()
    dem_count = dem_df_area_cnt.values
    cnt_date = [dates.date2num(x) for x in dem_df_area_cnt.index]

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # - Average Area Covered per sampling period
    ax1.set_title(title, weight='bold', loc='left')
    ax1.bar(cnt_date, dem_count, width=30, color='#397aec', edgecolor='#0a4dc2')
    ax1.set_ylabel(ylabel, weight='bold')
    ax1.set_xlabel(xlabel, weight='bold')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(int(n_month / 6)))
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%Y-%b'))
    # - add grid
    ax1.grid(color='k', linestyle='dotted', alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.savefig(output_path, dpi=200, format=fig_format)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="""TEST: Evaluate TanDEM-X temporal coverage of the
        considered region of interest."""
    )

    # - Absolute Path to project data directory.
    default_dir = os.path.join('/', 'Volumes', 'Extreme Pro')
    parser.add_argument('--directory', '-D',
                        type=lambda p: os.path.abspath(os.path.expanduser(p)),
                        default=default_dir,
                        help='Project data directory.')

    # - Absolute path to Output directory
    parser.add_argument('--outdir', '-O',
                        type=str,
                        default=default_dir,
                        help='Output directory.')

    # - Region of interest Boundaries provided as a bbox.
    parser.add_argument('--boundaries', '-B', type=str, default=None,
                        help='Domain BBOX (WGS84) - Lat Min,Lat Max,'
                             'Lon Min,Lon Max')

    # - Region of interest Boundaries provided as Shapefile
    parser.add_argument('--shapefile', '-S', type=str,
                        # - Uncomment this default value during Development
                        default=os.path.join(default_dir, 'GIS_Data',
                                             'Petermann_features_extraction',
                                             'Petermann_iceshelf_clipped_'
                                             'epsg3413.shp'),
                        # default=None,
                        help='Absolute path to the shapefile containing the'
                             ' coordinates of the locations to consider.')

    # - Default Coordinate Reference system.
    parser.add_argument('--crs', '-T',
                        type=int, default=3413,
                        help='Coordinate Reference System - def. EPSG:3413')

    # - Dataset Resolution.
    parser.add_argument('--res', '-R', type=int,
                        default=150,
                        help='Input raster resolution.')

    #  Define Reference temporal interval.
    parser.add_argument('--f_date', '-F', type=str, default='2011,1,1',
                        help='Starting Date of the Considered Period of time.'
                             'Passed as year,month,day or reference year only.')

    parser.add_argument('--l_date', '-L', type=str, default='2020,12,31',
                        help='Ending Date of the Considered Period of time.'
                             'Passed as year,month,day or reference year only.')

    parser.add_argument('--freq', '-Q', type=str, default='2M',
                        help='Sampling Frequency.')

    args = parser.parse_args()

    if not args.boundaries and not args.shapefile:
        print('# - Provide ROI of interest coordinates. See Options:')
        print('# - --boundaries, -B : Lat Min,Lat Max,Lon Min,Lon Max.')
        print('# - --shapefile, -S : Absolute path the Shapefile containing'
              ' the coordinates of the locations to consider.')
        sys.exit()
    elif args.boundaries and args.shapefile:
        print('# - Chose ROI boundaries using either (--boundaries, -B) or '
              '(--shapefile, -S) options.')
        sys.exit()

    # - Read Starting and Ending Date of the considered period of time
    t_start = [int(x) for x in args.f_date.split(',')]
    t_end = [int(x) for x in args.l_date.split(',')]
    # - Reference dates provided as a single year values
    if len(t_start) == 1:
        t_start = [t_start[0], 6, 1]
    elif len(t_start) == 2:
        t_start = [t_start[0], t_start[1], 1]

    if len(t_end) == 1:
        t_end = [t_end[0], 6, 30]
    elif len(t_end) == 2:
        t_end = [t_end[0], t_end[1], 1]
    # -
    t00 = datetime(*t_start)
    t11 = datetime(*t_end)
    n_month = (((t11.year - t00.year) * 12) + (t11.month - t00.month))
    print('# - Selected Period:')
    print(f'# - t00: {t00}')
    print(f'# - t11 {t11}')
    if t11 < t00:
        print('# - t_end < t_start selected.')
        sys.exit()

    # - Load TanDEM-X index shapefile.
    # -     To create the index file, see:
    # -     https://github.com/eciraci/DHDT-TanDEM-X_DEM
    index_file = os.path.join(args.directory, 'TanDEM-X',
                              'Petermann_Glacier_out',
                              'petermann_tandemx_dem_index.shp')

    # - Read DEM index
    print('# - Load TanDEM-X DEMs Index.')
    dem_df = gpd.read_file(index_file).to_crs(epsg=args.crs)
    print(f'# - Total Number of DEMs available: {len(dem_df.index)}')
    # - The TanDEM-X index files reports the DEMs bounds polygons in
    dem_df['datetime'] = pd.DatetimeIndex(dem_df['time'])
    dem_df['ntime'] = dem_df['datetime']
    dem_df = dem_df.drop(['time'], axis=1)  # - drop original time axis
    dem_df = dem_df.set_index('datetime')
    dem_df = dem_df.sort_index()
    dset_time_ref = list(dem_df['ntime'])
    t00_s = dset_time_ref[0].strftime('%Y-%m-%d')
    t11_s = dset_time_ref[-1].strftime('%Y-%m-%d')

    if dset_time_ref[0] > t00 or dset_time_ref[-1] < t11:
        print('# - Incomplete coverage for the chosen time period.')

        print(f'# - First date available: {t00_s}')
        print(f'# - Last date available: {t11_s}')
        print(f'# - Considered period: {t00_s} - {t11_s}')

    # - create year, month, and day dataframe columns
    print(dem_df.head())
    dem_df['year'] = \
        dem_df['ntime'].apply(lambda x: x.year)
    dem_df['month'] = \
        dem_df['ntime'].apply(lambda x: x.month)
    dem_df['day'] = \
        dem_df['ntime'].apply(lambda x: x.day)
    # - calculate raster area in km2
    dem_df['area'] = dem_df['Width'] * dem_df['Height'] * (args.res**2) / 1e6

    # - Define Region of Interest boundaries
    if args.shapefile:
        # - Read Region Of Interest Shapefile
        print('# - Load Region of Interest shapefile.')
        # - Load dhdt domain shapefile
        shapefile_name = args.shapefile.split('/')[-1][:-4]
        gdf_dhdt = gpd.read_file(args.shapefile) \
            .to_crs(epsg=args.crs)
        # - Create Output directory
        # out_dir = create_dir(out_dir, shapefile_name)
        print(f'# - ROI name: {shapefile_name}')

    elif args.boundaries:
        # - Import ROI Lat/Lon Boundaries and define GeodataFrame
        # - Polygon Corners.
        lat_min = None
        lat_max = None
        lon_min = None
        lon_max = None

        if args.boundaries is not None:
            bound_list = args.boundaries.split(',')
            lat_min = float(bound_list[0])
            lat_max = float(bound_list[1])
            lon_min = float(bound_list[2])
            lon_max = float(bound_list[3])

        point_list = [(lon_min, lat_min), (lon_min, lat_max),
                      (lon_max, lat_max), (lon_max, lat_min),
                      (lon_min, lon_min)]

        polygon_geom = Polygon(point_list)
        gdf_dhdt \
            = gpd.GeoDataFrame(index=[0], crs='epsg:4326',
                               geometry=[polygon_geom])\
            .to_crs(epsg=str(args.crs))

        shapefile_name = args.boundaries
        print(f'# - ROI name: {shapefile_name}')

    else:
        # - Consider all the available DEMs - define GeodataFrame
        # - Polygon Corners.
        lat_min = -90
        lat_max = 90
        lon_min = -180
        lon_max = 180

        point_list = [(lon_min, lat_min), (lon_min, lat_max),
                      (lon_max, lat_max), (lon_max, lat_min),
                      (lon_min, lon_min)]

        polygon_geom = Polygon(point_list)
        gdf_dhdt \
            = gpd.GeoDataFrame(index=[0], crs='epsg:4326',
                               geometry=[polygon_geom])\
            .to_crs(epsg=str(args.crs))

    # - Apply GeoPandas Spatial Join to include in the calculation
    # - only DEMs with valid data within the ROI.
    dem_df_inter = gpd.sjoin(dem_df, gdf_dhdt, predicate='intersects')
    print(f'# - Number of DEMs with valid'
          f' data within ROI: {len(dem_df_inter.index)}')

    # - Extract Reference Dataframe subset to evaluate data coverage.
    dem_df_area = dem_df_inter[['ntime', 'area']]
    # - Number of DEMs available for the selected sampling period
    dem_df_area_cnt = dem_df_area['area']\
        .groupby(pd.Grouper(freq=args.freq)).count()
    dem_count = dem_df_area_cnt.values
    cnt_date = [dates.date2num(x) for x in dem_df_area_cnt.index]

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # - Average Area Covered per sampling period
    ax1.set_title('TanDEM-X Data Availability', weight='bold', loc='left')
    ax1.bar(cnt_date, dem_count, width=30, color='#397aec', edgecolor='#0a4dc2')
    ax1.set_ylabel('[Count]', weight='bold')
    ax1.set_xlabel('[time]', weight='bold')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(int(n_month/6)))
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%Y-%b'))
    ax1.grid(color='k', linestyle='dotted', alpha=0.3)
    plt.gcf().autofmt_xdate()
    # - grid
    plt.show()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"# - Computation Time: {end_time - start_time}")
