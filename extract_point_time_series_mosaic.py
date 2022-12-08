#!/usr/bin/env python
u"""
extract_point_time_series_mosaic.py
Written by Enrico Ciraci' (11/2022)

Calculate elevation time series by employing estimates from time-tagged
TanDEM-X DEM MOSAICS. Time series are calculated at the selected coordinates
that can be provided in the form of a Point/Multipoint esri shapefile or as a
single coordinates pair provided as comma-separated values.

See extract_point_time_series.py for more info.

NOTE: In this version of the script, no DEMs index file is ude to verify if the
    selected sampling point are actually  located inside the area covered by
    the DEMs time series.


COMMAND LINE OPTIONS:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -D DIRECTORY
                        Project data directory.
  --outdir OUTDIR, -O OUTDIR
                        Output directory.
  --coords COORDS, -C COORDS
                        Point Coordinates (WGS84) - Pt. Lat, Pt.Lon
  --shapefile SHAPEFILE, -S SHAPEFILE
                        Absolute path to the shapefile containing the
                        coordinates of the locations to consider.
  --win_size WIN_SIZE, -W WIN_SIZE
                        Moving Median Filter Window Size.
  --out_thresh OUT_THRESH, -T OUT_THRESH
                        Outliers Detection Threshold.

Note: This preliminary version of the script has been developed to process
      TanDEM-X data available between 2011 and 2020 for the area surrounding
      Petermann Glacier (Northwest Greenland).

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    matplotlib: Library for creating static, animated, and interactive
           visualizations in Python.
           https://matplotlib.org/
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
    pyproj: Python interface to PROJ (cartographic projections and coordinate
           transformations library).
           https://pyproj4.github.io/pyproj/stable

UPDATE HISTORY:

"""
# - Python Dependencies
from __future__ import print_function
import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from shapely.geometry import Point
import geopandas as gpd
from pyproj import CRS
from pyproj import Transformer
import matplotlib.pyplot as plt
from utility_functions_tdx import dem_2_skip
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-v0_8-deep')


def create_dir(abs_path: str, dir_name: str) -> str:
    """
    Create directory
    :param abs_path: absolute path to the output directory
    :param dir_name: new directory name
    :return: absolute path to the new directory
    """
    import os
    dir_to_create = os.path.join(abs_path, dir_name)
    if not os.path.exists(dir_to_create):
        os.mkdir(dir_to_create)
    return dir_to_create


def main() -> None:
    # - Extract Elevation time series evaluated at the selected Geographic
    # - Locations.
    parser = argparse.ArgumentParser(
        description="""Extract Elevation time series for the provided 
        Geographic Coordinates [Use Daily Mosaics]"""
    )
    # - Project data directory.
    default_dir = os.environ['PYTHONDATA']
    parser.add_argument('--directory', '-D',
                        type=lambda p: os.path.abspath(os.path.expanduser(p)),
                        default=default_dir,
                        help='Project data directory.')
    # - Output directory
    parser.add_argument('--outdir', '-O',
                        type=str,
                        default=default_dir,
                        help='Output directory.')

    parser.add_argument('--coords', '-C', type=str, default=None,
                        help='Point Coordinates (WGS84) - Pt. Lat, Pt.Lon')

    parser.add_argument('--shapefile', '-S', type=str, default=None,
                        help='Absolute path to the shapefile containing the'
                             ' coordinates of the locations to consider.')

    parser.add_argument('--crs', type=int, default=3413,
                        help='Coordinate Reference System - def. EPSG:3413')

    parser.add_argument('--res', '-R', type=int,
                        default=150,
                        help='Input raster resolution.')

    parser.add_argument('--win_size', '-W', type=int, default=30,
                        help='Moving Median Filter Window Size.')

    parser.add_argument('--out_thresh', '-T', type=int, default=7,
                        help='Outliers Detection Threshold.')

    args = parser.parse_args()

    if not args.coords and not args.shapefile:
        print('# - Provide Points of interest coordinates. See Options:')
        print('# - --coords, -C : Single Point Coordinates - Pt. Lat, Pt.Lon')
        print('# - --shapefile, -S : Absolute path the Shapefile containing the'
              ' coordinates of the locations to consider.')
        sys.exit()

    # - GDAL Binding [Rasterio (rio) or GDAL (gdal)]
    gdal_binding = 'rio'
    # - TanDEM-X DEMs reprojection algorithm
    resampling_alg = 'average'

    # - create output directory
    out_dir = create_dir(os.path.join(args.outdir, 'TanDEM-X'),
                         'TanDEM-X_Point_Time_Series_Mosaic')

    # - Path to DEM directory
    dem_dir = os.path.join(args.directory, 'TanDEM-X',
                           'Petermann_Glacier_out', 'Mosaics',
                           f'Petermann_Glacier_Mosaics_EPSG-{args.crs}'
                           f'_res-{args.res}_ralg-{resampling_alg}'
                           f'_{gdal_binding}_poly0')

    # - Load TanDEM-X index shapefile
    index_file = os.path.join(dem_dir,
                              'petermann_tandemx_dem_mosaics_index.shp')

    # - Figure Parameters - Not Editable
    fig_format = 'jpeg'
    dpi = 150
    label_size_rc = 16
    p_color = '#0000cc'
    p_marker = "o"
    lw = 2
    m_size = 6
    # - Unit Conversion
    nano_sec_2_year = 365 * 24 * 60 * 60 * 1e9

    # - Read DEM index
    print('# - Load TanDEM-X DEMs Index.')
    dem_df = gpd.read_file(index_file).to_crs(epsg=3413)

    # - The TanDEM-X index files reports the DEMs bounds polygons in
    dem_df['datetime'] = pd.DatetimeIndex(dem_df['time'])
    dem_df['ntime'] = dem_df['datetime']
    # - Add time-tag column
    time_tag_list = []
    dem_df['time-tag'] = np.nan
    print('# - Number of DEMs available: {}'.format(len(dem_df.index)))

    print('# - Remove unusable DEMs.')
    for index, row in dem_df.iterrows():
        time_tag = (str(row['datetime'].year) + '-'
                    + str(row['datetime'].month).zfill(2) + '-'
                    + str(row['datetime'].day).zfill(2))
        time_tag_list.append(time_tag)
        dem_df.at[index, 'time-tag'] = time_tag

    for t_tag in dem_2_skip():
        index_d = dem_df[dem_df['time-tag'] == t_tag].index
        dem_df.drop(index_d, inplace=True)
    print('# - Number of DEMs available: {}'.format(len(dem_df.index)))

    # - Drop not necessary columns and set new index column
    dem_df = dem_df.drop(['time-tag'], axis=1)  # - drop original time axis
    dem_df = dem_df.set_index('datetime')
    dem_df = dem_df.sort_index()

    # - create year, month, and day axis
    dem_df['year'] = \
        dem_df['ntime'].apply(lambda x: x.year)
    dem_df['month'] = \
        dem_df['ntime'].apply(lambda x: x.month)
    dem_df['day'] = \
        dem_df['ntime'].apply(lambda x: x.day)

    # - Points of Interest Coordinates
    # - Single Point - Point Coordinates provided as csv pair.
    if args.coords is not None:
        # - Define Output Projection transformation
        crs_4326 = CRS.from_epsg(4326)  # - input projection
        crs_3413 = CRS.from_epsg(3413)  # - default projection
        transformer = Transformer.from_crs(crs_4326, crs_3413)
        coords_list = args.coords.split(',')  # - read coordinates
        c_points = transformer.transform(float(coords_list[1]),
                                         float(coords_list[0]))
        pt_coords = gpd.GeoSeries([Point(c_points)])
        df1 = gpd.GeoDataFrame({'geometry': pt_coords, 'df1': 1,
                                'x': coords_list[0], 'y': coords_list[1]}) \
            .set_crs(epsg=3413, inplace=True)

        print('# - Calculating Elevation Time Series at the following '
              'coordinates:')
        print('# - Latitude:  {}'.format(coords_list[1]))
        print('# - Longitude: {}'.format(coords_list[0]))

        # - Find DEMs with valid elevation data running a spatial join between
        # - the GeoDataFrame created using the point of interest coordinates
        # - and the GeoDataFrame associated with the TanDEM-X index shapefile.
        res_contains = gpd.sjoin(dem_df, df1, predicate='contains')

        # - read  single observation time stamp and generate time axis
        time_ax = pd.to_datetime(list(res_contains['time']))
        # - generate new index column which values go from 0 to n.rows-1
        res_contains['index'] = np.arange(len(res_contains['time']))
        res_contains = res_contains.set_index('index')

        # - Initialize elevation time series vector
        elev_ts = np.zeros(len(time_ax))
        for index, row in res_contains.iterrows():
            dem_name = row['Name']
            # - Import DEM data
            # - List input data directory content
            f_name = [os.path.join(dem_dir, x) for x in os.listdir(dem_dir)
                      if dem_name in x][0]

            # - extract elevation value at the considered location
            with rasterio.open(f_name) as src:
                elev_sample = [x[0] for x in src.sample([c_points])][0]
                elev_ts[index] = elev_sample

        # - Set input raster fill values equal to nan
        elev_ts[elev_ts == src.nodata] = np.nan
        # - Save - Point Elevation Time Series
        elev_ts_da = xr.DataArray(data=elev_ts, dims=["time"],
                                  name='Elevation',
                                  coords=dict(time=time_ax),
                                  attrs=dict(name='Elevation',
                                             description="Elevation [m] - "
                                                         "TanDEM-X",
                                             units="m",
                                             actual_range=[np.min(elev_ts),
                                                           np.max(elev_ts)])
                                  )
        print("# - Number of Observations: {}".format(len(elev_ts)))
        if len(elev_ts) >= 2 * args.win_size:
            if np.where(np.isfinite(elev_ts))[0].size == 0:
                print('# - Not Valid Observations Found.')
                sys.exit()
            # - Apply Moving Median Filter with window size - win_size
            # - NOTE: apply the filter only if the time series is longer
            # -       more than 2-times the selected windows size.
            elev_ts_da_median \
                = elev_ts_da.rolling(min_periods=args.win_size,
                                     center=True,
                                     time=args.win_size).median()
            elev_ts_da_median = elev_ts_da_median.bfill("time") \
                .ffill("time")
            # - Mark as outliers all the points that differ from the local
            # - median by a value larger or equal than "out_thresh".
            elev_ts_da = elev_ts_da \
                .where(np.abs(elev_ts_da.data - elev_ts_da_median.data)
                       <= args.out_thresh).dropna("time")
            elev_ts_no_out = elev_ts_da.data
            if np.where(np.isfinite(elev_ts_no_out))[0].size == 0:
                print('# - Not Valid Observations Found.')
                sys.exit()
            elev_ts_da.attrs['actual_range'] \
                = [np.nanmin(elev_ts_no_out), np.nanmax(elev_ts_no_out)]
            print("# - Number of Outliers Found: {}"
                  .format(len(elev_ts) - len(elev_ts_no_out)))
        print(' ')
        elev_ts_da.to_netcdf(os.path.join(out_dir,
                                          'point_elev_ts_meter_coords({},{}).nc'
                                          .format(coords_list[1],
                                                  coords_list[0])),
                             mode='w', format='NETCDF4')

        # - Calculate Linear Trend in Elevation
        poly_c = elev_ts_da.polyfit("time", 1,
                                    skipna=True)[
            'polyfit_coefficients'].values
        trend = np.round(poly_c[0] * nano_sec_2_year, decimals=3)

        # - plot elevation time series
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Lat: {} - Lon: {}"
                     .format(np.around(float(coords_list[1]), decimals=3),
                             np.around(float(coords_list[0]), decimals=3)),
                     weight='bold', loc='left', size=label_size_rc)
        elev_ts_da.plot.line(color=p_color, marker=p_marker, lw=lw,
                             markersize=m_size, ax=ax)
        ax.grid(color='k', linestyle='dotted', alpha=0.3)

        # - Annotate linear trend
        txt = r'Linear Trend {} m/year'.format(trend)
        ax.annotate(txt, xy=(0.03, 0.03), xycoords="axes fraction",
                    size=label_size_rc, zorder=100,
                    bbox=dict(boxstyle="square", fc="w"))
        # - ticks prop
        ax.xaxis.label.set_weight('bold')
        ax.xaxis.label.set_size(label_size_rc)
        ax.yaxis.label.set_weight('bold')
        ax.yaxis.label.set_size(label_size_rc)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(label_size_rc - 2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(label_size_rc - 2)
        # - save output figure
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, 'point_elev_ts_meter_coords({},{}).{}')
            .format(coords_list[1], coords_list[0], fig_format),
            format=fig_format, dpi=dpi)
        plt.close()

    # - Multipoint - Shapefile Input
    if args.shapefile is not None:
        # - Save time series for all the points loaded from the shapefile
        # - inside a subdirectory named as the shapefile.
        shapefile_name = args.shapefile.split('/')[-1][:-4]
        out_dir = create_dir(out_dir, shapefile_name)
        gdf_pt = gpd.read_file(args.shapefile).to_crs(epsg=3413)

        # - Define Output Projection transformation
        crs_4326 = CRS.from_epsg(4326)  # - input projection
        crs_3413 = CRS.from_epsg(3413)  # - default projection
        transformer = Transformer.from_crs(crs_3413, crs_4326)

        for point in gdf_pt.geometry:
            try:
                # - Point(Multipoint) Object
                pt_coords = gpd.GeoSeries([point[0]])
                df1 = gpd.GeoDataFrame({'geometry': pt_coords, 'df1': 1,
                                        'x': point[0].x,
                                        'y': point[0].y}).set_crs(epsg=3413,
                                                                  inplace=True)
                c_points = transformer.transform(float(point[0].x),
                                                 float(point[0].y))

            except TypeError:
                # - Point(Point) Object
                pt_coords = gpd.GeoSeries([point])
                df1 = gpd.GeoDataFrame({'geometry': pt_coords, 'df1': 1,
                                        'x': point.x,
                                        'y': point.y}).set_crs(epsg=3413,
                                                               inplace=True)
                c_points = transformer.transform(float(point.x),
                                                 float(point.y))

            print('# - Calculating Elevation Time Series at the following '
                  'coordinates:')
            print('# - Latitude:  {}'.format(c_points[0]))
            print('# - Longitude: {}'.format(c_points[1]))

            # - Find DEMs with valid elevation data running a spatial join
            # - between  the GeoDataFrame created using the point of interest
            # - coordinates and the GeoDataFrame associated with the
            # - TanDEM-X index shapefile.
            res_contains = gpd.sjoin(dem_df, df1, predicate='contains')
            # - Generate Time axis
            time_ax = pd.to_datetime(list(res_contains['time']))

            # - generate new index column which values go from 0 to n.rows-1
            res_contains['index'] = np.arange(len(res_contains['time']))
            res_contains = res_contains.set_index('index')

            # - Initialize elevation time series vector
            elev_ts = np.zeros(len(time_ax))
            for index, row in res_contains.iterrows():
                dem_name = row['Name']
                # - Import DEM data - List input data directory content
                f_name = [os.path.join(dem_dir, x) for x in os.listdir(dem_dir)
                          if dem_name in x][0]

                # - extract elevation value at the considered location
                with rasterio.open(f_name) as src:
                    try:
                        # - Point(Point) Object
                        elev_sample = [x[0] for x in
                                       src.sample([(point[0].x,
                                                    point[0].y)])][0]
                    except TypeError:
                        # - Point(Multipoint) Object
                        elev_sample = [x[0] for x in
                                       src.sample([(point.x,
                                                    point.y)])][0]
                    elev_ts[index] = elev_sample
            # - Set input raster fill values equal to nan
            elev_ts[elev_ts == src.nodata] = np.nan
            # - Save - Single-Point Elevation Time Series
            elev_ts_da = xr.DataArray(data=elev_ts, dims=["time"],
                                      name='Elevation',
                                      coords=dict(time=time_ax),
                                      attrs=dict(name='Elevation',
                                                 description="Elevation [m]",
                                                 units="m",
                                                 actual_range=[
                                                     np.min(elev_ts),
                                                     np.max(elev_ts)])
                                      )

            print("# - Number of Observations: {}".format(len(elev_ts)))
            if len(elev_ts) >= 2 * args.win_size:
                if np.where(np.isfinite(elev_ts))[0].size == 0:
                    print('# - Not Valid Observations Found.')
                    continue
                # - Apply Moving Median Filter with window size - win_size
                # - NOTE: apply the filter only if the time series is longer
                # -       more than 2-times the selected windows size.
                elev_ts_da_median \
                    = elev_ts_da.rolling(min_periods=int(args.win_size / 2),
                                         center=True,
                                         time=args.win_size).median()
                elev_ts_da_median = elev_ts_da_median.bfill("time") \
                    .ffill("time")
                # - Mark as outliers all the points that differ from the local
                # - median by a value larger or equal than "out_thresh".
                elev_ts_da = elev_ts_da \
                    .where(np.abs(elev_ts_da.data - elev_ts_da_median.data)
                           <= args.out_thresh).dropna("time")
                elev_ts_no_out = elev_ts_da.data
                if np.where(np.isfinite(elev_ts_no_out))[0].size == 0:
                    print('# - Not Valid Observations Found.')
                    continue
                elev_ts_da.attrs['actual_range'] \
                    = [np.nanmin(elev_ts_no_out), np.nanmax(elev_ts_no_out)]
                print("# - Number of Outliers Found: {}"
                      .format(len(elev_ts) - len(elev_ts_no_out)))

            print('\n')
            elev_ts_da \
                .to_netcdf(os.path.join(out_dir,
                                        'point_elev_ts_meter'
                                        '_coords({},{}).nc'
                                        .format(c_points[0], c_points[1])),
                           mode='w', format='NETCDF4')

            # - Calculate Linear Trend in Elevation
            poly_c = elev_ts_da.polyfit("time", 1,
                                        skipna=True)[
                'polyfit_coefficients'].values
            trend = np.round(poly_c[0] * nano_sec_2_year, decimals=3)
            # - plot elevation time series
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("Lat: {} - Lon: {}"
                         .format(np.around(float(c_points[0]), decimals=3),
                                 np.around(float(c_points[1]), decimals=3)),
                         weight='bold', loc='left', size=label_size_rc)
            elev_ts_da.plot.line(color=p_color, marker=p_marker, lw=lw,
                                 markersize=m_size, ax=ax)
            ax.grid(color='k', linestyle='dotted', alpha=0.3)

            # - Annotate linear trend
            txt = r'Linear Trend {} m/year'.format(trend)
            ax.annotate(txt, xy=(0.03, 0.03), xycoords="axes fraction",
                        size=label_size_rc, zorder=100,
                        bbox=dict(boxstyle="square", fc="w"))
            # - ticks prop
            ax.xaxis.label.set_weight('bold')
            ax.xaxis.label.set_size(label_size_rc)
            ax.yaxis.label.set_weight('bold')
            ax.yaxis.label.set_size(label_size_rc)

            for t in ax.xaxis.get_major_ticks():
                t.label1.set_fontsize(label_size_rc - 2)
            for t in ax.xaxis.get_major_ticks():
                t.label1.set_fontsize(label_size_rc - 2)

            # - save output figure
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir,
                                     'point_elev_ts_meter_coords({},{}).{}')
                        .format(c_points[0], c_points[1], fig_format),
                        format=fig_format, dpi=dpi)
            plt.close()


# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print("# - Computation Time: {}".format(end_time - start_time))
