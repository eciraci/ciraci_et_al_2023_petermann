# - Python Dependencies
from __future__ import print_function
import os
import sys
import argparse
import datetime
import numpy as np
from scipy import signal
import pandas as pd
import xarray as xr
import rasterio
import fiona
from rasterio import features
import geopandas as gpd
from shapely.geometry import Polygon
from fiona.crs import from_epsg
from calendar import monthrange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from utility_functions import create_dir, add_colorbar
from utility_functions_rio import clip_raster, virtual_warp_rio, \
    load_dem_tiff, save_raster
from tandemx_proc.dem_tdx_temporal_coverage import dem_tdx_temporal_coverage
from utility_functions_tdx import dem_2_skip
# - change matplotlib default setting
plt.rc('font', family='monospace')
plt.rc('font', weight='bold')
plt.style.use('seaborn-v0_8-deep')


def load_ice_mask(data_dir: str, res: int = 150, crs: int = 3413,
                  resampling_alg: str = 'average') -> dict:
    """
    Load ICE Mask From BedMachine v.5.0
    :param data_dir: absolute path to project data directory
    :param res: mask resolution
    :param crs: map coordinate reference system
    :param resampling_alg: gdal warp resampling meth0d.
    :return: python dictionary
    """
    # - Load ICE Mask From BedMachine v.5.0
    bed_machine_path = os.path.join(data_dir, 'GIS_Data',
                                    'BedMachineGreenland',
                                    'BedMachineGreenlandIceMaskv5_'
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

    return {'ice_mask': ice_mask, 'x_coords': ice_x_coords,
            'y_coords': ice_y_coords}


def plot_dhdt_map(data_dir: str, img: np.array, xx: np.array, yy: np.array,
                  out_path: str, ref_crs: int, vp_shp=None,
                  npts_tresh: int = 100, extent: int = 1,
                  land_color: str = 'black', ice_color: str = 'grey',
                  grnd_ln_color: str = 'k', grnd_zn_color: str = 'g',
                  grnd_zn_buffer: float = 0, title: str = '',
                  annotate: str = '', fig_format: str = 'jpeg',
                  cmap=plt.get_cmap('bwr_r'),
                  vmin: int = -10, vmax: int = 10,
                  unit: str = 'm/year',
                  ) -> None:
    """
    Plot Elevation Change DhDt Map - Wide Domain
    :param data_dir: Project Data Directory
    :param img: elevation change [numpy array]
    :param xx: xx m-grid [numpy array]
    :param yy: yy m-grid [numpy array]
    :param out_path: absolute path to output file
    :param ref_crs: coordinates reference system
    :param vp_shp: valid data points boundaries - esri shp
    :param npts_tresh: minimum number of observations required to accept dhdt
    :param extent: figure extent [1,2,3]
    :param land_color: land edges color
    :param ice_color: ice edges color
    :param grnd_ln_color: Grounding Line Color
    :param grnd_zn_color: Grounding Zone Color
    :param grnd_zn_buffer: Grounding Zone Buffer
    :param title: figure title
    :param annotate: add annotate object
    :param fig_format: figure format [jpeg]
    :param cmap: imshow/pcolomeh color map
    :param vmin: imshow/pcolomeh vmin
    :param vmax: imshow/pcolomeh vmax
    :param unit: elevation change unit - label
    :return: None
    """
    if extent == 1:
        # - Map Extent 1 - wide
        map_extent = [-68, -55, 80, 82]
        figsize = (9, 9)
    elif extent == 2:
        # - Map Extent 2 - zoom 1
        map_extent = [-61.5, -57, 79.5, 81.5]
        figsize = (5, 8)
    else:
        # - Map Extent 3 - zoom2
        map_extent = [-61.5, -57.6, 80.2, 81.2]
        figsize = (9, 9)

    # - text size
    txt_size = 14       # - main text size
    leg_size = 13       # - legend text size
    label_size = 12     # - label text size

    # - Path to Ice and Land Masks
    ics_shp = os.path.join('..', 'esri_shp', 'GIMP',
                           'Petermann_Domain_glaciers_wgs84.shp')
    land_shp = os.path.join('..', 'esri_shp', 'GIMP',
                            'GSHHS_i_L1_Petermann_clip.shp')

    # - Petermann Grounding Line - 2021
    gnd_ln_shp = os.path.join(data_dir, 'GIS_Data',
                              'Petermann_GL_selected_yr',
                              '2021.shp')
    gnd_ln_df = gpd.read_file(gnd_ln_shp).to_crs(epsg=ref_crs)
    # -
    xg, yg = gnd_ln_df['geometry'].geometry[0].xy

    # - Petermann Grounding Zone - 2011/2021
    gnd_zn_shp = os.path.join(data_dir, 'GIS_Data',
                              'Petermann_features_extraction',
                              'Petermann_grounding_line_migration_'
                              'range_epsg3413.shp')

    if grnd_zn_buffer:
        # - Clip Output Raster
        clip_shp_mask_path \
            = os.path.join(data_dir, 'GIS_Data',
                           'Petermann_features_extraction',
                           'Petermann_iceshelf_clipped_epsg3413.shp')
        clip_mask = gpd.read_file(clip_shp_mask_path).to_crs(epsg=ref_crs)

        gnd_zn_shp_buff = os.path.join(data_dir, 'GIS_Data',
                                       'Petermann_features_extraction',
                                       'Petermann_grounding_line_migration_'
                                       'range_buff{}_epsg3413.shp'
                                       .format(grnd_zn_buffer))
        if not os.path.isfile(gnd_zn_shp_buff):
            gnd_zn_to_bf = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)
            gnd_zn_to_bf['geometry'] = gnd_zn_to_bf.geometry\
                .buffer(grnd_zn_buffer)
            # - clip the obtained buffered mask with the
            # - ice shelf perimeter mask.
            gnd_zn_to_bf = gpd.overlay(gnd_zn_to_bf, clip_mask,
                                       how='intersection')
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
    leg_label_list = list()
    ax = fig.add_subplot(1, 1, 1, projection=ref_crs)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Plot Coastlines
    shape_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor='None', edgecolor=land_color)
    # - Plot Glaciers Mask
    shape_feature = ShapelyFeature(Reader(ics_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor='None', edgecolor=ice_color)

    # - Plot Grounding Line 2020/2021
    l1, = ax.plot(xg, yg, color=grnd_ln_color, lw=2, zorder=10, ls='-.')
    leg_label_list.append('Grounding Line 2021')
    # - Plot Grounding Zone 2011/2021
    shape_feature = ShapelyFeature(Reader(gnd_zn_shp).geometries(),
                                   ref_crs)
    ax.add_feature(shape_feature, facecolor='None',
                   edgecolor=grnd_zn_color, linestyle='--',
                   linewidth=2)
    leg_label_list.append('Grounding Zone 2011-2021')
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor=grnd_zn_color, facecolor='none',
                            linestyle='--')
    # - Valid Point Shapefile
    if vp_shp is not None:
        shape_feature = ShapelyFeature(Reader(vp_shp).geometries(), ref_crs)
        ax.add_feature(shape_feature, facecolor='None',
                       edgecolor='r', linestyle='dotted', lw=2)
        leg_label_list.append('N. Obs > {}'.format(npts_tresh))
        l3 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                                edgecolor='r', facecolor='none',
                                linestyle='dotted')
    else:
        l3 = None

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
    gl.xlabel_style = {'rotation': 0, 'weight': 'bold', 'size': label_size}
    gl.ylabel_style = {'rotation': 0, 'weight': 'bold', 'size': label_size}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # - Figure title
    ax.set_title(title, weight='bold', loc='left', size=txt_size)
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
    cb.set_label(label='['+unit+']', weight='bold', size=label_size)
    cb.ax.tick_params(labelsize='large')

    # - Add Legend to DhDt Maps
    ax.legend([l1, l2, l3], leg_label_list, loc='upper right',
              fontsize=leg_size, framealpha=0.8,
              facecolor='w', edgecolor='k')

    # - Add ScaleBar
    ax.add_artist(ScaleBar(1, units='m', location='lower right',
                           border_pad=1, pad=0.5, box_color='w',
                           frameon=True))

    # - save output figure
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()


def plot_dhdt_map_zoom(data_dir: str, dhdt_path: str, out_path: str,
                       ref_crs: int, extent: int = 1, land_color: str = 'black',
                       ice_color: str = 'grey', grnd_ln_color: str = 'k',
                       grnd_zn_color: str = 'g', grnd_zn_buffer: float = 0,
                       title: str = '', annotate: str = '',
                       fig_format: str = 'jpeg', cmap=plt.get_cmap('bwr_r'),
                       vmin: int = -10, vmax: int = 10,
                       unit: str = 'm/year',
                       ) -> None:
    """
    Plot Ice Elevation Change DhDt Map - Zoom Domain
    :param data_dir: absolute path to project data directory
    :param dhdt_path: absolute path to dhdt in GeoTiff format
    :param out_path: absolute path to output figure
    :param ref_crs: reference CRS
    :param extent: map extend
    :param land_color: land edges color
    :param ice_color: ice edges color
    :param grnd_ln_color: Grounding Line Color
    :param grnd_zn_color: Grounding Zone Color
    :param grnd_zn_buffer: Grounding Zone Buffer
    :param title: figure title
    :param annotate: add annotate object
    :param fig_format: figure format [jpeg]
    :param cmap: imshow/pcolomeh color map
    :param vmin: imshow/pcolomeh vmin
    :param vmax: imshow/pcolomeh vmax
    :param unit: elevation change unit - label
    :return:
    """
    # - text size
    txt_size = 14       # - main text size
    leg_size = 13       # - legend text size
    label_size = 12     # - label text siz

    # - Plot DhDt Map
    dhdt_plot = load_dem_tiff(dhdt_path)
    dhdt_fig = dhdt_plot['data']
    dhdt_fig[dhdt_fig == dhdt_plot['nodata']] = np.nan
    x_coords = dhdt_plot['x_coords']
    y_coords = dhdt_plot['y_coords']
    xx, yy = np.meshgrid(x_coords, y_coords)

    # - Map Extent
    if extent == 1:
        map_extent = [-61.1, -59.9, 80.4, 81.2]
        figsize = (6, 9)
    else:
        map_extent = [-60.8, -59.1, 80.4, 80.7]
        figsize = (9, 9)

    # - Path to Ice and Land Masks
    ics_shp = os.path.join('..', 'esri_shp', 'GIMP',
                           'Petermann_Domain_glaciers_wgs84.shp')
    land_shp = os.path.join('..', 'esri_shp', 'GIMP',
                            'GSHHS_i_L1_Petermann_clip.shp')
    # - Petermann Grounding Line - 2021
    gnd_ln_shp = os.path.join(data_dir, 'GIS_Data',
                              'Petermann_GL_selected_yr',
                              '2021.shp')
    gnd_ln_df = gpd.read_file(gnd_ln_shp).to_crs(epsg=ref_crs)
    # -
    xg, yg = gnd_ln_df['geometry'].geometry[0].xy

    # - Petermann Grounding Zon - 2011/2021
    gnd_zn_shp = os.path.join(data_dir, 'GIS_Data',
                              'Petermann_features_extraction',
                              'Petermann_grounding_line_migration_'
                              'range_epsg3413.shp')

    if grnd_zn_buffer:
        # - Clip Output Raster
        clip_shp_mask_path \
            = os.path.join(data_dir, 'GIS_Data',
                           'Petermann_features_extraction',
                           'Petermann_iceshelf_clipped_epsg3413.shp')
        clip_mask = gpd.read_file(clip_shp_mask_path).to_crs(epsg=ref_crs)

        gnd_zn_shp_buff = os.path.join(data_dir, 'GIS_Data',
                                       'Petermann_features_extraction',
                                       'Petermann_grounding_line_migration_'
                                       'range_buff{}_epsg3413.shp'
                                       .format(grnd_zn_buffer))
        if not os.path.isfile(gnd_zn_shp_buff):
            gnd_zn_to_bf = gpd.read_file(gnd_zn_shp).to_crs(epsg=ref_crs)
            gnd_zn_to_bf['geometry'] = gnd_zn_to_bf.geometry\
                .buffer(grnd_zn_buffer)
            # - clip the obtained buffered mask with the
            # - ice shelf perimeter mask.
            gnd_zn_to_bf = gpd.overlay(gnd_zn_to_bf, clip_mask,
                                       how='intersection')
            # - save buffered mask to file
            gnd_zn_to_bf.to_file(gnd_zn_shp_buff)

        # -
        gnd_zn_shp = gnd_zn_shp_buff

    # - set Coordinate Reference System
    ref_crs = ccrs.NorthPolarStereo(central_longitude=-45,
                                    true_scale_latitude=70)
    fig = plt.figure(figsize=figsize)
    # - initialize legend labels
    leg_label_list = list()

    # - Plot DhDt Map
    ax = fig.add_subplot(1, 1, 1, projection=ref_crs)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    # - Plot Coastlines
    shape_feature = ShapelyFeature(Reader(land_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor='None', edgecolor=land_color)
    ax.add_feature(shape_feature, facecolor='None', edgecolor=land_color)
    # - Plot Glaciers Mask
    shape_feature = ShapelyFeature(Reader(ics_shp).geometries(),
                                   ccrs.PlateCarree())
    ax.add_feature(shape_feature, facecolor='None', edgecolor=ice_color)
    # - Plot Grounding Line 2020/2021
    l1, = ax.plot(xg, yg, color=grnd_ln_color, lw=2, zorder=10, ls='-.')
    leg_label_list.append('Grounding Line 2021')
    # - Plot Grounding Zone 2011/2021
    shape_feature = ShapelyFeature(Reader(gnd_zn_shp).geometries(),
                                   ref_crs)
    ax.add_feature(shape_feature, facecolor='None',
                   edgecolor=grnd_zn_color, linestyle='--',
                   linewidth=2)
    leg_label_list.append('Grounding Zone 2011-2021')
    l2 = mpatches.Rectangle((0, 0), 1, 0.1, linewidth=2,
                            edgecolor=grnd_zn_color, facecolor='none',
                            linestyle='--')

    # - Set Map Grid
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False,
                      y_inline=False, color='k', linestyle='dotted',
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
    gl.xlabel_style = {'rotation': 0, 'weight': 'bold', 'size': label_size}
    gl.ylabel_style = {'rotation': 0, 'weight': 'bold', 'size': label_size}
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
    cb.set_label(label='['+unit+']', weight='bold', size=label_size)
    cb.ax.tick_params(labelsize='medium')

    # - Figure title
    ax.set_title(title, weight='bold', loc='left', size=txt_size)
    # - Add Figure Annotation
    ax.annotate(annotate, xy=(0.03, 0.03), xycoords="axes fraction",
                size=label_size, zorder=100,
                bbox=dict(boxstyle="square", fc="w", alpha=0.8))

    # - Add Legend to DhDt Maps
    ax.legend([l1, l2], leg_label_list, loc='upper right',
              fontsize=leg_size, framealpha=0.8,
              facecolor='w', edgecolor='k')

    # - Add ScaleBar
    ax.add_artist(ScaleBar(1, units='m', location='lower right',
                           border_pad=1, pad=0.5, box_color='w',
                           frameon=True))

    # - save output figure
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="""Calculate Elevation Change [Dh/Dt] maps over the 
        selected regions of interest."""
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

    parser.add_argument('--boundaries', '-B', type=str, default=None,
                        help='Domain BBOX (WGS84) - Lat Min,Lat Max,'
                             'Lon Min,Lon Max')

    parser.add_argument('--shapefile', '-S', type=str,
                        # - Uncomment this default value during Development
                        default=os.path.join(default_dir, 'GIS_Data',
                                             'Petermann_Domain',
                                             'dhdt_trend_domain.shp'),
                        # default=None,
                        help='Absolute path to the shapefile containing the'
                             ' coordinates of the locations to consider.')

    parser.add_argument('--crs', '-T',
                        type=int, default=3413,
                        help='Coordinate Reference System - def. EPSG:3413')

    parser.add_argument('--res', '-R', type=int,
                        default=150,
                        help='Input raster resolution.')

    # - NOTE: f_date and l_date are hardcoded here, and they should be kept
    # -       is the TanDEM-X dataset is extended in time.
    parser.add_argument('--f_date', '-F', type=str, default='2011,6,9,0,0,0',
                        help='Starting Date of the Considered Period of time.'
                             'Passed as year,month,day or reference year only.')

    parser.add_argument('--l_date', '-L', type=str,
                        default='2021,12,1,23,59,59',
                        help='Ending Date of the Considered Period of time.'
                             'Passed as year,month,day or reference year only.')

    parser.add_argument('--smooth', type=str,
                        choices=['nosmooth', 'median', 'average'],
                        help='Smoothing Filter.', default='nosmooth')

    # - Spatial Smoothing: Kernel Radius
    parser.add_argument('--w_size', '-W', type=int, default=1000,
                        help='Smoothing Window Size [in meters]')

    args = parser.parse_args()

    if not args.boundaries and not args.shapefile:
        print('# - Provide ROI of interest coordinates. See Options:')
        print('# - --boundaries, -B : SLat Min,Lat Max,Lon Min,Lon Max')
        print('# - --shapefile, -S : Absolute path the Shapefile containing'
              ' the coordinates of the locations to consider.')
        sys.exit()
    elif args.boundaries and args.shapefile:
        print('# - Chose ROI boundaries using either (--boundaries, -B) or '
              '(--shapefile, -S) options.')
        sys.exit()

    # - Starting and Ending Date of the considered period of time
    t_start = [int(x) for x in args.f_date.split(',')]
    t_end = [int(x) for x in args.l_date.split(',')]

    # - Reference dates provided as a single year values
    if len(t_start) == 1:
        # - If only a year value is provided as input,
        # - consider the period starting from the first of January
        # - of that yer.
        t_start = [t_start[0], 1, 1]
    elif len(t_start) == 2:
        # - Year and month values are provided as input.
        # - Consider the entire  month.
        t_start = [t_start[0], t_start[1], 1]

    if len(t_end) == 1:
        # - If only a year value is provided as input, consider the
        # - last day of January of that year as the ending date.
        t_end = [t_end[0], 12, 31, 23, 59, 59]

    elif len(t_end) == 2:
        # - Year and month values are provided as input.
        # - Consider the entire  month.
        t_end = [t_end[0], t_end[1], monthrange(t_end[0], t_end[1])[1],
                 23, 59, 59]
    # -
    t00 = datetime.datetime(*t_start)
    t11 = datetime.datetime(*t_end)
    n_month = (((t11.year - t00.year) * 12) + (t11.month - t00.month))

    if t11 < t00:
        print('# - t_end < t_start selected.')
        sys.exit()

    # - GDAL Binding [Rasterio (rio) or GDAL (gdal)]
    gdal_binding = 'rio'
    # - TanDEM-X DEMs resampling algorithm
    resampling_alg = 'average'

    # - minimum number of valid data points tha must be available
    # - to consider the estimated trend valid.
    npts_tresh = (t11.year - t00.year) * 12
    print('# - npts_tresh = {}'.format(npts_tresh))

    # - Calculate Dhdt Smoothing filter Kernel size in #pixels
    smth_kernel_size = int(args.w_size/args.res)
    if smth_kernel_size % 2 == 0:
        smth_kernel_size += 1       # - Kernel Size must be an odd number.

    # - smoothing kernel - suffix str
    if args.smooth == 'nosmooth':
        smth_suff = ''
    else:
        smth_suff = '_{}'.format(args.w_size)

    # - Path to DEM directory
    input_data_path \
        = os.path.join(args.directory, 'TanDEM-X', 'Petermann_Glacier_out',
                       'Mosaics', f'Petermann_Glacier_Mosaics_EPSG-{args.crs}'
                                  f'_res-{args.res}_ralg-{resampling_alg}'
                                  f'_{gdal_binding}_poly0')
    # - Load TanDEM-X index shapefile
    index_file = os.path.join(input_data_path,
                              'petermann_tandemx_dem_mosaics_index.shp')

    # - Read DEM index
    print('# - Load TanDEM-X DEMs Index.')
    dem_df = gpd.read_file(index_file).to_crs(epsg=3413)
    print('# - Number of DEMs available: {}'.format(len(dem_df.index)))
    # - The TanDEM-X index files reports the DEMs bounds polygons in
    dem_df['datetime'] = pd.DatetimeIndex(dem_df['time'])
    dem_df['ntime'] = dem_df['datetime']
    # - Add time-tag column
    time_tag_list = []
    dem_df['time-tag'] = np.nan

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
    dset_time_ref = list(dem_df['ntime'])
    # -
    print('# - Number of DEM Mosaics available: {}'.format(len(dem_df.index)))

    if dset_time_ref[0] > t00 or dset_time_ref[-1] < t11:
        print('# - Full data coverage not available for the selected period.')
        print('# - First date available: {}'
              .format(dset_time_ref[0].strftime('%Y-%m-%d')))
        print('# - Last date available:  {}'
              .format(dset_time_ref[-1].strftime('%Y-%m-%d')))
        if dset_time_ref[0] > t00:
            t00 = dset_time_ref[0]
        if dset_time_ref[-1] < t11:
            t11 = dset_time_ref[-1]
    else:
        print(f'# - Considered period: {t00.strftime("%Y-%m-%d")} '
              f'- {t11.strftime("%Y-%m-%d")}')

    # - Petermann Glacier Maks extracted BedMachine
    ice_mask_input = load_ice_mask(args.directory, res=args.res,
                                   crs=args.crs, resampling_alg=resampling_alg)
    ice_mask = ice_mask_input['ice_mask']
    ice_x_coords = ice_mask_input['x_coords']
    ice_y_coords = ice_mask_input['y_coords']

    # - Create Output Directory
    out_dir = create_dir(os.path.join(args.outdir, 'TanDEM-X'),
                         'TanDEM-X_DHDT_Eulerian_Maps_Mosaics')
    out_dir = create_dir(out_dir, '{}_{}'.format(t00.strftime('%Y-%m-%d'),
                                                 t11.strftime('%Y-%m-%d')))

    # - extract data relative to the considered time period
    t_range = ((dem_df['ntime'] > t00) & (dem_df['ntime'] <= t11))
    dem_df = dem_df.loc[t_range]

    print('# - Number of DEMs available for the selected period: '
          '{}'.format(len(dem_df.index)))

    # - Define Region of Interest boundaries
    if args.shapefile:
        # - Read Region Of Interest Shapefile
        print('# - Load Region of Interest shapefile.')
        # - Load dhdt domain shapefile
        shapefile_name = args.shapefile.split('/')[-1][:-4]
        gdf_dhdt = gpd.read_file(args.shapefile)\
            .to_crs(epsg=args.crs)
        # - Create Output directory
        out_dir = create_dir(out_dir, shapefile_name)
        print('# - ROI name: {}'.format(shapefile_name))

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
        out_dir = create_dir(out_dir, shapefile_name)
        print('# - ROI name: {}'.format(shapefile_name))

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

        shapefile_name = args.boundaries
        out_dir = create_dir(out_dir, shapefile_name)
        print('# - ROI name: {}'.format(shapefile_name))

    # - create output directory
    out_dir_name = 'dhdt_[m*year-1]_crs{}' \
                   '_res{}_{}{}'.format(args.crs, args.res,
                                        args.smooth, smth_suff)

    out_dir = create_dir(out_dir, out_dir_name)

    dem_df_inter = dem_df
    print('# - Number of DEMs with valid data within ROI: {}'
          .format(len(dem_df_inter.index)))

    # -  List files available for the selected reference period
    input_file_list = list(dem_df_inter['Name'])
    # - read single observation time stamp and generate time axis
    time_ax = pd.to_datetime(list(dem_df_inter['ntime']))

    # - Save figure showing TanDEM-X elevation data coverage for the
    # - considered temporal interval.
    # - Extract Reference Dataframe subset to evaluate data coverage.
    # - calculate raster area in km2
    dem_df_inter['area'] = (dem_df_inter['Width'] * dem_df_inter['Height']
                            * (args.res ** 2) / 1e6)
    dem_df_area = dem_df_inter[['ntime', 'area']]
    # -
    fig_format = 'jpeg'
    data_coverage_fig = os.path.join(out_dir,
                                     'tdem_x_data_coverage.' + fig_format)
    dem_tdx_temporal_coverage(data_coverage_fig, dem_df_area, n_month=n_month,
                              fig_format=fig_format)
    # # - Define Computation Grid
    min_x, min_y, max_x, max_y = 0, 0, 0, 0

    mosaic_dict = dict()
    for cnt, dem_name in enumerate(input_file_list):
        # - Import DEM data
        # - List input data directory content
        f_name = [os.path.join(input_data_path, x)
                  for x in os.listdir(input_data_path)
                  if dem_name in x][0]

        # - Initialize mosaic_dict dictionary
        # - It will be used to temporary store the elevation data
        # - needed for the computation.
        mosaic_dict[dem_name] = dict()
        src_dem = load_dem_tiff(f_name)
        # - read band #1 - elevation in meters
        mosaic_dict[dem_name]['elev'] = src_dem['data']
        # - set no-data grid points to NaN
        mosaic_dict[dem_name]['elev'][mosaic_dict[dem_name]['elev']
                                      == src_dem['nodata']] = np.nan
        # - evaluate raster size
        mosaic_dict[dem_name]['npts'] = src_dem['width'] * src_dem['height']
        # - raster upper-left and lower-right corners
        mosaic_dict[dem_name]['ul_corner'] = src_dem['ul_corner']
        mosaic_dict[dem_name]['lr_corner'] = src_dem['lr_corner']

        # - Define Patch Bounds
        ptch_min_x = mosaic_dict[dem_name]['ul_corner'][0]
        ptch_min_y = mosaic_dict[dem_name]['lr_corner'][1]
        ptch_max_x = mosaic_dict[dem_name]['lr_corner'][0]
        ptch_max_y = mosaic_dict[dem_name]['ul_corner'][1]
        mosaic_dict[dem_name]['bounds'] = [ptch_min_x, ptch_min_y,
                                           ptch_max_x, ptch_max_y]
        # - Find Mosaic Domain Bounds
        if cnt == 0:
            min_x = ptch_min_x
            min_y = ptch_min_y
            max_x = ptch_max_x
            max_y = ptch_max_y
        else:
            if ptch_min_x < min_x:
                min_x = ptch_min_x
            if ptch_min_y < min_y:
                min_y = ptch_min_y
            if ptch_max_x > max_x:
                max_x = ptch_max_x
            if ptch_max_y > max_y:
                max_y = ptch_max_y

    # - Round Bounding-box values and define mosaic coordinate axes grids
    mosaic_vect_x = np.arange(min_x, max_x + 1, args.res)
    mosaic_vect_y = np.arange(min_y, max_y + 1, args.res)

    # - create mosaic domain coordinates grids meshgrid
    m_xx, m_yy = np.meshgrid(mosaic_vect_x, mosaic_vect_y)

    # - Mosaic Stack to be used to save temporary aligned DEMs
    mosaic_shape = (len(input_file_list), m_xx.shape[0], m_xx.shape[1])
    mosaic_stack = np.full(mosaic_shape, np.nan, dtype=np.float32)

    # - crop Ice Mask
    ice_ind_x = np.where((ice_x_coords >= mosaic_vect_x[0])
                         & (ice_x_coords <= mosaic_vect_x[-1]))[0]
    ice_ind_y = np.where((ice_y_coords >= mosaic_vect_y[0])
                         & (ice_y_coords <= mosaic_vect_y[-1]))[0]
    ice_ind_xx, ice_ind_yy = np.meshgrid(ice_ind_x, ice_ind_y)
    ice_mask_crop = ice_mask[ice_ind_yy, ice_ind_xx]

    # - Add each of the DEMs available for the period to the stack.
    for cnt, dem_name in enumerate(mosaic_dict.keys()):
        dem_info_d = mosaic_dict[dem_name]
        ind_x = np.where((mosaic_vect_x >= mosaic_dict[dem_name]['bounds'][0])
                         & (mosaic_vect_x
                            < mosaic_dict[dem_name]['bounds'][2]))[0]
        ind_y = np.where((mosaic_vect_y >= mosaic_dict[dem_name]['bounds'][1])
                         & (mosaic_vect_y
                            < mosaic_dict[dem_name]['bounds'][3]))[0]
        ind_xx, ind_yy = np.meshgrid(ind_x, ind_y)

        # - NOTE: flipud is necessary here because of the ordering
        # - of the y-axis in GeoTiff rasters. The grid point with
        # - coordinates (0, 0) represents the upper-left corner of
        # - the dataset (i.e. the grid-point associated with
        # - the maximum y value).
        # - secondary raster
        mosaic_stack[cnt, ind_yy, ind_xx] = dem_info_d['elev']

    ds_mosaic = xr.DataArray(mosaic_stack,
                             dims=('time', 'y', 'x'),
                             coords={'time': time_ax,
                                     'y': mosaic_vect_y,
                                     'x': mosaic_vect_x})

    ds_trend = ds_mosaic.polyfit("time", 1, skipna=True)

    # - Unit Conversion - needed beacaue the output of polyfit is provided in
    # - nanoseconds.
    nano_sec_2_year = 365 * 24 * 60 * 60 * 1e9
    trend_coeff = ds_trend['polyfit_coefficients'].values[0] * nano_sec_2_year

    # - extract trend binary mask
    t_binary_mask = np.where(np.isnan(trend_coeff))

    # - set unrealistic elevation change values - abs(dh/dt) > 10 m/year
    # - equal to np.nan
    trend_coeff[np.abs(trend_coeff) >= 10] = np.nan
    # - Set dh/dt values over areas with no ice coverage to np.nan
    trend_coeff[ice_mask_crop == 0] = np.nan

    if args.smooth == 'average':
        # - option 1  - Average filter
        ave_filter = np.ones((smth_kernel_size, smth_kernel_size))
        smoothed = signal.convolve2d(trend_coeff, ave_filter,
                                     mode='same') / np.sum(ave_filter)
    elif args.smooth == 'median':
        # - option 2  - Median filter
        smoothed = signal.medfilt2d(trend_coeff,
                                    kernel_size=smth_kernel_size)
    else:
        # - No smoothing filter applied
        smoothed = trend_coeff

    smoothed[t_binary_mask] = np.nan

    # - save the obtained mosaic
    print('# - Save Elevation Change Trend Map.')
    print('# - Output File Name: ' + shapefile_name
          + '_dhdt_[m*year-1]_crs{}_res{}_{}{}_{}-{}.tiff'
          .format(args.crs, args.res, args.smooth, smth_suff,
                  t00.strftime('%Y-%m-%d'), t11.strftime('%Y-%m-%d')))

    out_path_dhdt = os.path.join(out_dir, shapefile_name
                                 + '_dhdt_[m*year-1]_crs{}_res{}_{}_'
                                   'ws{}_{}-{}.tiff'
                                 .format(args.crs, args.res, args.smooth,
                                         args.w_size, t00.strftime('%Y-%m-%d'),
                                         t11.strftime('%Y-%m-%d')))

    save_raster(smoothed, args.res, mosaic_vect_x, mosaic_vect_y,
                out_path_dhdt, args.crs)

    # - save elevation change map thumbnail
    fig_format = 'jpeg'
    label_size_rc = 12

    # - Save N-Points Mask -> Number of valid elevation measurements x pixel
    out_path_nptsf = os.path.join(out_dir, shapefile_name
                                  + '_dhdt_[m*year-1]_crs{}_res{}_'
                                    '{}_{}-{}_NPTS.{}'
                                  .format(args.crs, args.res, args.smooth,
                                          t00.strftime('%Y-%m-%d'),
                                          t11.strftime('%Y-%m-%d'),
                                          fig_format))

    npts_mask = np.count_nonzero(np.isfinite(mosaic_stack), axis=0)
    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Number of Valid Data Values', weight='bold', loc='left')
    im = ax.imshow(npts_mask, interpolation='nearest',
                   cmap=plt.get_cmap('Blues'))
    # - colorbar
    cb = add_colorbar(fig, ax, im)
    cb.set_label(label='# Obs.', weight='bold')
    cb.ax.tick_params(labelsize='large')
    # - grid
    ax.grid(color='k', linestyle='dotted', alpha=0.3)
    # - save output figure
    plt.tight_layout()
    plt.savefig(out_path_nptsf, dpi=200, format=fig_format)
    plt.close()

    # - convert npts_mask to float
    npts_mask = np.array(npts_mask, dtype=float)

    # - save valid data points mask in geotiff format
    out_path_npts = os.path.join(out_dir, shapefile_name
                                 + '_dhdt_[m*year-1]_crs{}_res{}_'
                                   '{}_{}-{}_NPTS.{}'
                                 .format(args.crs, args.res, args.smooth,
                                         t00.strftime('%Y-%m-%d'),
                                         t11.strftime('%Y-%m-%d'), 'tiff'))

    save_raster(npts_mask, args.res, mosaic_vect_x, mosaic_vect_y,
                out_path_npts, args.crs)

    # - Save valid data point mask > 100 pts boundaries in esri shapefile format
    # - Define output index shapefile schema
    schema = {
        'geometry': 'Polygon',
        'properties': [('Id', 'str')]
    }
    out_path_shp = os.path.join(out_dir, shapefile_name
                                + '_dhdt_[m*year-1]_crs{}_res{}_{}'
                                  '_{}-{}_NPTS.{}'
                                .format(args.crs, args.res, args.smooth,
                                        t00.strftime('%Y-%m-%d'),
                                        t11.strftime('%Y-%m-%d'), 'shp'))
    with rasterio.open(out_path_npts, mode="r+") as src:
        # - read band #1
        pts_mask = src.read(1, masked=True)
        pts_mask_bin = np.zeros(pts_mask.shape).astype('float32')
        pts_mask_bin[pts_mask > npts_tresh] = 1.
        # - save dhdt map in geotiff format
        out_f_name = out_path_shp
        with fiona.open(out_f_name, mode='w', driver='ESRI Shapefile',
                        schema=schema,
                        crs=from_epsg(src.crs.to_epsg())) as poly_shp:
            # - Use rasterio.features.shapes to get valid data region
            # - boundaries. For more details:
            # - https://rasterio.readthedocs.io/en/latest/api/
            # -         rasterio.features.html
            b_shapes = list(features.shapes(pts_mask_bin,
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
                'properties': {'Id': '1'},
            }
            poly_shp.write(row_dict)

    # - Clip Elevation Change Map using NPS valid mask
    out_path_clip = out_path_dhdt.replace('.tiff', '_clipped.tiff')
    clip_raster(out_path_dhdt, out_path_shp, out_path_clip)

    # - Save Smoothed Elevation Change Map - Pixel Domain
    out_path = os.path.join(out_dir, shapefile_name
                            + '_dhdt_[m*year-1]_crs{}_res{}_{}{}_{}-{}.{}'
                            .format(args.crs, args.res, args.smooth, smth_suff,
                                    t00.strftime('%Y-%m-%d'),
                                    t11.strftime('%Y-%m-%d'),
                                    fig_format))
    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(smoothed, vmin=-10, vmax=+10, cmap=plt.get_cmap('bwr_r'),
                   interpolation='nearest')
    title_str = 'DH/Dt: \n{} $-$ {}'.format(t00.strftime('%Y-%m-%d'),
                                            t11.strftime('%Y-%m-%d'))
    ax.set_title(title_str, weight='bold', loc='left')
    # - colorbar
    cb = add_colorbar(fig, ax, im)
    cb.set_label(label='[m/year]', weight='bold')
    cb.ax.tick_params(labelsize='large')
    ax.grid(color='k', linestyle='dotted', alpha=0.3)
    # - Annotate linear trend
    if args.smooth in ['median', 'average']:
        txt = r'Filter [m]: {} ({}x{})'.format(args.smooth,
                                               args.w_size, args.w_size)
    else:
        txt = 'No Smoothing Applied.'
    txt += '\nGrid Spacing [m]: {}'.format(args.res)
    ax.annotate(txt, xy=(0.03, 0.03), xycoords="axes fraction",
                size=label_size_rc, zorder=100)
    # - ticks prop
    ax.xaxis.label.set_weight('bold')
    ax.xaxis.label.set_size(label_size_rc)
    ax.yaxis.label.set_weight('bold')
    ax.yaxis.label.set_size(label_size_rc)

    for t in ax.xaxis.get_major_ticks():
        t.label1.set_fontsize(label_size_rc)
    for t in ax.xaxis.get_major_ticks():
        t.label1.set_fontsize(label_size_rc)

    # - save output figure
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, format=fig_format)
    plt.close()

    # - Save Smoothed Elevation Change Map - Geographic Domain
    # -     Plot DHDT map using cartopy
    out_path = os.path.join(out_dir, shapefile_name
                            + '_dhdt_[m*year-1]_crs{}_res{}_{}{}'
                              '_{}-{}_cartopy.{}'
                            .format(args.crs, args.res, args.smooth, smth_suff,
                                    t00.strftime('%Y-%m-%d'),
                                    t11.strftime('%Y-%m-%d'),
                                    fig_format))
    # - Plot DhDt Map - Wide Domain
    plot_dhdt_map(args.directory, smoothed, m_xx, m_yy, out_path,
                  vp_shp=out_path_shp, npts_tresh=npts_tresh, extent=3,
                  title=title_str, ref_crs=args.crs, annotate=txt,
                  fig_format=fig_format)

    # - Plot DhDt Map - Zoom Domain
    out_path = os.path.join(out_dir, shapefile_name
                            + '_dhdt_[m*year-1]_crs{}_res{}_{}{}'
                              '_{}-{}_cartopy_ZOOM.{}'
                            .format(args.crs, args.res, args.smooth, smth_suff,
                                    t00.strftime('%Y-%m-%d'),
                                    t11.strftime('%Y-%m-%d'),
                                    fig_format))
    # - Plot DhDt Map - Zoom Domain
    plot_dhdt_map_zoom(args.directory, out_path_clip, out_path,
                       ref_crs=args.crs, title=title_str, annotate=txt,
                       grnd_zn_buffer=args.w_size / 2)

    # -
    out_path = out_path.replace('ZOOM', 'GL_ZOOM')
    # - Plot DhDt Map - Zoom Domain - Around Grounding line
    plot_dhdt_map_zoom(args.directory, out_path_clip, out_path, extent=2,
                       ref_crs=args.crs, title=title_str, annotate=txt,
                       grnd_zn_buffer=args.w_size / 2)



# - run main program
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print("# - Computation Time: {}".format(end_time - start_time))
