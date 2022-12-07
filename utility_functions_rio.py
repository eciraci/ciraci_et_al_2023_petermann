#!/usr/bin/env python
"""
Enrico Ciraci (12/2021)
Set of utility functions used to read/write raster data using Rasterio.
Rasterio is and alternative GDAL’s Python bindings using more idiomatic Python
types and protocols compared to the standard GDAL library.

For more info about the Rasterio project see:
https://rasterio.readthedocs.io/en/latest/

For more info about GDAL/OGR in Python see:
https://gdal.org/api/python.html
"""
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
import fiona
import rasterio.mask
from rasterio import MemoryFile
import affine
from pyproj import CRS
import numpy as np
from typing import Union, Any, List


def load_dem_tiff(in_path: str) -> dict:
    # - Load TanDEM-X DEM raster saved in GeoTiff format
    with rasterio.open(in_path, mode="r+") as src:
        # - read band #1 - DEM elevation in meters
        dem_input = src.read(1).astype(src.dtypes[0])
        # - raster upper-left and lower-right corners
        ul_corner = src.transform * (0, 0)
        lr_corner = src.transform * (src.width, src.height)
        grid_res = src.res

        # - compute x- and y-axis coordinates
        x_coords = np.arange(ul_corner[0], lr_corner[0], grid_res[0])
        y_coords = np.arange(lr_corner[1] + grid_res[1],
                             ul_corner[1] + grid_res[1], grid_res[1])
        # - compute raster coordinates mesh-grids
        m_xx, m_yy = np.meshgrid(x_coords, y_coords)

        # - compute raster extent - (left, right, bottom, top)
        extent = [ul_corner[0], lr_corner[0], lr_corner[1], ul_corner[1]]
        # - compute cell centroids
        x_centroids = x_coords + (grid_res[0]/2.)
        y_centroids = y_coords + (grid_res[1]/2.)
        # - rotate the output numpy array in such a way that
        # - the lower-left corner of the raster is considered
        # - the origin of the reference system.
        if src.transform.e < 0:
            dem_input = np.flipud(dem_input)
        # - Compute New Affine Transform
        transform = (Affine.translation(x_coords[0], y_coords[0])
                     * Affine.scale(src.res[0], src.res[1]))

        return{'data': dem_input, 'crs': src.crs, 'res': src.res,
               'y_coords': y_coords, 'x_coords': x_coords,
               'm_xx': m_xx, 'm_yy': m_yy,
               'y_centroids': y_centroids, 'x_centroids': x_centroids,
               'transform': transform, 'src_transform': src.transform,
               'width': src.width, 'height': src.height, 'extent': extent,
               'ul_corner': ul_corner, 'lr_corner': lr_corner,
               'nodata': src.nodata, 'dtype': src.dtypes[0]}


def save_raster(raster: np.ndarray, res: int, x: np.ndarray,
                y: np.ndarray, out_path: str, crs: int,
                nbands: int = 1, nodata: int = -9999) -> None:
    """
    Save the Provided Raster in GeoTiff format
    :param raster: input raster - np.ndarray
    :param res: raster resolution - integer
    :param x: x-axis - np.ndarray
    :param y: y-axis in a figure coordinate system- np.ndarray
    :param crs: - coordinates reference system
    :param out_path: absolute path to output file
    :param nbands: number of raster bands
    :param nodata: output raster no data value
    :return: None
    """
    # - Calculate Affine Transformation of the output raster
    y_t = y.copy()
    if y_t[1] > y_t[0]:
        y_t = np.flipud(y_t)
        y_t += res
        raster = np.flipud(raster)
    transform = (Affine.translation(x[0], y_t[0])
                 * Affine.scale(res, -res))

    out_meta = {'driver': 'GTiff',
                'height': raster.shape[0],
                'width': raster.shape[1],
                'nodata': nodata,
                'dtype': str(raster.dtype),
                'compress': 'lzw',
                'count': nbands,
                'crs': crs,
                'transform': transform}

    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(raster, 1)


def write_tiff(raster: np.ndarray, res_x: int,  res_y: int, x: np.ndarray,
               y: np.ndarray, out_path: str, crs: int, nbands: int = 1,
               nodata: int = -9999) -> None:
    """
    Save the Raster in GeoTiff format
    :param raster: input raster - np.ndarray
    :param res_x: raster resolution x - integer
    :param res_y: raster resolution y - integer
    :param x: x-axis - np.ndarray
    :param y: y-axis in a figure coordinate system - np.ndarray
    :param crs: - coordinates reference system
    :param nbands: number of raster bands - def.1
    :param out_path: absolute path to output file
    :param nodata: output raster no data value
    :return: None
    """
    # - Calculate Affine Transformation of the output raster
    y_t = y.copy()
    if y_t[1] > y_t[0]:
        y_t = np.flipud(y_t)
        y_t += res_y
        raster = np.flipud(raster)
    transform = (Affine.translation(x[0], y_t[0])
                 * Affine.scale(res_x, -res_y))

    out_meta = {'driver': 'GTiff',
                'height': raster.shape[0],
                'width': raster.shape[1],
                'nodata': nodata,
                'dtype': str(raster.dtype),
                'compress': 'lzw',
                'count': nbands,
                'crs': crs,
                'transform': transform}

    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(raster, 1)


def vrt_param(crs, res_x: int, res_y: int, bounds: List[int],
              resampling_alg: str, dtype: str,
              src_nodata: int = -9999, nodata: int = -9999) -> dict:
    """
    Virtual Warp Parameters
    :param crs: destination coordinate reference system
    :param res_x: output resolution x-axis
    :param res_y: output resolution y-axis
    :param bounds: Interpolation Domain Boundaries
    :param resampling_alg: Interpolation Algorithm
    :param dtype: the working data type for warp operation and output.
    :param src_nodata: source nodata value
    :param nodata: output nodata value
    :return: dictionary containing vrt options.
    """
    # - Re-projection Parameters
    dst_crs = CRS.from_epsg(crs)  # - Destination CRS
    # - Output image transform
    left, bottom, right, top = bounds
    dst_width = (right - left) / res_x
    dst_height = (top - bottom) / res_y
    # - Affine transformation matrix
    dst_transform = affine.Affine(res_x, 0.0, left,
                                  0.0, -res_y, top)
    # - Virtual Warping Options
    vrt_options = {
        'resampling': Resampling[resampling_alg],
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
        'src_nodata': src_nodata,
        'nodata': nodata,
        'dtype': dtype,
    }
    return vrt_options


def virtual_warp_rio(src_file: str, out_file: str,
                     res_x: int = 250, res_y: int = 250,
                     bounds: List[int] = None,
                     crs: int = 3413, method: str = 'med',
                     dtype=None, src_nodata: int = -9999,
                     nodata: int = -9999) -> np.ndarray:
    """
    Rasterio Virtual Warp
    :param src_file: absolute path to source file
    :param out_file: absolute path to output file
    :param res_x: output resolution - x-axis
    :param res_y: output resolution = y-axis
    :param bounds: warp domain boundaries
    :param crs: output coordinate reference system
    :param method: resampling method
    :param dtype: output data type
    :param src_nodata: source nodata value
    :param nodata: nodata value
    :return: None
    """
    # - Define output grid - with regular step equal to the
    # - selected resolution
    dem_src = load_dem_tiff(src_file)
    # - raster upper - left and lower - right corners
    ul_corner_1 = dem_src['ul_corner']
    lr_corner_1 = dem_src['lr_corner']
    if bounds is None:
        minx = int((ul_corner_1[0] // res_x) * res_x) - res_x
        miny = int((lr_corner_1[1] // res_y) * res_y) - res_y
        maxx = int((lr_corner_1[0] // res_x) * res_x) + res_x
        maxy = int((ul_corner_1[1] // res_y) * res_y) + res_y
        output_bounds = [minx, miny, maxx, maxy]
    else:
        output_bounds = bounds

    with rasterio.open(src_file) as src:
        # - virtual Warp Parameters
        if dtype is None:
            # - if not selected, source data type
            dtype = src.dtypes[0]
        vrt_options = vrt_param(crs, res_x, res_y,
                                output_bounds, method, dtype,
                                src_nodata, nodata)
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
            rio_shutil.copy(vrt, out_file, driver='GTiff')

        return data


def clip_raster(src_file: str, ref_shp: str, out_file: str,
                nodata: int = -9999) -> Union[str, None]:
    """
    Clip Input Raster Using Rasterio. Find more info here:
    https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
    :param src_file: absolute path to input raster file
    :param ref_shp: absolute path to reference shapefile
    :param out_file: absolute path to output raster file
    :param nodata: output raster nodata
    :return: None
    """
    # - Open Reference shapefile
    with fiona.open(ref_shp, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]

    # - Open Input Raster
    with rasterio.open(src_file) as src:
        out_raster, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    # - Define Output raster metadata
    out_meta.update({'driver': 'GTiff',
                     'height': out_raster.shape[1],
                     'width': out_raster.shape[2],
                     'nodata': nodata,
                     'dtype': src.dtypes[0],
                     'compress': 'lzw',
                     'transform': out_transform})
    out_raster[out_raster == src.nodata] = np.nan

    if out_raster[np.isfinite(out_raster)].shape[0] == 0:
        return None
    # - Save clipped raster
    # - [only if valid data are found within the clipped area]
    out_raster[np.isnan(out_raster)] = nodata
    with rasterio.open(out_file, 'w', **out_meta) as dest:
        dest.write(out_raster)

    return out_file


def sample_in_memory_dataset(raster: np.ndarray, res: Any, x: np.ndarray,
                             y: np.ndarray, crs: int,
                             sample_ps_iter: list[tuple], nodata=np.nan):
    """
    Sample In-Memory Dataset
    :param raster: input raster - np.ndarray
    :param res: raster resolution - [xres, yres]
    :param x: x-axis coordinates - np.ndarray
    :param y: y-axis coordinates - np.ndarray
    :param crs: epsg code crs
    :param sample_ps_iter: list of tuples containing x, y coordinates
    :param nodata: input raster nodata
    :return: list of sampled values
    """
    if y[1] > y[0]:
        y = np.flipud(y)
        raster = np.flipud(raster)
        y += res[1]
    transform = (Affine.translation(x[0], y[0])
                 * Affine.scale(res[0], -res[1]))
    out_meta = {'driver': 'GTiff',
                'height': raster.shape[0],
                'width': raster.shape[1],
                'nodata': nodata,
                'dtype': str(raster.dtype),
                'compress': 'lzw',
                'count': 1,
                'crs': crs,
                'transform': transform}
    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as dataset:
            dataset.write(raster, 1)
        with memfile.open() as src:
            sampled_pts = [x for x in src.sample(sample_ps_iter)]
            sampled_pts = [x[0] for x in sampled_pts]

    return sampled_pts

