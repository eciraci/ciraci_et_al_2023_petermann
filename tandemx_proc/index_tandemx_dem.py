#!/usr/bin/env python
u"""
index_tandemx_dem.py
Written by Enrico Ciraci' (08/2021)

Create an Index Shapefile of time-tagged Raw Digital Elevation Models
from the German Aerospace Centre (DLR) TanDEM-X mission (TerraSAR-X add-on for
Digital Elevation Measurement).

The index file will contain an entry for each time-tagged DEM file in the
dateset. Each index entry will be composed by a polygon showing the area of
valid elevation data covered by the DEM, and an attribute tables containing
the following attiributes:

- DEM Name,
- Raster Width in pixels,
- Raster Height in pixel;
- DEM Time Tag: Year,Month,Day - isoformat;
- Elevation Units.

Note: This script was developed to process TanDEM-X elevation data  available
    between  2011 and 2021 for the area surrounding  Petermann Glacier
    (Northwest Greenland).

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    rasterio: access to geospatial raster data
           https://rasterio.readthedocs.io
    fiona: Fiona reads and writes geographic data files
           https://fiona.readthedocs.io
    tqdm: A Fast, Extensible Progress Bar for Python and CLI
           https://tqdm.github.io
    shapely: Geometric objects, predicates, and operations
           https://pypi.org/project/Shapely/

UPDATE HISTORY:
04/13/2022: Clip Input raster by employing the estimated valid data mask.
            Use Lempel–Ziv–Welch (lws) lossless data compression to write
            output GeoTiffs.
"""
# - Python Dependencies
from __future__ import print_function
import os
import datetime
import numpy as np
from tqdm import tqdm
import fiona
from shapely import geometry
import rasterio
from rasterio import features
from rasterio.mask import mask
# - Project Utility Functions
from utility_functions import create_dir


def main() -> None:
    """Create TanDEM-X DEM index shapefile """
    # - Input data directory - Directory containing the TanDEM-X DEM files
    input_data_path \
        = os.path.join(os.getenv('PYTHONDATA'), 'TanDEM-X', 'RAW_DEMs')
    # - Create Output directory
    output_data_path \
        = create_dir(os.path.join(os.getenv('PYTHONDATA'), 'TanDEM-X'),
                     'Processed_DEMs')
    # - List input data directory content
    input_data_dir = [os.path.join(input_data_path, x)
                      for x in os.listdir(input_data_path) if not
                      x.startswith('.') and x.endswith('.tiff')]

    print('# - Create Shapefile Index of the time-tagged Digital Elevation '
          'Models from the DLR/TanDEM-X mission.')
    # -
    print(f'# - Number of DEMs Found: {len(input_data_dir)}')

    # - Processing Parameters
    crs = "EPSG:4326"       # - default coordinate reference system
    # - Define output index shapefile schema
    schema = {
        'geometry': 'Polygon',
        'properties': [('Name', 'str'), ('time', 'str'),
                       ('Width', 'int'), ('Height', 'int'),
                       ('Units', 'str'), ('npts', 'str')]
    }
    out_f_name = os.path.join(output_data_path, 'roi_tandemx_dem_index.shp')
    with fiona.open(out_f_name, mode='w', driver='ESRI Shapefile',
                    schema=schema, crs=crs) as poly_shp:
        for dem in tqdm(sorted(input_data_dir), ncols=70,
                        desc="# - Indexing TanDEM-X DEM: "):
            # - Extract time-tag from each of the considered DEM
            f_name = dem.split('/')[-1].split('_')
            data_str = f_name[-3]
            year_d = int(data_str[:4])
            month_d = int(data_str[4:6])
            day_d = int(data_str[6:8])
            hour_d = int(data_str[9:11])
            minute_d = int(data_str[11:13])
            second_d = int(data_str[13:15])
            # - convert dat string to datetime
            dem_date = datetime.datetime(year=year_d, month=month_d, day=day_d,
                                         hour=hour_d, minute=minute_d,
                                         second=second_d)
            # - DEM name
            d_name = f_name[-3]+'_'+f_name[-2]+'_'+f_name[-1][:-5]
            # - Import DEM data
            with rasterio.open(dem, mode="r+") as src:
                # - read band #1
                m_arr = src.read(1, masked=True)
                # - generate valid data binary mask
                # - valid data - msk = 255
                # - not valid data - msk = 0
                raster_shape = list(m_arr.shape)
                b_raster_shape = [dim+20 for dim in raster_shape]
                msk = np.full(m_arr.shape, 255).astype('float32')
                msk[m_arr == -9999] = 0
                # - Add a 10-pixels buffer around the binary mask.
                buffered_mask = np.zeros(b_raster_shape).astype('float32')
                buffered_mask[10:-10, 10:-10] = msk
                # - Use rasterio.features.shapes to get valid data region
                # - boundaries. For more details:
                # - https://rasterio.readthedocs.io/en/latest/api/
                # -         rasterio.features.html
                b_shapes = list(features.shapes(buffered_mask,
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
                    'properties': {'Name': d_name, 'time': dem_date.isoformat(),
                                   'Width': src.width, 'Height': src.height,
                                   'Units': src.units[0],
                                   'npts': len(shp_bound[0]['coordinates'][0])},
                }
                poly_shp.write(row_dict)

                # - Clip Input raster by employing the valid data mask
                poly_mask = [geometry.Polygon(shp_bound[0]['coordinates'][0])]
                out_raster, out_transform \
                    = mask(src, poly_mask, crop=True, nodata=-9999)

                # - Define Output raster metadata
                out_meta = src.meta
                out_meta.update({"driver": "GTiff",
                                 "height": out_raster.shape[1],
                                 "width": out_raster.shape[2],
                                 "nodata": -9999,
                                 "dtype": src.dtypes[0],
                                 'compress': 'lzw',
                                 "transform": out_transform})
                # - Save clipped raster
                if out_raster[np.isfinite(out_raster)].shape[0] != 0:
                    with rasterio.open(dem, "w", **out_meta) as dest:
                        dest.write(out_raster)


# -- run main program
if __name__ == '__main__':
    main()
