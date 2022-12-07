## Process TanDEM-X data

#### Set of script used to index and process TanDEM-X raw data

1. **index_tandemx_data.py** - create a shapefile index of the TanDEM-X data.
2. **reproject_dem_rasterio.py** - reproject TanDEM-X DEMs to new coordinate reference system and resolution. Use rasterio as gdal Python's binding.
3. **reproject_dem_gdal.py** - reproject TanDEM-X DEMs to new coordinate reference system and resolution. Use standard as gdal Python's binding.
4. **mosaicing_dem_rasterio.py** - generate daily mosaics of TanDEM-X DEMs. Use rasterio as gdal Python's binding.
5. **mosaicing_dem_gdal.py** - generate daily mosaics of TanDEM-X DEMs. Use standard as gdal Python's binding.
6. **dem_tdx_iceshelf_corrections_compute.py** - Convert TanDEM-X data from elevation above the WGS84 Standard Ellipsoid  into
elevation above the mean sea level.
7. **dem_tdx_iceshelf_mosaic_compute.py** -  generate daily mosaics of calibrated TanDEM-X DEMs. 