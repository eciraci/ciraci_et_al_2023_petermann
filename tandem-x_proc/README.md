## Process TanDEM-X data

#### Set of script used to index and process TanDEM-X raw data

1. **index_tandemx_data.py** - create a shapefile index of the TanDEM-X data.
2. **reproject_dem_rasterio.py** - reproject TanDEM-X DEMs to new coordinate reference system and resolution. Use rasterio as gdal Python's binding.
3. **reproject_dem_gdal.py** - reproject TanDEM-X DEMs to new coordinate reference system and resolution. Use standard as gdal Python's binding.
4. **mosaicing_dem_rasterio.py** - create daily mosaics of TanDEM-X DEMs. Use rasterio as gdal Python's binding.
5. **mosaicing_dem_gdal.py** - create daily mosaics of TanDEM-X DEMs. Use standard as gdal Python's binding.