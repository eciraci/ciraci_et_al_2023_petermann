# ciraci_rignot_et_al_2022
Compute Ice Shelf Basal Melt Rate in a Lagrangian Framework

[![Language][]][1]

### Installation:

1. Setup minimal **conda** installation using [Miniconda][]

2. Create Python Virtual Environment

    > -   Creating an environment with commands ([Link][]);
    > -   Creating an environment from an environment.yml file
    >     ([Link][2])  -> **Recommended**;


---
### Lagrangian Workflow
```mermaid
    graph LR;
        A[(fa:fa-area-chart  TanDEM-X - Mosaics)] --> 
        A1((+)) --> B
        B[(fa:fa-area-chart  Calibrated DEMs)] --> B2
        B2((+)) --> L
        C[fa:fa-cloud I.B.E. ERA5] --> A1
        D[fa:fa-globe EIGEN-6C4 Geoid] --> A1
        E[fa:fa-earth MDT-CNES-CLS18] --> A1
        F[Ocean Tide - AOTIM-5] --> A1
        G[InSAR Ice Velocity] --> B2
        L[Lagrangian dh/dt] --> N
        M[SMB - RACMO2.3p2] --> N
        N((fa:fa-spinner Melt Rate))

        style A fill:#007b25,stroke:#333,stroke-width:4px
        style B fill:#00758f,stroke:#333,stroke-width:4px
        style N fill:#ec7c37,stroke:#333,stroke-width:4px

```


#### PYTHON DEPENDENCIES:
- [rasterio: access to geospatial raster data][]
- [gdal: Python's GDAL binding.][]
- [fiona: Fiona is GDAL’s neat and nimble vector API for Python programmers.][]
- [numpy: The fundamental package for scientific computing with Python.][]
- [xarray: Labelled multi-dimensional arrays in Python.][]
- [matplotlib: Library for creating static, animated, and interactive visualizations in Python.][]
- [tqdm: A Fast, Extensible Progress Bar for Python and CLI.][]
- [necdft4: Provides an object-oriented python interface to the netCDF version 4 library.][]
- [PyTMD: Python package for the analysis of tidal data.][]

[Language]: https://img.shields.io/badge/python%20-3.7%2B-brightgreen
[License]: https://img.shields.io/badge/license-MIT-green.svg
[1]: ..%20image::%20https://www.python.org/
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html
[Link]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
[2]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

[xarray: Labelled multi-dimensional arrays in Python.]:https://docs.xarray.dev
[rasterio: access to geospatial raster data]:https://rasterio.readthedocs.io/en/latest/
[gdal: Python's GDAL binding.]: https://gdal.org/index.html
[matplotlib: Library for creating static, animated, and interactive visualizations in Python.]:https://matplotlib.org
[tqdm: A Fast, Extensible Progress Bar for Python and CLI.]: https://github.com/tqdm/tqdm
[necdft4: Provides an object-oriented python interface to the netCDF version 4 library.]:https://pypi.org/project/netCDF4/
[fiona: Fiona is GDAL’s neat and nimble vector API for Python programmers.]:https://fiona.readthedocs.io/en/latest/
[numpy: The fundamental package for scientific computing with Python.]:https://numpy.org
[PyTMD: Python package for the analysis of tidal data.]: https://github.com/tsutterley/pyTMD
