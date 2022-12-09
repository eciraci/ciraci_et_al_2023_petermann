"""
Enrico Ciraci 12/2021
Era5Loader - Utility class to import ERA5 Mean Sea Level Pressure data.

PYTHON DEPENDENCIES:
    numpy: package for scientific computing with Python
           https://numpy.org
    scipy: library for mathematics, science, and engineering
          https://scipy.org
    pandas: Python Data Analysis Library
           https://pandas.pydata.org/
    xarray: N-D labeled arrays and datasets in Python
          https://xarray.pydata.org/en/stable/

"""
# - python dependencies
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import xarray as xr
from utility_functions import do_kdtree


class Era5Loader:
    """Load ERA5 hourly data."""
    def __init__(self, d_path):
        # - class attributes
        self.path = d_path
        self.msl_input = None
        self.lat_ax = None
        self.lon_ax = None
        self.time = None
        self.time_ax = None
        self.ds = None
        files_to_load = [
            'mean_sea_level_pressure_utc_petermann_1979_1988',
            'mean_sea_level_pressure_utc_petermann_1989_1998',
            'mean_sea_level_pressure_utc_petermann_1999_2009',
            'mean_sea_level_pressure_utc_petermann_2010_2021'
        ]
        msl_array = np.array([])
        time_ax = np.array([])
        lat_ax = np.array([])
        lon_ax = np.array([])

        for cnt, f_input in enumerate(sorted(files_to_load)):
            # - load reanalysis data
            era5_path = os.path.join(self.path, 'Reanalysis', 'ERA5',
                                     'reanalysis-era5-single-levels',
                                     f_input + '.nc')
            # - load hourly mean sea level pressure data
            d_input = xr.open_dataset(era5_path)
            msl_input_temp = d_input['msl'].values

            if msl_input_temp.shape[1] == 2:
                msl_input_temp = np.nanmean(msl_input_temp, axis=1)
                lat_ax = d_input['latitude'].values
                lon_ax = d_input['longitude'].values
            if cnt == 0:
                msl_array = msl_input_temp
                time_ax = d_input['time'].values
            else:
                msl_array = np.append(msl_array, msl_input_temp, axis=0)
                time_ax = np.append(time_ax, d_input['time'].values)

        # - mean sea level pressure in Pascal
        self.msl_input = msl_array
        self.lat_ax = lat_ax
        self.lon_ax = lon_ax
        self.time = time_ax
        self.time_ax = pd.to_datetime(list(self.time))

        self.ds = xr.Dataset(
            data_vars=dict(msl=(["time", "lat", "lon"], self.msl_input)),
            coords=dict(time=(["time"], self.time_ax),
                        lat=(["lat"], self.lat_ax),
                        lon=(["lon"], self.lon_ax))
        )

    def sample_msl_pt_coords(self, pt_lat: float, pt_lon: float,
                             year: int, month: int, day: int,
                             hour: int, verbose: bool = False) -> dict:
        # - - extract MSL at the selected UTC time
        msl_point_s = self.ds.where(
            ((self.ds['time.year'] == year)
             & (self.ds['time.month'] == month)
             & (self.ds['time.day'] == day)
             & (self.ds['time.hour'] == hour)),
            drop=True)['msl'].values
        msl_sample = np.squeeze(msl_point_s)

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

        return {'msl_sample': msl_sample.ravel()[index],
                'index': index}

    def compute_ibe_correction(self, pt_lat: float, pt_lon: float,
                               year: int, month: int, day: int, hour: int,
                               median: bool = False,
                               m_r_year: int = 1992,
                               verbose: bool = False) -> float:
        # - Calculate Inverse Barometer Effect  [m]
        rho_sea = 1028     # - seawater density  kg/m3
        gravity_acc = 9.8  # - gravity acceleration in m/sec^2 [N/kg]
        std_atm = 101325   # - standard atmosphere in Pascal

        # - Extract MSL at the selected location for the considered date
        msl_pt = self.sample_msl_pt_coords(pt_lat, pt_lon,
                                           year, month, day, hour)
        msl_sample = msl_pt['msl_sample']
        index = msl_pt['index']
        if verbose:
            print('# - Sea Level Pressure:')
            print(msl_sample)

        if median:
            # - Use Long-term median of Meas Sea Level Pressure at the selected
            # - location as reference.
            # - Calculate MSL  hourly time series.
            msl_input_ts = self.ds['msl'].values
            msl_ts = np.array([msl_input_ts[t, :, :].ravel()[index]
                               for t in range(len(self.ds['time.year']))])
            ind_median = np.where(self.ds['time.year'].values >= m_r_year)[0]
            mls_lt_median = np.median(msl_ts[ind_median])
            ibe = (msl_sample - mls_lt_median) / (rho_sea * gravity_acc)
            if verbose:
                print('# - Using Long-Term MSLP to evaluate pressure anomaly.')
                print('# - mls_lt_median = {}'.format(mls_lt_median))
                print('# - DeltaP: {}'.format(msl_sample - mls_lt_median))
        else:
            # - Using Standard Atmosphere to evaluate pressure anomaly.
            ibe = (msl_sample - std_atm) / (rho_sea * gravity_acc)

        return -ibe

    def compute_ibe_t_series(self, pt_lat: float, pt_lon: float,
                             median: bool = False,
                             m_r_year: int = 1992,
                             verbose: bool = False) -> xr.Dataset:
        # - Calculate Inverse Barometer Effect  [m]
        rho_sea = 1028     # - seawater density  kg/m3
        gravity_acc = 9.8  # - gravity acceleration in m/sec^2 [N/kg]
        std_atm = 101325   # - standard atmosphere in Pascal

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
        # - Calculate Mean Sea Level Pressure at the selected location
        msl_input_ts = self.ds['msl'].values
        msl_ts = np.array([msl_input_ts[t, :, :].ravel()[index]
                           for t in range(len(self.ds['time.year']))])

        if median:
            # - Use Long-term median of Meas Sea Level Pressure at the selected
            # - location as reference.
            ind_median = np.where(self.ds['time.year'].values >= m_r_year)[0]
            mls_lt_median = np.median(msl_ts[ind_median])
            ibe_ts = (msl_ts - mls_lt_median) / (rho_sea * gravity_acc)
        else:
            # - Using Standard Atmosphere to evaluate pressure anomaly.
            ibe_ts = (msl_ts - std_atm) / (rho_sea * gravity_acc)

        ds_ts = xr.Dataset(
            data_vars=dict(ibe=(["time"], ibe_ts)),
            coords=dict(time=(["time"], self.time_ax))
        )
        return -ds_ts

    def compute_ibe_climatology(self, pt_lat: float, pt_lon: float,
                                median: bool = False,
                                m_r_year: int = 1992,
                                verbose: bool = False) -> xr.Dataset:
        # - Calculate Inverse Barometer Effect  time series
        ts_dx = self.compute_ibe_t_series(pt_lat, pt_lon, median,
                                          m_r_year, verbose)
        # - Return Climatology
        return ts_dx.groupby("time.hour").mean("time")

