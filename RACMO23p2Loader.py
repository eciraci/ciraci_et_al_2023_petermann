"""
Enrico Ciraci 12/2021
RACMO23p2Loader - Utility class to import Surface Mass Balance Data
estimates from the Regional Climate Model RACMO version 2.3p2.
"""
# - python dependencies
from __future__ import print_function
import os
import numpy as np
import xarray as xr
from datetime import datetime
from utility_functions import do_kdtree


class RACMO23p2Loader:
    """Load RACMO23p2 SMB data."""
    def __init__(self, d_path, verbose=False):
        # - class attributes
        self.path = d_path
        self.smb_array = np.array([])
        self.smb_array_rsp = np.array([])
        self.lat_grid = np.array([])
        self.lon_grid = np.array([])
        self.y_coords = np.array([])
        self.x_coords = np.array([])
        self.y_coords_mm = np.array([])
        self.x_coords_mm = np.array([])
        self.time_ax = []
        self.ds = None
        # - list input directory content
        smb_data_list = sorted([os.path.join(d_path, x)
                                for x in os.listdir(d_path)
                                if x.endswith('nc')
                                and not x.startswith('.')])

        # - append the available yearly estimates
        if verbose:
            print('# - Loading SMB data from RACMO2.3p2:')
        for cnt, y_dset in enumerate(smb_data_list[:]):
            # - extract year value from input file name
            year_f = int(y_dset.split('/')[-1][8:12])
            if verbose:
                print(f'# - {year_f}')

            # - load yearly SMB files
            smb_input = xr.open_dataset(y_dset, decode_times=False)
            if cnt == 0:
                self.smb_array = smb_input['smb'].values
                self.x_coords = smb_input['x'].values
                self.y_coords = smb_input['y'].values
                self.lat_grid = smb_input['latitude'].values
                self.lon_grid = smb_input['longitude'].values

                # - create x/y coords mesh grid
                self.x_coords_mm, self.y_coords_mm \
                    = np.meshgrid(self.x_coords, self.y_coords)
            else:
                self.smb_array = np.append(self.smb_array,
                                           smb_input['smb'].values,
                                           axis=0)

            # - create time axis in datetime format
            time_ax_temp = [datetime(year=year_f, month=m, day=1)
                            for m in range(1, smb_input['smb']
                                           .values.shape[0] + 1)]
            # - extend time axis
            self.time_ax.extend(time_ax_temp)

        # - Reshape/Vectorize Monthly SMB estimates
        self.smb_array_rsp \
            = self.smb_array.reshape([len(self.time_ax),
                                      len(self.x_coords_mm.ravel())])
        # - Create Xarray Dataset Containing Xarray data.
        self.ds = xr.Dataset(
            data_vars=dict(smb_array=(["time", 'y', 'x'], self.smb_array)
                           ),
            coords=dict(time=(["time"], self.time_ax),
                        x=(["x"], self.x_coords),
                        y=(["y"], self.y_coords),
                        )
        )

    def __str__(self):
        return '# - RACMO23p2Loader() - Data Coverage: {} - {}'\
            .format(self.ds['time'].values[0], self.ds['time'].values[-1])

    def sample_smb_pt_coords(self, pt_x: float, pt_y: float,
                             year: int, month: int) -> dict:
        """
        Extract SMB monthly values at the selected location[ pt_x, pt_y]
        and  time - month/year
        NOTE: point coordinates must be provided in North Polar
              Stereographic Projection
        :param pt_x: point x coordinate
        :param pt_y: point y coordinate
        :param year: year
        :param month: month
        :return: python dictionary containing SMB monthly values
        """
        # - extract SMB data at the selected time
        smb_point_s = self.ds.where(
            ((self.ds['time.year'] == year)
             & (self.ds['time.month'] == month)),
            drop=True)['smb_array'].values
        smb_sample = np.squeeze(smb_point_s)

        # - Flatten Domain Coordinates
        combined_x_y_arrays = np.dstack([self.x_coords_mm.ravel(),
                                         self.y_coords_mm.ravel()])[0]
        s_points = [pt_x, pt_y]
        # - Use kd-tree to find the index of the closest location
        index = do_kdtree(combined_x_y_arrays, s_points)

        return {'smn_sample': smb_sample.ravel()[index], 'index': index}

    def smb_pt_t_series(self, pt_x: float, pt_y: float,
                        rm_mean: bool = False, ref_year: int = 1991) \
            -> xr.Dataset:
        """
        Extract SMB time series at the selected location [pt_x, pt_y]
        NOTE: point coordinates must be provided in North Polar
              Stereographic Projection
        :param pt_x: point x coordinate
        :param pt_y: point y coordinate
        :param rm_mean: if True, remove SMB mean value for the selected
            reference time period
        :param ref_year: last year of the reference period.
        :return: xarray dataset containing the SMB time series.
        """
        s_points = [pt_x, pt_y]
        # - Flatten Domain Coordinates
        combined_x_y_arrays = np.dstack([self.x_coords_mm.ravel(),
                                         self.y_coords_mm.ravel()])[0]
        # - Use kd-tree to find the index of the closest location
        index = do_kdtree(combined_x_y_arrays, s_points)

        # - Calculate smb time series at the selected location
        smb_pt = self.smb_array_rsp[:, index]

        # - Save the Obtained time series inside a Xarray Dataset
        ds_ts = xr.Dataset(
            data_vars=dict(smb_pt=(["time"], smb_pt),
                           ),
            coords=dict(time=(["time"], self.time_ax))
        )

        if rm_mean:
            # - If selected use the period 1958-1991 as reference
            ref_mean = ds_ts.where((ds_ts['time.year'] <= ref_year),
                                   drop=True).mean()
            smb_pt_anom = smb_pt - ref_mean['smb_pt'].values

            # - Calculate Cumulative SMB - Check Unit Conversion
            cum_smb_pt = np.cumsum(smb_pt_anom)
        else:
            # - Calculate Cumulative SMB - Check Unit Conversion
            # - WO removing reference mean.
            cum_smb_pt = np.cumsum(smb_pt)

        # - Save the Obtained time series inside a Xarray Dataset
        # - Include also the cumulative SMB time series.
        ds_ts = xr.Dataset(
            data_vars=dict(smb_pt=(["time"], smb_pt),
                           cum_smb_pt=(["time"], cum_smb_pt),
                           ),
            coords=dict(time=(["time"], self.time_ax))
        )
        return ds_ts

    def smb_t_series(self, rm_mean: bool = False,
                     ref_year: int = 1991) -> xr.Dataset:
        """
        Extract SMB time series over the entire SMB dataset domain
        :param rm_mean: if True, remove SMB mean value for the selected
            reference time period
        :param ref_year: last year of the reference period
        :return: xarray dataset containing the SMB time series [gridded format].
        """
        ds_ts = xr.Dataset(
            data_vars=dict(smb_array=(["time", 'y', 'x'], self.smb_array),
                           ),
            coords=dict(time=(["time"], self.time_ax),
                        x=(["x"], self.x_coords),
                        y=(["y"], self.y_coords),
                        )
        )

        if rm_mean:
            # - If selected use the period 1958-1991 as reference
            ref_mean = ds_ts.where((ds_ts['time.year'] <= ref_year),
                                   drop=True).mean()
            smb_array_anom = self.smb_array - ref_mean['smb_array'].values
            # - Calculate Cumulative SMB - Check Unit Conversion
            cum_smb_grid = np.cumsum(smb_array_anom, axis=0)
        else:
            # - Calculate Cumulative SMB - Check Unit Conversion
            # - WO removing reference mean.
            cum_smb_grid = np.cumsum(self.smb_array, axis=0)

        ds_cum_ts = xr.Dataset(
            data_vars=dict(cum_smb_grid=(["time", 'y', 'x'], cum_smb_grid),
                           ),
            coords=dict(time=(["time"], self.time_ax),
                        x=(["x"], self.x_coords), y=(["y"], self.y_coords),
                        )
        )

        return ds_cum_ts

    def sel_slice(self, date_1: str, date_2: str, drop: bool = True)\
            -> xr.Dataset:
        """
        Select SMB data for a selected time slice
        :param date_1: time 1 - time stamp format
        :param date_2: time 2 - time stamp format
        :param drop: if True drop values outside the considered interval.
        :return: Xarray Dataset - containing data for the selected time slice.
        """
        return self.ds.sel(time=slice(date_1, date_2), drop=drop)
