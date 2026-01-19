# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Class for performing preprocessing on the loaded data."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd

from pyelq.meteorology.meteorology import Meteorology, MeteorologyGroup
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.support_functions.spatio_temporal_interpolation import temporal_resampling


@dataclass
class Preprocessor:
    """Class which implements generic functionality for pre-processing of sensor and meteorology information.

    Attributes:
        time_bin_edges (Union[pd.arrays.DatetimeArray, None]): edges of the time bins to be used for
        smoothing/interpolation. If None, no smoothing/interpolation is performed.
        sensor_object (SensorGroup): sensor group object containing raw data.
        met_object (Meteorology): met object containing raw data.
        aggregate_function (str): function to be used for aggregation of data. Defaults to mean.
        sensor_fields (list): standard list of sensor attributes that we wish to regularize and/or filter.
        met_fields (list): standard list of meteorology attributes that we wish to regularize/filter.
        is_regularized (bool): flag indicating whether the met and sensor data has been regularized.
    """

    time_bin_edges: Union[pd.arrays.DatetimeArray, None]
    sensor_object: SensorGroup
    met_object: Union[Meteorology, MeteorologyGroup]
    aggregate_function: str = "mean"
    sensor_fields = ["time", "concentration", "source_on"]
    met_fields = [
        "time",
        "wind_direction",
        "wind_speed",
        "pressure",
        "temperature",
        "u_component",
        "v_component",
        "w_component",
        "wind_turbulence_horizontal",
        "wind_turbulence_vertical",
    ]
    is_regularized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Initialise the class.

        Attaching the sensor and meteorology objects as attributes, and running initial regularization and NaN filtering
        steps. If time_bin_edges is provided, the data will be smoothed/interpolated onto the specified time grid and
        NaN values will be filtered out. If time_bin_edges is None, the data will be filtered for NaN values only.

        Before running the regularization (for the case where time_bin_edges is provided) & NaN filtering, the function
        ensures that u_component and v_component are present as fields on met_object. The post-smoothing wind speed and
        direction are then calculated from the smoothed u and v components, to eliminate the need to take means of
        directions when binning.

        If time_bin_edges is provided, the sensor and meteorology group objects attached to the class will have
        identical numbers of data points per device, identical time stamps, and be free of NaNs. If time_bin_edges is
        None, the sensor and meteorology group objects will not have identical time stamps, but will be free of NaNs.
        The time stamps for the sensor objects and meteorology objects will be the original time stamps.

        """        
        self.met_object.calculate_uv_from_wind_speed_direction()
        if self.time_bin_edges is not None:
            self.regularize_data()

        self.met_object.calculate_wind_direction_from_uv()
        self.met_object.calculate_wind_speed_from_uv()
        if self.is_regularized:
            self.filter_nans()
        else:
            self.filter_nans_no_regularize()
            

    def regularize_data(self) -> None:
        """Smoothing or interpolation of data onto a common set of time points.

        Function which takes in sensor and meteorology objects containing raw data (on original time points), and
        smooths or interpolates these onto a common set of time points.

        When a SensorGroup object is supplied, the function will return a SensorGroup object with the same number of
        sensors. When a MeteorologyGroup object is supplied, the function will return a MeteorologyGroup object with the
        same number of objects. When a Meteorology object is supplied, the function will return a MeteorologyGroup
        object with the same number of objects as there is sensors in the SensorGroup object. The individual Meteorology
        objects will be identical.

        Assumes that sensor_object and met_object attributes contain the RAW data, on the original time stamps, as
        loaded from file/API using the relevant data access class.

        After the function has been run, the sensor and meteorology group objects attached to the class as attributes
        will have identical time stamps, but may still contain NaNs.

        """
        self.is_regularized = True
        sensor_out = deepcopy(self.sensor_object)
        for sns_new, sns_old in zip(sensor_out.values(), self.sensor_object.values()):
            for field in self.sensor_fields:
                if (field != "time") and (getattr(sns_old, field) is not None):
                    time_out, resampled_values = temporal_resampling(
                        sns_old.time, getattr(sns_old, field), self.time_bin_edges, self.aggregate_function
                    )
                    setattr(sns_new, field, resampled_values)
            sns_new.time = time_out

        met_out = MeteorologyGroup()
        if isinstance(self.met_object, Meteorology):
            single_met_object = self.interpolate_single_met_object(met_in_object=self.met_object)
            for key in sensor_out.keys():
                met_out[key] = single_met_object
        else:
            for key, temp_met_object in self.met_object.items():
                met_out[key] = self.interpolate_single_met_object(met_in_object=temp_met_object)

        self.sensor_object = sensor_out
        self.met_object = met_out
        

    def filter_nans(self) -> None:
        """Filter out data points where any of the specified sensor or meteorology fields has a NaN value.

        Assumes that sensor_object and met_object attributes have first been passed through the regularize_data
        function, and thus have fields on aligned time grids.

        Function first works through all sensor and meteorology fields and finds indices of all times where there is a
        NaN value in any field. Then, it uses the resulting index to filter all fields.

        The result of this function is that the sensor_object and met_object attributes of the class are updated, any
        NaN values having been removed.

        """
        for sns_key, met_key in zip(self.sensor_object, self.met_object):
            sns_in = self.sensor_object[sns_key]
            met_in = self.met_object[met_key]

            filter_index_sensor = self.get_nan_filter_index(sns_in, self.sensor_fields)
            filter_index_met = self.get_nan_filter_index(met_in, self.met_fields)
            filter_index = np.logical_and(filter_index_sensor, filter_index_met)

            self.sensor_object[sns_key] = self.filter_object_fields(sns_in, self.sensor_fields, filter_index)
            self.met_object[met_key] = self.filter_object_fields(met_in, self.met_fields, filter_index)

    def filter_nans_no_regularize(self) -> None:
        """Filter out data points where any of the specified sensor or meteorology fields has a NaN value.

        Function first works through all sensor and meteorology fields and finds indices of all times where there is a
        NaN value in any field. Then, it uses the resulting index to filter all fields.
        The sensor_object and met_object are not assumed to have the same time stamps and they are treated separately.

        The result of this function is that the sensor_object and met_object attributes of the class are updated, any
        NaN values having been removed.

        """
        for sns_key in self.sensor_object:
            sns_in = self.sensor_object[sns_key]
            filter_index = self.get_nan_filter_index(sns_in, self.sensor_fields)
            self.sensor_object[sns_key] = self.filter_object_fields(sns_in, self.sensor_fields, filter_index)

        if isinstance(self.met_object, Meteorology):
            filter_index = self.get_nan_filter_index(self.met_object, self.met_fields)
            self.met_object = self.filter_object_fields(self.met_object, self.met_fields, filter_index)
        else:
            raise TypeError("MeteorologyGroup not required in case with no regularization.")

    @staticmethod
    def get_nan_filter_index(obj: Union[Sensor, Meteorology], field_list: list) -> np.ndarray:
        """Get a index for a given object to be able to filter out on NaN values in listed fields.

        Args:
            obj: Sensor or Meteorology object.
            field_list (list): list of field names to be checked for NaN values.

        Returns:
            filter_index (np.ndarray): boolean array indicating which indices do not have NaN values in
                any of the specified fields.

        """
        filter_index = np.ones(obj.nof_observations, dtype=bool)

        for field_name in field_list:
            if (field_name != "time") and (getattr(obj, field_name) is not None):
                filter_index = np.logical_and(filter_index, np.logical_not(np.isnan(getattr(obj, field_name))))

        return filter_index

    def filter_on_met(self, filter_variable: list, lower_limit: list = None, upper_limit: list = None) -> None:
        """Filter the supplied data on given properties of the meteorological data.

       
        If self.is_regularized, the filtering is done on both sensor_object and met_object and assumes that the
        SensorGroup and MeteorologyGroup objects attached as attributes have corresponding values (one per sensor 
        device), and have attributes that have been pre-smoothed/interpolated onto a common time grid per device.

        If the data is not regularized, the filtering is done on the met_object only. In this case a MeteorologyGroup 
        is not allowed.

        The result of this function is that the sensor_object and met_object attributes are updated with the filtered
        versions.

        Args:
            filter_variable (list of str): list of meteorology variables that we wish to use for filtering.
            lower_limit (list of float): list of lower limits associated with the variables in filter_variables.
                Defaults to None.
            upper_limit (list of float): list of upper limits associated with the variables in filter_variables.
                Defaults to None.

        """
        if lower_limit is None:
            lower_limit = [-np.inf] * len(filter_variable)
        if upper_limit is None:
            upper_limit = [np.inf] * len(filter_variable)

        if self.is_regularized:
            for vrb, low, high in zip(filter_variable, lower_limit, upper_limit):
                for sns_key, met_key in zip(self.sensor_object, self.met_object):
                    sns_in = self.sensor_object[sns_key]
                    met_in = self.met_object[met_key]
                    index_keep = np.logical_and(getattr(met_in, vrb) >= low, getattr(met_in, vrb) <= high)
                    self.sensor_object[sns_key] = self.filter_object_fields(sns_in, self.sensor_fields, index_keep)
                    self.met_object[met_key] = self.filter_object_fields(met_in, self.met_fields, index_keep)
        else:
            if isinstance(self.met_object, MeteorologyGroup):
                raise TypeError("MeteorologyGroup not required in case with no regularization.")
            for vrb, low, high in zip(filter_variable, lower_limit, upper_limit):                
                index_keep = np.logical_and(
                    getattr(self.met_object, vrb) >= low, getattr(self.met_object, vrb) <= high
                )
                self.met_object = self.filter_object_fields(self.met_object, self.met_fields, index_keep)                                           

    def block_data(
        self, time_edges: pd.arrays.DatetimeArray, data_object: Union[SensorGroup, MeteorologyGroup]
    ) -> list:
        """Break the supplied data group objects into time-blocked chunks.

        Returning a list of sensor and meteorology group objects per time chunk.

        If there is no data for a given device in a particular period, then that device is simply dropped from the group
        object in that block.

        Either a SensorGroup or a MeteorologyGroup object can be supplied, and the list of blocked objects returned will
        be of the same type.

        Args:
            time_edges (pd.Arrays.DatetimeArray): [(n_period + 1) x 1] array of edges of the time bins to be used for
                dividing the data into blocks.
            data_object (SensorGroup or MeteorologyGroup): data object containing either or meteorological data, to be
                divided into blocks.

        Returns:
            data_list (list): list of [n_period x 1] data objects, each list element being either a SensorGroup or
                MeteorologyGroup object (depending on the input) containing the data for the corresponding period.

        """
        data_list = []
        nof_periods = len(time_edges) - 1
        if isinstance(data_object, SensorGroup):
            field_list = self.sensor_fields
        elif isinstance(data_object, MeteorologyGroup):
            field_list = self.met_fields
        else:
            raise TypeError("Data input must be either a SensorGroup or MeteorologyGroup.")

        for k in range(nof_periods):
            data_list.append(type(data_object)())
            for key, dat in data_object.items():
                idx_time = (dat.time >= time_edges[k]) & (dat.time <= time_edges[k + 1])
                if np.any(idx_time):
                    data_list[-1][key] = deepcopy(dat)
                    data_list[-1][key] = self.filter_object_fields(data_list[-1][key], field_list, idx_time)
        return data_list

    @staticmethod
    def filter_object_fields(
        data_object: Union[Sensor, Meteorology], fields: list, index: np.ndarray
    ) -> Union[Sensor, Meteorology]:
        """Apply a filter index to all the fields in a given data object.

        Can be used for either a Sensor or Meteorology object.

        Args:
            data_object (Union[Sensor, Meteorology]): sensor or meteorology object (corresponding to a single device)
                for which fields are to be filtered.
            fields (list): list of field names to be filtered.
            index (np.ndarray): filter index.

        Returns:
            Union[Sensor, Meteorology]: filtered data object.

        """
        return_object = deepcopy(data_object)
        for field_name in fields:
            if getattr(return_object, field_name) is not None:
                setattr(return_object, field_name, getattr(return_object, field_name)[index])
        return return_object

    def interpolate_single_met_object(self, met_in_object: Meteorology) -> Meteorology:
        """Interpolate a single Meteorology object onto the time grid of the class.

        Args:
            met_in_object (Meteorology): Meteorology object to be interpolated onto the time grid of the class.

        Returns:
            met_out_object (Meteorology): interpolated Meteorology object.

        """
        met_out_object = Meteorology()
        time_out = None
        for field_name in self.met_fields:
            if (field_name != "time") and (getattr(met_in_object, field_name) is not None):
                time_out, resampled_values = temporal_resampling(
                    met_in_object.time,
                    getattr(met_in_object, field_name),
                    self.time_bin_edges,
                    self.aggregate_function,
                )
                setattr(met_out_object, field_name, resampled_values)

        if time_out is not None:
            met_out_object.time = time_out

        return met_out_object
