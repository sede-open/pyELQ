# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Sensor module.

The superclass for the sensor classes. This module provides the higher level Sensor and SensorGroup classes. The Sensor
class is a single sensor, the SensorGroup is a dictionary of Sensors. The SensorGroup class is created to deal with the
properties over all sensors together.

"""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.arrays import DatetimeArray

from pyelq.coordinate_system import ECEF, ENU, LLA, Coordinate


@dataclass
class Sensor:
    """Defines the properties and methods of the sensor class.

    Attributes:
        label (str, optional): String label for sensor
        time (pandas.arrays.DatetimeArray, optional): Array containing time values associated with concentration
            reading
        location (Coordinate, optional): Coordinate object specifying the observation locations
        concentration (np.ndarray, optional): Array containing concentration values associated with time reading
        source_on (np.ndarray, optional): Array of size nof_observations containing boolean values indicating
            whether a source is on or off for each observation, i.e. we are assuming the sensor can/can't see a source

    """

    label: str = field(init=False)
    time: DatetimeArray = field(init=False, default=None)
    location: Coordinate = field(init=False)
    concentration: np.ndarray = field(default_factory=lambda: np.array([]))
    source_on: np.ndarray = field(init=False, default=None)

    @property
    def nof_observations(self) -> int:
        """Int: Number of observations contained in concentration array."""
        return self.concentration.size

    def plot_sensor_location(self, fig: go.Figure(), color=None) -> go.Figure:
        """Plotting the sensor location.

        Args:
            fig (go.Figure): Plotly figure object to add the trace to
            color (`optional`): When specified, the color to be used

        Returns:
            fig (go.Figure): Plotly figure object with sensor location trace added to it

        """
        lla_object = self.location.to_lla()

        marker_dict = {"size": 10, "opacity": 0.8}
        if color is not None:
            marker_dict["color"] = color

        fig.add_trace(
            go.Scattermapbox(
                mode="markers+lines",
                lat=np.array(lla_object.latitude),
                lon=np.array(lla_object.longitude),
                marker=marker_dict,
                line={"width": 3},
                name=self.label,
            )
        )
        return fig

    def plot_timeseries(self, fig: go.Figure(), color=None, mode: str = "markers") -> go.Figure:
        """Timeseries plot of the sensor concentration observations.

        Args:
            fig (go.Figure): Plotly figure object to add the trace to
            color (`optional`): When specified, the color to be used
            mode (str, optional): Mode used for plotting, i.e. markers, lines or markers+lines

        Returns:
            fig (go.Figure): Plotly figure object with sensor concentration timeseries trace added to it

        """
        marker_dict = {"size": 5, "opacity": 1}
        if color is not None:
            marker_dict["color"] = color

        fig.add_trace(
            go.Scatter(
                x=self.time,
                y=self.concentration.flatten(),
                mode=mode,
                marker=marker_dict,
                name=self.label,
                legendgroup=self.label,
            )
        )

        return fig


@dataclass
class SensorGroup(dict):
    """A dictionary containing multiple Sensors.

    This class is used when we want to combine a collection of sensors and be able to store/access overall properties.

    Attributes:
        color_map (list, optional): Default colormap to use for plotting

    """

    color_map: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.color_map = px.colors.qualitative.Pastel

    @property
    def nof_observations(self) -> int:
        """Int: The total number of observations across all the sensors."""
        return int(np.sum([sensor.nof_observations for sensor in self.values()], axis=None))

    @property
    def concentration(self) -> np.ndarray:
        """np.ndarray: Column vector of concentration values across all sensors, unwrapped per sensor."""
        return np.concatenate([sensor.concentration.flatten() for sensor in self.values()], axis=0)

    @property
    def time(self) -> pd.arrays.DatetimeArray:
        """DatetimeArray: Column vector of time values across all sensors."""
        return pd.array(np.concatenate([sensor.time for sensor in self.values()]), dtype="datetime64[ns]")

    @property
    def location(self) -> Coordinate:
        """Coordinate: Coordinate object containing observation locations from all sensors in the group."""
        location_object = deepcopy(list(self.values())[0].location)
        if isinstance(location_object, ENU):
            attr_list = ["east", "north", "up"]
        elif isinstance(location_object, LLA):
            attr_list = ["latitude", "longitude", "altitude"]
        elif isinstance(location_object, ECEF):
            attr_list = ["x", "y", "z"]
        else:
            raise TypeError(
                f"Location object should be either ENU, LLA or ECEF, while currently it is{type(location_object)}"
            )
        for attr in attr_list:
            setattr(
                location_object,
                attr,
                np.concatenate([np.array(getattr(sensor.location, attr), ndmin=1) for sensor in self.values()], axis=0),
            )
        return location_object

    @property
    def sensor_index(self) -> np.ndarray:
        """np.ndarray: Column vector of integer indices linking concentration observation to a particular sensor."""
        return np.concatenate(
            [np.ones(sensor.nof_observations, dtype=int) * i for i, sensor in enumerate(self.values())]
        )

    @property
    def source_on(self) -> np.ndarray:
        """Column vector of booleans indicating whether sources are expected to be on, unwrapped over sensors.

        Assumes source is on when None is specified for a specific sensor.

        Returns:
            np.ndarray: Source on attribute, unwrapped over sensors.

        """
        overall_idx = np.array([])
        for curr_key in list(self.keys()):
            if self[curr_key].source_on is None:
                temp_idx = np.ones(self[curr_key].nof_observations).astype(bool)
            else:
                temp_idx = self[curr_key].source_on

            overall_idx = np.concatenate([overall_idx, temp_idx])
        return overall_idx.astype(bool)

    @property
    def nof_sensors(self) -> int:
        """Int: Number of sensors contained in the SensorGroup."""
        return len(self)

    def add_sensor(self, sensor: Sensor):
        """Add a sensor to the SensorGroup."""
        self[sensor.label] = sensor

    def plot_sensor_location(self, fig: go.Figure, color_map: list = None) -> go.Figure:
        """Plotting of the locations of all sensors in the SensorGroup.

        Args:
            fig (go.Figure): Plotly figure object to add the trace to
            color_map (list, optional): When specified, the colormap to be used, plotting will cycle through
                the colors

        Returns:
            fig (go.Figure): Plotly figure object with sensor location traces added to it

        """
        if color_map is None:
            color_map = self.color_map

        for i, sensor in enumerate(self.values()):
            color_idx = i % len(color_map)
            fig = sensor.plot_sensor_location(fig, color=color_map[color_idx])

        return fig

    def plot_timeseries(self, fig: go.Figure, color_map: list = None, mode: str = "markers") -> go.Figure:
        """Plotting of the concentration timeseries of all sensors in the SensorGroup.

        Args:
            fig (go.Figure): Plotly figure object to add the trace to
            color_map (list, optional): When specified, the colormap to be used, plotting will cycle through
                the colors
            mode (str, optional): Mode used for plotting, i.e. markers, lines or markers+lines

        Returns:
            fig (go.Figure): Plotly figure object with sensor concentration time series traces added to it

        """
        if color_map is None:
            color_map = self.color_map

        for i, sensor in enumerate(self.values()):
            color_idx = i % len(color_map)
            fig = sensor.plot_timeseries(fig, color=color_map[color_idx], mode=mode)

        return fig
