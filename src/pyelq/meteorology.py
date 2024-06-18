# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Meteorology module.

The superclass for the meteorology classes

"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.arrays import DatetimeArray
from scipy.stats import circstd

from pyelq.coordinate_system import Coordinate
from pyelq.sensor.sensor import SensorGroup


@dataclass
class Meteorology:
    """Defines the properties and methods of the meteorology class.

    Sizes of all attributes should match.

    Attributes:
        wind_speed (np.ndarray, optional): Wind speed [m/s]
        wind_direction (np.ndarray, optional): Meteorological wind direction (from) [deg], see
            https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
        u_component (np.ndarray, optional): u component of wind [m/s] in the easterly direction
        v_component (np.ndarray, optional): v component of wind [m/s] in the northerly direction
        w_component (np.ndarray, optional): w component of wind [m/s] in the vertical direction
        wind_turbulence_horizontal (np.ndarray, optional): Parameter of the wind stability in
            horizontal direction [deg]
        wind_turbulence_vertical (np.ndarray, optional): Parameter of the wind stability in
            vertical direction [deg]
        pressure (np.ndarray, optional): Pressure [kPa]
        temperature (np.ndarray, optional): Temperature [K]
        atmospheric_boundary_layer (np.ndarray, optional): Atmospheric boundary layer [m]
        surface_albedo (np.ndarray, optional): Surface reflectance parameter [unitless]
        time (pandas.arrays.DatetimeArray, optional): Array containing time values associated with the
            meteorological observation
        location: (Coordinate, optional): Coordinate object specifying the meteorological observation locations
        label (str, optional): String label for object

    """

    wind_speed: np.ndarray = field(init=False, default=None)
    wind_direction: np.ndarray = field(init=False, default=None)
    u_component: np.ndarray = field(init=False, default=None)
    v_component: np.ndarray = field(init=False, default=None)
    w_component: np.ndarray = field(init=False, default=None)
    wind_turbulence_horizontal: np.ndarray = field(init=False, default=None)
    wind_turbulence_vertical: np.ndarray = field(init=False, default=None)
    pressure: np.ndarray = field(init=False, default=None)
    temperature: np.ndarray = field(init=False, default=None)
    atmospheric_boundary_layer: np.ndarray = field(init=False, default=None)
    surface_albedo: np.ndarray = field(init=False, default=None)
    time: DatetimeArray = field(init=False, default=None)
    location: Coordinate = field(init=False, default=None)
    label: str = field(init=False)

    @property
    def nof_observations(self) -> int:
        """Number of observations."""
        if self.location is None:
            return 0
        return self.location.nof_observations

    def calculate_wind_speed_from_uv(self) -> None:
        """Calculate wind speed.

        Calculate the wind speed from u and v components. Result gets stored in the wind_speed attribute

        """
        self.wind_speed = np.sqrt(self.u_component**2 + self.v_component**2)

    def calculate_wind_direction_from_uv(self) -> None:
        """Calculate wind direction: meteorological convention 0 is wind from the North.

        Calculate the wind direction from u and v components. Result gets stored in the wind_direction attribute
        See: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

        """
        self.wind_direction = (270 - 180 / np.pi * np.arctan2(self.v_component, self.u_component)) % 360

    def calculate_uv_from_wind_speed_direction(self) -> None:
        """Calculate u and v components from wind speed and direction.

        Results get stored in the u_component and v_component attributes.
        See: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

        """
        self.u_component = -1 * self.wind_speed * np.sin(self.wind_direction * (np.pi / 180))
        self.v_component = -1 * self.wind_speed * np.cos(self.wind_direction * (np.pi / 180))

    def calculate_wind_turbulence_horizontal(self, window: str) -> None:
        """Calculate the horizontal wind turbulence values from the wind direction attribute.

        Wind turbulence values are calculated as the circular standard deviation based on a rolling window.
        Outputted values are calculated at the center of the window and at least 3 observations are required in a
        window for the calculation. If the window contains less values the result will be np.nan.
        The result of the calculation will be stored as the wind_turbulence_horizontal attribute.

        Args:
            window (str): The size of the window in which values are aggregated specified as an offset alias:
                https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

        """
        data_series = pd.Series(data=self.wind_direction, index=self.time)
        aggregated_data = data_series.rolling(window=window, center=True, min_periods=3).apply(
            circstd, kwargs={"low": 0, "high": 360}
        )
        self.wind_turbulence_horizontal = aggregated_data.values

    def plot_polar_hist(self, nof_sectors: int = 16, nof_divisions: int = 5, template: object = None) -> go.Figure():
        """Plots a histogram of wind speed and wind direction in polar Coordinates.

        Args:
            nof_sectors (int, optional): The number of wind direction sectors into which the data is binned.
            nof_divisions (int, optional): The number of wind speed divisions into which the data is binned.
            template (object): A layout template which can be applied to the plot. Defaults to None.

        Returns:
            fig (go.Figure): A plotly go figure containing the trace of the rose plot.

        """
        sector_half_width = 0.5 * (360 / nof_sectors)
        wind_direction_bin_edges = np.linspace(-sector_half_width, 360 - sector_half_width, nof_sectors + 1)
        wind_speed_bin_edges = np.linspace(np.min(self.wind_speed), np.max(self.wind_speed), nof_divisions)

        dataframe = pd.DataFrame()
        dataframe["wind_direction"] = [x - 360 if x > (360 - sector_half_width) else x for x in self.wind_direction]
        dataframe["wind_speed"] = self.wind_speed

        dataframe["sector"] = pd.cut(dataframe["wind_direction"], wind_direction_bin_edges, include_lowest=True)
        if np.allclose(wind_speed_bin_edges[0], wind_speed_bin_edges):
            dataframe["speed"] = wind_speed_bin_edges[0]
        else:
            dataframe["speed"] = pd.cut(dataframe["wind_speed"], wind_speed_bin_edges, include_lowest=True)

        dataframe = dataframe.groupby(["sector", "speed"], observed=False).count()
        dataframe = dataframe.rename(columns={"wind_speed": "count"}).drop(columns=["wind_direction"])
        dataframe["%"] = dataframe["count"] / dataframe["count"].sum()

        dataframe = dataframe.reset_index()
        dataframe["theta"] = dataframe.apply(lambda x: x["sector"].mid, axis=1)

        fig = px.bar_polar(
            dataframe,
            r="%",
            theta="theta",
            color="speed",
            direction="clockwise",
            start_angle=90,
            color_discrete_sequence=px.colors.sequential.Sunset_r,
        )

        ticktext = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        polar_dict = {
            "radialaxis": {"tickangle": 90},
            "radialaxis_angle": 90,
            "angularaxis": {
                "tickmode": "array",
                "ticktext": ticktext,
                "tickvals": list(np.linspace(0, 360 - (360 / 8), 8)),
            },
        }
        fig.add_annotation(
            x=1,
            y=1,
            yref="paper",
            xref="paper",
            xanchor="right",
            yanchor="top",
            align="left",
            font={"size": 18, "color": "#000000"},
            showarrow=False,
            borderwidth=2,
            borderpad=10,
            bgcolor="#ffffff",
            bordercolor="#000000",
            opacity=0.8,
            text="<b>Radial Axis:</b> Proportion<br>of wind measurements<br>in a given direction.",
        )

        fig.update_layout(polar=polar_dict)
        fig.update_layout(template=template)
        fig.update_layout(title="Distribution of Wind Speeds and Directions")

        return fig

    def plot_polar_scatter(self, fig: go.Figure, sensor_object: SensorGroup, template: object = None) -> go.Figure():
        """Plots a scatter plot of concentration with respect to wind direction in polar Coordinates.

        This function implements the polar scatter functionality for a (single) Meteorology object. Assuming the all
        Sensors in the SensorGroup are consistent with the Meteorology object.

        Note we do plot the sensors which do not contain any values when present in the SensorGroup to keep consistency
        in plot colors.

        Args:
            fig (go.Figure): A plotly figure onto which traces can be drawn.
            sensor_object (SensorGroup): SensorGroup object which contains the concentration information
            template (object): A layout template which can be applied to the plot. Defaults to None.

        Returns:
            fig (go.Figure): A plotly go figure containing the trace of the rose plot.

        """
        max_concentration = 0

        for i, (sensor_key, sensor) in enumerate(sensor_object.items()):
            if sensor.concentration.shape != self.wind_direction.shape:
                warnings.warn(
                    f"Concentration values for sensor {sensor_key} are of shape "
                    + f"{sensor.concentration.shape}, but self.wind_direction has shape "
                    + f"{self.wind_direction.shape}. It will not be plotted on the polar scatter plot."
                )
            else:
                theta = self.wind_direction
                color_idx = i % len(sensor_object.color_map)

                fig.add_trace(
                    go.Scatterpolar(
                        r=sensor.concentration,
                        theta=theta,
                        mode="markers",
                        name=sensor_key,
                        marker={"color": sensor_object.color_map[color_idx]},
                    )
                )
                if sensor.concentration.size > 0:
                    max_concentration = np.maximum(np.nanmax(sensor.concentration), max_concentration)

        fig = set_plot_polar_scatter_layout(max_concentration=max_concentration, fig=fig, template=template)

        return fig


@dataclass
class MeteorologyGroup(dict):
    """A dictionary containing multiple Meteorology objects.

    This class is used when we want to define/store a collection of meteorology objects consistent with an associated
    SensorGroup which can then be used in further processing, e.g. Gaussian plume coupling computation.

    """

    @property
    def nof_objects(self) -> int:
        """Int: Number of meteorology objects contained in the MeteorologyGroup."""
        return len(self)

    def add_object(self, met_object: Meteorology):
        """Add an object to the MeteorologyGroup."""
        self[met_object.label] = met_object

    def calculate_uv_from_wind_speed_direction(self):
        """Calculate the u and v components for each member of the group."""
        for met in self.values():
            met.calculate_uv_from_wind_speed_direction()

    def calculate_wind_direction_from_uv(self):
        """Calculate wind direction from the u and v components for each member of the group."""
        for met in self.values():
            met.calculate_wind_direction_from_uv()

    def calculate_wind_speed_from_uv(self):
        """Calculate wind speed from the u and v components for each member of the group."""
        for met in self.values():
            met.calculate_wind_speed_from_uv()

    def plot_polar_scatter(self, fig: go.Figure, sensor_object: SensorGroup, template: object = None) -> go.Figure():
        """Plots a scatter plot of concentration with respect to wind direction in polar coordinates.

        This function implements the polar scatter functionality for a MeteorologyGroup object. It assumes each object
        in the SensorGroup has an associated Meteorology object in the MeteorologyGroup.

        Note we do plot the sensors which do not contain any values when present in the SensorGroup to keep consistency
        in plot colors.

        Args:
            fig (go.Figure): A plotly figure onto which traces can be drawn.
            sensor_object (SensorGroup): SensorGroup object which contains the concentration information
            template (object): A layout template which can be applied to the plot. Defaults to None.

        Returns:
            fig (go.Figure): A plotly go figure containing the trace of the rose plot.

        Raises
            ValueError: When there is a sensor key which is not present in the MeteorologyGroup.

        """
        max_concentration = 0

        for i, (sensor_key, sensor) in enumerate(sensor_object.items()):
            if sensor_key not in self.keys():
                raise ValueError(f"Key {sensor_key} not found in MeteorologyGroup.")
            temp_met_object = self[sensor_key]
            if sensor.concentration.shape != temp_met_object.wind_direction.shape:
                warnings.warn(
                    f"Concentration values for sensor {sensor_key} are of shape "
                    + f"{sensor.concentration.shape}, but wind_direction values for meteorology object {sensor_key} "
                    f"has shape {temp_met_object.wind_direction.shape}. It will not be plotted on the polar scatter "
                    f"plot."
                )
            else:
                theta = temp_met_object.wind_direction
                color_idx = i % len(sensor_object.color_map)

                fig.add_trace(
                    go.Scatterpolar(
                        r=sensor.concentration,
                        theta=theta,
                        mode="markers",
                        name=sensor_key,
                        marker={"color": sensor_object.color_map[color_idx]},
                    )
                )

                if sensor.concentration.size > 0:
                    max_concentration = np.maximum(np.nanmax(sensor.concentration), max_concentration)

        fig = set_plot_polar_scatter_layout(max_concentration=max_concentration, fig=fig, template=template)

        return fig


def set_plot_polar_scatter_layout(max_concentration: float, fig: go.Figure(), template: object) -> go.Figure:
    """Helper function to set the layout of the polar scatter plot.

    Helps avoid code duplication.

    Args:
        max_concentration (float): The maximum concentration value used to update radial axis range.
        fig (go.Figure): A plotly figure onto which traces can be drawn.
        template (object): A layout template which can be applied to the plot.

    Returns:
        fig (go.Figure): A plotly go figure containing the trace of the rose plot.

    """
    ticktext = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    polar_dict = {
        "radialaxis": {"tickangle": 0, "range": [0.0, 1.01 * max_concentration]},
        "radialaxis_angle": 0,
        "angularaxis": {
            "tickmode": "array",
            "ticktext": ticktext,
            "direction": "clockwise",
            "rotation": 90,
            "tickvals": list(np.linspace(0, 360 - (360 / 8), 8)),
        },
    }

    fig.add_annotation(
        x=1,
        y=1,
        yref="paper",
        xref="paper",
        xanchor="right",
        yanchor="top",
        align="left",
        font={"size": 18, "color": "#000000"},
        showarrow=False,
        borderwidth=2,
        borderpad=10,
        bgcolor="#ffffff",
        bordercolor="#000000",
        opacity=0.8,
        text="<b>Radial Axis:</b> Wind<br>speed in m/s.",
    )

    fig.update_layout(polar=polar_dict)
    fig.update_layout(template=template)
    fig.update_layout(title="Measured Concentration against Wind Direction.")
    return fig
