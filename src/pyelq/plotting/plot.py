# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Plot module.

Large module containing all the plotting code used to create various plots. Contains helper functions and the Plot class
definition.

"""
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Type, Union

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from geojson import Feature, FeatureCollection
from openmcmc.mcmc import MCMC
from scipy.ndimage import label
from shapely import geometry

from pyelq.component.background import TemporalBackground
from pyelq.component.error_model import ErrorModel
from pyelq.component.offset import PerSensor
from pyelq.component.source_model import SlabAndSpike, SourceModel
from pyelq.coordinate_system import ENU, LLA
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.sensor.sensor import Sensor, SensorGroup

if TYPE_CHECKING:
    from pyelq.model import ELQModel


def lighter_rgb(rbg_string: str) -> str:
    """Takes in an RGB string and returns a lighter version of this colour.

    The colour is made lighter by increasing the magnitude of the RGB values by half of the difference between the
    original value and the number 255.

    Arguments:
        rbg_string (str): An RGB string.

    """
    rbg_string = rbg_string[4:-1]
    rbg_string = rbg_string.replace(" ", "")
    colors = rbg_string.split(",")
    colors_out = [np.nan, np.nan, np.nan]

    for i, color in enumerate(colors):
        color = int(color)
        color = min(int(round(color + ((255 - color) * 0.5))), 255)
        colors_out[i] = color

    return f"rgb({colors_out[0]}, {colors_out[1]}, {colors_out[2]})"


def plot_quantiles_from_array(
    fig: go.Figure,
    x_values: Union[np.ndarray, pd.arrays.DatetimeArray],
    y_values: np.ndarray,
    quantiles: Union[tuple, list, np.ndarray],
    color: str,
    name: str = None,
) -> go.Figure:
    """Plot quantiles over y-values against x-values.

    Assuming x-values have size N and y-values have size [N x M] where the second dimension is the dimension to
    calculate the quantiles over.

    Will plot the median of the y-values as a solid line and a filled area between the lower and upper specified
    quantile.

    Args:
        fig (go.Figure): Plotly figure to add the traces on.
        x_values (Union[np.ndarray, pd.arrays.DatetimeArray]): Numpy array containing the x-values to plot.
        y_values (np.ndarray): Numpy array containing the y-values to calculate the quantiles for.
        quantiles (Union[tuple, list, np.ndarray]): Values of upper and lower quantile to plot in range (0-100)
        color (str): RGB string specifying color for quantile fill plot.
        name (str, optional): Optional string name to show in the legend.

    Returns:
         fig (go.Figure): Plotly figure with the quantile filled traces and median trace added on it.

    """
    color_fill = f"rgba{color[3:-1]}, 0.3)"

    median_trace = go.Scatter(
        x=x_values,
        y=np.median(y_values, axis=1),
        mode="lines",
        line={"width": 3, "color": color},
        name=f"Median for {name}",
        legendgroup=name,
        showlegend=False,
    )

    lower_quantile_trace = go.Scatter(
        x=x_values,
        y=np.quantile(y_values, axis=1, q=quantiles[0] / 100),
        mode="lines",
        line={"width": 0, "color": color_fill},
        name=f"{quantiles[0]}% quantile",
        legendgroup=name,
        showlegend=False,
    )

    upper_quantile_trace = go.Scatter(
        x=x_values,
        y=np.quantile(y_values, axis=1, q=quantiles[1] / 100),
        fill="tonexty",
        fillcolor=color_fill,
        mode="lines",
        line={"width": 0, "color": color_fill},
        name=f"{quantiles[1]}% quantile",
        legendgroup=name,
        showlegend=False,
    )

    fig.add_trace(median_trace)
    fig.add_trace(lower_quantile_trace)
    fig.add_trace(upper_quantile_trace)

    return fig


def create_trace_specifics(object_to_plot: Union[Type[SlabAndSpike], SourceModel, MCMC], **kwargs) -> dict:
    """Specification of different traces of single variables.

    Provides all details for plots where we want to plot a single variable as a line plot. Based on the object_to_plot
    we select the correct plot to show.

    Args:
        object_to_plot (Union[Type[SlabAndSpike], SourceModel, MCMC]): Object which we want to plot a single
            variable from
        **kwargs (dict): Additional key word arguments, e.g. burn_in or dict_key, used in some specific plots but not
            applicable to all.

    Returns:
        dict: A dictionary with the following key/values:
            x_values (Union[np.ndarray, pd.arrays.DatetimeArray]): Array containing the x-values to plot.
            y_values (np.ndarray): Numpy array containing the y-values to use in plotting.
            dict_key (str): String key associated with this plot to be used in the figure_dict attribute of the Plot
                class.
            title_text (str): String title of the plot.
            x_label (str): String label of x-axis.
            y_label (str) : String label of y-axis.
            name (str): String name to show in the legend.
            color (str): RGB string specifying color for plot.

    Raises:
        ValueError: When no specifics are defined for the inputted object to plot.

    """
    if isinstance(object_to_plot, SourceModel):
        dict_key = kwargs.pop("dict_key", "number_of_sources_plot")
        title_text = "Number of Sources 'on' against MCMC iterations"
        x_label = "MCMC Iteration Number"
        y_label = "Number of Sources 'on'"
        emission_rates = object_to_plot.emission_rate
        if isinstance(object_to_plot, SlabAndSpike):
            total_nof_sources = emission_rates.shape[0]
            y_values = total_nof_sources - np.sum(object_to_plot.allocation, axis=0)
        elif object_to_plot.reversible_jump:
            y_values = np.count_nonzero(np.logical_not(np.isnan(emission_rates)), axis=0)
        else:
            raise TypeError("No plotting routine implemented for this SourceModel type.")
        x_values = np.array(range(y_values.size))
        color = "rgb(248, 156, 116)"
        name = "Number of Sources 'on'"

    elif isinstance(object_to_plot, MCMC):
        dict_key = kwargs.pop("dict_key", "log_posterior_plot")
        title_text = "Log posterior values against MCMC iterations"
        x_label = "MCMC Iteration Number"
        y_label = "Log Posterior<br>Value"
        y_values = object_to_plot.store["log_post"].flatten()
        x_values = np.array(range(y_values.size))
        color = "rgb(102, 197, 204)"
        name = "Log Posterior"

        if "burn_in" not in kwargs:
            warnings.warn("Burn in is not specified for the Log Posterior plot, are you sure this is correct?")

    else:
        raise ValueError("No values to plot")

    return {
        "x_values": x_values,
        "y_values": y_values,
        "dict_key": dict_key,
        "title_text": title_text,
        "x_label": x_label,
        "y_label": y_label,
        "name": name,
        "color": color,
    }


def create_plot_specifics(
    object_to_plot: Union[ErrorModel, PerSensor, MCMC], sensor_object: SensorGroup, plot_type: str = "", **kwargs
) -> dict:
    """Specification of different traces where we want to plot a trace for each sensor.

    Provides all details for plots where we want to plot a single variable for each sensor as a line or box plot.
    Based on the object_to_plot we select the correct plot to show.

    When plotting the MCMC Observations and Predicted Model Values Against Time plot we are assuming time axis is the
    same for all sensors w.r.t. the fitted values from the MCMC store attribute, so we are only using the time axis
    from the first sensor.

    Args:
        object_to_plot (Union[ErrorModel, PerSensor, MCMC]): Object which we want to plot a single variable from
        sensor_object (SensorGroup): SensorGroup object associated with the object_to_plot
        plot_type (str, optional): String specifying either a line or a box plot.
        **kwargs (dict): Additional key word arguments, e.g. burn_in or dict_key, used in some specific plots but not
            applicable to all.

    Returns:
        dict: A dictionary with the following key/values:
            x_values (Union[np.ndarray, pd.arrays.DatetimeArray]): Array containing the x-values to plot.
            y_values (np.ndarray): Numpy array containing the y-values to use in plotting.
            dict_key (str): String key associated with this plot to be used in the figure_dict attribute of the
                Plot class.
            title_text (str): String title of the plot.
            x_label (str): String label of x-axis.
            y_label (str): String label of y-axis.
            plot_type (str): Type of plot which needs to be generated.

    Raises:
        ValueError: When no specifics are defined for the inputted object to plot.

    """
    if isinstance(object_to_plot, ErrorModel):
        y_values = np.sqrt(1 / object_to_plot.precision)
        x_values = np.array(range(y_values.shape[1]))

        if plot_type == "line":
            dict_key = kwargs.pop("dict_key", "error_model_iterations")
            title_text = "Estimated Error Model Values"
            x_label = "MCMC Iteration Number"
            y_label = "Estimated Error Model<br>Standard Deviation (ppm)"

        elif plot_type == "box":
            dict_key = kwargs.pop("dict_key", "error_model_distributions")
            title_text = "Distributions of Estimated Error Model Values After Burn-In"
            x_label = "Sensor"
            y_label = "Estimated Error Model<br>Standard Deviation (ppm)"

        else:
            raise ValueError("Only line and box are allowed for the plot_type argument for ErrorModel")

        if "burn_in" not in kwargs:
            warnings.warn("Burn in is not specified for the ErrorModel plot, are you sure this is correct?")

    elif isinstance(object_to_plot, PerSensor):
        offset_sensor_name = list(sensor_object.values())[0].label
        y_values = object_to_plot.offset
        nan_row = np.tile(np.nan, (1, y_values.shape[1]))
        y_values = np.concatenate((nan_row, y_values), axis=0)
        x_values = np.array(range(y_values.shape[1]))

        if plot_type == "line":
            dict_key = kwargs.pop("dict_key", "offset_iterations")
            title_text = f"Estimated Value of Offset w.r.t. {offset_sensor_name}"
            x_label = "MCMC Iteration Number"
            y_label = "Estimated Offset<br>Value (ppm)"

        elif plot_type == "box":
            dict_key = kwargs.pop("dict_key", "offset_distributions")
            title_text = f"Distributions of Estimated Offset Values w.r.t. {offset_sensor_name} After Burn-In"
            x_label = "Sensor"
            y_label = "Estimated Offset<br>Value (ppm)"

        else:
            raise ValueError("Only line and box are allowed for the plot_type argument for PerSensor OffsetModel")

        if "burn_in" not in kwargs:
            warnings.warn("Burn in is not specified for the PerSensor OffsetModel plot, are you sure this is correct?")

    elif isinstance(object_to_plot, MCMC):
        y_values = object_to_plot.store["y"]
        x_values = list(sensor_object.values())[0].time
        dict_key = kwargs.pop("dict_key", "fitted_values")
        title_text = "Observations and Predicted Model Values Against Time"
        x_label = "Time"
        y_label = "Concentration (ppm)"
        plot_type = "line"

    else:
        raise ValueError("No values to plot")

    return {
        "x_values": x_values,
        "y_values": y_values,
        "dict_key": dict_key,
        "title_text": title_text,
        "x_label": x_label,
        "y_label": y_label,
        "plot_type": plot_type,
    }


def plot_single_scatter(
    fig: go.Figure,
    x_values: Union[np.ndarray, pd.arrays.DatetimeArray],
    y_values: np.ndarray,
    color: str,
    name: str,
    **kwargs,
) -> go.Figure:
    """Plots a single scatter trace on the supplied figure object.

    Args:
        fig (go.Figure): Plotly figure to add the trace to.
        x_values (Union[np.ndarray, pd.arrays.DatetimeArray]): X values to plot
        y_values (np.ndarray): Numpy array containing the y-values to use in plotting.
        color (str): RGB color string to use for this trace.
        name (str): String name to show in the legend.
        **kwargs (dict): Additional key word arguments, e.g. burn_in, legend_group, show_legend, used in some specific plots
            but not applicable to all.

    Returns:
        fig (go.Figure): Plotly figure with the trace added to it.

    """
    burn_in = kwargs.pop("burn_in", 0)
    legend_group = kwargs.pop("legend_group", name)
    show_legend = kwargs.pop("show_legend", True)
    if burn_in > 0:
        fig.add_trace(
            go.Scatter(
                x=x_values[: burn_in + 1],
                y=y_values[: burn_in + 1],
                name=name,
                mode="lines",
                line={"width": 3, "color": lighter_rgb(color)},
                legendgroup=legend_group,
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_values[burn_in:],
            y=y_values[burn_in:],
            name=name,
            mode="lines",
            line={"width": 3, "color": color},
            legendgroup=legend_group,
            showlegend=show_legend,
        )
    )

    return fig


def plot_single_box(fig: go.Figure, y_values: np.ndarray, color: str, name: str) -> go.Figure:
    """Plot a single box plot trace on the plot figure.

    Args:
        fig (go.Figure): Plotly figure to add the trace to.
        y_values (np.ndarray): Numpy array containing the y-values to use in plotting.
        color (str): RGB color string to use for this trace.
        name (str): String name to show in the legend.

    Returns:
        fig (go.Figure): Plotly figure with the trace added to it.

    """
    fig.add_trace(go.Box(y=y_values, name=name, legendgroup=name, marker={"color": color}))

    return fig


def plot_polygons_on_map(
    polygons: Union[np.ndarray, list], values: np.ndarray, opacity: float, map_color_scale: str, **kwargs
) -> go.Choroplethmapbox:
    """Plot a set of polygons on a map.

    Args:
        polygons (Union[np.ndarray, list]): Numpy array or list containing the polygons to plot.
        values (np.ndarray): Numpy array consistent with polygons containing the value which is
                             used in coloring the polygons on the map.
        opacity (float): Float between 0 and 1 specifying the opacity of the polygon fill color.
        map_color_scale (str): The string which defines which plotly color scale.
        **kwargs (dict): Additional key word arguments which can be passed on the go.Choroplethmapbox object (will override
            the default values as specified in this function)

    Returns:
        trace: go.Choroplethmapbox trace with the colored polygons which can be added to a go.Figure object.

    """
    polygon_id = list(range(values.shape[0]))
    feature_collection = FeatureCollection([Feature(geometry=polygons[idx], id_value=idx) for idx in polygon_id])
    text_box = [
        f"<b>Polygon ID</b>: {counter:d}<br><b>Center (lon, lat)</b>: "
        f"({polygons[counter].centroid.coords[0][0]:.4f}, {polygons[counter].centroid.coords[0][1]:.4f})<br>"
        f"<b>Value</b>: {values[counter]:f}<br>"
        for counter in polygon_id
    ]

    trace_options = {
        "geojson": feature_collection,
        "featureidkey": "id_value",
        "locations": polygon_id,
        "z": values,
        "marker": {"line": {"width": 0}, "opacity": opacity},
        "hoverinfo": "text",
        "text": text_box,
        "name": "Values",
        "colorscale": map_color_scale,
        "colorbar": {"title": "Values"},
        "showlegend": True,
    }

    for key, value in kwargs.items():
        trace_options[key] = value

    trace = go.Choroplethmapbox(**trace_options)

    return trace


def plot_regular_grid(
    coordinates: LLA,
    values: np.ndarray,
    opacity: float,
    map_color_scale: str,
    tolerance: float = 1e-7,
    unit: str = "kg/hr",
    name="Values",
) -> go.Choroplethmapbox:
    """Plots a regular grid of LLA data onto a map.

    So long as the input array is regularly spaced, the value of the spacing is found. A set of rectangles are defined
    where the centre of the rectangle is the LLA coordinate.

    Args:
        coordinates (LLA object): A LLA coordinate object containing a set of locations.
        values (np.array): A set of values that correspond to locations specified in the coordinates.
        opacity (float): The opacity of the grid cells when they are plotted.
        map_color_scale (str): The string which defines which plotly color scale should be used when plotting
            the values.
        tolerance (float, optional): Absolute value above which the difference between values is considered significant.
                                     Used to calculate the regular grid of coordinate values. Defaults to 1e-7.
        unit (str, optional): The unit to be added to the colorscale. Defaults to kg/hr.
        name (str, optional): Name for the trace to be used in the color bar as well

    Returns:
        trace (go.Choroplethmapbox): Trace with the colored polygons which can be added to a go.Figure object.

    """
    _, gridsize_lat = is_regularly_spaced(coordinates.latitude, tolerance=tolerance)
    _, gridsize_lon = is_regularly_spaced(coordinates.longitude, tolerance=tolerance)

    polygons = [
        geometry.box(
            coordinates.longitude[idx] - gridsize_lon / 2,
            coordinates.latitude[idx] - gridsize_lat / 2,
            coordinates.longitude[idx] + gridsize_lon / 2,
            coordinates.latitude[idx] + gridsize_lat / 2,
        )
        for idx in range(coordinates.nof_observations)
    ]

    trace = plot_polygons_on_map(
        polygons=polygons,
        values=values,
        opacity=opacity,
        name=name,
        colorbar={"title": name + "<br>" + unit},
        map_color_scale=map_color_scale,
    )

    return trace


def plot_hexagonal_grid(
    coordinates: LLA,
    values: np.ndarray,
    opacity: float,
    map_color_scale: str,
    num_hexagons: Union[int, None],
    show_positions: bool,
    aggregate_function: Callable = np.sum,
):
    """Plots a set of values into hexagonal bins with respect to the location of the values.

    Any data points that fall within the area of a hexagon are used to perform aggregation and bin the data.
    See: https://plotly.com/python-api-reference/generated/plotly.figure_factory.create_hexbin_mapbox.html

    Args:
        coordinates (LLA object): A LLA coordinate object containing a set of locations.
        values (np.array): A set of values that correspond to locations specified in the coordinates.
        opacity (float): The opacity of the hexagons when they are plotted.
        map_color_scale (str): Colour scale for plotting values.
        num_hexagons (Union[int, None]): The number of hexagons which define the *horizontal* axis of the plot.
        show_positions (bool): A flag to determine whether the original data should be shown alongside
            the binning hexagons.
        aggregate_function (Callable, optional): Function which to apply on the data in each hexagonal bin to aggregate
            the data and visualise the result.

    Returns:
        (go.Figure): A plotly go figure representing the data which was submitted to this function.

    """
    if num_hexagons is None:
        num_hexagons = max(1, np.ceil((np.max(coordinates.longitude) - np.min(coordinates.longitude)) / 0.25))

    coordinates = coordinates.to_lla()

    hex_plot = ff.create_hexbin_mapbox(
        lat=coordinates.latitude,
        lon=coordinates.longitude,
        color=values,
        nx_hexagon=num_hexagons,
        opacity=opacity,
        agg_func=aggregate_function,
        color_continuous_scale=map_color_scale,
        show_original_data=show_positions,
        original_data_marker={"color": "black"},
    )

    return hex_plot


@dataclass
class Plot:
    """Defines the plot class.

    Can be used to generate various figures from model components while storing general settings to get consistent
    figure appearance.

    Attributes:
        figure_dict (dict): Figure dictionary, used as storage using keys to identify the different figures.
        mapbox_token (str, optional): Optional mapbox token, used for plotting mapbox backgrounds.
        layout (dict, optional): Layout template for plotly figures, used in all figures generated using this class
            instance.

    """

    figure_dict: dict = field(default_factory=dict)
    mapbox_token: str = "empty"
    layout: dict = field(default_factory=dict)

    def __post_init__(self):
        """Using post init to set the default layout, not able to do this in attribute definition/initialization."""
        self.layout = {
            "layout": go.Layout(
                font={"family": "Futura", "size": 20},
                title={"x": 0.5},
                title_font={"size": 30},
                xaxis={"ticks": "outside", "showline": True, "linewidth": 2},
                yaxis={"ticks": "outside", "showline": True, "linewidth": 2},
                legend={
                    "orientation": "v",
                    "yanchor": "middle",
                    "y": 0.5,
                    "xanchor": "right",
                    "x": 1.2,
                    "font": {"size": 14, "color": "black"},
                },
            )
        }

    def show_all(self, renderer="browser"):
        """Show all the figures which are in the figure dictionary.

        Args:
            renderer (str, optional): Default renderer to use when showing the figures.

        """
        for fig in self.figure_dict.values():
            fig.show(renderer=renderer)

    def plot_single_trace(self, object_to_plot: Union[Type[SlabAndSpike], SourceModel, MCMC], **kwargs):
        """Plotting a trace of a single variable.

        Depending on the object to plot it creates a figure which is stored in the figure_dict attribute.
        First it grabs all the specifics needed for the plot and then plots the trace.

        Args:
            object_to_plot (Union[Type[SlabAndSpike], SourceModel, MCMC]): The object from which to plot a variable
            **kwargs (dict): Additional key word arguments, e.g. burn_in, legend_group, show_legend, dict_key, used in some
                specific plots but not applicable to all.

        """
        plot_specifics = create_trace_specifics(object_to_plot=object_to_plot, **kwargs)

        burn_in = kwargs.pop("burn_in", 0)

        fig = go.Figure()
        fig = plot_single_scatter(
            fig=fig,
            x_values=plot_specifics["x_values"],
            y_values=plot_specifics["y_values"],
            color=plot_specifics["color"],
            name=plot_specifics["name"],
            burn_in=burn_in,
        )

        if burn_in > 0:
            fig.add_vline(
                x=burn_in, line_width=3, line_dash="dash", line_color="black", annotation_text=f"\tBurn in: {burn_in}"
            )
        if isinstance(object_to_plot, SlabAndSpike) and isinstance(object_to_plot, SourceModel):
            prior_num_sources_on = round(object_to_plot.emission_rate.shape[0] * object_to_plot.slab_probability, 2)

            fig.add_hline(
                y=prior_num_sources_on,
                line_width=3,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Prior sources 'on': {prior_num_sources_on}",
            )

        if self.layout is not None:
            fig.update_layout(template=self.layout)

        fig.update_layout(title=plot_specifics["title_text"])
        fig.update_xaxes(title_standoff=20, automargin=True, title_text=plot_specifics["x_label"])
        fig.update_yaxes(title_standoff=20, automargin=True, title_text=plot_specifics["y_label"])

        self.figure_dict[plot_specifics["dict_key"]] = fig

    def plot_trace_per_sensor(
        self,
        object_to_plot: Union[ErrorModel, PerSensor, MCMC],
        sensor_object: Union[SensorGroup, Sensor],
        plot_type: str,
        **kwargs,
    ):
        """Plotting a trace of a single variable per sensor.

        Depending on the object to plot it creates a figure which is stored in the figure_dict attribute.
        First it grabs all the specifics needed for the plot and then plots the trace per sensor.

        Args:
            object_to_plot (Union[ErrorModel, PerSensor, MCMC]): The object which to plot a variable from
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the object_to_plot
            plot_type (str): String specifying a line or box plot.
            **kwargs (dict): Additional key word arguments, e.g. burn_in, legend_group, show_legend, dict_key, used in some
                specific plots but not applicable to all.

        """
        if isinstance(sensor_object, Sensor):
            temp = SensorGroup()
            temp.add_sensor(sensor_object)
            sensor_object = deepcopy(temp)
        plot_specifics = create_plot_specifics(
            object_to_plot=object_to_plot, sensor_object=sensor_object, plot_type=plot_type, **kwargs
        )
        burn_in = kwargs.pop("burn_in", 0)

        fig = go.Figure()
        for sensor_idx, sensor_key in enumerate(sensor_object.keys()):
            color_idx = sensor_idx % len(sensor_object.color_map)
            color = sensor_object.color_map[color_idx]

            if plot_specifics["plot_type"] == "line":
                fig = plot_single_scatter(
                    fig=fig,
                    x_values=plot_specifics["x_values"],
                    y_values=plot_specifics["y_values"][sensor_idx, :],
                    color=color,
                    name=sensor_key,
                    burn_in=burn_in,
                )
            elif plot_specifics["plot_type"] == "box":
                fig = plot_single_box(
                    fig=fig,
                    y_values=plot_specifics["y_values"][sensor_idx, burn_in:].flatten(),
                    color=color,
                    name=sensor_key,
                )

        if burn_in > 0 and plot_specifics["plot_type"] == "line":
            fig.add_vline(
                x=burn_in, line_width=3, line_dash="dash", line_color="black", annotation_text=f"\tBurn in: {burn_in}"
            )

        if self.layout is not None:
            fig.update_layout(template=self.layout)

        fig.update_layout(title=plot_specifics["title_text"])
        fig.update_xaxes(title_standoff=20, automargin=True, title_text=plot_specifics["x_label"])
        fig.update_yaxes(title_standoff=20, automargin=True, title_text=plot_specifics["y_label"])

        self.figure_dict[plot_specifics["dict_key"]] = fig

    def plot_fitted_values_per_sensor(
        self,
        mcmc_object: MCMC,
        sensor_object: Union[SensorGroup, Sensor],
        background_model: TemporalBackground = None,
        burn_in: int = 0,
    ):
        """Plot the fitted values from the mcmc object against time, also shows the estimated background when inputted.

        Based on the inputs it plots the results of the mcmc analysis, being the fitted values of the concentration
        measurements together with the 10th and 90th quantile lines to show the goodness of fit of the estimates.

        The created figure is stored in the figure_dict attribute.

        Args:
            mcmc_object (MCMC): MCMC object which contains the fitted values in the store attribute of the object.
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the object_to_plot
            background_model (TemporalBackground, optional): Background model containing the estimated background.
            burn_in (int, optional): Number of burn-in iterations to discard before calculating the quantiles
                and median. Defaults to 0.

        """
        if "y" not in mcmc_object.store:
            raise ValueError("Missing fitted values ('y') in mcmc_store_object")

        if isinstance(sensor_object, Sensor):
            temp = SensorGroup()
            temp.add_sensor(sensor_object)
            sensor_object = deepcopy(temp)

        y_values_overall = mcmc_object.store["y"]
        dict_key = "fitted_values"
        title_text = "Observations and Predicted Model Values Against Time"
        x_label = "Time"
        y_label = "Concentration (ppm)"
        fig = go.Figure()

        for sensor_idx, sensor_key in enumerate(sensor_object.keys()):
            plot_idx = np.array(sensor_object.sensor_index == sensor_idx)

            x_values = sensor_object[sensor_key].time
            y_values = y_values_overall[plot_idx, burn_in:]

            color_idx = sensor_idx % len(sensor_object.color_map)
            color = sensor_object.color_map[color_idx]

            fig = plot_quantiles_from_array(
                fig=fig, x_values=x_values, y_values=y_values, quantiles=[10, 90], color=color, name=sensor_key
            )

        if isinstance(background_model, TemporalBackground):
            fig = plot_quantiles_from_array(
                fig=fig,
                x_values=background_model.time,
                y_values=background_model.bg,
                quantiles=[10, 90],
                color="rgb(186, 186, 186)",
                name="Background",
            )

            fig.for_each_trace(
                lambda trace: (
                    trace.update(showlegend=True, name="Background") if trace.name == "Median for Background" else ()
                ),
            )

        fig = sensor_object.plot_timeseries(fig=fig, color_map=sensor_object.color_map, mode="markers")

        fig.add_annotation(
            x=1,
            y=1.1,
            yref="paper",
            xref="paper",
            xanchor="left",
            yanchor="top",
            font={"size": 12, "color": "#000000"},
            align="left",
            showarrow=False,
            borderwidth=2,
            borderpad=10,
            bgcolor="#ffffff",
            bordercolor="#000000",
            opacity=0.8,
            text=(
                "<b>Point</b>: Real observation<br><b>Line</b>: Predicted Value<br><b>Shading</b>: " + "Quantiles 10-90"
            ),
        )

        if self.layout is not None:
            fig.update_layout(template=self.layout)

        fig.update_layout(title=title_text)
        fig.update_xaxes(title_standoff=20, automargin=True, title_text=x_label)
        fig.update_yaxes(title_standoff=20, automargin=True, title_text=y_label)

        self.figure_dict[dict_key] = fig

    def plot_emission_rate_estimates(self, source_model_object, y_axis_type="linear", **kwargs):
        """Plot the emission rate estimates source model object against MCMC iteration.

        Based on the inputs it plots the results of the mcmc analysis, being the estimated emission rate values for
        each source location together with the total emissions estimate, which is the sum over all source locations.

        The created figure is stored in the figure_dict attribute.

        After the loop over all sources we add an empty trace to have the legend entry and desired legend group
        behaviour.

        Args:
            source_model_object (SourceModel): Source model object which contains the estimated emission rate estimates.
            y_axis_type (str, optional): String to indicate whether the y-axis should be linear of log scale.
            **kwargs (dict): Additional key word arguments, e.g. burn_in, dict_key, used in some specific plots but not
                applicable to all.

        """
        total_emissions = np.nansum(source_model_object.emission_rate, axis=0)
        x_values = np.array(range(total_emissions.size))

        burn_in = kwargs.pop("burn_in", 0)

        dict_key = "estimated_values_plot"
        title_text = "Estimated Values of Sources With Respect to MCMC Iterations"
        x_label = "MCMC Iteration Number"
        y_label = "Estimated Emission<br>Values (kg/hr)"

        fig = go.Figure()

        fig = plot_single_scatter(
            fig=fig,
            x_values=x_values,
            y_values=total_emissions,
            color="rgb(239, 85, 59)",
            name="Total Site Emissions",
            burn_in=burn_in,
            show_legend=True,
        )

        for source_idx in range(source_model_object.emission_rate.shape[0]):
            y_values = source_model_object.emission_rate[source_idx, :]

            fig = plot_single_scatter(
                fig=fig,
                x_values=x_values,
                y_values=y_values,
                color="rgb(102, 197, 204)",
                name=f"Source {source_idx}",
                burn_in=burn_in,
                show_legend=False,
                legend_group="Source traces",
            )

        fig = plot_single_scatter(
            fig=fig,
            x_values=np.array([None]),
            y_values=np.array([None]),
            color="rgb(102, 197, 204)",
            name="Source traces",
            burn_in=0,
            show_legend=True,
        )

        if burn_in > 0:
            fig.add_vline(
                x=burn_in, line_width=3, line_dash="dash", line_color="black", annotation_text=f"\tBurn in: {burn_in}"
            )

        if self.layout is not None:
            fig.update_layout(template=self.layout)

        fig.add_annotation(
            x=1.05,
            y=1.05,
            yref="paper",
            xref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 12, "color": "#000000"},
            showarrow=False,
            borderwidth=2,
            borderpad=10,
            bgcolor="#ffffff",
            bordercolor="#000000",
            opacity=0.8,
            text=(
                "<b>Total Site Emissions</b> are<br>the sum of all estimated<br>"
                "emission rates at a given<br>iteration number."
            ),
        )

        fig.update_layout(title=title_text)
        fig.update_xaxes(title_standoff=20, automargin=True, title_text=x_label)
        fig.update_yaxes(title_standoff=20, automargin=True, title_text=y_label)
        if y_axis_type == "log":
            fig.update_yaxes(type="log")
            dict_key = "log_estimated_values_plot"
        elif y_axis_type != "linear":
            raise ValueError(f"Only linear or log y axis type is allowed, {y_axis_type} was currently specified.")

        self.figure_dict[dict_key] = fig

    def create_empty_mapbox_figure(self, dict_key: str = "map_plot") -> None:
        """Creating an empty mapbox figure to use when you want to add additional traces on a map.

        Args:
            dict_key (str, optional): String key for figure dictionary

        """
        self.figure_dict[dict_key] = go.Figure(
            data=go.Scattermapbox(),
            layout={
                "mapbox_style": "carto-positron",
                "mapbox_center_lat": 0,
                "mapbox_center_lon": 0,
                "mapbox_zoom": 0,
                "mapbox_accesstoken": self.mapbox_token,
            },
        )

    def plot_values_on_map(
        self, dict_key: str, coordinates: LLA, values: np.ndarray, aggregate_function: Callable = np.sum, **kwargs
    ):
        """Plot values on a map based on coordinates.

        Args:
            dict_key (str): Sting key to use in the figure dictionary
            coordinates (LLA): LLA coordinates to use in plotting the values on the map
            values (np.ndarray): Numpy array of values consistent with coordinates to plot on the map
            aggregate_function (Callable, optional): Function which to apply on the data in each hexagonal bin to
                aggregate the data and visualise the result.
            **kwargs (dict): Additional keyword arguments for plotting behaviour (opacity, map_color_scale, num_hexagons,
                show_positions)

        """
        map_color_scale = kwargs.pop("map_color_scale", "YlOrRd")
        num_hexagons = kwargs.pop("num_hexagons", None)
        opacity = kwargs.pop("opacity", 0.8)
        show_positions = kwargs.pop("show_positions", False)

        latitude_check, _ = is_regularly_spaced(coordinates.latitude)
        longitude_check, _ = is_regularly_spaced(coordinates.longitude)
        if latitude_check and longitude_check:
            self.create_empty_mapbox_figure(dict_key=dict_key)
            trace = plot_regular_grid(
                coordinates=coordinates,
                values=values,
                opacity=opacity,
                map_color_scale=map_color_scale,
                tolerance=1e-7,
                unit="",
            )
            self.figure_dict[dict_key].add_trace(trace)
        else:
            fig = plot_hexagonal_grid(
                coordinates=coordinates,
                values=values,
                opacity=opacity,
                map_color_scale=map_color_scale,
                num_hexagons=num_hexagons,
                show_positions=show_positions,
                aggregate_function=aggregate_function,
            )
            fig.update_layout(mapbox_accesstoken=self.mapbox_token, mapbox_style="carto-positron")
            self.figure_dict[dict_key] = fig

        center_longitude = np.mean(coordinates.longitude)
        center_latitude = np.mean(coordinates.latitude)
        self.figure_dict[dict_key].update_layout(
            mapbox={"zoom": 10, "center": {"lon": center_longitude, "lat": center_latitude}}
        )

        if self.layout is not None:
            self.figure_dict[dict_key].update_layout(template=self.layout)

    def plot_quantification_results_on_map(
        self,
        model_object: "ELQModel",
        bin_size_x: float = 1,
        bin_size_y: float = 1,
        normalized_count_limit: float = 0.005,
        burn_in: int = 0,
        show_summary_results: bool = True,
    ):
        """Placeholder for the quantification plots."""
        nof_iterations = model_object.n_iter
        ref_latitude = model_object.components["source"].dispersion_model.source_map.location.ref_latitude
        ref_longitude = model_object.components["source"].dispersion_model.source_map.location.ref_longitude
        ref_altitude = model_object.components["source"].dispersion_model.source_map.location.ref_altitude
        datetime_min_string = model_object.sensor_object.time.min().strftime("%d-%b-%Y, %H:%M:%S")
        datetime_max_string = model_object.sensor_object.time.max().strftime("%d-%b-%Y, %H:%M:%S")

        all_source_locations = model_object.mcmc.store["z_src"]
        min_x = np.nanmin(all_source_locations[0, :, :])
        max_x = np.nanmax(all_source_locations[0, :, :])
        min_y = np.nanmin(all_source_locations[1, :, :])
        max_y = np.nanmax(all_source_locations[1, :, :])

        bin_min_x = np.floor(min_x - 0.1)
        bin_max_x = np.ceil(max_x + 0.1)
        bin_min_y = np.floor(min_y - 0.1)
        bin_max_y = np.ceil(max_y + 0.1)
        bin_min_iteration = burn_in + 0.5
        bin_max_iteration = nof_iterations + 0.5

        max_nof_sources = all_source_locations.shape[1]

        x_edges = np.arange(start=bin_min_x, stop=bin_max_x + bin_size_x, step=bin_size_x)
        y_edges = np.arange(start=bin_min_y, stop=bin_max_y + bin_size_y, step=bin_size_y)
        iteration_edges = np.arange(start=bin_min_iteration, stop=bin_max_iteration + bin_size_y, step=1)

        result_x_vals = all_source_locations[0, :, :].flatten()
        result_y_vals = all_source_locations[1, :, :].flatten()
        result_z_vals = all_source_locations[2, :, :].flatten()
        # 1-indexing for iterations effectively
        result_iteration_vals = np.array(range(nof_iterations)).reshape(1, -1) + 1
        result_iteration_vals = np.tile(result_iteration_vals, (max_nof_sources, 1)).flatten()
        results_estimates = model_object.mcmc.store["s"].flatten()

        result_weighted, _ = np.histogramdd(
            sample=np.array([result_x_vals, result_y_vals, result_iteration_vals]).T,
            bins=[x_edges, y_edges, iteration_edges],
            weights=results_estimates,
            density=False,
        )

        count_result, edges_result = np.histogramdd(
            sample=np.array([result_x_vals, result_y_vals, result_iteration_vals]).T,
            bins=[x_edges, y_edges, iteration_edges],
            density=False,
        )

        enu_x = edges_result[0]
        enu_x = enu_x[:-1] + np.diff(enu_x) / 2
        enu_y = edges_result[1]
        enu_y = enu_y[:-1] + np.diff(enu_y) / 2

        enu_x, enu_y = np.meshgrid(enu_x, enu_y, indexing="ij")

        enu_object_full_grid = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)
        enu_object_full_grid.east = enu_x.flatten()
        enu_object_full_grid.north = enu_y.flatten()
        enu_object_full_grid.up = np.zeros_like(enu_object_full_grid.north)
        lla_object_full_grid = enu_object_full_grid.to_lla()

        _, gridsize_lat = is_regularly_spaced(lla_object_full_grid.latitude, tolerance=1e-6)
        _, gridsize_lon = is_regularly_spaced(lla_object_full_grid.longitude, tolerance=1e-6)

        overall_count = np.sum(count_result, axis=2)
        normalized_count = overall_count / (nof_iterations - burn_in)

        count_boolean = normalized_count >= normalized_count_limit

        enu_object = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)
        enu_object.east = enu_x[count_boolean].flatten()
        enu_object.north = enu_y[count_boolean].flatten()
        enu_object.up = np.zeros_like(enu_object.north)
        lla_object = enu_object.to_lla()

        polygons = [
            geometry.box(
                lla_object.longitude[idx] - gridsize_lon / 2,
                lla_object.latitude[idx] - gridsize_lat / 2,
                lla_object.longitude[idx] + gridsize_lon / 2,
                lla_object.latitude[idx] + gridsize_lat / 2,
            )
            for idx in range(lla_object.nof_observations)
        ]

        if show_summary_results:
            summary_trace = self.create_summary_trace(
                result_iteration_vals=result_iteration_vals,
                burn_in=burn_in,
                result_x_vals=result_x_vals,
                result_y_vals=result_y_vals,
                result_z_vals=result_z_vals,
                results_estimates=results_estimates,
                count_boolean=count_boolean,
                x_edges=x_edges,
                y_edges=y_edges,
                nof_iterations=nof_iterations,
                ref_latitude=ref_latitude,
                ref_longitude=ref_longitude,
                ref_altitude=ref_altitude,
            )

        self.create_empty_mapbox_figure(dict_key="count_map")
        trace = plot_polygons_on_map(
            polygons=polygons,
            values=normalized_count[count_boolean].flatten(),
            opacity=0.8,
            name="normalized_count",
            colorbar={"title": "Normalized Count", "orientation": "h"},
            map_color_scale="Bluered",
        )
        self.figure_dict["count_map"].add_trace(trace)
        self.figure_dict["count_map"].update_layout(
            mapbox_accesstoken=self.mapbox_token,
            mapbox_style="carto-positron",
            mapbox={"zoom": 15, "center": {"lon": ref_longitude, "lat": ref_latitude}},
            title=f"Source location probability "
            f"(>={normalized_count_limit}) for "
            f"{datetime_min_string} to {datetime_max_string}",
            font_family="Futura",
            font_size=15,
        )
        model_object.sensor_object.plot_sensor_location(self.figure_dict["count_map"])
        self.figure_dict["count_map"].update_traces(showlegend=False)

        adjusted_result_weights = result_weighted.copy()
        adjusted_result_weights[adjusted_result_weights == 0] = np.nan

        median_of_all_emissions = np.nanmedian(adjusted_result_weights, axis=2)

        self.create_empty_mapbox_figure(dict_key="median_map")

        trace = plot_polygons_on_map(
            polygons=polygons,
            values=median_of_all_emissions[count_boolean].flatten(),
            opacity=0.8,
            name="median_emission",
            colorbar={"title": "Median Emission", "orientation": "h"},
            map_color_scale="Bluered",
        )
        self.figure_dict["median_map"].add_trace(trace)
        self.figure_dict["median_map"].update_layout(
            mapbox_accesstoken=self.mapbox_token,
            mapbox_style="carto-positron",
            mapbox={"zoom": 15, "center": {"lon": ref_longitude, "lat": ref_latitude}},
            title=f"Median emission rate estimate for {datetime_min_string} to {datetime_max_string}",
            font_family="Futura",
            font_size=15,
        )
        model_object.sensor_object.plot_sensor_location(self.figure_dict["median_map"])
        self.figure_dict["median_map"].update_traces(showlegend=False)

        iqr_of_all_emissions = np.nanquantile(a=adjusted_result_weights, q=0.75, axis=2) - np.nanquantile(
            a=adjusted_result_weights, q=0.25, axis=2
        )
        self.create_empty_mapbox_figure(dict_key="iqr_map")

        trace = plot_polygons_on_map(
            polygons=polygons,
            values=iqr_of_all_emissions[count_boolean].flatten(),
            opacity=0.8,
            name="iqr_emission",
            colorbar={"title": "IQR", "orientation": "h"},
            map_color_scale="Bluered",
        )
        self.figure_dict["iqr_map"].add_trace(trace)
        self.figure_dict["iqr_map"].update_layout(
            mapbox_accesstoken=self.mapbox_token,
            mapbox_style="carto-positron",
            mapbox={"zoom": 15, "center": {"lon": ref_longitude, "lat": ref_latitude}},
            title=f"Inter Quartile range (25%-75%) of emission rate "
            f"estimate for {datetime_min_string} to {datetime_max_string}",
            font_family="Futura",
            font_size=15,
        )
        model_object.sensor_object.plot_sensor_location(self.figure_dict["iqr_map"])
        self.figure_dict["iqr_map"].update_traces(showlegend=False)

        if show_summary_results:
            self.figure_dict["count_map"].add_trace(summary_trace)
            self.figure_dict["count_map"].update_traces(showlegend=True)
            self.figure_dict["median_map"].add_trace(summary_trace)
            self.figure_dict["median_map"].update_traces(showlegend=True)
            self.figure_dict["iqr_map"].add_trace(summary_trace)
            self.figure_dict["iqr_map"].update_traces(showlegend=True)

    def plot_coverage(
        self,
        coordinates: LLA,
        couplings: np.ndarray,
        threshold_function: Callable = np.max,
        coverage_threshold: float = 6,
        opacity: float = 0.8,
        map_color_scale="jet",
    ):
        """Creates a coverage plot using the coverage function from Gaussian Plume.

        Args:
            coordinates (LLA object): A LLA coordinate object containing a set of locations.
            couplings (np.array): The calculated values of coupling (The 'A matrix') for a set of wind data.
            threshold_function (Callable, optional): Callable function which returns some single value that defines the
                                         maximum or 'threshold' coupling. Examples: np.quantile(q=0.9),
                                         np.max, np.mean. Defaults to np.max.
            coverage_threshold (float, optional): The threshold value of the estimated emission rate which is
                                                  considered to be within the coverage. Defaults to 6 kg/hr.
            opacity (float): The opacity of the grid cells when they are plotted.
            map_color_scale (str): The string which defines which plotly colour scale should be used when plotting
                                   the values.

        """
        coverage_values = GaussianPlume(source_map=None).compute_coverage(
            couplings=couplings, threshold_function=threshold_function, coverage_threshold=coverage_threshold
        )
        self.plot_values_on_map(
            dict_key="coverage_map",
            coordinates=coordinates,
            values=coverage_values,
            aggregate_function=np.max,
            opacity=opacity,
            map_color_scale=map_color_scale,
        )

    @staticmethod
    def create_summary_trace(
        result_x_vals: np.ndarray,
        result_y_vals: np.ndarray,
        result_z_vals: np.ndarray,
        results_estimates: np.ndarray,
        result_iteration_vals: np.ndarray,
        count_boolean: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        nof_iterations: int,
        burn_in: int,
        ref_latitude: float,
        ref_longitude: float,
        ref_altitude: float,
    ) -> go.Scattermapbox:
        """Helper function to create the summary information to plot on top of map type plots.

        We identify all blobs of estimates which appear close together on the map by looking at connected pixels in the
        count_boolean array. Next we find the summary statistics for all estimates in that blob like overall median and
        IQR estimate, mean location and the likelihood of that blob.

        When multiple sources are present in the same blob at the same iteration we first sum those emission rate
        estimates before taking the median.

        The summary statistics are also printed out on screen.

        Args:
            result_x_vals (np.ndarray): X-coordinate of estimates, flattened array of (n_sources_max * nof_iterations,).
            result_y_vals (np.ndarray): Y-coordinate of estimates, flattened array of (n_sources_max * nof_iterations,).
            result_z_vals (np.ndarray): Z-coordinate of estimates, flattened array of (n_sources_max * nof_iterations,).
            results_estimates (np.ndarray): Emission rate estimates, flattened array of
                (n_sources_max * nof_iterations,).
            result_iteration_vals (np.ndarray): Iteration number corresponding each estimated value, flattened array
                of (n_sources_max * nof_iterations,).
            count_boolean (np.ndarray): Boolean array which indicates if likelihood of pixel is over threshold.
            x_edges (np.ndarray): Pixel edges x-coordinates.
            y_edges (np.ndarray): Pixel edges y-coordinates.
            nof_iterations (int): Number of iterations used in MCMC.
            burn_in (int): Burn-in used in MCMC.
            ref_latitude (float): Reference latitude in degrees of ENU coordinate system.
            ref_longitude (float): Reference longitude in degrees of ENU coordinate system.
            ref_altitude (float): Reference altitude in meters of ENU coordinate system.

        Returns:
            summary_trace (go.Scattermapbox): Trace with summary information to plot on top of map type plots.

        """
        labeled_array, num_features = label(input=count_boolean, structure=np.ones((3, 3)))

        burn_in_bool = result_iteration_vals > burn_in
        nan_x_vals = np.isnan(result_x_vals)
        nan_y_vals = np.isnan(result_y_vals)
        nan_z_vals = np.isnan(result_z_vals)
        no_nan_idx = np.logical_not(np.logical_or(np.logical_or(nan_x_vals, nan_y_vals), nan_z_vals))
        no_nan_and_burn_in_bool = np.logical_and(no_nan_idx, burn_in_bool)
        result_x_vals_no_nan = result_x_vals[no_nan_and_burn_in_bool]
        result_y_vals_no_nan = result_y_vals[no_nan_and_burn_in_bool]
        result_z_vals_no_nan = result_z_vals[no_nan_and_burn_in_bool]
        results_estimates_no_nan = results_estimates[no_nan_and_burn_in_bool]
        result_iteration_vals_no_nan = result_iteration_vals[no_nan_and_burn_in_bool]

        x_idx = np.digitize(result_x_vals_no_nan, x_edges, right=False) - 1
        y_idx = np.digitize(result_y_vals_no_nan, y_edges, right=False) - 1
        bin_numbers = np.ravel_multi_index((x_idx, y_idx), labeled_array.shape)

        bin_numbers_per_label = [
            np.ravel_multi_index(np.nonzero(labeled_array == value), labeled_array.shape)
            for value in np.array(range(num_features)) + 1
        ]

        summary_result = pd.DataFrame()
        for label_idx, curr_bins in enumerate(bin_numbers_per_label):
            boolean_for_result = np.isin(bin_numbers, curr_bins)
            mean_x = np.mean(result_x_vals_no_nan[boolean_for_result])
            mean_y = np.mean(result_y_vals_no_nan[boolean_for_result])
            mean_z = np.mean(result_z_vals_no_nan[boolean_for_result])

            unique_iteration_vals, indices, counts = np.unique(
                result_iteration_vals_no_nan[boolean_for_result], return_inverse=True, return_counts=True
            )
            nof_iterations_present = unique_iteration_vals.size
            blob_likelihood = nof_iterations_present / (nof_iterations - burn_in)
            single_idx = np.argwhere(counts == 1)
            results_estimates_for_blob = results_estimates_no_nan[boolean_for_result]
            temp_estimate_result = results_estimates_for_blob[indices[single_idx.flatten()]]
            multiple_idx = np.argwhere(counts > 1)
            for single_idx in multiple_idx:
                temp_val = np.sum(results_estimates_for_blob[indices == single_idx])
                temp_estimate_result = np.append(temp_estimate_result, temp_val)

            median_estimate = np.median(temp_estimate_result)
            iqr_estimate = np.nanquantile(a=temp_estimate_result, q=0.75) - np.nanquantile(
                a=temp_estimate_result, q=0.25
            )
            lower_bound = np.nanquantile(a=temp_estimate_result, q=0.025)
            upper_bound = np.nanquantile(a=temp_estimate_result, q=0.975)
            enu_object = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)
            enu_object.east = mean_x
            enu_object.north = mean_y
            enu_object.up = mean_z
            lla_object = enu_object.to_lla()

            summary_result.loc[label_idx, "latitude"] = lla_object.latitude
            summary_result.loc[label_idx, "longitude"] = lla_object.longitude
            summary_result.loc[label_idx, "altitude"] = lla_object.altitude
            summary_result.loc[label_idx, "height"] = mean_z
            summary_result.loc[label_idx, "median_estimate"] = median_estimate
            summary_result.loc[label_idx, "quantile_025"] = lower_bound
            summary_result.loc[label_idx, "quantile_975"] = upper_bound
            summary_result.loc[label_idx, "iqr_estimate"] = iqr_estimate
            summary_result.loc[label_idx, "absolute_count_iterations"] = nof_iterations_present
            summary_result.loc[label_idx, "blob_likelihood"] = blob_likelihood

        summary_text_values = [
            f"<b>Source ID</b>: {value}<br>"
            f"<b>(Lon, Lat, Alt)</b> ([deg], [deg], [m]):<br>"
            f"({summary_result.longitude[value]:.7f}, "
            f"{summary_result.latitude[value]:.7f}, {summary_result.altitude[value]:.3f})<br>"
            f"<b>Height</b>: {summary_result.height[value]:.3f} [m]<br>"
            f"<b>Median emission rate</b>: {summary_result.median_estimate[value]:.4f} [kg/hr]<br>"
            f"<b>2.5% quantile</b>: {summary_result.quantile_025[value]:.3f} [kg/hr]<br>"
            f"<b>97.5% quantile</b>: {summary_result.quantile_975[value]:.3f} [kg/hr]<br>"
            f"<b>IQR</b>: {summary_result.iqr_estimate[value]:.4f} [kg/hr]<br>"
            f"<b>Blob present during</b>: "
            f"{summary_result.absolute_count_iterations[value]:.0f} iterations<br>"
            f"<b>Blob likelihood</b>: {summary_result.blob_likelihood[value]:.5f}<br>"
            for value in summary_result.index
        ]

        summary_trace = go.Scattermapbox(
            lat=summary_result.latitude,
            lon=summary_result.longitude,
            mode="markers",
            marker=go.scattermapbox.Marker(size=14, color="black"),
            text=summary_text_values,
            name="Summary",
            hoverinfo="text",
        )

        summary_result.index.name = "source_ID"
        summary_result = summary_result.astype({"absolute_count_iterations": "int"})
        print("Summary results:")
        print(summary_result.to_string(float_format=lambda x: "%.7f" % x))

        return summary_trace


def is_regularly_spaced(array: np.ndarray, tolerance: float = 0.01, return_delta: bool = True):
    """Determines whether an input array is regularly spaced, within some (absolute) tolerance.

    Gets the large differences (defined by tolerance) in the array, and sees whether all of them are within 5% of one
    another.

    Args:
        array (np.ndarray): Input array to be analysed.
        tolerance (float, optional): Absolute value above which the difference between values is considered significant.
            Defaults to 0.01.
        return_delta (bool, optional): Whether to return the value of the regular grid spacing. Defaults to True.

    Returns:
        (bool): Whether or not the grid is regularly spaced.
        (float): The value of the regular grid spacing.

    """
    unique_vals = np.unique(array)
    diff_unique_vals = np.diff(unique_vals)
    diff_big = diff_unique_vals[diff_unique_vals > tolerance]

    boolean = np.all([np.isclose(diff_big[i], diff_big[i + 1], rtol=0.05) for i in range(len(diff_big) - 1)])

    if return_delta:
        return boolean, np.mean(diff_big)

    return boolean, None
