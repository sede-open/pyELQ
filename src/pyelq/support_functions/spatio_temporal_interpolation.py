# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Spatio-temporal interpolation module.

Support function to perform interpolation in various ways

"""
import warnings
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def interpolate(
    location_in: np.ndarray = None,
    time_in: Union[np.ndarray, pd.arrays.DatetimeArray] = None,
    values_in: np.ndarray = None,
    location_out: np.ndarray = None,
    time_out: Union[np.ndarray, pd.arrays.DatetimeArray] = None,
    **kwargs,
) -> np.ndarray:
    """Interpolates data based on input.

    Interpolation using scipy.griddata function. Which in turn uses linear barycentric interpolation.

    It is assumed that the shape of location_in, time_in and values_in is consistent

    When time_out has the same size as number of rows of location_out, it is assumed these are aligned and be treated as
    consistent, hence the output will be a column vector.
    If this is not the case an interpolation will be performed for all combinations of rows in location out with times
    of time_out and output wil be shaped as [nof_location_values x dimension]

    If location_out == None, we only perform temporal (1D) interpolation.
    If time_out == None we only perform spatial  interpolation

    If linear interpolation is not possible for spatio or spatiotemporal interpolation, we use nearest neighbor
    interpolation, a warning will be displayed

    Args:
        location_in (np.ndarray): Array of size [nof_values x dimension] with locations to interpolate from
        time_in (Union[np.ndarray, pd.arrays.DatetimeArray]): Array of size [nof_values x 1] with timestamps or some
            form of time values (seconds) to interpolate from
        values_in (np.ndarray): Array of size [nof_values x 1] with values to interpolate from
        location_out (np.ndarray): Array of size [nof_location_values x dimension] with locations to interpolate to
        time_out (Union[np.ndarray, pd.arrays.DatetimeArray]): Array of size [nof_time_values x 1] with
            timestamps or some form of time values (seconds) to interpolate to
        **kwargs (dict): Other keyword arguments which get passed into the griddata interpolation function

    Returns:
        result (np.ndarray): Array of size [nof_location_values x nof_time_values] with interpolated values

    """
    _sense_check_interpolate_inputs(
        location_in=location_in, time_in=time_in, values_in=values_in, location_out=location_out, time_out=time_out
    )

    if (
        time_out is not None
        and isinstance(time_out, pd.arrays.DatetimeArray)
        and isinstance(time_in, pd.arrays.DatetimeArray)
    ):
        min_time_out = np.amin(time_out)
        time_out = (time_out - min_time_out).total_seconds()
        time_in = (time_in - min_time_out).total_seconds()

    if location_out is None:
        return _griddata(points_in=time_in, values=values_in, points_out=time_out, **kwargs)

    if time_out is None:
        return _griddata(points_in=location_in, values=values_in, points_out=location_out, **kwargs)

    if location_in.shape[0] != time_in.size:
        raise ValueError("Location and time are do not have consistent sizes")

    if location_out.shape[0] != time_out.size:
        location_temp = np.tile(location_out, (time_out.size, 1))
        time_temp = np.repeat(time_out.squeeze(), location_out.shape[0])
        out_array = np.column_stack((location_temp, time_temp))
    else:
        out_array = np.column_stack((location_out, time_out))

    in_array = np.column_stack((location_in, time_in))

    result = _griddata(points_in=in_array, values=values_in, points_out=out_array, **kwargs)

    if location_out.shape[0] != time_out.size:
        result = result.reshape((location_out.shape[0], time_out.size), order="C")

    return result


def _sense_check_interpolate_inputs(
    location_in: np.ndarray,
    time_in: Union[np.ndarray, pd.arrays.DatetimeArray],
    values_in: np.ndarray,
    location_out: np.ndarray,
    time_out: Union[np.ndarray, pd.arrays.DatetimeArray],
):
    """Helper function to sense check inputs and raise errors when applicable.

    Args:
        location_in (np.ndarray): Array of size [nof_values x dimension] with locations to interpolate from
        time_in (Union[np.ndarray, pd.arrays.DatetimeArray]): Array of size [nof_values x 1] with timestamps or some
            form of time values (seconds) to interpolate from
        values_in (np.ndarray): Array of size [nof_values x 1] with values to interpolate from
        location_out (np.ndarray): Array of size [nof_location_values x dimension] with locations to interpolate to
        time_out (Union[np.ndarray, pd.arrays.DatetimeArray]): Array of size [nof_time_values x 1] with
        timestamps or some form of time values (seconds) to interpolate to

    Raises:
        ValueError: When inputs do not match up.

    """
    if location_out is not None and location_in is None:
        raise ValueError("Cannot specify output location without input location")
    if time_out is not None and time_in is None:
        raise ValueError("Cannot specify output time without input time")
    if values_in is None:
        raise ValueError("Must provide values_in")
    if location_out is None and time_out is None:
        raise ValueError("location_out or time_out not specified. Need to specify somewhere to interpolate to")


def _griddata(points_in: np.ndarray, values: np.ndarray, points_out: np.ndarray, **kwargs):
    """Wrapped function to handle special cases around the gridded interpolate.

    Will try nearest neighbour method when few enough points that spatial cases fail.

    Syntax like scipy.griddata

    Args:
        points_in (np.ndarray): 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D
        nd-arrays with shape (n,). Data point coordinates.
        values (np.ndarray): _ndarray of float or complex, shape (n,). Data values
        points_out (np.ndarray): 2-D ndarray of floats with shape (m, D), or length D tuple of nd-arrays
          broadcastable to the same shape. Points at which to interpolate data.

    Returns:
        ndarray: Array of interpolated values.

    """
    if values.size == 1:
        return np.ones((points_out.shape[0], 1)) * values

    try:
        return griddata(points=points_in, values=values.flatten(), xi=points_out, **kwargs)
    except RuntimeError:
        warnings.warn(
            "Warning linear interpolation did not succeed, most likely too few input points (<5),"
            "trying again with method==nearest"
        )
        if "method" in kwargs:
            del kwargs["method"]
        return griddata(points=points_in, values=values, xi=points_out, method="nearest", **kwargs)


def temporal_resampling(
    time_in: Union[np.ndarray, pd.arrays.DatetimeArray],
    values_in: np.ndarray,
    time_bin_edges: Union[np.ndarray, pd.arrays.DatetimeArray],
    aggregate_function: str = "mean",
    side: str = "center",
) -> Tuple[Union[np.ndarray, pd.arrays.DatetimeArray], np.ndarray]:
    """Resamples data into a set of time bins.

    Checks which values of time_in are withing 2 consecutive values of time_bin_edges and performs the aggregate
    function on the corresponding values from values_in. time_in values outside the time_bin_edges are ignored.
    Empty bins will be assigned a 'NaN' value.

    When 'time_in' is a sequence of time stamps, a DatetimeArray should be used. Otherwise, a np.ndarray should be used.

    Args:
        time_in (Union[np.ndarray, pd.arrays.DatetimeArray]): A vector of times which correspond to values_in.
        values_in (np.ndarray): A vector of the values to be resampled.
        time_bin_edges (Union[np.ndarray, pd.arrays.DatetimeArray]): A vector of times which define the edges of the
                                                                     bins into which the data will be resampled.
        aggregate_function (str, optional): The function which is used to aggregate the data after it has been
                                            sorted into bins. Defaults to mean.
        side (str, optional): Which side of the time bins should be used to generate times_out. Possible values are:
                              'left', 'center', and 'right'. Defaults to 'center'.

    Returns:
        time_out (Union[np.ndarray, pd.arrays.DatetimeArray]): Vector-like object containing the times of the resampled
                                                               values consistent with time_in dtype and side input
                                                               argument.
        values_out (np.ndarray): A vector of resampled values, according to the time bins and the aggregate function.

    Raises:
        ValueError: If any of the input arguments are not of the correct type or shape, this error is raised.

    """
    if not isinstance(time_bin_edges, type(time_in)) or values_in.size != time_in.size:
        raise ValueError("Arguments 'time_in', 'time_bin_edges' and/or 'values_in' are not of consistent type or size.")

    if not isinstance(aggregate_function, str):
        raise ValueError("The supplied 'aggregate_function' is not a string.")

    if side == "center":
        time_out = np.diff(time_bin_edges) / 2 + time_bin_edges[:-1]
    elif side == "left":
        time_out = time_bin_edges[:-1]
    elif side == "right":
        time_out = time_bin_edges[1:]
    else:
        raise ValueError(f"The 'side' argument must be 'left', 'center', or 'right', but received '{side}'.")

    zero_value = 0
    if isinstance(time_bin_edges, pd.arrays.DatetimeArray):
        zero_value = np.array(0).astype("<m8[ns]")

    if not np.all(np.diff(time_bin_edges) > zero_value):
        raise ValueError("Argument 'time_bin_edges' does not monotonically increase.")

    if np.any(time_in < time_bin_edges[0]) or np.any(time_in > time_bin_edges[-1]):
        warnings.warn("Values in time_in are outside of range of time_bin_edges. These values will be ignored.")

    index = np.searchsorted(time_bin_edges, time_in, side="left")
    grouped_vals = pd.Series(values_in).groupby(index).agg(aggregate_function)
    grouped_vals = grouped_vals.drop(index=[0, time_bin_edges.size], errors="ignore").sort_index()

    values_out = np.full(time_out.shape, np.nan)
    values_out[grouped_vals.index - 1] = grouped_vals.to_numpy()

    return time_out, values_out
