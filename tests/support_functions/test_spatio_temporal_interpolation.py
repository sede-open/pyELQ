# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for spatio-temporal interpolation module."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import pyelq.support_functions.spatio_temporal_interpolation as sti


def test_error_catching():
    """Test to see if errors are being thrown with incorrect inputs."""
    with pytest.raises(ValueError):
        sti.interpolate(location_out=np.array(3))
    with pytest.raises(ValueError):
        sti.interpolate(time_out=np.array(3))
    with pytest.raises(ValueError):
        sti.interpolate(values_in=None)
    with pytest.raises(ValueError):
        sti.interpolate(location_out=None, time_out=None, location_in=np.array(3), values_in=np.array(3))


def test_default_returns():
    """Tests if default values are returned Tests if same values are returned when input/output locations/times are the
    same."""

    loc_in = np.array([[0, 0, 0], [1, 1, 1]])
    time_in = pd.date_range(pd.Timestamp.now(), periods=loc_in.shape[0], freq="h").array[:, None]

    rng = np.random.default_rng(42)
    vals = rng.random((loc_in.shape[0], 1))
    # check if same input/output locations and time give the same answer
    return_vals = sti.interpolate(
        location_in=loc_in, time_in=time_in, values_in=vals, location_out=loc_in, time_out=time_in
    )
    assert np.all(return_vals == vals)
    return_vals = sti.interpolate(location_in=loc_in, time_in=time_in, values_in=vals, location_out=loc_in)
    assert np.all(return_vals == vals)
    return_vals = sti.interpolate(location_in=loc_in, time_in=time_in, values_in=vals, time_out=time_in)
    assert np.all(return_vals == vals)


def test_single_value():
    """Tests if all interpolated values are set to the same value when 1 input value is provided."""
    loc_in = np.array([[0, 0, 0], [1, 1, 1]])
    n_obs = loc_in.shape[0]
    time_in = pd.date_range(pd.Timestamp.now(), periods=n_obs, freq="h").array[:, None]
    rng = np.random.default_rng(42)
    vals = rng.random((loc_in.shape[0], 1))

    # Check if we get the same output for all values when 1 value is provided
    return_vals = sti.interpolate(
        location_in=loc_in[[0], :], time_in=time_in[[0]], values_in=vals[[0]], location_out=loc_in, time_out=time_in
    )
    assert np.all(return_vals == vals[0])
    assert return_vals.shape == (n_obs, 1)
    return_vals = sti.interpolate(
        location_in=loc_in[[0], :], time_in=time_in[[0]], values_in=vals[[0]], location_out=loc_in
    )
    assert np.all(return_vals == vals[0])
    assert return_vals.shape == (n_obs, 1)
    return_vals = sti.interpolate(
        location_in=loc_in[[0], :], time_in=time_in[[0]], values_in=vals[[0]], time_out=time_in
    )
    assert np.all(return_vals == vals[0])
    assert return_vals.shape == (n_obs, 1)


def test_temporal_interpolation():
    """Check interpolation value with simple manually calculated value for temporal interpolation (hence linear
    interpolation in 1d) Also checks if we get the same values when an array of integers (representing seconds) is
    supplied instead of an array of datetimes."""
    periods = 10
    time_in = pd.date_range(pd.Timestamp.now(), periods=periods, freq="s").array[:, None]
    time_in_array = np.array(range(periods))[:, None]

    rng = np.random.default_rng(42)
    vals = rng.random(time_in.size)
    random_index = rng.integers(0, periods - 1)
    random_factor = rng.random()
    return_vals = sti.interpolate(
        time_in=time_in, values_in=vals, time_out=time_in[[random_index]] + random_factor * pd.Timedelta(1, unit="sec")
    )
    assert np.allclose(return_vals, random_factor * (vals[random_index + 1] - vals[random_index]) + vals[random_index])
    return_vals_array = sti.interpolate(
        time_in=time_in_array, values_in=vals, time_out=time_in_array[[random_index]] + random_factor
    )
    assert np.allclose(return_vals, return_vals_array)


def test_nearest_neighbour():
    """Test to check spatial interpolation when we don't have more than 5 points and hence want to check for nearest
    value."""
    loc_in = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    rng = np.random.default_rng(42)
    vals = rng.random(loc_in.shape[0])
    return_vals = sti.interpolate(location_in=loc_in, values_in=vals, location_out=loc_in[0, :] + 1e-6, method="linear")
    assert np.all(return_vals == vals[0])


def test_spatial_interpolation():
    """Test spatial interpolation by checking if value interpolated at center of tetrahedron is actually the mean of the
    values on the vertices of the tetrahedron."""
    # check spatial interpolation with a cube
    loc_in = np.array([[0, 0, 0], [0, 1, 0], [1, 0.5, 0], [0.5, 0.5, 1]])
    rng = np.random.default_rng(42)
    vals = rng.random((loc_in.shape[0], 1))
    return_vals = sti.interpolate(
        location_in=loc_in, values_in=vals, location_out=np.mean(loc_in, axis=0, keepdims=True)
    )
    assert np.allclose(return_vals, np.mean(vals))


def test_same_value():
    """Test to check if all values are the same we get that value."""
    loc_in = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    vals = np.ones((loc_in.shape[0], 1))
    return_vals = sti.interpolate(
        location_in=loc_in, values_in=vals, location_out=np.mean(loc_in, axis=0, keepdims=True)
    )

    assert np.all(return_vals == 1)


def test_fill_value():
    """Test to check if fill value argument works for point outside of interpolation points."""
    loc_in = np.array([[0, 0, 0], [0, 1, 0], [1, 0.5, 0], [0.5, 0.5, 1]])
    rng = np.random.default_rng(42)
    vals = rng.random((loc_in.shape[0], 1))
    return_vals = sti.interpolate(
        location_in=loc_in, values_in=vals, location_out=np.array([[-1, -1, -1]]), fill_value=-99
    )
    assert np.all(return_vals == -99)


def test_consistent_shapes():
    """Test if output shapes are consistent with provided input."""
    loc_in = np.array([[0, 0, 0], [1, 1, 1]])
    time_in = pd.date_range(pd.Timestamp.now(), periods=loc_in.shape[0] - 1, freq="h").array[:, None]
    rng = np.random.default_rng(42)
    vals = rng.random((loc_in.shape[0], 1))
    with pytest.raises(ValueError):
        sti.interpolate(location_in=loc_in, time_in=time_in, values_in=vals, location_out=loc_in, time_out=time_in)

    loc_in = np.array([[0, 0, 0], [0, 1, 0], [1, 0.5, 0], [0.5, 0.5, 1]])
    time_in = pd.date_range(pd.Timestamp.now(), periods=loc_in.shape[0], freq="h").array[:, None]
    vals = rng.random((loc_in.shape[0], 1))
    return_vals = sti.interpolate(
        location_in=loc_in, time_in=time_in, values_in=vals, location_out=loc_in, time_out=time_in
    )
    assert return_vals.shape == (loc_in.shape[0], 1)

    time_out = time_in[:-1, :]
    return_vals = sti.interpolate(
        location_in=loc_in, time_in=time_in, values_in=vals, location_out=loc_in, time_out=time_out
    )
    assert return_vals.shape == (loc_in.shape[0], time_out.size)


def test_temporal_resampling():
    """This test function generates a set of 100 synthetic data points from 1st January 2000.

    It then finds what
    the correct values_out would be, and afterwards shuffles the data. It uses this to check that temporal_resampling()
    can do the following:
    1: Run without error.
    2: Handle any incorrect input arguments including incorrect types and shapes.
    3. Test that the values are correctly resampled in time, regardless of:
        A: The settings used
        B: The order of the values in time_in
        C: Any values outside the time bins.

    """
    n_values_in = 100
    n_time_out = 10

    rng = np.random.default_rng(42)
    values_in = np.array(rng.random(n_values_in))
    time_in = [datetime(2000, 1, 1, 0, 0, 1) + timedelta(minutes=i) for i in range(n_values_in)]
    time_bin_edges = pd.to_datetime(
        [datetime(2000, 1, 1) + timedelta(minutes=i * 10) for i in range(n_time_out + 1)]
    ).array

    correct_values_out_mean = np.array([np.mean(i) for i in np.split(values_in, n_time_out)])
    correct_values_out_max = np.array([np.max(i) for i in np.split(values_in, n_time_out)])
    correct_values_out_min = np.array([np.min(i) for i in np.split(values_in, n_time_out)])

    time_bin_edges_non_monotonic = pd.Series(list(time_bin_edges)[:-1] + [datetime(1999, 1, 1)]).array
    time_in = pd.to_datetime(time_in + [datetime(2001, 1, 1)]).array
    values_in = np.append(values_in, 1000000)

    p = rng.permutation(len(time_in))
    time_in = time_in[p]
    values_in = values_in[p]

    incorrect_arguments_list = [
        [time_in[:3], values_in, time_bin_edges],
        [np.array(time_in), values_in, time_bin_edges],
        [time_in, values_in, time_bin_edges, "mean", "nonsense_text"],
        [time_in, values_in, time_bin_edges, np.mean],
        [time_in, values_in, time_bin_edges_non_monotonic],
    ]

    for incorrect_arguments in incorrect_arguments_list:
        with pytest.raises(ValueError):
            sti.temporal_resampling(*incorrect_arguments)

    time_out, values_out = sti.temporal_resampling(time_in, values_in, time_bin_edges, "mean", "center")
    correct_time_out = np.diff(time_bin_edges) / 2 + time_bin_edges[:-1]
    assert (time_out == correct_time_out).all()
    assert np.allclose(values_out, correct_values_out_mean)

    time_out, values_out = sti.temporal_resampling(time_in, values_in, time_bin_edges, "max", "left")
    correct_time_out = time_bin_edges[:-1]
    assert (time_out == correct_time_out).all()
    assert np.allclose(values_out, correct_values_out_max)

    time_out, values_out = sti.temporal_resampling(time_in, values_in, time_bin_edges, "min", "right")
    correct_time_out = time_bin_edges[1:]
    assert (time_out == correct_time_out).all()
    assert np.allclose(values_out, correct_values_out_min)


def test_temporal_resampling_empty_bins():
    """This test function test to see if the temporal resampling provides a nan value for an empty bin."""
    time_in = np.array([1, 3])
    values_in = np.array([1, 3])
    time_bin_edges = np.array([0.5, 1.5, 2.5, 3.5])
    correct_values_out = np.array([1, np.nan, 3])
    _, values_out = sti.temporal_resampling(time_in, values_in, time_bin_edges, "mean", "center")
    assert np.allclose(values_out, correct_values_out, equal_nan=True)
