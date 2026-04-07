# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pre-processing class."""

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pyelq.meteorology import Meteorology, MeteorologyGroup
from pyelq.preprocessing import Preprocessor


def get_time_lims(sensor_group):
    """Extract the time limits from the sensor group."""
    min_time, max_time = datetime.now(), datetime.now()
    for sns in sensor_group.values():
        min_time = np.minimum(min_time, np.min(sns.time))
        max_time = np.maximum(max_time, np.max(sns.time))
    return min_time, max_time


@pytest.fixture(name="time_bin_edges")
def fix_time_bin_edges(sensor_group):
    """Fix the time bin edges to be used for aggregation."""
    min_time, max_time = get_time_lims(sensor_group=sensor_group)
    min_time, max_time = min_time - timedelta(seconds=60), max_time + timedelta(seconds=60)
    time_bin_edges = pd.array(pd.date_range(min_time, max_time, freq="120s"), dtype="datetime64[ns]")
    return time_bin_edges


@pytest.fixture(name="block_times")
def fix_block_times(sensor_group):
    """Fix the time bin edges for re-blocking the processed data."""
    min_time, max_time = get_time_lims(sensor_group=sensor_group)
    min_time, max_time = min_time - timedelta(hours=1), max_time + timedelta(hours=1)
    block_times = pd.array(pd.date_range(min_time, max_time, freq="1200s"), dtype="datetime64[ns]")
    return block_times


def add_random_nans(data_object, fields, percent_nan):
    """Take in a data object (Sensor or Meteorology) and add NaNs in random locations."""
    for field in fields:
        idx_nans = np.random.choice(
            np.arange(data_object.time.shape[0]),
            int(np.floor(data_object.time.shape[0] * percent_nan / 100)),
            replace=False,
        )
        data_with_nan = getattr(data_object, field)
        data_with_nan[idx_nans] = np.nan
        setattr(data_object, field, data_with_nan)
    return data_object


@pytest.fixture(name="sensor_mod", params=[False, True], ids=["no_nans", "sns_nans"])
def fix_sensor_mod(request, sensor_group):
    """Generate versions of the supplied sensor object that do/don't have NaNs."""
    with_nans = request.param
    if with_nans:
        for sns in sensor_group.values():
            sns = add_random_nans(sns, ["concentration"], percent_nan=5.0)
    return sensor_group


@pytest.fixture(name="meteorology", params=[False, True], ids=["no_nans", "met_nans"])
def fix_meteorology(request, sensor_group):
    """Fix a meteorology object for the preprocessing test.

    Sets up the wind direction to be between 358 and 2 degrees, so that we can check that the binning in the
    preprocessing can recover values in this range.

    """
    with_nans = request.param
    min_time, max_time = get_time_lims(sensor_group=sensor_group)
    meteorology = Meteorology()
    meteorology.time = pd.array(pd.date_range(min_time, max_time, freq="1s"), dtype="datetime64[ns]")
    meteorology.wind_speed = 1.9 + 0.2 * np.random.random_sample(size=meteorology.time.shape)
    meteorology.wind_direction = np.mod(358.0 + 4.0 * np.random.random_sample(size=meteorology.time.shape), 360)
    meteorology.wind_turbulence_horizontal = 10.0 * np.ones(shape=meteorology.time.shape)
    meteorology.wind_turbulence_vertical = 10.0 * np.ones(shape=meteorology.time.shape)
    meteorology.temperature = 293.0 * np.ones(shape=meteorology.time.shape)
    meteorology.pressure = 101.0 * np.ones(shape=meteorology.time.shape)
    if with_nans:
        meteorology = add_random_nans(
            meteorology,
            [
                "wind_speed",
                "wind_direction",
                "wind_turbulence_horizontal",
                "wind_turbulence_vertical",
                "temperature",
                "pressure",
            ],
            percent_nan=5.0,
        )
    return meteorology


def check_field_values(data_object, field_list):
    """Helper function to check whether all the listed fields on a given object are not NaN or Inf.

    Args:
        data_object (Union[SensorGroup, MeteorologyGroup]): data object on which to check the fields.
        field_list (list): list of fields to check.

    """
    for data in data_object.values():
        for field in field_list:
            if (field != "time") and (getattr(data, field) is not None):
                assert np.all(np.logical_not(np.isnan(getattr(data, field))))
                assert np.all(np.logical_not(np.isinf(getattr(data, field))))


def test_initialize(sensor_mod, meteorology, time_bin_edges):
    """Test that the preprocessing class initialises successfully.

    Using the wrapper construction to test both a single Meteorology input object as well as a MeteorologyGroup.

    """
    wrapper_initialise(sensor_mod, meteorology, time_bin_edges)
    met_group = MeteorologyGroup()
    for key in sensor_mod.keys():
        temp_object = deepcopy(meteorology)
        temp_object.label = key
        met_group.add_object(temp_object)
    wrapper_initialise(sensor_mod, met_group, time_bin_edges)


def wrapper_initialise(sensor_mod_input, meteorology_input, time_bin_edges_input):
    """Tests that the preprocessing class initialises successfully, and that the attached attributes have the correct
    properties.

    Checks that:
        - the time bin edges are correctly stored on the Preprocessor object.
        - the same time stamps are assigned to the processed meteorology and sensor objects.
        - the wind directions are between 358 and 2 degrees after averaging.
        - the wind speeds are all between 1.9 and 2.1 m/s after averaging.
        - there are no NaNs or Infs in the fields of the processed object.

    """
    preprocess = Preprocessor(
        time_bin_edges=time_bin_edges_input, sensor_object=sensor_mod_input, met_object=meteorology_input
    )

    assert np.allclose(
        np.array(preprocess.time_bin_edges.to_numpy() - time_bin_edges_input.to_numpy(), dtype=float),
        np.zeros(preprocess.time_bin_edges.shape),
    )

    for sns, met in zip(preprocess.sensor_object.values(), preprocess.met_object.values()):
        assert np.allclose(np.array(sns.time - met.time, dtype=float), np.zeros(sns.time.shape))

    for met in preprocess.met_object.values():
        assert np.all(
            np.logical_or(
                np.logical_and(met.wind_direction >= 358.0, met.wind_direction <= 360.0),
                np.logical_and(met.wind_direction >= 0.0, met.wind_direction <= 2.0),
            )
        )
        assert np.all(np.logical_and(met.wind_speed >= 1.9, met.wind_speed <= 2.1))

    check_field_values(data_object=preprocess.sensor_object, field_list=preprocess.sensor_fields)
    check_field_values(data_object=preprocess.met_object, field_list=preprocess.met_fields)

    preprocess_limit_high = deepcopy(preprocess)
    preprocess_limit_low = deepcopy(preprocess)
    limit = 2.0
    preprocess_limit_low.filter_on_met(filter_variable=["wind_speed"], lower_limit=[limit])
    preprocess_limit_high.filter_on_met(filter_variable=["wind_speed"], upper_limit=[limit])

    for met in preprocess_limit_high.met_object.values():
        assert np.all(met.wind_speed <= limit)

    for met in preprocess_limit_low.met_object.values():
        assert np.all(met.wind_speed >= limit)


def test_block_data(sensor_mod, meteorology, time_bin_edges, block_times):
    """Test that the data blocking functionality returns expected results.

    Checks that:
        - the field values after blocking do not contain any NaNs or Infs.
        _ that empty SensorGroup and MeteorologyGroup objects are returned in the list elements for any time blocks
            which lie entirely outside the time range of the data.

    """
    preprocess = Preprocessor(time_bin_edges=time_bin_edges, sensor_object=sensor_mod, met_object=meteorology)

    with pytest.raises(TypeError):
        preprocess.block_data(block_times, data_object="bad_argument")

    sensor_list = preprocess.block_data(block_times, preprocess.sensor_object)
    met_list = preprocess.block_data(block_times, preprocess.met_object)

    for sns in sensor_list:
        check_field_values(data_object=sns, field_list=preprocess.sensor_fields)
    for met in met_list:
        check_field_values(data_object=met, field_list=preprocess.met_fields)

    min_time, max_time = get_time_lims(sensor_mod)
    for k in range(len(block_times) - 1):
        if ((block_times[k] < min_time) and (block_times[k + 1] < min_time)) or (
            (block_times[k] > max_time) and (block_times[k + 1] > max_time)
        ):
            assert not list(sensor_list[k].keys())
            assert not list(met_list[k].keys())
