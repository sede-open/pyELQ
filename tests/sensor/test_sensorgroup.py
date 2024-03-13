# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for SensorGroup superclass.

This module provides tests for the SensorGroup superclass in pyELQ

"""
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyelq.coordinate_system import LLA
from pyelq.sensor.sensor import Sensor, SensorGroup


def test_sensorgroup():
    """Tests to check all properties of the SensorGroup class have the correct output shapes."""
    nof_sensors = 3
    total_observations = 0
    group = SensorGroup()
    for idx in range(nof_sensors):
        sensor = Sensor()
        nof_observations = np.random.randint(1, 10)
        total_observations += nof_observations
        sensor.concentration = np.random.rand(nof_observations, 1)
        sensor.time = pd.array(pd.date_range(start="1/1/2022", periods=nof_observations), dtype="datetime64[ns]")
        sensor.location = LLA(
            latitude=0.01 * np.random.rand(), longitude=0.01 * np.random.rand(), altitude=0.01 * np.random.rand()
        )
        sensor.label = str(idx)
        group.add_sensor(sensor=sensor)

    assert group.nof_sensors == nof_sensors
    assert group.nof_observations == total_observations
    assert group.concentration.shape == (total_observations,)
    assert group.time.shape == (total_observations,)
    assert group.sensor_index.shape == (total_observations,)

    enu_location = group.location.to_enu()
    assert enu_location.east.shape == (nof_sensors,)
    assert enu_location.north.shape == (nof_sensors,)
    assert enu_location.up.shape == (nof_sensors,)


def test_plotting():
    """Tests to check if plotting methods provide a plotly figure with the correct amount of traces."""
    nof_sensors = 3
    total_observations = 0
    group = SensorGroup()
    for idx in range(nof_sensors):
        sensor = Sensor()
        nof_observations = np.random.randint(5, 10)
        total_observations += nof_observations
        sensor.concentration = np.random.rand(nof_observations, 1)
        sensor.time = pd.array(pd.date_range(start="1/1/2022", periods=nof_observations), dtype="datetime64[ns]")
        location = LLA()
        location.latitude = np.array(idx)
        location.longitude = np.array(idx)
        sensor.location = location
        sensor.label = str(idx)
        group.add_sensor(sensor=sensor)

    fig_1 = go.Figure()
    fig_1 = group.plot_timeseries(fig_1)
    assert isinstance(fig_1, go.Figure)
    assert len(fig_1.data) == group.nof_sensors
    # fig_1.show(renderer='browser')

    fig_2 = go.Figure()
    fig_2 = group.plot_sensor_location(fig_2)
    fig_2.update_layout(mapbox={"style": "open-street-map", "center": {"lon": 0, "lat": 0}, "zoom": 7})
    assert isinstance(fig_2, go.Figure)
    assert len(fig_2.data) == group.nof_sensors
    # fig_2.show(renderer='browser')


def test_source_on_attribute():
    """Simple test to check correct concatenation of source_on attribute of SensorGroup."""
    location = LLA()
    location.from_array(np.ones((5, 3)))
    sensor = Sensor()
    sensor.concentration = np.array([1, 2, 3, 4, 5])
    sensor.location = location
    sensor.source_on = np.array([True, True, False, False, False])
    sensor.label = "1"

    sensor_2 = deepcopy(sensor)
    sensor_2.source_on = None
    sensor_2.label = "2"

    sns_group = SensorGroup()
    sns_group.add_sensor(sensor)
    sns_group.add_sensor(sensor_2)

    correct_results = np.array([True, True, False, False, False, True, True, True, True, True])
    assert np.all(sns_group.source_on == correct_results)
