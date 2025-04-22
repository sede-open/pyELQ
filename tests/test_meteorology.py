# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for meteorology superclass.

This module provides tests for the meteorology superclass in pyELQ

"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pyelq.coordinate_system import LLA
from pyelq.meteorology import Meteorology, MeteorologyGroup


@pytest.mark.parametrize(
    "u_component, v_component, truth",
    [
        (0, 1, 1),
        (np.sqrt(0.5), np.sqrt(0.5), 1),
        (1, 0, 1),
        (np.sqrt(0.5), -np.sqrt(0.5), 1),
        (0, -1, 1),
        (-np.sqrt(0.5), -np.sqrt(0.5), 1),
        (-1, 0, 1),
        (-np.sqrt(0.5), np.sqrt(0.5), 1),
    ],
)
def test_wind_speed(u_component, v_component, truth):
    """Basic test to check wind speed calculation from u and v.

    Args:
        u_component (float): u component
        v_component (float): v component
        truth (float): true wind speed

    """

    met_object = Meteorology()
    met_object.u_component = u_component
    met_object.v_component = v_component
    met_object.calculate_wind_speed_from_uv()

    assert met_object.wind_speed == truth


@pytest.mark.parametrize(
    "u_component, v_component, truth",
    [
        (0, 1, 180),
        (np.sqrt(0.5), np.sqrt(0.5), 225),
        (1, 0, 270),
        (np.sqrt(0.5), -np.sqrt(0.5), 315),
        (0, -1, 0),
        (-np.sqrt(0.5), -np.sqrt(0.5), 45),
        (-1, 0, 90),
        (-np.sqrt(0.5), np.sqrt(0.5), 135),
    ],
)
def test_wind_direction(u_component, v_component, truth):
    """Basic test to check wind direction (from) calculation from u and v.

    Args:
        u_component (float): u component
        v_component (float): v component
        truth (float): true wind direction (from)

    """

    met_object = Meteorology()
    met_object.u_component = u_component
    met_object.v_component = v_component
    met_object.calculate_wind_direction_from_uv()

    assert met_object.wind_direction == truth


def test_nof_observations():
    """Test if nof_observation property works as expected."""
    n_samples = np.random.randint(1, 100)
    array = np.random.random((n_samples, 2))

    lla_object = LLA()
    lla_object.from_array(array)

    met_object = Meteorology()
    assert met_object.nof_observations == 0

    met_object.location = lla_object
    assert met_object.nof_observations == n_samples


@pytest.mark.parametrize(
    "wind_speed, wind_direction, u_component, v_component",
    [
        (2, 0, 0, -2),
        (1, 45, -np.sqrt(0.5), -np.sqrt(0.5)),
        (1, 90, -1, 0),
        (1, 135, -np.sqrt(0.5), np.sqrt(0.5)),
        (1, 180, 0, 1),
        (1, 225, np.sqrt(0.5), np.sqrt(0.5)),
        (1, 270, 1, 0),
        (1, 315, np.sqrt(0.5), -np.sqrt(0.5)),
    ],
)
def test_calculate_uv_from_wind_speed_direction(wind_speed, wind_direction, u_component, v_component):
    """Basic test to check the calculation of u and v components from wind speed and direction.

    Args:
        wind_speed (float): Example wind speed.
        wind_direction (float): Example wind direction.
        u_component (float): True u value.
        v_component (float): True v value.

    """

    met_object = Meteorology()
    met_object.wind_speed = wind_speed
    met_object.wind_direction = wind_direction
    met_object.calculate_uv_from_wind_speed_direction()

    assert np.isclose(met_object.u_component, u_component)
    assert np.isclose(met_object.v_component, v_component)


@pytest.mark.parametrize(
    "wind_speed, wind_direction, u_component, v_component",
    [
        (2, 0, 0, -2),
        (1, 45, -np.sqrt(0.5), -np.sqrt(0.5)),
        (1, 90, -1, 0),
        (1, 135, -np.sqrt(0.5), np.sqrt(0.5)),
        (1, 180, 0, 1),
        (1, 225, np.sqrt(0.5), np.sqrt(0.5)),
        (1, 270, 1, 0),
        (1, 315, np.sqrt(0.5), -np.sqrt(0.5)),
    ],
)
def test_consistency_of_functions(wind_speed, wind_direction, u_component, v_component):
    """Basic test to check the consistency between the conversion functions of u and v components to/from wind speed and
    direction.

    Args:
        wind_speed (float): Example wind speed.
        wind_direction (float): Example wind direction.
        u_component (float): True u value.
        v_component (float): True v value.

    """

    met_object = Meteorology()
    met_object.wind_speed = wind_speed
    met_object.wind_direction = wind_direction
    met_object.calculate_uv_from_wind_speed_direction()
    met_object.calculate_wind_speed_from_uv()
    met_object.calculate_wind_direction_from_uv()

    assert np.isclose(met_object.wind_speed, wind_speed)
    assert np.isclose(met_object.wind_direction, wind_direction)

    met_object = Meteorology()
    met_object.u_component = u_component
    met_object.v_component = v_component
    met_object.calculate_wind_speed_from_uv()
    met_object.calculate_wind_direction_from_uv()

    met_object.calculate_uv_from_wind_speed_direction()

    assert np.isclose(met_object.u_component, u_component)
    assert np.isclose(met_object.v_component, v_component)


def test_meteorology_group():
    """Basic function to test MeteorologyGroup functionality, seeing if we can add objects to the group and check if the
    group returns the right number of objects as well as if the uv calculation works for a group object."""
    object_1 = Meteorology()
    object_1.label = "One"
    object_1.wind_speed = np.array([1, 1, 1, 1])
    object_1.wind_direction = np.array([0, 90, 180, 270])
    object_2 = Meteorology()
    object_2.label = "Two"
    object_2.wind_speed = np.array([1, 1, 1, 1])
    object_2.wind_direction = np.array([0, 90, 180, 270])

    group_object = MeteorologyGroup()
    group_object.add_object(object_1)
    group_object.add_object(object_2)
    assert group_object.nof_objects == 2
    group_object.calculate_uv_from_wind_speed_direction()
    for _, temp_object in group_object.items():
        assert np.allclose(temp_object.u_component, np.array([0, -1, 0, 1]))
        assert np.allclose(temp_object.v_component, np.array([-1, 0, 1, 0]))


def test_calculate_wind_turbulence_horizontal_deg():
    """Checks that the wind turbulence values are calculated correctly.

    To verify horizontal wind turbulence calculations, we define winds as draws from a normal distribution. We then check that the mean of the
    calculated turbulence values is within 3 standard deviations of the true value.

    """

    met = Meteorology()
    met.time = pd.array(
        np.array([dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)]).astype("datetime64[ns]"),
        dtype="datetime64[ns]",
    )
    met.wind_direction = np.linspace(0, 360, met.time.shape[0])

    sigma = 3

    met.time = pd.array(
        pd.date_range(dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 2), freq="5s"), dtype="datetime64[ns]"
    )
    met.wind_direction = np.random.normal(180, sigma, met.time.shape[0])
    met.calculate_wind_turbulence_horizontal_deg(window="300s")

    tolerance = 3 * np.std(met.wind_turbulence_horizontal_deg)
    mean_turbulence = np.mean(met.wind_turbulence_horizontal_deg)
    assert (mean_turbulence - tolerance) < sigma < (mean_turbulence + tolerance)
