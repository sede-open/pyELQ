# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for source map class.

This module provides tests for the source map class in pyELQ.

"""
import numpy as np
import pytest

from pyelq import coordinate_system
from pyelq.coordinate_system import ENU, LLA
from pyelq.sensor.sensor import Sensor
from pyelq.source_map import SourceMap


def test_n_sources():
    """Test if the nof_sources property works as expected."""
    n_samples = np.random.randint(1, 100)
    array = np.random.random((n_samples, 2))

    lla_object = LLA()
    lla_object.from_array(array)

    source_object = SourceMap()
    assert source_object.nof_sources == 0

    source_object.location = lla_object
    assert source_object.nof_sources == n_samples


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube", "grid", "grid_sphere", "error"])
@pytest.mark.parametrize("dim", [2, 3])
def test_generate_sources(sourcemap_type, dim):
    """Test the generate_sources method.

    Checks if the not implement error gets raised. Checks for correct number of sources generated. Checks if all
    sources are within the specified limits

    Args:
        sourcemap_type (str): Type of source map to generate (1 or 2)
        dim (int): Dimension of each source location (2 or 3)

    """
    source_object = SourceMap()
    enu_object = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    sourcemap_limits = np.array([[-100, 100], [-100, 100], [-100, 100]])
    sourcemap_limits = sourcemap_limits[:dim, :]

    if sourcemap_type in ["central", "hypercube", "grid", "grid_sphere"]:
        if sourcemap_type in ["central", "hypercube"]:
            random_integer = np.random.randint(1, 100)
            source_object.generate_sources(
                coordinate_object=enu_object,
                sourcemap_limits=sourcemap_limits,
                sourcemap_type=sourcemap_type,
                nof_sources=random_integer,
            )
            if sourcemap_type == "central":
                assert source_object.nof_sources == 1
            else:
                assert source_object.nof_sources == random_integer
        elif sourcemap_type in ["grid", "grid_sphere"]:
            random_shape = np.random.randint(1, 100, size=dim)
            source_object.generate_sources(
                coordinate_object=enu_object,
                sourcemap_limits=sourcemap_limits,
                sourcemap_type=sourcemap_type,
                grid_shape=random_shape,
            )
            assert source_object.nof_sources == random_shape.prod()

        array_object = source_object.location.to_array()
        for idx in range(dim):
            assert np.all(array_object[:, idx] >= sourcemap_limits[idx, 0])
            assert np.all(array_object[:, idx] <= sourcemap_limits[idx, 1])
    else:
        with pytest.raises(NotImplementedError):
            source_object.generate_sources(
                coordinate_object=enu_object, sourcemap_limits=sourcemap_limits, sourcemap_type=sourcemap_type
            )


@pytest.mark.parametrize("source_coordinate_system", ["LLA", "ENU", "ECEF"])
@pytest.mark.parametrize("sensor_coordinate_system", ["LLA", "ENU", "ECEF"])
def test_calculate_inclusion_idx(source_coordinate_system, sensor_coordinate_system):
    """Test the calculate_inclusion_idx method.

    Defines a source map of 2 sources source far apart and observations close to one of those sources and far away,
    checks if inclusion idx are correct and inclusion_n_obs are correct.

    Calculate the source and sensor locations first in an ENU system, then convert to the desired coordinate system and
    then do the actual calculation.

    """
    source_object = SourceMap()
    if source_coordinate_system == "ENU":
        coordinate_object = getattr(coordinate_system, source_coordinate_system)(
            ref_latitude=0, ref_longitude=0, ref_altitude=0
        )
    else:
        coordinate_object = getattr(coordinate_system, source_coordinate_system)()
    enu_coordinate = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    enu_coordinate.from_array(array=np.array([[0, 0, 0], [80, 80, 80]]))
    source_object.location = enu_coordinate.to_object_type(coordinate_object)

    sensor_object = Sensor()
    if sensor_coordinate_system == "ENU":
        coordinate_object = getattr(coordinate_system, sensor_coordinate_system)(
            ref_latitude=0, ref_longitude=0, ref_altitude=0
        )
    else:
        coordinate_object = getattr(coordinate_system, sensor_coordinate_system)()

    points_inside = np.random.randint(1, 100)
    inside_idx = list(range(points_inside))

    points_outside = np.random.randint(1, 100)
    outside_idx = list(range(points_outside))
    outside_idx = [value + points_inside for value in outside_idx]

    inside_locations = np.random.normal(0, 0.001, (points_inside, 3))
    outside_locations = np.random.normal(80, 0.001, (points_outside, 3))
    array = np.concatenate((inside_locations, outside_locations), axis=0)

    enu_coordinate = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    enu_coordinate.from_array(array)
    sensor_object.location = enu_coordinate.to_object_type(coordinate_object)
    source_object.calculate_inclusion_idx(sensor_object=sensor_object, inclusion_radius=100)

    assert np.all(source_object.inclusion_n_obs == np.array([points_inside, points_outside]))
    assert np.all(source_object.inclusion_idx[0] == inside_idx)
    assert np.all(source_object.inclusion_idx[1] == outside_idx)
