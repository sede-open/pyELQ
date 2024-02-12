# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for beam sensor class.

This module provides tests for the beam sensor class in pyELQ.

"""
import numpy as np

from pyelq.coordinate_system import ENU
from pyelq.sensor.beam import Beam


def test_midpoint():
    """Basic test to check if midpoint is working correctly.

    Setting the absolute tolerance to 1e-6 because of rounding errors in the conversion

    """

    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    location.east = np.array([-1, 1])
    location.north = np.array([-1, 1])
    location.up = np.array([-1, 1])
    midpoint = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    midpoint.from_array(np.array([[0, 0, 0]]))

    sensor = Beam()
    sensor.location = location
    test_enu = sensor.midpoint
    assert np.allclose(test_enu, midpoint.to_array())
    sensor.location = location.to_lla()
    test_lla = sensor.midpoint
    assert np.allclose(test_lla, midpoint.to_lla().to_array(), atol=1e-06)
    sensor.location = location.to_ecef()
    test_ecef = sensor.midpoint
    assert np.allclose(test_ecef, midpoint.to_ecef().to_array())


def test_make_beam_knots():
    """Basic test to check if make_beam_knots is working correctly.

    Checking all beam locations are inside bounding box and number of points correct. As well ass if they are linearly
    spaced

    """
    sensor = Beam()
    sensor.location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    sensor.location.east = np.array([-1, 1])
    sensor.location.north = np.array([-2, 0])
    sensor.location.up = np.array([0, 2])
    beam_knot_array = sensor.make_beam_knots(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    assert np.all(beam_knot_array[:, 0] >= sensor.location.east[0])
    assert np.all(beam_knot_array[:, 0] <= sensor.location.east[1])
    assert np.all(beam_knot_array[:, 1] >= sensor.location.north[0])
    assert np.all(beam_knot_array[:, 1] <= sensor.location.north[1])
    assert np.all(beam_knot_array[:, 2] >= sensor.location.up[0])
    assert np.all(beam_knot_array[:, 2] <= sensor.location.up[1])
    assert beam_knot_array.shape[0] == sensor.n_beam_knots
    assert np.unique(np.round(np.diff(beam_knot_array, axis=0), 10), axis=0).shape[0] == 1
