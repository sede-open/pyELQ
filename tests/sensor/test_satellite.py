# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for satellite sensor class.

This module provides tests for the satellite sensor class in pyELQ.

"""

import numpy as np
import pytest

from pyelq.sensor.satellite import Satellite


def test_orbits():
    """Basic test to check is Satellite can be instantiated and if it correctly finds the unique orbits."""
    sensor = Satellite()
    assert isinstance(sensor, Satellite)

    with pytest.raises(ValueError):
        sensor.get_orbits()

    orbits = np.array([1, 2, 3, 4, 5])
    rng = np.random.default_rng(42)
    random_integer = rng.integers(low=1, high=10 + 1)
    random_repeat = np.repeat(orbits, random_integer)
    sensor.orbit = random_repeat
    result = sensor.get_orbits()
    assert np.all(result == orbits)
