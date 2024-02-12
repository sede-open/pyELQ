# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for sensor superclass.

This module provides tests for the sensor superclass in pyELQ

"""

import numpy as np
import pytest

from pyelq.sensor.sensor import Sensor


@pytest.mark.parametrize("nof_observations", [0, 1, 10])
def test_nof_observables(nof_observations: int):
    """Basic test to check Sensor class method.

    Args:
        nof_observations (int): Number observations

    """
    sensor = Sensor()
    if nof_observations > 0:
        sensor.concentration = np.random.rand(nof_observations, 1)

    assert sensor.nof_observations == nof_observations
