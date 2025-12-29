# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for sensor superclass.

This module provides tests for the sensor superclass in pyELQ

"""

import numpy as np
import pandas as pd
import pytest

from pyelq.coordinate_system import LLA
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


@pytest.mark.parametrize(
    "source_on",
    [
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0]),
        np.ones(10, dtype=bool),
    ],
    ids=["multiple_sections", "single_section"],
)
def test_subset_sensor(source_on):
    """Test subset_sensor method for correct behavior when splitting sensor observations based on source_on.
    The subset_sensor method is designed to return a new `Sensor` object containing only the observations corresponding
    to a specified section index. Sections are defined by unique values in the `source_on` attribute. The case 
    "multiple_sections" represents a situation where the source is turned on and off multiple times (0 values in 
    `source_on` indicate off periods and positive integers indicate different on periods). The case "single_section"
    represents a situation where the source is continuously on.

    For each case, the test verifies that:
    - The number of observations from the output subset_sensor matches the expected count for the specified section.
    - The size of the source_on array in the subset_sensor matches the expected count for the specified section.
    - If the subset_sensor contains observations, all source_on values in the subset_sensor match those in the original
    sensor for the specified section.

    """
    sensor = Sensor()
    sensor.label = "sensor_0"
    sensor.time = pd.array(pd.date_range("2024-01-01", periods=len(source_on)), dtype="datetime64[ns]")
    sensor.concentration = np.random.rand(len(source_on), 1)
    sensor.location = LLA(
        latitude=np.ones((len(source_on), 1)) * 10,
        longitude=np.ones((len(source_on), 1)) * 40,
        altitude=np.ones((len(source_on), 1)) * 0,
    )
    sensor.source_on = source_on

    number_of_sections = max(sensor.source_on) + 1
    for section in range(1, number_of_sections + 1):
        subset_sensor = sensor.subset_sensor(section_index=section)

        assert subset_sensor.nof_observations == np.sum(sensor.source_on == section)
        assert subset_sensor.source_on.size == np.sum(sensor.source_on == section)
        if subset_sensor.nof_observations > 0:
            assert np.all(subset_sensor.source_on == sensor.source_on[sensor.source_on == section])
