# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Beam module.

Subclass of Sensor. Used for beam sensors

"""

from dataclasses import dataclass

import numpy as np

from pyelq.sensor.sensor import Sensor


@dataclass
class Beam(Sensor):
    """Defines Beam sensor class.

    Location attribute from superclass is assumed to be a Coordinate class object containing 2 locations, the first of
    the sensor and the second of the retro.

    Attributes:
        n_beam_knots (int, optional): Number of beam knots to evaluate along a single beam

    """

    n_beam_knots: int = 50

    @property
    def midpoint(self) -> np.ndarray:
        """np.ndarray: Midpoint of the beam."""
        return np.mean(self.location.to_array(), axis=0)

    def make_beam_knots(self, ref_latitude, ref_longitude, ref_altitude=0) -> np.ndarray:
        """Create beam knot locations.

        Creates beam knot locations based on location attribute and n_beam_knot attribute.
        Results in an array of beam knot locations of shape [n_beam_knots x 3]. Have to provide a reference point in
        order to create the beam knots in a local frame, spaced in meters

        Args:
            ref_latitude (float): Reference latitude in degrees
            ref_longitude (float): Reference longitude in degrees
            ref_altitude (float, optional): Reference altitude in meters

        """
        temp_location = self.location.to_enu(
            ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude
        ).to_array()
        beam_knot_array = np.linspace(temp_location[0, :], temp_location[1, :], num=self.n_beam_knots, endpoint=True)
        return beam_knot_array
