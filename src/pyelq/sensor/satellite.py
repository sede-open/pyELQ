# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Satellite module.

Subclass of Sensor. Mainly used to accommodate satellite sensor TROPOMI. See:
http://www.tropomi.eu/data-products/methane
: http: //www.tropomi.eu/data-products/methane and http://www.tropomi.eu/data-products/nitrogen-dioxide

"""

from dataclasses import dataclass, field

import numpy as np

from pyelq.sensor.sensor import Sensor


@dataclass
class Satellite(Sensor):
    """Defines Satellite sensor class.

    Attributes:
        qa_value (np.ndarray, optional): Array containing quality values associated with the observations.
        precision (np.ndarray, optional): Array containing precision values associated with the observations.
        precision_kernel (np.ndarray, optional): Array containing precision kernel values associated with the
            observations.
        ground_pixel (np.ndarray, optional): Array containing ground pixels values associated with the observations.
            Ground pixels are indicating the dimension perpendicular to the flight direction.
        scanline (np.ndarray, optional): Array containing scanline values associated with the observations.
            Scanlines are indicating the dimension in the direction of flight.
        orbit (np.ndarray, optional): Array containing orbit values associated with the observations.
        pixel_bounds (np.ndarray, optional): Array containing Polygon features which define the pixel bounds.

    """

    qa_value: np.ndarray = field(init=False)
    precision: np.ndarray = field(init=False)
    precision_kernel: np.ndarray = field(init=False)
    ground_pixel: np.ndarray = field(init=False)
    scanline: np.ndarray = field(init=False)
    orbit: np.ndarray = field(init=False, default=None)
    pixel_bounds: np.ndarray = field(init=False)

    def get_orbits(self) -> np.ndarray:
        """Gets the unique orbits which are present in the data.

        Raises:
            ValueError: When orbits attribute is None

        Returns:
            np.ndarray: Unique orbits present in the data.

        """
        if self.orbit is None:
            raise ValueError("Orbits attribute is None")
        return np.unique(self.orbit)
