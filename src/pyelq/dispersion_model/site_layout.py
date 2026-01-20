# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Site layout module.

This module defines the SiteLayout class, which represents the layout of a site, giving the option to include obstacles
(e.g. buildings, tanks, equipment) as cylinders obstructing the flow field.

"""

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import spatial

from pyelq.coordinate_system import ENU


@dataclass
class SiteLayout:
    """Class for site layout defining cylindrical obstacles in the environment.

    These are used with MeteorologyWindfield to calculate the wind field at each grid point with a potential flow
    around the cylindrical obstacles.

    Attributes:
        cylinder_coordinates (Union[ENU, None]): The coordinates of the cylindrical obstacles in the site layout.
        cylinder_radius (np.ndarray): The radius of the cylindrical obstacles in the site layout.
        id_obstacles (np.ndarray): Boolean array indicating which grid points are within obstacle regions.
        id_obstacles_index (np.ndarray): The indices of the grid points that are within obstacle regions.

    """

    cylinder_coordinates: Union[ENU, None]
    cylinder_radius: np.ndarray

    id_obstacles: np.ndarray = field(init=False)
    id_obstacles_index: np.ndarray = field(init=False)

    @property
    def nof_cylinders(self) -> int:
        """Returns the number of cylinders in the site layout."""
        if self.cylinder_coordinates is None:
            return 0
        return self.cylinder_coordinates.nof_observations

    def find_index_obstacles(self, grid_coordinates: ENU):
        """Find the indices of the grid_coordinates that are within the radius of the obstacles.

        This method uses a KDTree to efficiently find the indices of the grid points that are within the radius of the
        cylindrical obstacles. It also checks the height of the grid points against the height of the obstacles.
        If the height of the grid point is greater than the height of the obstacle, it is not considered an obstacle.

        Args:
            grid_coordinates (ENU): The coordinates of the grid points to check.

        """
        if self.cylinder_coordinates is None or self.cylinder_coordinates.nof_observations == 0:
            self.id_obstacles = np.zeros((grid_coordinates.nof_observations, 1), dtype=bool)
            return

        self.check_reference_coordinates(grid_coordinates)

        grid_coordinates_array = grid_coordinates.to_array(dim=2)
        tree = spatial.KDTree(grid_coordinates_array)
        indices = tree.query_ball_point(x=self.cylinder_coordinates.to_array(dim=2), r=self.cylinder_radius.flatten())

        if grid_coordinates.up is not None:

            for i, height in enumerate(self.cylinder_coordinates.up):
                indices[i] = np.array(indices[i])
                if len(indices[i]) > 0:
                    indices[i] = indices[i][grid_coordinates.up[indices[i]].flatten() <= height]
        indices_conc = np.unique(np.concatenate(indices, axis=0)).astype(int)
        self.id_obstacles_index = indices_conc
        self.id_obstacles = np.zeros((grid_coordinates.nof_observations, 1), dtype=bool)
        self.id_obstacles[[indices_conc]] = True

    def check_reference_coordinates(self, grid_coordinates: ENU):
        """Check if the reference coordinates of the grid and cylinder coordinates match.

        Args:
            grid_coordinates (ENU): The coordinates of the grid points to check.

        Raises:
            ValueError: If the reference coordinates do not match.
        """
        if grid_coordinates.ref_altitude != self.cylinder_coordinates.ref_altitude:
            raise ValueError("Grid coordinates and cylinder coordinates must have the same reference altitude.")
        if grid_coordinates.ref_longitude != self.cylinder_coordinates.ref_longitude:
            raise ValueError("Grid coordinates and cylinder coordinates must have the same reference longitude.")
        if grid_coordinates.ref_latitude != self.cylinder_coordinates.ref_latitude:
            raise ValueError("Grid coordinates and cylinder coordinates must have the same reference latitude.")
