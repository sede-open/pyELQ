# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Site layout module.

This module defines the SiteLayout class, which represents the layout of a site, giving the option to include obstacles
(e.g. buildings, tanks, equipment) as cylinders obstructing the flow field.

"""

from dataclasses import dataclass, field

import numpy as np
from scipy import spatial

from pyelq.coordinate_system import ENU


@dataclass
class SiteLayout:
    """Class for site layout defining cylindrical obstacles in the environment.

    These are used with MeteorologyWindfield to calculate the wind field at each grid point with a potential flow
    around the cylindrical obstacles.

    Attributes:
        cylinders_coordinate (ENU): The coordinates of the cylindrical obstacles in the site layout.
        cylinders_radius (np.ndarray): The radius of the cylindrical obstacles in the site layout.
        id_obstacles (np.ndarray): Boolean array indicating which grid points are within obstacle regions.
        id_obstacles_index (np.ndarray): The indices of the grid points that are within obstacle regions.

    Methods:
        find_index_obstacles(grid_coordinates: ENU) -> None:
            Find the indices of the grid_coordinates that are within the radius of the obstacles.

    """

    cylinders_coordinate: ENU = None
    cylinders_radius: np.ndarray = None

    id_obstacles: np.ndarray = field(init=False)
    id_obstacles_index: np.ndarray = field(init=False)

    @property
    def nof_cylinders(self) -> int:
        """Returns the number of cylinders in the site layout."""
        if self.cylinders_coordinate is None:
            return 0
        return self.cylinders_coordinate.nof_observations

    def find_index_obstacles(self, grid_coordinates: ENU):
        """Find the indices of the grid_coordinates that are within the radius of the obstacles.

        This method uses a KDTree to efficiently find the indices of the grid points that are within the radius of the
        cylindrical obstacles. It also checks the height of the grid points against the height of the obstacles.
        If the height of the grid point is greater than the height of the obstacle, it is not considered an obstacle.

        Args:
            grid_coordinates (ENU): The coordinates of the grid points to check.

        """
        if self.cylinders_coordinate is None or self.cylinders_coordinate.nof_observations == 0:
            self.id_obstacles = np.zeros((grid_coordinates.nof_observations, 1), dtype=bool)
            return
        grid_coordinates_array = grid_coordinates.to_array(dim=2)
        tree = spatial.KDTree(grid_coordinates_array)
        indices = tree.query_ball_point(x=self.cylinders_coordinate.to_array(dim=2), r=self.cylinders_radius.flatten())

        if grid_coordinates.up is not None:
            for i, height in enumerate(self.cylinders_coordinate.up):
                indices[i] = np.array(indices[i])
                if len(indices[i]) > 0:
                    indices[i] = indices[i][grid_coordinates.up[indices[i]].flatten() <= height]
        indices_conc = np.unique(np.concatenate(indices, axis=0)).astype(int)
        self.id_obstacles_index = indices_conc
        self.id_obstacles = np.zeros((grid_coordinates.nof_observations, 1), dtype=bool)
        self.id_obstacles[indices_conc, 0] = True
