# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Site layout module.

This module defines the SiteLayout class, which represents the layout of a site, giving the option to include obstacles
(e.g. buildings, tanks, equipment) as cylinders obstructing the flow field.

"""

from dataclasses import dataclass, field
from typing import Union, Tuple

import numpy as np
from scipy import spatial

from pyelq.coordinate_system import ENU


@dataclass
class SiteLayout:
    """Class for site layout defining cylindrical obstacles in the environment.

    These are used with MeteorologyWindfield to calculate the wind field at each grid point with a potential flow
    around the cylindrical obstacles.

    Attributes:
        cylinder_coordinates (Union[ENU, None]): The coordinates of the cylindrical obstacles in the site layout. The
            east, north represent the the center of the cylinder and the up coordinate represents the cylinder height.
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
        """Int: Returns the number of cylinders in the site layout."""
        if self.cylinder_coordinates is None:
            return 0
        return self.cylinder_coordinates.nof_observations

    def find_index_obstacles(self, coordinates: ENU) -> Tuple[np.ndarray, np.ndarray]:
        """Find the indices of the coordinates that are within the radius of the obstacles.

        This method uses a KDTree to efficiently find the indices of the points that are within the radius of the
        cylindrical obstacles. It also checks the height of the points against the height of the obstacles.
        If the height of the point is greater than the height of the obstacle, it is not considered an obstacle.

        Args:
            coordinates (ENU): The coordinates of the points to check.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the indices of the points that are within the obstacles
            and a boolean array indicating which points are within obstacles.

        """
        if self.cylinder_coordinates is None or self.cylinder_coordinates.nof_observations == 0:
            id_obstacles = np.zeros((coordinates.nof_observations, 1), dtype=bool)
            return np.array([], dtype=int), id_obstacles

        self.check_reference_coordinates(coordinates)

        coordinates_array = coordinates.to_array(dim=2)
        tree = spatial.KDTree(coordinates_array)
        indices = tree.query_ball_point(x=self.cylinder_coordinates.to_array(dim=2), r=self.cylinder_radius.flatten())

        if coordinates.up is not None:

            for i, height in enumerate(self.cylinder_coordinates.up):
                indices[i] = np.array(indices[i])
                if len(indices[i]) > 0:
                    indices[i] = indices[i][coordinates.up[indices[i]].flatten() <= height]
        indices_conc = np.unique(np.concatenate(indices, axis=0)).astype(int)
        id_obstacles_index = indices_conc
        id_obstacles = np.zeros((coordinates.nof_observations, 1), dtype=bool)
        id_obstacles[[indices_conc]] = True
        return id_obstacles_index, id_obstacles
    
    def set_index_obstacles_grid(self, grid_coordinates: ENU):
        """Set the indices of the grid points that are within obstacle regions based on the grid coordinates.
        
        Args:
            grid_coordinates (ENU): The coordinates of the grid points to check.

        """
        self.id_obstacles_index, self.id_obstacles = self.find_index_obstacles(grid_coordinates)

    def check_reference_coordinates(self, coordinates: ENU):
        """Check if the reference coordinates of the coordinates and cylinder coordinates match.

        Args:
            coordinates (ENU): The coordinates of the points to check.

        Raises:
            ValueError: If the reference coordinates do not match.
        """
        if coordinates.ref_altitude != self.cylinder_coordinates.ref_altitude:
            raise ValueError("Coordinates and cylinder coordinates must have the same reference altitude.")
        if coordinates.ref_longitude != self.cylinder_coordinates.ref_longitude:
            raise ValueError("Coordinates and cylinder coordinates must have the same reference longitude.")
        if coordinates.ref_latitude != self.cylinder_coordinates.ref_latitude:
            raise ValueError("Coordinates and cylinder coordinates must have the same reference latitude.")
