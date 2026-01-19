# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Meteorology windfield module.

Version of the meteorology class that deals with spatial wind fields and can calculate the wind field around
cylindrical obstacles.

"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.site_layout import SiteLayout
from pyelq.meteorology.meteorology import Meteorology


@dataclass
class MeteorologyWindfield(Meteorology):
    """Represents a spatially resolved wind field based on meteorological measurements and the presence of obstacles.

    This class extends the base `Meteorology` class by providing methods to compute the local wind vector (u and v
    components) at every grid point, factoring in obstacle perturbations using an analytical method. It accounts for
    spatial rotation to align with the instantaneous wind direction at each time step.

    Attributes:
        static_wind_field (Meteorology): The static wind field used for calculations.
        site_layout (SiteLayout): The layout of the site, including cylinder coordinates and radii.

    """

    static_wind_field: Meteorology
    site_layout: Optional[Union[SiteLayout, None]] = None

    def calculate_spatial_wind_field(self, grid_coordinates, time_index: int = None):
        """Calculates the spatial wind field over a grid considering obstacles.

        Computes the full spatial wind field over a grid considering both ambient meteorological conditions and local
        distortions due to obstacles.

        The method:
        - Rotates grid coordinates into the wind-aligned frame based on mathematical wind direction.
        - Calculates the distorted wind field due to the presence of cylindrical obstacles.
        - Rotates the resulting local wind field back into the original frame.
        - Updates the object's `u_component` and `v_component` accordingly.
        - If w_component is present in the static wind field, it is broadcasted to match the grid points.
        - If no site layout is provided, the wind field remains undisturbed and is simply broadcasted across the grid.

        The method updates the following properties in place:
        - u_component np.ndarray: (n_grid x n_time) The x-component of the wind field at the grid points.
        - v_component np.ndarray: (n_grid x n_time) The y-component of the wind field at the grid points.
        - w_component np.ndarray: (n_grid x n_time) The y-component of the wind field at the grid points.

        Args:
            grid_coordinates (ENU): The coordinates of the grid points.
            time_index (int): The time index for the meteorological data.

        """
        if time_index is not None:
            u = self.static_wind_field.u_component.reshape(-1, 1)[time_index]
            v = self.static_wind_field.v_component.reshape(-1, 1)[time_index]
            if self.static_wind_field.w_component is not None:
                self.w_component = np.broadcast_to(
                    self.static_wind_field.w_component[time_index].T,
                    (grid_coordinates.nof_observations, u.shape[0]),
                )
        else:
            u = self.static_wind_field.u_component.reshape(-1, 1)
            v = self.static_wind_field.v_component.reshape(-1, 1)
            if self.static_wind_field.w_component is not None:
                self.w_component = np.broadcast_to(
                    self.static_wind_field.w_component.T,
                    (grid_coordinates.nof_observations, self.static_wind_field.w_component.shape[0]),
                )

        if self.site_layout is None:
            self.u_component = np.broadcast_to(u.T, (grid_coordinates.nof_observations, u.shape[0]))
            self.v_component = np.broadcast_to(v.T, (grid_coordinates.nof_observations, v.shape[0]))
            return
        mathematical_wind_direction = np.arctan2(v, u).flatten()
        (rotation_matrix, rotated_grid, rotated_cylinders) = self._rotate_coordinates(
            grid_coordinates, mathematical_wind_direction
        )
        u_rot, v_rot = self._calculate_wind_field_cardinal(
            u=u,
            v=v,
            grid_coordinates=grid_coordinates,
            rotated_grid=rotated_grid,
            rotated_cylinders=rotated_cylinders,
        )
        u_stacked = np.stack((u_rot, v_rot), axis=2)
        inverse_rot = np.transpose(rotation_matrix, axes=(1, 0, 2))
        rotated_wind = np.einsum("ijt,ntj-> nti", inverse_rot, u_stacked)
        self.u_component = rotated_wind[:, :, 0]
        self.v_component = rotated_wind[:, :, 1]

    def _rotate_coordinates(
        self, grid_coordinates: ENU, wind_direction: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotates the x, y coordinates based on the wind direction.

        Args:
            grid_coordinates (ENU): The coordinates to be rotated.
            wind_direction (np.array): The wind direction in radians.

        Returns:
            tuple: The rotated coordinates for the grid and cylinders.

        """
        rotation_matrix = np.array(
            [
                [np.cos(wind_direction), np.sin(wind_direction)],
                [-np.sin(wind_direction), np.cos(wind_direction)],
            ]
        )
        rotated_grid = np.einsum("ijt,nj->nit", rotation_matrix, grid_coordinates.to_array(dim=2))
        rotated_cylinders = np.einsum(
            "ijt,nj->nit", rotation_matrix, self.site_layout.cylinder_coordinates.to_array(dim=2)
        )
        return (rotation_matrix, rotated_grid, rotated_cylinders)

    def _calculate_wind_field_cardinal(
        self,
        u: np.ndarray,
        v: np.ndarray,
        grid_coordinates: ENU,
        rotated_grid: np.ndarray,
        rotated_cylinders: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the distorted wind field components (u_x, u_y) in the wind-aligned (cardinal) frame.

        The method:
        - Determines whether each grid point is influenced by nearby cylinders based on distance and cylinder radius.
        - If no obstacles are relevant at the evaluation height, the wind field remains undisturbed.
        - If obstacles are present, modifies the wind field using an analytical perturbation formula based on
            potential flow theory.
        - Wind inside obstacles is set to zero.

        If no height information is provided (e.g. in the 2-dimensional solver case), the function assumes that
        all cylinders and input points are at the same height, and applies the mask accordingly.

        Args:
            u (np.ndarray n_time x 1): The x-component of the wind vector.
            v (np.ndarray  n_time x 1): The y-component of the wind vector.
            grid_coordinates (ENU): location object containing information about the finite volume solve grid points.
            rotated_grid (np.ndarray n_grid x 2 x n_time): The grid coordinates where the wind field is to be calculated
            in the wind-aligned frame.
            rotated_cylinders (np.ndarray n_cylinders x 2 x n_time): The coordinates of the cylinders in the
            wind-aligned frame.

        Returns:
            u_rot (np.ndarray): The x-component of the wind field at the grid points.
            v_rot (np.ndarray): The y-component of the wind field at the grid points.

        """
        diff = rotated_grid[:, np.newaxis, :, :] - rotated_cylinders[np.newaxis, :, :, :]
        radial_distance = np.linalg.norm(diff, axis=2)
        x_diff = diff[:, :, 0, :]
        y_diff = diff[:, :, 1, :]
        radius_squared = self.site_layout.cylinder_radius.T**2
        radial_distance_sq = radial_distance**2
        radial_distance_quad = radial_distance_sq**2
        radius_sq_over_r4 = radius_squared[:, :, np.newaxis] / radial_distance_quad

        if grid_coordinates.up is None:
            sum_term_x = np.einsum("nct, nct->nt", radius_sq_over_r4, (y_diff**2 - x_diff**2))
            sum_term_y = np.einsum("nct, nct->nt", radius_sq_over_r4, (y_diff * x_diff))
        else:
            height_mask = grid_coordinates.up <= self.site_layout.cylinder_coordinates.up.T
            height_mask = height_mask.reshape(grid_coordinates.nof_observations, self.site_layout.nof_cylinders)
            sum_term_x = np.einsum("nc, nct, nct->nt", height_mask, radius_sq_over_r4, (y_diff**2 - x_diff**2))
            sum_term_y = np.einsum("nc, nct, nct->nt", height_mask, radius_sq_over_r4, (y_diff * x_diff))
        wind_speed = np.sqrt(u**2 + v**2).T
        u_rot = wind_speed * (1 + sum_term_x)
        v_rot = -2 * wind_speed * sum_term_y
        u_rot[self.site_layout.id_obstacles.flatten(), :] = 0
        v_rot[self.site_layout.id_obstacles.flatten(), :] = 0
        return u_rot, v_rot
