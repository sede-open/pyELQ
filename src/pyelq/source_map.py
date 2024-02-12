# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""SourceMap module.

The class for the source maps used in pyELQ

"""
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from pyelq.coordinate_system import Coordinate, make_latin_hypercube
from pyelq.sensor.sensor import Sensor


@dataclass
class SourceMap:
    """Defines SourceMap class.

    Attributes:
        location (Coordinate, optional): Coordinate object specifying the potential source locations
        prior_value (np.ndarray, optional): Array with prior values for each source
        inclusion_idx (np.ndarray, optional): Array of lists containing indices of the observations of a
            corresponding sensor_object which are within the inclusion_radius of that particular source
        inclusion_n_obs (list, optional): Array containing number of observations of a sensor_object within
            radius for each source

    """

    location: Coordinate = field(init=False, default=None)
    prior_value: np.ndarray = None
    inclusion_idx: np.ndarray = field(init=False, default=None)
    inclusion_n_obs: np.ndarray = field(init=False, default=None)

    @property
    def nof_sources(self) -> int:
        """Number of sources."""
        if self.location is None:
            return 0
        return self.location.nof_observations

    def calculate_inclusion_idx(self, sensor_object: Sensor, inclusion_radius: Union[int, np.ndarray]) -> None:
        """Find observation indices which are within specified radius of each source location.

        This method takes the sensor object and for each source in the source_map object it calculates which
        observations are within the specified radius.
        When sensor_object location and sourcemap_object location are not of the same type, simply convert both to ECEF
        and calculate inclusion indices accordingly.
        The result is an array of lists which are the indices of the observations in sensor_object which are within the
        specified radius. Result is stored in the corresponding attribute.
        Also calculating number of observations in radius per source and storing result as a list in inclusion_n_obs
        attribute
        When a location attribute is in LLA we convert to ECEF for the inclusion radius to make sense

        Args:
            sensor_object (Sensor): Sensor object containing location information on the observations under
                consideration
            inclusion_radius (Union[float, np.ndarray], optional): Inclusion radius in [m] radius from source
                for which we take observations into account

        """
        sensor_kd_tree = sensor_object.location.to_ecef().create_tree()
        source_points = self.location.to_ecef().to_array()

        inclusion_idx = sensor_kd_tree.query_ball_point(source_points, inclusion_radius)
        idx_array = np.array(inclusion_idx, dtype=object)
        self.inclusion_idx = idx_array
        self.inclusion_n_obs = np.array([len(value) for value in self.inclusion_idx])

    def generate_sources(
        self,
        coordinate_object: Coordinate,
        sourcemap_limits: np.ndarray,
        sourcemap_type: str = "central",
        nof_sources: int = 5,
        grid_shape: Union[tuple, np.ndarray] = (5, 5, 1),
    ) -> None:
        """Generates source locations based on specified inputs.

        The result gets stored in the location attribute

        In grid_sphere we scale the latitude and longitude from -90/90 and -180/180 to 0/1 for the use in temp_lat_rad
        and temp_lon_rad

        Args:
            coordinate_object (Coordinate): Empty coordinate object which specifies the coordinate class to populate
                location with
            sourcemap_limits (np.ndarray): Limits of the sourcemap on which to generate the sources of size [dim x 2]
                if dim == 2 we assume the third dimension will be zeros. Assuming the units of the limits are defined in
                the desired coordinate system
            sourcemap_type (str, optional): Type of sourcemap to generate: central == 1 central source,
                hypercube == nof_sources through a Latin Hypercube design, grid == grid of shape grid_shape
                filled with sources, grid_sphere == grid of shape grid_shape taking into account a spherical spacing
            nof_sources (int, optional): Number of sources to generate (used in 'hypercube' case)
            grid_shape: (tuple, optional): Number of sources to generate in each dimension, total number of
                sources will be the product of the entries of this tuple (used in 'grid' and 'grid_sphere' case)

        """
        sourcemap_dimension = sourcemap_limits.shape[0]
        if sourcemap_type == "central":
            array = sourcemap_limits.mean(axis=1).reshape(1, sourcemap_dimension)
        elif sourcemap_type == "hypercube":
            array = make_latin_hypercube(bounds=sourcemap_limits, nof_samples=nof_sources)
        elif sourcemap_type == "grid":
            array = coordinate_object.make_grid(bounds=sourcemap_limits, grid_type="rectangular", shape=grid_shape)
        elif sourcemap_type == "grid_sphere":
            array = coordinate_object.make_grid(bounds=sourcemap_limits, grid_type="spherical", shape=grid_shape)
        else:
            raise NotImplementedError("Please provide a valid sourcemap type")
        coordinate_object.from_array(array=array)
        self.location = coordinate_object
