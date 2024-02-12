# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Coordinate System.

This code provides the definition of, and the functionality for, all the main coordinate systems that are used in
pyELQ. Each coordinate system has relevant methods for features that are commonly required. Also provided is a set of
conversions between each of the systems, alongside some functionality for interpolation.

"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pymap3d as pm
from scipy.spatial import KDTree
from scipy.stats import qmc

import pyelq.support_functions.spatio_temporal_interpolation as sti


def make_latin_hypercube(bounds: np.ndarray, nof_samples: int) -> np.ndarray:
    """Latin Hypercube samples.

    Draw samples according to a Latin Hypercube design within the specified bounds.

    Args:
        bounds (np.ndarray): Limits of the resulting hypercube of size [dim x 2]
        nof_samples (int): Number of samples to draw

    Returns:
        array (np.ndarray): Samples forming the Latin Hypercube

    """
    dimension = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler.random(n=nof_samples)
    array = qmc.scale(sample, np.min(bounds, axis=1), np.max(bounds, axis=1))
    return array


@dataclass
class Coordinate(ABC):
    """Abstract base class for coordinate transformations.

    Attributes:
        use_degrees (bool): Flag if reference uses degrees (True) or radians (False). Defaults to True.
        ellipsoid (pm.Ellipsoid): Definition of the Ellipsoid used in the coordinate system, for which the default is
            WGS84. See: https://en.wikipedia.org/wiki/World_Geodetic_System.

    """

    use_degrees: bool = field(init=False)
    ellipsoid: pm.Ellipsoid = field(init=False)

    def __post_init__(self):
        self.use_degrees = True
        self.ellipsoid = pm.Ellipsoid.from_name("wgs84")

    @property
    @abstractmethod
    def nof_observations(self) -> int:
        """Number of observations contained in the class instance, implemented as dependent property."""

    @abstractmethod
    def from_array(self, array: np.ndarray) -> None:
        """Unstack a numpy array into the corresponding coordinates.

        The method has no return as it sets the corresponding attributes of the coordinate class instance.

        Args:
            array (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single
                array

        """

    @abstractmethod
    def to_array(self, dim: int = 3) -> np.ndarray:
        """Stacks coordinates together into a numpy array.

        Args:
            dim (int, optional): Number of dimensions to use, which is either 2 or 3.

        Returns:
            np.ndarray: Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single array

        """

    @abstractmethod
    def to_lla(self):
        """LLA: Converts coordinates to latitude/longitude/altitude system."""

    @abstractmethod
    def to_ecef(self):
        """ECEF: Convert coordinates to earth centered earth fixed coordinates."""

    @abstractmethod
    def to_enu(self, ref_latitude: float = None, ref_longitude: float = None, ref_altitude: float = None):
        """Converts coordinates to East North Up system.

        If a reference is not provided, the  minimum of coordinates in Lat/Lon/Alt is used as the reference.

        Args:
            ref_latitude (float, optional): reference latitude for ENU
            ref_longitude (float, optional): reference longitude for ENU
            ref_altitude (float, optional):  reference altitude for ENU

        Returns:
           (ENU): East North Up coordinate object

        """

    def to_object_type(self, coordinate_object):
        """Converts current object to same class as input coordinate_object.

        Args:
            coordinate_object (Coordinate): An coordinate object which provides the coordinate system to convert self to

        Returns:
            (Coordinate): The converted coordinate object

        """
        if type(coordinate_object) is not type(self):
            if isinstance(coordinate_object, LLA):
                temp_object = self.to_lla()
            elif isinstance(coordinate_object, ENU):
                temp_object = self.to_enu(
                    ref_latitude=coordinate_object.ref_latitude,
                    ref_longitude=coordinate_object.ref_longitude,
                    ref_altitude=coordinate_object.ref_altitude,
                )
            elif isinstance(coordinate_object, ECEF):
                temp_object = self.to_ecef()
            else:
                raise TypeError("Please provide a valid coordinate type")

            return temp_object

        return self

    def interpolate(self, values: np.ndarray, locations, dim: int = 3, **kwargs) -> np.ndarray:
        """Interpolate data using coordinate object.

        If locations coordinate system does not match self's coordinate system it will be converted to same type as
        self. In the ENU case extra checking needs to take place to check reference locations match up.

        If only 1 value is provided which needs to be interpolated to many other locations we just set the value at all
        these locations to the single input value

        Args:
            values (np.ndarray): Values to interpolate,  consistent with location in self
            locations (Coordinate): Coordinate object containing locations to which you want to interpolate
            dim (int): Number of dimensions to use for interpolation (2 or 3)
            **kwargs (dict):  Other arguments available in scipy.interpolate.griddata e.g. method, fill_value

        Returns:
            Result (np.ndarray): Interpolated values at requested locations.

        """
        locations = locations.to_object_type(coordinate_object=self)

        if isinstance(self, ENU):
            if (
                self.ref_latitude != locations.ref_latitude
                or self.ref_longitude != locations.ref_longitude
                or self.ref_altitude != locations.ref_altitude
            ):
                locations = locations.to_lla()
                locations = locations.to_enu(
                    ref_latitude=self.ref_latitude, ref_longitude=self.ref_longitude, ref_altitude=self.ref_altitude
                )
        result = sti.interpolate(
            location_in=self.to_array(dim),
            values_in=values.flatten(),
            location_out=locations.to_array(dim=dim),
            **kwargs,
        )

        return result

    def make_grid(
        self, bounds: np.ndarray, grid_type: str = "rectangular", shape: Union[tuple, np.ndarray] = (5, 5, 1)
    ) -> np.ndarray:
        """Generates grid of values locations based on specified inputs.

        If the grid type is 'spherical', we scale the latitude and longitude from -90/90 and -180/180 to 0/1 for the
        use in temp_lat_rad and temp_lon_rad.

        Args:
            bounds (np.ndarray): Limits of the grid on which to generate the grid of size [dim x 2]
                if dim == 2 we assume the third dimension will be zeros
            grid_type (str, optional): Type of grid to generate, default 'rectangular':
                     rectangular == rectangular grid of shape grd_shape,
                     spherical == grid of shape grid_shape taking into account a spherical spacing
            shape: (tuple, optional): Number of grid cells to generate in each dimension, total number of
                grid cells will be the product of the entries of this tuple

        Returns
            np.ndarray: gridded of locations

        """
        dimension = bounds.shape[0]

        if grid_type == "rectangular":
            dim_0 = np.linspace(bounds[0, 0], bounds[0, 1], num=shape[0])
            dim_1 = np.linspace(bounds[1, 0], bounds[1, 1], num=shape[1])
            if dimension == 3:
                dim_2 = np.linspace(bounds[2, 0], bounds[2, 1], num=shape[2])
            else:
                dim_2 = np.array(0)

            dim_0, dim_1, dim_2 = np.meshgrid(dim_0, dim_1, dim_2)
            array = np.stack([dim_0.flatten(), dim_1.flatten(), dim_2.flatten()], axis=1)
        elif grid_type == "spherical":
            temp_object = deepcopy(self)
            temp_object.from_array(array=bounds)
            temp_object = temp_object.to_lla()
            temp_object.latitude = (temp_object.latitude - (-90)) / 180
            temp_object.longitude = (temp_object.longitude - (-180)) / 360

            temp_lat_rad = np.linspace(start=temp_object.latitude[0], stop=temp_object.latitude[1], num=shape[0])
            temp_lon_rad = np.linspace(start=temp_object.longitude[0], stop=temp_object.longitude[1], num=shape[1])

            longitude = (2 * np.pi * temp_lon_rad - np.pi) * 180 / np.pi
            latitude = (np.arccos(1 - 2 * temp_lat_rad) - 0.5 * np.pi) * 180 / np.pi
            if dimension == 3:
                altitude = np.linspace(start=temp_object.altitude[0], stop=temp_object.altitude[1], num=shape[2])
                latitude, longitude, altitude = np.meshgrid(latitude, longitude, altitude)
                array = np.stack(
                    [latitude.flatten() * np.pi / 180, longitude.flatten() * np.pi / 180, altitude.flatten()], axis=1
                )
            else:
                latitude, longitude = np.meshgrid(latitude, longitude)
                array = np.stack([latitude.flatten() * np.pi / 180, longitude.flatten() * np.pi / 180], axis=1)

            temp_object.from_array(array=array)
            temp_object = temp_object.to_object_type(self)
            array = temp_object.to_array()
        else:
            raise NotImplementedError("Please provide a valid grid type")

        return array

    def create_tree(self) -> KDTree:
        """Create KD tree for the purpose of fast distance computation.

        Returns:
                KDTree: Spatial KD tree

        """
        return KDTree(self.to_array())


@dataclass
class LLA(Coordinate):
    """Defines the properties and functionality of the latitude/ longitude/ altitude coordinate system.

    Attributes:
        latitude (np.ndarray): Latitude values in degrees.
        longitude (np.ndarray): Longitude values in degrees.
        altitude (np.ndarray): Altitude values in meters with respect to a spheroid.

    """

    latitude: np.ndarray = None
    longitude: np.ndarray = None
    altitude: np.ndarray = None

    @property
    def nof_observations(self):
        """Number of observations contained in the class instance, implemented as dependent property."""
        if self.latitude is None:
            return 0
        return self.latitude.size

    def from_array(self, array):
        """Unstack a numpy array into the corresponding coordinates.

        The method has no return as it sets the corresponding attributes of the coordinate class instance.

        Args:
            array (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single
                array

        """
        dim = array.shape[1]
        self.latitude = array[:, 0]
        self.longitude = array[:, 1]
        self.altitude = np.zeros_like(self.latitude)
        if dim == 3:
            self.altitude = array[:, 2]

    def to_array(self, dim=3):
        """Stacks coordinates together into a numpy array.

        Args:
            dim (int, optional): Number of dimensions to use, which is either 2 or 3.

        Returns:
            (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single array

        """
        if dim == 2:
            return np.stack((self.latitude.flatten(), self.longitude.flatten()), axis=1)
        return np.stack((self.latitude.flatten(), self.longitude.flatten(), self.altitude.flatten()), axis=1)

    def to_lla(self):
        """LLA: Converts coordinates to latitude/longitude/altitude system."""
        return self

    def to_ecef(self):
        """ECEF: Convert coordinates to earth centered earth fixed coordinates."""
        if self.altitude is None:
            self.altitude = np.zeros(self.latitude.shape)
        ecef_object = ECEF()
        ecef_object.x, ecef_object.y, ecef_object.z = pm.geodetic2ecef(
            lat=self.latitude, lon=self.longitude, alt=self.altitude, ell=self.ellipsoid, deg=self.use_degrees
        )

        return ecef_object

    def to_enu(self, ref_latitude=None, ref_longitude=None, ref_altitude=None):
        """Converts coordinates to East North Up system.

        If a reference is not provided, the  minimum of coordinates in Lat/Lon/Alt is used as the reference.

        Args:
            ref_latitude (float, optional): reference latitude for ENU
            ref_longitude (float, optional): reference longitude for ENU
            ref_altitude (float, optional):  reference altitude for ENU

        Returns:
           (ENU): East North Up coordinate object

        """
        if self.altitude is None:
            self.altitude = np.zeros(self.latitude.shape)

        if ref_altitude is None:
            ref_altitude = np.amin(self.altitude)

        if ref_latitude is None:
            ref_latitude = np.amin(self.latitude)

        if ref_longitude is None:
            ref_longitude = np.amin(self.longitude)

        enu_object = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)

        enu_object.east, enu_object.north, enu_object.up = pm.geodetic2enu(
            lat=self.latitude,
            lon=self.longitude,
            h=self.altitude,
            lat0=ref_latitude,
            lon0=ref_longitude,
            h0=ref_altitude,
            ell=self.ellipsoid,
            deg=self.use_degrees,
        )

        return enu_object


@dataclass
class ENU(Coordinate):
    """Defines the properties and functionality of a local East-North-Up coordinate system.

     Positions relative to some reference location in metres.

    Attributes:
        ref_latitude (float): Reference latitude for current ENU system.
        ref_longitude (float): Reference longitude for current ENU system.
        ref_altitude (float): Reference altitude for current ENU system.
        east (np.ndarray): East values.
        north (np.ndarray): North values.
        up: (np.ndarray): Up values.

    """

    ref_latitude: float
    ref_longitude: float
    ref_altitude: float
    east: np.ndarray = None
    north: np.ndarray = None
    up: np.ndarray = None

    @property
    def nof_observations(self):
        """Number of observations contained in the class instance, implemented as dependent property."""
        if self.east is None:
            return 0
        return self.east.size

    def from_array(self, array):
        """Unstack a numpy array into the corresponding coordinates.

        The method has no return as it sets the corresponding attributes of the coordinate class instance.

        Args:
            array (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single
                array

        """
        dim = array.shape[1]
        self.east = array[:, 0]
        self.north = array[:, 1]
        self.up = np.zeros_like(self.east)
        if dim == 3:
            self.up = array[:, 2]

    def to_array(self, dim=3):
        """Stacks coordinates together into a numpy array.

        Args:
            dim (int, optional): Number of dimensions to use, which is either 2 or 3.

        Returns:
            (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single array

        """
        if dim == 2:
            return np.stack((self.east.flatten(), self.north.flatten()), axis=1)
        return np.stack((self.east.flatten(), self.north.flatten(), self.up.flatten()), axis=1)

    def to_enu(self, ref_latitude=None, ref_longitude=None, ref_altitude=None):
        """Converts coordinates to East North Up system.

        If a reference is not provided, the  minimum of coordinates in Lat/Lon/Alt is used as the reference.

        Args:
            ref_latitude (float, optional): reference latitude for ENU
            ref_longitude (float, optional): reference longitude for ENU
            ref_altitude (float, optional):  reference altitude for ENU

        Returns:
           (ENU): East North Up coordinate object

        """
        if ref_latitude is None:
            ref_latitude = self.ref_latitude

        if ref_longitude is None:
            ref_longitude = self.ref_longitude

        if ref_altitude is None:
            ref_altitude = self.ref_altitude

        if (
            self.ref_latitude == ref_latitude
            and self.ref_longitude == ref_longitude
            and self.ref_altitude == ref_altitude
        ):
            return self

        ecef_temp = self.to_ecef()

        return ecef_temp.to_enu(ref_longitude=ref_longitude, ref_latitude=ref_latitude, ref_altitude=ref_altitude)

    def to_lla(self):
        """LLA: Converts coordinates to latitude/longitude/altitude system."""
        lla_object = LLA()

        lla_object.latitude, lla_object.longitude, lla_object.altitude = pm.enu2geodetic(
            e=self.east,
            n=self.north,
            u=self.up,
            lat0=self.ref_latitude,
            lon0=self.ref_longitude,
            h0=self.ref_altitude,
            ell=self.ellipsoid,
            deg=self.use_degrees,
        )

        return lla_object

    def to_ecef(self):
        """ECEF: Convert coordinates to earth centered earth fixed coordinates."""
        ecef_object = ECEF()

        ecef_object.x, ecef_object.y, ecef_object.z = pm.enu2ecef(
            e1=self.east,
            n1=self.north,
            u1=self.up,
            lat0=self.ref_latitude,
            lon0=self.ref_longitude,
            h0=self.ref_altitude,
            ell=self.ellipsoid,
            deg=self.use_degrees,
        )

        return ecef_object


@dataclass
class ECEF(Coordinate):
    """Defines the properties and functionality of an Earth-Centered, Earth-Fixed coordinate system.

    See: https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system

    Attributes:
        x (np.ndarray): Eastings values [metres]
        y (np.ndarray): Northings values [metres]
        z (np.ndarray): Altitude values [metres]

    """

    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None

    @property
    def nof_observations(self):
        """Number of observations contained in the class instance, implemented as dependent property."""
        if self.x is None:
            return 0
        return self.x.size

    def from_array(self, array):
        """Unstack a numpy array into the corresponding coordinates.

        The method has no return as it sets the corresponding attributes of the coordinate class instance.

        Args:
            array (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single
                array

        """
        dim = array.shape[1]
        self.x = array[:, 0]
        self.y = array[:, 1]
        self.z = np.zeros_like(self.x)
        if dim == 3:
            self.z = array[:, 2]

    def to_array(self, dim=3):
        """Stacks coordinates together into a numpy array.

        Args:
            dim (int, optional): Number of dimensions to use, which is either 2 or 3.

        Returns:
            (np.ndarray): Numpy array of size [n x dim] with n>0 containing the coordinates stacked into a single array

        """
        if dim == 2:
            return np.stack((self.x.flatten(), self.y.flatten()), axis=1)
        return np.stack((self.x.flatten(), self.y.flatten(), self.z.flatten()), axis=1)

    def to_ecef(self):
        """ECEF: Convert coordinates to earth centered earth fixed coordinates."""
        return self

    def to_lla(self):
        """LLA: Converts coordinates to latitude/longitude/altitude system."""
        lla_object = LLA()

        lla_object.latitude, lla_object.longitude, lla_object.altitude = pm.ecef2geodetic(
            self.x, self.y, self.z, ell=self.ellipsoid, deg=self.use_degrees
        )

        return lla_object

    def to_enu(self, ref_latitude=None, ref_longitude=None, ref_altitude=None):
        """Converts coordinates to East North Up system.

        If a reference is not provided, the  minimum of coordinates in Lat/Lon/Alt is used as the reference.

        Args:
            ref_latitude (float, optional): reference latitude for ENU
            ref_longitude (float, optional): reference longitude for ENU
            ref_altitude (float, optional):  reference altitude for ENU

        Returns:
           (ENU): East North Up coordinate object

        """
        if ref_latitude is None or ref_longitude is None or ref_altitude is None:
            lla_object = self.to_lla()
            return lla_object.to_enu()

        enu_object = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)

        enu_object.east, enu_object.north, enu_object.up = pm.ecef2enu(
            x=self.x,
            y=self.y,
            z=self.z,
            lat0=ref_latitude,
            lon0=ref_longitude,
            h0=ref_altitude,
            ell=self.ellipsoid,
            deg=self.use_degrees,
        )

        return enu_object
