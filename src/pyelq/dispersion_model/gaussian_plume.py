# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Gaussian Plume module.

The class for the Gaussian Plume dispersion model used in pyELQ.

The Mathematics of Atmospheric Dispersion Modeling, John M. Stockie, DOI. 10.1137/10080991X

"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import numpy as np

from pyelq.coordinate_system import ENU, LLA
from pyelq.dispersion_model.dispersion_model import DispersionModel
from pyelq.gas_species import GasSpecies
from pyelq.meteorology.meteorology import Meteorology, MeteorologyGroup
from pyelq.sensor.beam import Beam
from pyelq.sensor.satellite import Satellite
from pyelq.sensor.sensor import Sensor, SensorGroup


@dataclass
class GaussianPlume(DispersionModel):
    """Defines the Gaussian plume dispersion model class.

    Attributes:
        source_half_width (float): Source half width (radius) to be used in the Gaussian plume model (in meters)

    """

    source_half_width: float = 1

    def compute_coupling(
        self,
        sensor_object: Union[SensorGroup, Sensor],
        meteorology_object: Union[MeteorologyGroup, Meteorology],
        gas_object: GasSpecies = None,
        output_stacked: bool = False,
        run_interpolation: bool = True,
    ) -> Union[list, np.ndarray, dict]:
        """Top level function to calculate the Gaussian plume coupling.

        Calculates the coupling for either a single sensor object or a dictionary of sensor objects.

        When both a SensorGroup and a MeteorologyGroup have been passed in, we assume they are consistent and contain
        exactly the same keys for each item in both groups. Also assuming interpolation has been performed and time axes
        are consistent, so we set run_interpolation to False

        When you input a SensorGroup and a single Meteorology object we convert this object into a dictionary, so we
        don't have to duplicate the same code.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Single sensor object or SensorGroup object which is used in the
                calculation of the plume coupling.
            meteorology_object (Union[MeteorologyGroup, Meteorology]): Meteorology object or MeteorologyGroup object
                which is used in the calculation of the plume coupling.
            gas_object (GasSpecies, optional): Optional input, a gas species object to correctly calculate the
                gas density which is used in the conversion of the units of the Gaussian plume coupling
            output_stacked (bool, optional): if true outputs as stacked np.array across sensors if not
                outputs as dict
            run_interpolation (bool, optional): logical indicating whether interpolation of the meteorological data to
                the sensor/source is required. Defaults to True.

        Returns:
            plume_coupling (Union[list, np.ndarray, dict]): List of arrays, single array or dictionary containing the
                plume coupling in hr/kg. When a single source object is passed in as input this function returns a list
                or an array depending on the sensor type.
                If a dictionary of sensor objects is passed in as input and output_stacked=False  this function returns
                a dictionary consistent with the input dictionary keys, containing the corresponding plume coupling
                outputs for each sensor.
                If a dictionary of sensor objects is passed in as input and output_stacked=True  this function returns
                a np.array containing the stacked coupling matrices.

        """
        if isinstance(sensor_object, SensorGroup):
            output = {}
            if isinstance(meteorology_object, Meteorology):
                meteorology_object = dict.fromkeys(sensor_object.keys(), meteorology_object)
            elif isinstance(meteorology_object, MeteorologyGroup):
                run_interpolation = False

            for sensor_key in sensor_object:
                output[sensor_key] = self.compute_coupling_single_sensor(
                    sensor_object=sensor_object[sensor_key],
                    meteorology=meteorology_object[sensor_key],
                    gas_object=gas_object,
                    run_interpolation=run_interpolation,
                )
            if output_stacked:
                output = np.concatenate(tuple(output.values()), axis=0)

        elif isinstance(sensor_object, Sensor):
            if isinstance(meteorology_object, MeteorologyGroup):
                raise TypeError("Please provide a single Meteorology object when using a single Sensor object")

            output = self.compute_coupling_single_sensor(
                sensor_object=sensor_object,
                meteorology=meteorology_object,
                gas_object=gas_object,
                run_interpolation=run_interpolation,
            )
        else:
            raise TypeError("Please provide either a Sensor or SensorGroup as input argument")

        return output

    def compute_coupling_single_sensor(
        self,
        sensor_object: Sensor,
        meteorology: Meteorology,
        gas_object: Union[GasSpecies, None] = None,
        run_interpolation: bool = True,
    ) -> Union[list, np.ndarray]:
        """Wrapper function to compute the gaussian plume coupling for a single sensor.

        Wrapper is used to identify specific cases and calculate the Gaussian plume coupling accordingly.

        When the sensor object contains the source_on attribute we set all coupling values to 0 for observations for
        which source_on is False. Making sure the source_on is column array, aligning with the 1st dimension
        (nof_observations) of the plume coupling array.

        Args:
            sensor_object (Sensor): Single sensor object which is used in the calculation of the plume coupling
            meteorology (Meteorology): Meteorology object which is used in the calculation of the plume coupling
            gas_object (GasSpecies, optional): Optionally input a gas species object to correctly calculate the
                gas density which is used in the conversion of the units of the Gaussian plume coupling
            run_interpolation (bool): logical indicating whether interpolation of the meteorological data to
                the sensor/source is required. Default passed from compute_coupling.

        Returns:
            plume_coupling (Union[list, np.ndarray]): List of arrays or single array containing the plume coupling
                in 1e6*[hr/kg]. Entries of the list are per source in the case of a satellite sensor, if a single array
                is returned the coupling for each observation (first dimension) to each source (second dimension) is
                provided.

        """
        if not isinstance(sensor_object, Sensor):
            raise NotImplementedError("Please provide a valid sensor type")

        (
            gas_density,
            u_interpolated,
            v_interpolated,
            wind_turbulence_horizontal,
            wind_turbulence_vertical,
        ) = self.interpolate_all_meteorology(
            meteorology=meteorology,
            sensor_object=sensor_object,
            gas_object=gas_object,
            run_interpolation=run_interpolation,
        )

        wind_speed = np.sqrt(u_interpolated**2 + v_interpolated**2)
        theta = np.arctan2(v_interpolated, u_interpolated)

        if isinstance(sensor_object, Satellite):
            plume_coupling = self.compute_coupling_satellite(
                sensor_object=sensor_object,
                wind_speed=wind_speed,
                theta=theta,
                wind_turbulence_horizontal=wind_turbulence_horizontal,
                wind_turbulence_vertical=wind_turbulence_vertical,
                gas_density=gas_density,
            )

        else:
            plume_coupling = self.compute_coupling_ground(
                sensor_object=sensor_object,
                wind_speed=wind_speed,
                theta=theta,
                wind_turbulence_horizontal=wind_turbulence_horizontal,
                wind_turbulence_vertical=wind_turbulence_vertical,
                gas_density=gas_density,
            )

        if sensor_object.source_on is not None:                        
            plume_coupling = plume_coupling *  np.where(sensor_object.source_on != 0, 1, 0)[:, None]

        return plume_coupling

    def compute_coupling_array(
        self,
        sensor_x: np.ndarray,
        sensor_y: np.ndarray,
        sensor_z: np.ndarray,
        source_z: np.ndarray,
        wind_speed: np.ndarray,
        theta: np.ndarray,
        wind_turbulence_horizontal: np.ndarray,
        wind_turbulence_vertical: np.ndarray,
        gas_density: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute the Gaussian plume coupling.

        Most low level function to calculate the Gaussian plume coupling. Assuming input shapes are consistent but no
        checking is done on this.

        Setting sigma_vert to 1e-16 when it is identically zero (distance_x == 0) so we don't get a divide by 0 error
        all the time.

        Args:
            sensor_x (np.ndarray): sensor x location relative to source [m].
            sensor_y (np.ndarray): sensor y location relative to source [m].
            sensor_z (np.ndarray): sensor z location relative to ground height [m].
            source_z (np.ndarray): source z location relative to ground height [m].
            wind_speed (np.ndarray): wind speed at source locations in [m/s].
            theta (np.ndarray): Mathematical wind direction at source locations [radians]:
                calculated as np.arctan2(v_component_wind, u_component_wind).
            wind_turbulence_horizontal (np.ndarray): Horizontal wind turbulence [deg].
            wind_turbulence_vertical (np.ndarray): Vertical wind turbulence [deg].
            gas_density (Union[float, np.ndarray]): Gas density to use in coupling calculation [kg/m^3].

        Returns:
            plume_coupling (np.ndarray): Gaussian plume coupling in (1e6)*[hr/kg]: gives concentrations
                in [ppm] when multiplied by sources in [kg/hr].

        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        distance_x = cos_theta * sensor_x + sin_theta * sensor_y
        if np.all(distance_x < 0):
            return np.zeros_like(distance_x)

        distance_y = -sin_theta * sensor_x + cos_theta * sensor_y

        sigma_hor = np.tan(wind_turbulence_horizontal * (np.pi / 180)) * np.abs(distance_x) + self.source_half_width
        sigma_vert = np.tan(wind_turbulence_vertical * (np.pi / 180)) * np.abs(distance_x)

        sigma_vert[sigma_vert == 0] = 1e-16

        plume_coupling = (
            (1 / (2 * np.pi * wind_speed * sigma_hor * sigma_vert))
            * np.exp(-0.5 * (distance_y / sigma_hor) ** 2)
            * (
                np.exp(-0.5 * (((sensor_z + source_z) / sigma_vert) ** 2))
                + np.exp(-0.5 * (((sensor_z - source_z) / sigma_vert) ** 2))
            )
        )

        plume_coupling = np.divide(np.multiply(plume_coupling, 1e6), (gas_density * 3600))
        plume_coupling[np.logical_or(distance_x < 0, plume_coupling < self.minimum_contribution)] = 0

        return plume_coupling

    def compute_coupling_satellite(
        self,
        sensor_object: Sensor,
        wind_speed: np.ndarray,
        theta: np.ndarray,
        wind_turbulence_horizontal: np.ndarray,
        wind_turbulence_vertical: np.ndarray,
        gas_density: np.ndarray,
    ) -> list:
        """Compute Gaussian plume coupling for satellite sensor.

        When the sensor is a Satellite object we calculate the plume coupling per source. Given the large number of
        sources and the possibility of using the inclusion radius and inclusion indices here and validity of a local
        ENU system over large distances we loop over each source and calculate the coupling on a per-source basis.

        If source_map.inclusion_n_obs is None, we do not do any filtering on observations and we want to include all
        observations in the plume coupling calculations.

        All np.ndarray inputs should have a shape of [1 x nof_sources]

        Args:
            sensor_object (Sensor): Sensor object used in plume coupling calculation
            wind_speed (np.ndarray): Wind speed [m/s]
            theta (np.ndarray): Mathematical angle between the u- and v-components of wind [radians]
            wind_turbulence_horizontal (np.ndarray): Parameter of the wind stability in horizontal direction [deg]
            wind_turbulence_vertical (np.ndarray): Parameter of the wind stability in vertical direction [deg]
            gas_density: (np.ndarray): Numpy array containing the gas density values to use [kg/m^3]

        Returns:
            plume_coupling (list): List of Gaussian plume coupling 1e6*[hr/kg] arrays. The list has a length of
                nof_sources, each array has the shape [nof_observations x 1] or [inclusion_n_obs x 1] when
                inclusion_idx is used.

        """
        plume_coupling = []

        source_map_location_lla = self.source_map.location.to_lla()
        for current_source in range(self.source_map.nof_sources):
            if self.source_map.inclusion_n_obs is None:
                enu_sensor_array = sensor_object.location.to_enu(
                    ref_latitude=source_map_location_lla.latitude[current_source],
                    ref_longitude=source_map_location_lla.longitude[current_source],
                    ref_altitude=0,
                ).to_array()

            else:
                if self.source_map.inclusion_n_obs[current_source] == 0:
                    plume_coupling.append(np.array([]))
                    continue

                enu_sensor_array = _create_enu_sensor_array(
                    inclusion_idx=self.source_map.inclusion_idx[current_source],
                    sensor_object=sensor_object,
                    source_map_location_lla=source_map_location_lla,
                    current_source=current_source,
                )

            temp_coupling = self.compute_coupling_array(
                enu_sensor_array[:, [0]],
                enu_sensor_array[:, [1]],
                enu_sensor_array[:, [2]],
                source_map_location_lla.altitude[current_source],
                wind_speed[:, current_source],
                theta[:, current_source],
                wind_turbulence_horizontal[:, current_source],
                wind_turbulence_vertical[:, current_source],
                gas_density[:, current_source],
            )

            plume_coupling.append(temp_coupling)

        return plume_coupling

    def compute_coupling_ground(
        self,
        sensor_object: Sensor,
        wind_speed: np.ndarray,
        theta: np.ndarray,
        wind_turbulence_horizontal: np.ndarray,
        wind_turbulence_vertical: np.ndarray,
        gas_density: np.ndarray,
    ) -> np.ndarray:
        """Compute Gaussian plume coupling for a ground sensor.

        If the source map is already defined as ENU the reference location is maintained but the sensor is checked
        to make sure the same reference location is used. Otherwise, when converting to ENU object for the sensor
        observations we use a single source and altitude 0 as the reference location. This way our ENU system is a
        system w.r.t. ground level which is required for the current implementation of the actual coupling calculation.

        When the sensor is a Beam object we calculate the plume coupling for all sources to all beam knot locations at
        once in the same ENU coordinate system and finally averaged over the beam knots to get the final output.

        In general, we calculate the coupling from all sources to all sensor observation locations. In order to achieve
        this we input the sensor array as column and source array as row vector in calculating relative x etc.,
        with the beam knot locations being the third dimension. When the sensor is a single point Sensor or a Drone
        sensor we effectively have one beam knot, making the mean operation at the end effectively a reshape operation
        which gets rid of the third dimension.

        All np.ndarray inputs should have a shape of [nof_observations x 1]

        Args:
            sensor_object (Sensor): Sensor object used in plume coupling calculation
            wind_speed (np.ndarray): Wind speed [m/s]
            theta (np.ndarray): Mathematical angle between the u- and v-components of wind [radians]
            wind_turbulence_horizontal (np.ndarray): Parameter of the wind stability in horizontal direction [deg]
            wind_turbulence_vertical (np.ndarray): Parameter of the wind stability in vertical direction [deg]
            gas_density: (np.ndarray): Numpy array containing the gas density values to use [kg/m^3]

        Returns:
            plume_coupling (np.ndarray): Gaussian plume coupling 1e6*[hr/kg] array. The array has the
                shape [nof_observations x nof_sources]

        """
        if not isinstance(self.source_map.location, ENU):
            source_map_lla = self.source_map.location.to_lla()
            source_map_enu = source_map_lla.to_enu(
                ref_latitude=source_map_lla.latitude[0], ref_longitude=source_map_lla.longitude[0], ref_altitude=0
            )
        else:
            source_map_enu = self.source_map.location

        enu_source_array = source_map_enu.to_array()

        if isinstance(sensor_object, Beam):
            enu_sensor_array = sensor_object.make_beam_knots(
                ref_latitude=source_map_enu.ref_latitude,
                ref_longitude=source_map_enu.ref_longitude,
                ref_altitude=source_map_enu.ref_altitude,
            )
            relative_x = np.subtract(enu_sensor_array[:, 0][None, None, :], enu_source_array[:, 0][None, :, None])
            relative_y = np.subtract(enu_sensor_array[:, 1][None, None, :], enu_source_array[:, 1][None, :, None])
            z_sensor = enu_sensor_array[:, 2][None, None, :]
        else:
            enu_sensor_array = sensor_object.location.to_enu(
                ref_latitude=source_map_enu.ref_latitude,
                ref_longitude=source_map_enu.ref_longitude,
                ref_altitude=source_map_enu.ref_altitude,
            ).to_array()
            relative_x = np.subtract(enu_sensor_array[:, 0][:, None, None], enu_source_array[:, 0][None, :, None])
            relative_y = np.subtract(enu_sensor_array[:, 1][:, None, None], enu_source_array[:, 1][None, :, None])
            z_sensor = enu_sensor_array[:, 2][:, None, None]

        z_source = enu_source_array[:, 2][None, :, None]

        plume_coupling = self.compute_coupling_array(
            relative_x,
            relative_y,
            z_sensor,
            z_source,
            wind_speed[:, :, None],
            theta[:, :, None],
            wind_turbulence_horizontal[:, :, None],
            wind_turbulence_vertical[:, :, None],
            gas_density[:, :, None],
        )

        plume_coupling = plume_coupling.mean(axis=2)

        return plume_coupling


def _create_enu_sensor_array(
    inclusion_idx: np.ndarray, sensor_object: Sensor, source_map_location_lla: LLA, current_source: int
):
    """Helper function to create ENU sensor array when we only want ot include specific observation locations.

    This function gets called when we need to create the enu_sensor_array when we only want to include specific
    observation locations. First we obtain the subset of locations from the sensor object and convert that to an array.
    Given we don't know which coordinate system the sensor_object is created in, we make a copy of the original sensor
    object, thereby keeping all key details of the coordinate system and repopulate the location values accordingly
    through the from_array method using the subset of locations from the sensor object. Finally, we convert the subset
    to ENU and return that as output.

    Args:
        inclusion_idx (np.ndarray): Numpy array containing the indices of observations in the sensor_object to be used
            in the Gaussian plume coupling.
        sensor_object (Sensor): Sensor object to be used in the Gaussian Plume calculation.
        source_map_location_lla (LLA): LLA coordinate object of the source map locations.
        current_source (int): Integer index of the current source for which we want to use in the Gaussian plume
            calculation.

    """
    temp_array = sensor_object.location.to_array()[inclusion_idx, :]
    temp_object = deepcopy(sensor_object.location)
    temp_object.from_array(array=temp_array)
    enu_sensor_array = temp_object.to_enu(
        ref_latitude=source_map_location_lla.latitude[current_source],
        ref_longitude=source_map_location_lla.longitude[current_source],
        ref_altitude=0,
    ).to_array()

    return enu_sensor_array
