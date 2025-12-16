# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""DispersionModel module.

The super class for the Gaussian Plume and Finite Volume dispersion models used in pyELQ.

The Mathematics of Atmospheric Dispersion Modeling, John M. Stockie, DOI. 10.1137/10080991X

"""
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

import pyelq.support_functions.spatio_temporal_interpolation as sti
from pyelq.gas_species import GasSpecies
from pyelq.meteorology import Meteorology
from pyelq.sensor.satellite import Satellite
from pyelq.sensor.sensor import Sensor
from pyelq.source_map import SourceMap


@dataclass
class DispersionModel(ABC):
    """Defines the dispersion model class.

    Attributes:
        source_map (Sourcemap): SourceMap object used for the dispersion model.
        minimum_contribution (float): All elements in the plume coupling smaller than this number will be set
            to 0. Helps to speed up matrix multiplications/matrix inverses, also helps with stability.

    """

    source_map: SourceMap
    minimum_contribution: float = 0

    def calculate_gas_density(
        self,
        meteorology: Meteorology,
        sensor_object: Sensor,
        gas_object: Union[GasSpecies, None],
        run_interpolation: bool = True,
    ) -> np.ndarray:
        """Helper function to calculate the gas density using ideal gas law.

        https://en.wikipedia.org/wiki/Ideal_gas

        When a gas object is passed as input we calculate the density according to that gas. We check if the
        meteorology object has a temperature and/or pressure value and use those accordingly. Otherwise, we use Standard
        Temperature and Pressure (STP).

        If run_interpolation is True, we interpolate the temperature and pressure values to the source locations/times
        such that this is consistent with the other calculations, i.e. we only do spatial interpolation when the sensor
        is a Satellite object and temporal interpolation otherwise.

        When no gas_object is passed in we just set the gas density value to 1.

        Args:
            meteorology (Meteorology): Meteorology object potentially containing temperature or pressure values
            sensor_object (Sensor): Sensor object containing information about where to interpolate to
            gas_object (Union[GasSpecies, None]): Gas species object which actually calculates the correct density
            run_interpolation (bool): Flag indicating whether to run interpolation, defaults to True.

        Returns:
            gas_density (np.ndarray): Numpy array of shape [1 x nof_sources] (Satellite sensor)
                or [nof_observations x 1] (otherwise) containing the gas density values to use

        """
        if not isinstance(gas_object, GasSpecies):
            if isinstance(sensor_object, Satellite):
                return np.ones((1, self.source_map.nof_sources))
            return np.ones((sensor_object.nof_observations, 1))

        if meteorology.temperature is None:
            temperature = np.array([[273.15]])

        elif run_interpolation:
            temperature = self.interpolate_meteorology(
                meteorology=meteorology, variable_name="temperature", sensor_object=sensor_object
            )
        else:
            temperature = meteorology.temperature

        if meteorology.pressure is None:
            pressure = np.array([[101.325]])
        elif run_interpolation:
            pressure = self.interpolate_meteorology(
                meteorology=meteorology, variable_name="pressure", sensor_object=sensor_object
            )
        else:
            pressure = meteorology.pressure

        gas_density = gas_object.gas_density(temperature=temperature, pressure=pressure)
        return gas_density

    def interpolate_all_meteorology(
        self, sensor_object: Sensor, meteorology: Meteorology, gas_object: GasSpecies, run_interpolation: bool
    ):
        """Function which carries out interpolation of all meteorological information.

        The flag run_interpolation determines whether the interpolation should be carried out. If this is set to be
        False, the meteorological parameters are simply set to the values stored on the meteorology object (i.e. we
        assume that the meteorology has already been interpolated). This functionality is required to avoid wasted
        computation in the case of e.g. a reversible jump run.

        Args:
            sensor_object (Sensor): object containing locations/times onto which met information should
                be interpolated.
            meteorology (Meteorology): object containing meteorology information for interpolation.
            gas_object (GasSpecies): object containing gas information.
            run_interpolation (bool): logical indicating whether the meteorology information needs to be interpolated.

        Returns:
            gas_density (np.ndarray): numpy array of shape [n_data x 1] of gas densities.
            u_interpolated (np.ndarray): numpy array of shape [n_data x 1] of northerly wind components.
            v_interpolated (np.ndarray): numpy array of shape [n_data x 1] of easterly wind components.
            wind_turbulence_horizontal (np.ndarray): numpy array of shape [n_data x 1] of horizontal turbulence
                parameters.
            wind_turbulence_vertical (np.ndarray): numpy array of shape [n_data x 1] of vertical turbulence
                parameters.

        """
        if run_interpolation:
            gas_density = self.calculate_gas_density(
                meteorology=meteorology, sensor_object=sensor_object, gas_object=gas_object
            )
            u_interpolated = self.interpolate_meteorology(
                meteorology=meteorology, variable_name="u_component", sensor_object=sensor_object
            )
            v_interpolated = self.interpolate_meteorology(
                meteorology=meteorology, variable_name="v_component", sensor_object=sensor_object
            )
            wind_turbulence_horizontal = self.interpolate_meteorology(
                meteorology=meteorology, variable_name="wind_turbulence_horizontal", sensor_object=sensor_object
            )
            wind_turbulence_vertical = self.interpolate_meteorology(
                meteorology=meteorology, variable_name="wind_turbulence_vertical", sensor_object=sensor_object
            )
        else:
            gas_density = gas_object.gas_density(temperature=meteorology.temperature, pressure=meteorology.pressure)
            gas_density = gas_density.reshape((gas_density.size, 1))
            u_interpolated = meteorology.u_component.reshape((meteorology.u_component.size, 1))
            v_interpolated = meteorology.v_component.reshape((meteorology.v_component.size, 1))
            wind_turbulence_horizontal = meteorology.wind_turbulence_horizontal.reshape(
                (meteorology.wind_turbulence_horizontal.size, 1)
            )
            wind_turbulence_vertical = meteorology.wind_turbulence_vertical.reshape(
                (meteorology.wind_turbulence_vertical.size, 1)
            )

        return gas_density, u_interpolated, v_interpolated, wind_turbulence_horizontal, wind_turbulence_vertical

    def interpolate_meteorology(
        self, meteorology: Meteorology, variable_name: str, sensor_object: Sensor
    ) -> Union[np.ndarray, None]:
        """Helper function to interpolate meteorology variables.

        This function interpolates meteorological variables to times in Sensor or Sources in sourcemap. It also
        calculates the wind speed and mathematical angle between the u- and v-components which in turn gets used in the
        calculation of the Gaussian plume.

        When the input sensor object is a Satellite type we use spatial interpolation using the interpolation method
        from the coordinate system class as this takes care of the coordinate systems.
        When the input sensor object is of another time we use temporal interpolation (assumption is spatial uniformity
        for all observations over a small(er) area).

        Args:
            meteorology (Meteorology): Meteorology object containing u- and v-components of wind including their
                spatial location
            variable_name (str): String name of an attribute in the meteorology input object which needs to be
                interpolated
            sensor_object (Sensor): Sensor object containing information about where to interpolate to

        Returns:
            variable_interpolated (np.ndarray): Interpolated values

        """
        variable = getattr(meteorology, variable_name)
        if variable is None:
            return None

        if isinstance(sensor_object, Satellite):
            variable_interpolated = meteorology.location.interpolate(variable, self.source_map.location)
            variable_interpolated = variable_interpolated.reshape(1, self.source_map.nof_sources)
        else:
            variable_interpolated = sti.interpolate(
                time_in=meteorology.time, values_in=variable, time_out=sensor_object.time
            )
            variable_interpolated = variable_interpolated.reshape(sensor_object.nof_observations, 1)
        return variable_interpolated

    @staticmethod
    def compute_coverage(
        couplings: np.ndarray, threshold_function: Callable, coverage_threshold: float = 6, **kwargs
    ) -> Union[np.ndarray, dict]:
        """Returns a logical vector that indicates which sources in the couplings are, or are not, within the coverage.

        The 'coverage' is the area inside which all sources are well covered by wind data. E.g. If wind exclusively
        blows towards East, then all sources to the East of any sensor are 'invisible', and are not within the coverage.

        Couplings are returned in hr/kg. Some threshold function defines the largest allowed coupling value. This is
        used to calculate estimated emission rates in kg/hr. Any emissions which are greater than the value of
        'coverage_threshold' are defined as not within the coverage.

        Args:
            couplings (np.ndarray): Array of coupling values. Dimensions: n_data points x n_sources.
            threshold_function (Callable): Callable function which returns some single value that defines the
                maximum or 'threshold' coupling. For example: np.quantile(., q=0.95)
            coverage_threshold (float, optional): The threshold value of the estimated emission rate which is
                considered to be within the coverage. Defaults to 6 kg/hr.
            kwargs (dict, optional): Keyword arguments required for the threshold function.

        Returns:
            coverage (Union[np.ndarray, dict]): A logical array specifying which sources are within the coverage.

        """
        coupling_threshold = threshold_function(couplings, **kwargs)
        no_warning_threshold = np.where(coupling_threshold <= 1e-100, 1, coupling_threshold)
        no_warning_estimated_emission_rates = np.where(coupling_threshold <= 1e-100, np.inf, 1 / no_warning_threshold)
        coverage = no_warning_estimated_emission_rates < coverage_threshold
        return coverage
