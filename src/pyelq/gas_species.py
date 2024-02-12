# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Gas Species module.

The superclass for the Gas species classes. It contains a few gas species with its properties and functionality to
calculate the density of the gas and do emission rate conversions from m^3/s to kg/hr and back

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np


@dataclass
class GasSpecies(ABC):
    """Defines the properties of a particular gas species.

    Attributes:
        global_background (float, optional): Global background concentration [ppm]
        half_life (float, optional): Half life of gas [hr]
        __molar_gas_constant (float): R, molar gas constant [JK^-1mol^-1]

    """

    global_background: float = field(init=False)
    half_life: float = field(init=False)
    __molar_gas_constant: float = 8.31446261815324

    @property
    @abstractmethod
    def name(self) -> str:
        """Str: Name of gas."""

    @property
    @abstractmethod
    def molar_mass(self) -> float:
        """Float: Molar Mass [g/mol]."""

    @property
    @abstractmethod
    def formula(self) -> str:
        """Str: Chemical formula of gas."""

    def gas_density(
        self, temperature: Union[np.ndarray, float] = 273.15, pressure: Union[np.ndarray, float] = 101.325
    ) -> np.ndarray:
        """Calculating the density of the gas.

        Calculating the density of the gas given temperature and pressure if temperature and pressure are not provided
        we use Standard Temperature and Pressure (STP).

        https://en.wikipedia.org/wiki/Ideal_gas_law

        Args:
            temperature (Union[np.ndarray, float], optional): Array of temperatures [Kelvin],
                defaults to 273.15 [K]
            pressure (Union[np.ndarray, float], optional): Array of pressures [kPa],
                defaults to 101.325 [kPa]

        Returns:
             density (np.ndarray): Array of gas density values [kg/m^3]

        """
        specific_gas_constant = self.__molar_gas_constant / self.molar_mass
        density = np.divide(pressure, (temperature * specific_gas_constant))
        return density

    def convert_emission_m3s_to_kghr(
        self,
        emission_m3s: Union[np.ndarray, float],
        temperature: Union[np.ndarray, float] = 273.15,
        pressure: Union[np.ndarray, float] = 101.325,
    ) -> np.ndarray:
        """Converting emission rates from m^3/s to kg/hr given temperature and pressure.

         If temperature and pressure are not provided we use Standard Temperature and Pressure (STP).

        Args:
            emission_m3s (Union[np.ndarray, float]): Array of emission rates [m^3/s]
            temperature (Union[np.ndarray, float], optional): Array of temperatures [Kelvin],
                defaults to 273.15 [K]
            pressure (Union[np.ndarray, float], optional): Array of pressures [kPa],
                defaults to 101.325 [kPa]

        Returns:
             emission_kghr (np.ndarray): [p x 1] array of emission rates in  [kg/hr]

        """
        density = self.gas_density(temperature=temperature, pressure=pressure)
        emission_kghr = np.multiply(emission_m3s, density) * 3600
        return emission_kghr

    def convert_emission_kghr_to_m3s(
        self,
        emission_kghr: Union[np.ndarray, float],
        temperature: Union[np.ndarray, float] = 273.15,
        pressure: Union[np.ndarray, float] = 101.325,
    ) -> np.ndarray:
        """Converting emission rates from  kg/hr to m^3/s given temperature and pressure.

        If temperature and pressure are not provided we use Standard Temperature and Pressure (STP).

        Args:
            emission_kghr (np.ndarray): Array of emission rates in  [kg/hr]
            temperature (Union[np.ndarray, float], optional): Array of temperatures [Kelvin],
                defaults to 273.15 [K]
            pressure (Union[np.ndarray, float], optional): Array of pressures [kPa],
                defaults to 101.325 [kPa]

        Returns:
             emission_m3s (Union[np.ndarray, float]): Array of emission rates [m^3/s]

        """
        density = self.gas_density(temperature=temperature, pressure=pressure)
        emission_m3s = np.divide(emission_kghr, density) / 3600
        return emission_m3s


@dataclass
class CH4(GasSpecies):
    """Defines the properties of CH4."""

    @property
    def name(self):
        """Str: Name of gas."""
        return "Methane"

    @property
    def molar_mass(self):
        """Float: Molar Mass [g/mol]."""
        return 16.04246

    @property
    def formula(self):
        """Str: Chemical formula of gas."""
        return "CH4"

    global_background = 1.85


@dataclass
class C2H6(GasSpecies):
    """Defines the properties of C2H6."""

    @property
    def name(self):
        """Str: Name of gas."""
        return "Ethane"

    @property
    def molar_mass(self):
        """Float: Molar Mass [g/mol]."""
        return 30.06904

    @property
    def formula(self):
        """Str: Chemical formula of gas."""
        return "C2H6"

    global_background = 5e-4


@dataclass
class C3H8(GasSpecies):
    """Defines the properties of C3H8."""

    @property
    def name(self):
        """Str: Name of gas."""
        return "Propane"

    @property
    def molar_mass(self):
        """Float: Molar Mass [g/mol]."""
        return 46.0055

    @property
    def formula(self):
        """Str: Chemical formula of gas."""
        return "C3H8"

    global_background = 5e-4


@dataclass
class CO2(GasSpecies):
    """Defines the properties of CO2."""

    @property
    def name(self):
        """Str: Name of gas."""
        return "Carbon Dioxide"

    @property
    def molar_mass(self):
        """Float: Molar Mass [g/mol]."""
        return 44.0095

    @property
    def formula(self):
        """Str: Chemical formula of gas."""
        return "CO2"

    global_background = 400


@dataclass
class NO2(GasSpecies):
    """Defines the properties of NO2."""

    @property
    def name(self):
        """Str: Name of gas."""
        return "Nitrogen Dioxide"

    @property
    def molar_mass(self):
        """Float: Molar Mass [g/mol]."""
        return 46.0055

    @property
    def formula(self):
        """Str: Chemical formula of gas."""
        return "NO2"

    global_background = 0
    half_life = 12
