# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for gas species superclass.

This module provides tests for the gas species superclass in pyELQ

"""

import numpy as np
import pytest

from pyelq.gas_species import C2H6, C3H8, CH4, CO2, H2, NO2


@pytest.mark.parametrize("gas_species", [CH4, C2H6, C3H8, CO2, NO2, H2])
def test_consistency_emission_rate(gas_species):
    """Basic test to check consistency in gas species methods.

    Checks density conversions to/from kg/hr and m^3 /s to check match up.
    Checks with default (STP) temp and pressure and then with randomly close temp are pressure.

    Args:
        gas_species (pyelq.gas_species): gas species type to check

    """
    rng = np.random.default_rng(42)
    emission_m3s_start = rng.random((5, 1)) * 10
    emission_kghr_start = rng.random((5, 1)) * 1000
    alternate_temperature = 273.15 * (rng.random(1) + 0.5)
    alternate_pressure = 100 * (rng.random(1) + 0.5)
    gas_object = gas_species()

    kghr_intermediate = gas_object.convert_emission_m3s_to_kghr(emission_m3s_start)
    m3s_result = gas_object.convert_emission_kghr_to_m3s(kghr_intermediate)
    assert np.allclose(emission_m3s_start, m3s_result)

    kghr_intermediate = gas_object.convert_emission_m3s_to_kghr(
        emission_m3s_start, temperature=alternate_temperature, pressure=alternate_pressure
    )
    m3s_result = gas_object.convert_emission_kghr_to_m3s(
        kghr_intermediate, temperature=alternate_temperature, pressure=alternate_pressure
    )
    assert np.allclose(emission_m3s_start, m3s_result)

    m3s_intermediate = gas_object.convert_emission_kghr_to_m3s(emission_kghr_start)
    kghr_result = gas_object.convert_emission_m3s_to_kghr(m3s_intermediate)
    assert np.allclose(emission_kghr_start, kghr_result)

    m3s_intermediate = gas_object.convert_emission_kghr_to_m3s(
        emission_kghr_start, temperature=alternate_temperature, pressure=alternate_pressure
    )
    kghr_result = gas_object.convert_emission_m3s_to_kghr(
        m3s_intermediate, temperature=alternate_temperature, pressure=alternate_pressure
    )
    assert np.allclose(emission_kghr_start, kghr_result)


@pytest.mark.parametrize(
    "gas_species, temperature, density",
    [
        (CH4, 293.15, 0.668),
        (CH4, 273.15, 0.717),
        (C2H6, 273.15, 1.3547),
        (C3H8, 293.15, 1.8988),
        (C3H8, 303.15, 1.8316),
        (CO2, 293.15, 1.842),
        (CO2, 273.15, 1.977),
        (NO2, 273.15, 2.05),
        (H2, 273.15, 0.08988),
    ],
)
def test_density_calculation(gas_species, temperature, density):
    """Test density calculation against known values for a set of gases https://www.engineeringtoolbox.com/gas-density-
    d_158.html https://encyclopedia.airliquide.com/ethane#properties
    https://encyclopedia.airliquide.com/propane#properties https://www.thermopedia.com/content/980/

    Assumes atmospheric pressure of 101.325 kPa

    Args:
        gas_species (pyelq.gas_species): gas species type to check
        temperature (float): temperature
        density (float): true density from reference.

    """
    gas_object = gas_species()
    result = gas_object.gas_density(temperature=temperature, pressure=101.325)
    assert np.isclose(result, density, rtol=1e-2)


@pytest.mark.parametrize("gas_species", [CH4, C2H6, C3H8, CO2, NO2, H2])
def test_name_and_formula(gas_species):
    """Test to see if name and formula give back a string output."""
    gas_object = gas_species()
    assert isinstance(gas_object.name, str)
    assert isinstance(gas_object.formula, str)
