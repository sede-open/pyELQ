# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for gaussian plume module.

This module provides various tests for the Gaussian plume related code part of pyELQ

"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.gas_species import CH4
from pyelq.meteorology.meteorology import Meteorology, MeteorologyGroup
from pyelq.sensor.beam import Beam
from pyelq.sensor.satellite import Satellite
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap


def make_met_object(location):
    """Function to create a meteorology object for testing purposes."""

    rng = np.random.default_rng(42)
    time = pd.date_range(
        pd.Timestamp.fromisoformat("2022-01-01 00:00:00"), periods=location.nof_observations, freq="s"
    ).array[:, None]
    met_object = Meteorology()
    met_object.location = location
    met_object.time = time
    met_object.u_component = rng.integers(low=1, high=5, size=time.shape)
    met_object.v_component = rng.integers(low=1, high=5, size=time.shape)
    met_object.calculate_wind_direction_from_uv()
    met_object.calculate_wind_speed_from_uv()
    met_object.temperature = rng.integers(low=270, high=275, size=time.shape)
    met_object.pressure = rng.integers(low=99, high=103, size=time.shape)
    met_object.wind_turbulence_horizontal = 5 + 10 * rng.random(size=time.shape)
    met_object.wind_turbulence_vertical = 5 + 10 * rng.random(size=time.shape)

    return met_object


@pytest.fixture(name="met_object")
def fixture_met_object():
    """Fixture to define a meteorology object."""
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    location.from_array(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

    met_object = make_met_object(location)

    return met_object


@pytest.fixture(name="met_object_single")
def fixture_met_object_single():
    """Fixture to define a meteorology object with a single observation."""
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    location.from_array(np.array([[0, 0, 0]]))
    met_object = make_met_object(location)
    return met_object


@pytest.fixture(name="sensor_object")
def fixture_sensor_object():
    """Fixture to define a generic sensor object."""
    sensor_object = Sensor()
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    location.from_array(np.array([[25, 0, 0]]))
    sensor_object.location = location
    time = pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:00"), periods=5, freq="ns").array[:, None]
    sensor_object.time = time
    sensor_object.concentration = np.zeros(time.size)
    sensor_object.label = "Generic"
    return sensor_object


@pytest.fixture(name="drone_object")
def fixture_drone_object():
    """Fixture to define a drone sensor object."""
    sensor_object = Sensor()
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    loc_in = np.array([[0, 50, 0], [25, 25, 0], [50, 0, 0]])
    location.from_array(loc_in)
    sensor_object.location = location
    time = pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:00"), periods=loc_in.shape[0], freq="s").array[
        :, None
    ]
    sensor_object.time = time
    sensor_object.concentration = np.zeros(time.size)
    sensor_object.label = "Generic"
    return sensor_object


@pytest.fixture(name="beam_object")
def fixture_beam_object():
    """Fixture to define a beam sensor object."""
    beam_location = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    beam_location.from_array(np.array([[24.99, 0, 0], [25.01, 0, 0]]))
    beam_object = Beam()
    beam_object.location = beam_location
    time = pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:00"), periods=4, freq="ns").array[:, None]
    beam_object.time = time
    beam_object.concentration = np.zeros(time.size)
    beam_object.label = "Beam"
    return beam_object


@pytest.fixture(name="satellite_object")
def fixture_satellite_object():
    """Fixture to define a satellite sensor object."""
    satellite_location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    temp_array = np.array(
        [[-25, 25, 0], [0, 25, 0], [25, 25, 0], [25, 0, 0], [25, -25, 0], [0, -25, 0], [-25, -25, 0], [-25, 0, 0]]
    )
    satellite_location.from_array(temp_array)
    satellite_object = Satellite()
    satellite_object.location = satellite_location
    time = None
    satellite_object.time = time
    satellite_object.concentration = np.zeros(temp_array.shape[0])
    satellite_object.label = "Satellite"
    return satellite_object


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube"])
def test_compute_coupling_array(sourcemap_type):
    """Test to check compute_coupling_array method.

    Tests two configurations:
        1- Places a sensor on the upwind edge of the possible source domain, computes
            the couplings to randomly-generated sources, and then checks that the coupling values
            are all 0.
        2- Generates a random number of sensor locations, at random downwind locations, then checks
            that the computed coupling array has the correct shape, that all values are >=0, and that
            values for which the raw coupling is less than the minimum contribution have been correctly
            set to 0.

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location, sourcemap_type=sourcemap_type, nof_sources=3, sourcemap_limits=np.array([[-1, 1], [-1, 1], [-1, 1]])
    )
    plume_object = GaussianPlume(source_map=source_object)
    coupling_array = plume_object.compute_coupling_array(
        sensor_x=np.array([-1]),
        sensor_y=np.array([0]),
        sensor_z=np.array([0]),
        source_z=np.array([0]),
        wind_speed=np.array([0]),
        theta=np.array([0]),
        wind_turbulence_horizontal=np.array([5]),
        wind_turbulence_vertical=np.array([5]),
        gas_density=1,
    )
    assert np.all(coupling_array == 0)

    rng = np.random.default_rng(42)
    random_shape = rng.integers(1, 5, 3)
    coupling_array = plume_object.compute_coupling_array(
        sensor_x=rng.random(random_shape) * 6 - 3,
        sensor_y=rng.random(random_shape) * 6 - 3,
        sensor_z=rng.random(random_shape) * 6 - 3,
        source_z=np.array(0),
        wind_speed=rng.integers(1, 5),
        theta=rng.random(1) * 2 * np.pi,
        wind_turbulence_horizontal=np.array([5]),
        wind_turbulence_vertical=np.array([5]),
        gas_density=1,
    )
    assert np.all(coupling_array.shape == random_shape)
    assert np.all(coupling_array >= 0)
    assert not np.any(np.logical_and(coupling_array > 0, coupling_array < plume_object.minimum_contribution))


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube"])
@pytest.mark.parametrize("coordinate_type", ["ENU", "LLA"])
def test_compute_coupling_single_sensor_non_satellite(
    sourcemap_type, coordinate_type, met_object, sensor_object, beam_object, met_object_single, drone_object
):
    """Test to check compute_coupling_single_sensor method for a non satellite sensor.

    Coordinate_type is varied from ENU to LLA to check the way coupling handles the coordinate conversions.

    Takes in a generic sensor object and a beam sensor object. The observation locations for the generic sensor object
    are defined at the mid-points of the beam sensors. Performs the following tests:     1- Checks the shape of the
    coupling array produced (in both the general sensor and beam cases).     2- Compares the beam sensor coupling values
    with point sensor values evaluated at the        midpoints of the beams, checks that they are close (for the first
    four time points).     3- Checks that inputting a sensor with multiple location values and a single meteorological
    observation gives        the correct shape, and calculates the right values used to check the drone type sensor
    case.

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location,
        sourcemap_type=sourcemap_type,
        nof_sources=3,
        sourcemap_limits=np.array([[-100, 100], [-100, 100], [-100, 100]]),
    )

    if coordinate_type == "LLA":
        source_object.location = source_object.location.to_lla()

    plume_object = GaussianPlume(source_map=source_object)
    plume_object.minimum_contribution = 1e-15

    coupling = plume_object.compute_coupling_single_sensor(
        sensor_object=sensor_object, meteorology=met_object, gas_object=None
    )
    assert coupling.shape == (sensor_object.nof_observations, source_object.nof_sources)

    coupling_beam = plume_object.compute_coupling_single_sensor(
        sensor_object=beam_object, meteorology=met_object, gas_object=None
    )

    assert coupling_beam.shape == (beam_object.nof_observations, source_object.nof_sources)
    assert np.allclose(coupling_beam, coupling[:-1, :])

    coupling_drone = plume_object.compute_coupling_single_sensor(
        sensor_object=drone_object, meteorology=met_object_single, gas_object=None
    )

    assert coupling_drone.shape == (drone_object.nof_observations, source_object.nof_sources)

    drone_location = drone_object.location.to_array()
    for idx in range(drone_object.nof_observations):
        drone_object_single = deepcopy(drone_object)

        drone_object_single.time = drone_object.time[idx, :]
        drone_object_single.concentration = drone_object.concentration[[idx]]
        drone_object_single.location.from_array(drone_location[[idx], :])

        coupling_drone_single = plume_object.compute_coupling_single_sensor(
            sensor_object=drone_object_single, meteorology=met_object_single, gas_object=None
        )
        assert np.allclose(coupling_drone[idx, :], coupling_drone_single)


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube"])
@pytest.mark.parametrize("inclusion_radius", [26, 1])
def test_compute_coupling_single_sensor_satellite(sourcemap_type, inclusion_radius, met_object, satellite_object):
    """Test to check compute_coupling_single_sensor method for a satellite sensor.

    Checks for consistent shapes, i.e. length of list output and shapes of array with and without usage of
    inclusion_idx.

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location,
        sourcemap_type=sourcemap_type,
        nof_sources=3,
        sourcemap_limits=np.array([[-100, 100], [-100, 100], [-100, 100]]),
    )

    plume_object = GaussianPlume(source_map=source_object)
    plume_object.minimum_contribution = 1e-15

    coupling_sat = plume_object.compute_coupling_single_sensor(
        sensor_object=satellite_object, meteorology=met_object, gas_object=None
    )

    assert len(coupling_sat) == source_object.nof_sources
    assert np.all([value.shape == (satellite_object.nof_observations, 1) for value in coupling_sat])

    source_object.calculate_inclusion_idx(sensor_object=satellite_object, inclusion_radius=inclusion_radius)
    coupling_sat_subset = plume_object.compute_coupling_single_sensor(
        sensor_object=satellite_object, meteorology=met_object, gas_object=None
    )
    assert len(coupling_sat_subset) == source_object.nof_sources
    assert np.all(
        [
            coupling_sat_subset[value].size == source_object.inclusion_n_obs[value]
            for value in range(source_object.nof_sources)
        ]
    )


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube"])
def test_coupling_non_stp(sourcemap_type, met_object, sensor_object):
    """Test to check if magnitude of coupling value changes correctly when using a non-Standard Pressure and
    Temperature.

    Performs two tests:
        1- The temperature is decreased to 100 degrees below the default: we check that the coupling
            values all decrease as expected (due to higher density).
        2- The pressure is decreased to 80 kPa below the default: we check that the coupling values
            all increase as expected (due to lower density).

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location,
        sourcemap_type=sourcemap_type,
        nof_sources=3,
        sourcemap_limits=np.array([[-100, 100], [-100, 100], [-100, 100]]),
    )

    plume_object = GaussianPlume(source_map=source_object)
    plume_object.minimum_contribution = 1e-15

    gas_species = CH4()
    met_object.temperature = np.array(273.15)
    met_object.pressure = np.array(101.325)
    coupling_stp = plume_object.compute_coupling_single_sensor(
        sensor_object=sensor_object, meteorology=met_object, gas_object=gas_species
    )

    met_object.temperature = np.array(273.15 - 100)
    met_object.pressure = None
    coupling_low_temp = plume_object.compute_coupling_single_sensor(
        sensor_object=sensor_object, meteorology=met_object, gas_object=gas_species
    )
    assert np.all(coupling_stp >= coupling_low_temp)

    met_object.temperature = None
    met_object.pressure = np.array(101.325 - 80)
    coupling_low_pressure = plume_object.compute_coupling_single_sensor(
        sensor_object=sensor_object, meteorology=met_object, gas_object=gas_species
    )

    assert np.all(coupling_stp <= coupling_low_pressure)


def test_not_implemented_error():
    """Simple test to check if correct error is thrown."""
    with pytest.raises(NotImplementedError):
        plume_object = GaussianPlume(source_map=SourceMap())
        plume_object.compute_coupling_single_sensor(sensor_object=None, meteorology=Meteorology())


def test_compute_coupling(monkeypatch):
    """Test the high level function to see if the return is of the correct type."""

    def mock_coupling(*args, **kwargs):
        """Return an empty array instead of computing the actual coupling."""
        return np.array([])

    monkeypatch.setattr(GaussianPlume, "compute_coupling_single_sensor", mock_coupling)

    plume_object = GaussianPlume(source_map=SourceMap())
    coupling_object = plume_object.compute_coupling(sensor_object=Sensor(), meteorology_object=Meteorology())
    assert isinstance(coupling_object, np.ndarray)

    sensor_group = SensorGroup()
    sensor = Sensor()
    sensor.label = "sensor_1"
    sensor_group.add_sensor(sensor)
    sensor = Sensor()
    sensor.label = "sensor_2"
    sensor_group.add_sensor(sensor)
    coupling_object = plume_object.compute_coupling(sensor_object=sensor_group, meteorology_object=Meteorology())
    assert isinstance(coupling_object, dict)
    assert np.all(coupling_object.keys() == sensor_group.keys())
    assert np.all([isinstance(value, np.ndarray) for value in coupling_object.values()])

    coupling_object = plume_object.compute_coupling(
        sensor_object=sensor_group, meteorology_object=Meteorology(), output_stacked=True
    )
    assert isinstance(coupling_object, np.ndarray)

    object_1 = Meteorology()
    object_1.label = "sensor_1"
    object_2 = Meteorology()
    object_2.label = "sensor_2"
    group_object = MeteorologyGroup()
    group_object.add_object(object_1)
    group_object.add_object(object_2)
    coupling_object = plume_object.compute_coupling(sensor_object=sensor_group, meteorology_object=group_object)
    assert isinstance(coupling_object, dict)
    assert np.all(coupling_object.keys() == sensor_group.keys())
    assert np.all([isinstance(value, np.ndarray) for value in coupling_object.values()])

    with pytest.raises(TypeError):
        plume_object.compute_coupling(sensor_object=None, meteorology_object=Meteorology())

    with pytest.raises(TypeError):
        plume_object.compute_coupling(sensor_object=sensor, meteorology_object=group_object)


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube"])
def test_interpolate_meteorology(sourcemap_type, met_object, sensor_object, satellite_object, beam_object):
    """Test to check interpolate_meteorology method.

    Tests as follows:
        1- For each sensor type, check that the meteorology interpolation returns the correct
            number of values.
        2- Checks that when the horizontal wind turbulence on the met object is a single value, the
            interpolate function correctly returns the same value for all entries.
        3- In the generic sensor case only checks when a field (pressure) is set to None, the
            interpolate function correctly returns None.

    Note that the specific values returned are not checked, it is assumes that this is being tested
    for the underlying interpolation function.

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location,
        sourcemap_type=sourcemap_type,
        nof_sources=3,
        sourcemap_limits=np.array([[-100, 100], [-100, 100], [-100, 100]]),
    )

    plume_object = GaussianPlume(source_map=source_object)

    for temp_sensor in [beam_object, sensor_object]:
        return_values = plume_object.interpolate_meteorology(
            meteorology=met_object, variable_name="u_component", sensor_object=temp_sensor
        )
        assert return_values.shape == (temp_sensor.nof_observations, 1)

        return_values = plume_object.interpolate_meteorology(
            meteorology=met_object, variable_name="wind_turbulence_horizontal", sensor_object=temp_sensor
        )
        assert return_values.shape == (temp_sensor.nof_observations, 1)
        if met_object.wind_turbulence_horizontal.size == 1:
            assert np.all(return_values == met_object.wind_turbulence_horizontal)

    return_values = plume_object.interpolate_meteorology(
        meteorology=met_object, variable_name="u_component", sensor_object=satellite_object
    )
    assert return_values.shape == (1, plume_object.source_map.nof_sources)

    return_values = plume_object.interpolate_meteorology(
        meteorology=met_object, variable_name="wind_turbulence_horizontal", sensor_object=satellite_object
    )
    assert return_values.shape == (1, plume_object.source_map.nof_sources)
    if met_object.wind_turbulence_horizontal.size == 1:
        assert np.all(return_values == met_object.wind_turbulence_horizontal)

    met_object.pressure = None
    return_values = plume_object.interpolate_meteorology(
        meteorology=met_object, variable_name="pressure", sensor_object=sensor_object
    )
    assert return_values is None


def test_interpolate_all_meteorology(met_object, sensor_object):
    """Checks interpolate_all_meteorology for correct output when run_interpolation flag is set to False."""
    plume_object = GaussianPlume(source_map=SourceMap())
    (
        gas_density,
        u_interpolated,
        v_interpolated,
        wind_turbulence_horizontal,
        wind_turbulence_vertical,
    ) = plume_object.interpolate_all_meteorology(
        sensor_object=sensor_object, meteorology=met_object, gas_object=CH4(), run_interpolation=False
    )
    assert np.all(gas_density == CH4().gas_density(temperature=met_object.temperature, pressure=met_object.pressure))
    assert np.all(u_interpolated == met_object.u_component)
    assert np.all(v_interpolated == met_object.v_component)
    assert np.all(wind_turbulence_horizontal == met_object.wind_turbulence_horizontal)
    assert np.all(wind_turbulence_vertical == met_object.wind_turbulence_vertical)


@pytest.mark.parametrize("sourcemap_type", ["central", "hypercube"])
def test_calculate_gas_density(sourcemap_type, met_object, sensor_object, satellite_object, beam_object):
    """Test to check calculate_gas_density method.

    The following tests are performed (all performed for all sensor types):     1- Checks that when the temperature and
    pressure values are fixed for all time (at         standard temperature and pressure), the interpolation returns the
    density at STP for         all times.     2- Checks that the returned vector has the correct shape.     3- Checks
    that when the pressure and temperature are set to None, a vector of ones (of         the correct shape) is returned
    by the interpolate function.

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location,
        sourcemap_type=sourcemap_type,
        nof_sources=3,
        sourcemap_limits=np.array([[-100, 100], [-100, 100], [-100, 100]]),
    )

    met_object.temperature = np.ones_like(met_object.temperature) * 273.15
    met_object.pressure = np.ones_like(met_object.pressure) * 101.325

    gas_species = CH4()
    plume_object = GaussianPlume(source_map=source_object)
    return_values = plume_object.calculate_gas_density(
        meteorology=met_object, sensor_object=sensor_object, gas_object=gas_species
    )
    assert np.all(return_values == gas_species.gas_density())
    assert return_values.shape == (sensor_object.nof_observations, 1)

    return_values = plume_object.calculate_gas_density(
        meteorology=met_object, sensor_object=beam_object, gas_object=gas_species
    )
    assert np.all(return_values == gas_species.gas_density())
    assert return_values.shape == (beam_object.nof_observations, 1)

    return_values = plume_object.calculate_gas_density(
        meteorology=met_object, sensor_object=satellite_object, gas_object=gas_species
    )
    assert np.all(return_values == gas_species.gas_density())
    assert return_values.shape == (1, plume_object.source_map.nof_sources)

    met_object.pressure = None
    met_object.temperature = None
    return_values = plume_object.calculate_gas_density(
        meteorology=met_object, sensor_object=sensor_object, gas_object=None
    )
    assert np.all(return_values == 1)
    assert return_values.shape == (sensor_object.nof_observations, 1)

    return_values = plume_object.calculate_gas_density(
        meteorology=met_object, sensor_object=beam_object, gas_object=None
    )
    assert np.all(return_values == 1)
    assert return_values.shape == (beam_object.nof_observations, 1)

    return_values = plume_object.calculate_gas_density(
        meteorology=met_object, sensor_object=satellite_object, gas_object=None
    )
    assert np.all(return_values == 1)
    assert return_values.shape == (1, plume_object.source_map.nof_sources)


def test_source_on_switch(met_object, sensor_object):
    """Test to check the implementation of the source_on attribute.

    Hence, the coupling should be 0 when the sources_on switch is False for that observation. We check if any value was
    nonzero before using the switch to be sure the switch is the reason the value is 0 after applying it and hence check
    the correct working of the switch.

    """
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    source_object = SourceMap()
    source_object.generate_sources(
        location,
        sourcemap_type="central",
        nof_sources=1,
        sourcemap_limits=np.array([[-100, 100], [-100, 100], [-100, 100]]),
    )

    plume_object = GaussianPlume(source_map=source_object)
    coupling = plume_object.compute_coupling_single_sensor(
        sensor_object=sensor_object, meteorology=met_object, gas_object=None
    )
    change_point = int(np.floor(sensor_object.nof_observations / 2))
    switch = np.ones(sensor_object.nof_observations)
    switch[change_point:] = 0
    sensor_object.source_on = switch.astype(bool)

    coupling_switch = plume_object.compute_coupling_single_sensor(
        sensor_object=sensor_object, meteorology=met_object, gas_object=None
    )

    assert np.all(coupling_switch[change_point:] == 0) and np.any(coupling[change_point:] > 0)
