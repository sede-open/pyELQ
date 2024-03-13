# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Generic fixtures that can be used for any component tests."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from openmcmc.model import Model

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.gas_species import CH4
from pyelq.meteorology import Meteorology, MeteorologyGroup
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap


@pytest.fixture(name="ref_longitude", scope="module")
def fix_ref_longitude():
    """Fix the reference longitude to be used in the tests."""
    return 0.0


@pytest.fixture(name="ref_latitude", scope="module")
def fix_ref_latitude():
    """Fix the reference latitude to be used in the tests."""
    return 0.0


@pytest.fixture(name="site_limits", scope="module")
def fix_site_limits():
    """Fix the site limits to be used in the tests."""
    return np.array([[-100, 100], [-100, 100], [0, 5]])


@pytest.fixture(
    params=[(1, 1), (1, 3), (100, 1), (100, 3)],
    ids=["1_1", "1_3", "100_1", "100_3"],
    name="sensor_group",
    scope="module",
)
def fix_sensor_group(request, ref_longitude, ref_latitude):
    """Create sensor fixture.

    We add n_sensor-1 sensors to the sensor group, and one Beam sensor to make sure we cover both cases.

    """
    [n_time, n_sensor] = request.param
    locations = np.concatenate(
        (100 * np.random.random_sample(size=(n_sensor, 1)), 100 * np.random.random_sample(size=(n_sensor, 1))), axis=1
    )

    sensor = SensorGroup()
    for k in range(n_sensor - 1):
        device_name = "device_" + str(k)
        sensor[device_name] = Sensor()
        sensor[device_name].time = pd.array(
            pd.date_range(start=datetime.now(), end=datetime.now() + timedelta(hours=1.0), periods=n_time),
            dtype='datetime64[ns]')
        sensor[device_name].concentration = np.random.random_sample(size=(n_time,))
        sensor[device_name].location = ENU(
            east=locations[k, 0],
            north=locations[k, 1],
            up=5.0,
            ref_longitude=ref_longitude,
            ref_latitude=ref_latitude,
            ref_altitude=0.0,
        ).to_lla()
        sensor[device_name].source_on = np.random.choice(a=[False, True], size=(n_time,), p=[0.5, 0.5])

    k = n_sensor - 1
    device_name = "device_" + str(k)
    sensor[device_name] = Beam()
    sensor[device_name].time = pd.array(
        pd.date_range(start=datetime.now(), end=datetime.now() + timedelta(hours=1.0), periods=n_time),
        dtype='datetime64[ns]')
    sensor[device_name].concentration = np.random.random_sample(size=(n_time,))
    sensor[device_name].location = ENU(
        east=np.array([0, locations[k, 0]]),
        north=np.array([0, locations[k, 1]]),
        up=np.array([5.0, 5.0]),
        ref_longitude=ref_longitude,
        ref_latitude=ref_latitude,
        ref_altitude=0.0,
    ).to_lla()
    sensor[device_name].source_on = np.random.choice(a=[False, True], size=(n_time,), p=[0.5, 0.5])
    return sensor


@pytest.fixture(name="met_group", scope="module")
def fix_met_group(sensor_group):
    """Create meteorology fixture."""
    met_group = MeteorologyGroup()
    for name, sns in sensor_group.items():
        met_group[name] = Meteorology()
        met_group[name].time = sns.time
        met_group[name].wind_speed = 2.0 + 3.0 * np.random.random_sample(size=met_group[name].time.shape)
        met_group[name].wind_direction = 360.0 * np.random.random_sample(size=met_group[name].time.shape)
        met_group[name].wind_turbulence_horizontal = 10.0 * np.ones(shape=met_group[name].time.shape)
        met_group[name].wind_turbulence_vertical = 10.0 * np.ones(shape=met_group[name].time.shape)
        met_group[name].temperature = 293.0 * np.ones(shape=met_group[name].time.shape)
        met_group[name].pressure = 101.0 * np.ones(shape=met_group[name].time.shape)
        met_group[name].calculate_uv_from_wind_speed_direction()
    return met_group


@pytest.fixture(name="gas_species", scope="module")
def fix_gas_species():
    """Create gas species fixture."""
    return CH4()


@pytest.fixture(name="dispersion_model", scope="module")
def fix_dispersion_model(ref_longitude, ref_latitude, site_limits):
    """Set up the dispersion model."""
    source_map = SourceMap()
    coordinate_object = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=0.0)
    source_map.generate_sources(
        coordinate_object=coordinate_object, sourcemap_limits=site_limits, sourcemap_type="hypercube"
    )
    dispersion_model = GaussianPlume(source_map=source_map)
    return dispersion_model


def initialise_sampler(component):
    """Helper function to initialise the sampler for any given component."""
    model = component.make_model(model=[])
    sampler_object = component.make_sampler(model=Model(model), sampler_list=None)
    return sampler_object
