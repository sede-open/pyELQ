# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the offset model."""

import numpy as np
import pytest
from openmcmc.distribution.location_scale import Normal
from openmcmc.sampler.sampler import NormalNormal
from scipy import sparse

from pyelq.component.offset import PerSensor
from pyelq.sensor.sensor import SensorGroup
from tests.conftest import initialise_sampler


@pytest.fixture(name="offset_model", params=[PerSensor()], ids=["PerSensor"])
def fix_offset_model(request, sensor_group, met_group, gas_species):
    """Set up the specific error model, based on sensor, met and gas fixtures."""
    offset_model = request.param
    offset_model.update_precision = True
    offset_model.initialise(sensor_object=sensor_group, meteorology=met_group, gas_species=gas_species)
    return offset_model


def test_make_state(offset_model: PerSensor, sensor_group: SensorGroup):
    """Test the function which initialises the state.

    Tests:
        1) that the parameters have the correct size.
        2) that the basis and precision matrix are both sparse.
        3) that the allocation basis allocates each parameter to the correct number of observations.

    """
    state = offset_model.make_state(state=None)
    n_obs = sensor_group.nof_observations
    n_sensor = len(sensor_group)
    n_param = n_sensor - 1

    assert state["mu_d"].shape == (n_param, 1)
    assert state["B_d"].shape == (n_obs, n_param)
    assert state["P_d"].shape == (n_param, n_param)
    assert isinstance(state["B_d"], sparse.csc_matrix)
    assert isinstance(state["P_d"], sparse.csc_matrix)
    assert isinstance(state["lambda_d"], float)
    if offset_model.update_precision:
        assert isinstance(state["a_lam_d"], float)
        assert isinstance(state["b_lam_d"], float)

    sum_basis = np.array(np.sum(state["B_d"], axis=0)).flatten()
    for k, sns in enumerate(sensor_group.values()):
        if k > 0:
            assert sns.nof_observations == sum_basis[k - 1]


def test_make_model(offset_model: PerSensor):
    """Test the construction of the model object.

    Tests:
        1) that a normal distribution is added to the model with the correct parameters.

    """
    model = offset_model.make_model(model=None)
    assert isinstance(model[0], Normal)
    assert model[0].response == "d"


def test_make_sampler(offset_model: PerSensor):
    """Test the construction of the sampler object.

    Tests:
        1) that the sampler is a conjugate NormalNormal object.

    """
    sampler_object = initialise_sampler(offset_model)
    assert isinstance(sampler_object[0], NormalNormal)
