# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the error model."""

import numpy as np
import pytest
from openmcmc import parameter
from openmcmc.distribution.distribution import Gamma
from openmcmc.distribution.location_scale import Normal
from openmcmc.model import Model as mcmcModel
from openmcmc.sampler.sampler import NormalGamma
from scipy import sparse

from pyelq.component.error_model import ByRelease, BySensor, ErrorModel
from pyelq.sensor.sensor import SensorGroup


@pytest.fixture(name="error_model", params=[BySensor, ByRelease], ids=["BySensor", "ByRelease"])
def fix_error_model(request, sensor_group, met_group, gas_species):
    """Set up the error model based on the sensor, met and gas fixtures."""
    call_fun = request.param
    error_model = call_fun()
    error_model.initialise(sensor_object=sensor_group, meteorology=met_group, gas_species=gas_species)
    return error_model


def test_make_state(error_model: ErrorModel, sensor_group: SensorGroup):
    """Test the initialisation of the state.

    Tests:
        1) that the parameters are the correct size for the particular type of error model.
        2) that the assignment index has one element for every observation.
        3) that every precision parameter in the set is used at least once.

    """
    state = error_model.make_state(state=None)
    n_obs = sensor_group.nof_observations
    n_sensor = len(sensor_group)
    if isinstance(error_model, BySensor):
        n_param = n_sensor
    elif isinstance(error_model, ByRelease):
        n_param = 2 * n_sensor
    else:
        raise TypeError("Unknown error model type.")

    assert state["tau"].shape == (n_param,)
    assert state["a_tau"].shape == (n_param,)
    assert state["b_tau"].shape == (n_param,)
    assert state["precision_index"].shape == (n_obs,)
    if n_obs > n_sensor:
        assert np.all(np.isin(state["precision_index"], np.arange(n_param)))


def test_make_model(error_model: ErrorModel):
    """Test the construction of the model object.

    Tests:
        1) that a Gamma distribution is added to the model with the correct parameters
        2) that the predictor for the data distribution has the correct form.

    """
    model = error_model.make_model(model=None)
    assert isinstance(model[0], Gamma)
    assert model[0].response == "tau"

    assert isinstance(error_model.precision_parameter, parameter.MixtureParameterMatrix)


def test_precision_predictor(error_model: ErrorModel):
    """Test that the precision predictor gives expected values.

    Tests:
        1) that the precision predictor gives a sparse diagonal matrix with expected values on the diagonal.
        2) that when we assign parameter index to state["tau"], we recover the precision index on the diagonal.

    """
    state = error_model.make_state(state=None)
    state["tau"] = np.arange(state["tau"].shape[0], dtype=float)
    precision_matrix = error_model.precision_parameter.predictor(state)

    assert sparse.issparse(precision_matrix)
    assert np.array_equal(precision_matrix.toarray(), np.diag(np.diag(precision_matrix.toarray())))

    assert np.allclose(np.diag(precision_matrix.toarray()), error_model.precision_index)


def test_make_sampler(error_model: ErrorModel):
    """Test the construction of the sampler object.

    Tests:
        1) that the sampler object created for the precisions is of NormalGamma type.

    """
    model = [Normal(response="y", mean="mu", precision=error_model.precision_parameter)]
    model = error_model.make_model(model=model)
    sampler_object = error_model.make_sampler(model=mcmcModel(model), sampler_list=None)
    assert isinstance(sampler_object[0], NormalGamma)
