# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for background models."""

import numpy as np
import pytest
from openmcmc.distribution.distribution import Gamma
from openmcmc.distribution.location_scale import Normal
from openmcmc.model import Model as mcmcModel
from openmcmc.sampler.sampler import NormalGamma, NormalNormal
from scipy import sparse

from pyelq.component.background import SpatioTemporalBackground, TemporalBackground
from tests.conftest import initialise_sampler


@pytest.fixture(
    name="background_model",
    params=[TemporalBackground(), SpatioTemporalBackground()],
    ids=["Temporal", "Spatiotemporal"],
)
def fix_background_model(request, sensor_group, met_group, gas_species):
    """Fix the background models to be tested."""
    background_model = request.param
    background_model.update_precision = True
    background_model.initialise(sensor_object=sensor_group, meteorology=met_group, gas_species=gas_species)
    if isinstance(background_model, SpatioTemporalBackground):
        background_model.spatial_dependence = True
    return background_model


def test_background_init(background_model):
    """Check that the background object initialises with properties that make sense."""
    assert np.allclose(np.sum(background_model.basis_matrix, axis=1), np.ones(background_model.n_obs))


def test_make_state(background_model, sensor_group):
    """Check that the state is constructed properly."""
    state = background_model.make_state(state={})
    n_param = background_model.n_parameter
    n_obs = sensor_group.nof_observations

    assert state["B_bg"].shape == (n_obs, n_param)
    assert sparse.issparse(state["B_bg"])

    assert state["P_bg"].shape == (n_param, n_param)
    if n_param > 1:
        assert sparse.issparse(state["P_bg"])
    else:
        assert isinstance(state["P_bg"], np.ndarray)

    assert state["bg"].shape == (n_param, 1)
    assert state["mu_bg"].shape == (n_param, 1)


def test_make_model(background_model):
    """Check that the model is constructed as expected."""
    model = mcmcModel(background_model.make_model(model=[]))

    assert isinstance(model["bg"], Normal)
    if background_model.update_precision:
        assert isinstance(model["lambda_bg"], Gamma)


def test_make_sampler(background_model):
    """Check that the sampler is constructed as expected."""
    sampler_object = initialise_sampler(background_model)
    assert isinstance(sampler_object[0], NormalNormal)
    if background_model.update_precision:
        assert isinstance(sampler_object[1], NormalGamma)
