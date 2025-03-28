# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the main ELQModel class."""

from copy import deepcopy

import pytest

from pyelq.component.background import SpatioTemporalBackground
from pyelq.component.error_model import BySensor
from pyelq.component.offset import PerSensor
from pyelq.component.source_model import Normal, NormalSlabAndSpike
from pyelq.model import ELQModel


@pytest.fixture(name="model_default")
def fix_model_default(sensor_group, met_group, gas_species):
    """Fix an instance of model based on the sensor, meteorology and gas species provided."""
    model = ELQModel(sensor_object=sensor_group, meteorology=met_group, gas_species=gas_species)
    return model


@pytest.fixture(params=[None, SpatioTemporalBackground], ids=["none", "spt"], name="background_model")
def fix_background_model(request):
    """Fix a particular type of background model."""
    background_model = request.param
    if background_model is None:
        return None
    return background_model()


@pytest.fixture(
    params=[
        None,
        Normal(),
        NormalSlabAndSpike(),
        Normal(label_string="fixed"),
        [Normal(), Normal(label_string="fixed")],
    ],
    ids=["none", "normal", "normal-ssp", "normal_label", "source_model_list"],
    name="source_model",
)
def fix_source_model(request):
    """Fix a particular type of source model."""
    return request.param


@pytest.fixture(params=[None, PerSensor], ids=["none", "per-sns"], name="offset_model")
def fix_offset_model(request):
    """Fix a particular type of offset model."""
    offset_model = request.param
    if offset_model is None:
        return None
    return offset_model()


@pytest.fixture(params=[None, BySensor], ids=["none", "by-sns"], name="error_model")
def fix_error_model(request):
    """Fix a particular type of error model.

    We make sure we don't pass None to the model, as this will raise a UserWarning, instead we set it to the default
    BySensor model.

    """
    error_model = request.param
    if error_model is None:
        return BySensor()
    return error_model()


@pytest.fixture(name="model")
def fix_model(sensor_group, met_group, gas_species, background_model, source_model, error_model, offset_model):
    """Create the ELQModel object using the data/model specifications."""
    if background_model is not None:
        background_model.update_precision = True
    if offset_model is not None:
        offset_model.update_precision = True
    if source_model is not None:

        if isinstance(source_model, list):
            for source_model_i in source_model:
                source_model_i.update_precision = True
        else:
            source_model.update_precision = True
    model = ELQModel(
        sensor_object=sensor_group,
        meteorology=met_group,
        gas_species=gas_species,
        background=background_model,
        source_model=source_model,
        error_model=error_model,
        offset_model=offset_model,
    )
    model.initialise()
    return model


def test_default(model_default):
    """Test whether the default ELQModel case will initialise (with default component settings)."""
    model_default.initialise()
    model_default.n_iter = 5
    model_default.to_mcmc()
    model_default.run_mcmc()
    model_default.from_mcmc()


def test_run_mcmc(model):
    """Test running a small number of iterations of the MCMC."""
    model.n_iter = 5
    model.to_mcmc()
    model.run_mcmc()
    model.from_mcmc()


def test_mcmc_iterations(model):
    """Run one iteration of the MCMC for each of the samplers on the model, and check that the variables stored in the
    results dictionary are of the shape expected."""
    model.n_iter = 1
    model.to_mcmc()
    original_state = deepcopy(model.mcmc.state)
    model.run_mcmc()
    for var in model.mcmc.state.keys():
        assert model.mcmc.state[var].shape == original_state[var].shape
