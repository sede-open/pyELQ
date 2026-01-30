# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the source model."""

from copy import deepcopy

import numpy as np
import pytest
from openmcmc import parameter
from openmcmc.distribution.distribution import Gamma, Poisson, Uniform
from openmcmc.distribution.location_scale import Normal as mcmcNormal
from openmcmc.sampler.sampler import NormalNormal

from pyelq.component.source_model import Normal, NormalSlabAndSpike, SourceModel
from tests.conftest import initialise_sampler


@pytest.fixture(
    name="distribution_number_sources_prior",
    params=[Uniform, Poisson],
    ids=lambda p: p.__name__,
)
def fix_distribution_number_sources(request):
    """Set up the distribution class and id for number of sources prior."""
    distribution_class = request.param
    distribution_id = distribution_class.__name__
    return distribution_class, distribution_id


@pytest.fixture(
    name="source_model",
    params=[(Normal, False), (NormalSlabAndSpike, False), (Normal, True), (NormalSlabAndSpike, True)],
    ids=["Normal", "Normal_SlabAndSpike", "Normal_RJ", "Normal_SlabAndSpike_RJ"],
)
def fix_source_model(
    request, sensor_group, met_group, gas_species, dispersion_model, site_limits, distribution_number_sources_prior
):
    """Set up the source model based on all previous fixtures."""
    call_fun, rj_flag = request.param
    _, distribution_number_sources_prior_id = distribution_number_sources_prior
    source_model = call_fun()
    source_model.dispersion_model = dispersion_model
    source_model.update_precision = True
    source_model.reversible_jump = rj_flag
    source_model.site_limits = site_limits
    source_model.distribution_number_sources = distribution_number_sources_prior_id
    source_model.initialise(sensor_object=sensor_group, meteorology=met_group, gas_species=gas_species)
    return source_model


@pytest.fixture(name="fix_coupling_matrix")
def fix_coupling_matrix(monkeypatch):
    """Mock source_model.update_coupling_column to simply return a column of ones."""

    def mock_update_coupling_column(self, state, update_column):
        state["A"][:, update_column] = np.ones((state["A"].shape[0],))
        return state

    monkeypatch.setattr(SourceModel, "update_coupling_column", mock_update_coupling_column)


def test_make_state(source_model, sensor_group):
    """Test the make_state() function.

    Tests that the parameters stored in the state have the correct sizes based on the inputs.

    """
    state = source_model.make_state(state={})
    n_obs = sensor_group.nof_observations
    n_source = source_model.dispersion_model.source_map.nof_sources

    assert state["A"].shape == (n_obs, n_source)
    assert state["s"].shape == (n_source, 1)
    assert state["alloc_s"].shape == (n_source, 1)
    if isinstance(source_model, NormalSlabAndSpike):
        prior_param_shape = (2, 1)
    elif isinstance(source_model, Normal):
        prior_param_shape = (1,)
    else:
        prior_param_shape = None
    assert state["lambda_s"].shape == prior_param_shape
    assert state["mu_s"].shape == prior_param_shape


def test_make_model(source_model, distribution_number_sources_prior):
    """Test the make_model() function.

    Tests that:
        - The source distribution is of type openmcmc.distribution.location_scale.Normal.
        - The mean and precision parameters are of the expected types.
        - If source_model.update_precision is True, then checks that a Gamma distribution for lambda_s is created as
            expected.

    """
    model = source_model.make_model(model=[])
    distribution_number_sources_prior_class, _ = distribution_number_sources_prior

    if isinstance(source_model, Normal):
        assert model[0].response == "s"
        assert isinstance(model[0], mcmcNormal)
        assert isinstance(model[0].mean, parameter.MixtureParameterVector)
        assert isinstance(model[0].precision, parameter.MixtureParameterMatrix)
    elif isinstance(source_model, NormalSlabAndSpike):
        assert model[1].response == "s"
        assert isinstance(model[1], mcmcNormal)
        assert isinstance(model[1].mean, parameter.MixtureParameterVector)
        assert isinstance(model[1].precision, parameter.MixtureParameterMatrix)
    if source_model.update_precision:
        if source_model.reversible_jump:
            assert model[-3].response == "lambda_s"
            assert isinstance(model[-3], Gamma)
            assert isinstance(model[-1], distribution_number_sources_prior_class)
        else:
            assert model[-1].response == "lambda_s"
            assert isinstance(model[-1], Gamma)


def test_make_sampler(source_model):
    """Test the construction of the sampler."""
    sampler_object = initialise_sampler(source_model)
    assert isinstance(sampler_object[0], NormalNormal)


def test_coverage_function(source_model):
    """Test that the coverage function has defaulted correctly."""
    rng = np.random.default_rng(42)
    random_vars = rng.normal(0, 1, size=(10000, 1))
    threshold_value = source_model.threshold_function(random_vars)
    assert threshold_value.shape == (1,)
    assert np.allclose(threshold_value, np.quantile(random_vars, 0.95))


def test_birth_function(source_model):
    """Test the birth_function implementation, and some aspects of the reversible jump sampler.

    Runs source_model.birth_function on the initialised state and
    checks the following:
        1. That the coupling matrix in the proposed state has one
            additional column.
        2. That the new coupling column has been appended to the
            right-hand side of the matrix.
        3. That log(p(current|proposed)) is 0 after this step.

    Following this, runs ReversibleJump.matched_birth_transition and
    checks the following properties of the result:
        4. That there is one extra element appended to the source
            vector in the state.
        5. That the existing elements of the source vector in the
            state remain unchanged.

    """
    rng = np.random.default_rng(42)
    if not source_model.reversible_jump:
        return
    current_state = source_model.make_state(state={})
    current_state["A"] = rng.random(size=current_state["A"].shape)

    prop_state = deepcopy(current_state)
    prop_state["n_src"] = current_state["n_src"] + 1
    prop_state["z_src"] = np.concatenate((prop_state["z_src"], np.zeros((3, 1))), axis=1)

    prop_state, logp_pr_g_cr, logp_cr_g_pr = source_model.birth_function(current_state, prop_state)

    assert prop_state["A"].shape[1] == (current_state["A"].shape[1] + 1)
    assert np.allclose(prop_state["A"][:, :-1], current_state["A"])
    assert logp_cr_g_pr == 0

    sampler_object = initialise_sampler(source_model)
    prop_state, logp_pr_g_cr, logp_cr_g_pr = sampler_object[-1].matched_birth_transition(
        current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr
    )

    assert prop_state["s"].shape[0] == (current_state["s"].shape[0] + 1)
    assert np.allclose(prop_state["s"][:-1], current_state["s"])


def test_death_function(source_model):
    """Test the death_function implementation, and some aspects of the reversible jump sampler.

    Performs the equivalent checks as in the birth case, adapted for the death move.

    """
    rng = np.random.default_rng(42)
    if not source_model.reversible_jump:
        return
    current_state = source_model.make_state(state={})
    if current_state["n_src"] == 0:
        return
    current_state["A"] = rng.random(size=current_state["A"].shape)

    prop_state = deepcopy(current_state)
    prop_state["n_src"] = current_state["n_src"] - 1
    deletion_index = rng.integers(low=0, high=current_state["n_src"].item())
    prop_state["z_src"] = np.delete(prop_state["z_src"], obj=deletion_index, axis=1)

    prop_state, logp_pr_g_cr, logp_cr_g_pr = source_model.death_function(current_state, prop_state, deletion_index)

    assert prop_state["A"].shape[1] == (current_state["A"].shape[1] - 1)
    assert np.allclose(prop_state["A"], np.delete(current_state["A"], obj=deletion_index, axis=1))
    assert logp_pr_g_cr == 0

    sampler_object = initialise_sampler(source_model)
    prop_state, logp_pr_g_cr, logp_cr_g_pr = sampler_object[-1].matched_death_transition(
        current_state, prop_state, logp_pr_g_cr, logp_cr_g_pr, deletion_index
    )

    assert prop_state["s"].shape[0] == (current_state["s"].shape[0] - 1)
    assert np.allclose(np.delete(current_state["s"], obj=deletion_index, axis=0), prop_state["s"])


def test_move_function(source_model):
    """Test the move_function, which updates the coupling matrix after a source is relocated by the sampler.

    The source_model.update_coupling_function is mocked so that it always
    returns a column of ones.

    Checks the following:
        1. That the size of the coupling matrix is the same before and
            after the move.
        2. That the other elements of the coupling matrix are unchanged
            by the move_function.
        3. That the column of the coupling matrix corresponding to the
            relocated source has changed

    """
    rng = np.random.default_rng(42)
    if not source_model.reversible_jump:
        return
    current_state = source_model.make_state(state={})
    if current_state["n_src"] == 0:
        return
    current_state["A"] = rng.random(size=current_state["A"].shape)

    prop_state = deepcopy(current_state)
    move_index = rng.integers(low=0, high=current_state["n_src"].item())
    prop_state["z_src"][:, move_index] = np.zeros((3,))
    prop_state, _, _ = source_model.move_function(prop_state, update_column=move_index)

    assert prop_state["A"].shape == current_state["A"].shape
    assert np.allclose(
        np.delete(current_state["A"], obj=move_index, axis=1), np.delete(prop_state["A"], obj=move_index, axis=1)
    )
    assert np.logical_not(np.allclose(current_state["A"][:, move_index], prop_state["A"][:, move_index]))


def test_compute_coverage(source_model):
    """Test to check whether the compute coverage function can correctly determine which sources are, or are not, within
    the coverage.

    We define some coupling where there are two sources, and one source is coupled half of the time. We then check that
    all the inputs work as intended.

    """
    test_source_model = deepcopy(source_model)
    test_source_model.coverage_detection = 1

    test_source_model.sensor_object["device_0"].source_on = np.ones((4,), dtype=bool)
    if test_source_model.sensor_object.nof_sensors > 1:
        test_source_model.sensor_object.pop("device_1")
        test_source_model.sensor_object.pop("device_2")

    couplings = np.array(
        [
            [1, 0],
            [0, 0],
            [0, 0],
            [1, 0],
        ]
    )

    test_source_model.threshold_function = lambda x: np.quantile(x, 0.95, axis=0)
    coverage = test_source_model.compute_coverage(couplings)
    assert np.all(np.equal(coverage, np.array([True, False])))

    test_source_model.coverage_test_source = 0.3
    coverage = test_source_model.compute_coverage(couplings)
    assert np.all(np.equal(coverage, np.array([False, False])))

    test_source_model.threshold_function = lambda x: np.mean(x, axis=0)
    coverage = test_source_model.compute_coverage(couplings)
    assert np.all(np.equal(coverage, np.array([False, False])))

    test_source_model.coverage_test_source = 6
    coverage = test_source_model.compute_coverage(couplings)
    assert np.all(np.equal(coverage, np.array([True, False])))
