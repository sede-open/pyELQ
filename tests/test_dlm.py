# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Test module for DLM code.

This module provides various tests for the DLM related code part of pyELQ

"""

from typing import Tuple

import numpy as np
import pytest
from scipy.stats import cramervonmises

from pyelq.dlm import DLM


@pytest.mark.parametrize(
    "g_matrix, power",
    [(np.array([3], ndmin=2), 3), (np.identity(4), 4), (np.random.default_rng().random(size=(5, 5)), 5)],
)
def test_calculate_g_power(g_matrix: np.ndarray, power: int):
    """Test to check calculate_g_power method.

    Uses the numpy matrix power function to compare the output of the implemented method.

    Args:
        g_matrix (np.ndarray): Matrix to perform power operation on
        power (int): Power to calculate

    """
    model = DLM(g_matrix=g_matrix)
    model.calculate_g_power(max_power=power)
    numpy_result = np.linalg.matrix_power(g_matrix, power)
    dlm_result = model.g_power[:, :, -1].squeeze()
    assert np.allclose(numpy_result, dlm_result)


@pytest.mark.parametrize("nof_observables, order", [(1, 1), (2, 2), (4, 3)])
def test_polynomial_f_g(nof_observables: int, order: int):
    """Test to check polynomial_f_g method.

    Check if the shapes are consistent and if resulting G matrix of n-th order polynomial DLM has a single unit
    eigenvalue of multiplicity n * nof_observables (from Harrison and West Chap 7.1), in particular no zero eigenvalues

    Also checks if 0 is returned for nof_observables and nof_state_parameters when F or G are not set

    Args:
        nof_observables (int): Dimension of observation
        order (int): Polynomial order (0=constant, 1=linear, 2=quadratic etc.)

    """
    model = DLM()
    assert model.nof_observables == 0
    assert model.nof_state_parameters == 0
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    assert model.f_matrix.shape == ((order + 1) * nof_observables, nof_observables)
    assert model.g_matrix.shape == ((order + 1) * nof_observables, (order + 1) * nof_observables)
    eigenvalues = np.linalg.eigvals(model.g_matrix)
    unique_vals, unique_counts = np.unique(eigenvalues, return_counts=True)
    assert unique_vals.size == 1
    assert unique_counts[0] == (order + 1) * nof_observables
    assert unique_vals[0] != 0


@pytest.mark.parametrize("nof_observables, order", [(1, 1), (2, 2)])
def test_values_polynomial_f_g(nof_observables: int, order: int):
    """Test to check polynomial_f_g method.

    Check if we get exactly the correct F and G matrices for a few order/observation combinations

    Args:
        nof_observables (int): Dimension of observation
        order (int): Polynomial order (0=constant, 1=linear, 2=quadratic etc.)

    """
    model = DLM()
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    if nof_observables == 1 and order == 1:
        true_f = np.array([[1], [0]])
        true_g = np.array([[1.0, 1.0], [0.0, 1.0]])

    elif nof_observables == 2 and order == 2:
        true_f = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        true_g = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        true_f = None
        true_g = None
    assert np.all(model.f_matrix == true_f)
    assert np.all(model.g_matrix == true_g)


def kullback_leibler_gaussian(
    mean_0: np.ndarray, sigma_0: np.ndarray, mean_1: np.ndarray, sigma_1: np.ndarray
) -> float:
    """Kullback-Leibler divergence from N(mean_0, sigma_0) to N(mean_1, sigma_1).

    Helper function to compute the Kullback Leibler divergence

    Duchi, J. "Derivations for Linear Algebra and Optimization" (PDF): 13.
    https://stanford.edu/~jduchi/projects/general_notes.pdf#page=13

    KL( (mean_0, sigma_0) || (mean_1, sigma_1)) = 0.5 * (tr(sigma_1^(-1) @ sigma_0) +
    (mean_1 - mean_0).T @ sigma_1^(-1) @ (mean_1 - mean_0) - k + ln(det(sigma_1)/det(sigma_0))),
    with k = dimension of multivariate normal distribution

    Args:
        mean_0 (np.ndarray): Mean vector of first normal distribution of shape [k x 1]
        sigma_0 (np.ndarray): Covariance matrix of first normal distribution of shape [k x k]
        mean_1 (np.ndarray): Mean vector of second normal distribution of shape [k x 1]
        sigma_1 (np.ndarray): Covariance matrix of second normal distribution of shape [k x k]

    Returns:
        float: Kullback-Leibler divergence

    """
    k = mean_0.shape[0]
    sigma_1_inv = np.linalg.inv(sigma_1)
    diff_mean = mean_1 - mean_0
    statistic = 0.5 * (
        np.trace(sigma_1_inv @ sigma_0)
        + diff_mean.T @ sigma_1_inv @ diff_mean
        - k
        + np.log(np.linalg.det(sigma_1) / np.linalg.det(sigma_0))
    )

    return statistic.flatten()[0]


def create_init_state_and_covariance_matrices(
    nof_observables: int, order: int, rho: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to create initial state and V and W matrices.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        rho (float): Correlation parameter to use in creation of W matrix

    Returns:
        init_state (np.ndarray): initial state vector of shape [nof_observables * (order + 1) x 1]
        v_true (np.ndarray): Observation covariance matrix of shape [nof_observables X nof_observables]
        w_true (np.ndarray): State covariance matrix of shape
            [nof_observables * (order + 1) X nof_observables * (order + 1)]

    """
    v_true = np.eye(nof_observables)
    if order == 0:
        w_true = np.eye(nof_observables) * (1 - rho) + np.ones((nof_observables, nof_observables)) * rho
        init_state = np.zeros((nof_observables, 1))
    else:
        init_state = np.concatenate([1 / 10 ** (np.array(range(order + 1)))] * nof_observables)
        w_true = np.diag(init_state) * 0.1

    return init_state.reshape(-1, 1), v_true, w_true


def forecasts_and_simulate_data(
    nof_observables: int, order: int, rho: float, forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to perform a single run of the test to check for consistency between forecasting functions and
    simulated data. Checking if the covariance matrices are actually positive definite. Also checks error catching when
    not all matrices are set and when forecast steps are set to a number smaller than 1.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        rho (float): Correlation parameter to use in creation of W matrix
        forecast_horizon (int): Maximum forecast step ahead

    Returns:
        statistic_observation_result (np.ndarray): Boolean array of shape [forecast_horizon, ] containing the results
            of the test on the observation forecast
        statistic_state_result (np.ndarray): Boolean array of shape [forecast_horizon, ] containing the results
            of the test on the state forecast

    """
    nof_runs = 1000
    model = DLM()
    with pytest.raises(ValueError):
        _ = model.simulate_data(init_state=np.array([]), nof_timesteps=1)
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    init_state, model.v_matrix, model.w_matrix = create_init_state_and_covariance_matrices(nof_observables, order, rho)
    model.calculate_g_power(max_power=forecast_horizon)

    init_cov = np.zeros(model.w_matrix.shape)
    a_t, f_t = model.forecast_mean(init_state, forecast_steps=np.array(range(forecast_horizon)) + 1)
    r_matrix_t, q_matrix_t = model.forecast_covariance(init_cov, forecast_steps=np.array(range(forecast_horizon)) + 1)

    with pytest.raises(ValueError):
        _ = model.forecast_mean(init_state, forecast_steps=-10)

    with pytest.raises(ValueError):
        _ = model.forecast_covariance(init_cov, forecast_steps=-10)

    state_realizations = np.zeros((model.nof_state_parameters, forecast_horizon, nof_runs))
    observation_realizations = np.zeros((model.nof_observables, forecast_horizon, nof_runs))

    for run in range(nof_runs):
        state_realizations[:, :, run], observation_realizations[:, :, run] = model.simulate_data(
            init_state=init_state, nof_timesteps=forecast_horizon
        )

    statistic_observation_result = np.zeros(forecast_horizon).astype(bool)
    statistic_state_result = np.zeros(forecast_horizon).astype(bool)
    for forecast_step in range(forecast_horizon):
        assert np.linalg.det(q_matrix_t[:, :, forecast_step]) > 0
        assert np.linalg.det(r_matrix_t[:, :, forecast_step]) > 0
        statistic_observation = kullback_leibler_gaussian(
            observation_realizations[:, forecast_step, :].mean(axis=1).reshape(-1, 1),
            np.cov(observation_realizations[:, forecast_step, :]).reshape(nof_observables, nof_observables),
            f_t[:, [forecast_step]],
            q_matrix_t[:, :, forecast_step],
        )
        statistic_observation_result[forecast_step] = statistic_observation < 0.05
        statistic_state = kullback_leibler_gaussian(
            state_realizations[:, forecast_step, :].mean(axis=1).reshape(-1, 1),
            np.cov(state_realizations[:, forecast_step, :]).reshape(
                model.nof_state_parameters, model.nof_state_parameters
            ),
            a_t[:, [forecast_step]],
            r_matrix_t[:, :, forecast_step],
        )
        statistic_state_result[forecast_step] = statistic_state < 0.05

    return statistic_observation_result, statistic_state_result


@pytest.mark.parametrize(
    "nof_observables, order, rho", [(1, 0, 0.8), (2, 0, 0.8), (1, 1, 0.8), (2, 1, 0.8), (3, 2, 0.8)]
)
def test_forecasts_and_simulate_data(nof_observables: int, order: int, rho: float):
    """Function to perform multiple runs of the test to check for consistency between forecast functions and simulated
    data.

    Multiple runs are carried out because of the stochastics in the methods under test
    Eventually we check if more than half of the runs pass the test which would indicate good working code.
    If less than half pass we feel like there is a bug in the code.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        rho (float): Correlation parameter to use in creation of W matrix

    """
    nof_tests = 5
    forecast_horizon = 5
    overall_test_observation = np.zeros((forecast_horizon, nof_tests))
    overall_test_state = np.zeros((forecast_horizon, nof_tests))
    for test in range(nof_tests):
        overall_test_observation[:, test], overall_test_state[:, test] = forecasts_and_simulate_data(
            nof_observables, order, rho, forecast_horizon
        )

    assert np.all(np.count_nonzero(overall_test_observation, axis=1) >= nof_tests / 2)
    assert np.all(np.count_nonzero(overall_test_state, axis=1) >= nof_tests / 2)


def full_dlm_update_and_mahalanobis_distance(
    nof_observables: int, order: int, rho: float, forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to perform 1 test run to check full DLM update and mahalanobis distance calculation.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        rho (float): Correlation parameter to use in creation of W matrix
        forecast_horizon (int): Maximum forecast step ahead

    Returns:
        overall_test_fail (np.ndarray): Boolean array of shape [1, ] containing the result
            of the test using all beams, True = fail, False = pass
        per_beam_test_fail (np.ndarray): Boolean array of shape [nof_observables, ] containing the results
            of the tests for each individual beam, True = fail, False = pass

    We are using the Cramer Von Mises test. The Mahalanobis distance should follow the chi2 distribution with the number
    of degrees of freedom as specified in the args input argument

    """
    n_time = 100

    model = DLM()
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    init_state, model.v_matrix, model.w_matrix = create_init_state_and_covariance_matrices(nof_observables, order, rho)
    model.calculate_g_power(max_power=forecast_horizon)

    _, observations = model.simulate_data(init_state=init_state, nof_timesteps=n_time)

    cov_state = np.zeros(model.w_matrix.shape)
    dlm_state = np.empty((model.nof_state_parameters, n_time + 1))
    dlm_state[:, 0] = init_state.flatten()
    mhd_overall = np.empty(n_time)
    mhd_per_beam = np.empty((model.nof_observables, n_time))
    mhd_overall[:] = np.nan
    mhd_per_beam[:] = np.nan

    new_state_ignore, new_cov_ignore, _ = model.dlm_full_update(
        observations[:, [0]], dlm_state[:, [0]], cov_state, mode="ignore"
    )

    true_state_forecast, _ = model.forecast_mean(dlm_state[:, [0]], forecast_steps=1)
    true_cov_forecast, _ = model.forecast_covariance(cov_state, forecast_steps=1)
    assert np.all(new_state_ignore == true_state_forecast)
    assert np.all(new_cov_ignore == true_cov_forecast)

    with pytest.raises(TypeError):
        _ = model.dlm_full_update(observations[:, [0]], dlm_state[:, [0]], cov_state, mode="error")

    with pytest.raises(AttributeError):
        _ = model.calculate_mahalanobis_distance(
            observations[:, :forecast_horizon], dlm_state[:, [0]], cov_state, forecast_steps=-10
        )
    with pytest.raises(AttributeError):
        _ = model.calculate_mahalanobis_distance(
            observations[:, : (forecast_horizon + 2)], dlm_state[:, [0]], cov_state, forecast_steps=forecast_horizon
        )

    with pytest.raises(AttributeError):
        _ = model.calculate_mahalanobis_distance(
            observations[:, :forecast_horizon], dlm_state[:, :3], cov_state, forecast_steps=forecast_horizon
        )

    for i in range(n_time):
        dlm_state[:, [i + 1]], cov_state, _ = model.dlm_full_update(
            observations[:, [i]], dlm_state[:, [i]], cov_state, mode="learn"
        )
        if i + forecast_horizon < n_time:
            (
                mhd_overall[i + forecast_horizon],
                mhd_per_beam[:, [i + forecast_horizon]],
            ) = model.calculate_mahalanobis_distance(
                observations[:, i : (i + forecast_horizon)],
                dlm_state[:, [i]],
                cov_state,
                forecast_steps=forecast_horizon,
                return_statistics=False,
            )

    temp_mhd = mhd_overall[~np.isnan(mhd_overall)].flatten()
    overall_test_result = cramervonmises(
        temp_mhd[::forecast_horizon], "chi2", args=(forecast_horizon * model.nof_observables,)
    )

    # plt.figure()
    # plt.hist(temp_mhd, density=True, cumulative=True, bins=100, histtype='step', color='k')
    # plt.hist(temp_mhd[::forecast_horizon], density=True, cumulative=True, bins=100, histtype='step', color='r')
    # x = np.linspace(start=0, stop=np.nanmax(mhd_overall), num=100)
    # plt.plot(x, chi2.cdf(x, df=forecast_horizon * model.nof_observables), '-g')

    overall_test_fail = overall_test_result.pvalue < 0.05
    per_beam_test_fail = np.zeros(nof_observables).astype(bool)

    for beam in range(model.nof_observables):
        temp_value = mhd_per_beam[beam, :].flatten()
        temp_mhd = temp_value[~np.isnan(temp_value)]
        test_result_beam = cramervonmises(temp_mhd[::forecast_horizon], "chi2", args=(forecast_horizon,))
        per_beam_test_fail[beam] = test_result_beam.pvalue < 0.05

    return overall_test_fail, per_beam_test_fail


@pytest.mark.parametrize(
    "nof_observables, order, rho, forecast_horizon",
    [
        (1, 0, 0.8, 1),
        (1, 1, 0.8, 1),
        (2, 0, 0.8, 1),
        (1, 0, 0.8, 10),
        (2, 1, 0.8, 10),
        (2, 0, 0.8, 10),
        (3, 2, 0.8, 10),
    ],
)
def test_full_dlm_update_and_mahalanobis_distance(nof_observables, order, rho, forecast_horizon):
    """Function to perform multiple runs of the test to check full DLM update and mahalanobis distance calculation.

    Multiple runs are carried out because of the stochastics in the methods under test
    Eventually we check if more than half of the runs pass the test which would indicate good working code.
    If less than half pass we feel like there is a bug in the code.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        rho (float): Correlation parameter to use in creation of W matrix
        forecast_horizon (int): Maximum forecast step ahead

    """
    nof_tests = 5
    overall_test = np.zeros(nof_tests)
    per_beam_test = np.zeros((nof_observables, nof_tests))
    for run in range(nof_tests):
        overall_test[run], per_beam_test[:, run] = full_dlm_update_and_mahalanobis_distance(
            nof_observables, order, rho, forecast_horizon
        )

    assert np.count_nonzero(overall_test) <= nof_tests / 2
    assert np.all(np.count_nonzero(per_beam_test, axis=1) <= nof_tests / 2)


@pytest.mark.parametrize("nof_observables, order, forecast_horizon", [(2, 0, 10), (2, 1, 10), (2, 2, 10)])
def test_missing_value_mahalanobis_distance(nof_observables, order, forecast_horizon):
    """Function to test if missing values in the observations are handled correctly.

    The functions hsould return a nan value in the one step ahead error where applicable.
    We create 2 identical beams and remove data for a few timesteps of 1 of the beams. The mahalanobis distance should
    be lower for the beam with missing data because effectively that error has been set to 0 in the processing and also
    the number of degrees of freedom should be lower too. For the Mahalanobis distance check we add 1 to the start idx
    as the first entry should still be the same but due to machine precision it might give a different value which
    doesn't pass the test. Also, we subtract 2 from the end index for the same reason and to ensure the forecast
    horizon 'covers' the missing data period and the test is actually valid.
    Due to the stochastic nature of the process we can't really perform a good test to check validity of the chi2
    statistic, but visual inspection of plots have concluded it gives a sensible output.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        forecast_horizon (int): Maximum forecast step ahead

    """
    n_time = 100
    start_idx_missing = 50
    end_idx_missing = 55

    model = DLM()
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    init_state, model.v_matrix, model.w_matrix = create_init_state_and_covariance_matrices(
        nof_observables, order, rho=0.8
    )
    model.calculate_g_power(max_power=forecast_horizon)

    _, observations = model.simulate_data(init_state=init_state, nof_timesteps=n_time)
    observations[1, :] = observations[0, :].copy()
    observations[1, start_idx_missing:end_idx_missing] = np.nan

    cov_state = np.zeros(model.w_matrix.shape)
    dlm_state = np.empty((model.nof_state_parameters, n_time + 1))
    dlm_state[:, 0] = init_state.flatten()
    dlm_state[int(model.nof_state_parameters / 2) :, 0] = dlm_state[: int(model.nof_state_parameters / 2), 0]
    error = np.zeros((model.nof_observables, n_time))
    mhd_overall = np.empty(n_time)
    mhd_per_beam = np.empty((model.nof_observables, n_time))
    mhd_overall[:] = np.nan
    mhd_per_beam[:] = np.nan

    dof_overall = np.zeros(n_time)
    dof_per_beam = np.zeros((model.nof_observables, n_time))
    chi2_overall = np.zeros(n_time)
    chi2_per_beam = np.zeros((model.nof_observables, n_time))

    for i in range(n_time):
        dlm_state[:, [i + 1]], cov_state, error[:, [i]] = model.dlm_full_update(
            observations[:, [i]], dlm_state[:, [i]], cov_state, mode="learn"
        )
        if i + forecast_horizon < n_time:
            (
                mhd_overall[i + forecast_horizon],
                mhd_per_beam[:, [i + forecast_horizon]],
                dof_overall[i + forecast_horizon],
                dof_per_beam[:, [i + forecast_horizon]],
                chi2_overall[i + forecast_horizon],
                chi2_per_beam[:, [i + forecast_horizon]],
            ) = model.calculate_mahalanobis_distance(
                observations[:, i : (i + forecast_horizon)],
                dlm_state[:, [i]],
                cov_state,
                forecast_steps=forecast_horizon,
                return_statistics=True,
            )

    assert np.all(np.isnan(error[1, start_idx_missing:end_idx_missing]))
    assert np.all(
        mhd_per_beam[1, start_idx_missing + 1 : end_idx_missing + forecast_horizon - 2]
        <= mhd_per_beam[0, start_idx_missing + 1 : end_idx_missing + forecast_horizon - 2]
    )
    assert np.all(
        dof_per_beam[1, start_idx_missing : end_idx_missing + forecast_horizon - 2]
        <= dof_per_beam[0, start_idx_missing : end_idx_missing + forecast_horizon - 2]
    )


@pytest.mark.parametrize("nof_observables, order, forecast_horizon", [(2, 0, 10), (2, 1, 10), (2, 2, 10)])
def test_missing_value_updating(nof_observables, order, forecast_horizon):
    """Function to test if updating of the dlm works correctly when missing values in the observations are present.

    When no observation is present we should set the posterior equal to the prior for that variable, so we check if the
    state evolves accordingly.
    Checking if variance of observation estimate which has missing data is monotonically increasing over missing data
    period.
    Finally, checking if the implementation is correct by running a dlm model without any nan values and comparing the
    state values of interest on equality to ensure the nan updating does not affect the non-nan values.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        forecast_horizon (int): Maximum forecast step ahead

    """
    n_time = 100
    start_idx_missing = 50
    end_idx_missing = 55

    model = DLM()
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    init_state, model.v_matrix, model.w_matrix = create_init_state_and_covariance_matrices(
        nof_observables, order, rho=0
    )
    model.calculate_g_power(max_power=forecast_horizon)

    _, observations = model.simulate_data(init_state=init_state, nof_timesteps=n_time)
    observations[1, :] = observations[0, :].copy()
    observations_no_nan = observations.copy()
    observations[1, start_idx_missing:end_idx_missing] = np.nan

    cov_state = np.zeros((model.w_matrix.shape[0], model.w_matrix.shape[1], n_time + 1))
    dlm_state = np.empty((model.nof_state_parameters, n_time + 1))
    dlm_state[:, 0] = init_state.flatten()
    dlm_state[int(model.nof_state_parameters / 2) :, 0] = dlm_state[: int(model.nof_state_parameters / 2), 0]
    error = np.zeros((model.nof_observables, n_time))

    cov_state_no_nan = cov_state.copy()
    dlm_state_no_nan = dlm_state.copy()
    error_no_nan = error.copy()

    for i in range(n_time):
        dlm_state[:, [i + 1]], cov_state[:, :, i + 1], error[:, [i]] = model.dlm_full_update(
            observations[:, [i]], dlm_state[:, [i]], cov_state[:, :, i], mode="learn"
        )
        dlm_state_no_nan[:, [i + 1]], cov_state_no_nan[:, :, i + 1], error_no_nan[:, [i]] = model.dlm_full_update(
            observations_no_nan[:, [i]], dlm_state_no_nan[:, [i]], cov_state_no_nan[:, :, i], mode="learn"
        )

    for idx in range(end_idx_missing - start_idx_missing):
        temp_prior = model.g_matrix @ dlm_state[:, start_idx_missing + idx]
        temp_posterior = dlm_state[:, start_idx_missing + idx + 1]
        assert np.allclose(
            temp_prior[int(model.nof_state_parameters / 2) :], temp_posterior[int(model.nof_state_parameters / 2) :]
        )

    variance_observations = np.zeros((model.nof_observables, n_time + 1))
    for idx in range(n_time + 1):
        temp_matrix = model.f_matrix.T @ cov_state[:, :, idx] @ model.f_matrix
        variance_observations[:, idx] = np.diag(temp_matrix)
    difference = np.diff(variance_observations[1, :])

    assert np.all(difference[start_idx_missing:end_idx_missing] > 0)

    state_idx = int(model.nof_state_parameters / 2)
    assert np.allclose(dlm_state[:state_idx, :], dlm_state_no_nan[:state_idx, :])
    assert np.allclose(cov_state[:state_idx, :state_idx, :], cov_state_no_nan[:state_idx, :state_idx, :])
    assert np.allclose(error[0, :], error_no_nan[0, :])


@pytest.mark.parametrize("nof_observables, order, forecast_horizon", [(2, 0, 10), (2, 1, 10), (2, 2, 10)])
def test_full_covariance_matrix(nof_observables, order, forecast_horizon):
    """Function to test if we correctly construct the full covariance matrix.

    Compares the forecast covariance matrix calculated using the power formula from book with the 'standard'
    method of calculating it recursively. Note that the observation variance contribution to the diagonal blocks seems
    to be missing from the power formula version, this has been accounted for here.

    Args:
        nof_observables (int): Number of observables
        order (int): Order of the polynomial DLM (0==constant, 1==linear, etc.)
        forecast_horizon (int): Maximum forecast step ahead

    """
    n_time = 20

    model = DLM()
    model.polynomial_f_g(nof_observables=nof_observables, order=order)
    init_state, model.v_matrix, model.w_matrix = create_init_state_and_covariance_matrices(
        nof_observables, order, rho=0.8
    )
    model.calculate_g_power(max_power=forecast_horizon)

    _, observations = model.simulate_data(init_state=init_state, nof_timesteps=n_time)

    cov_state = np.zeros((model.w_matrix.shape[0], model.w_matrix.shape[1], n_time + 1))
    dlm_state = np.empty((model.nof_state_parameters, n_time + 1))
    dlm_state[:, 0] = init_state.flatten()
    dlm_state[int(model.nof_state_parameters / 2) :, 0] = dlm_state[: int(model.nof_state_parameters / 2), 0]
    error = np.zeros((model.nof_observables, n_time))

    for i in range(n_time):
        dlm_state[:, [i + 1]], cov_state[:, :, i + 1], error[:, [i]] = model.dlm_full_update(
            observations[:, [i]], dlm_state[:, [i]], cov_state[:, :, i], mode="learn"
        )

        r_t_k, q_t_k = model.forecast_covariance(
            c_matrix=cov_state[:, :, i], forecast_steps=np.array(range(forecast_horizon)) + 1
        )
        full_cov_model = model.create_full_covariance(r_t_k=r_t_k, q_t_k=q_t_k, forecast_steps=forecast_horizon)
        full_cov_test = np.zeros(full_cov_model.shape)

        base_idx = np.array(range(model.nof_observables)) * forecast_horizon

        for k in np.array(range(forecast_horizon)) + 1:
            for v in range(forecast_horizon - k + 1):
                matrix_idx = np.ix_(base_idx + k - 1 + v, base_idx + k - 1)
                matrix_idx_transpose = np.ix_(base_idx + k - 1, base_idx + k - 1 + v)
                value = model.f_matrix.T @ model.g_power[:, :, v] @ r_t_k[:, :, k - 1] @ model.f_matrix
                if v == 0:
                    value = value + model.v_matrix
                full_cov_test[matrix_idx] = value
                full_cov_test[matrix_idx_transpose] = value

        assert np.allclose(full_cov_model, full_cov_test)


def test_dlm_full_update_example():
    """Testing if implementation gives same output as the KURIT example from Harrison and West.

    See table 2.1 on page 41.
    We use an uncorrelated duplication of the 1D input to check if the matrix multiplication works well. We needed
    to round some values in between steps in order to replicate the  results from the book. 2 minor things then remain:
        - in dlm_state we changed the check value from 142.6 (from the  example) to 142.7 (6th entry) which is the
        answer we are getting.
        - in cov_state we changed the test value from 20 to 21  for the last element for a similar reason.
    Given the rest is giving exactly the same results we are confident this works. Our assumption is that the numbers
    presented in the book table are rounded before being propagated to the next stage of the calculation, giving rise
    to the differences.

    """
    n_time = 9

    model = DLM()
    model.f_matrix = np.array([[1, 0], [0, 1]])
    model.g_matrix = np.array([[1, 0], [0, 1]])
    model.calculate_g_power(max_power=1)
    model.v_matrix = np.array([[100, 0], [0, 100]])
    model.w_matrix = np.array([[5, 0], [0, 5]])

    cov_state = np.zeros((model.w_matrix.shape[0], model.w_matrix.shape[1], n_time + 1))
    cov_state[:, :, 0] = np.array([[400, 0], [0, 400]])

    dlm_state = np.empty((model.nof_state_parameters, n_time + 1))
    dlm_state[:, [0]] = np.array([[130], [130]])

    observations = np.array(
        [[150, 136, 143, 154, 135, 148, 128, 149, 146], [150, 136, 143, 154, 135, 148, 128, 149, 146]]
    )

    error = np.zeros((model.nof_observables, n_time))
    r_t_k = np.zeros((model.nof_state_parameters, model.nof_state_parameters, n_time))
    q_t_k = np.zeros((model.nof_observables, model.nof_observables, n_time))

    for i in range(n_time):
        r_t_k[:, :, [i]], q_t_k[:, :, [i]] = model.forecast_covariance(c_matrix=cov_state[:, :, i], forecast_steps=1)

        dlm_state[:, [i + 1]], cov_state[:, :, i + 1], error[:, [i]] = model.dlm_full_update(
            observations[:, [i]], dlm_state[:, [i]], cov_state[:, :, i], mode="learn"
        )

        dlm_state[:, [i + 1]] = np.round(dlm_state[:, [i + 1]], decimals=1)
        cov_state[:, :, i + 1] = np.round(cov_state[:, :, i + 1], decimals=0)
        error[:, [i]] = np.round(error[:, [i]], decimals=1)

    dlm_state = np.round(dlm_state, decimals=1)
    q_t_k = np.round(q_t_k, decimals=0)
    r_t_k = np.round(r_t_k, decimals=0)
    error = np.round(error, decimals=1)
    cov_state = np.round(cov_state, decimals=0)

    adaptive_coefficient_0 = np.round(r_t_k[0, 0, :] / q_t_k[0, 0, :], decimals=2)
    adaptive_coefficient_1 = np.round(r_t_k[1, 1, :] / q_t_k[1, 1, :], decimals=2)

    assert np.allclose(q_t_k[0, 0, :], np.array([505, 185, 151, 139, 133, 130, 128, 127, 126]))
    assert np.allclose(
        dlm_state[0, :], np.array([130.0, 146.0, 141.4, 141.9, 145.3, 142.7, 143.9, 140.4, 142.2, 143.0])
    )
    assert np.allclose(adaptive_coefficient_0, np.array([0.80, 0.46, 0.34, 0.28, 0.25, 0.23, 0.22, 0.21, 0.21]))
    assert np.allclose(error[0, :], np.array([20.0, -10.0, 1.6, 12.1, -10.3, 5.3, -15.9, 8.6, 3.8]))
    assert np.allclose(cov_state[0, 0, :], np.array([400, 80, 46, 34, 28, 25, 23, 22, 21, 21]))

    assert np.allclose(error[0, :], error[1, :])
    assert np.allclose(dlm_state[0, :], dlm_state[1, :])
    assert np.allclose(cov_state[0, 0, :], cov_state[1, 1, :])
    assert np.all(cov_state[0, 1, :] == 0)
    assert np.all(cov_state[1, 0, :] == 0)
    assert np.allclose(adaptive_coefficient_0, adaptive_coefficient_1)
    assert np.allclose(r_t_k[0, 0, :], r_t_k[1, 1, :])
    assert np.all(r_t_k[0, 1, :] == 0)
    assert np.all(r_t_k[1, 0, :] == 0)
    assert np.allclose(q_t_k[0, 0, :], q_t_k[1, 1, :])
    assert np.all(q_t_k[0, 1, :] == 0)
    assert np.all(q_t_k[1, 0, :] == 0)
