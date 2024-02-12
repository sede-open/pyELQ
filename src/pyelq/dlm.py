# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""DLM module.

This module provides a class definition for the Dynamic Linear Models following Harrison and West
'Bayesian Forecasting and Dynamic Models' (2nd ed), Springer New York, NY, Chapter 4, https://doi.org/10.1007/b98971

"""
from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
from scipy.stats import chi2


@dataclass
class DLM:
    """Defines the DLM in line with Harrison and West (2nd edition) Chapter 4.

    Attributes:
        f_matrix (np.ndarray, optional): F matrix linking the state to the observables of
            size [nof_state_parameters x nof_observables]
        g_matrix (np.ndarray, optional): G matrix characterizing the state evolution of
            size [nof_state_parameters x nof_state parameters]
        v_matrix (np.ndarray, optional): V matrix being the covariance matrix of the zero mean observation noise
            of size [nof_state_parameters x nof_observables]
        w_matrix (np.ndarray, optional): W matrix being the covariance matrix of the zero mean system noise of
            size [nof_state_parameters x nof_state parameters]
        g_power (np.ndarray, optional): Attribute to store G^k, does not get initialized

    """

    f_matrix: np.ndarray = None
    g_matrix: np.ndarray = None
    v_matrix: np.ndarray = None
    w_matrix: np.ndarray = None
    g_power: np.ndarray = field(init=False)

    @property
    def nof_observables(self) -> int:
        """Int: Number of observables as derived from the associated F matrix."""
        if self.f_matrix is not None and isinstance(self.f_matrix, np.ndarray):
            return self.f_matrix.shape[1]
        return 0

    @property
    def nof_state_parameters(self) -> int:
        """Int: Number of state parameters as derived from the associated G matrix."""
        if self.g_matrix is not None and isinstance(self.g_matrix, np.ndarray):
            return self.g_matrix.shape[0]
        return 0

    def calculate_g_power(self, max_power: int) -> None:
        """Calculate the powers of the G matrix.

        Calculate the powers upfront, so we don't have to calculate it at every iteration. Result gets stored in the
        g_power attribute of the DLM class. We use an iterative way of calculating the power to have the fewest matrix
        multiplications necessary, i.e. we are not using numpy.linalg.matrix_power as that would leak to k factorial
        multiplications instead of the k we have now.

        Args:
            max_power (int): Maximum power to compute

        """
        if self.nof_state_parameters == 1:
            self.g_power = self.g_matrix ** np.array([[range(max_power + 1)]])
        else:
            self.g_power = np.zeros((self.nof_state_parameters, self.nof_state_parameters, max_power + 1))
            self.g_power[:, :, 0] = np.identity(self.nof_state_parameters)
            for i in range(max_power):
                self.g_power[:, :, i + 1] = self.g_power[:, :, i] @ self.g_matrix

    def polynomial_f_g(self, nof_observables: int, order: int) -> None:
        """Create F and G matrices associated with a polynomial DLM.

        Following Harrison and West (Chapter 7 on polynomial DLMs) with the exception that we use order==0 for a
        "constant" DLM and order==1 for linear growth DLM, order==2 for quadratic growth etc.
        Hence, the definition of n-th order polynomial DLM in Harrison & West is implemented here with order=n-1
        We stack the observables in a block diagonal form. So the first #order of rows belong to the first observable,
        the second #order rows belong to the second observable etc.
        Results are being stored in the f_matrix and g_matrix attributes respectively

        Args:
            nof_observables (int): Dimension of observation
            order (int): Polynomial order (0=constant, 1=linear, 2=quadratic etc.)

        """
        e_n = np.append(1, np.zeros(order))[:, None]
        self.f_matrix = np.kron(np.eye(nof_observables), e_n)

        l_n = np.triu(np.ones((order + 1, order + 1)))
        self.g_matrix = np.kron(np.eye(nof_observables), l_n)

    def simulate_data(self, init_state: np.ndarray, nof_timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate data from DLM model.

        Function to simulate state evolution and corresponding observations according to model as specified through DLM
        class attributes (F, G, V and W matrices)

        Args:
            init_state (np.ndarray): Initial state vector to start simulating from of size [nof_state_parameters x 1]
            nof_timesteps (int): Number of timesteps to simulate

        Returns:
            state (np.ndarray): Simulated state vectors of size [nof_state_parameters x nof_timesteps]
            obs (np.ndarray): Simulated observations of size [nof_observables x nof_timesteps]

        """
        if self.f_matrix is None or self.g_matrix is None or self.v_matrix is None or self.w_matrix is None:
            raise ValueError("Please specify all matrices (F, G, V and W)")

        obs = np.empty((self.nof_observables, nof_timesteps))
        state = np.empty((self.nof_state_parameters, nof_timesteps))

        state[:, [0]] = init_state
        mean_state_noise = np.zeros(self.nof_state_parameters)
        mean_observation_noise = np.zeros(self.nof_observables)

        for i in range(nof_timesteps):
            if i == 0:
                state[:, [i]] = (
                    self.g_matrix @ init_state
                    + np.random.multivariate_normal(mean_state_noise, self.w_matrix, size=1).T
                )
            else:
                state[:, [i]] = (
                    self.g_matrix @ state[:, [i - 1]]
                    + np.random.multivariate_normal(mean_state_noise, self.w_matrix, size=1).T
                )
            obs[:, [i]] = (
                self.f_matrix.T @ state[:, [i]]
                + np.random.multivariate_normal(mean_observation_noise, self.v_matrix, size=1).T
            )

        return state, obs

    def forecast_mean(
        self, current_mean_state: np.ndarray, forecast_steps: Union[int, list, np.ndarray] = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform forecasting of the state and observation mean parameters.

        Following Harrison and West (2nd ed) Chapter 4.4 (Forecast Distributions), corollary 4.1, assuming F and G are
        constant over time.
        Note that in the output the second axis of the output arrays is the forecast dimension consistent with the
        forecast steps input, all forecast steps contained in the forecast steps argument are returned.

        Args:
            current_mean_state (np.ndarray): Current mean parameter for the state of size [nof_state_parameters x 1]
            forecast_steps (Union[int, list, np.ndarray], optional): Steps ahead to forecast

        Returns:
            a_t_k (np.array): Forecast values of state mean parameter of the size
                [nof_observables x size(forecast_steps)]
            f_t_k (np.array): Forecast values of observation mean parameter of the size
                [nof_observables x size(forecast_steps)]

        """
        min_forecast = np.amin(forecast_steps)

        if min_forecast < 1:
            raise ValueError(f"Minimum forecast should be >= 1, currently it is {min_forecast}")
        if isinstance(forecast_steps, int):
            forecast_steps = [forecast_steps]

        a_t_k = np.hstack([self.g_power[:, :, step] @ current_mean_state for step in forecast_steps])
        f_t_k = self.f_matrix.T @ a_t_k

        return a_t_k, f_t_k

    def forecast_covariance(
        self, c_matrix: np.ndarray, forecast_steps: Union[int, list, np.ndarray] = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform forecasting of the state and observation covariance parameters.

        Following Harrison and West (2nd ed) Chapter 4.4 (Forecast Distributions), assuming F, G, V and W are
        constant over time.
        Note that in the output the third axis of the output arrays is the forecast dimension consistent with the
        forecast steps input, all forecast steps contained in the forecast steps argument are returned.
        sum_g_w_g is initialized as G^k @ W @ G^k for k==0, hence we initialize as W
        Because of zero based indexing, in the for loop i==1 means 2-step ahead forecast which requires element
        (i+1) of the g_power attribute as the third dimension serves as the actual power of the G matrix

        Args:
            c_matrix (np.ndarray): Current posterior covariance estimate for the state of size
                [nof_state_parameters x nof_state_parameters]
            forecast_steps (Union[int, list, np.ndarray], optional): Steps ahead to forecast

        Returns:
            r_t_k (np.array): Forecast values of estimated prior state covariance of the size
                [nof_state_parameters x nof_state_parameters x size(forecast_steps)]
            q_t_k (np.array): Forecast values of estimated observation covariance of the size
                [nof_observables x nof_observables x size(forecast_steps)]

        """
        min_forecast = np.amin(forecast_steps)
        max_forecast = np.amax(forecast_steps)

        if min_forecast < 1:
            raise ValueError(f"Minimum forecast should be >= 1, currently it is {min_forecast}")
        if isinstance(forecast_steps, int):
            forecast_steps = [forecast_steps]

        sum_g_w_g = np.zeros((self.nof_state_parameters, self.nof_state_parameters, max_forecast))
        sum_g_w_g[:, :, 0] = self.w_matrix
        for i in np.arange(1, max_forecast, step=1):
            sum_g_w_g[:, :, i] = (
                sum_g_w_g[:, :, i - 1] + self.g_power[:, :, i] @ self.w_matrix @ self.g_power[:, :, i].T
            )

        r_t_k = np.dstack(
            [
                self.g_power[:, :, step] @ c_matrix @ self.g_power[:, :, step].T + sum_g_w_g[:, :, step - 1]
                for step in forecast_steps
            ]
        )
        q_t_k = np.dstack(
            [self.f_matrix.T @ r_t_k[:, :, idx] @ self.f_matrix + self.v_matrix for idx in range(r_t_k.shape[2])]
        )

        return r_t_k, q_t_k

    def update_posterior(
        self, a_t: np.ndarray, r_matrix_t: np.ndarray, q_matrix_t: np.ndarray, error: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update of the posterior mean and covariance of the state.

        Following Harrison and West (2nd ed) Chapter 4.4 (Forecast Distributions), assuming F, G, V and W are
        constant over time.
        We are using a solver instead of calculating the inverse of Q directly
        Setting inf values in Q equal to 0 after the solver function for computational issues, otherwise we would
        get 0 * inf = nan, where we want the result to be 0.

        Args:
            a_t (np.ndarray): Current prior mean of the state of size [nof_state_parameters x 1]
            r_matrix_t (np.ndarray): Current prior covariance of the state of size [nof_state_parameters x nof_state_parameters]
            q_matrix_t (np.ndarray): Current one step ahead forecast covariance estimate of the observations of size [nof_observables x nof_observables]
            error (np.ndarray): Error associated with the one step ahead forecast (observation - forecast) of size [nof_observables x 1]

        Returns:
            m_t (np.array): Posterior mean estimate of the state of size [nof_state_parameters x 1]
            c_matrix (np.array): Posterior covariance estimate of the state of size [nof_state_parameters x nof_state_parameters]

        """
        if self.nof_state_parameters == 1:
            a_matrix_t = r_matrix_t @ self.f_matrix.T @ (1 / q_matrix_t)
        else:
            a_matrix_t = r_matrix_t @ np.linalg.solve(q_matrix_t.T, self.f_matrix.T).T
        m_t = a_t + a_matrix_t @ error
        q_matrix_t[np.isinf(q_matrix_t)] = 0
        c_matrix = r_matrix_t - a_matrix_t @ q_matrix_t @ a_matrix_t.T

        return m_t, c_matrix

    def dlm_full_update(
        self,
        new_observation: np.ndarray,
        current_mean_state: np.ndarray,
        current_cov_state: np.ndarray,
        mode: str = "learn",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform 1 step of the full DLM update.

        Following Harrison and West (2nd ed) we perform all steps to update the entire DLM model and obtain new
        estimates for all parameters involved, including nan value handling.
        When mode == 'learn' the parameters are updated, when mode == 'ignore' the current observation is ignored and
        the posterior is set equal to the prior
        When no observation is present (i.e. a nan value) we let the covariance (V matrix) for that particular sensor
        such that we set the variance of that sensor for that time instance to infinity and set all cross (covariance)
        terms to 0. Instead of changing this in the V matrix, we simply adjust the Q matrix accordingly. Effectively,
        we set the posterior equal to the prior for that particular sensor and the uncertainty associated with the new
        forecast gets increased. We set the error equal to zero for computational issues, first but finally set it equal
        to nan in the end.

        Args:
            new_observation (np.ndarray): New observations to use in the updating of the estimates of size [nof_observables x 1]
            current_mean_state (np.ndarray):  Current mean estimate for the state of size [nof_state_parameters x 1]
            current_cov_state (np.ndarray):  Current covariance estimate for the state of size [nof_state_parameters x nof_state_parameters]
            mode (str, optional): String indicating whether the DLM needs to be updated using the new observation or not. Currently, `learn` and `ignore` are implemented

        Returns:
            new_mean_state (np.ndarray): New mean estimate for the state of size [nof_state_parameters x 1]
            new_cov_state (np.ndarray): New covariance estimate for the state of size [nof_state_parameters x nof_state_parameters]
            error (np.ndarray): Error between the observation and the forecast (observation - forecast) of size [nof_observables x 1]

        """
        a_t, f_t = self.forecast_mean(current_mean_state, forecast_steps=1)
        r_matrix_t, q_matrix_t = self.forecast_covariance(current_cov_state, forecast_steps=1)
        error = new_observation - f_t

        nan_bool = np.isnan(new_observation)
        nan_idx = np.argwhere(nan_bool.flatten())
        if np.any(nan_bool):
            q_matrix_t[nan_idx, :, 0] -= self.v_matrix[nan_idx, :]
            q_matrix_t[:, nan_idx, 0] -= self.v_matrix[:, nan_idx]
            q_matrix_t[nan_idx, nan_idx, 0] = np.inf
            error[nan_idx] = 0

        if mode == "learn":
            new_mean_state, new_cov_state = self.update_posterior(a_t, r_matrix_t[:, :, 0], q_matrix_t[:, :, 0], error)
        elif mode == "ignore":
            new_mean_state = a_t
            new_cov_state = r_matrix_t
        else:
            raise TypeError(f"Mode {mode} not implemented")

        error[nan_idx] = np.nan

        return new_mean_state, new_cov_state, error

    def calculate_mahalanobis_distance(
        self,
        new_observations: np.ndarray,
        current_mean_state: np.ndarray,
        current_cov_state: np.ndarray,
        forecast_steps: int = 1,
        return_statistics=False,
    ) -> Union[Tuple[float, np.ndarray], Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Calculate the mahalanobis distance.

        Calculating the Mahalanobis distance which is defined as error.T @ covariance^(-1) @ error
        The error is flatted in row-major (C-style) This returns the stacked rows, which in our case is the errors per
        observation parameter stacked and this is exactly what we want: array([[1, 2], [3, 4]]).reshape((-1, 1),
        order='C') becomes column array([1, 2 3, 4])
        Using a solve method instead of calculating inverse matrices directly
        When calculating mhd_per_obs_param we use the partial result and reshape the temporary output such that we can
        sum the correct elements associated with the same observable together
        When no observation is present (i.e. a nan value) we let the covariance (V matrix) for that particular sensor
        such that we set the variance of that sensor for that time instance to infinity and set all cross (covariance)
        terms to 0. Instead of changing this in the V matrix, we simply adjust the Q matrix accordingly. Effectively,
        we set the posterior equal to the prior for that particular sensor and the uncertainty associated with the new
        forecast gets increased. We set the error equal to zero for computational issues, but this does decrease the
        number of degrees of freedom for that particular Mahalanobis distance calculation, basically decreasing the
        Mahalanobis distance. We allow the option to output the number of degrees of freedom and chi2 statistic which
        allows to take this decrease in degrees of freedom into account.

        Args:
            new_observations (np.ndarray): New observations to use in the calculation of the mahalanobis distance of
                size [nof_observables x forecast_steps]
            current_mean_state (np.ndarray): Current mean estimate for the state of size [nof_state_parameters x 1]
            current_cov_state (np.ndarray): Current covariance estimate for the state of size
                [nof_state_parameters x nof_state_parameters]
            forecast_steps (int, optional): Number of steps ahead to forecast and use in the mahalanobis distance
                calculation
            return_statistics (bool, optional): Boolean to return used degrees of freedom and chi2 statistic
        Returns:
            mhd_overall (float): mahalanobis distance over all observables
            mhd_per_obs_param (np.ndarray): mahalanobis distance per observation parameter of size [nof_observables, 1]

        """
        if forecast_steps <= 0:
            raise AttributeError("Forecast steps should be a positive integer")

        if new_observations.size / self.nof_observables != forecast_steps:
            raise AttributeError("Sizes of new observations and forecast steps are not aligning")

        _, f_t_k = self.forecast_mean(current_mean_state, forecast_steps=np.array(range(forecast_steps)) + 1)

        if new_observations.shape != f_t_k.shape:
            raise AttributeError("Dimensions of new_observations are not aligning with dimensions of forecast")

        error = np.subtract(new_observations, f_t_k).reshape((-1, 1), order="C")

        r_t_k, q_t_k = self.forecast_covariance(current_cov_state, forecast_steps=np.array(range(forecast_steps)) + 1)

        nan_bool = np.isnan(new_observations)
        if np.any(nan_bool):
            nan_idx = np.argwhere(nan_bool)
            for value in nan_idx:
                q_t_k[value[0], :, value[1]] -= self.v_matrix[value[0], :]
                q_t_k[:, value[0], value[1]] -= self.v_matrix[:, value[0]]

            q_t_k[nan_idx[:, 0], nan_idx[:, 0], nan_idx[:, 1]] = np.inf
            error[nan_bool.reshape((-1, 1), order="C")] = 0

        if forecast_steps > 1:
            full_covariance = self.create_full_covariance(r_t_k=r_t_k, q_t_k=q_t_k, forecast_steps=forecast_steps)
        else:
            full_covariance = q_t_k[:, :, 0]

        mhd_overall = mahalanobis_distance(error=error, cov_matrix=full_covariance)
        mhd_per_obs_param = np.empty((self.nof_observables, 1))

        for i_obs in range(self.nof_observables):
            ind_hrz = np.array(range(forecast_steps)) + i_obs * forecast_steps
            mhd_per_obs_param[i_obs] = mahalanobis_distance(
                error=error[ind_hrz], cov_matrix=full_covariance[np.ix_(ind_hrz, ind_hrz)]
            )

        if self.nof_observables == 1:
            mhd_per_obs_param = mhd_per_obs_param.item()

        if return_statistics:
            dof_per_obs_param = (nan_bool.shape[1] - np.count_nonzero(nan_bool, axis=1)).reshape(
                self.nof_observables, 1
            )
            dof_overall = dof_per_obs_param.sum()
            chi2_cdf_per_obs_param = chi2.cdf(mhd_per_obs_param.flatten(), dof_per_obs_param.flatten()).reshape(
                self.nof_observables, 1
            )
            chi2_cdf_overall = chi2.cdf(mhd_overall, dof_overall)

            return (
                mhd_overall,
                mhd_per_obs_param,
                dof_overall,
                dof_per_obs_param,
                chi2_cdf_overall,
                chi2_cdf_per_obs_param,
            )

        return mhd_overall, mhd_per_obs_param

    def create_full_covariance(self, r_t_k: np.ndarray, q_t_k: np.ndarray, forecast_steps: int) -> np.ndarray:
        """Helper function to construct the full covariance matrix.

        Following Harrison and West (2nd ed) Chapter 4.4 (Forecast distributions) Theorem 4.2 and corollary 4.2
        we construct the full covariance matrix. This full covariance matrix is the covariance matrix of all forecasted
        observations with respect to each other. Hence, it's COV[Y_{t+k}, Y_{t+j}] with j and k 1<=j,k<=forecast steps
        input argument and Y_{t+k} the k step ahead forecast of the observation at time t

        The matrix is build up using the different blocks for different covariances between observations i and j.
        The diagonals of each block are calculated first as q_t_k[i, j, :].
        Next the i, j-th (lower triangular) entry of the m, n-th block is calculated as
        (F.T @ G^(i-j) r_t_k[:, :, j] @ F)[i, j]
        Next each upper triangular part of each lower diagonal block is calculated and next the entire upper triangular
        part of the full matrix is calculated

        Args:
            r_t_k (np.array): Forecast values of estimated prior state covariance of the size
                [nof_state_parameters x nof_state_parameters x forecast_steps]
            q_t_k (np.array): Forecast values of estimated observation covariance of the size
                [nof_observables x nof_observables x forecast_steps]
            forecast_steps (int): Maximum number of steps ahead to forecast and use all of those in the mahalanobis
                distance calculation

        Returns:
            full_covariance (np.array): Full covariance matrix of all forecasted observations with respect to each other
            having size [(nof_observables * forecast_steps) X (nof_observables * forecast_steps)]

        """
        full_covariance = np.zeros((forecast_steps * self.nof_observables, forecast_steps * self.nof_observables))
        base_idx = np.array(range(forecast_steps))
        for block_i in range(self.nof_observables):
            for block_j in range(block_i + 1):
                block_rows = base_idx + block_i * forecast_steps
                block_cols = base_idx + block_j * forecast_steps
                full_covariance[block_rows, block_cols] = q_t_k[block_i, block_j, :]

        temp_idx = np.array(range(self.nof_observables))
        for sub_i in np.arange(start=1, stop=forecast_steps, step=1):
            sub_row = temp_idx * forecast_steps + sub_i
            for sub_j in range(sub_i):
                sub_col = temp_idx * forecast_steps + sub_j
                sub_idx = np.ix_(sub_row, sub_col)
                full_covariance[sub_idx] = (
                    self.f_matrix.T @ self.g_power[:, :, sub_i - sub_j] @ r_t_k[:, :, sub_j] @ self.f_matrix
                )

        for block_i in range(self.nof_observables):
            for block_j in range(block_i):
                block_rows = base_idx + block_i * forecast_steps
                block_cols = base_idx + block_j * forecast_steps
                block_idx = np.ix_(block_rows, block_cols)
                full_covariance[block_idx] = full_covariance[block_idx] + np.tril(full_covariance[block_idx], k=-1).T

        full_covariance = np.tril(full_covariance) + np.tril(full_covariance, k=-1).T

        return full_covariance


def mahalanobis_distance(error: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate Mahalanobis distance for multivariate observations.

    m = e.T @ inv(cov) @ e
    Sometimes the solution does not exist when np.inf value is present in cov_matrix (computational limitations?)
    Hence, we set it to a large value instead

    Args:
        error (np.ndarray):  n x p   observation error
        cov_matrix (np.ndarray): p x p covariance matrix

    Returns:
        np.ndarray: n x 1  mahalanobis distance score for each observation

    """
    if cov_matrix.size == 1:
        return error.item() ** 2 / cov_matrix.item()

    partial_solution = np.linalg.solve(cov_matrix, error)
    if np.any(np.isnan(partial_solution)):
        cov_matrix[np.isinf(cov_matrix)] = 1e100
        partial_solution = np.linalg.solve(cov_matrix, error)

    return np.sum(error * partial_solution, axis=0).item()
