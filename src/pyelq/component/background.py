# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model components for background modelling."""

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from openmcmc import gmrf, parameter
from openmcmc.distribution.distribution import Gamma
from openmcmc.distribution.location_scale import Normal
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalGamma, NormalNormal
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from pyelq.component.component import Component
from pyelq.coordinate_system import Coordinate
from pyelq.gas_species import GasSpecies
from pyelq.meteorology import MeteorologyGroup
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import SensorGroup


@dataclass
class Background(Component):
    """Superclass for background models.

    Attributes:
        n_obs (int): total number of observations in the background model (across all sensors).
        n_parameter (int): number of parameters in the background model
        bg (np.ndarray): array of sampled background values, populated in self.from_mcmc() after the MCMC run is
            completed.
        precision_scalar (np.ndarray): array of sampled background precision values, populated in self.from_mcmc() after
            the MCMC run is completed. Only populated if update_precision is True.
        precision_matrix (Union[np.ndarray, sparse.csr_array]): un-scaled precision matrix for the background parameter
            vector.
        mean_bg (float): global mean background value. Should be populated from the value specified in the GasSpecies
            object.
        update_precision (bool): logical determining whether the background (scalar) precision parameter should be
            updated as part of the MCMC. Defaults to False.
        prior_precision_shape (float): shape parameter for the prior gamma distribution for the scalar precision
            parameter(s).
        prior_precision_rate (float): rate parameter for the prior gamma distribution for the scalar precision
            parameter(s).
        initial_precision (float): initial value for the scalar precision parameter.
        basis_matrix (sparse.csr_array): [n_obs x n_time] matrix mapping the background model parameters on to the
            observations.

    """

    n_obs: int = field(init=False)
    n_parameter: int = field(init=False)
    bg: np.ndarray = field(init=False)
    precision_scalar: np.ndarray = field(init=False)
    precision_matrix: Union[np.ndarray, sparse.csc_matrix] = field(init=False)
    mean_bg: Union[float, None] = None
    update_precision: bool = False
    prior_precision_shape: float = 1e-3
    prior_precision_rate: float = 1e-3
    initial_precision: float = 1.0
    basis_matrix: sparse.csr_array = field(init=False)

    @abstractmethod
    def initialise(self, sensor_object: SensorGroup, meteorology: MeteorologyGroup, gas_species: GasSpecies):
        """Take data inputs and extract relevant properties.

        Args:
            sensor_object (SensorGroup): sensor data
            meteorology (MeteorologyGroup): meteorology data
            gas_species (GasSpecies): gas species information

        """

    def make_model(self, model: list = None) -> list:
        """Take model list and append new elements from current model component.

        Args:
            model (list, optional): Current list of model elements. Defaults to None.

        Returns:
            list: model output list.

        """
        bg_precision_predictor = parameter.ScaledMatrix(matrix="P_bg", scalar="lambda_bg")
        model.append(Normal("bg", mean="mu_bg", precision=bg_precision_predictor))
        if self.update_precision:
            model.append(Gamma("lambda_bg", shape="a_lam_bg", rate="b_lam_bg"))
        return model

    def make_sampler(self, model: Model, sampler_list: list = None) -> list:
        """Take sampler list and append new elements from current model component.

        Args:
            model (Model): Full model list of distributions.
            sampler_list (list, optional): Current list of samplers. Defaults to None.

        Returns:
            list: sampler output list.

        """
        if sampler_list is None:
            sampler_list = []
        sampler_list.append(NormalNormal("bg", model))
        if self.update_precision:
            sampler_list.append(NormalGamma("lambda_bg", model))
        return sampler_list

    def make_state(self, state: dict = None) -> dict:
        """Take state dictionary and append initial values from model component.

        Args:
            state (dict, optional): current state vector. Defaults to None.

        Returns:
            dict: current state vector with components added.

        """
        state["mu_bg"] = np.ones((self.n_parameter, 1)) * self.mean_bg
        state["B_bg"] = self.basis_matrix
        state["bg"] = np.ones((self.n_parameter, 1)) * self.mean_bg
        state["P_bg"] = self.precision_matrix
        state["lambda_bg"] = self.initial_precision
        if self.update_precision:
            state["a_lam_bg"] = self.prior_precision_shape
            state["b_lam_bg"] = self.prior_precision_rate
        return state

    def from_mcmc(self, store: dict):
        """Extract results of mcmc from mcmc.store and attach to components.

        Args:
            store (dict): mcmc result dictionary.

        """
        self.bg = store["bg"]
        if self.update_precision:
            self.precision_scalar = store["lambda_bg"]


@dataclass
class TemporalBackground(Background):
    """Model which imposes only temporal correlation on the background parameters.

    Assumes that the prior mean concentration of the background at every location/time point is the global average
    background concentration as defined in the input GasSpecies object.

    Generates the (un-scaled) prior background precision matrix using the function gmrf.precision_temporal: this
    precision matrix imposes first-oder Markov structure for the temporal dependence.

    By default, the times used for the model definition are the set of unique times in the observation set.

    This background model only requires the initialise function, and does not require the implementation of any further
    methods.

    Attributes:
        time (Union[np.ndarray, pd.arrays.DatetimeArray]): vector of times used in defining the model.

    """

    time: Union[np.ndarray, pd.arrays.DatetimeArray] = field(init=False)

    def initialise(self, sensor_object: SensorGroup, meteorology: MeteorologyGroup, gas_species: GasSpecies):
        """Create temporal background model from sensor, meteorology and gas species inputs.

        Args:
            sensor_object (SensorGroup): sensor data object.
            meteorology (MeteorologyGroup): meteorology data object.
            gas_species (GasSpecies): gas species data object.

        """
        self.n_obs = sensor_object.nof_observations
        self.time, unique_inverse = np.unique(sensor_object.time, return_inverse=True)
        self.time = pd.array(self.time, dtype="datetime64[ns]")
        self.n_parameter = len(self.time)
        self.basis_matrix = sparse.csr_array((np.ones(self.n_obs), (np.array(range(self.n_obs)), unique_inverse)))
        self.precision_matrix = gmrf.precision_temporal(time=self.time)
        if self.mean_bg is None:
            self.mean_bg = gas_species.global_background


@dataclass
class SpatioTemporalBackground(Background):
    """Model which imposes both spatial and temporal correlation on the background parameters.

    Defines a grid in time, and assumes a correlated time-series per sensor using the defined time grid.

    The background parameter is an [n_location * n_time x 1] (if self.spatial_dependence is True) or an [n_time x 1]
    vector (if self.spatial_dependence is False). In the spatio-temporal case, the background vector is assumed to
    unwrap over space and time as follows:
    bg = [b_1(t_1), b_2(t_1),..., b_nlct(t_1),...,b_1(t_k),..., b_nlct(t_k),...].T
    where nlct is the number of sensor locations.
    This unwrapping mechanism is chosen as it greatly speeds up the sparse matrix operations in the solver (vs. the
    alternative).

    self.basis_matrix is set up to map the elements of the full background vector onto the observations, on the basis
    of spatial location and nearest time knot.

    The temporal background correlation is computed using gmrf.precision_temporal, and the spatial correlation is
    computed using a squared exponential correlation function, parametrized by self.spatial_correlation_param (spatial
    correlation, measured in metres). The full precision matrix is simply a Kronecker product between the two
    component precision matrices.

    Attributes:
        n_time (int): number of time knots for which the model is defined. Note that this does not need to be the same
            as the number of concentration observations in the analysis.
        n_location (int): number of spatial knots in the model.
        time (pd.arrays.DatetimeArray): vector of times used in defining the model.
        spatial_dependence (bool): flag indicating whether the background parameters should be spatially correlated. If
            True, the model assumes a separate background time-series per sensor location, and assumes these
            time-series to be spatially correlated. If False (default), the background parameters are assumed to be
            common between sensors (only temporally correlated).
        spatial_correlation_param (float): correlation length parameter, determining the degree of spatial correlation
            imposed on the background time-series. Units are metres. Assumes equal correlation in all spatial
            directions. Defaults to 1.0.
        location (np.ndarray): [n_location x 3] array of sensor locations, used for calculating the spatial correlation
            between the sensor background values. If self.spatial_dependence is False, this attribute is simply set to
            be the location of the first sensor in the sensor object.
        temporal_precision_matrix (Union[np.ndarray, sparse.csc_matrix]): temporal component of the precision matrix.
            The full model precision matrix is the Kronecker product of this matrix with self.spatial_precision_matrix.
        spatial_precision_matrix (np.ndarray): spatial component of the precision matrix. The full model precision
            matrix is the Kronecker product of this matrix with the self.temporal_precision_matrix. Simply set to 1 if
            self.spatial_dependence is False.
        precision_time_0 (float): precision relating to the first time stamp in the model. Defaults to 0.01.

    """

    n_time: Union[int, None] = None
    n_location: int = field(init=False)
    time: pd.arrays.DatetimeArray = field(init=False)
    spatial_dependence: bool = False
    spatial_correlation_param: float = field(init=False, default=1.0)
    location: Coordinate = field(init=False)
    temporal_precision_matrix: Union[np.ndarray, sparse.csc_matrix] = field(init=False)
    spatial_precision_matrix: np.ndarray = field(init=False)
    precision_time_0: float = field(init=False, default=0.01)

    def initialise(self, sensor_object: SensorGroup, meteorology: MeteorologyGroup, gas_species: GasSpecies):
        """Take data inputs and extract relevant properties.

        Args:
            sensor_object (SensorGroup): sensor data
            meteorology (MeteorologyGroup): meteorology data wind data
            gas_species (GasSpecies): gas species information

        """
        self.make_temporal_knots(sensor_object)
        self.make_spatial_knots(sensor_object)
        self.n_parameter = self.n_time * self.n_location
        self.n_obs = sensor_object.nof_observations

        self.make_precision_matrix()
        self.make_parameter_mapping(sensor_object)

        if self.mean_bg is None:
            self.mean_bg = gas_species.global_background

    def make_parameter_mapping(self, sensor_object: SensorGroup):
        """Create the mapping of parameters onto observations, through creation of the associated basis matrix.

        The background vector unwraps first over the spatial (sensor) location dimension, then over the temporal
        dimension. For more detail, see the main class docstring.

        The data vector in the solver state is assumed to consist of the individual sensor data vectors stacked
        consecutively.

        Args:
            sensor_object (SensorGroup): group of sensor objects.

        """
        nn_object = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(self.time.to_numpy().reshape(-1, 1))
        for k, sensor in enumerate(sensor_object.values()):
            _, time_index = nn_object.kneighbors(sensor.time.to_numpy().reshape(-1, 1))
            basis_matrix = sparse.csr_array(
                (np.ones(sensor.nof_observations), (np.array(range(sensor.nof_observations)), time_index.flatten())),
                shape=(sensor.nof_observations, self.n_time),
            )
            if self.spatial_dependence:
                basis_matrix = sparse.kron(basis_matrix, np.eye(N=self.n_location, M=1, k=-k).T)

            if k == 0:
                self.basis_matrix = basis_matrix
            else:
                self.basis_matrix = sparse.vstack([self.basis_matrix, basis_matrix])

    def make_temporal_knots(self, sensor_object: SensorGroup):
        """Create the temporal grid for the model.

        If self.n_time is not specified, then the model will use the unique set of times from the sensor data.

        If self.n_time is specified, then the model will define a time grid with self.n_time elements.

        Args:
            sensor_object (SensorGroup): group of sensor objects.

        """
        if self.n_time is None:
            self.time = pd.array(np.unique(sensor_object.time), dtype="datetime64[ns]")
            self.n_time = len(self.time)
        else:
            self.time = pd.array(
                pd.date_range(start=np.min(sensor_object.time), end=np.max(sensor_object.time), periods=self.n_time),
                dtype="datetime64[ns]",
            )

    def make_spatial_knots(self, sensor_object: SensorGroup):
        """Create the spatial grid for the model.

        If self.spatial_dependence is False, the code assumes that only a single (arbitrary) location is used, thereby
        eliminating any spatial dependence.

        If self.spatial_dependence is True, a separate but correlated time-series of background parameters is assumed
        for each sensor location.

        Args:
            sensor_object (SensorGroup): group of sensor objects.

        """
        if self.spatial_dependence:
            self.n_location = sensor_object.nof_sensors
            self.get_locations_from_sensors(sensor_object)
        else:
            self.n_location = 1
            self.location = sensor_object[list(sensor_object.keys())[0]].location

    def make_precision_matrix(self):
        """Create the full precision matrix for the background parameters.

        Defined as the Kronecker product of the temporal precision matrix and the spatial precision matrix.

        """
        self.temporal_precision_matrix = gmrf.precision_temporal(time=self.time)
        lam = self.temporal_precision_matrix[0, 0]
        self.temporal_precision_matrix[0, 0] = lam * (2.0 - lam / (self.precision_time_0 + lam))

        if self.spatial_dependence:
            self.make_spatial_precision_matrix()
            self.precision_matrix = sparse.kron(self.temporal_precision_matrix, self.spatial_precision_matrix)
        else:
            self.precision_matrix = self.temporal_precision_matrix
        if (self.n_parameter == 1) and sparse.issparse(self.precision_matrix):
            self.precision_matrix = self.precision_matrix.toarray()

    def make_spatial_precision_matrix(self):
        """Create the spatial precision matrix for the model.

        The spatial precision matrix is simply calculated as the inverse of a squared exponential covariance matrix
        calculated using the sensor locations.

        """
        location_array = self.location.to_array()
        spatial_covariance_matrix = np.exp(
            -(1 / (2 * np.power(self.spatial_correlation_param, 2)))
            * (
                np.power(location_array[:, [0]] - location_array[:, [0]].T, 2)
                + np.power(location_array[:, [1]] - location_array[:, [1]].T, 2)
                + np.power(location_array[:, [2]] - location_array[:, [2]].T, 2)
            )
        )
        self.spatial_precision_matrix = np.linalg.inv(
            spatial_covariance_matrix + (1e-6) * np.eye(spatial_covariance_matrix.shape[0])
        )

    def get_locations_from_sensors(self, sensor_object: SensorGroup):
        """Extract the location information from the sensor object.

        Attaches a Coordinate.ENU object as the self.location attribute, with all the sensor locations stored on the
        same object.

        Args:
            sensor_object (SensorGroup): group of sensor objects.

        """
        self.location = deepcopy(sensor_object[list(sensor_object.keys())[0]].location.to_enu())
        self.location.east = np.full(shape=(self.n_location,), fill_value=np.nan)
        self.location.north = np.full(shape=(self.n_location,), fill_value=np.nan)
        self.location.up = np.full(shape=(self.n_location,), fill_value=np.nan)
        for k, sensor in enumerate(sensor_object.values()):
            if isinstance(sensor, Beam):
                self.location.east[k] = np.mean(sensor.location.to_enu().east, axis=0)
                self.location.north[k] = np.mean(sensor.location.to_enu().north, axis=0)
                self.location.up[k] = np.mean(sensor.location.to_enu().up, axis=0)
            else:
                self.location.east[k] = sensor.location.to_enu().east
                self.location.north[k] = sensor.location.to_enu().north
                self.location.up[k] = sensor.location.to_enu().up
