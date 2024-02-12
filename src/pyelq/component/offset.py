# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Offset module."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from openmcmc import parameter
from openmcmc.distribution.distribution import Gamma
from openmcmc.distribution.location_scale import Normal
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalGamma, NormalNormal
from scipy import sparse

from pyelq.component.component import Component
from pyelq.gas_species import GasSpecies
from pyelq.meteorology import Meteorology
from pyelq.sensor.sensor import Sensor, SensorGroup

if TYPE_CHECKING:
    from pyelq.plotting.plot import Plot


@dataclass
class PerSensor(Component):
    """Offset implementation which assumes an additive offset between sensors.

    The offset is which is constant in space and time and accounts for calibration differences between sensors.
    To maintain parameter identifiability, the offset for the first sensor (with index 0) is assumed to be 0, and other
    sensor offsets are defined relative to this beam.

    Attributes:
        n_sensor (int): number of sensors in the sensor object used for analysis.
        offset (np.ndarray): array of sampled offset values, populated in self.from_mcmc() after the MCMC run is
            completed.
        precision_scalar (np.ndarray): array of sampled offset precision values, populated in self.from_mcmc() after
            the MCMC run is completed. Only populated if update_precision is True.
        indicator_basis (sparse.csc_matrix): [nof_observations x (nof_sensors - 1)] sparse matrix which assigns the
            offset parameters to the correct observations.
        update_precision (bool): logical indicating whether the offset prior precision parameter should be updated as
            part of the analysis.
        mean_offset (float): prior mean parameter for the offsets, assumed to be the same for each beam. Default is 0.
        prior_precision_shape (float): shape parameter for the prior gamma distribution for the scalar precision
            parameter. Default is 1e-3.
        prior_precision_rate (float): rate parameter for the prior gamma distribution for the scalar precision
            parameter(s). Default is 1e-3.
        initial_precision (float): initial value for the scalar precision parameter. Default is 1.0.

    """

    n_sensor: int = field(init=False)
    offset: np.ndarray = field(init=False)
    precision_scalar: np.ndarray = field(init=False)
    indicator_basis: sparse.csc_matrix = field(init=False)
    update_precision: bool = False
    mean_offset: float = 0.0
    prior_precision_shape: float = 1e-3
    prior_precision_rate: float = 1e-3
    initial_precision: float = 1.0

    def initialise(self, sensor_object: SensorGroup, meteorology: Meteorology, gas_species: GasSpecies):
        """Take data inputs and extract relevant properties.

        Args:
            sensor_object (SensorGroup): sensor data
            meteorology (MeteorologyGroup): meteorology data wind data
            gas_species (GasSpecies): gas species information

        """
        self.n_sensor = len(sensor_object)
        self.indicator_basis = sparse.csc_matrix(
            np.equal(sensor_object.sensor_index[:, np.newaxis], np.array(range(1, self.n_sensor)))
        )

    def make_model(self, model: list = None) -> list:
        """Take model list and append new elements from current model component.

        Args:
            model (list, optional): Current list of model elements. Defaults to [].

        Returns:
            list: model output list.

        """
        if model is None:
            model = []
        off_precision_predictor = parameter.ScaledMatrix(matrix="P_d", scalar="lambda_d")
        model.append(Normal("d", mean="mu_d", precision=off_precision_predictor))
        if self.update_precision:
            model.append(Gamma("lambda_d", shape="a_lam_d", rate="b_lam_d"))
        return model

    def make_sampler(self, model: Model, sampler_list: list = None) -> list:
        """Take sampler list and append new elements from current model component.

        Args:
            model (Model): Full model list of distributions.
            sampler_list (list, optional): Current list of samplers. Defaults to [].

        Returns:
            list: sampler output list.

        """
        if sampler_list is None:
            sampler_list = []
        sampler_list.append(NormalNormal("d", model))
        if self.update_precision:
            sampler_list.append(NormalGamma("lambda_d", model))
        return sampler_list

    def make_state(self, state: dict = None) -> dict:
        """Take state dictionary and append initial values from model component.

        Args:
            state (dict, optional): current state vector. Defaults to {}.

        Returns:
            dict: current state vector with components added.

        """
        if state is None:
            state = {}
        state["mu_d"] = np.ones((self.n_sensor - 1, 1)) * self.mean_offset
        state["d"] = np.zeros((self.n_sensor - 1, 1))
        state["B_d"] = self.indicator_basis
        state["P_d"] = sparse.eye(self.n_sensor - 1, format="csc")
        state["lambda_d"] = self.initial_precision
        if self.update_precision:
            state["a_lam_d"] = self.prior_precision_shape
            state["b_lam_d"] = self.prior_precision_rate
        return state

    def from_mcmc(self, store: dict):
        """Extract results of mcmc from mcmc.store and attach to components.

        Args:
            store (dict): mcmc result dictionary.

        """
        self.offset = store["d"]
        if self.update_precision:
            self.precision_scalar = store["lambda_d"]

    def plot_iterations(self, plot: "Plot", sensor_object: Union[SensorGroup, Sensor], burn_in_value: int) -> "Plot":
        """Plots the offset values for every sensor with respect to the MCMC iterations.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the offset_model
            burn_in_value (int): Burn in value to show in plot.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'offset_iterations'

        """
        plot.plot_trace_per_sensor(
            object_to_plot=self, sensor_object=sensor_object, plot_type="line", burn_in=burn_in_value
        )

        return plot

    def plot_distributions(self, plot: "Plot", sensor_object: Union[SensorGroup, Sensor], burn_in_value: int) -> "Plot":
        """Plots the distribution of the offset values after the burn in for every sensor.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the offset_model
            burn_in_value (int): Burn in value to use for plot.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'offset_distributions'

        """
        plot.plot_trace_per_sensor(
            object_to_plot=self, sensor_object=sensor_object, plot_type="box", burn_in=burn_in_value
        )

        return plot
