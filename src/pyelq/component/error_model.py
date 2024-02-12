# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Error model module."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from openmcmc import parameter
from openmcmc.distribution.distribution import Gamma
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalGamma

from pyelq.component.component import Component
from pyelq.gas_species import GasSpecies
from pyelq.meteorology import MeteorologyGroup
from pyelq.sensor.sensor import Sensor, SensorGroup

if TYPE_CHECKING:
    from pyelq.plotting.plot import Plot


@dataclass
class ErrorModel(Component):
    """Measurement precision model component for the model.

    Attributes:
        n_sensor (int): number of sensors in the sensor object used for analysis.
        precision_index (np.ndarray): index mapping precision parameters onto observations. Will be set up differently
            for different model types.
        precision_parameter (parameter.Parameter): parameter object which constructs the full measurement error
            precision matrix from the components stored in state. Will be passed to the distribution for the observed
            when the full model is constructed.
        prior_precision_shape (Union[np.ndarray, float]): prior shape parameters for the precision model. Set up
            differently per model type.
        prior_precision_rate (Union[np.ndarray, float]): prior rate parameters for the precision model. Set up
            differently per model type.
        initial_precision (Union[np.ndarray, float]): initial value for the precision to be passed to the analysis
            routine. Set up differently per model type.
        precision (np.ndarray): array of sampled measurement error precision values, populated in self.from_mcmc() after
            the MCMC run is completed.

    """

    n_sensor: int = field(init=False)
    precision_index: np.ndarray = field(init=False)
    precision_parameter: parameter.Parameter = field(init=False)
    prior_precision_shape: Union[np.ndarray, float] = field(init=False)
    prior_precision_rate: Union[np.ndarray, float] = field(init=False)
    initial_precision: Union[np.ndarray, float] = field(init=False)
    precision: np.ndarray = field(init=False)

    def initialise(
        self, sensor_object: SensorGroup, meteorology: MeteorologyGroup = None, gas_species: GasSpecies = None
    ):
        """Take data inputs and extract relevant properties.

        Args:
            sensor_object (SensorGroup): sensor data.
            meteorology (MeteorologyGroup): meteorology data. Defaults to None.
            gas_species (GasSpecies): gas species information. Defaults to None.

        """
        self.n_sensor = sensor_object.nof_sensors

    def make_model(self, model: list = None) -> list:
        """Take model list and append new elements from current model component.

        Args:
            model (list, optional): Current list of model elements. Defaults to None.

        Returns:
            list: model output list.

        """
        if model is None:
            model = []
        model.append(Gamma("tau", shape="a_tau", rate="b_tau"))
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
        sampler_list.append(NormalGamma("tau", model))
        return sampler_list

    def make_state(self, state: dict = None) -> dict:
        """Take state dictionary and append initial values from model component.

        Args:
            state (dict, optional): current state vector. Defaults to None.

        Returns:
            dict: current state vector with components added.

        """
        if state is None:
            state = {}
        state["a_tau"] = self.prior_precision_shape.flatten()
        state["b_tau"] = self.prior_precision_rate.flatten()
        state["precision_index"] = self.precision_index
        state["tau"] = self.initial_precision.flatten()
        return state

    def from_mcmc(self, store: dict):
        """Extract results of mcmc from mcmc.store and attach to components.

        Args:
            store (dict): mcmc result dictionary.

        """
        self.precision = store["tau"]


@dataclass
class BySensor(ErrorModel):
    """Version of measurement precision where each sensor object has a different precision.

    Attributes:
        prior_precision_shape (Union[np.ndarray, float]): prior shape parameters for the precision model, can be
            specified either as a float or as a (nof_sensors, ) np.ndarray: a float specification will result in
            the same parameter value for each sensor. Defaults to 1e-3.
        prior_precision_rate (Union[np.ndarray, float]): prior rate parameters for the precision model, can be
            specified either as a float or as a (nof_sensors, ) np.ndarray: a float specification will result in
            the same parameter value for each sensor. Defaults to 1e-3.
        initial_precision (Union[np.ndarray, float]): initial value for the precision parameters, can be specified
            either as a float or as a (nof_sensors, ) np.ndarray: a float specification will result in the same
            parameter value for each sensor. Defaults to 1.
        precision_index (np.ndarray): index mapping precision parameters onto observations. Parameters 1:n_sensor are
            mapped as the measurement error precisions of the corresponding sensors.
        precision_parameter (Parameter.MixtureParameterMatrix): parameter specification for this model, maps the
            current value of the parameter in the state dict onto the concentration data precisions.

    """

    prior_precision_shape: Union[np.ndarray, float] = 1e-3
    prior_precision_rate: Union[np.ndarray, float] = 1e-3
    initial_precision: Union[np.ndarray, float] = 1.0

    def initialise(
        self, sensor_object: SensorGroup, meteorology: MeteorologyGroup = None, gas_species: GasSpecies = None
    ):
        """Set up the error model using sensor properties.

        Args:
            sensor_object (SensorGroup): sensor data.
            meteorology (MeteorologyGroup): meteorology data. Defaults to None.
            gas_species (GasSpecies): gas species information. Defaults to None.

        """
        super().initialise(sensor_object=sensor_object, meteorology=meteorology, gas_species=gas_species)
        self.prior_precision_shape = self.prior_precision_shape * np.ones((self.n_sensor,))
        self.prior_precision_rate = self.prior_precision_rate * np.ones((self.n_sensor,))
        self.initial_precision = self.initial_precision * np.ones((self.n_sensor,))
        self.precision_index = sensor_object.sensor_index
        self.precision_parameter = parameter.MixtureParameterMatrix(param="tau", allocation="precision_index")

    def plot_iterations(self, plot: "Plot", sensor_object: Union[SensorGroup, Sensor], burn_in_value: int) -> "Plot":
        """Plots the error model values for every sensor with respect to the MCMC iterations.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the error_model
            burn_in_value (int): Burn in value to show in plot.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'error_model_iterations'

        """
        plot.plot_trace_per_sensor(
            object_to_plot=self, sensor_object=sensor_object, plot_type="line", burn_in=burn_in_value
        )

        return plot

    def plot_distributions(self, plot: "Plot", sensor_object: Union[SensorGroup, Sensor], burn_in_value: int) -> "Plot":
        """Plots the distribution of the error model values after the burn in for every sensor.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the error_model
            burn_in_value (int): Burn in value to show in plot.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'error_model_distributions'

        """
        plot.plot_trace_per_sensor(
            object_to_plot=self, sensor_object=sensor_object, plot_type="box", burn_in=burn_in_value
        )

        return plot


@dataclass
class ByRelease(ErrorModel):
    """ByRelease error model, special case of the measurement precision model.

    Version of the measurement precision model where each sensor object has a different precision, and there are
    different precisions for periods inside and outside controlled release periods. For all parameters: the first
    element corresponds to the case where the sources are OFF; the second element corresponds to the case where the
    sources are ON.

    Attributes:
        prior_precision_shape (np.ndarray): prior shape parameters for the precision model, can be
            specified either as a (2, 1) np.ndarray or as a (2, nof_sensors) np.ndarray: the former specification
            will result in the same prior specification for the off/on precisions for each sensor. Defaults to
            np.array([1e-3, 1e-3]).
        prior_precision_rate (np.ndarray): prior rate parameters for the precision model, can be
            specified either as a (2, 1) np.ndarray or as a (2, nof_sensors) np.ndarray: the former specification
            will result in the same prior specification for the off/on precisions for each sensor. Defaults to
            np.array([1e-3, 1e-3]).
        initial_precision (np.ndarray): initial value for the precision parameters, can be
            specified either as a (2, 1) np.ndarray or as a (2, nof_sensors) np.ndarray: the former specification
            will result in the same prior specification for the off/on precisions for each sensor. Defaults to
            np.array([1.0, 1.0]).
        precision_index (np.ndarray): index mapping precision parameters onto observations. Parameters 1:n_sensor are
            mapped onto each sensor for the periods where the sources are OFF; parameters (n_sensor + 1):(2 * n_sensor)
            are mapped onto each sensor for the periods where the sources are ON.
        precision_parameter (Parameter.MixtureParameterMatrix): parameter specification for this model, maps the
            current value of the parameter in the state dict onto the concentration data precisions.

    """

    prior_precision_shape: np.ndarray = field(default_factory=lambda: np.array([1e-3, 1e-3], ndmin=2).T)
    prior_precision_rate: np.ndarray = field(default_factory=lambda: np.array([1e-3, 1e-3], ndmin=2).T)
    initial_precision: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0], ndmin=2).T)

    def initialise(
        self, sensor_object: SensorGroup, meteorology: MeteorologyGroup = None, gas_species: GasSpecies = None
    ):
        """Set up the error model using sensor properties.

        Args:
            sensor_object (SensorGroup): sensor data.
            meteorology (MeteorologyGroup): meteorology data. Defaults to None.
            gas_species (GasSpecies): gas species information. Defaults to None.

        """
        super().initialise(sensor_object=sensor_object, meteorology=meteorology, gas_species=gas_species)
        self.prior_precision_shape = self.prior_precision_shape * np.ones((2, self.n_sensor))
        self.prior_precision_rate = self.prior_precision_rate * np.ones((2, self.n_sensor))
        self.initial_precision = self.initial_precision * np.ones((2, self.n_sensor))
        self.precision_index = sensor_object.sensor_index + sensor_object.source_on * self.n_sensor
        self.precision_parameter = parameter.MixtureParameterMatrix(param="tau", allocation="precision_index")

    def plot_iterations(self, plot: "Plot", sensor_object: Union[SensorGroup, Sensor], burn_in_value: int) -> "Plot":
        """Plot the estimated error model parameters against iterations of the MCMC chain.

        Works by simply creating a separate plot for each of the two categories of precision parameter (when the
        sources are on/off). Creates a BySensor() object for each of the off/on precision cases, and then makes a
        call to its plot function.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the error_model
            burn_in_value (int): Burn in value to show in plot.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'error_model_iterations'

        """
        figure_keys = ["error_model_off_iterations", "error_model_on_iterations"]
        figure_titles = [
            "Estimated error parameter values: sources off",
            "Estimated error parameter values: sources on",
        ]
        precision_arrays = [
            self.precision[: sensor_object.nof_sensors, :],
            self.precision[sensor_object.nof_sensors :, :],
        ]
        for key, title, array in zip(figure_keys, figure_titles, precision_arrays):
            error_model = BySensor()
            error_model.precision = array
            plot = error_model.plot_iterations(plot, sensor_object, burn_in_value)
            plot.figure_dict[key] = plot.figure_dict.pop("error_model_iterations")
            plot.figure_dict[key].update_layout(title=title)
        return plot

    def plot_distributions(self, plot: "Plot", sensor_object: Union[SensorGroup, Sensor], burn_in_value: int) -> "Plot":
        """Plot the estimated distributions of error model parameters.

        Works by simply creating a separate plot for each of the two categories of precision parameter (when the
        sources are off/on). Creates a BySensor() object for each of the off/on precision cases, and then makes a
        call to its plot function.

        Args:
            sensor_object (Union[SensorGroup, Sensor]): Sensor object associated with the error_model
            burn_in_value (int): Burn in value to show in plot.
            plot (Plot): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'error_model_distributions'

        """
        figure_keys = ["error_model_off_distributions", "error_model_on_distributions"]
        figure_titles = [
            "Estimated error parameter distribution: sources off",
            "Estimated error parameter distribution: sources on",
        ]
        precision_arrays = [
            self.precision[: sensor_object.nof_sensors, :],
            self.precision[sensor_object.nof_sensors :, :],
        ]
        for key, title, array in zip(figure_keys, figure_titles, precision_arrays):
            error_model = BySensor()
            error_model.precision = array
            plot = error_model.plot_distributions(plot, sensor_object, burn_in_value)
            plot.figure_dict[key] = plot.figure_dict.pop("error_model_distributions")
            plot.figure_dict[key].update_layout(title=title)
        return plot
