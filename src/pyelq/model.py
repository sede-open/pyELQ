# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""ELQModel module.

This module provides a class definition for the main functionalities of the codebase, providing the interface with the
openMCMC repo and defining some plotting wrappers.

"""
import warnings
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from openmcmc import parameter
from openmcmc.distribution import location_scale
from openmcmc.mcmc import MCMC
from openmcmc.model import Model

from pyelq.component.background import Background, SpatioTemporalBackground
from pyelq.component.error_model import BySensor, ErrorModel
from pyelq.component.offset import PerSensor
from pyelq.component.source_model import Normal, SourceModel
from pyelq.gas_species import GasSpecies
from pyelq.meteorology import Meteorology, MeteorologyGroup
from pyelq.plotting.plot import Plot
from pyelq.sensor.sensor import SensorGroup


@dataclass
class ELQModel:
    """Class for setting up, running, and post-processing the full ELQModel analysis.

    Attributes:
        form (dict): dictionary detailing the form of the predictor for the concentration data. For details of the
            required specification, see parameter.LinearCombinationWithTransform() in the openMCMC repo.
        transform (dict): dictionary detailing transformations applied to the model components. For details of the
            required specification, see parameter.LinearCombinationWithTransform() in the openMCMC repo.
        model (Model): full model specification for the analysis, constructed in self.to_mcmc().
        mcmc (MCMC): MCMC object containing model and sampler specification for the problem. Constructed from the
            other components in self.to_mcmc().
        n_iter (int): number of MCMC iterations to be run.
        n_thin (int): number of iterations to thin by.
        fitted_values (np.ndarray): samples of fitted values (i.e. model predictions for the data) generated during the
            MCMC sampler. Attached in self.from_mcmc().

    """

    form: dict = field(init=False)
    transform: dict = field(init=False)
    model: Model = field(init=False)
    mcmc: MCMC = field(init=False)
    n_iter: int = 1000
    n_thin: int = 1
    fitted_values: np.ndarray = field(init=False)

    def __init__(
        self,
        sensor_object: SensorGroup,
        meteorology: Union[Meteorology, MeteorologyGroup],
        gas_species: GasSpecies,
        background: Background = SpatioTemporalBackground(),
        source_model: SourceModel = Normal(),
        error_model: ErrorModel = BySensor(),
        offset_model: PerSensor = None,
    ):
        """Initialise the ELQModel model.

        Model form is as follows:
        y = A*s + b + d + e
        where:
        - y is the vector of observed concentration data (extracted from the sensor object).
        - A*s is the source contribution (from the source model and dispersion model).
        - b is from the background model.
        - d is from the offset model.
        - e is residual error term and var(e) comes from the error precision model.

        Args:
            sensor_object (SensorGroup): sensor data.
            meteorology (Union[Meteorology, MeteorologyGroup]): meteorology data.
            gas_species (GasSpecies): gas species object.
            background (Background): background model specification. Defaults to SpatioTemporalBackground().
            source_model (SourceModel): source model specification. Defaults to Normal().
            error_model (Precision): measurement precision model specification. Defaults to BySensor().
            offset_model (PerSensor): offset model specification. Defaults to None.

        """
        self.sensor_object = sensor_object
        self.meteorology = meteorology
        self.gas_species = gas_species
        self.components = {
            "background": background,
            "source": source_model,
            "error_model": error_model,
            "offset": offset_model,
        }
        if error_model is None:
            self.components["error_model"] = BySensor()
            warnings.warn("None is not an allowed type for error_model: resetting to default BySensor model.")
        for key in list(self.components.keys()):
            if self.components[key] is None:
                self.components.pop(key)

    def initialise(self):
        """Take data inputs and extract relevant properties."""
        self.form = {}
        self.transform = {}
        component_keys = list(self.components.keys())
        if "background" in component_keys:
            self.form["bg"] = "B_bg"
            self.transform["bg"] = False
        if "source" in component_keys:
            self.transform["s"] = False
            self.form["s"] = "A"
        if "offset" in component_keys:
            self.form["d"] = "B_d"
            self.transform["d"] = False
        for key in component_keys:
            self.components[key].initialise(self.sensor_object, self.meteorology, self.gas_species)

    def to_mcmc(self):
        """Convert the ELQModel specification into an MCMC solver object that can be run.

        Executing the following steps:
            - Initialise the model object with the data likelihood (response distribution for y), and add all the
                associated prior distributions, as specified by the model components.
            - Initialise the state dictionary with the observed sensor data, and add parameters associated with all
                the associated prior distributions, as specified by the model components.
            - Initialise the MCMC sampler objects associated with each of the model components.
            - Create the MCMC solver object, using all of the above information.

        """
        response_precision = self.components["error_model"].precision_parameter
        model = [
            location_scale.Normal(
                "y",
                mean=parameter.LinearCombinationWithTransform(self.form, self.transform),
                precision=response_precision,
            )
        ]

        initial_state = {"y": self.sensor_object.concentration}

        for component in self.components.values():
            model = component.make_model(model)
            initial_state = component.make_state(initial_state)

        self.model = Model(model, response={"y": "mean"})

        sampler_list = []
        for component in self.components.values():
            sampler_list = component.make_sampler(self.model, sampler_list)

        self.mcmc = MCMC(initial_state, sampler_list, self.model, n_burn=0, n_iter=self.n_iter, n_thin=self.n_thin)

    def run_mcmc(self):
        """Run the mcmc function."""
        self.mcmc.run_mcmc()

    def from_mcmc(self):
        """Extract information from MCMC solver class once its has run.

        Performs two operations:
            - For each of the components of the model: extracts the related sampled parameter values and attaches these
                to the component class.
            - For all keys in the mcmc.store dictionary: extracts the sampled parameter values from self.mcmc.store and
                puts them into the equivalent fields in the state

        """
        state = self.mcmc.state
        for component in self.components.values():
            component.from_mcmc(self.mcmc.store)
        for key in self.mcmc.store:
            state[key] = self.mcmc.store[key]

    def plot_log_posterior(self, burn_in_value: int, plot: Plot = Plot()) -> Plot():
        """Plots the trace of the log posterior over the iterations of the MCMC.

        Args:
            burn_in_value (int): Burn in value to show in plot.
            plot (Plot, optional): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with
                key 'log_posterior_plot'

        """
        plot.plot_single_trace(object_to_plot=self.mcmc, burn_in=burn_in_value)
        return plot

    def plot_fitted_values(self, plot: Plot = Plot()) -> Plot:
        """Plot the fitted values from the mcmc object against time, also shows the estimated background when possible.

        Based on the inputs it plots the results of the mcmc analysis, being the fitted values of the concentration
        measurements together with the 10th and 90th quantile lines to show the goodness of fit of the estimates.

        Args:
            plot (Plot, optional): Plot object to which this figure will be added in the figure dictionary

        Returns:
            plot (Plot): Plot object to which this figure is added in the figure dictionary with key 'fitted_values'

        """
        plot.plot_fitted_values_per_sensor(
            mcmc_object=self.mcmc, sensor_object=self.sensor_object, background_model=self.components["background"]
        )
        return plot
