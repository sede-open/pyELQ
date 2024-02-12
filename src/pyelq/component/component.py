# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Superclass for model components."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openmcmc.model import Model

from pyelq.gas_species import GasSpecies
from pyelq.meteorology import MeteorologyGroup
from pyelq.sensor.sensor import SensorGroup


@dataclass
class Component(ABC):
    """Abstract class defining methods and rules for model elements.

    The bulk of attributes will be defined in the subclasses inheriting from this superclass.

    """

    @abstractmethod
    def initialise(self, sensor_object: SensorGroup, meteorology: MeteorologyGroup, gas_species: GasSpecies):
        """Take data inputs and extract relevant properties.

        Args:
            sensor_object (SensorGroup): sensor data
            meteorology (MeteorologyGroup): meteorology data
            gas_species (GasSpecies): gas species information

        """

    @abstractmethod
    def make_model(self, model: list) -> list:
        """Take model list and append new elements from current model component.

        Args:
            model (list, optional): Current list of model elements. Defaults to [].

        Returns:
            list: model output list.

        """

    @abstractmethod
    def make_sampler(self, model: Model, sampler_list: list) -> list:
        """Take sampler list and append new elements from current model component.

        Args:
            model (Model): Full model list of distributions.
            sampler_list (list, optional): Current list of samplers. Defaults to [].

        Returns:
            list: sampler output list.

        """

    @abstractmethod
    def make_state(self, state: dict) -> dict:
        """Take state dictionary and append initial values from model component.

        Args:
            state (dict, optional): current state vector. Defaults to {}.

        Returns:
            dict: current state vector with components added.

        """

    @abstractmethod
    def from_mcmc(self, store: dict):
        """Extract results of mcmc from mcmc.store and attach to components.

        Args:
            store (dict): mcmc result dictionary.

        """
