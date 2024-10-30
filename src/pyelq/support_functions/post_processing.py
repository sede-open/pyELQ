# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Post-processing module.

Module containing some functions used in post-processing of the results.

"""
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Type, Union

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from geojson import Feature, FeatureCollection
from openmcmc.mcmc import MCMC
from scipy.ndimage import label
from shapely import geometry

from pyelq.component.background import TemporalBackground
from pyelq.component.error_model import ErrorModel
from pyelq.component.offset import PerSensor
from pyelq.component.source_model import SlabAndSpike, SourceModel
from pyelq.coordinate_system import ENU, LLA
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.sensor.sensor import Sensor, SensorGroup

if TYPE_CHECKING:
    from pyelq.model import ELQModel


def is_regularly_spaced(array: np.ndarray, tolerance: float = 0.01, return_delta: bool = True):
    """Determines whether an input array is regularly spaced, within some (absolute) tolerance.

    Gets the large differences (defined by tolerance) in the array, and sees whether all of them are within 5% of one
    another.

    Args:
        array (np.ndarray): Input array to be analysed.
        tolerance (float, optional): Absolute value above which the difference between values is considered significant.
            Defaults to 0.01.
        return_delta (bool, optional): Whether to return the value of the regular grid spacing. Defaults to True.

    Returns:
        (bool): Whether the grid is regularly spaced.
        (float): The value of the regular grid spacing.

    """
    unique_vals = np.unique(array)
    diff_unique_vals = np.diff(unique_vals)
    diff_big = diff_unique_vals[diff_unique_vals > tolerance]

    boolean = np.all([np.isclose(diff_big[i], diff_big[i + 1], rtol=0.05) for i in range(len(diff_big) - 1)])

    if return_delta:
        return boolean, np.mean(diff_big)

    return boolean, None
