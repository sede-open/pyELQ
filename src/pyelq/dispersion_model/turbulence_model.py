"""Turbulence Models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

@dataclass
class TurbulenceModel(ABC):
    """Abstract class to define specific turbulence model behaviour"""

    @abstractmethod
    def calculate(
        self, turbulence_vector: np.ndarray, source_z: np.ndarray, wind_speed: np.ndarray, distance_x: np.ndarray
    ) -> np.ndarray:
        """Abstract method to define behaviour for turbulence calculation. All inputs are
        expected to have the same shape.

        Args:
            turbulence_vector (np.ndarray): wind turbulence parameter [unit depends on model]
            source_z (np.ndarray): height of source relative to ground [m]
            wind_speed (np.ndarray): wind speed at sensor location [m/s]
            distance_x (np.ndarray): distance along plume axis from source to sensor location [m]

        Returns:
            np.ndarray: stability of the plume according to the specific model in use

        """


@dataclass
class AngularModel(TurbulenceModel):
    """Basic turbulence model"""

    def calculate(
        self, turbulence_vector: np.ndarray, source_z: np.ndarray, wind_speed: np.ndarray, distance_x: np.ndarray
    ):
        """Calculate the effective wind component perpendicular to the wind direction at the source location.

        Expects that 'turbulence_vector' is in degrees.

        """

        return np.tan(turbulence_vector * (np.pi / 180)) * np.abs(distance_x)


@dataclass
class DraxlerModel(TurbulenceModel):
    """Draxler Turbulence Model

    This turbulence model attempts to characterise the dispersion of the plume based on the travel
    time (T) of the gas from source location to sensor.

    The calculation and default parameters used in this class are derived from 'Determination of Atmospheric
    Diffusion Parameters, R. R. Draxler, DOI: 10.1016/0004-6981(76)90226-2'.

    The calculation relies on the use of universal functions that are used to characterise the turbulence
    in the horizontal and vertical directions under particular conditions of atmospheric stability and source
    heights. The functions take the form:

        f = 1 / (1 + scale * (T/t_i) ** exp)

    Taking T as input, and as parameters:
        scale: an increase in scale reduces the effective dispersion along the entire plume, constraining
            plume growth with distance
        exp: an increase in exp creates a more abrupt transition point from greater to lesser dispersion
        t_i: the time scale at which the transition from greater to lower dispersion is expected to occur

    This class differentiates between the stability parameters for ground and elevated sources, and provides
    defaults for both of these in both the horizontal and vertical planes. The scale and exp values are drawn
    from Equations 2.7 and 2.9 in the Draxler paper and the t_i values from Table 3.

    Attributes:
        scale_ground (float): the scale parameter as described above for a ground source
        exp_ground (float): the exp parameter as described above for a ground source
        t_i_ground (float): the t_i parameter as described above for a ground source
        scale_elevated (float): the scale parameter as described above for an elevated source
        exp_elevated (float): the exp parameter as described above for an elevated source
        t_i_elevated (float): the t_i parameter as described above for an elevated source
        ground_threshold (float): maximum height above ground at which source can be considered 'ground'

    """

    scale_ground: float
    scale_elevated: float
    exp_ground: float
    exp_elevated: float
    t_i_ground: float
    t_i_elevated: float
    ground_threshold: float = 0.5

    DEFAULT_DRAXLER_HORIZONTAL = {
        "scale_ground": 0.9,
        "scale_elevated": 0.9,
        "exp_ground": 0.5,
        "exp_elevated": 0.5,
        "t_i_ground": 300.0,
        "t_i_elevated": 1000.0,
    }
    DEFAULT_DRAXLER_VERTICAL = {
        "scale_ground": 0.9,
        "scale_elevated": 0.945,
        "exp_ground": 0.5,
        "exp_elevated": 0.806,
        "t_i_ground": 50.0,
        "t_i_elevated": 100.0,
    }

    def calculate(
        self, turbulence_vector: np.ndarray, source_z: np.ndarray, wind_speed: np.ndarray, distance_x: np.ndarray
    ) -> np.ndarray:
        """Draxler calculation for atmospheric diffusion. See DOI above for details.
        
        Expects that 'turbulence_vector' is in m/s
        
        """

        positive_part_x = np.maximum(distance_x, 0.0)
        diffusion_time = positive_part_x / wind_speed

        is_ground_source: np.ndarray[bool] = source_z <= self.ground_threshold
        scale = np.where(is_ground_source, self.scale_ground, self.scale_elevated)
        power = np.where(is_ground_source, self.exp_ground, self.exp_elevated)
        t_i = np.where(is_ground_source, self.t_i_ground, self.t_i_elevated)

        f = 1.0 / (1.0 + scale * ((diffusion_time / t_i) ** power))

        return (turbulence_vector / wind_speed) * positive_part_x * f

    @staticmethod
    def default_horizontal(**kwargs):
        """Construct a DraxlerModel object for horizontal turbulence, overriding defaults where provided."""

        params = DraxlerModel.DEFAULT_DRAXLER_HORIZONTAL | kwargs
        return DraxlerModel(**params)

    @staticmethod
    def default_vertical(**kwargs):
        """Construct a DraxlerModel object for vertical turbulence, overriding defaults where provided."""

        params = DraxlerModel.DEFAULT_DRAXLER_VERTICAL | kwargs
        return DraxlerModel(**params)
