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
            wind_speed (np.ndarray): wind speed at source location [m/s]
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

    Attributes:
        ground_threshold (float): height for a source to be considered ground, informs which
            scale, exponent and t_i should be used for a given source [m]

        The following parameters are all of type float and are given in pairs. Their use in the calculation
        is dependent on the location of the source relative to ground, and the threshold given by the parameter above.

            scale_ground, scale_air
            exp_ground, exp_air
            t_i_ground, t_i_air

        The calculation and default parameters are derived from:

        Determination of Atmospheric Diffusion Parameters, R. R. Draxler, DOI: 10.1016/0004-6981(76)90226-2

    """

    scale_ground: float
    scale_air: float
    exp_ground: float
    exp_air: float
    t_i_ground: float
    t_i_air: float

    ground_threshold: float = 0.5

    DEFAULT_DRAXLER_HORIZONTAL = {
        "scale_ground": 0.9,
        "scale_air": 0.9,
        "exp_ground": 0.5,
        "exp_air": 0.5,
        "t_i_ground": 300.0,
        "t_i_air": 1000.0,
    }
    DEFAULT_DRAXLER_VERTICAL = {
        "scale_ground": 0.9,
        "scale_air": 0.945,
        "exp_ground": 0.5,
        "exp_air": 0.806,
        "t_i_ground": 50.0,
        "t_i_air": 100.0,
    }

    def calculate(
        self, turbulence_vector: np.ndarray, source_z: np.ndarray, wind_speed: np.ndarray, distance_x: np.ndarray
    ) -> np.ndarray:
        """Draxler calculation for atmospheric diffusion. See DOI above for details.
        
        Expects that 'turbulence_vector' is in m/s
        
        """

        positive_part_x = np.maximum(distance_x, 0.0)
        diff_time = positive_part_x / wind_speed

        is_ground_source: np.ndarray[bool] = source_z <= self.ground_threshold
        scale = np.where(is_ground_source, self.scale_ground, self.scale_air)
        power = np.where(is_ground_source, self.exp_ground, self.exp_air)
        t_i = np.where(is_ground_source, self.t_i_ground, self.t_i_air)

        f = 1.0 / (1.0 + scale * ((diff_time / t_i) ** power))

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
