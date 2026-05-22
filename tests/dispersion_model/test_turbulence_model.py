"""Test module for TurbulenceModel objects."""

import numpy as np
import pytest

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.turbulence_model import AngularModel, DraxlerModel
from tests.dispersion_model.test_gaussian_plume import make_met_object


@pytest.mark.parametrize("n_wind", [1, 10], ids=["1", "10"])
@pytest.mark.parametrize("parameters", [{}, {"scale_ground": 0.8, "exp_ground": 0.4}], ids=["default", "override"])
@pytest.mark.parametrize("type", ["horizontal", "vertical"], ids=["horizontal", "vertical"])
def test_draxler_model(n_wind, parameters, type):
    """Tests for the Draxler turbulence model.

    Performs the following checks:
        - A DraxlerModel can be instantiated successfully with both default and overridden parameters,
            in both the horizontal and vertical cases.
        - The scale_ground and exp_ground parameters are correctly set either to the default or
            user-supplied values (horizontal and vertical defaults are equivalent for these parameters).
        - The calculated plume spread has the expected shape.
        - The universal turbulence function f has the expected values in the case where the wind speed and
            turbulence_vector are both set to 1.0, for both ground and elevated sources.

    """
    if type == "horizontal":
        model = DraxlerModel.default_horizontal(**parameters)
    else:
        model = DraxlerModel.default_vertical(**parameters)
    turbulence_vector = np.ones((n_wind, 1)) * 1.0
    wind_speed = np.ones((n_wind, 1)) * 1.0
    source_z = np.linspace(0, 3, n_wind).reshape((n_wind, 1))
    distance_x = np.linspace(1, 50, n_wind).reshape((n_wind, 1))
    plume_spread = model.calculate(
        turbulence_vector=turbulence_vector, source_z=source_z, wind_speed=wind_speed, distance_x=distance_x
    )

    if parameters:
        assert model.scale_ground == parameters["scale_ground"]
        assert model.exp_ground == parameters["exp_ground"]
    else:
        assert model.scale_ground == DraxlerModel.DEFAULT_DRAXLER_HORIZONTAL["scale_ground"]
        assert model.exp_ground == DraxlerModel.DEFAULT_DRAXLER_HORIZONTAL["exp_ground"]

    idx_ground_source = source_z[:, 0] <= model.ground_threshold
    f = plume_spread * (wind_speed / turbulence_vector) / distance_x
    expected_f_ground = 1.0 / (1.0 + model.scale_ground * ((distance_x / model.t_i_ground) ** model.exp_ground))
    expected_f_elevated = 1.0 / (1.0 + model.scale_elevated * ((distance_x / model.t_i_elevated) ** model.exp_elevated))
    assert plume_spread.shape == (n_wind, 1)
    assert np.allclose(f[idx_ground_source, 0], expected_f_ground[idx_ground_source, 0])
    assert np.allclose(f[~idx_ground_source, 0], expected_f_elevated[~idx_ground_source, 0])


@pytest.mark.parametrize("n_wind", [1, 10], ids=["1", "10"])
def test_angular_model(n_wind):
    """Tests for the Angular turbulence model.

    Performs the following checks:
        - An AngularModel can be instantiated successfully.
        - The calculated plume spread has the expected shape and is positive.

    """
    model = AngularModel()
    source_z = np.linspace(0, 3, n_wind).reshape((n_wind, 1))
    distance_x = np.linspace(0, 100, n_wind).reshape((n_wind, 1))
    turbulence_vector = np.ones((n_wind, 1)) * 10
    wind_speed = np.ones((n_wind, 1)) * 1
    plume_spread = model.calculate(
        turbulence_vector=turbulence_vector, source_z=source_z, wind_speed=wind_speed, distance_x=distance_x
    )

    assert plume_spread.shape == (n_wind, 1)
    assert np.all(plume_spread > 0)
