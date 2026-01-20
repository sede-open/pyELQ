"""Unit Testing for the SiteLayout and MeteorologyWindfield classes."""

import numpy as np
import pytest

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.site_layout import SiteLayout
from pyelq.meteorology.meteorology import Meteorology
from pyelq.meteorology.meteorology_windfield import MeteorologyWindfield


@pytest.fixture(
    params=[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [[10.0, 6.0, 7.0]]],
    ids=["wind_u=1_v=0_w=0", "wind_2time_u=0_v=1_w=0", "wind_u=10_v=6_w=7"],
    name="meteorology",
)
def fixture_meteorology(request):
    """Create a wind component for the test."""
    wind_vector = np.array(request.param)
    meteorology = Meteorology()
    meteorology.u_component = wind_vector[:, 0]
    meteorology.v_component = wind_vector[:, 1]
    meteorology.w_component = wind_vector[:, 2]
    return meteorology


@pytest.fixture(name="grid_coordinates")
def fixture_grid_coordinates():
    """Fixture for grid coordinates."""
    return ENU(
        east=np.array([[0, 1, 2]]).T,
        north=np.array([[0, 1, 2]]).T,
        up=np.array([[0, 10, 20]]).T,
        ref_latitude=0,
        ref_longitude=0,
        ref_altitude=0,
    )


@pytest.fixture(params=[0, 10, 20], ids=["0-m", "10-m", "20-m"], name="height")
def fixture_height(request):
    """Fixture for the cylinder heights."""
    return request.param


@pytest.fixture(params=[0, 1, 2], ids=["0-cyl", "1-cyl", "2-cyl"], name="site_layout")
def fixture_site_layout(request, height, grid_coordinates):
    """Fixture for site layout."""
    number_cylinders = request.param
    radius = np.array([[1, 2, 3]]).T[:number_cylinders]
    east = np.array([[0, 0, 0]]).T[:number_cylinders]
    north = np.array([[0, 1, 2]]).T[:number_cylinders]
    height = height * np.ones_like(radius)

    site_layout = SiteLayout(
        cylinder_coordinates=ENU(
            east=east,
            north=north,
            up=height,
            ref_latitude=0,
            ref_longitude=0,
            ref_altitude=0,
        ),
        cylinder_radius=radius,
    )
    site_layout.find_index_obstacles(grid_coordinates)
    return site_layout


@pytest.fixture(name="meteorology_windfield")
def fixture_meteorology_windfield(site_layout, meteorology):
    """Fixture for MeteorologyWindfield."""
    return MeteorologyWindfield(site_layout=site_layout, static_wind_field=meteorology)


def test_find_index_obstacles(site_layout, grid_coordinates, height):
    """Test the find_index_obstacles method of SiteLayout.

    Test the method with different cylinder configurations and check the output.

    Check that grid points above the height of the cylinders are not considered obstacles.

    Pass cylinder locations in as grid coordinates and check that the method correctly identifies them as obstacles or
    not.

    """
    assert site_layout.id_obstacles.shape == (grid_coordinates.nof_observations, 1)
    assert site_layout.id_obstacles.dtype == bool
    assert not np.any(site_layout.id_obstacles[grid_coordinates.up > height])
    site_layout.find_index_obstacles(site_layout.cylinder_coordinates)
    assert np.all(site_layout.id_obstacles)


def test_meteorology_windfield(meteorology_windfield, meteorology, grid_coordinates):
    """Test the MeteorologyWindfield class.

    Check that the wind field components that get created by the class have the correct shapes.

    """
    meteorology_windfield.calculate_spatial_wind_field(grid_coordinates)
    assert meteorology_windfield.u_component.shape == (
        grid_coordinates.nof_observations,
        meteorology.u_component.shape[0],
    )
    assert meteorology_windfield.v_component.shape == (
        grid_coordinates.nof_observations,
        meteorology.u_component.shape[0],
    )
    assert meteorology_windfield.w_component.shape == (
        grid_coordinates.nof_observations,
        meteorology.u_component.shape[0],
    )

    meteorology_windfield.calculate_spatial_wind_field(grid_coordinates, time_index=0)
    assert meteorology_windfield.u_component.shape == (grid_coordinates.nof_observations, 1)
    assert meteorology_windfield.v_component.shape == (grid_coordinates.nof_observations, 1)
    assert meteorology_windfield.w_component.shape == (grid_coordinates.nof_observations, 1)
