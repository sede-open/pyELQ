"""Test module for finite volume dispersion model.

This module provides various tests for the Finite volume related code part of pyELQ

"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.finite_volume import (
    FiniteVolume,
    FiniteVolumeDimension,
    FiniteVolumeFace,
    FiniteVolumeFaceLeft,
    FiniteVolumeFaceRight,
)
from pyelq.gas_species import CH4
from pyelq.meteorology import Meteorology
from pyelq.meteorology_windfield import MeteorologyWindfield, SiteLayout
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap


@pytest.fixture(
    params=[[[10.0, 6.0, 7.0]]],
    ids=["wind_10,6,7"],
    name="meteorology",
)
def fixture_meteorology(request):
    """Create a wind component for the test."""
    wind_vector = np.array(request.param)
    meteorology = Meteorology()
    time = pd.array(
        pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:00"), periods=35, freq="s"), dtype="datetime64[ns]"
    )[:, None]
    meteorology.u_component = wind_vector[:, 0] * np.ones(time.size)
    meteorology.v_component = wind_vector[:, 1] * np.ones(time.size)
    meteorology.w_component = wind_vector[:, 2] * np.ones(time.size)
    meteorology.time = time
    return meteorology


@pytest.fixture(params=[0, 3], ids=["GrdSrc", "3Src"], name="source_map")
def fixture_source_map(request):
    """Fixture for source map."""
    source_map = SourceMap()
    source_map.location = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)

    if request.param > 0:
        source_map.location.east = np.zeros((request.param, 1))
        source_map.location.north = np.zeros((request.param, 1))
        source_map.location.up = np.zeros((request.param, 1))

    return source_map


@pytest.fixture(params=[[3, 4, 2]], ids=["342"], name="number_cells")
def fixture_number_cells(request):
    """Fixture for number of cells in each dimension."""
    return request.param


@pytest.fixture(
    params=[["neumann"], ["dirichlet"]],
    ids=["neu", "dir"],
    name="external_boundary_type",
)
def fixture_external_boundary_type(request):
    """Fixture for external boundary type."""
    return request.param


@pytest.fixture(params=[["x", "y"], ["x", "y", "z"]], ids=["2D", "3D"], name="dimension")
def fixture_dimension(request, number_cells, external_boundary_type):
    """Fixture for dimension of the finite volume.

    We create a list of FiniteVolumeDimension objects based on the dimension_labels, number_cells, and
    external_boundary_type parameters.

    """
    dimension_labels = request.param
    limits = [-10, 10]

    dimension = []
    for i, dim in enumerate(dimension_labels):
        dimension.append(
            FiniteVolumeDimension(
                label=dim,
                number_cells=number_cells[i],
                limits=limits,
                external_boundary_type=external_boundary_type,
            )
        )

    return dimension


@pytest.fixture(params=[False, True], ids=["explicit", "implicit"], name="implicit_solver")
def fixture_implicit_solver(request):
    """Fixture for implicit solver."""
    return request.param


@pytest.fixture(params=[False, True], ids=["no_obstacle", "obstacle"], name="use_obstacle")
def fixture_use_obstacle(request):
    """Fixture for obstacle."""
    return request.param


@pytest.fixture(params=[False, True], ids=["no_lookup", "lookup"], name="use_lookup_table")
def fixture_use_lookup_table(request):
    """Fixture for using a lookup table."""
    return request.param


@pytest.fixture(name="finite_volume")
def fixture_finite_volume(implicit_solver, use_obstacle, dimension, source_map, use_lookup_table):
    """Create a finite volume object with the given dimension.

    Diffusion constants are set to 1.0 for all dimensions. The time step is set to 0.1. An obstacle is created in the
    middle of the grid if use_obstacle is True.

    Arguments:
        implicit_solver (bool): Whether to use an implicit solver.
        use_obstacle (bool): Whether to include an obstacle in the grid.
        dimension (list): List of FiniteVolumeDimension objects defining the grid.
        source_map (SourceMap): Source map object defining the sources.
        use_lookup_table (bool): Whether to use a lookup table for the coupling matrix.

    """
    diffusion_constants = [1.0] * len(dimension)

    if use_obstacle:
        cylinders_coordinate = ENU(
            east=np.array(0.0, ndmin=2),
            north=np.array(0.0, ndmin=2),
            up=np.array(10.0, ndmin=2),
            ref_latitude=0,
            ref_longitude=0,
            ref_altitude=0,
        )
        site_layout = SiteLayout(cylinders_coordinate=cylinders_coordinate, cylinders_radius=np.array([[1.0]]))

    else:
        site_layout = None

    return FiniteVolume(
        dimensions=dimension,
        diffusion_constants=diffusion_constants,
        site_layout=site_layout,
        source_map=source_map,
        use_lookup_table=use_lookup_table,
        implicit_solver=implicit_solver,
        minimum_contribution=1e-6,
    )


@pytest.fixture(name="sensor_object")
def fixture_sensor_object():
    """Fixture to define a generic sensor object."""
    sensor_object = Sensor()
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    location.from_array(np.array([[5, 0, 0]]))
    sensor_object.location = location
    time = pd.array(
        pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:05"), periods=5, freq="5s"), dtype="datetime64[ns]"
    )[:, None]
    sensor_object.time = time
    sensor_object.concentration = np.zeros(time.size)
    sensor_object.label = "Generic"
    return sensor_object


@pytest.fixture(name="beam_object")
def fixture_beam_object():
    """Fixture to define a beam sensor object."""
    beam_location = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    beam_location.from_array(np.array([[-5, 0, 0], [5, 0, 0]]))
    beam_object = Beam()
    beam_object.location = beam_location
    time = pd.array(
        pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:05"), periods=4, freq="6s"), dtype="datetime64[ns]"
    )[:, None]
    beam_object.time = time
    beam_object.concentration = np.zeros(time.size)
    beam_object.label = "Beam"
    return beam_object


@pytest.fixture(params=["point+beam"], name="sensor_group")
# @pytest.fixture(params=["1 point", "1 beam", "point+beam"], name="sensor_group")
def fixture_sensor_group(request, sensor_object, beam_object):
    """Fixture to define a sensor group."""
    sensor_group = SensorGroup()
    if request.param == "1 point":
        sensor_group.add_sensor(sensor_object)
    elif request.param == "1 beam":
        sensor_group.add_sensor(beam_object)
    elif request.param == "point+beam":
        sensor_group.add_sensor(sensor_object)
        sensor_group.add_sensor(beam_object)
    return sensor_group


def test_dimension(dimension):
    """Check that the dimension object is created correctly."""
    for dim in dimension:
        assert isinstance(dim, FiniteVolumeDimension)
        assert len(dim.label) == 1
        assert isinstance(dim.label, str)
        assert dim.number_cells > 0
        assert len(dim.limits) == 2
        assert len(dim.external_boundary_type) == 2
        assert dim.cell_edges[0] == dim.limits[0]
        assert dim.cell_edges[-1] == dim.limits[1]
        assert dim.cell_width == (dim.limits[1] - dim.limits[0]) / dim.number_cells
        assert len(dim.faces) == 2
        assert isinstance(dim.faces[0], FiniteVolumeFaceLeft)
        assert isinstance(dim.faces[1], FiniteVolumeFaceRight)
        for i, face in enumerate(dim.faces):
            assert face.external_boundary_type == dim.external_boundary_type[i]


def test_finite_volume(finite_volume):
    """Check that the finite volume object is created correctly."""
    assert isinstance(finite_volume, FiniteVolume)
    assert len(finite_volume.dimensions) == finite_volume.number_dimensions
    assert finite_volume.total_number_cells == np.prod([dim.number_cells for dim in finite_volume.dimensions])
    assert isinstance(finite_volume.grid_coordinates, ENU)

    if finite_volume.site_layout is not None:
        assert finite_volume.site_layout.id_obstacles.shape == (finite_volume.total_number_cells, 1)
        assert finite_volume.site_layout.id_obstacles.dtype == bool

    for dim in finite_volume.dimensions:
        assert isinstance(dim, FiniteVolumeDimension)
        for face in dim.faces:
            assert isinstance(face, FiniteVolumeFace)
            assert face.cell_face_area > 0
            assert face.cell_volume > 0
            assert face.boundary_type.shape == (finite_volume.total_number_cells, 1)
            assert face.neighbour_index.shape == (finite_volume.total_number_cells, 1)


def test_forward_matrix(finite_volume, meteorology):
    """Unit test to verify that mass balance is correctly preserved in the numerical scheme implemented by the finite
    volume solver.

    This test evaluates conservation properties for a discretized advection-diffusion
    transport model. The physical principle being tested is mass conservation, i.e.,
    the net change in mass within a control volume (grid cell) must equal the net fluxes
    across its boundaries plus any sources/sinks. The test ensures that the numerical
    implementation of the flux terms (diffusion and advection) respects this balance.
    Parameters:

    The test performs mass balance checks at multiple levels of the finite volume assembly:
    1. Face-Level Diffusion Check:
        For each face of each spatial dimension, it verifies that the sum of all diffusion
        flux contributions (neighbor term, central term, Dirichlet, and Neumann)
        equals zero. This ensures local consistency of the diffusion operator.
    2. Global Advection and Diffusion Check:
        For each term (advection and diffusion), the test checks that the combined contribution
        from neighbor interactions, central terms, and boundary conditions sum to zero for each cell.
        Additionally, it checks that all term arrays have the expected shapes.
    3. Combined Operator Check:
        It checks that the total contribution from the assembled transport operator,
        including all internal and boundary effects (from `adv_diff_terms['combined']`),
        balances with the implicit time-stepping sink term (`cell_volume / dt`).
    4. Matrix-Level Mass Balance:
        It validates that the assembled system matrix `A` and right-hand-side vector `b`,
        obtained from `solver_matrix`, also satisfy the mass balance against the
        implicit time sink (`cell_volume / dt`), ensuring the global solver preserves conservation.

    """

    fe = finite_volume
    fe.set_dt_cfl(meteorology)

    assert fe.dt > 0

    meteorology_windfield = MeteorologyWindfield(site_layout=fe.site_layout, static_wind_field=meteorology)
    meteorology_windfield.calculate_spatial_wind_field(time_index=0, grid_coordinates=fe.grid_coordinates)

    fe.compute_forward_matrix(meteorology_windfield)

    for dim in fe.dimensions:
        for term in ["diffusion"]:
            for face in dim.faces:
                check_value = (
                    face.adv_diff_terms[term].B_neighbour
                    + face.adv_diff_terms[term].B_central
                    + face.adv_diff_terms[term].b_dirichlet
                    + face.adv_diff_terms[term].b_neumann
                )

                assert np.allclose(check_value, 0, atol=1e-10)

    if fe.implicit_solver:
        volume_term = fe.cell_volume / fe.dt
    else:
        volume_term = -fe.cell_volume / fe.dt

    for term in ["advection", "diffusion"]:
        check_value = (
            np.sum(fe.adv_diff_terms[term].B_neighbour, axis=1).reshape(-1, 1)
            + fe.adv_diff_terms[term].B_central
            + fe.adv_diff_terms[term].b_dirichlet
            + fe.adv_diff_terms[term].b_neumann
        )
        assert np.allclose(check_value, 0, atol=1e-10)
        assert fe.adv_diff_terms[term].B_central.shape == (fe.total_number_cells, 1)
        assert fe.adv_diff_terms[term].b_dirichlet.shape == (fe.total_number_cells, 1)
        assert fe.adv_diff_terms[term].b_neumann.shape == (fe.total_number_cells, 1)
        assert fe.adv_diff_terms[term].B_neighbour.shape == (fe.total_number_cells, len(fe.dimensions) * 2)

        check_value = (
            np.sum(fe.adv_diff_terms["combined"].B, axis=1).reshape(-1, 1)
            + fe.adv_diff_terms["combined"].b_dirichlet
            + volume_term
        )
    assert np.allclose(check_value, 0, atol=1e-10)

    check_value = (
        np.sum(fe.forward_matrix, axis=1).reshape(-1, 1) + fe.adv_diff_terms["combined"].b_dirichlet + volume_term
    )

    assert np.allclose(check_value, 0, atol=1e-10)


def test_finite_volume_time_step_solver(finite_volume, meteorology):
    """Test the compute_coupling method of the FiniteVolume class.

    This test checks that the coupling matrix is correctly computed for different wind components.

    """
    finite_volume.set_dt_cfl(meteorology)

    meteorology_windfield = MeteorologyWindfield(
        site_layout=finite_volume.site_layout,
        static_wind_field=meteorology,
    )
    meteorology_windfield.calculate_spatial_wind_field(time_index=0, grid_coordinates=finite_volume.grid_coordinates)

    coupling_matrix = None
    for _ in range(2):
        coupling_matrix = finite_volume.propagate_solver_single_time_step(
            meteorology_windfield,
            coupling_matrix=coupling_matrix,
        )

        assert coupling_matrix.shape == (finite_volume.total_number_cells, finite_volume.source_grid_link.shape[1])
        assert sparse.issparse(coupling_matrix)
        assert np.min(coupling_matrix) >= 0


@pytest.mark.parametrize("output_stacked", [False, True], ids=["dict", "stacked"])
@pytest.mark.parametrize("sections", [False, True], ids=["single", "2 sections"])
def test_compute_coupling(finite_volume, meteorology, sensor_group, output_stacked, sections):
    """Test the compute_coupling method of the FiniteVolume class."""

    meteorology_windfield = MeteorologyWindfield(
        site_layout=None,
        static_wind_field=meteorology,
    )
    if sections is True:
        for sensor in sensor_group.values():
            sensor.source_on = np.round(np.linspace(1, 2, sensor.nof_observations)).reshape(-1, 1)

    output = finite_volume.compute_coupling(
        sensor_object=sensor_group, met_windfield=meteorology_windfield, gas_object=CH4(), output_stacked=output_stacked
    )

    if output_stacked:
        assert output.shape == (sensor_group.nof_observations, finite_volume.source_map.nof_sources)
        assert output.dtype == "float64"
        assert np.all(output >= 0)
        assert np.all(np.isfinite(output))
    else:
        assert isinstance(output, dict)
        assert len(output) == len(sensor_group)
        for key, sensor in sensor_group.items():
            assert output[key].shape == (sensor.nof_observations, finite_volume.source_map.nof_sources)
            assert output[key].dtype == "float64"
            assert np.all(output[key] >= 0)
            assert np.all(np.isfinite(output[key]))


def test_compute_time_bins(finite_volume, sensor_group, meteorology):
    """Test the compute_time_bins method of the FiniteVolume class.

    Time bins are defined from range(sensor) so all time_index_sensor should be well defined

    Meteorology time bins may not be well defined if the time range is not the same as the sensor time range.

    """

    # Compute the time bins
    (time_bins, time_index_sensor, time_index_met) = finite_volume.compute_time_bins(sensor_group, meteorology)

    n_bins = len(time_bins)
    # Check the result
    assert time_bins[1] - time_bins[0] == pd.Timedelta(finite_volume.dt, unit="s")

    for key, sensor in sensor_group.items():
        assert time_index_sensor[key].shape == (sensor.nof_observations,)
        assert time_index_sensor[key].shape == (sensor.nof_observations,)
        assert time_index_sensor[key].dtype == "int64"
        assert np.all(time_index_sensor[key] >= 0)
        assert np.all(time_index_sensor[key] <= n_bins)

    assert time_index_met.shape == (n_bins,)
    assert time_index_met.dtype == "int64"
    assert np.all(time_index_met >= 0)
    assert np.all(time_index_met <= meteorology.u_component.shape[0])
