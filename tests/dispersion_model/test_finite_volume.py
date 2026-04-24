# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
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
from pyelq.meteorology.meteorology import Meteorology
from pyelq.meteorology.meteorology_windfield import MeteorologyWindfield, SiteLayout
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap


@pytest.fixture(
    params=[[[10.0, 6.0, 7.0]]],
    ids=["wind_u=10_v=6_w=7"],
    name="meteorology",
)
def fixture_meteorology(request):
    """Create a wind time series for the tests."""
    wind_vector = np.array(request.param)
    meteorology = Meteorology()
    time = pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:00"), periods=35, freq="s").array[:, None]
    meteorology.u_component = wind_vector[:, 0] * np.ones(time.size)
    meteorology.v_component = wind_vector[:, 1] * np.ones(time.size)
    meteorology.w_component = wind_vector[:, 2] * np.ones(time.size)
    meteorology.time = time
    return meteorology


@pytest.fixture(params=[0, 3], ids=["GrdSrc", "3Src"], name="source_map")
def fixture_source_map(request):
    """Create a source map for the tests."""
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


@pytest.fixture(params=[False, True], ids=["explicit", "implicit"], name="solver_type")
def fixture_solver_type(request):
    """Fixture for the type of solver (implicit or explicit)."""
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
def fixture_finite_volume(solver_type, use_obstacle, dimension, source_map, use_lookup_table):
    """Create a finite volume object with the given dimension.

    Diffusion constants are set to 1.0 for all dimensions. The time step is set to 0.1. An obstacle is created in the
    middle of the grid if use_obstacle is True.

    Arguments:
        solver_type (bool): Whether to use an implicit solver.
        use_obstacle (bool): Whether to include an obstacle in the grid.
        dimension (list): List of FiniteVolumeDimension objects defining the grid.
        source_map (SourceMap): Source map object defining the sources.
        use_lookup_table (bool): Whether to use a lookup table for the coupling matrix.

    """
    diffusion_constants = [1.0] * len(dimension)
    if use_obstacle:
        cylinder_coordinates = ENU(
            east=np.array(0.0, ndmin=2),
            north=np.array(0.0, ndmin=2),
            up=np.array(10.0, ndmin=2),
            ref_latitude=0,
            ref_longitude=0,
            ref_altitude=0,
        )
        site_layout = SiteLayout(cylinder_coordinates=cylinder_coordinates, cylinder_radius=np.array([[1.0]]))
    else:
        site_layout = None

    return FiniteVolume(
        dimensions=dimension,
        diffusion_constants=diffusion_constants,
        site_layout=site_layout,
        source_map=source_map,
        use_lookup_table=use_lookup_table,
        implicit_solver=solver_type,
        minimum_contribution=1e-6,
    )


@pytest.fixture(name="sensor_object")
def fixture_sensor_object():
    """Fixture to define a generic sensor object."""
    sensor_object = Sensor()
    location = ENU(ref_longitude=0, ref_latitude=0, ref_altitude=0)
    location.from_array(np.array([[5, 0, 0]]))
    sensor_object.location = location
    time = pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:05"), periods=5, freq="5s").array[:, None]
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
    time = pd.date_range(pd.Timestamp.fromisoformat("2022-01-01 00:00:05"), periods=4, freq="6s").array[:, None]
    beam_object.time = time
    beam_object.concentration = np.zeros(time.size)
    beam_object.label = "Beam"
    return beam_object


@pytest.fixture(params=["1 point", "1 beam", "point+beam"], name="sensor_group")
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

    This test evaluates conservation properties for a discretized advection-diffusion transport model. The physical
    principle being tested is mass conservation, i.e. the net change in mass within a control volume (grid cell) must
    equal the net fluxes across its boundaries plus any sources/sinks. The test ensures that the numerical
    implementation of the flux terms (diffusion and advection) respects this balance.

    The test performs mass balance checks at multiple levels of the finite volume assembly:
        1. Face-Level Diffusion Check:
            For each face of each spatial dimension, it verifies that the sum of all diffusion flux contributions
            (neighbor terms, central term, Dirichlet, and Neumann) equals zero. This ensures local consistency of the
            diffusion operator.
        2. Global Advection and Diffusion Check:
            For each term (advection and diffusion), the test checks that the combined contribution from neighbor
            interactions, central terms, and boundary conditions sum to zero for each cell. Additionally, it checks that
            all term arrays have the expected shapes.
        3. Combined Operator Check:
            It checks that the total contribution from the assembled transport operator, including all internal and
            boundary effects (from `adv_diff_terms['combined']`), balances with the implicit time derivative term
            (`cell_volume / dt`).
        4. Matrix-Level Mass Balance:
            It validates that the assembled system matrix `A` and right-hand-side vector `b`, obtained from
            `solver_matrix`, also satisfy the mass balance against the implicit time derivative term
            (`cell_volume / dt`), ensuring the global solver preserves conservation.

    """
    fe = finite_volume
    fe.set_delta_time_cfl(meteorology)
    assert fe.dt > 0
    meteorology_windfield = MeteorologyWindfield(site_layout=fe.site_layout, static_wind_field=meteorology)
    meteorology_windfield.calculate_spatial_wind_field(time_index=0, grid_coordinates=fe.grid_coordinates)
    fe.compute_forward_matrix(meteorology_windfield)
    for dim in fe.dimensions:
        for term in ["diffusion"]:
            for face in dim.faces:
                face_diffusion_flux_balance = (
                    face.adv_diff_terms[term].B_neighbour
                    + face.adv_diff_terms[term].B_central
                    + face.adv_diff_terms[term].b_dirichlet
                    + face.adv_diff_terms[term].b_neumann
                )
                assert np.allclose(face_diffusion_flux_balance, 0, atol=1e-10)

    if fe.implicit_solver:
        volume_term = fe.cell_volume / fe.dt
    else:
        volume_term = -fe.cell_volume / fe.dt
    for term in ["advection", "diffusion"]:
        cell_term_flux_balance = (
            np.sum(fe.adv_diff_terms[term].B_neighbour, axis=1).reshape(-1, 1)
            + fe.adv_diff_terms[term].B_central
            + fe.adv_diff_terms[term].b_dirichlet
            + fe.adv_diff_terms[term].b_neumann
        )
        assert np.allclose(cell_term_flux_balance, 0, atol=1e-10)
        assert fe.adv_diff_terms[term].B_central.shape == (fe.total_number_cells, 1)
        assert fe.adv_diff_terms[term].b_dirichlet.shape == (fe.total_number_cells, 1)
        assert fe.adv_diff_terms[term].b_neumann.shape == (fe.total_number_cells, 1)
        assert fe.adv_diff_terms[term].B_neighbour.shape == (fe.total_number_cells, len(fe.dimensions) * 2)
        combined_operator_balance = (
            np.sum(fe.adv_diff_terms["combined"].B, axis=1).reshape(-1, 1)
            + fe.adv_diff_terms["combined"].b_dirichlet
            + volume_term
        )
        assert np.allclose(combined_operator_balance, 0, atol=1e-10)

    forward_matrix_balance = (
        np.sum(fe.forward_matrix, axis=1).reshape(-1, 1) + fe.adv_diff_terms["combined"].b_dirichlet + volume_term
    )
    assert np.allclose(forward_matrix_balance, 0, atol=1e-10)


def test_finite_volume_time_step_solver(finite_volume, meteorology):
    """Test the compute_coupling method of the FiniteVolume class.

    This test runs two time steps of the finite volume solver, and checks that the resulting solver matrix after each
    step is sparse, has the correct shape, and contains only non-negative values.

    """
    finite_volume.set_delta_time_cfl(meteorology)
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
    """Test the compute_coupling method of the FiniteVolume class.

    For each input configuration, this test computes the coupling matrix, and then checks that the result:
        - Has the correct shape.
        - Is of type float64.
        - Contains only non-negative values.
        - Contains only finite values.

    These checks are carried out for both the case where sections are used (i.e., multiple sections means that a source
    can be active only during a subset of the sensor observations), and the case where no sections are used (i.e., a
    source is active for the entire duration of observations). In case sections are used, the source_on attribute of
    each sensor is set to activate sources in distinct sections of the observation period.
    Additionally, the checks are carried out for both the case where a single stacked matrix is returned, and the case
    where a dict of couplings per sensor is returned. For the case with single stacked matrix (output_stacked=True), a
    single 2D matrix is expected with the shape `(sensor_group.nof_observations, finite_volume.source_map.nof_sources)`.
    For the case with dict output (output_stacked=False), a dictionary is expected where each key corresponds to
    a sensor, and each value is a 2D matrix of shape `(sensor.nof_observations, finite_volume.source_map.nof_sources)`.

    """
    meteorology_windfield = MeteorologyWindfield(
        site_layout=finite_volume.site_layout,
        static_wind_field=meteorology,
    )
    if sections is True:
        for sensor in sensor_group.values():
            index = np.linspace(0, sensor.nof_observations - 1, sensor.nof_observations).astype(int)
            index = np.floor(index / sensor.nof_observations * 2 * 2)
            index = index * np.mod(index, 2)
            index = np.ceil(index * 0.5).astype(int)
            sensor.source_on = index

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
            if sensor.source_on is not None:
                assert np.all(output[key][sensor.source_on == 0, :] == 0)


def test_compute_time_bins(finite_volume, sensor_group, meteorology):
    """Test the compute_time_bins method of the FiniteVolume class.

    Time bins are defined from range(sensor) so all time_index_sensor should be well defined.
    Meteorology time bins may not be well defined if the time range is not the same as the sensor time range.

    """
    time_bins, time_index_sensor, time_index_met = finite_volume.compute_time_bins(sensor_group, meteorology)
    n_bins = len(time_bins)
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


def manually_construct_1d_advection_matrix(wind_vector):
    """Construct advection matrix F using an upwind scheme for a 1D grid.

     Upwind scheme for a single dimension has the following form:
            F_i = A * [(u_{i-1/2})^{+} * (c_{i} - c_{i-1}) + (u_{i+1/2})^{-} * (c_{i+1} - c_{i})]
    where:
        A = cell face area (= 1 in this case, because of grid setup)
        (u_{i-1/2})^{+} = max(u_{i-1/2}, 0)
        (u_{i+1/2})^{-} = min(u_{i+1/2}, 0)
    The winds u_{i-1/2} and u_{i+1/2} are computed as the average of the winds in the adjacent cells,
    i.e. u_{i-1/2} = (u_i + u_{i-1})/2 and u_{i+1/2} = (u_i + u_{i+1})/2. For the boundary cells, the
    face wind is taken as the wind in the cell itself, i.e. u_{-1/2} = u_0 and u_{n+1/2} = u_n.

    The (.)^{+} and (.)^{-} expressions and sign conventions are chosen to coincide with those used in the
    Wikipedia page on upwind schemes: https://en.wikipedia.org/wiki/Upwind_scheme

    Args:
        wind_vector (np.ndarray): wind speed at each grid cell

    Returns:
        F (np.ndarray): advection matrix with shape=(n_grid, n_grid).

    """
    wind_vector = wind_vector.flatten()
    n_grid = wind_vector.shape[0]
    F = np.zeros((n_grid, n_grid))

    for i in range(n_grid):
        if i == 0:
            u_face_l = wind_vector[i]
        else:
            u_face_l = (wind_vector[i] + wind_vector[i - 1]) / 2
        u_plus = max(u_face_l, 0)
        if i == n_grid - 1:
            u_face_r = wind_vector[i]
        else:
            u_face_r = (wind_vector[i] + wind_vector[i + 1]) / 2
        u_minus = min(u_face_r, 0)

        F[i, i] = u_plus - u_minus
        if i > 0:
            F[i, i - 1] = -u_plus
        if i < n_grid - 1:
            F[i, i + 1] = +u_minus

    return F


def manually_construct_2d_advection_matrix(wind_vector):
    """Construct advection matrix F using an upwind scheme for 2D grid.

    Upwind scheme for a single dimension has the following form:
            F_{ij} = A * [(u_{i-1/2,j})^{+} * (c_{i,j} - c_{i-1,j}) +
                                (u_{i+1/2,j})^{-} * (c_{i+1,j} - c_{i,j}) +
                          (v_{i,j-1/2})^{+} * (c_{i,j} - c_{i,j-1}) +
                                (v_{i,j+1/2})^{-} * (c_{i,j+1} - c_{i,j})]
    where:
        A = cell face area (= 1 in this case, because of grid setup)
        (u_{i-1/2,j})^{+} = max(u_{i-1/2,j}, 0)
        (u_{i+1/2,j})^{-} = min(u_{i+1/2,j}, 0)
        (v_{i,j-1/2})^{+} = max(v_{i,j-1/2}, 0)
        (v_{i,j+1/2})^{-} = min(v_{i,j+1/2}, 0)
    The winds u_{i-1/2,j}, u_{i+1/2,j}, v_{i,j-1/2}, and v_{i,j+1/2} are computed as the average of the winds
    in the adjacent cells. E.g. u_{i-1/2,j} = (u_{i,j} + u_{i-1,j})/2. For boundary cells, the face wind is
    set to be the same as at the cell centre.

    The (.)^{+} and (.)^{-} expressions and sign conventions are chosen to coincide with those used in the
    Wikipedia page on upwind schemes: https://en.wikipedia.org/wiki/Upwind_scheme

    Args:
        wind_vector (np.ndarray): array with shape=(n_grid^2, 2) of wind vectors in each grid cell. The i^th
            row is a wind vector (u_{i}, v_{i}) corresponding to the i^th grid cell, using the usual
            np.meshgrid unwrapping convention.

    Returns:
        F (np.ndarray): advection matrix with shape=(n_grid^2, n_grid^2).

    """
    u_vector = wind_vector[:, 0].flatten()
    v_vector = wind_vector[:, 1].flatten()
    n_total = u_vector.shape[0]
    n_grid = int(np.sqrt(n_total))
    F = np.zeros((n_total, n_total))

    for i in range(n_grid):
        for j in range(n_grid):
            i_central = i * n_grid + j
            i_left = (i - 1) * n_grid + j
            i_right = (i + 1) * n_grid + j
            i_down = i * n_grid + (j - 1)
            i_up = i * n_grid + (j + 1)

            if i == 0:
                u_face_l = u_vector[i_central]
            else:
                u_face_l = (u_vector[i_central] + u_vector[i_left]) / 2
            u_plus = max(u_face_l, 0)
            if i == n_grid - 1:
                u_face_r = u_vector[i_central]
            else:
                u_face_r = (u_vector[i_central] + u_vector[i_right]) / 2
            u_minus = min(u_face_r, 0)

            if j == 0:
                v_face_d = v_vector[i_central]
            else:
                v_face_d = (v_vector[i_central] + v_vector[i_down]) / 2
            v_plus = max(v_face_d, 0)
            if j == n_grid - 1:
                v_face_u = v_vector[i_central]
            else:
                v_face_u = (v_vector[i_central] + v_vector[i_up]) / 2
            v_minus = min(v_face_u, 0)

            F[i_central, i_central] += u_plus - u_minus + v_plus - v_minus
            if i > 0:
                F[i_central, i_left] -= u_plus
            if i < (n_grid - 1):
                F[i_central, i_right] += u_minus
            if j > 0:
                F[i_central, i_down] -= v_plus
            if j < (n_grid - 1):
                F[i_central, i_up] += v_minus

    return F


@pytest.mark.parametrize("n_grid", [3, 5, 10], ids=["3 cells", "5 cells", "10 cells"])
@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann"], ids=["Dirichlet", "Neumann"])
def test_two_dimensional_advection_matrix(n_grid, boundary_type):
    """This test checks that the advection matrix for a 2D finite volume discretization is correctly constructed.

    The test constructs an advection matrix for a 2D grid element-by-element (in a loop). This "manually constructed"
    solver matrix is then compared to the advection matrix computed using the main implementation (constructed using
    sparse diagonal methods). These should give exactly the same result.

    For simplicity in this case, the solver grid is set up so that the cell volume is 1 and time step dt = 1, so the
    multiplicative factor is also V/dt = 1.

    The wind vector is randomly generated for each test case.

    Args:
        n_grid (int): number of cells in each dimension of the 2D grid
        boundary_type (str): type of boundary condition to apply at the edges of the grid
            (either "dirichlet" or "neumann"). The same boundary type is applied to all grid edges.

    """
    source_map = SourceMap()
    source_map.location = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    wind_vector = np.random.normal(0, 1, size=(n_grid**2, 2))
    dim_x = FiniteVolumeDimension("x", number_cells=n_grid, limits=[0, n_grid], external_boundary_type=[boundary_type])
    dim_y = FiniteVolumeDimension("y", number_cells=n_grid, limits=[0, n_grid], external_boundary_type=[boundary_type])
    fe = FiniteVolume(dimensions=[dim_x, dim_y], source_map=source_map, dt=1, diffusion_constants=[0.0, 0.0])
    met = MeteorologyWindfield(static_wind_field=None)
    met.u_component = wind_vector[:, [0]]
    met.v_component = wind_vector[:, [1]]
    fe.compute_forward_matrix(met)

    F = np.eye(n_grid**2) - fe.forward_matrix.toarray() + np.diag(fe.adv_diff_terms["combined"].b_neumann.flatten())
    F_manual = manually_construct_2d_advection_matrix(wind_vector)
    assert np.allclose(F, F_manual, atol=1e-10)


@pytest.mark.parametrize("n_grid", [3, 5, 10], ids=["3 cells", "5 cells", "10 cells"])
@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann"], ids=["Dirichlet", "Neumann"])
def test_one_dimensional_advection_matrix(n_grid, boundary_type):
    """This test checks that the advection matrix for a 1D finite volume discretization is correctly constructed.

    The test compares the advection matrix computed by the finite volume implementation matches a
    manually constructed advection matrix for a 1D grid with n_grid cells that doesn't use the sparse diagonal
    construction.

    For simplicity in this case the finite volume is setup so that the volume is 1 and the dt = 1 so the
    multiplicative factor is simply V/dt = 1.

    The wind vector is randomly generated for each test case.

    Args:
        n_grid (int): number of cells in the 1D grid
        boundary_type (str): type of boundary condition to apply at the edges of the grid
            (either "dirichlet" or "neumann").

    """
    source_map = SourceMap()
    source_map.location = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    u = np.random.normal(0, 1, size=(n_grid, 1))
    dim = FiniteVolumeDimension("x", number_cells=n_grid, limits=[0, n_grid], external_boundary_type=[boundary_type])
    fe = FiniteVolume(dimensions=[dim], source_map=source_map, dt=1, diffusion_constants=[0.0])
    met = MeteorologyWindfield(static_wind_field=None)
    met.u_component = u
    fe.compute_forward_matrix(met)

    F = np.eye(n_grid) - fe.forward_matrix.toarray() + np.diag(fe.adv_diff_terms["combined"].b_neumann.flatten())
    F_manual = manually_construct_1d_advection_matrix(u)
    assert np.allclose(F, F_manual, atol=1e-10)


def manually_construct_1d_diffusion_matrix(diffusion_constant, n_grid):
    """Make a 1D diffusion matrix by assigning each term individually.

    Finite difference scheme for a single dimension has the following form:
        G_i = A * K * [(c_{i+1} - c_{i})/dx - (c_{i} - c_{i-1})/dx]
    where:
        A = cell face area (= 1 in this case, because of grid setup)
        K = diffusion constant
        dx = cell width (= 1 in this case, because of grid setup)

    Args:
        diffusion_constant (np.ndarray): diffusion constant.

    """
    G = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        G[i, i] = -diffusion_constant[0] * 2
        if i > 0:
            G[i, i - 1] = diffusion_constant[0]
        if i < n_grid - 1:
            G[i, i + 1] = diffusion_constant[0]
    return G


def manually_construct_2d_diffusion_matrix(diffusion_constant, n_grid):
    """Make a 2D diffusion matrix by assigning each term individually.

    Finite difference scheme for two dimensions has the following form:
        G_{ij} = A * K_x * [(c_{i+1,j} - c_{i,j})/dx - (c_{i,j} - c_{i-1,j})/dx] +
                 A * K_y * [(c_{i,j+1} - c_{i,j})/dy - (c_{i,j} - c_{i,j-1})/dy]
    where:
        A = cell face area (= 1 in this case, because of grid setup)
        K_x, K_y = diffusion constants in x and y directions
        dx, dy = cell widths in x and y directions (= 1 in this case, because of grid setup)

    Args:
        diffusion_constant (np.ndarray): diffusion constant.
        n_grid (int): number of cells in each dimension of the 2D grid.

    """
    G = np.zeros((n_grid**2, n_grid**2))
    for i in range(n_grid):
        for j in range(n_grid):
            i_central = i * n_grid + j
            i_left = (i - 1) * n_grid + j
            i_right = (i + 1) * n_grid + j
            i_down = i * n_grid + (j - 1)
            i_up = i * n_grid + (j + 1)
            G[i_central, i_central] = -diffusion_constant[0] * 2 - diffusion_constant[1] * 2
            if i > 0:
                G[i_central, i_left] = diffusion_constant[0]
            if i < (n_grid - 1):
                G[i_central, i_right] = diffusion_constant[0]
            if j > 0:
                G[i_central, i_down] = diffusion_constant[1]
            if j < (n_grid - 1):
                G[i_central, i_up] = diffusion_constant[1]
    return G


@pytest.mark.parametrize("n_grid", [3, 5, 10], ids=["3 cells", "5 cells", "10 cells"])
@pytest.mark.parametrize("boundary_type", ["dirichlet", "neumann"], ids=["Dirichlet", "Neumann"])
@pytest.mark.parametrize("dimension", [1, 2], ids=["1D", "2D"])
def test_diffusion_matrix(n_grid, boundary_type, dimension):
    """This test checks that the diffusion matrix for a finite volume discretization is correctly constructed.

    The test compares the diffusion matrix computed by the finite volume implementation matches a manually
    constructed diffusion matrix which doesn't use the sparse diagonal construction.

    For simplicity in this case the finite volume is setup so that the cell volume is 1 and the dt = 1 so the
    multiplicative factor is simply V/dt = 1.

    The diffusion constant is randomly generated for each test case.

    Args:
        n_grid (int): number of cells in each dimension of the grid
        boundary_type (str): type of boundary condition to apply at the edges of the grid
            (either "dirichlet" or "neumann").
        dimension (int): dimensionality of the grid (either 1 or 2).

    """
    source_map = SourceMap()
    source_map.location = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    diffusion_constant = np.random.uniform(0, 1, size=(dimension,))
    dim = [FiniteVolumeDimension("x", number_cells=n_grid, limits=[0, n_grid], external_boundary_type=[boundary_type])]
    if dimension == 2:
        dim.append(
            FiniteVolumeDimension("y", number_cells=n_grid, limits=[0, n_grid], external_boundary_type=[boundary_type])
        )
    fe = FiniteVolume(
        dimensions=dim,
        source_map=source_map,
        dt=1,
        diffusion_constants=[diffusion_constant[k] for k in range(dimension)],
    )
    met = MeteorologyWindfield(static_wind_field=None)
    if dimension == 1:
        met.u_component = np.zeros((n_grid, 1))
    if dimension == 2:
        met.u_component = np.zeros((n_grid**2, 1))
        met.v_component = np.zeros((n_grid**2, 1))
    fe.compute_forward_matrix(met)

    G = (
        fe.forward_matrix.toarray()
        - np.eye(fe.forward_matrix.shape[0])
        - np.diag(fe.adv_diff_terms["combined"].b_neumann.flatten())
    )
    if dimension == 1:
        G_manual = manually_construct_1d_diffusion_matrix(diffusion_constant, n_grid)
    else:
        G_manual = manually_construct_2d_diffusion_matrix(diffusion_constant, n_grid)
    assert np.allclose(G, G_manual, atol=1e-10)
