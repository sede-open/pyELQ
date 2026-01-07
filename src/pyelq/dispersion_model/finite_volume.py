# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Finite Volume Dispersion Model module.

Methods and classes for the finite volume method for the dispersion model.

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_array, dia_array
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
from tqdm import tqdm

from pyelq.coordinate_system import ENU
from pyelq.dispersion_model.dispersion_model import DispersionModel
from pyelq.dispersion_model.site_layout import SiteLayout
from pyelq.gas_species import GasSpecies
from pyelq.meteorology.meteorology import Meteorology
from pyelq.meteorology.meteorology_windfield import MeteorologyWindfield
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import SensorGroup


@dataclass
class FiniteVolume(DispersionModel):
    """Dispersion model object which creates a coupling matrix using a finite volume solver.

    Uses an advection-diffusion solver to create the coupling matrix between a set of source locations and a set of
    sensor locations.

    Args:
        dimensions (list): list of FiniteVolumeDimension for each grid dimension (e.g., x, y, z).
        diffusion_constants (np.ndarray): array of diffusion constants ([x,y,z], m^2/s).
        site_layout (Union[SiteLayout, None]): the layout of the site including cylinder coordinates and radii.
            (default is None). If None, no obstacles are considered in the model.
        dt (float): time step (s) (default is None). (If None, the time step is set using the CFL condition).
        implicit_solver (bool): if True, the solver uses implicit methods. (default is False).
        courant_number (float): Courant number which and represents the fraction of the grid cell that a fluid particle
            can travel in one time step. It is used in calculating dt when not specified. Default is 0.5 which means
            that a fluid particle can travel half the grid cell in one time step.
        burn_in_steady_state (bool): if True, the model runs a burn-in period to reach steady state before
            computing coupling. (default is True).
        use_lookup_table (bool): if True, uses a lookup table for coupling matrix interpolation (default is True).

    Attributes:
        grid_coordinates (np.ndarray): shape=(total_number_cells, number_dimensions), coordinates of the grid points.
        source_grid_link (csr_array): is a sparse matrix linking the source map to the grid coordinates.
        cell_volume (float): volume of a single grid cell.
        total_number_cells (int): total number of cells in the grid.
        grid_size (tuple): size of the grid in each dimension.
        grid_centers (list): centers of the grid cells in each dimension.
        number_dimensions (int): number of dimensions in the grid.
        adv_diff_terms (dict): contains advection and diffusion terms for the solver matrix.
        coupling_lookup_table (np.ndarray): coupling matrix calculated for each grid cell in grid_coordinates computed
            when use_lookup_table=True. It is used for interpolation of coupling values for new source locations without
            the need to re-run the FV solver.
        forward_matrix (dia_array): the solver matrix for the finite volume method.
        _forward_matrix_transpose (dia_array): the transpose of the solver matrix for the finite volume method.

    """

    dimensions: list = field(default_factory=list)
    diffusion_constants: np.ndarray = field(default_factory=lambda: np.zeros((3, 1)))
    site_layout: Union[SiteLayout, None] = field(default=None)
    dt: Union[float, None] = field(default=None)
    implicit_solver: bool = field(default=False)
    courant_number: float = field(default=0.5)
    burn_in_steady_state: bool = field(default=True)
    use_lookup_table: bool = field(default=True)

    grid_coordinates: np.ndarray = field(init=False)
    source_grid_link: csr_array = field(init=False)
    cell_volume: float = field(init=False)
    total_number_cells: int = field(init=False)
    grid_size: tuple = field(init=False)
    grid_centers: list = field(init=False)
    number_dimensions: int = field(init=False)
    adv_diff_terms: dict = field(init=False)
    coupling_lookup_table: np.ndarray = field(init=False, default=None)
    forward_matrix: dia_array = field(init=False, default=None)
    _forward_matrix_transpose: dia_array = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Post-initialization checks and setup.

        Creates the grid and neighbourhood for the finite volume solver, and uses the site layout to mask any obstacles
        from the solver grid.

        """
        if not isinstance(self.source_map.location, ENU):
            raise ValueError("source_map.location must be an ENU object.")
        self.number_dimensions = len(self.dimensions)
        self._setup_grid()
        if self.site_layout is not None:
            self.site_layout.find_index_obstacles(self.grid_coordinates)
        self._setup_neighbourhood()

    def compute_coupling(
        self,
        sensor_object: SensorGroup,
        met_windfield: MeteorologyWindfield,
        gas_object: Union[GasSpecies, None] = None,
        output_stacked: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, dict]:
        """Compute the coupling matrix for the finite volume method using a lookup table.

        If self.use_lookup_table == False, or if self.coupling_lookup_table is None, the coupling matrix is computed
        using the FV solver and stored in self.coupling_lookup_table. Otherwise, the coupling matrix is computed using a
        lookup table approach from the previously computed coupling matrix.

        Args:
            sensor_object (SensorGroup): sensor object containing sensor observations.
            met_windfield (MeteorologyWindfield): meteorology object containing site layout and timeseries of wind data.
            gas_object (Union[GasSpecies, None]): optional input, a gas species object to correctly calculate the
                gas density which is used in the conversion of the units of the Gaussian plume coupling. Defaults to
                None.
            output_stacked (bool): if True, the coupling is stacked across sensors into a single np.ndarray. Otherwise,
                the coupling is returned as a dictionary with an entry per sensor. Defaults to False.
            **kwargs: additional keyword arguments. To accommodate some arguments used in
                GaussianPlume.compute_coupling but not required in FiniteVolume.

        Returns:
            output (Union[np.ndarray, dict]): List of arrays, single array or dictionary containing the plume coupling
                in hr/kg. If a dictionary of sensor objects is passed in and output_stacked=False, this function returns
                a dictionary consistent with the input dictionary keys, containing the corresponding plume coupling
                outputs for each sensor. If a dictionary of sensor objects is passed in and output_stacked=True, this
                function returns an np.ndarray containing the stacked coupling matrices.

        """
        if (met_windfield.site_layout is not None) | (self.site_layout is not None):
            if np.any(met_windfield.site_layout.id_obstacles != self.site_layout.id_obstacles):
                raise ValueError("MeteorologyWindfield site layout does not match FiniteVolume site layout.")
        if not isinstance(self.source_map.location, ENU):
            raise ValueError("source_map.location must be an ENU object.")

        if (not self.use_lookup_table) or (self.coupling_lookup_table is None):
            coupling_sensor = self.compute_coupling_sections(sensor_object, met_windfield, gas_object)
            if self.use_lookup_table:
                self.coupling_lookup_table = coupling_sensor

        if self.use_lookup_table:
            output = self.interpolate_coupling_lookup_to_source_map(sensor_object)
        else:
            output = coupling_sensor
        if output_stacked:
            output = np.concatenate(tuple(output.values()), axis=0)
        return output

    def compute_coupling_sections(
        self, sensor_object: SensorGroup, met_windfield: MeteorologyWindfield, gas_object: GasSpecies
    ) -> dict:
        """Compute the coupling sections for the finite volume method.

        Sections are defined by the source_on attribute of the sensor object. If source_on is None (not specified) or
        all ones, then it is treated as a single section of data and directly moves on to computing the coupling matrix.

        If there are multiple sections, then the coupling matrix is computed for each section separately and combined
        into a single coupling matrix. This avoids computational effort computing the forward model through time steps
        that are not required and can speed up the computational time substantially in this case. Sections are
        defined by the source_on attribute of the sensor object which indicates which time steps the source is on where
        0 indicates the source is off and integers starting from 1 indicate different source on sections.

        Args:
            sensor_object (SensorGroup): sensor data object.
            meteorology_object (MeteorologyWindfield): wind field data object.
            gas_object (GasSpecies): gas species object.

        Returns:
            coupling_sensor (dict): coupling for each sensor, keys corresponding to each sensor: e.g.
                coupling_sensor['sensor_1'] is the coupling matrix for sensor 1.

        """
        if sensor_object.source_on is None or np.all(sensor_object.source_on == 1):
            return self.finite_volume_time_step_solver(sensor_object, met_windfield, gas_object)

        number_of_sections = max(sensor_object.source_on)
        coupling_sensor = {}
        for key, sensor in sensor_object.items():
            coupling_sensor[key] = np.full((sensor.time.shape[0], self.source_grid_link.shape[1]), fill_value=0.0)
        for section in range(1, number_of_sections + 1):
            subset_sensor_object = sensor_object.subset_sensor(section_index=section)
            coupling_sensor_section = self.finite_volume_time_step_solver(
                subset_sensor_object, met_windfield, gas_object
            )
            for key, sensor in sensor_object.items():
                section_index = (sensor.source_on == section).flatten()
                coupling_sensor[key][section_index, :] = coupling_sensor_section[key]
        return coupling_sensor

    def finite_volume_time_step_solver(
        self,
        sensor_object: SensorGroup,
        met_windfield: MeteorologyWindfield,
        gas_object: GasSpecies,
    ) -> dict:
        """Compute the finite volume coupling matrix, by time-stepping the solver.

        This function calculates the coupling between emission sources and sensor measurements based on a spatial wind
        field derived from meteorological data. The resulting coupling matrices model the transport of gas through a
        discretized domain. The coupling between emissions in all solver grid cells and concentrations in the same set
        of grid cells is calculated by time-stepping a finite volume solver for the advection-diffusion equation. In
        time bins where sensor observations occur, the coupling between any source locations in the source map and the
        locations where sensor observations were obtained are extracted and stored in the rows of the coupling matrix.

        If dt is not specified, it will be set automatically using a CFL-like condition via self.set_delta_time_cfl().
        If burn_in_steady_state is True, the model runs a burn-in period to reach steady state before computing any
        coupling values. The wind field during the burn-in period is assumed to be constant and the same as the wind
        field at the first time-step.

        If the coupling matrix is unstable (norm > 1e3), an error is raised suggesting to check the CFL number and dt.
        This condition is only checked at time t = 0.

        Args:
            sensor_object (SensorGroup): sensor data object.
            meteorology_object (MeteorologyWindfield): wind field data object.
            gas_object (GasSpecies): gas species object.

        Returns:
            coupling_sensor (dict): coupling matrix for each sensor and sources defined by source_grid_link units hr/kg.
                coupling_sensor keys corresponding to each source, e.g. coupling_sensor['sensor_1'] =
                coupling matrix for sensor 1 with shape=(number of observations (sensor_1), number of sources).

        """
        coupling_sensor = {}
        for key, sensor in sensor_object.items():
            coupling_sensor[key] = np.full((sensor.time.shape[0], self.source_grid_link.shape[1]), fill_value=0.0)
        coupling_grid = None
        time_bins, time_index_sensor, time_index_met = self.compute_time_bins(
            sensor_object=sensor_object, meteorology_object=met_windfield.static_wind_field
        )
        sensor_object = self._prepare_sensor(sensor_object)
        n_burn_steps = self._calculate_number_burn_steps(met_windfield.static_wind_field)
        gas_density = self.calculate_gas_density(
            met_windfield.static_wind_field, sensor_object, gas_object, run_interpolation=False
        )
        met_windfield.calculate_spatial_wind_field(time_index=0, grid_coordinates=self.grid_coordinates)

        for i_time in tqdm(range(-n_burn_steps, time_bins.size), desc="Computing coupling matrix"):
            if i_time > 0 and (time_index_met[i_time] != time_index_met[i_time - 1]):
                met_windfield.calculate_spatial_wind_field(
                    time_index=time_index_met[i_time], grid_coordinates=self.grid_coordinates
                )
            if gas_density.size > 1:
                gas_density_i = gas_density[time_index_met[i_time]]
            else:
                gas_density_i = gas_density
            coupling_grid = self.propagate_solver_single_time_step(met_windfield, coupling_matrix=coupling_grid)
            scaled_coupling = coupling_grid * (1e6 / (gas_density_i.item() * 3600))
            coupling_sensor = self.interpolate_coupling_grid_to_sensor(
                sensor_object,
                scaled_coupling=scaled_coupling,
                time_index_sensor=time_index_sensor,
                i_time=i_time,
                coupling_sensor=coupling_sensor,
            )
            if i_time == np.floor(0.1 * (time_bins.size + n_burn_steps)):
                coupling_grid_sourcemap_norm = sp.linalg.norm(coupling_grid)
                if coupling_grid_sourcemap_norm > 1e3:
                    raise ValueError(
                        f"The coupling matrix is unstable, with matrix norm: {coupling_grid_sourcemap_norm:.3g}, "
                        f"check the courant_number={self.courant_number:.3f} and calculated dt="
                        f"{self.dt:.3f} s"
                    )

        return coupling_sensor

    def interpolate_coupling_lookup_to_source_map(self, sensor_object: SensorGroup) -> dict:
        """Compute the coupling matrix by interpolation from a lookup table.

        A coupling matrix from all solver grid centres to all observations is pre-computed and stored on the class.
        Coupling columns for new source locations can then be computed by interpolation from these pre-computed values.

        The coupling matrix used for lookup is taken from self.coupling_lookup_table which is a sparse matrix computed
        in self.finite_volume_time_step_solver().

        Args:
            sensor_object (SensorGroup): sensor data object.

        Returns:
            interpolated_coupling (dict): interpolated coupling matrix for each sensor and sources (units hr/kg).

        """
        interpolated_coupling = {}
        source_location = self.source_map.location.to_array(dim=self.number_dimensions)
        for key, sensor in sensor_object.items():
            interpolated_coupling[key] = np.full((sensor.time.shape[0], source_location.shape[0]), fill_value=0.0)
            lookup_table_values = self.coupling_lookup_table[key].T
            interpolated_coupling[key] = self._build_interpolator(
                lookup_table_values, locations_to_interpolate=source_location
            ).T
        return interpolated_coupling

    def propagate_solver_single_time_step(
        self, met_windfield: MeteorologyWindfield, coupling_matrix: np.ndarray = None
    ) -> sp.csr_array:
        """Time-step the finite volume solver.

        Time-step the finite volume solver to map the coupling matrix at time t to the coupling matrix at time (t +
        dt).

        For each time step, the forward matrix is computed based on the current wind field. The coupling matrix is then
        evolved by a single time-step using either an implicit or explicit solver approach, depending on the value of
        self.implicit_solver.

        coupling_matrix will be a sparse csr_array with shape=(total_number_cells, number of sources)

        If minimum_contribution is set, all elements in the coupling matrix smaller than this number will be set to 0.
        This can speed up computation.

        Args:
            met_windfield (MeteorologyWindfield): meteorology object containing wind field information.
            coupling_matrix (np.ndarray): shape=(self.total_number_cells, number of sources). Coupling matrix at the
                current time. Units are s/m^3.

        Returns:
            coupling_matrix (sparse.csr_array): shape=(self.total_number_cells, number of sources). Coupling
                matrix on the finite volume grid. Represents the contribution of each cell to the source term in the
                transport equation.

        """
        self.compute_forward_matrix(met_windfield)
        if coupling_matrix is None:
            coupling_matrix = sp.csr_array(self.source_grid_link.shape, dtype=self.forward_matrix.dtype)
        scale_factor = self.dt / self.cell_volume
        if self.implicit_solver:
            rhs = (1.0 / scale_factor) * coupling_matrix + self.source_grid_link
            coupling_matrix = -spsolve(self.forward_matrix, rhs).reshape(self.source_grid_link.shape)
            if not sp.issparse(coupling_matrix):
                coupling_matrix = sp.csr_array(coupling_matrix)
        else:
            coupling_matrix = scale_factor * (self.forward_matrix @ coupling_matrix + self.source_grid_link)
        if self.minimum_contribution > 0 and sp.issparse(coupling_matrix):
            coupling_matrix.data[abs(coupling_matrix.data) <= self.minimum_contribution] = 0
            coupling_matrix.eliminate_zeros()
        if self.site_layout is not None:
            coupling_matrix[self.site_layout.id_obstacles_index, :] = 0
        return coupling_matrix

    def compute_forward_matrix(self, met_windfield: MeteorologyWindfield) -> None:
        """Construct the forward solver matrix. This can be used to step the solution forward in time.

        The matrix forward_matrix is constructed using the advection and diffusion terms computed for each face in the
        grid.

        The overall matrix equation for the FV solver is:
            (V / dt) * [c^(n+1) - c^(n)] + F @ c^(n) - G @ c^(n) = s
        where F is the matrix of advection term coefficients, G is the matrix of diffusion term coefficients, and s is
        the source term.

        Rearranging gives:
            c^(n+1) = R @ c^(n) + (dt / V) * s
        where R = I - (dt / V) * (F - G).

        The diagonals of the matrix are constructed using self._construct_diagonals_advection_diffusion() and combined
        using self._combine_advection_diffusion_terms().

        On first run, the matrix is constructed using self._construct_diagonal_matrix(). On subsequent runs, the matrix
        is updated using self._update_diagonal_matrix() which saves computational time by updating the sparse matrix in
        place.

        Args:
            met_windfield (MeteorologyWindfield): meteorology object containing wind field information.

        """
        self._compute_advection_diffusion_terms_by_face(met_windfield)
        self._construct_diagonals_advection_diffusion()
        self._combine_advection_diffusion_terms()
        if self.forward_matrix is None:
            self._construct_diagonal_matrix()
        else:
            self._update_diagonal_matrix()

    def _compute_advection_diffusion_terms_by_face(self, met_windfield: MeteorologyWindfield) -> None:
        """Compute advection and diffusion terms for each face in the grid.

        Loops over each dimension and face in the grid and computes the advection using the wind vector and the
        diffusion terms using the diffusion constants.

        Args:
            met_windfield (MeteorologyWindfield): meteorology object containing wind field information.

        """
        for i_dim, dim in enumerate(self.dimensions):
            if i_dim == 0:
                wind_component = met_windfield.u_component
            elif i_dim == 1:
                wind_component = met_windfield.v_component
            elif i_dim == 2:
                wind_component = met_windfield.w_component
            else:
                wind_component = None
            for face in dim.faces:
                face.assign_advection(wind_component)
                face.assign_diffusion(self.diffusion_constants[i_dim])

    def _construct_diagonals_advection_diffusion(self) -> None:
        """Construct the diagonals of the advection and diffusion contributions to the overall solver matrix.

        In the docstring of the function self._combine_advection_diffusion_terms(), the coefficients of the advection
        terms are stored in the matrix F, and the coefficients of the diffusion terms are stored in the matrix G. This
        function creates the diagonals of the F and G matrices.

        The overall diagonals are cumulated by looping over the solver dimensions, and the cell faces in each dimension.

        """
        num_off_diags = self.number_dimensions * 2
        self.adv_diff_terms = {"advection": SolverDiagonals(), "diffusion": SolverDiagonals()}
        for key, term in self.adv_diff_terms.items():
            term = self.adv_diff_terms[key]
            term.B_central = np.zeros((self.total_number_cells, 1))
            term.B_neighbour = np.zeros((self.total_number_cells, num_off_diags))
            term.b_dirichlet = np.zeros((self.total_number_cells, 1))
            term.b_neumann = np.zeros((self.total_number_cells, 1))
            count = 0
            for dim in self.dimensions:
                for face in dim.faces:
                    face_term = face.adv_diff_terms[key]
                    term.B_central += face_term.B_central
                    term.B_neighbour[:, count] = face_term.B_neighbour.flatten()
                    term.b_dirichlet += face_term.b_dirichlet
                    term.b_neumann += face_term.b_neumann
                    count += 1
            term.B = np.concatenate((term.B_central, term.B_neighbour), axis=1)

    def _combine_advection_diffusion_terms(self) -> None:
        """Combine the advection and diffusion terms into the solver matrix.

        The overall matrix equation for the FV solver is:
            (V / dt) * [c^(n+1) - c^(n)] + F @ c^(n) - G @ c^(n) = s
        where F is the matrix of advection term coefficients, G is the matrix of diffusion term coefficients, and s is
        the source term.

        Rearranging gives:
            c^(n+1) = R @ c^(n) + (dt / V) * s
        where R = I - (dt / V) * (F - G).

        This function calculates the diagonals of the matrix R by combining the advection and diffusion terms. These
        diagonals are stored in self.adv_diff_terms['combined'].B.

        """
        num_diags = 1 + self.number_dimensions * 2
        terms = self.adv_diff_terms
        terms["combined"] = SolverDiagonals()
        terms["combined"].B = np.zeros((self.total_number_cells, num_diags))
        if self.implicit_solver:
            terms["combined"].B[:, 0] = terms["combined"].B[:, 0] - self.cell_volume / self.dt
        else:
            terms["combined"].B[:, 0] = terms["combined"].B[:, 0] + self.cell_volume / self.dt
        terms["combined"].B = terms["combined"].B + terms["advection"].B + terms["diffusion"].B
        terms["combined"].b_dirichlet = terms["advection"].b_dirichlet + terms["diffusion"].b_dirichlet
        terms["combined"].b_neumann = terms["advection"].b_neumann + terms["diffusion"].b_neumann
        terms["combined"].B[:, 0] = terms["combined"].B[:, 0] + terms["combined"].b_neumann.flatten()

    def _construct_diagonal_matrix(self) -> None:
        """Construct the diagonal matrix for the solver.

        This method creates a sparse diagonal matrix using the diagonals and the specified grid size.

        The diagonal index is constructed based on the number of dimensions and the grid size using ravel_multi_index.
        This index is used to place the diagonal elements in the correct location within the sparse matrix. It is
        designed to be consistent with the meshgrid with indexing="ij" used to construct grid_coordinates in
        self._setup_grid().

        The transposed matrix self._forward_matrix_transpose is constructed to deal with the way the zero-padding works
        in dia_array then transposed to self.forward_matrix which is required for forward simulation.

        The matrix self._forward_matrix_transpose is also stored to allow quick updating in self._update_diagonal_matrix

        """
        diagonal_index = np.array([0])
        start_coord = np.zeros(self.number_dimensions, dtype=int)
        for i in range(self.number_dimensions):
            diag_coord = start_coord.copy()
            diag_coord[i] += 1
            diag_coord = np.ravel_multi_index(diag_coord, self.grid_size, mode="clip")
            diagonal_index = np.concatenate((diagonal_index, np.array([diag_coord, -diag_coord])))

        self._forward_matrix_transpose = dia_array(
            (self.adv_diff_terms["combined"].B.T, diagonal_index),
            shape=(self.total_number_cells, self.total_number_cells),
        )
        self.forward_matrix = self._forward_matrix_transpose.T

    def _update_diagonal_matrix(self) -> None:
        """Update the diagonal matrix for the solver.

        This method updates the diagonal matrix using the specified forward matrix. Avoid reconstructing the matrix and
        just updates the data in place.

        Since the forward_matrix is transposed, we need to update the transposed version of the forward matrix then
        transpose it back.

        """
        self._forward_matrix_transpose.data = self.adv_diff_terms["combined"].B.T
        self.forward_matrix = self._forward_matrix_transpose.T

    def _setup_grid(self) -> None:
        """Initializes a structured Cartesian grid using the site limits and number of cells in each dimension.

        ENU CoordinateSystem reference location is taken from self.source_map.

        This method builds a multi-dimensional grid by discretizing the spatial domain into equally spaced cells
        along each axis (e.g., x, y, z).

        Grid construction uses np.meshgrid with indexing="ij" to be consistent with the way the diagonals are
        constructed in self._construct_diagonal_matrix() and the way the neighbourhood is constructed in
        self._setup_neighbourhood(). "ij" is the matrix indexing convention, which means that the first dimension
        corresponds to rows and the second dimension corresponds to columns.

        Volume and Area Calculations:
            - self.cell_volume stores the volume of a single grid cell (product of widths).
            - For each dimension, self.cell_face_area is computed as the ratio of cell volume to that dimension's width,
                representing the area of a face perpendicular to the given axis.

        """
        self.cell_volume = np.prod([dim.cell_width for dim in self.dimensions])
        self.grid_centers = [dim.cell_centers for dim in self.dimensions]
        for dim in self.dimensions:
            for face in dim.faces:
                face.cell_volume = self.cell_volume
                face.cell_face_area = self.cell_volume / dim.cell_width
        grid_coordinates = np.meshgrid(*[dim.cell_centers for dim in self.dimensions], indexing="ij")
        self.grid_size = grid_coordinates[0].shape
        self.grid_coordinates = ENU(
            ref_longitude=self.source_map.location.ref_longitude,
            ref_latitude=self.source_map.location.ref_latitude,
            ref_altitude=self.source_map.location.ref_altitude,
        )
        self.grid_coordinates.east = grid_coordinates[0].reshape(-1, 1)
        if self.number_dimensions > 1:
            self.grid_coordinates.north = grid_coordinates[1].reshape(-1, 1)
        if self.number_dimensions > 2:
            self.grid_coordinates.up = grid_coordinates[2].reshape(-1, 1)
        self.total_number_cells = self.grid_coordinates.east.shape[0]
        self._setup_source_link()

    def _setup_source_link(self) -> None:
        """Setup the source link between the source map and the grid coordinates.

        This method creates a sparse matrix that links the source map to the grid coordinates.

        Used in the coupling matrix to link the source map to the grid coordinates.

        If there are no sources in the source map or if use_lookup_table is True, the source map locations are set to
        the grid coordinates and the source_grid_link is set to an identity matrix.

        If there are sources in the source map and use_lookup_table is False, a KDTree is used to find the nearest grid
        point for each source location. The source_grid_link is then created as a sparse matrix with ones at the
        locations of the nearest grid points and zeros elsewhere.

        self.source_grid_link is a sparse matrix linking the source map to the grid coordinates.

        """
        if self.use_lookup_table or self.source_map.nof_sources == 0:
            if self.source_map.nof_sources == 0:
                self.source_map.location = self.grid_coordinates
            self.source_grid_link = sp.eye_array(self.total_number_cells, format="csr")
        else:
            n_sources = self.source_map.nof_sources
            tree = KDTree(self.grid_coordinates.to_array(dim=self.number_dimensions))
            source_index = tree.query(self.source_map.location.to_array(dim=self.number_dimensions), k=1)[1]
            self.source_grid_link = sp.csr_array(
                (np.ones(n_sources), (source_index, np.array(range(n_sources)))),
                shape=(self.total_number_cells, n_sources),
            )

    def _setup_neighbourhood(self) -> None:
        """Initializes the neighborhood relationships for each cell in the grid across all dimensions.

        For a given dim and face, to find the neighbor indices for each cell, the index is unwrapped and converted to
        multi-dimensional indices using np.unravel_index.
            e.g. if grid_size = (10,10) then for the cell index 27,
                index_center = np.unravel_index(27, (10,10))  = (2,7)
            we find the neighbor index by shifting the multi-dimensional indices by the face shift:
            for left face in x-dimension, shift = -1, so the new multi-dimensional indices are (1, 7))
            then we convert back to the unwrapped index using
                index_neighbour = np.ravel_multi_index((1,7), (10,10)) = 17
            for right face in y-dimension, shift = 1, so the new multi-dimensional indices are (2, 8))
            then we convert back to the unwrapped index using
                index_neighbour = np.ravel_multi_index((2,8), (10,10)) = 28
        Cells that lie at the domain boundary (i.e., where a shift would move them outside the grid extent) are detected
        and handled:
            - Their neighbor index is set to `-9999` to indicate an invalid or non-existent neighbor.
            - They are classified as external boundaries.

        Cells adjacent to user-defined obstacles (as indicated by `self.site_layout.id_obstacle`) are specially treated.
        If a neighboring cell lies within an obstacle region, it's considered an invalid neighbor for flow or
        interaction purposes.

        For external boundaries, the method assigns Dirichlet or Neumann boundary conditions depending on the
        specification in the grid metadata for that dimension.

        Each grid dimension is updated with the following information for both 'left' and 'right' directions:
            - neighbour_index: array of neighbor indices for each cell (-9999 for out-of-bounds).
            - boundary_condition: array indicating the type of boundary condition ('internal', 'dirichlet', 'neumann').
            - boundary_conditions: the boundary condition type for the current direction.

        """
        index_center = np.unravel_index(range(self.total_number_cells), self.grid_size)
        for i, dim in enumerate(self.dimensions):
            for face in dim.faces:
                index_center_shift = list(index_center)
                index_center_shift[i] = index_center_shift[i] + face.shift
                face.neighbour_index = np.ravel_multi_index(index_center_shift, self.grid_size, mode="clip")
                face.neighbour_index = face.neighbour_index.reshape((self.total_number_cells, 1))
                external_boundaries = np.logical_or(
                    index_center_shift[i] < 0, index_center_shift[i] >= dim.number_cells
                )
                face.neighbour_index[external_boundaries] = -9999
                face.set_boundary_type(external_boundaries, self.site_layout)

    def compute_time_bins(
        self, sensor_object: SensorGroup, meteorology_object: Meteorology
    ) -> Tuple[pd.DatetimeIndex, dict, np.ndarray]:
        """Compute discretized time bins for aligning sensor observations and meteorological data.

        This method constructs a uniform time grid (bins) based on the observation time range of the given sensors.
        The time resolution is determined by `self.dt`. If `self.dt` is not specified, it will be set automatically
        using a CFL-like condition via `self.set_delta_time_cfl()` based on the meteorology object.

        Once the time bins are established:
            - Each sensor's observation times are digitized to determine which time bin each observation belongs to.
            - A KDTree is used to find the closest meteorological time index corresponding to each time bin, mapping the
            wind field to the solver grid.

        Args:
            sensor_object (SensorGroup): Sensor data object
            meteorology_object (Meteorology): Meteorology data object.

        Returns:
            time_bins (pd.DatetimeIndex): The array of uniformly spaced time bins (based on `self.dt`).
            time_index_sensor (dict): A dictionary mapping each sensor ID to its array of time bin indices.
            time_index_met (np.ndarray): An array mapping each time bin to the closest meteorological time index.

        """
        if self.dt is None:
            self.set_delta_time_cfl(meteorology_object)
        sensor_time = sensor_object.time.reshape(
            -1,
        )
        time_bins = pd.date_range(
            start=sensor_time.min() - pd.Timedelta(self.dt, unit="s"),
            end=sensor_time.max() + pd.Timedelta(self.dt, unit="s"),
            freq=f"{self.dt}s",
            inclusive="both",
        )
        time_index_sensor = {}
        for key, sensor in sensor_object.items():
            time_index_sensor[key] = np.digitize(
                sensor.time.reshape(
                    -1,
                ).astype(np.int64),
                time_bins.astype(np.int64),
            )
        tree = KDTree(meteorology_object.time.reshape(-1, 1).astype(np.int64))
        _, time_index_met = tree.query(np.array(time_bins.astype(np.int64)).reshape(-1, 1), k=1)
        return time_bins, time_index_sensor, time_index_met

    def set_delta_time_cfl(self, meteorology_object: Meteorology) -> None:
        """Use CFL condition to set the time step.

        The CFL condition is a stability criterion for numerical methods used in solving partial differential equations.
        It ensures that the numerical solution remains stable and converges to the true solution.

        The CFL condition for advection is given by:
            dt <= min(dx / |u|)
        for all dimensions, where dx is the grid spacing and u is the velocity. This method calculates the maximum
        velocity in each dimension and sets the time step accordingly.

        The diffusion term is also considered in the CFL condition:
            dt <= (dx^2) / (2 * K)
        for all dimensions, where K is self.diffusion_constants.

        dt is set to the minimum of the advection and diffusion time steps multiplied by self.courant_number.

        Args:
            meteorology_object (Meteorology): meteorology object containing timeseries of wind data.

        """
        if meteorology_object.wind_speed is None:
            meteorology_object.calculate_wind_speed_from_uv()
        u_max = np.max(meteorology_object.wind_speed)
        dx = np.min([dim.cell_width for dim in self.dimensions])

        dt_adv = np.round(self.courant_number * dx / u_max, decimals=1)
        dt_diff = (self.courant_number * dx**2) / (2 * np.max(self.diffusion_constants))
        self.dt = np.minimum(dt_adv, dt_diff)

    def interpolate_coupling_grid_to_sensor(
        self,
        sensor_object: SensorGroup,
        scaled_coupling: sp.csr_array,
        time_index_sensor: np.ndarray,
        i_time: int,
        coupling_sensor: dict,
    ) -> dict:
        """Interpolate coupling grid values to sensor locations.

        Calculate the coupling for each sensor at a given time step. This function interpolates plume coupling values
        from the coupling matrix to each sensor's location for a specific time step, and updates the output dictionary
        with the results.

        Args:
            sensor_object (SensorGroup): object containing sensor data.
            scaled_coupling (sp.csr_array): The sparse matrix representing coupling values between sources and grid
                cells for the current time step.
            time_index_sensor (np.ndarray): An array mapping each sensor to its corresponding time step index.
            i_time (int): The index of the current time step.
            coupling_sensor (dict): The output dictionary to be updated with coupling values for each sensor.

        Returns:
            coupling_sensor (dict): The updated output dictionary with interpolated coupling values at each sensor
                location for the current time step.

        """
        for key, sensor in sensor_object.items():
            observation_index = time_index_sensor[key] == i_time
            if np.any(observation_index):
                sensor_location = sensor.location.to_array(dim=self.number_dimensions)
                coupling_interp = self._build_interpolator(
                    scaled_coupling.toarray(), locations_to_interpolate=sensor_location, method="nearest"
                )
                if isinstance(sensor, Beam):
                    coupling_sensor[key][observation_index, :] = np.mean(coupling_interp, axis=0)
                else:
                    coupling_sensor[key][observation_index, :] = coupling_interp.flatten()
        return coupling_sensor

    def _build_interpolator(
        self, tabular_values: np.ndarray, locations_to_interpolate: np.ndarray, method: str = "linear"
    ) -> np.ndarray:
        """Build an interpolator for given tabular values and interpolate at specified locations.

        Interpolates values at specified locations using interpolation with the method of choosing within the grid,
        and nearest-neighbor extrapolation for out-of-bounds points.

        Args:
            tabular_values (np.ndarray): Array of data values defined on the grid.
            locations_to_interpolate (np.ndarray): Points at which to evaluate the interpolator, shape (M, D), where D
                is the number of dimensions.
            method (str): Interpolation method to use. Options are 'linear', 'nearest', etc.

        Returns:
            combined_result (np.ndarray): Interpolated values at the specified locations.

        """
        shape = list(self.grid_size) + [-1]
        reshaped_values = tabular_values.reshape(*shape)
        method_interp = RegularGridInterpolator(
            self.grid_centers,
            reshaped_values,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        nearest_interp = RegularGridInterpolator(
            self.grid_centers,
            reshaped_values,
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )
        method_result = method_interp(locations_to_interpolate)
        nearest_result = nearest_interp(locations_to_interpolate)
        combined_result = np.where(np.isnan(method_result), nearest_result, method_result)
        return combined_result

    def _prepare_sensor(self, sensor_object: SensorGroup) -> SensorGroup:
        """Add beam knots to the sensor object for Beam sensors and convert all sensor locations to ENU coordinates.

        Args:
            sensor_object (SensorGroup): SensorGroup object containing sensor observations.

        Returns:
            sensor_object_beam_knots_added (SensorGroup): A new SensorGroup object with beam knots added for Beam
                sensors.

        """
        sensor_object_beam_knots_added = deepcopy(sensor_object)
        for _, sensor in sensor_object_beam_knots_added.items():
            sensor.location = sensor.location.to_enu(
                ref_latitude=self.grid_coordinates.ref_latitude,
                ref_longitude=self.grid_coordinates.ref_longitude,
                ref_altitude=self.grid_coordinates.ref_altitude,
            )
            if isinstance(sensor, Beam):
                sensor_array = sensor.make_beam_knots(
                    ref_latitude=self.grid_coordinates.ref_latitude,
                    ref_longitude=self.grid_coordinates.ref_longitude,
                    ref_altitude=self.grid_coordinates.ref_altitude,
                )
                sensor.location.from_array(sensor_array)
                if self.number_dimensions == 2:
                    sensor_array = np.delete(sensor_array, 2, axis=1)
        return sensor_object_beam_knots_added

    def _calculate_number_burn_steps(self, meteorology_object: Meteorology) -> int:
        """Compute the number of burn-in steps for plume stabilization.

        Computes the approximate amount of time required for a gas parcel to traverse the entire solver domain, based on
        the initial wind conditions. Then, based on the model time step (self.dt), computes the approximate number of
        time steps required for the plume to stabilize before the main analysis begins.

        If burn_in_steady_state is False, the function returns 0.

        burn steps are calculated as:
        n_burn_steps = ceil(2 * max_domain_size / (max_wind_speed * dt))
        roughly the time for a plume to travel across the domain twice. Note only consider the horizontal dimensions.

        Args:
            meteorology_object (Meteorology): Object providing wind field or other meteorological data over time.

        Returns:
            n_burn_steps (int): The number of burn steps to be used in the coupling calculations.

        """
        if self.burn_in_steady_state is False:
            return 0
        meteorology_object.calculate_wind_speed_from_uv()
        n_burn_steps = int(
            np.ceil(
                2
                * np.max([(dim.limits[1] - dim.limits[0]) for dim in self.dimensions[:1]])
                / (meteorology_object.wind_speed[0] * self.dt)
            )
        )
        return n_burn_steps


@dataclass
class FiniteVolumeDimension:
    """Individual grid dimension for the finite volume method.

    Assuming that each solver dimension is a regular grid, this class stores grid properties, such as cell edges,
    centre points and cell widths.

    Args:
        label (str): name of this dimension (e.g., 'x', 'y', 'z').
        number_cells (int): number of cells in this dimension.
        limits (list): limits of this dimension (e.g., [0, 100]).
        external_boundary_type (list): type of boundary condition for the faces in this dimension
            e.g., external_boundary_type=['dirichlet', 'neumann'].
            If only 1 type is specified, it is used for both faces of this dimension.

    Attributes:
        cell_edges (np.ndarray): shape=(self.number_cells + 1,) edge locations for the cells in this dimension.
        cell_centers (np.ndarray): shape=(self.number_cells,) central locations of the cells in this dimension.
        cell_width (float): width of the cells in this dimension.
        faces (list(FiniteVolumeFaceLeft, FiniteVolumeFaceRight)): list of objects corresponding to the left and right
            (-ve and +ve) faces of this dimension.

    """

    label: str
    number_cells: int
    limits: list
    external_boundary_type: list = field(default_factory=list)
    cell_edges: np.ndarray = field(init=False)
    cell_centers: np.ndarray = field(init=False)
    cell_width: float = field(init=False)
    faces: list = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization processing.

        Validates the external boundary types and initializes the face objects for the dimension. Also calls
        get_dimensions to calculate and store geometric properties of the dimension.

        Raises:
            ValueError: external_boundary_type must one of ['dirichlet', 'neumann'].
            ValueError: number_cells must be at least 2.

        """
        if not isinstance(self.external_boundary_type, list):
            raise ValueError("external_boundary_type must be a list.")
        if self.number_cells < 2:
            raise ValueError("number_cells must be at least 2")
        if len(self.external_boundary_type) == 1:
            self.external_boundary_type = [self.external_boundary_type[0], self.external_boundary_type[0]]
        self.faces = [
            FiniteVolumeFaceLeft(self.external_boundary_type[0]),
            FiniteVolumeFaceRight(self.external_boundary_type[1]),
        ]
        self.get_dimensions()

    def get_dimensions(self) -> None:
        """Setup the face properties for the finite volume method.

        This function calculates and stores the grid cell edges, cell centres and cell widths, and assigns the cell
        width values to the cell faces.

        """
        self.cell_edges = np.linspace(self.limits[0], self.limits[1], self.number_cells + 1)
        self.cell_centers = 0.5 * (self.cell_edges[:-1] + self.cell_edges[1:])
        self.cell_width = self.cell_edges[1] - self.cell_edges[0]
        for face in self.faces:
            face.cell_width = self.cell_width


@dataclass
class FiniteVolumeFace(ABC):
    """Face type for a grid cell in the finite volume method.

    Args:
        external_boundary_type (str): The type of boundary condition for the face. either 'dirichlet' or 'neumann'.

    Attributes:
        cell_face_area (float): The area of the face.
        cell_volume (float): The volume of the face.
        cell_width (float): The width of the cell in the direction normal to the face.
        boundary_type (np.ndarray): shape=(total_number_cells, 1). The type of boundary condition for the face. Each
            entry is a string, either 'internal', 'dirichlet' or 'neumann'.
        neighbour_index (np.ndarray): shape=(total_number_cells, 1). The index of the neighboring cell across the face.
        adv_diff_terms (dict): The advection and diffusion terms for the face. Dictionary has two entries: "advection"
            and "diffusion", each containing a SolverDiagonals object.

    """

    external_boundary_type: str
    cell_face_area: float = field(init=False)
    cell_volume: float = field(init=False)
    cell_width: float = field(init=False)
    boundary_type: np.ndarray = field(init=False)
    neighbour_index: np.ndarray = field(init=False)
    adv_diff_terms: dict = field(init=False)

    @property
    @abstractmethod
    def normal(self):
        """Abstract property to be defined in subclasses."""

    def __post_init__(self) -> None:
        if self.external_boundary_type not in ["dirichlet", "neumann"]:
            raise ValueError(f"Invalid external boundary type: {self.external_boundary_type}. ")
        self.adv_diff_terms = {"advection": SolverDiagonals(), "diffusion": SolverDiagonals()}

    def set_boundary_type(self, external_boundaries, site_layout: SiteLayout = None) -> None:
        """Set the boundary condition for the face based on the external boundary type.

        External boundaries are set to 'dirichlet' or 'neumann' based on the specified external_boundary_type. Internal
        boundaries are set to 'internal'.

        The function also handles the case where the face is affected by an obstacle. Obstacle boundaries are set to
        'neumann'.

        Args:
            external_boundaries (np.ndarray): shape=(total_number_cells, 1). Boolean array indicating which faces are
                external boundaries.
            site_layout (SiteLayout): SiteLayout object containing obstacle information. Defaults to None.

        """
        self.boundary_type = np.full(self.neighbour_index.shape, "internal", dtype="<U10")
        self.boundary_type[external_boundaries] = self.external_boundary_type
        if site_layout is not None:
            faces_affected_obstacle = np.isin(self.neighbour_index, np.nonzero(site_layout.id_obstacles)[0])
            self.boundary_type[np.logical_or(faces_affected_obstacle, site_layout.id_obstacles)] = "neumann"

    def assign_advection(self, wind_vector: np.ndarray) -> None:
        """Assigns the advection terms for the defined set of interfaces to adv_diff_terms['advection'].

        Uses an upwind scheme for the discretization of the advection term:
        https://en.wikipedia.org/wiki/Upwind_scheme#:~:text=In#20computational#20physics#2C#20the#20term,derivatives#20in#20a#20flow#20field.

        Upwind scheme for a single dimension has the following form:
            F_i = A * [u^{+} * (c_i - c_{i-1}) + u^{-} * (c_{i+1} - c_{i})]
        where u^{+} = -min(-u, 0) and u^{-} = max(-u, 0), A is the face area, and indices corresponding to other
        dimensions have been dropped.

        Args:
            wind_vector (np.ndarray): shape=(total_number_cells, 1). Wind speed vector in dimension of this face
                e.g. x, y, z.

        """
        term = self.adv_diff_terms["advection"]
        u_norm = wind_vector * self.normal
        term.B_central = -self.cell_face_area * -np.minimum(-u_norm, 0)
        neighbour_advection = self.cell_face_area * np.maximum(-u_norm, 0)
        term.B_neighbour = (self.boundary_type == "internal") * neighbour_advection
        term.b_dirichlet = (self.boundary_type == "dirichlet") * neighbour_advection
        term.b_neumann = (self.boundary_type == "neumann") * neighbour_advection

    def assign_diffusion(self, diffusion_constants: float) -> None:
        """Assigns the diffusion terms for the defined set of interfaces to adv_diff_terms['diffusion'].

        If diffusion is already set this function is skipped as the diffusion term is constant.

        The diffusion term for a single dimension has the following form:
            G_i = K * A * [(c_{i+1} - c_i) / delta - (c_i - c_{i-1}) / delta]
        where K is the diffusion constant, A is the face area, delta is the cell width, and indices corresponding to
        other dimensions have been dropped.

        Args:
            diffusion_constants (scalar) : diffusion coefficient in this dimension.

        """
        term = self.adv_diff_terms["diffusion"]
        if term.B_central is None:
            diffusion_coefficient = self.cell_face_area * diffusion_constants / self.cell_width
            term.B_central = -diffusion_coefficient * np.ones(self.boundary_type.shape)
            term.B_neighbour = (self.boundary_type == "internal") * diffusion_coefficient
            term.b_dirichlet = (self.boundary_type == "dirichlet") * diffusion_coefficient
            term.b_neumann = (self.boundary_type == "neumann") * diffusion_coefficient


@dataclass
class FiniteVolumeFaceLeft(FiniteVolumeFace):
    """Set up face properties specific to a left-facing cell (i.e. outward normal is the negative unit vector).

    Attributes:
        direction (str): direction of the face, either 'left' or 'right'.
        shift (int): shift in the grid index to find the neighbour cell. -1 for left face.
        normal (int): normal vector for the face. -1 for left face.

    """

    direction: str = "left"
    shift: int = -1
    normal: int = -1


@dataclass
class FiniteVolumeFaceRight(FiniteVolumeFace):
    """Set up face properties specific to a right-facing cell (i.e. outward normal is the positive unit vector).

    Attributes:
        direction (str): direction of the face, either 'left' or 'right'.
        shift (int): shift in the grid index to find the neighbour cell. +1 for right face.
        normal (int): normal vector for the face. +1 for right face.

    """

    direction: str = "right"
    shift: int = 1
    normal: int = 1


@dataclass
class SolverDiagonals:
    """Storage for the diagonals of the solver matrix for the finite volume method on a regular grid.

    This class holds the diagonal components to construct the solver matrix. It is used for advection, diffusion and
    combined terms.

    Attributes:
        B (Union[np.ndarray, None]): shape=(total_number_cells, 1 + number_faces). Array containing all solver
            diagonals, i.e. containing all diagonals from self.B_central and self.B_neighbour. The first column is the
            central diagonal and the remaining columns are the off-diagonal terms.
        B_central (Union[np.ndarray, None]): shape=(total_number_cells, 1). Array containing the central diagonal of the
            solver matrix.
        B_neighbour (Union[np.ndarray, None]): shape=(total_number_cells, number_faces). Array containing the
            off-diagonals of the solver matrix.
        b_dirichlet (Union[np.ndarray, None]): shape=(total_number_cells, 1). Vector containing contributions from
            Dirichlet boundary conditions at edge cells.
        b_neumann (Union[np.ndarray, None]): shape=(total_number_cells, 1). Vector containing contributions from Neumann
            boundary conditions.

    """

    B: Union[np.ndarray, None] = field(default=None, init=False)
    B_central: Union[np.ndarray, None] = field(default=None, init=False)
    B_neighbour: Union[np.ndarray, None] = field(default=None, init=False)
    b_dirichlet: Union[np.ndarray, None] = field(default=None, init=False)
    b_neumann: Union[np.ndarray, None] = field(default=None, init=False)
