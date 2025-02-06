# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Post-processing module.

Module containing some functions used in post-processing of the results.

"""
import warnings
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import label
from shapely import geometry

from pyelq.coordinate_system import ENU

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


def calculate_rectangular_statistics(
    model_object: "ELQModel",
    bin_size_x: float = 1,
    bin_size_y: float = 1,
    burn_in: int = 0,
    normalized_count_limit: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, pd.DataFrame]:
    """Function which aggregates the pyELQ results into rectangular bins and outputs the related summary statistics.

    The function creates a pixel grid (binning) in East-North coordinates based on the bin_size_x and bin_size_y
    parameters. For each bin both a count as well as a weighted sum of the emission rate estimates is calculated. The
    count is normalized by the number of iterations used in the MCMC and a boolean array is created which indicates if
    the count is above a certain threshold. Connected pixels where the count is above this threshold are considered to
    be a single blob/source and emission estimates per blob are summed over all pixels in the blob. The function
    then calculates the summary statistics for each blob of estimates which are connected pixels. The summary
    statistics include the median and IQR of the emission rate estimates, the mean location of the blob and the
    likelihood of the blob.

    Args:
        model_object (ELQModel): ELQModel object containing the results of the MCMC run.
        bin_size_x (float, optional): Size of the bins in the x-direction. Defaults to 1.
        bin_size_y  (float, optional): Size of the bins in the y-direction. Defaults to 1.
        burn_in (int, optional): Number of burn-in iterations used in the MCMC. Defaults to 0.
        normalized_count_limit  (float, optional): Threshold for the normalized count to be considered a blob.

    Returns:
        result_weighted (np.ndarray): Weighted sum of the emission rate estimates in each bin.
        overall_count (np.ndarray): Count of the number of estimates in each bin.
        normalized_count (np.ndarray): Normalized count of the number of estimates in each bin.
        count_boolean (np.ndarray): Boolean array which indicates if likelihood of pixel is over threshold.
        edges_result (list): Centers of the pixels in the x and y direction.
        summary_result (pd.DataFrame): Summary statistics for each blob of estimates.

    """
    nof_iterations = model_object.n_iter
    ref_latitude = model_object.components["source"].dispersion_model.source_map.location.ref_latitude
    ref_longitude = model_object.components["source"].dispersion_model.source_map.location.ref_longitude
    ref_altitude = model_object.components["source"].dispersion_model.source_map.location.ref_altitude

    if model_object.components["source"].reversible_jump:
        all_source_locations = model_object.mcmc.store["z_src"]
    else:
        source_locations = (
            model_object.components["source"]
            .dispersion_model.source_map.location.to_enu(
                ref_longitude=ref_longitude, ref_latitude=ref_latitude, ref_altitude=ref_altitude
            )
            .to_array()
        )
        all_source_locations = np.repeat(source_locations.T[:, :, np.newaxis], model_object.mcmc.n_iter, axis=2)

    if np.all(np.isnan(all_source_locations[:2, :, :])):
        warnings.warn("No sources found")
        result_weighted = np.array([[[np.nan]]])
        overall_count = np.array([[0]])
        normalized_count = np.array([[0]])
        count_boolean = np.array([[False]])
        edges_result = [np.array([np.nan])] * 2
        summary_result = pd.DataFrame()
        summary_result.index.name = "source_ID"
        summary_result.loc[0, "latitude"] = np.nan
        summary_result.loc[0, "longitude"] = np.nan
        summary_result.loc[0, "altitude"] = np.nan
        summary_result.loc[0, "height"] = np.nan
        summary_result.loc[0, "median_estimate"] = np.nan
        summary_result.loc[0, "quantile_025"] = np.nan
        summary_result.loc[0, "quantile_975"] = np.nan
        summary_result.loc[0, "iqr_estimate"] = np.nan
        summary_result.loc[0, "absolute_count_iterations"] = np.nan
        summary_result.loc[0, "blob_likelihood"] = np.nan

        return result_weighted, overall_count, normalized_count, count_boolean, edges_result[:2], summary_result

    min_x = np.nanmin(all_source_locations[0, :, :])
    max_x = np.nanmax(all_source_locations[0, :, :])
    min_y = np.nanmin(all_source_locations[1, :, :])
    max_y = np.nanmax(all_source_locations[1, :, :])

    bin_min_x = np.floor(min_x - 0.1)
    bin_max_x = np.ceil(max_x + 0.1)
    bin_min_y = np.floor(min_y - 0.1)
    bin_max_y = np.ceil(max_y + 0.1)
    bin_min_iteration = burn_in + 0.5
    bin_max_iteration = nof_iterations + 0.5

    max_nof_sources = all_source_locations.shape[1]

    x_edges = np.arange(start=bin_min_x, stop=bin_max_x + bin_size_x, step=bin_size_x)
    y_edges = np.arange(start=bin_min_y, stop=bin_max_y + bin_size_y, step=bin_size_y)
    iteration_edges = np.arange(start=bin_min_iteration, stop=bin_max_iteration + bin_size_y, step=1)

    result_x_vals = all_source_locations[0, :, :].flatten()
    result_y_vals = all_source_locations[1, :, :].flatten()
    result_z_vals = all_source_locations[2, :, :].flatten()

    result_iteration_vals = np.array(range(nof_iterations)).reshape(1, -1) + 1
    result_iteration_vals = np.tile(result_iteration_vals, (max_nof_sources, 1)).flatten()
    results_estimates = model_object.mcmc.store["s"].flatten()

    result_weighted, _ = np.histogramdd(
        sample=np.array([result_x_vals, result_y_vals, result_iteration_vals]).T,
        bins=[x_edges, y_edges, iteration_edges],
        weights=results_estimates,
        density=False,
    )

    count_result, edges_result = np.histogramdd(
        sample=np.array([result_x_vals, result_y_vals, result_iteration_vals]).T,
        bins=[x_edges, y_edges, iteration_edges],
        density=False,
    )

    overall_count = np.array(np.sum(count_result, axis=2))
    normalized_count = overall_count / (nof_iterations - burn_in)
    count_boolean = normalized_count >= normalized_count_limit

    summary_result = create_aggregation(
        result_iteration_vals=result_iteration_vals,
        burn_in=burn_in,
        result_x_vals=result_x_vals,
        result_y_vals=result_y_vals,
        result_z_vals=result_z_vals,
        results_estimates=results_estimates,
        count_boolean=count_boolean,
        x_edges=x_edges,
        y_edges=y_edges,
        nof_iterations=nof_iterations,
        ref_latitude=ref_latitude,
        ref_longitude=ref_longitude,
        ref_altitude=ref_altitude,
    )

    return result_weighted, overall_count, normalized_count, count_boolean, edges_result[:2], summary_result


def create_lla_polygons_from_xy_points(
    points_array: list[np.ndarray],
    ref_latitude: float,
    ref_longitude: float,
    ref_altitude: float,
    boolean_mask: Union[np.ndarray, None] = None,
) -> list[geometry.Polygon]:
    """Function to create polygons in LLA coordinates from a grid of points in ENU coordinates.

    This function takes a grid of East-North points, these points are used as center points for a pixel grid. The pixel
    grid is then converted to LLA coordinates and these center points are used to create a polygon in LLA coordinates.
    A polygon is only created if the boolean mask for that pixel is True. In case one unique East-North point is
    available, a predefined grid size of 1e-6 (equaling to 0.0036 seconds) is assumed. 

    Args:
        points_array (list[np.ndarray]): List of arrays of grid of points in ENU coordinates.
        ref_latitude (float): Reference latitude in degrees of ENU coordinate system.
        ref_longitude (float): Reference longitude in degrees of ENU coordinate system.
        ref_altitude (float): Reference altitude in meters of ENU coordinate system.
        boolean_mask (np.ndarray, optional): Boolean mask to indicate which pixels to create polygons for.
            Defaults to None which means all pixels are used.

    Returns:
        list[geometry.Polygon]: List of polygons in LLA coordinates
    """
    if boolean_mask is None:
        boolean_mask = np.ones_like(points_array, dtype=bool)

    enu_x = points_array[0]
    enu_x = enu_x[:-1] + np.diff(enu_x) / 2
    enu_y = points_array[1]
    enu_y = enu_y[:-1] + np.diff(enu_y) / 2

    enu_x, enu_y = np.meshgrid(enu_x, enu_y, indexing="ij")

    enu_object_full_grid = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)
    enu_object_full_grid.east = enu_x.flatten()
    enu_object_full_grid.north = enu_y.flatten()
    enu_object_full_grid.up = np.zeros_like(enu_object_full_grid.north)
    lla_object_full_grid = enu_object_full_grid.to_lla()

    _, gridsize_lat = is_regularly_spaced(lla_object_full_grid.latitude, tolerance=1e-6)
    _, gridsize_lon = is_regularly_spaced(lla_object_full_grid.longitude, tolerance=1e-6)

    if np.isnan(gridsize_lat):
        gridsize_lat = 1e-6
    if np.isnan(gridsize_lon):
        gridsize_lon = 1e-6

    polygons = [
        geometry.box(
            lla_object_full_grid.longitude[idx] - gridsize_lon / 2,
            lla_object_full_grid.latitude[idx] - gridsize_lat / 2,
            lla_object_full_grid.longitude[idx] + gridsize_lon / 2,
            lla_object_full_grid.latitude[idx] + gridsize_lat / 2,
        )
        for idx in np.argwhere(boolean_mask.flatten()).flatten()
    ]

    return polygons


def create_aggregation(
    result_x_vals: np.ndarray,
    result_y_vals: np.ndarray,
    result_z_vals: np.ndarray,
    results_estimates: np.ndarray,
    result_iteration_vals: np.ndarray,
    count_boolean: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    nof_iterations: int,
    burn_in: int,
    ref_latitude: float,
    ref_longitude: float,
    ref_altitude: float,
) -> pd.DataFrame:
    """Function to create the aggregated information for the blobs of estimates.

    We identify all blobs of estimates which appear close together on the map by looking at connected pixels in the
    count_boolean array. Next we find the summary statistics for all estimates in that blob like overall median and
    IQR estimate, mean location and the likelihood of that blob.

    When multiple sources are present in the same blob at the same iteration we first sum those emission rate
    estimates before taking the median.

    If no blobs are found a dataframe with nan values is return to avoid breaking plotting code which calls this
    function.

    Args:
        result_x_vals (np.ndarray): X-coordinate of estimates, flattened array of (n_sources_max * nof_iterations,).
        result_y_vals (np.ndarray): Y-coordinate of estimates, flattened array of (n_sources_max * nof_iterations,).
        result_z_vals (np.ndarray): Z-coordinate of estimates, flattened array of (n_sources_max * nof_iterations,).
        results_estimates (np.ndarray): Emission rate estimates, flattened array of
            (n_sources_max * nof_iterations,).
        result_iteration_vals (np.ndarray): Iteration number corresponding each estimated value, flattened array
            of (n_sources_max * nof_iterations,).
        count_boolean (np.ndarray): Boolean array which indicates if likelihood of pixel is over threshold.
        x_edges (np.ndarray): Pixel edges x-coordinates.
        y_edges (np.ndarray): Pixel edges y-coordinates.
        nof_iterations (int): Number of iterations used in MCMC.
        burn_in (int): Burn-in used in MCMC.
        ref_latitude (float): Reference latitude in degrees of ENU coordinate system.
        ref_longitude (float): Reference longitude in degrees of ENU coordinate system.
        ref_altitude (float): Reference altitude in meters of ENU coordinate system.

    Returns:
        summary_result (pd.DataFrame): Summary statistics for each blob of estimates.

    """
    labeled_array, num_features = label(input=count_boolean, structure=np.ones((3, 3)))

    if num_features == 0:
        summary_result = pd.DataFrame()
        summary_result.index.name = "source_ID"
        summary_result.loc[0, "latitude"] = np.nan
        summary_result.loc[0, "longitude"] = np.nan
        summary_result.loc[0, "altitude"] = np.nan
        summary_result.loc[0, "height"] = np.nan
        summary_result.loc[0, "median_estimate"] = np.nan
        summary_result.loc[0, "quantile_025"] = np.nan
        summary_result.loc[0, "quantile_975"] = np.nan
        summary_result.loc[0, "iqr_estimate"] = np.nan
        summary_result.loc[0, "absolute_count_iterations"] = np.nan
        summary_result.loc[0, "blob_likelihood"] = np.nan

        return summary_result

    burn_in_bool = result_iteration_vals > burn_in
    nan_x_vals = np.isnan(result_x_vals)
    nan_y_vals = np.isnan(result_y_vals)
    nan_z_vals = np.isnan(result_z_vals)
    no_nan_idx = np.logical_not(np.logical_or(np.logical_or(nan_x_vals, nan_y_vals), nan_z_vals))
    no_nan_and_burn_in_bool = np.logical_and(no_nan_idx, burn_in_bool)
    result_x_vals_no_nan = result_x_vals[no_nan_and_burn_in_bool]
    result_y_vals_no_nan = result_y_vals[no_nan_and_burn_in_bool]
    result_z_vals_no_nan = result_z_vals[no_nan_and_burn_in_bool]
    results_estimates_no_nan = results_estimates[no_nan_and_burn_in_bool]
    result_iteration_vals_no_nan = result_iteration_vals[no_nan_and_burn_in_bool]

    x_idx = np.digitize(result_x_vals_no_nan, x_edges, right=False) - 1
    y_idx = np.digitize(result_y_vals_no_nan, y_edges, right=False) - 1
    bin_numbers = np.ravel_multi_index((x_idx, y_idx), labeled_array.shape)

    bin_numbers_per_label = [
        np.ravel_multi_index(np.nonzero(labeled_array == value), labeled_array.shape)
        for value in np.array(range(num_features)) + 1
    ]

    summary_result = pd.DataFrame()
    summary_result.index.name = "source_ID"

    for label_idx, curr_bins in enumerate(bin_numbers_per_label):
        boolean_for_result = np.isin(bin_numbers, curr_bins)
        mean_x = np.mean(result_x_vals_no_nan[boolean_for_result])
        mean_y = np.mean(result_y_vals_no_nan[boolean_for_result])
        mean_z = np.mean(result_z_vals_no_nan[boolean_for_result])

        unique_iteration_vals, indices, counts = np.unique(
            result_iteration_vals_no_nan[boolean_for_result], return_inverse=True, return_counts=True
        )
        nof_iterations_present = unique_iteration_vals.size
        blob_likelihood = nof_iterations_present / (nof_iterations - burn_in)
        single_idx = np.argwhere(counts == 1)
        results_estimates_for_blob = results_estimates_no_nan[boolean_for_result]
        temp_estimate_result = results_estimates_for_blob[indices[single_idx.flatten()]]
        multiple_idx = np.argwhere(counts > 1)
        for single_idx in multiple_idx:
            temp_val = np.sum(results_estimates_for_blob[indices == single_idx])
            temp_estimate_result = np.append(temp_estimate_result, temp_val)

        median_estimate = np.median(temp_estimate_result)
        iqr_estimate = np.nanquantile(a=temp_estimate_result, q=0.75) - np.nanquantile(a=temp_estimate_result, q=0.25)
        lower_bound = np.nanquantile(a=temp_estimate_result, q=0.025)
        upper_bound = np.nanquantile(a=temp_estimate_result, q=0.975)
        enu_object = ENU(ref_latitude=ref_latitude, ref_longitude=ref_longitude, ref_altitude=ref_altitude)
        enu_object.east = mean_x
        enu_object.north = mean_y
        enu_object.up = mean_z
        lla_object = enu_object.to_lla()

        summary_result.loc[label_idx, "latitude"] = lla_object.latitude
        summary_result.loc[label_idx, "longitude"] = lla_object.longitude
        summary_result.loc[label_idx, "altitude"] = lla_object.altitude
        summary_result.loc[label_idx, "height"] = mean_z
        summary_result.loc[label_idx, "median_estimate"] = median_estimate
        summary_result.loc[label_idx, "quantile_025"] = lower_bound
        summary_result.loc[label_idx, "quantile_975"] = upper_bound
        summary_result.loc[label_idx, "iqr_estimate"] = iqr_estimate
        summary_result.loc[label_idx, "absolute_count_iterations"] = nof_iterations_present
        summary_result.loc[label_idx, "blob_likelihood"] = blob_likelihood

    summary_result = summary_result.astype({"absolute_count_iterations": "int"})

    return summary_result
