# SPDX-FileCopyrightText: 2026 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test module for coordinate system functions.

This module provides tests for the coordinate system

"""

import numpy as np
import pytest

from pyelq.coordinate_system import ECEF, ENU, LLA, make_latin_hypercube


@pytest.mark.parametrize("n", [1, 100])
def test_lla(n):
    """Testing conversion functions in LLA class to convert to all other types and check get back to the same place.

    Args:
        n (int): Parameter to define size of location vector

    """
    rng = np.random.default_rng(42)
    lat = rng.random(n) * 10 + 40
    lon = (rng.random(n) - 0.5) * 5
    alt = rng.random(n) * 100
    crd = LLA(latitude=lat, longitude=lon, altitude=alt)

    crd2 = crd.to_lla()
    crd3 = crd2.to_enu()
    crd4 = crd3.to_ecef()
    crd5 = crd4.to_lla()

    assert np.all(crd5.latitude == pytest.approx(lat))
    assert np.all(crd5.longitude == pytest.approx(lon))
    assert np.all(crd5.altitude == pytest.approx(alt))

    crd.create_tree()


@pytest.mark.parametrize("n", [1, 100])
def test_enu(n):
    """Testing conversion functions in ENU class to convert to all other types and check get back to the same place.

    Args:
        n (int): Parameter to define size of location vector

    """
    rng = np.random.default_rng(42)
    east = rng.random(n) * 10000
    north = (rng.random(n)) * 5000
    up = rng.random(n) * 100
    ref_lat = 50
    ref_lon = 0
    ref_alt = 0
    crd = ENU(ref_longitude=ref_lon, ref_latitude=ref_lat, ref_altitude=ref_alt, east=east, north=north, up=up)

    ref_lat_2 = 54
    ref_lon_2 = 2
    ref_alt_2 = 10

    crd2 = crd.to_enu()
    crd3 = crd2.to_enu(ref_longitude=ref_lon_2, ref_latitude=ref_lat_2, ref_altitude=ref_alt_2)
    crd4 = crd3.to_lla()
    crd5 = crd4.to_ecef()
    crd6 = crd5.to_enu(ref_longitude=ref_lon, ref_latitude=ref_lat, ref_altitude=ref_alt)

    assert np.all(crd6.east == pytest.approx(east))
    assert np.all(crd6.north == pytest.approx(north))
    assert np.all(crd6.up == pytest.approx(up))

    # test tree is created successfully
    crd.create_tree()


@pytest.mark.parametrize("n", [1, 100])
def test_ecef(n):
    """Testing conversion functions in ECEF class to convert to all other types and check get back to the same place.

    Args:
        n (int): Parameter to define size of location vector

    """
    rng = np.random.default_rng(42)
    x = rng.random(n) * 1000 + 4107864.0912067825
    y = rng.random(n) * 1000
    z = rng.random(n) * 5 + 4862789.037706433
    crd = ECEF(x=x, y=y, z=z)

    crd2 = crd.to_ecef()
    crd3 = crd2.to_enu()
    crd4 = crd3.to_lla()
    crd5 = crd4.to_ecef()

    assert np.all(crd5.x == pytest.approx(x))
    assert np.all(crd5.y == pytest.approx(y))
    assert np.all(crd5.z == pytest.approx(z))

    # test tree is created successfully
    crd.create_tree()


@pytest.mark.parametrize("grid_crd", ["to_lla", "to_enu", "to_ecef"])
@pytest.mark.parametrize("mid_crd", ["to_lla", "to_enu", "to_ecef"])
@pytest.mark.parametrize("dim", [2, 3])
def test_interpolate(dim, grid_crd, mid_crd):
    """Test interpolation function to/from different coordinate systems.

    Define a box or cube in of which all corners are relative close (so we don't get round-off errors due to the shape
    of the Earth) see if the interpolate function gives exactly the midpoint

    Args:
        dim (int): dimension for interpolation
        grid_crd (str): function for the coordinate transform  for grid
        mid_crd (str): unction for the coordinate transform for midpoint

    """

    if dim == 2:
        lon, lat = np.meshgrid([10, 10.01], [50, 50.01])
        z = np.array([[0, 1], [0, 1]])
        grid = LLA(longitude=lon, latitude=lat)
        mid = LLA(longitude=np.array(10.005), latitude=np.array(50.005))
    else:
        z = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
        lon, lat, alt = np.meshgrid([10, 10.01], [50, 50.01], [0, 10])
        grid = LLA(longitude=lon, latitude=lat, altitude=alt)
        mid = LLA(longitude=np.array(10.005), latitude=np.array(50.005), altitude=np.array(5))

    conv_func = getattr(grid, grid_crd)
    grid = conv_func()

    conv_func = getattr(mid, mid_crd)
    mid = conv_func()

    z_interp = grid.interpolate(values=z, locations=mid, dim=dim)

    assert z_interp == pytest.approx(0.5, abs=0.05)


def test_interpolate_outside():
    """Test interpolation fill extrapolation.

    Only use dim==2 because the test_interpolate already takes care of dim==3, main reason for this test is to see if
    **kwargs are passed on correctly.

    """
    lon, lat = np.meshgrid([10, 14], [50, 51])
    z = np.array([[0, 1], [0, 1]])
    grid = LLA(longitude=lon, latitude=lat)
    out = LLA(longitude=np.array(15), latitude=np.array(50.5))

    z_interp = grid.interpolate(values=z, locations=out, dim=2, fill_value=-99)

    assert z_interp == -99


def test_interpolate_single_values():
    """Test interpolation single input value.

    Test to see if all values are set to the single input value when only 1 input value is provided

    """
    location = LLA()
    location.from_array(np.array([[0, 0, 0]]))

    rng = np.random.default_rng(42)
    temp_array = rng.uniform(-30, 30, (5, 3))
    output_location = LLA()
    output_location.from_array(temp_array)

    interpolated_values = location.interpolate(values=np.array([39]), locations=output_location)

    assert np.all(interpolated_values == 39)


@pytest.mark.parametrize("dim", [2, 3])
def test_consistency_from_array_to_array(dim):
    """Test to_array and from_array methods.

    This test is designed to check for consistency between the to_array and from_array methods by just filling the
    attributes with random data and see if we get back the same data

    """
    rng = np.random.default_rng(42)
    n_samples = rng.integers(1, 100)
    array = rng.random((n_samples, dim))

    lla_object = LLA()
    lla_object.from_array(array)
    assert np.allclose(lla_object.to_array(dim=dim), array)

    enu_object = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    enu_object.from_array(array)
    assert np.allclose(enu_object.to_array(dim=dim), array)

    ecef_object = ECEF()
    ecef_object.from_array(array)
    assert np.allclose(ecef_object.to_array(dim=dim), array)


def test_nof_observations():
    """Test nof_observations calculation."""
    rng = np.random.default_rng(42)
    n_samples = rng.integers(1, 100)
    array = rng.random((n_samples, 3))

    lla_object = LLA()
    assert lla_object.nof_observations == 0
    lla_object.from_array(array)
    assert lla_object.nof_observations == n_samples

    enu_object = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    assert enu_object.nof_observations == 0
    enu_object.from_array(array)
    assert enu_object.nof_observations == n_samples

    ecef_object = ECEF()
    assert ecef_object.nof_observations == 0
    ecef_object.from_array(array)
    assert ecef_object.nof_observations == n_samples


@pytest.mark.parametrize("grid_type", ["rectangular", "spherical", "test"])
@pytest.mark.parametrize("dim", [2, 3])
def test_make_grid(grid_type, dim):
    """Test the make_grid method.

    Checks if the not implement error gets raised. Checks for correct number of samples generated. Checks if all
    samples are within the specified limits

    Args:
        grid_type (str): Type of grid to generate
        dim (int): Dimension of each grid (2 or 3)

    """
    enu_object = ENU(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    grid_limits = np.array([[-100, 100], [-100, 100], [-100, 100]])
    grid_limits = grid_limits[:dim, :]
    rng = np.random.default_rng(42)
    random_shape = rng.integers(1, 100, size=dim)

    if grid_type in ["rectangular", "spherical"]:
        grid = enu_object.make_grid(bounds=grid_limits, grid_type=grid_type, shape=random_shape)
        assert grid.shape[0] == random_shape.prod()

        for idx in range(dim):
            assert np.all(grid[:, idx] >= grid_limits[idx, 0])
            assert np.all(grid[:, idx] <= grid_limits[idx, 1])
    else:
        with pytest.raises(NotImplementedError):
            enu_object.make_grid(bounds=grid_limits, grid_type=grid_type, shape=random_shape)


@pytest.mark.parametrize("input_system", [LLA, ENU, ECEF])
@pytest.mark.parametrize("output_system", [LLA, ENU, ECEF])
def test_to_object_type(input_system, output_system):
    """Test the to_object method.

    Creates a very basic coordinate system object and check if it is converted to the right output system.
    Also checks if an error gets thrown when applicable

    Args:
        input_system (Coordinate): Input coordinate class
        output_system (Coordinate): Output coordinate class

    """
    if input_system == ENU:
        input_object = input_system(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    else:
        input_object = input_system()
    if output_system == ENU:
        output_object = output_system(ref_latitude=0, ref_longitude=0, ref_altitude=0)
    else:
        output_object = output_system()

    input_object.from_array(np.array([[0, 0, 0]]))
    test_object = input_object.to_object_type(output_object)
    assert isinstance(test_object, output_system)

    with pytest.raises(TypeError):
        input_object.to_object_type("test")


@pytest.mark.parametrize("dim", [2, 3])
def test_make_latin_hypercube(dim):
    """Test the make_latin_hypercube method.

    Checks for correct number of samples generated. Checks if all samples are within the specified limits

    Args:
        dim (int): Dimension of the hypercube (2 or 3)

    """
    grid_limits = np.array([[-100, 100], [-100, 100], [-100, 100]])
    grid_limits = grid_limits[:dim, :]
    rng = np.random.default_rng(42)
    random_number = rng.integers(1, 100)
    array = make_latin_hypercube(bounds=grid_limits, nof_samples=random_number)

    assert array.shape == (random_number, dim)

    for idx in range(dim):
        assert np.all(array[:, idx] >= grid_limits[idx, 0])
        assert np.all(array[:, idx] <= grid_limits[idx, 1])
