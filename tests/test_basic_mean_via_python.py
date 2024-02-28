import math
import warnings
import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

import tests.test_utils as utils
from pressiolinalg.linalg import _basic_mean_via_python


########################
###  Set up problem  ###
########################

def _mean_setup(ndim, dtype=None, axis=None, comm=None):
    local_arr, global_arr = utils.generate_local_and_global_arrays(ndim, comm)
    mean_result = _basic_mean_via_python(local_arr, dtype=dtype, axis=axis, comm=comm)
    return mean_result, np.mean(global_arr, dtype=dtype, axis=axis)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_mean_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _mean_setup(ndim=1, comm=comm)
    np.testing.assert_almost_equal(result, expected, decimal=10)

@pytest.mark.mpi(min_size=3)
def test_python_mean_null_vector_mpi():
    comm = MPI.COMM_WORLD

    # Both pla.mean and np.mean will output warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, expected = _mean_setup(ndim=0, comm=comm)
        assert math.isnan(result)
        assert math.isnan(expected)

@pytest.mark.mpi(min_size=3)
def test_python_mean_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _mean_setup(ndim=2, dtype=np.float32, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _mean_setup(ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)

@pytest.mark.mpi(min_size=3)
def test_python_mean_array_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _mean_setup(ndim=2, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _mean_setup(ndim=3, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0

    result_03, expected_03 = _mean_setup(ndim=3, axis=2, comm=comm)
    assert len(np.setdiff1d(result_03, expected_03)) == 0

def test_python_mean_serial():
    vector = np.random.rand(10)
    np.testing.assert_almost_equal(_basic_mean_via_python(vector), np.mean(vector), decimal=10)

    array = np.random.rand(3, 10)
    np.testing.assert_almost_equal(_basic_mean_via_python(array), np.mean(array), decimal=10)


if __name__ == "__main__":
    test_python_mean_null_vector_mpi()
    test_python_mean_vector_mpi()
    test_python_mean_array_mpi()
    test_python_mean_array_axis_mpi()
    test_python_mean_serial()
