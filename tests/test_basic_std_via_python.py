import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

import tests.test_utils as utils
from pressiolinalg.linalg import _basic_std_via_python


########################
###  Set up problem  ###
########################

def _std_setup(ndim, dtype=None, axis=None, ddof=0, comm=None):
    n_procs = comm.Get_size()
    local_arr, global_arr = utils.generate_local_and_global_arrays(ndim, comm)

    std_result = _basic_std_via_python(local_arr, dtype=dtype, axis=axis, ddof=ddof, comm=comm)
    return std_result, np.std(global_arr, dtype=dtype, axis=axis, ddof=ddof)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_std_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _std_setup(ndim=1, comm=comm)
    np.testing.assert_almost_equal(result, expected, decimal=10)

@pytest.mark.mpi(min_size=3)
def test_python_std_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _std_setup(ndim=2, dtype=np.float32, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _std_setup(ndim=3, ddof=1, comm=comm)
    assert np.allclose(result_02, expected_02)

@pytest.mark.mpi(min_size=3)
def test_python_std_array_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _std_setup(ndim=2, dtype=np.float32, axis=0, comm=comm)
    print("result_01")
    print(result_01)
    print("expected_01")
    print(expected_01)

    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _std_setup(ndim=2, dtype=np.float32, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0

    result_03, expected_03 = _std_setup(ndim=3, ddof=1, axis=2, comm=comm)
    assert len(np.setdiff1d(result_03, expected_03)) == 0

def test_python_std_serial():
    vector = np.random.rand(10)
    np.testing.assert_almost_equal(_basic_std_via_python(vector), np.std(vector), decimal=10)

    array = np.random.rand(3, 10)
    np.testing.assert_almost_equal(_basic_std_via_python(array), np.std(array), decimal=10)


if __name__ == "__main__":
    test_python_std_vector_mpi()
    test_python_std_array_mpi()
    test_python_std_array_axis_mpi()
    test_python_std_serial()
