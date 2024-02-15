import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from pressiolinalg import utils
from pressiolinalg.linalg import _basic_max_via_python
from pressiolinalg.linalg import _basic_min_via_python


########################
###  Set up problem  ###
########################

def _min_max_setup(operation, rank, axis=None, out=None, comm=None):
    num_processors = comm.Get_size()
    local_arr, global_arr = utils.get_local_and_global_arrays(rank, comm)

    if operation == "min":
        min_result = _basic_min_via_python(local_arr, axis=axis, out=out, comm=comm)
        return min_result, np.min(global_arr)
    elif operation == "max":
        max_result = _basic_max_via_python(local_arr, axis=axis, out=out, comm=comm)
        return max_result, np.max(global_arr, axis=axis)
    else:
        return None, max(global_arr)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_max_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _min_max_setup(operation="max", rank=1, comm=comm)
    assert result == expected

@pytest.mark.mpi(min_size=3)
def test_python_min_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected_min = _min_max_setup(operation="min", rank=1, comm=comm)
    assert result == expected_min

@pytest.mark.mpi(min_size=3)
def test_python_max_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="max", rank=2, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _min_max_setup(operation="max", rank=3, comm=comm)
    assert np.allclose(result_02, expected_02)

    test_out = np.empty(1)
    _, expected_03 = _min_max_setup(operation="max", rank=3, out=test_out, comm=comm)
    assert np.allclose(test_out, expected_03)

@pytest.mark.mpi(min_size=3)
def test_python_min_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="min", rank=2, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _min_max_setup(operation="min", rank=3, comm=comm)
    assert np.allclose(result_02, expected_02)

    test_out = np.empty(1)
    _, expected_03 = _min_max_setup(operation="min", rank=3, out=test_out, comm=comm)
    assert np.allclose(test_out, expected_03)

def test_python_max_serial():
    vector = np.random.rand(10)
    assert _basic_max_via_python(vector) == np.max(vector)

    array = np.random.rand(3, 10)
    assert _basic_max_via_python(array) == np.max(array)

def test_python_min_serial():
    vector = np.random.rand(10)
    assert _basic_min_via_python(vector) == np.min(vector)

    array = np.random.rand(3, 10)
    assert _basic_min_via_python(array) == np.min(array)


if __name__ == "__main__":
    test_python_max_vector_mpi()
    test_python_min_vector_mpi()
    test_python_max_array_mpi()
    test_python_min_array_mpi()
    test_python_max_serial()
    test_python_min_serial()
