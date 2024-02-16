import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from pressiolinalg import utils
from pressiolinalg.linalg import _basic_mean_via_python


########################
###  Set up problem  ###
########################

def _mean_setup(rank, dtype=None, out=None, comm=None):
    n_procs = comm.Get_size()
    local_arr, global_arr = utils.get_local_and_global_arrays(rank, comm)
    mean_result = _basic_mean_via_python(local_arr, dtype=dtype, out=out, comm=comm)
    return mean_result, np.mean(global_arr, dtype=dtype)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_mean_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _mean_setup(rank=1, comm=comm)
    assert result == expected

@pytest.mark.mpi(min_size=3)
def test_python_mean_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _mean_setup(rank=2, dtype=np.float32, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _mean_setup(rank=3, comm=comm)
    assert np.allclose(result_02, expected_02)

    test_out = np.empty(1)
    _, expected_03 = _mean_setup(rank=3, out=test_out, comm=comm)
    assert np.allclose(test_out, expected_03)

def test_python_mean_serial():
    vector = np.random.rand(10)
    assert _basic_mean_via_python(vector) == np.mean(vector)

    array = np.random.rand(3, 10)
    assert _basic_mean_via_python(array) == np.mean(array)


if __name__ == "__main__":
    test_python_mean_vector_mpi()
    test_python_mean_array_mpi()
    test_python_mean_serial()
