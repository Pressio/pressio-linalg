import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from pressiolinalg import utils
from pressiolinalg.linalg import _basic_std_via_python


########################
###  Set up problem  ###
########################

def _std_setup(rank, dtype=None, out=None, ddof=0, comm=None):
    n_procs = comm.Get_size()
    local_arr, global_arr = utils.get_local_and_global_arrays(rank, comm)

    std_result = _basic_std_via_python(local_arr, dtype=dtype, out=out, ddof=ddof, comm=comm)
    return std_result, np.std(global_arr, dtype=dtype, ddof=ddof)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_std_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _std_setup(rank=1, comm=comm)
    assert result == expected

@pytest.mark.mpi(min_size=3)
def test_python_std_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _std_setup(rank=2, dtype=np.float32, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _std_setup(rank=3, ddof=1, comm=comm)
    assert np.allclose(result_02, expected_02)

    test_out = np.empty(1)
    _, expected_03 = _std_setup(rank=3, out=test_out, comm=comm)
    assert np.allclose(test_out, expected_03)

def test_python_std_serial():
    vector = np.random.rand(10)
    assert _basic_std_via_python(vector) == np.std(vector)

    array = np.random.rand(3, 10)
    assert _basic_std_via_python(array) == np.std(array)


if __name__ == "__main__":
    test_python_std_vector_mpi()
    test_python_std_array_mpi()
    test_python_std_serial()
