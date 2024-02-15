import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from pressiolinalg.linalg import _basic_mean_via_python


################################################################
####################### Helper functions #######################
################################################################

def _distribute_vector(global_vector, comm):
    n_procs = comm.Get_size()

    local_size = len(global_vector) // n_procs
    local_vector = np.zeros(local_size, dtype=int)

    comm.Scatter(global_vector, local_vector, root=0)

    return local_vector

def _distribute_array(global_array, comm):
    n_procs = comm.Get_size()

    dim = len(global_array.shape)

    if dim == 2:
        rows, cols = global_array.shape
        local_rows = rows // n_procs
        local_array = np.zeros((local_rows, cols), dtype=int)

    elif dim == 3:
        rows, cols, depth = global_array.shape
        local_rows = rows // n_procs
        local_array = np.zeros((local_rows, cols, depth), dtype=int)

    else:
        return None

    comm.Scatter(global_array, local_array, root=0)

    return local_array

def _get_inputs(rank, comm):
    n_procs = comm.Get_size()

    if rank == 1:
        global_arr = np.empty(n_procs * 2, dtype=np.int64)
        for i in range(n_procs * 2):
            global_arr[i] = i
        local_arr = _distribute_vector(global_arr, comm)

    elif rank == 2:
        global_arr = np.empty((n_procs * 2, 3), dtype=np.int64)
        for i in range(n_procs * 2):
            for j in range(3):
                global_arr[i][j] = i+j
        local_arr = _distribute_array(global_arr, comm)

    elif rank == 3:
        global_arr = np.empty((n_procs * 2, 3, 4), dtype=np.int64)
        for i in range(n_procs * 2):
            for j in range(3):
                for k in range(4):
                    global_arr[i][j][k] = i+j+k
        local_arr = _distribute_array(global_arr, comm)

    return local_arr, global_arr

def _mean_setup(rank, axis=None, dtype=None, out=None, comm=None):
    n_procs = comm.Get_size()
    local_arr, global_arr = _get_inputs(rank, comm)

    mean_result = _basic_mean_via_python(local_arr, axis=axis, dtype=dtype, out=out, comm=comm)
    return mean_result, np.mean(global_arr, axis=axis, dtype=dtype)

###############################################################
###################### Max and Min Tests ######################
###############################################################

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
