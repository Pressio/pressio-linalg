import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from pressiolinalg.linalg import _basic_max_via_python
from pressiolinalg.linalg import _basic_min_via_python


################################################################
####################### Helper functions #######################
################################################################

def _distribute_vector(global_vector, comm):
    num_processes = comm.Get_size()

    local_size = len(global_vector) // num_processes
    local_vector = np.zeros(local_size, dtype=int)

    comm.Scatter(global_vector, local_vector, root=0)

    return local_vector

def _distribute_array(global_array, comm):
    num_processes = comm.Get_size()

    dim = len(global_array.shape)

    if dim == 2:
        rows, cols = global_array.shape
        local_rows = rows // num_processes
        local_array = np.zeros((local_rows, cols), dtype=int)

    elif dim == 3:
        rows, cols, depth = global_array.shape
        local_rows = rows // num_processes
        local_array = np.zeros((local_rows, cols, depth), dtype=int)

    else:
        return None

    comm.Scatter(global_array, local_array, root=0)

    return local_array

def _get_inputs(rank, comm):
    num_processors = comm.Get_size()
    if rank == 1:
        global_arr = np.empty(num_processors * 2, dtype=np.int64)
        for i in range(num_processors * 2):
            global_arr[i] = i
        local_arr = _distribute_vector(global_arr, comm)
    elif rank == 2:
        global_arr = np.empty((num_processors * 2, 3), dtype=np.int64)
        for i in range(num_processors * 2):
            for j in range(3):
                global_arr[i][j] = i+j
        local_arr = _distribute_array(global_arr, comm)
    elif rank == 3:
        global_arr = np.empty((num_processors * 2, 3, 4), dtype=np.int64)
        for i in range(num_processors * 2):
            for j in range(3):
                for k in range(4):
                    global_arr[i][j][k] = i+j+k
        local_arr = _distribute_array(global_arr, comm)

    return local_arr, global_arr

def _min_max_setup(operation, rank, axis=None, out=None, comm=None):
    num_processors = comm.Get_size()
    local_arr, global_arr = _get_inputs(rank, comm)

    if operation == "min":
        min_result = _basic_min_via_python(local_arr, axis=axis, out=out, comm=comm)
        return min_result, np.min(global_arr)
    elif operation == "max":
        max_result = _basic_max_via_python(local_arr, axis=axis, out=out, comm=comm)
        return max_result, np.max(global_arr, axis=axis)
    else:
        return None, max(global_arr)

###############################################################
###################### Max and Min Tests ######################
###############################################################

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

def test_python_min_serial():
    vector = np.random.rand(10)
    assert _basic_min_via_python(vector) == np.min(vector)


if __name__ == "__main__":
    test_python_max_vector_mpi()
    test_python_min_vector_mpi()
    test_python_max_array_mpi()
    test_python_min_array_mpi()
    test_python_max_serial()
    test_python_min_serial()
