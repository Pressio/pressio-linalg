import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

import tests.test_utils as utils
from pressiolinalg.linalg import _basic_max_via_python
from pressiolinalg.linalg import _basic_min_via_python


########################
###  Set up problem  ###
########################

def _min_max_setup(operation, ndim, axis=None, comm=None):
    num_processors = comm.Get_size()
    local_arr, global_arr = utils.generate_local_and_global_arrays(ndim, comm)

    if operation == "min":
        min_result = _basic_min_via_python(local_arr, comm=comm)
        return min_result, np.min(global_arr)
    elif operation == "max":
        max_result = _basic_max_via_python(local_arr, axis=axis, comm=comm)
        return max_result, np.max(global_arr, axis=axis)
    else:
        return None, max(global_arr)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_max_examples_mpi():
    """Specifically tests the documented examples in _basic_max_via_python."""

    # Example 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        local_arr_1 = np.array([2.2, 3.3])
    elif rank == 1:
        local_arr_1 = np.array([40., 51., -24., 45.])
    elif rank == 2:
        local_arr_1 = np.array([-4.])
    res = _basic_max_via_python(local_arr_1, comm=comm)
    assert res == 51.

    # Example 2
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        local_arr_2 = np.array([[2.2, 1.3, 4.],
                                [3.3, 5.0, 33.]])
    elif rank == 1:
        local_arr_2 = np.array([[40., -2., -4.],
                                [51., 4., 6.],
                                [-24., 8., 9.],
                                [45., -3., -4.]])
    elif rank == 2:
        local_arr_2 = np.array([[-4., 8., 9.]])

    # Axis 0
    res_0 = _basic_max_via_python(local_arr_2, axis=0, comm=comm)
    assert np.allclose(res_0, np.array([51., 8., 33.]))

    # Axis 1
    res_1 = _basic_max_via_python(local_arr_2, axis=1, comm=comm)
    if rank == 0:
        assert np.allclose(res_1, np.array([4., 33.]))
    elif rank == 1:
        assert np.allclose(res_1, np.array([40., 51., 9., 45.]))
    elif rank == 2:
        assert np.allclose(res_1, np.array([9.]))

    # Example 3
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    global_arr = np.array(
        [[[2.,3.],[1.,6.],[4.,-7]],
         [[3.,4.],[5.,-1.],[3.,5.]],
         [[4.,2],[-2.,-2.],[-4.,5]],
         [[5.,8.],[4.,-1.],[6.,0]],
         [[-2.,2,],[8.,0.],[9.,3]],
         [[4.,1],[-3.,-6.],[-4.,1]],
         [[-4.,2.],[8.,0.],[9.,3.]]])
    if rank == 0:
        local_arr = np.array(
            [[[2.,3.],[1.,6.],[4.,-7]],
             [[3.,4.],[5.,-1.],[3.,5.]]])
    elif rank == 1:
        local_arr = np.array(
            [[[4.,2],[-2.,-2.],[-4.,5]],
             [[5.,8.],[4.,-1.],[6.,0]],
             [[-2.,2,],[8.,0.],[9.,3]],
             [[4.,1],[-3.,-6.],[-4.,1]]])
    elif rank == 2:
        local_arr = np.array(
            [[[-4.,2.],[8.,0.],[9.,3.]]])

    # Axis 0
    res_ex3_0 = _basic_max_via_python(local_arr, axis=0, comm=comm)
    assert np.allclose(res_ex3_0, np.max(global_arr, axis=0))

    # Axis 1
    res_ex3_1 = _basic_max_via_python(local_arr, axis=1, comm=comm)
    if rank == 0:
        expected_1 = np.array([[4., 6.],
                               [5., 5.]])
    elif rank == 1:
        expected_1 = np.array([[4., 5.],
                               [6., 8.],
                               [9., 3.],
                               [4., 1.]])
    elif rank == 2:
        expected_1 = np.array([[9., 3.]])
    assert np.allclose(res_ex3_1, expected_1)

    # Axis 2
    res_ex3_2 = _basic_max_via_python(local_arr, axis=2, comm=comm)
    if rank == 0:
        expected_2 = np.array([[3., 6., 4.],
                               [4., 5., 5.]])
    elif rank == 1:
        expected_2 = np.array([[4., -2., 5.],
                               [8., 4., 6.],
                               [2., 8., 9.],
                               [4., -3., 1.]])
    elif rank == 2:
        expected_2 = np.array([[2., 8., 9.]])
    assert np.allclose(res_ex3_2, expected_2)

@pytest.mark.mpi(min_size=3)
def test_python_min_examples_mpi():
    """Specifically tests the documented examples in _basic_min_via_python."""

    # Example 1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        local_arr_1 = np.array([2.2, 3.3])
    elif rank == 1:
        local_arr_1 = np.array([40., 51., -24., 45.])
    elif rank == 2:
        local_arr_1 = np.array([-4.])
    res = _basic_min_via_python(local_arr_1, comm=comm)
    assert res == -24.

    # Example 2
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        local_arr_2 = np.array([[2.2, 1.3, 4.],
                                [3.3, 5.0, 33.]])
    elif rank == 1:
        local_arr_2 = np.array([[40., -2., -4.],
                                [51., 4., 6.],
                                [-24., 8., 9.],
                                [45., -3., -4.]])
    elif rank == 2:
        local_arr_2 = np.array([[-4., 8., 9.]])

    # Axis 0
    res_0 = _basic_min_via_python(local_arr_2, axis=0, comm=comm)
    assert np.allclose(res_0, np.array([-24., -3., -4.]))

    # Axis 1
    res_1 = _basic_min_via_python(local_arr_2, axis=1, comm=comm)
    if rank == 0:
        assert np.allclose(res_1, np.array([1.3, 3.3]))
    elif rank == 1:
        assert np.allclose(res_1, np.array([-4., 4., -24., -4.]))
    elif rank == 2:
        assert np.allclose(res_1, np.array([-4.]))

@pytest.mark.mpi(min_size=3)
def test_python_max_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _min_max_setup(operation="max", ndim=1, comm=comm)
    assert result == expected

@pytest.mark.mpi(min_size=3)
def test_python_min_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected_min = _min_max_setup(operation="min", ndim=1, comm=comm)
    assert result == expected_min

@pytest.mark.mpi(min_size=3)
def test_python_max_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="max", ndim=2, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _min_max_setup(operation="max", ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)

@pytest.mark.mpi(min_size=3)
def test_python_max_on_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="max", ndim=2, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    # Make sure the result is a subset of the full max along the axis
    result_02, expected_02 = _min_max_setup(operation="max", ndim=3, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0


@pytest.mark.mpi(min_size=3)
def test_python_min_on_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="min", ndim=2, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    # Make sure the result is a subset of the full min along the axis
    result_02, expected_02 = _min_max_setup(operation="min", ndim=3, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0

    result_03, expected_03 = _min_max_setup(operation="min", ndim=3, axis=(1,2), comm=comm)
    assert len(np.setdiff1d(result_03, expected_03)) == 0


@pytest.mark.mpi(min_size=3)
def test_python_min_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="min", ndim=2, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _min_max_setup(operation="min", ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)


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
    test_python_max_examples_mpi()
    test_python_min_examples_mpi()
    test_python_max_vector_mpi()
    test_python_min_vector_mpi()
    test_python_max_array_mpi()
    test_python_min_array_mpi()
    test_python_max_on_axis_mpi()
    test_python_min_on_axis_mpi()
    test_python_max_serial()
    test_python_min_serial()
