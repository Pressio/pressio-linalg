import numpy as np
import random

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_A_transpose_dot_b_via_python


################################################################
####################### Helper functions #######################
################################################################

def distribute_matrix(matrix, comm):
    """
    Distribute a matrix among processes.
    """
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    rows, cols = matrix.shape
    local_rows = int(rows // num_processes)

    # Scatter the matrix rows to all processes
    local_matrix_rows = np.zeros((local_rows, cols))
    comm.Scatter([matrix, MPI.DOUBLE], [local_matrix_rows, MPI.DOUBLE], root=0)

    return local_matrix_rows

def setup_A_transpose_dot_b(A, b, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    A_dist = distribute_matrix(A, comm)
    b_dist = distribute_matrix(b, comm)

    result = _basic_A_transpose_dot_b_via_python(A_dist, b_dist, comm)
    if rank == 0:
        print(f"\n-----------------\nresult:\n {result}")

    expected_result = np.dot(A.transpose(), b).astype(float)
    if rank == 0:
        print(f"\nexpected_result: \n {expected_result}\n-----------------\n")

    return result, expected_result


##############################################################
####################### At dot b tests #######################
##############################################################

def test_basic_A_transpose_dot_b_via_python_gram():
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    A = np.random.rand(num_processes*2, 3)
    result, expected_result = setup_A_transpose_dot_b(A, A, comm)
    assert result.all() == expected_result.all()

# def test_basic_A_transpose_dot_b_via_python_standard():
#     comm = MPI.COMM_WORLD
#     A = np.array([[1,2,3],
#                   [4,5,6]])
#     b = np.array([2,3,4])
#     result, expected_result = setup_A_transpose_dot_b(A, b, comm)
#     np.testing.assert_allclose(result, expected_result, atol=1e-10)

if __name__ == "__main__":
    test_basic_A_transpose_dot_b_via_python_gram()
    # test_basic_A_transpose_dot_b_via_python_standard()
