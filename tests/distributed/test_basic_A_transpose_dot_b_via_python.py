import numpy as np
import random

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_A_transpose_dot_b_via_python


def distribute_array(global_array, comm):
    '''Distribute an array among all processes.'''
    num_processes = comm.Get_size()

    n_rows, n_cols = global_array.shape

    n_local_rows = n_rows // num_processes
    local_array = np.zeros((n_local_rows, n_cols), dtype=float)

    comm.Scatter(global_array, local_array, root=0)

    return local_array

def run_At_dot_B(A, B, comm):
    rank = comm.Get_rank()

    A_dist = distribute_array(A, comm)
    B_dist = distribute_array(B, comm)

    dot_result = _basic_A_transpose_dot_b_via_python(A_dist, B_dist, comm)

    if rank == 0:
        dot_expected = np.dot(A.transpose(), B)
        return dot_result, dot_expected

    else:
        return None, None

def test_basic_A_transpose_dot_b_via_python_gram():
    '''Tests A^T A where A is row-distributed'''
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    A = np.random.rand(num_processes*2, 3)

    gram_result, gram_expected = run_At_dot_B(A, A, comm) # A^T A

    if rank == 0:
        assert np.allclose(gram_result, gram_expected)

def test_basic_A_transpose_dot_b_via_python_mat_mat():
    '''Tests A^T B where A and B are different row-distributed matrices'''
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    A = np.random.rand(num_processes*2, 3)
    B = np.random.rand(num_processes*2, 3)

    mat_mat_result, mat_mat_expected = run_At_dot_B(A, A, comm) # A^T B

    if rank == 0:
        assert np.allclose(mat_mat_result, mat_mat_expected)


if __name__ == "__main__":
    test_basic_A_transpose_dot_b_via_python_gram()
    test_basic_A_transpose_dot_b_via_python_mat_mat()
