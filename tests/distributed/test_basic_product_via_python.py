import numpy as np
import random

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_product_via_python


def distribute_array(global_array, comm):
    '''Distribute an array among all processes.'''
    num_processes = comm.Get_size()

    n_rows, n_cols = global_array.shape

    n_local_rows = n_rows // num_processes
    local_array = np.zeros((n_local_rows, n_cols), dtype=float)

    comm.Scatter(global_array, local_array, root=0)

    return local_array

# def run_product(A, B, comm):
#     rank = comm.Get_rank()

#     A_dist = distribute_array(A, comm)
#     B_dist = distribute_array(B, comm)

#     dot_result = _basic_product_via_python(A_dist, B_dist, comm)

#     if rank == 0:
#         dot_expected = np.dot(A.transpose(), B)
#         return dot_result, dot_expected

#     else:
#         return None, None

def test_basic_product_via_python_gram():
    '''Tests 2A^T 3A where A is row-distributed'''
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    m = num_processes*2
    n = 3

    A = np.random.rand(m, n)
    A_dist = distribute_array(A, comm)
    C = np.zeros((n,n))

    _basic_product_via_python("T", "N", 2, A_dist, A_dist, 3, C, comm) # 2A^T 3A

    if rank == 0:
        gram_expected = np.dot(2*A.transpose(), 3*A)
        assert np.allclose(C, gram_expected)

if __name__ == "__main__":
    test_basic_product_via_python_gram()
