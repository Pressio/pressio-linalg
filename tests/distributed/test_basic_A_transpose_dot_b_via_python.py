import numpy as np
import random

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_A_transpose_dot_b_via_python


################################################################
####################### Helper functions #######################
################################################################

def distribute_array(global_array, comm):
    """Distribute an array among processes"""
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    n_rows, n_cols = global_array.shape
    n_local_rows = int(n_rows // num_processes)

    local_array = np.empty((n_local_rows, n_cols), dtype=float)
    comm.Scatter(global_array, local_array, root=0)

    return local_array

def setup_A_transpose_dot_b(A, b, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    A_dist = distribute_array(A, comm)
    b_dist = distribute_array(b, comm)

    tmp_result = _basic_A_transpose_dot_b_via_python(A_dist, b_dist, comm)
    expected_result = np.dot(A.transpose(), b)

    return tmp_result, expected_result

###############################################################
######################## At dot b test ########################
###############################################################

def test_basic_A_transpose_dot_b_via_python_gram():
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    A = np.random.rand(num_processes*2, 3)
    result, expected = setup_A_transpose_dot_b(A, A, comm)
    assert np.allclose(result, expected)


if __name__ == "__main__":
    test_basic_A_transpose_dot_b_via_python_gram()
    test_basic_A_transpose_dot_b_via_python_matrix_vector()
