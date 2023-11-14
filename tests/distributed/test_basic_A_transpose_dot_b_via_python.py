import numpy as np
import random

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_A_transpose_dot_b_via_python


def distribute_array(global_array, comm):
    """Distribute an array among processes."""
    num_processes = comm.Get_size()

    n_rows, n_cols = global_array.shape

    n_local_rows = n_rows // num_processes
    local_array = np.zeros((n_local_rows, n_cols), dtype=float)

    comm.Scatter(global_array, local_array, root=0)

    return local_array

def test_basic_A_transpose_dot_b_via_python_gram():
    # Tests A_transpose dot A
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    A = np.random.rand(num_processes*2, 3)

    A_dist = distribute_array(A, comm)
    dot_result = _basic_A_transpose_dot_b_via_python(A_dist, A_dist, comm)

    if rank == 0:
        dot_expected = np.dot(A.transpose(), A)
        assert np.allclose(dot_result, dot_expected)


if __name__ == "__main__":
    test_basic_A_transpose_dot_b_via_python_gram()
