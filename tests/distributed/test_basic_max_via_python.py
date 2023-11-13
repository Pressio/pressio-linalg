import numpy as np

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_max_via_python
from pressiolinalg.linalg import _basic_min_via_python


def _min_max_setup(operation="max"):
    global_vector = np.array([i for i in range(10)])
    if operation == "min":
        return global_vector, min(global_vector)
    else:
        return global_vector, max(global_vector)

def test_basic_max_via_python():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    vec, expected_max = _min_max_setup("max")

    result = _basic_max_via_python(vec, comm)

    if rank == 0:
        assert result == expected_max

def test_basic_min_via_python():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    vec, expected_min = _min_max_setup("min")

    result = _basic_min_via_python(vec, comm)

    if rank == 0:
        assert result == expected_min

if __name__ == "__main__":
    test_basic_max_via_python()
    test_basic_min_via_python()