import numpy as np

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_max_via_python
from pressiolinalg.linalg import _basic_min_via_python


################################################################
####################### Helper functions #######################
################################################################

def distribute_vector(global_vector, comm):
    num_processes = comm.Get_size()

    local_size = len(global_vector) // num_processes
    local_vector = np.zeros(local_size, dtype=int)

    comm.Scatter(global_vector, local_vector, root=0)

    return local_vector

def _min_max_setup(operation, comm):
    num_processors = comm.Get_size()
    global_vector = np.array([i + 1 for i in range(num_processors * 2)], dtype=np.int64)

    local_vector = distribute_vector(global_vector, comm)

    if operation == "min":
        min_result = _basic_min_via_python(local_vector, comm)
        return min_result, min(global_vector)
    elif operation == "max":
        max_result = _basic_max_via_python(local_vector, comm)
        return max_result, max(global_vector)
    else:
        return None, max(global_vector)  # Fails

###############################################################
###################### Max and Min Tests ######################
###############################################################

def test_basic_max_via_python():
    comm = MPI.COMM_WORLD
    result, expected = _min_max_setup("max", comm)
    assert result == expected

def test_basic_min_via_python():
    comm = MPI.COMM_WORLD
    result, expected_min = _min_max_setup("min", comm)
    assert result == expected_min


if __name__ == "__main__":
    test_basic_max_via_python()
    test_basic_min_via_python()
