import numpy as np

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_max_via_python
from pressiolinalg.linalg import _basic_min_via_python


################################################################
####################### Helper functions #######################
################################################################

def distribute_vector(global_vector, comm):
    """
    Distribute a global vector among processes.
    """
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    # Determine the size of the local portion of the vector
    local_size = int(len(global_vector) // num_processes)

    # Scatter the global vector to all processes
    local_vector = np.zeros(local_size)
    comm.Scatter([global_vector, MPI.DOUBLE], [local_vector, MPI.DOUBLE], root=0)

    return local_vector

def _min_max_setup(operation, comm):
    num_processors = comm.Get_size()
    global_vector = np.array([i for i in range(num_processors * 2)])
    local_vector = distribute_vector(global_vector, comm)
    if operation == "min":
        return global_vector, min(global_vector)
    else:
        return global_vector, max(global_vector)

###############################################################
###################### Max and Min Tests ######################
###############################################################

def test_basic_max_via_python():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    vec, expected_max = _min_max_setup("max", comm)

    result = _basic_max_via_python(vec, comm)

    if rank == 0:
        assert result == expected_max

def test_basic_min_via_python():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    vec, expected_min = _min_max_setup("min", comm)

    result = _basic_min_via_python(vec, comm)

    if rank == 0:
        assert result == expected_min

if __name__ == "__main__":
    test_basic_max_via_python()
    test_basic_min_via_python()
