import numpy as np

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_A_transpose_dot_b_via_python


def test_basic_A_transpose_dot_b_via_python():

    # Distribute the matrix A among communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize input array
    A = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

    # Calculate the local part of A and b for each process
    local_A = A[:, rank::size]

    # Test the function with the distributed A and b
    result = _basic_A_transpose_dot_b_via_python(local_A, local_A, comm)

    # Gather results to the root process for comparison
    all_results = comm.gather(result, root=0)

    if rank == 0:
        # Combine results to get the global result
        global_result = np.zeros_like(result)
        for j in range(size):
            global_result += all_results[j]

        # Calculate the expected result for the comparison
        expected_result = np.dot(A.transpose(), A)

        # Check if the results are equal within a tolerance
        np.testing.assert_allclose(global_result, expected_result, atol=1e-10)

if __name__ == "__main__":
    test_basic_A_transpose_dot_b_via_python()
