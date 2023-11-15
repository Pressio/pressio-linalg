import numpy as np

import mpi4py
from mpi4py import MPI

from pressiolinalg.linalg import _basic_svd_method_of_snapshots_impl_via_python


def distribute_array(global_array, comm):
    '''Distribute an array among processes.'''
    num_processes = comm.Get_size()

    n_rows, n_cols = global_array.shape
    n_local_rows = n_rows // num_processes
    local_array = np.zeros((n_local_rows, n_cols), dtype=int)

    comm.Scatter(global_array, local_array, root=0)

    return local_array

def create_snapshots(comm):
    num_processes = comm.Get_size()
    global_snapshots = np.array([np.arange(0, num_processes)]).transpose()
    local_snapshots = distribute_array(global_snapshots, comm)
    return global_snapshots, local_snapshots

def test_basic_svd_method_of_snapshots_impl_via_python():
    # Solve in parallel
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processes = comm.Get_size()

    global_snapshots, local_snapshots = create_snapshots(comm)
    U, s = _basic_svd_method_of_snapshots_impl_via_python(local_snapshots, comm)

    if rank == 0:
        # Solve in serial
        dot_product = np.dot(global_snapshots.transpose(), global_snapshots)
        Lam,E = np.linalg.eig(dot_product)
        sigma = np.sqrt(Lam)
        U = np.zeros(np.shape(global_snapshots))
        U[:] = np.dot(global_snapshots, np.dot(E, np.diag(1./sigma)))
        ordering = np.argsort(sigma)[::-1]
        U_serial = U[:, ordering]
        sigma_serial = sigma[ordering]

        # Compare values
        assert np.allclose(U_serial, U)
        assert sigma_serial == sigma


if __name__ == "__main__":
    test_basic_svd_method_of_snapshots_impl_via_python()
