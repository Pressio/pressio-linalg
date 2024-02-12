import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

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

def get_solution(snapshots):
    dot_product = np.dot(snapshots.transpose(), snapshots)
    Lam,E = np.linalg.eig(dot_product)
    sigma = np.sqrt(Lam)
    U = np.zeros(np.shape(snapshots))
    U[:] = np.dot(snapshots, np.dot(E, np.diag(1./sigma)))
    ordering = np.argsort(sigma)[::-1]

    return U[:, ordering], sigma[ordering]

@pytest.mark.mpi(min_size=3)
def test_basic_svd_method_of_snapshots_impl_via_python():
    # Solve in parallel
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processes = comm.Get_size()

    global_snapshots, local_snapshots = create_snapshots(comm)
    U, s = _basic_svd_method_of_snapshots_impl_via_python(local_snapshots, comm)

    if rank == 0:
        # Solve in serial
        U_test, s_test = get_solution(global_snapshots)

        # Compare values
        assert np.allclose(U, U_test)
        assert s == s_test

def test_basic_svd_serial():
    snapshots = np.array([np.arange(0, 3)]).transpose()
    U, s = _basic_svd_method_of_snapshots_impl_via_python(snapshots)
    U_test, s_test = get_solution(snapshots)
    assert np.allclose(U, U_test)
    assert s == s_test


if __name__ == "__main__":
    test_basic_svd_method_of_snapshots_impl_via_python()
