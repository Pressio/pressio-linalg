
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1

The corresponding numpy API is included so we know what features we can add.
'''

import numpy as np

import mpi4py
from mpi4py import MPI


def _basic_func_via_python(vec):
    print("myfunc purely python")

def _basic_mpi_func_via_python(vec, comm):
    rank = comm.Get_rank()
    print(f"Python rank: {rank}")

def _basic_print_comm(comm):
    if comm == MPI.COMM_WORLD:
        print("Python received the world")
    else:
        print("Python received something else")

# np.max(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
def _basic_max_via_python(vec, comm):
    '''
    Finds the maximum of a distributed vector.

    Args:
        vec (np.array): Local vector
        comm (MPI_Comm): MPI communicator

    Returns:
        float: The maximum of the vector
    '''
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    if num_processes == 1:
        return np.max(vec)

    local_max = np.max(vec)
    data = comm.gather(local_max, root=0)

    global_max = 0
    if mpi_rank == 0:
        global_max = np.max(data)
    else:
        global_max = None

    global_max = comm.bcast(global_max, root=0)

    return global_max

# np.min(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
def _basic_min_via_python(vec, comm):
    '''
    Finds the minimum of a distributed vector.

    Args:
        vec (np.array): Local vector
        comm (MPI_Comm): MPI communicator

    Returns:
        float: The minimum of the vector
    '''
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    if num_processes == 1:
        return np.min(vec)

    local_min = np.min(vec)
    data = comm.gather(local_min, root=0)

    global_min = 0
    if mpi_rank == 0:
        global_min = np.min(data)
    else:
        global_min = None

    global_min = comm.bcast(global_min, root=0)

    return global_min

# np.dot(a, b, out=None)
# numpy.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj, axes, axis])
def _basic_A_transpose_dot_b_via_python(A, b, comm):
    '''
    Computes A^T B when A and B's columns are row-distributed.

    Args:
        A (np.array): Local array
        B (np.array): Local array
        comm (MPI_Comm): MPI communicator

    Returns:
        np.array: The dot product of A^T and B
    '''
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    if num_processes == 1:
        return np.dot(A.transpose(), b)

    tmp = np.dot(A.transpose(), b)
    data = comm.gather(tmp.flatten(), root=0)

    ATb_glob = np.zeros(np.size(tmp))

    if mpi_rank == 0:
        for j in range(0, num_processes):
            ATb_glob[:] += data[j]
        for j in range(1, num_processes):
            comm.Send(ATb_glob, dest=j)
    else:
        comm.Recv(ATb_glob, source=0)

    return np.reshape(ATb_glob, np.shape(tmp))

# np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)
def _basic_svd_method_of_snapshots_impl_via_python(snapshots, comm):
    '''
    Performs SVD via method of snapshots.

    Args:
        snapshots (np.array): Distributed array of snapshots
        comm (MPI_Comm): MPI communicator

    Returns:
        U (np.array): Phi, or modes; a numpy array where each column is a POD mode
        sigma (float): Energy; the energy associated with each mode (singular values)
    '''
    STS = _basic_A_transpose_dot_b_via_python(snapshots, snapshots, comm)
    Lam,E = np.linalg.eig(STS)
    sigma = np.sqrt(Lam)
    U = np.zeros(np.shape(snapshots))
    U[:] = np.dot(snapshots, np.dot(E, np.diag(1./sigma)))
    ## sort by singular values
    ordering = np.argsort(sigma)[::-1]
    return U[:, ordering], sigma[ordering]

# def _basic_orthogonalization_method_of_snapshots():


##############
### import ###
##############
try:
    from ._linalg import _myfunc, _print_comm
    myfunc = _myfunc
    print_comm = _print_comm
    # myfuncMPI = _myfuncMPI
    # max = _max
    # min = _min
    # At_dot_b = _At_dot_b
    # svd_method_of_snapshots = _svd_methods_of_snapshots
except ImportError as e:
    print(f"ImportError: {e}")
    # print full traceback
    import traceback
    traceback.print_exc()
    myfunc = _basic_func_via_python
    print_comm = _basic_print_comm
    # myfuncMPI = _basic_mpi_func_via_python
    # max = _basic_max_via_python
    # min = _basic_min_via_python
    # At_dot_b = _basic_A_transpose_dot_b_via_python
    # svd_method_of_snapshots = _basic_svd_method_of_snapshots_impl_via_python
