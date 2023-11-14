
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np

def _basic_func_via_python(vec):
    print("myfunc purely python")

# def _basic_max_via_python(vec, mpiComm):
#   print("_basic_max_via_python")

def _basic_max_via_python(vec, mpiComm):
    '''
    Finds the maximum of a distributed vector.

    Args:
        vec (np.array): Local vector
        mpiComm (MPI_Comm): MPI communicator

    Returns:
        float: The maximum of the vector
    '''
    mpi_rank = mpiComm.Get_rank()
    num_processes = mpiComm.Get_size()

    if num_processes == 1:
        return max(vec)

    local_max = max(vec)
    data = mpiComm.gather(local_max, root=0)

    global_max = 0
    if mpi_rank == 0:
        global_max = max(data)
    else:
        global_max = None

    global_max = mpiComm.bcast(global_max, root=0)

    return global_max

def _basic_min_via_python(vec, mpiComm):
    '''
    Finds the minimum of a distributed vector.

    Args:
        vec (np.array): Local vector
        mpiComm (MPI_Comm): MPI communicator

    Returns:
        float: The minimum of the vector
    '''
    mpi_rank = mpiComm.Get_rank()
    num_processes = mpiComm.Get_size()

    if num_processes == 1:
        return min(vec)

    local_min = min(vec)
    data = mpiComm.gather(local_min, root=0)

    global_min = 0
    if mpi_rank == 0:
        global_min = min(data)
    else:
        global_min = None

    global_min = mpiComm.bcast(global_min, root=0)

    return global_min

def _basic_A_transpose_dot_b_via_python(A, b, comm):
    '''
    Compute A^T B when A's columns are distributed.
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

def _basic_svd_method_of_snapshots_via_python(snapshots, comm):
    '''Performs SVD via method of snapshots'''
    #
    # outputs:
    # modes, Phi: numpy array where each column is a POD mode
    # energy, sigma: energy associated with each mode (singular values)

    STS = A_transpose_dot_bImpl(snapshots, snapshots, comm)
    Lam,E = np.linalg.eig(STS)
    sigma = np.sqrt(Lam)
    U = np.zeros(np.shape(snapshots))
    U[:] = np.dot(snapshots, np.dot(E, np.diag(1./sigma)))
    ## sort by singular values
    ordering = np.argsort(sigma)[::-1]
    return U[:, ordering], sigma[ordering]

# def _basic_orthogonalization_method_of_snapshots():

# def _basic_stretch_svd():

# def _basic_stretch_qr():

# def _basic_stretch_basic_linear_solve():

# def _basic_unlikely_svd():

# def _basic_unlikely_qr():

# def _basic_unlikely_basic_linear_solve():

##############
### import ###
##############
try:
    from ._linalg import myfunc
except ImportError:
    myfunc = _basic_func_via_python
    #max = _basic_max_via_python
