
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np
import mpi4py
from mpi4py import MPI

# ----------------------------------------------------
def _basic_max_via_python(vec, comm):
    '''
    Finds the maximum of a distributed vector.

    Args:
        vec (np.array): Local part of the vector
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

# ----------------------------------------------------
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

# ----------------------------------------------------
def _basic_product_via_python(flagA, flagB, alpha, A, B, beta, C, comm):
    '''
    Computes C = AB, where A and B's columns are row-distributed.

    Args:
        flagA (str): Determines the orientation of A, "T" for transpose or "N" for non-transpose.
        flagB (str): Determines the orientation of B, "T" for transpose or "N" for non-transpose.
        alpha (float): Coefficient of A.
        A (np.array): Local array
        B (np.array): Local array
        beta (float): Coefficient of B.
        C (np.array): Array to be overwritten with product
        comm (MPI_Comm): MPI communicator

    Returns:
        C (np.array): The dot product of A^T and B
    '''
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    if flagA == "N":
        mat1 = A * alpha
    elif flagA == "T":
        mat1 = A.transpose() * alpha
    else:
        print("Error!")
        mat1 = np.zeros(3,3)

    if flagB == "N":
        mat2 = B * beta
    elif flagB == "T":
        mat2 = B.transpose() * beta
    else:
        print("Error!")
        mat2 = np.zeros(3,3)

    if num_processes == 1:
        product = np.dot(mat1, mat2)
        np.copyto(C, product)

    tmp = np.dot(mat1, mat2)
    data = comm.gather(tmp.flatten(), root=0)

    C = np.reshape(C, np.size(tmp))

    if mpi_rank == 0:
        for j in range(0, num_processes):
            C[:] += data[j]
        for j in range(1, num_processes):
            comm.Send(C, dest=j)
    else:
        comm.Recv(C, source=0)

    C = np.reshape(C, np.shape(tmp))

    return C

# ----------------------------------------------------
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
    STS = np.zeros((np.shape(snapshots)[1], np.shape(snapshots)[1]))
    _basic_product_via_python("T", "N", 1, snapshots, snapshots, 1, STS, comm)
    Lam,E = np.linalg.eig(STS)
    sigma = np.sqrt(Lam)
    U = np.zeros(np.shape(snapshots))
    U[:] = np.dot(snapshots, np.dot(E, np.diag(1./sigma)))
    ## sort by singular values
    ordering = np.argsort(sigma)[::-1]
    return U[:, ordering], sigma[ordering]
