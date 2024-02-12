
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np

# ----------------------------------------------------
def _basic_max_via_python(vec, comm=None):
    '''
    Finds the maximum of a distributed vector.

    Args:
        vec (np.array): Local vector
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        float: The maximum of the vector, returned to all processes.
    '''
    dim = vec.ndim
    if (dim != 1):
        raise ValueError("This operation is currently supported only for a rank-1 array.")

    if comm and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        local_max = np.max(vec)
        global_max = np.zeros(1, dtype=vec.dtype)

        comm.Allreduce(local_max, global_max, op=MPI.MAX)

        return global_max[0]

    else:
        return np.max(vec)

# ----------------------------------------------------
def _basic_min_via_python(vec, comm=None):
    '''
    Finds the minimum of a distributed vector.

    Args:
        vec (np.array): Local vector
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        float: The minimum of the vector, returned to all processes.
    '''
    dim = vec.ndim
    if (dim != 1):
        raise ValueError("This operation is currently supported only for a rank-1 array.")

    if comm and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        local_min = np.min(vec)
        global_min = np.zeros(1, dtype=vec.dtype)

        comm.Allreduce(local_min, global_min, op=MPI.MIN)

        return global_min[0]

    else:
        return np.min(vec)

# ----------------------------------------------------
def _basic_product_via_python(flagA, flagB, alpha, A, B, beta, C, comm=None):
    '''
    Computes C = beta*C + alpha*op(A)*op(B), where A and B are row-distributed matrices.

    Args:
        flagA (str): Determines the orientation of A, "T" for transpose or "N" for non-transpose.
        flagB (str): Determines the orientation of B, "T" for transpose or "N" for non-transpose.
        alpha (float): Coefficient of AB.
        A (np.array): 2-D matrix
        B (np.array): 2-D matrix
        beta (float): Coefficient of C.
        C (np.array): 2-D matrix to be filled with the product
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        C (np.array): The specified product
    '''
    if flagA == "N":
        mat1 = A * alpha
    elif flagA == "T":
        mat1 = A.transpose() * alpha
    else:
        raise ValueError("flagA not recognized; use either 'N' or 'T'")

    if flagB == "N":
        mat2 = B
    elif flagB == "T":
        mat2 = B.transpose()
    else:
        raise ValueError("flagB not recognized; use either 'N' or 'T'")

    # CONSTRAINTS
    mat1_shape = np.shape(mat1)
    mat2_shape = np.shape(mat2)

    if (mat1.ndim == 2) and (mat2.ndim == 2):
        if np.shape(C) != (mat1_shape[0], mat2_shape[1]):
            raise ValueError(f"Size of output array C ({np.shape(C)}) is invalid. For A (m x n) and B (n x l), C has dimensions (m x l)).")

        if mat1_shape[1] != mat2_shape[0]:
            raise ValueError(f"Invalid input array size. For A (m x n), B must be (n x l).")

    if (mat1.ndim != 2) | (mat2.ndim != 2):
        raise ValueError(f"This operation currently supports rank-2 tensors.")

    if comm and comm.Get_size() > 1:

        import mpi4py
        from mpi4py import MPI

        local_product = np.dot(mat1, mat2)
        global_product = np.zeros_like(C, dtype=local_product.dtype)
        comm.Allreduce(local_product, global_product, op=MPI.SUM)
        if beta == 0:
            np.copyto(C, global_product)
        else:
            new_C = beta * C + global_product
            np.copyto(C, new_C)

    else:
        product = np.dot(mat1, mat2)
        if beta == 0:
            np.copyto(C, product)
        else:
            new_C = beta * C + product
            np.copyto(C, new_C)

    return

# ----------------------------------------------------
def _basic_svd_method_of_snapshots_impl_via_python(snapshots, comm=None):
    '''
    Performs SVD via method of snapshots.

    Args:
        snapshots (np.array): Distributed array of snapshots
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        U (np.array): Phi, or modes; a numpy array where each column is a POD mode
        sigma (float): Energy; the energy associated with each mode (singular values)
    '''
    gram_matrix = np.zeros((np.shape(snapshots)[1], np.shape(snapshots)[1]))
    _basic_product_via_python("T", "N", 1, snapshots, snapshots, 0, gram_matrix, comm)
    eigenvalues,eigenvectors = np.linalg.eig(gram_matrix)
    sigma = np.sqrt(eigenvalues)
    modes = np.zeros(np.shape(snapshots))
    modes[:] = np.dot(snapshots, np.dot(eigenvectors, np.diag(1./sigma)))
    ## sort by singular values
    ordering = np.argsort(sigma)[::-1]
    print("function modes:", modes[:, ordering])
    return modes[:, ordering], sigma[ordering]

# ----------------------------------------------------
# ----------------------------------------------------

# Define public facing API
max = _basic_max_via_python
min = _basic_min_via_python
product = _basic_product_via_python
svd_method_of_snapshots = _basic_svd_method_of_snapshots_impl_via_python

