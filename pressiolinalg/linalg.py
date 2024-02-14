
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np

# ----------------------------------------------------
def _basic_max_via_python(a, axis=None, out=None, comm=None):
    '''
    Finds the maximum of a distributed vector.

    Args:
        a (np.ndarray): Local input data
        axis (int or tuple of ints): Axis or axes along which to operate (by default, flattened input is used)
        out (np.ndarray): Output array in which to place the result (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        max (np.ndarray or scalar): The maximum of the array, returned to all processes.
    '''
    max_dim = 1 if axis is None else a.ndim - 1 if isinstance(axis, int) else a.ndim - len(axis)

    if comm is not None and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        # TO DO: Add support for axis (and out)
        if axis is not None:
            raise ValueError("The axis argument is not currently supported.")

        if out is not None:
            assert len(out.shape) == max_dim, "out must have correct dimensions."

        local_max = np.max(a, axis=axis)
        global_max = np.zeros(max_dim, dtype=a.dtype)

        comm.Allreduce(local_max, global_max, op=MPI.MAX)

        if out is None:
            if global_max.shape == 1:
                return global_max[0]
            else:
                return global_max
        else:
            np.copyto(out, global_max)
            return

    else:
        return np.max(a, axis=axis, out=out)

# ----------------------------------------------------
def _basic_min_via_python(a, axis=None, out=None, comm=None):
    '''
    Finds the minimum of a distributed vector.

    Args:
        a (np.ndarray): Local input data
        axis (int or tuple of ints): Axis or axes along which to operate (by default, flattened input is used)
        out (np.ndarray): Output array in which to place the result (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        min (np.ndarray or scalar): The minimum of the array, returned to all processes.
    '''
    min_dim = 1 if axis is None else a.ndim - 1 if isinstance(axis, int) else a.ndim - len(axis)

    if comm is not None and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        # TO DO: Add support for axis (and out)
        if axis is not None:
            raise ValueError("The axis argument is not currently supported.")

        if out is not None:
            assert len(out.shape) == min_dim, "out must have correct dimensions."

        local_min = np.min(a, axis=axis)
        global_min = np.zeros(min_dim, dtype=a.dtype)

        comm.Allreduce(local_min, global_min, op=MPI.MIN)

        if out is None:
            if global_min.shape == 1:
                return global_min[0]
            else:
                return global_min
        else:
            np.copyto(out, global_min)
            return

    else:
        return np.min(a, axis=axis, out=out)

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

    if comm is not None and comm.Get_size() > 1:
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

