
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np
from pressiolinalg import utils

# ----------------------------------------------------
def _basic_max_via_python(a, out=None, comm=None):
    '''
    Finds the maximum of a distributed vector.

    Args:
        a (np.ndarray): Local input data
        out (np.ndarray): Output array in which to place the result (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        max (np.ndarray or scalar): The maximum of the array, returned to all processes.
    '''
    if comm is not None and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        utils.verify_out_size(out, 1)

        local_max = np.max(a)
        global_max = comm.allreduce(local_max, op=MPI.MAX)

        return utils.return_to_out_if_given(global_max, out)

    else:
        return np.max(a, out=out)

# ----------------------------------------------------
def _basic_min_via_python(a, out=None, comm=None):
    '''
    Finds the minimum of a distributed vector.

    Args:
        a (np.ndarray): Local input data
        out (np.ndarray): Output array in which to place the result (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        min (np.ndarray or scalar): The minimum of the array, returned to all processes.
    '''
    if comm is not None and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        utils.verify_out_size(out, 1)

        local_min = np.min(a)
        global_min = comm.allreduce(local_min, op=MPI.MIN)

        return utils.return_to_out_if_given(global_min, out)

    else:
        return np.min(a, out=out)

# ----------------------------------------------------
def _basic_mean_via_python(a, dtype=None, out=None, comm=None):
    '''
    Finds the mean of a distributed array.

    Args:
        a (np.ndarray): Local input data
        dtype (data-type): Type to use in computing the mean (by default, uses the input dtype, float32 for integer inputs)
        out (np.ndarray): Output array in which to place the result (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        mean (np.ndarray or scalar): The mean of the array, returned to all processes.
    '''
    if comm is not None and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        n_procs = comm.Get_size()

        utils.verify_out_size(out, 1)

        local_size = a.size
        global_size = comm.allreduce(local_size, op=MPI.SUM)

        local_sum = np.sum(a)
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)

        global_mean = global_sum / global_size

        return utils.return_to_out_if_given(global_mean, out)

    else:
        return np.mean(a, dtype=dtype, out=out)

# ----------------------------------------------------
def _basic_std_via_python(a, dtype=None, out=None, ddof=0, comm=None):
    '''
    Finds the standard deviation of a distributed array.

    Args:
        a (np.ndarray): Local input data
        dtype (data-type): Type to use in computing the standard deviation (by default, uses the input dtype, float32 for integer inputs)
        out (np.ndarray): Output array in which to place the result (default: None)
        ddof (int): Delta degrees of freedom used in divisor N - ddof (default: 0)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        mean (np.ndarray or scalar): The mean of the array, returned to all processes.
    '''
    if comm is not None and comm.Get_size() > 1:
        import mpi4py
        from mpi4py import MPI

        n_procs = comm.Get_size()

        utils.verify_out_size(out, 1)

        # Get total number of elements
        global_size = comm.allreduce(a.size, op=MPI.SUM)

        # Get standard deviation
        global_mean = _basic_mean_via_python(a, dtype=dtype, comm=comm)
        local_sq_diff = np.sum(np.square(a - global_mean))
        global_sq_diff = comm.allreduce(local_sq_diff, op=MPI.SUM)
        std_dev = np.sqrt(global_sq_diff / (global_size - ddof))

        return utils.return_to_out_if_given(std_dev, out)

    else:
        return np.std(a, dtype=dtype, out=out, ddof=ddof)

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
def _thin_svd_via_method_of_snaphosts(snapshots, comm=None):
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

def _thin_svd_auto_select_algo(M, comm):
    # for now this is it, improve later
    return _thin_svd_via_method_of_snaphosts(M, comm)

def _thin_svd(M, comm=None, method='auto'):
  '''
  Preconditions:
    - M is rank-2 tensor
    - if M is distributed, M is distributed over its 0-th axis (row distribution)
    - allowed choices for method are "auto", "method_of_snapshots"

  Returns:
    - left singular vectors and singular values

  Postconditions:
    - M is not modified
    - if M is distributed, the left singular vectors have the same distributions
  '''
  assert method in ['auto', 'method_of_snapshots'], \
      "thin_svd currently supports only method = 'auto' or 'method_of_snapshots'"

  # if user wants a specific algorithm, then call it
  if method == 'method_of_snapshots':
      return _thin_svd_via_method_of_snaphosts(M, comm)

  # otherwise we have some freedom to decide
  if comm is not None and comm.Get_size() > 1:
      return _thin_svd_auto_select_algo(M, comm)
  else:
    return np.linalg.svd(M, full_matrices=False, compute_uv=True)

# ----------------------------------------------------
# ----------------------------------------------------

# Define public facing API
max = _basic_max_via_python
min = _basic_min_via_python
mean = _basic_mean_via_python
std = _basic_std_via_python
product = _basic_product_via_python
thin_svd = _thin_svd
