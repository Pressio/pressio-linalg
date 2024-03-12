
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import warnings
import numpy as np
from pressiolinalg import utils

# ----------------------------------------------------

def _basic_max_via_python(a: np.ndarray, axis=None, comm=None):
    '''
    Return the maximum of a possibly distributed array or maximum along an axis.

    Parameters:
        a (np.ndarray): input data
        axis (None or int): the axis along which to compute the maximum. If None, computes the max of the flattened array. (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        if axis==None, returns a scalar
        if axis is not None, returns an array of dimension a.dim - 1

    Preconditions:
      - a is at most a rank-3 tensor and
      - if a is distributed, it must be so along the 0-th axis,
        and every rank must have the same a.shape[1] and a.shape[2]
      - if axis != None, then it must be an int

    Postconditions:
      - a and comm are not modified

    Example 1:
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = pla.max(a, comm)
    then ALL ranks will contain res = 51.

    Example 2:
    **********

       rank 0  2.2  1.3  4.
               3.3  5.0  33.
      =======================
       rank 1  40.  -2.  -4.
               51.   4.   6.
               -24.  8.   9.
               45.  -3.  -4.
      =======================
       rank 2  -4.  8.   9.

    Suppose that we do:

       res = pla.max(a, axis=0, comm)

    then every rank will contain the same res which is an array = ([51., 8., 33])
    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = pla.max(a, axis=1, comm)

    then res is now a rank-1 array as follows

       rank 0  4.
               33.
      =======================
       rank 1  40.
               51.
               9.
               45.
      =======================
       rank 2  9.

    because the axis queried for the max is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.


    Example 3:
    **********

                / 3.  6. -7.
       rank 0  /2.  1.  4.
               --------------

                / 4.  -1.  5.
               /3.  5.  3.
      =======================

       rank 1   / 2.  -2.  5.
               /4.  -2.  -4.
               --------------

                / 8.  -1.  0.
               /5.   4.   6.
               --------------

                / 2.   0.   3.
               /-2.   8.   9.
               --------------

                / 1.  -6.  1.
               /4.  -3.  -4.
      =======================

       rank 2   / 2.   0.  3.
               /-4.  8.   9.
               --------------

    Suppose that we do:

       res = pla.max(a, axis=0, comm)

    then this is effectively a reduction over the 0-th axis
    and every rank will contain the same res which is a rank-2 array as follows

          ([[5. 8.]
            [8. 6.]
            [9. 5.]])

    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = pla.max(a, axis=1, comm)

    then res is now a rank-2 array as follows

       rank 0  4, 6
               5, 5
      =======================
       rank 1  4, 5
               6, 8
               9, 3
               4, 1
      =======================
       rank 2  9, 3

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 1-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = pla.max(a, axis=2, comm)

    then res is now a rank-2 array as follows

       rank 0  3, 6, 4
               4, 5, 5
      =======================
       rank 1  4, -2, 5
               8, 4, 6
               2, 8, 9
               4, -3, 1
      =======================
       rank 2  2, 8, 9

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 1-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    '''
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    utils.assert_axis_is_none_or_within_rank(a, axis)

    if comm is None or comm.Get_size() == 1:
        return np.max(a, axis=axis)

    else:

        import mpi4py
        from mpi4py import MPI

        local_max = np.max(a, axis=axis)

        if axis is None:
            return comm.allreduce(local_max, op=MPI.MAX)
        elif axis==0:
            if a.ndim == 1:
                local_max = a
            global_max = np.zeros_like(local_max, dtype=local_max.dtype)
            comm.Allreduce(local_max, global_max, op=MPI.MAX)
            return global_max
        else:
            return local_max


# # ----------------------------------------------------
def _basic_min_via_python(a: np.ndarray, axis=None, comm=None):
    '''
    Return the minimum of a possibly distributed array or minimum along an axis.

    Parameters:
        a (np.ndarray): input data
        axis (None or int): the axis along which to compute the minimum. If None, computes the min of the flattened array. (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        if axis==None, returns a scalar
        if axis is not None, returns an array of dimension a.dim - 1

    Preconditions:
      - a is at most a rank-3 tensor and
      - if a is distributed, it must be so along the 0-th axis,
        and every rank must have the same a.shape[1] and a.shape[2]
      - if axis != None, then it must be an int

    Postconditions:
      - a and comm are not modified

    Example 1:
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = pla.min(a, comm)
    then ALL ranks will contain res = -4.

    Example 2:
    **********

       rank 0  2.2  1.3  4.
               3.3  5.0  33.
      =======================
       rank 1  40.  -2.  -4.
               51.   4.   6.
               -24.  8.   9.
               45.  -3.  -4.
      =======================
       rank 2  -4.  8.   9.

    Suppose that we do:

       res = pla.min(a, axis=0, comm)

    then every rank will contain the same res which is an array = ([-24., -3., -4])
    this is because the min is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = pla.min(a, axis=1, comm)

    then res is now a rank-1 array as follows

       rank 0  1.3
               3.3
      =======================
       rank 1  -4.
               4.
               -24.
               -4.
      =======================
       rank 2  -4.

    because the axis queried for the min is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.


    Example 3:
    **********

                / 3.  6. -7.
       rank 0  /2.  1.  4.
               --------------

                / 4.  -1.  5.
               /3.  5.  3.
      =======================

       rank 1   / 2.  -2.  5.
               /4.  -2.  -4.
               --------------

                / 8.  -1.  0.
               /5.   4.   6.
               --------------

                / 2.   0.  3.
               /-2.   8.   9.
               --------------

                / 1.  -6.  1.
               /4.  -3.  -4.
      =======================

       rank 2   / 2.   0.  3.
               /-4.  8.   9.
               --------------

    Suppose that we do:

       res = pla.min(a, axis=0, comm)

    then this is effectively a reduction over the 0-th axis
    and every rank will contain the same res which is a rank-2 array as follows

          ([[8., 6., 5.],
            [5, 8., 9.]])

    this is because the min is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = pla.min(a, axis=1, comm)

    then res is now a rank-2 array as follows

       rank 0  4, 6
               5, 5
      =======================
       rank 1  4, 5
               6, 8
               9, 3
               4, 1
      =======================
       rank 2  9, 3

    because the axis queried for the min is NOT a distributed axis
    and this is effectively a reduction over the 1-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = pla.min(a, axis=2, comm)

    then res is now a rank-2 array as follows

       rank 0  3, 6, 4
               4, 5, 5
      =======================
       rank 1  4, -2, 5
               8, 4, 6
               2, 8, 9
               4, -3, -4
      =======================
       rank 2  2, 8, 9

    because the axis queried for the min is NOT a distributed axis
    and this is effectively a reduction over the 1-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    '''
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    utils.assert_axis_is_none_or_within_rank(a, axis)

    if comm is None or comm.Get_size() == 1:
        return np.min(a, axis=axis)
    else:
        import mpi4py
        from mpi4py import MPI

        local_min = np.min(a, axis=axis)

        if axis is None:
            return comm.allreduce(local_min, op=MPI.MIN)
        elif axis==0:
            if a.ndim == 1:
                local_min = a
            global_min = np.zeros_like(local_min, dtype=local_min.dtype)
            comm.Allreduce(local_min, global_min, op=MPI.MIN)
            return global_min
        else:
            return local_min


# # ----------------------------------------------------
def _basic_mean_via_python(a: np.ndarray, dtype=None, axis=None, comm=None):
    '''
    Return the mean of a possibly distributed array over a given axis.

    Parameters:
        a (np.ndarray): input data
        dtype (data-type): Type to use in computing the mean (default: float64 for int arrays, same type as input for float arrays)
        axis (None or int): the axis along which to compute the mean. If None, computes the mean of the flattened array. (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        if axis==None, returns a scalar
        if axis is not None, returns an array of dimension a.dim - 1

    Preconditions:
      - a is at most a rank-3 tensor
      - if a is distributed:
          - 2-D arrays must be distributed along axis=0,
              and every rank must have the same a.shape[1] and a.shape[2]
          - 3-D tensors msut be distributed along axis=1,
              and every rank must have the same a.shape[0] and a.shape[2]
      - if axis != None, then it must be an int

    Postconditions:
      - a and comm are not modified

    Example 1:
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = pla.mean(a, comm)
    then ALL ranks will contain res = 16.21


    Example 2:
    **********

       rank 0  2.2  1.3  4.
               3.3  5.0  33.
      =======================
       rank 1  40.  -2.  -4.
               51.   4.   6.
               -24.  8.   9.
               45.  -3.  -4.
      =======================
       rank 2  -4.  8.   9.

    Suppose that we do:

       res = pla.mean(a, axis=0, comm)

    then every rank will contain the same res which is:

       res  = ([16.21,  3.04,  7.57])

    this is because the mean is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = pla.mean(a, axis=1, comm)

    then res is now a rank-1 array as follows

       rank 0  2.5
               13.77
      =======================
       rank 1  11.33
               20.33
               -2.33
               12.67
      =======================
       rank 2  4.33

    because the axis queried for the mean is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.


    Example 3:
    **********

       / 3.   4.   /  2.   8.   2.   1.   / 2.
      /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
     /  -7.   5. /    5.   0.   3.   1. /   3.
    |-----------|----------------------|--------
    | 2.   3.   |  4.   5.  -2.   4.   | -4.
    | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
    | 4.   3.   | -4.   6.   9.  -4.   |  9.

        r0                r1              r2

    Suppose that we do:

        res = pla.mean(a, axis=0, comm)

    then res is now a rank-2 array as follows:

        r0      2.33333333  0.66666667
                3.66666667  2.66666667
        ======================================
                -0.66666667 1.66666667
        r1      5.          2.33333333
                5.          1.66666667
                -1.         -1.33333333
        ======================================
        r2      4.33333333  1.66666667


    because the axis queried for the mean is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = pla.mean(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

              1.71428571  3.1428571
              3.         -0.5714285
              3.28571429  1.4285714

    this is because the mean is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = pla.mean(a, axis=2, comm)

    then res is now a rank-2 array as follows

           r0      ||          r1           ||  r2
                   ||                       ||
         2.5  3.5  ||   3.   6.5  0.   2.5  || -1.
         3.5  2.   ||  -2.   1.5  4.  -4.5  ||  4.
        -1.5  4.   ||   0.5  3.   6.  -1.5  ||  6.
                   ||                       ||

    because the axis queried for the mean is NOT a distributed axis
    and this is effectively a reduction over the 2-th axis
    so this operation is purely local and the result has the same distribution
    as the original array

    '''
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    utils.assert_axis_is_none_or_within_rank(a, axis)

    if comm is None or comm.Get_size() == 1:
        return np.mean(a, dtype=dtype, axis=axis)

    else:

        import mpi4py
        from mpi4py import MPI

        local_size = a.size if axis is None else a.shape[axis]
        global_size = comm.allreduce(local_size, op=MPI.SUM)

        distributed_axis = 0 if a.ndim < 3 else 1

        if global_size == 0:
            warnings.warn("Invalid value encountered in scalar divide (global_size = 0)")
            return np.nan

        if axis is None:
            local_sum = np.sum(a)
            global_sum = comm.allreduce(local_sum, op=MPI.SUM)
            return global_sum / global_size

        elif axis == distributed_axis:
            local_sum = np.sum(a, axis=axis)
            global_sum = np.zeros_like(np.mean(a, axis=axis))
            comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
            return global_sum / global_size

        else:
            return np.mean(a, dtype=dtype, axis=axis)

# ----------------------------------------------------
def _basic_std_via_python(a: np.ndarray, dtype=None, axis=None, ddof=0, comm=None):
    '''
    Return the standard deviation of a possibly distributed array over a given axis.

    Parameters:
        a (np.ndarray): input data
        dtype (data-type): Type to use in computing the mean (default: float64 for int arrays, same type as input for float arrays)
        axis (None or int): the axis along which to compute the mean. If None, computes the mean of the flattened array. (default: None)
        ddof (int): Delta degrees of freedom used in divisor N - ddof (default: 0)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        if axis==None, returns a scalar
        if axis is not None, returns an array of dimension a.dim - 1

    Preconditions:
      - a is at most a rank-3 tensor and
      - if a is distributed, it must be so along the 0-th axis,
        and every rank must have the same a.shape[1] and a.shape[2]
      - if axis != None, then it must be an int

    Postconditions:
      - a and comm are not modified

    Example 1:
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = pla.std(a, comm)
    then ALL ranks will contain res = 26.71

    Example 2:
    **********

       rank 0  2.2  1.3  4.
               3.3  5.0  33.
      =======================
       rank 1  40.  -2.  -4.
               51.   4.   6.
               -24.  8.   9.
               45.  -3.  -4.
      =======================
       rank 2  -4.  8.   9.

    Suppose that we do:

       res = pla.std(a, axis=0, comm)

    then every rank will contain the same res which is:

       res  = ([26.71,  4.12 , 11.55])

    this is because the standard deviation is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = pla.std(a, axis=1, comm)

    then res is now a rank-1 array as follows

       rank 0  1.12
               13.62
      =======================
       rank 1  20.29
               21.70
               15.33
               22.87
      =======================
       rank 2  5.91

    because the axis queried for the standard deviation is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Example 3:
    **********

                / 3.  6. -7.
       rank 0  /2.  1.  4.
               --------------

                / 4.  -1.  5.
               /3.  5.  3.
      =======================

       rank 1   / 2.  -2.  5.
               /4.  -2.  -4.
               --------------

                / 8.  -1.  0.
               /5.   4.   6.
               --------------

                / 2.   0.  3.
               /-2.   8.   9.
               --------------

                / 1.  -6.  1.
               /4.  -3.  -4.
      =======================

       rank 2   / 2.   0.  3.
               /-4.  8.   9.
               --------------

    Suppose that we do:

       res = pla.std(a, axis=0, comm)

    then this is effectively a reduction over the 0-th axis
    and every rank will contain the same res which is a rank-2 array as follows

      ([[3.14934396, 2.16653584],
       [4.14039336, 3.28881841],
       [5.06287004, 3.84919817]])

    this is because the standard deviation is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = pla.std(a, axis=1, comm)

    then res is now a rank-2 array as follows

       rank 0   1.24721913, 5.55777733
                0.94280904, 2.62466929
      ===================================
       rank 1   3.39934634, 2.86744176
                0.81649658, 4.02768199
                4.96655481, 1.24721913
                3.55902608, 3.29983165
      ===================================
       rank 2   5.90668172, 1.24721913

    because the axis queried for the standard deviation is NOT a distributed axis
    and this is effectively a reduction over the 1-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = pla.std(a, axis=2, comm)

    then res is now a rank-2 array as follows

       rank 0   0.5, 2.5, 5.5
                0.5, 3. , 1.
      ===========================
       rank 1   1. , 0. , 4.5
                1.5, 2.5, 3.
                2. , 4. , 3.
                1.5, 1.5, 2.5
      ===========================
       rank 2   3. , 4. , 3.

    because the axis queried for the standard deviation is NOT a distributed axis
    and this is effectively a reduction over the 1-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.
    '''
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    utils.assert_axis_is_none_or_within_rank(a, axis)

    if comm is None or comm.Get_size() == 1:
        return np.std(a, dtype=dtype, axis=axis, ddof=ddof)
    else:
        import mpi4py
        from mpi4py import MPI

        # Calculate quanities shared among axis=None or axis=0
        if axis is None or axis == 0:
            global_mean = _basic_mean_via_python(a, dtype=dtype, axis=axis, comm=comm)
            local_sq_diff = np.sum(np.square(a - global_mean), axis=axis)
            local_size = a.size if axis is None else a.shape[0]
            global_size = comm.allreduce(local_size, op=MPI.SUM)

        if axis is None:
            global_sq_diff = comm.allreduce(local_sq_diff, op=MPI.SUM)
            global_std_dev = np.sqrt(global_sq_diff / (global_size - ddof))
            return global_std_dev

        elif axis == 0:
            global_sq_diff = np.zeros_like(local_sq_diff)
            comm.Allreduce(local_sq_diff, global_sq_diff, op=MPI.SUM)
            global_std_dev = np.sqrt(global_sq_diff / (global_size - ddof))
            return global_std_dev

        else:
            return np.std(a, dtype=dtype, axis=axis, ddof=ddof)

# ----------------------------------------------------
def _basic_product_via_python(flagA, flagB, alpha, A, B, beta, C, comm=None):
    '''
    Computes C = beta*C + alpha*op(A)*op(B), where A and B are row-distributed matrices.

    Parameters:
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
