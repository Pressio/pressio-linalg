import numpy as np
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


#####################################################
########             MPI Helpers             ########
#####################################################

def distribute_array(global_array, comm, axis=0):
    '''
    Splts an np.array and distributes to all available MPI processes as evenly as possible

    Inputs:
        global_array: The global np.array to be distributed.
        comm: The MPI communicator
        axis: The axis along which to split the input array. By default, splits along the first axis (rows).

    Returns:
        local_array: The subset of global_array sent to the current MPI process.

    '''
    # Get comm info
    n_procs = comm.Get_size()
    rank = comm.Get_rank()

    # Handle null case
    if global_array.size == 0:
        return np.empty(0)

    # Split the global_array and send to corresponding MPI rank
    if rank == 0:
        splits = np.array_split(global_array, n_procs, axis=axis)
        for proc in range(n_procs):
            if proc == 0:
                local_array = splits[proc]
            else:
                comm.send(splits[proc], dest=proc)
    else:
        local_array = comm.recv(source=0)

    return local_array

def generate_random_local_and_global_arrays(ndim, comm, dim1=7, dim2=5, dim3=6):
    '''Randomly generates both local and global arrays using optional dim<x> arguments to specify the shape'''
    # Get comm info
    rank = comm.Get_rank()

    # Create global_array (using optional dim<x> arguments)
    if rank == 0:
        if ndim == 0:
            global_arr = np.empty(0)
        elif ndim == 1:
            global_arr = np.random.rand(dim1) if rank == 0 else np.empty(dim1)
        elif ndim == 2:
            global_arr = np.random.rand(dim1, dim2) if rank == 0 else np.empty((dim1, dim2))
        elif ndim == 3:
            global_arr = np.random.rand(dim1, dim2, dim3) if rank == 0 else np.empty((dim1, dim2, dim3))
        else:
            raise ValueError(f"This function only supports arrays up to rank 3 (received rank {ndim})")
    else:
        if ndim == 0:
            global_arr = np.empty(0)
        elif ndim == 1:
            global_arr = np.empty(dim1, dtype=float)
        elif ndim == 2:
            global_arr = np.empty((dim1, dim2), dtype=float)
        elif ndim == 3:
            global_arr = np.empty((dim1, dim2, dim3), dtype=float)
        else:
            raise ValueError(f"This function only supports arrays up to rank 3 (received rank {ndim})")

    # Broadcast global_array and create local_array
    comm.Bcast(global_arr, root=0)
    local_arr = distribute_array(global_arr, comm)

    return local_arr, global_arr

def generate_local_and_global_arrays_from_example(rank, slices, example: int):
    '''Generates both local and global arrays built from the example tensors in the documentation.
       Also returns "slices," which tells how the arrays have been distributed.'''
    # Create arrays
    if example == 1:
        global_arr = np.array([2.2, 3.3, 40., 51., -24., 45., -4.])
        local_arr = global_arr[slices[rank][0]:slices[rank][1]]

    elif example == 2:
        global_arr = np.array([[2.2, 1.3, 4.],
                               [3.3, 5.0, 33.],
                               [40., -2., -4.],
                               [51., 4., 6.],
                               [-24., 8., 9.],
                               [45., -3., -4.],
                               [-4., 8., 9.]])
        local_arr = global_arr[slices[rank][0]:slices[rank][1],:]

    elif example == 3:
        global_arr = np.array([[[2.,3.],[1.,6.],[4.,-7]],
                               [[3.,4.],[5.,-1.],[3.,5.]],
                               [[4.,2],[-2.,-2.],[-4.,5]],
                               [[5.,8.],[4.,-1.],[6.,0]],
                               [[-2.,2,],[8.,0.],[9.,3]],
                               [[4.,1],[-3.,-6.],[-4.,1]],
                               [[-4.,2.],[8.,0.],[9.,3.]]])
        local_arr = global_arr[slices[rank][0]:slices[rank][1],:,:]

    else:
        return None, None

    return local_arr, global_arr
