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

def generate_local_and_global_arrays(ndim, comm, dim1=7, dim2=5, dim3=6):
    '''Generates both local and global arrays using optional dim<x> arguments to specify the shape'''
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
