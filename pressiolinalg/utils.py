import numpy as np
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


#####################################################
########             MPI Helpers             ########
#####################################################

def distribute_vector(global_vector, comm):
    '''Distributes a global vector over all available MPI processes.'''
    n_procs = comm.Get_size()

    local_size = len(global_vector) // n_procs
    local_vector = np.zeros(local_size, dtype=int)

    comm.Scatter(global_vector, local_vector, root=0)

    return local_vector

def distribute_array(global_array, comm):
    '''Distributes a global np.array over all available MPI processes.'''
    n_procs = comm.Get_size()

    dim = len(global_array.shape)

    if dim == 2:
        rows, cols = global_array.shape
        local_rows = rows // n_procs
        local_array = np.zeros((local_rows, cols), dtype=int)

    elif dim == 3:
        rows, cols, depth = global_array.shape
        local_rows = rows // n_procs
        local_array = np.zeros((local_rows, cols, depth), dtype=int)

    else:
        return None

    comm.Scatter(global_array, local_array, root=0)

    return local_array

def get_local_and_global_arrays(rank, comm):
    '''Generates both local and global arrays for use in testing.'''
    n_procs = comm.Get_size()

    if rank == 1:
        global_arr = np.empty(n_procs * 2, dtype=np.int64)
        for i in range(n_procs * 2):
            global_arr[i] = i
        local_arr = distribute_vector(global_arr, comm)

    elif rank == 2:
        global_arr = np.empty((n_procs * 2, 3), dtype=np.int64)
        for i in range(n_procs * 2):
            for j in range(3):
                global_arr[i][j] = i+j
        local_arr = distribute_array(global_arr, comm)

    elif rank == 3:
        global_arr = np.empty((n_procs * 2, 3, 4), dtype=np.int64)
        for i in range(n_procs * 2):
            for j in range(3):
                for k in range(4):
                    global_arr[i][j][k] = i+j+k
        local_arr = distribute_array(global_arr, comm)

    else:
        raise ValueError(f"This function only supports arrays up to rank 3 (received rank {rank})")

    return local_arr, global_arr


#####################################################
########           General Helpers           ########
#####################################################

def verify_out_size(out, expected_size):
    '''Checks that the out parameter is the correct size for holding the operation's output.'''
    if out is not None:
        assert out.size == expected_size, f"out should have size {expected_size}"

def return_to_out_if_given(result, out):
    '''Copies the result of an operation to the out array, if one is provided.'''
    if out is None:
        return result
    else:
        if isinstance(result, np.ndarray):
            np.copyto(out, result)
        else:
            out.fill(result)
        return