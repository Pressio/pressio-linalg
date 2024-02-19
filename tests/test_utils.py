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
    '''Distributes a global vector such that all available MPI processes receive the same number of elements.'''
    n_procs = comm.Get_size()

    local_size = len(global_vector) // n_procs
    local_vector = np.zeros(local_size, dtype=int)

    comm.Scatter(global_vector, local_vector, root=0)

    return local_vector

def distribute_array(global_array, comm):
    '''Distributes a global np.array such that all available MPI processes receive the same number of rows.'''
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

def generate_local_and_global_arrays(dim, comm):
    '''Generates both local and global arrays (for use in testing).'''
    n_procs = comm.Get_size()

    if dim == 0:
      global_arr = np.empty(0)
      local_arr = distribute_vector(global_arr, comm)

    elif dim == 1:
        global_arr = np.empty(n_procs * 2, dtype=np.int64)
        for i in range(n_procs * 2):
            global_arr[i] = i
        local_arr = distribute_vector(global_arr, comm)

    elif dim == 2:
        global_arr = np.empty((n_procs * 2, 3), dtype=np.int64)
        for i in range(n_procs * 2):
            for j in range(3):
                global_arr[i][j] = i+j
        local_arr = distribute_array(global_arr, comm)

    elif dim == 3:
        global_arr = np.empty((n_procs * 2, 3, 4), dtype=np.int64)
        for i in range(n_procs * 2):
            for j in range(3):
                for k in range(4):
                    global_arr[i][j][k] = i+j+k
        local_arr = distribute_array(global_arr, comm)

    else:
        raise ValueError(f"This function only supports arrays up to rank 3 (received rank {dim})")

    return local_arr, global_arr