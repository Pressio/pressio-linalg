import numpy as np

#####################################################
########           General Helpers           ########
#####################################################

def calculate_shape_of_result(global_array, axis=None):
    '''Calculates the output shape of an arbitrary operation.'''
    if axis is None:
        return 1
    elif isinstance(axis, int):
        shape_list = []
        for d in range(global_array.ndim):
            if d != axis:
                shape_list.append(global_array.shape[d])
    elif isinstance(axis, tuple):
        shape_list = []
        for d in range(global_array.ndim):
            if d not in axis:
                shape_list.append(global_array.shape[d])
    else:
        raise ValueError("axis must be either an integer or a tuple.")

    return tuple(shape_list)

def get_global_shape_from_local_array(a, comm):
    import mpi4py
    from mpi4py import MPI

    global_shape_list = []
    for dim in range(a.ndim):
        global_dim = comm.allreduce(a.shape[dim], op=MPI.SUM)
        global_shape_list.append(global_dim)
    global_shape = tuple(global_shape_list)
    print("global shape: ", global_shape)
    return global_shape

def assert_axis_is_correct_type_and_within_range(a, axis):
    if axis is not None:
        if isinstance(axis, int):
            assert axis <= a.ndim
        elif isinstance(axis, tuple):
            for ax in axis:
                assert ax <= a.ndim
        else:
            raise ValueError("axis must be either an int or tuple")

def assert_out_size_matches_expected(result, out):
    '''Checks that the out parameter has the correct shape for holding the operation's output.'''
    if out is not None:
        if isinstance(result, np.ndarray):
            assert out.shape == result.shape, f"out should have shape {result.shape} (received {out.shape})"

def copy_result_to_out_if_not_none_else_return(result, out):
    '''Copies the result of an operation to the out array, if one is provided.'''
    if out is None:
        return result
    else:
        assert_out_size_matches_expected(result, out)
        if isinstance(result, np.ndarray):
            np.copyto(out, result)
        else:
            out.fill(result)
        return
