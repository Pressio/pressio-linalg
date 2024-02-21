import numpy as np

def assert_out_size_matches_expected(out, expected_size):
    '''Checks that the out parameter is the correct size for holding the operation's output.'''
    if out is not None:
        assert out.size == expected_size, f"out should have size {expected_size}"

def copy_result_to_out_if_not_none_else_return(result, out):
    '''Copies the result of an operation to the out array, if one is provided.'''
    if out is None:
        return result
    else:
        if isinstance(result, np.ndarray):
            np.copyto(out, result)
        else:
            out.fill(result)
        return
