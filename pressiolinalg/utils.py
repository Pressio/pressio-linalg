import numpy as np

def assert_axis_is_correct_type_and_within_range(a, axis):
    if axis is not None:
        if isinstance(axis, int):
            assert axis <= a.ndim
        else:
            raise ValueError("axis must be an int")
