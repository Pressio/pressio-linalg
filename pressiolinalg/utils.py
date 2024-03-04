import numpy as np

def assert_axis_is_none_or_within_rank(a, axis):
    assert isinstance(axis, int) or axis is None, "axis must be an int or None"
    if axis is not None:
        assert axis < a.ndim, "axis must be < rank of the array"
