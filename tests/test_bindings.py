import os

import numpy as np
import mpi4py

from pressiolinalg.linalg import *

python_only = True
cpp_bindings = False

if os.environ.get('PRESSIO-LINALG-CPP'):
    cpp_bindings = True
    python_only = False

# TESTING MYFUNC (makes sure install process worked)
def test_myfunc():
    vector = np.arange(1,10)
    func = myfunc(vector)
    if python_only:
        assert func == "Using only Python"
    elif cpp_bindings:
        assert func == "Using C++ bindings"

# TESTING PRINT_COMM
def test_print_comm():
    comm = MPI.COMM_WORLD
    comm_address = hex(MPI._addressof(comm))
    ftn_address = hex(print_comm(comm))

    print(f"    comm address:     {comm_address}")
    print(f"    function address: {ftn_address}")

    assert ftn_address == comm_address, f"{ftn_address} != {comm_address}"

if __name__ == "__main__":
    test_myfunc()
    test_print_comm()
