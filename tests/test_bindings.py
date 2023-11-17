import os

import numpy as np
import mpi4py

from pressiolinalg.linalg import *

# TESTING MYFUNC
print("Testing myfunc")
vector = np.arange(1,10)
func = myfunc(vector)

# TESTING PRINT_COMM
print("\nTesting print_comm")
comm = MPI.COMM_WORLD
comm_address = hex(MPI._addressof(comm))
ftn_address = hex(print_comm(comm))

print(f"    comm address:     {comm_address}")
print(f"    function address: {ftn_address}")

assert ftn_address == comm_address, f"{ftn_address} != {comm_address}"
