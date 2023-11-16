import numpy as np
import mpi4py

from pressiolinalg.linalg import *

vector = np.arange(1,10)
func = myfunc(vector)

comm = MPI.COMM_WORLD
output = print_comm(comm)
assert(output == "C++ received the world.")
