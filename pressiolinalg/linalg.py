
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

import numpy as np

def _basic_func_via_python(vec):
  print("myfunc purely python")

# def _basic_max_via_python(vec, mpiComm):
#   print("_basic_max_via_python")

##############
### import ###
##############
try:
  from ._linalg import myfunc
except ImportError:
  myfunc = _basic_func_via_python
  #max = _basic_max_via_python
