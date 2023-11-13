
#ifndef PRESSIOTOOLS_MAIN_BINDER_HPP_
#define PRESSIOTOOLS_MAIN_BINDER_HPP_

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include "mpi.h"

namespace{

using scalar_t  = double;
using py_c_arr  = pybind11::array_t<scalar_t, pybind11::array::c_style>;
using py_f_arr  = pybind11::array_t<scalar_t, pybind11::array::f_style>;

void myfunc(py_f_arr vec){
  std::cout << "my fancy func impl in C++\n";
}

// void max(py_f_arr vec, ...mpicomm){
//   // compute max
// }
}

PYBIND11_MODULE(MODNAME, mParent)
{
  mParent.def("myfunc", &myfunc);
  //mParent.def("max", &max);
};

#endif
