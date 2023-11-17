
#ifndef PRESSIOTOOLS_MAIN_BINDER_HPP_
#define PRESSIOTOOLS_MAIN_BINDER_HPP_

#include <iostream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mpi.h"
#include <mpi4py/mpi4py.h>


namespace py = pybind11;
namespace{

using scalar_t  = double;
using py_c_arr  = pybind11::array_t<scalar_t, pybind11::array::c_style>;
using py_f_arr  = pybind11::array_t<scalar_t, pybind11::array::f_style>;

// Simple function to serial bindings
std::string _myfunc(py_f_arr vec) {
  std::string status = "Using C++ bindings";
  std::cout << status << std::endl;
  return status;
}

MPI_Comm* get_mpi_comm(py::object py_comm) {
  auto comm_ptr = PyMPIComm_Get(py_comm.ptr());

  if (!comm_ptr)
    throw py::error_already_set();

  return comm_ptr;
}

// Recieve a communicator, print some attributes of it, and return the memory address (to compare to Python)
uintptr_t _print_comm(MPI_Comm* comm_ptr) {
  MPI_Comm& comm = *comm_ptr;
  int size = 0;
  MPI_Comm_size(comm, &size);

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  uintptr_t int_comm_ptr = reinterpret_cast<uintptr_t>(comm_ptr);

  return int_comm_ptr;
}
}

PYBIND11_MODULE(MODNAME, mParent)
{
  // Initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
    throw py::error_already_set();
  }

  mParent.def("_myfunc", &_myfunc);
  mParent.def("_print_comm",
              [](py::object py_comm) {
                auto comm_ptr = get_mpi_comm(py_comm);
                return _print_comm(comm_ptr);;
              });
};

#endif
