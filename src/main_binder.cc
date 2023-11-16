
#ifndef PRESSIOTOOLS_MAIN_BINDER_HPP_
#define PRESSIOTOOLS_MAIN_BINDER_HPP_

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "mpi.h"
#include <mpi4py/mpi4py.h>


namespace py = pybind11;

// Create a type that the C++ compiler will recognize
struct mpi4py_comm {
  mpi4py_comm() = default;
  mpi4py_comm(MPI_Comm value) : value(value) {}
  operator MPI_Comm () { return value; }

  MPI_Comm value;
};

// Define the type caster
namespace pybind11 { namespace detail {
  template <> struct type_caster<mpi4py_comm> {
    public:
      PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

      // Python -> C++
      bool load(handle src, bool) {
        PyObject *py_src = src.ptr();

        // Check that we have been passed an mpi4py communicator
        if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
          // Convert to regular MPI communicator
          value.value = *PyMPIComm_Get(py_src);
        } else {
          return false;
        }

        return !PyErr_Occurred();
      }

      // C++ -> Python
      static handle cast(mpi4py_comm src,
                         return_value_policy /* policy */,
                         handle /* parent */)
      {
        // Create an mpi4py handle
        return PyMPIComm_New(src.value);
      }
  };
}} // namespace pybind11::detail

namespace{

using scalar_t  = double;
using py_c_arr  = pybind11::array_t<scalar_t, pybind11::array::c_style>;
using py_f_arr  = pybind11::array_t<scalar_t, pybind11::array::f_style>;

// Simple function to serial bindings
void _myfunc(py_f_arr vec) {
  std::cout << "my fancy func impl in C++\n";
}

// Recieve a communicator and check if it equals MPI_COMM_WORLD
void _print_comm(mpi4py_comm comm) {
  if (comm == MPI_COMM_WORLD) {
    std::cout << "C++ received the world." << std::endl;
  } else {
    std::cout << "C++ received something else." << std::endl;
  }
}
}

PYBIND11_MODULE(MODNAME, mParent)
{
  // Initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
    throw py::error_already_set();
  }

  mParent.def("_myfunc", &_myfunc);
  mParent.def("_print_comm", &_print_comm);

  //mParent.def("max", &max);
};

#endif
