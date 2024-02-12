pressio-linalg: Parallel Linear Algebra
======================================

This Python library offers basic linear algebra functions that can be implemented in parallel.

Install
-------

pressio-linalg is tested on Python 3.8-3.11.

To install, use the following command:

.. code-block:: bash

  pip install pressio-linalg

With this installation, all kernels are implemented with pure Python calls and calls to MPI. This ensures that all the communication is handled explicitly internally and we do not rely on any external backend.