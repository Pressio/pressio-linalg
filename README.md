[![Unit Tests](https://github.com/Pressio/pressio-linalg/actions/workflows/test.yaml/badge.svg)](https://github.com/Pressio/pressio-linalg/actions/workflows/test.yaml/badge.svg)

# pressio-linalg

## Current installation (subject to change)

This section will change as we finalize the installation process for the repository. For now, we are working on supporting parallel C++ bindings. To install and test, begin by cloning the repository:
```sh
git clone https://github.com/Pressio/pressio-linalg.git
cd pressio-linalg
```
Then, set the following environment variables:
```sh
export MPI_BASE_DIR=<path-to-MPI-install>
export MPI4PY_BASE_DIR=<path-to-mpi4py-install>
```
You can find the base directory of mpi4py by running this Python code, either on the command line or in a script:
```python
import mpi4py
print(mpi4py.get_include())
```
Note that the environment variable above is for the _base directory_, not the include directory, which is what this command will output.

Once these environment variables have been set, run
```sh
pip install .
```
to install the package.

Once the package has installed, you can test the Python/C++ bindings by running
```sh
python tests/test_bindings.py
```
This file calls the public-facing function and should print the output from the functions implemented in C++ (the source of the function is clear in the output--e.g. `C++ received MPI_COMM_WORLD`).

---

## Installation

_This section outlines the end goal--at present, these commands do not work._

Pressio-linalg offers two different modes. "Light mode" has pure Python dependencies and supports some basic functions so that a user can utilize most of the rom-tools library without headaches. "Heavy mode" is more performant and uses bindings to Trilinos.

Begin by cloning the repository:
```sh
git clone https://github.com/Pressio/pressio-linalg.git
cd pressio-linalg
```

### Light Mode

To install Pressio-linalg in Light Mode, ensure you are in the project directory and run
```sh
pip install .
```

### Heavy Mode

To install with heavy mode, we envision the following scenarios:

#### a) You already have Trilinos installed somewhere

From the project directory, run

```sh
export MPI_BASE_DIR=<full-path-to-your-MPI-installation>

python3 setup.py install --trilinos-basedir=<full-path-to-your-trilinos-installation>
```

#### b) You do NOT have Trilinos and want pressio-linalg to build it

From the project directory, run

```sh
# set MPI base
export MPI_BASE_DIR=<full-path-to-your-MPI-installation>

cd pressio-tools
python3 setup.py install
```

## Testing

Once you have installed the package, run tests with
```sh
mpirun -n <np> python -m pytest tests/* --with-mpi
```
where, `<np>` is the number of processors you would like to use.
