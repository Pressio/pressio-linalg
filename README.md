[![Unit Tests](https://github.com/Pressio/pressio-linalg/actions/workflows/test.yaml/badge.svg)](https://github.com/Pressio/pressio-linalg/actions/workflows/test.yaml/badge.svg)

# pressio-linalg

Install with
```
pip install pressio-linalg
```

This installs the library such that all kernels are implemented with pure python calls and calls to MPI. 
Therefore, all the communication is handled explicitly internally and we do not rely on any external backend.


<!-- ---

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
where, `<np>` is the number of processors you would like to use. -->
