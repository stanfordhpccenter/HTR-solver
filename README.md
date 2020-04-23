Organization of the repository
==============================

./  
   > [src/](src/)  
   >  > [prometeo.rg](src/prometeo.rg): Main tasks of the solver  
   >  > [prometeo_const.rg](src/prometeo_const.rg): Module that contains constants used by the solver  
   >  > [prometeo_macro.rg](src/prometeo_macro.rg): Module that contains simple macros used by the solver  
   >  > [prometeo_init.rg](src/prometeo_init.rg): Module that contains initialization tasks  
   >  > [prometeo_grid.rg](src/prometeo_grid.rg): Module that contains the tasks to generate the computational grid  
   >  > [prometeo_cfl.rg](src/prometeo_cfl.rg): Module that contains the tasks to compute the CFL number  
   >  > [prometeo_chem.rg](src/prometeo_chem.rg): Module that contains the tasks to advance chemistry  
   >  > [prometeo_bc.rg](src/prometeo_bc.rg): Module that contains the tasks that handle boundary conditions  
   >  > [prometeo_profiles.rg](src/prometeo_profiles.rg): Module that contains the tasks that handle external profiles provided to the solver  
   >  > [prometeo_rk.rg](src/prometeo_rk.rg): Module that contains the tasks for Runge-Kutta algorithm  
   >  > [prometeo_stat.rg](src/prometeo_stat.rg): Module that contains the tasks that extract integral quantities from the solution  
   >  > [prometeo_flux.rg](src/prometeo_flux.rg): Module that contains the tasks that compute the fluxes  
   >  > [prometeo_rhs.rg](src/prometeo_rhs.rg): Module that contains the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_operators.rg](src/prometeo_operators.rg): Module that contains the tasks corresponding to some differential operators  
   >  > [prometeo_variables.rg](src/prometeo_variables.rg): Module that contains the tasks that compute auxiliary variables from the unknowns and the other way around  
   >  > [prometeo_IO.rg](src/prometeo_IO.rg): Module that contains the tasks that perform I/O operations  
   >  > [prometeo_average.rg](src/prometeo_average.rg): Module that contains the tasks that perform 2D averages of the solution  
   >  > [prometeo_metric.rg](src/prometeo_metric.rg): Module that contains the tasks that compute the metric of the solver  
   >  > [prometeo_mapper.h](src/prometeo_mapper.h) [prometeo_mapper.cc](src/prometeo_mapper.cc): Source files for the mapper of the solver  
   >  > [desugar.py](src/desugar.py): Script that substitutes some macros used for metaprogramming  
   >  > [Makefile](src/Makefile): Builds the solver  
   >  > [prometeo.sh](src/prometeo.sh): Script to run a calculation  
   >  > [run.sh](src/run.sh): Script called by [prometeo.sh](src/prometeo.sh) to generate the execution command (modify this script using the provided templates to add a new machine)  
   >  > [jobscript_shared.sh](src/jobscript_shared.sh): Script called by [run.sh](src/run.sh)  
   >  > [json.h](src/json.h) [json.c](src/json.c): Source files to interpret `*.json` files  
   >  > [Reaction.rg](src/Reaction.rg): Tasks and data structures related to chemical reactions  
   >  > [Species.rg](src/Species.rg): Tasks and data structures related to chemical species  
   >  > [ConstPropMix.rg](src/ConstPropMix.rg): Tasks and data structures describing a calorically perfect gas  
   >  > [AirMix.rg](src/AirMix.rg): Tasks and data structures describing a non-equilibrium dissociating air  
   >  > [config_schema.lua](src/config_schema.lua): Lua file descibing the fields of the input file  
   >  > [process_schema.rg](src/process_schema.rg): Interpreter for [config_schema.lua](src/config_schema.lua)  
   >  > [util.rg ](src/util.rg): Various Regent tasks and Lua functions deployed throughout the solver  
   >  > [math_utils.rg](src/math_utils.rg): Basic mathematical operations deployed in the solver  
   >  > [hdf_helper.rg](src/hdf_helper.rg): Scripts to read and write HDF5 files  
   >  > [certainty.slurm](src/certainty.slurm): Submission script for Certainty (@ Stanford) (use as a template script for slurm system)  
   >  > [quartz.slurm](src/quartz.slurm): Submission script for Quartz (@ LLNL) (use as a template script for slurm system)  
   >  > [lassen.lsf](src/lassen.lsf): Submission script for Lassen (@ LLNL) (use as a template script for IBM Spectrum LSF system)  
   >  > [blacklist](src/blacklist): Folder containing potential blacklists of nodes that should not be used

   > [scripts/](scripts/)  
   >  > [viz_fluid.py](scripts/viz_fluid.py): Script to produce Xdmf files for visualiztion  
   >  > [compare_hdf.py](scripts/compare_hdf.py): Script to compare two HDF5 files  
   >  > [merge.py](scripts/merge.py): Script to merge multiple HDF5 files into a single file  
   >  > [makeVirtualLayout.py](scripts/makeVirtualLayout.py): Script to merge multiple HDF5 files into a single file using a virtual layout  
   >  > [interpolate.py](scripts/interpolate.py): Script to interpolate a solution on a new grid  
   >  > [convert_output_for_viz.py](scripts/convert_output_for_viz.py): Script that automates the production of visualization files for multiple snapshots  
   >  > [modules/](scripts/modules/): Various utility modules used by the python scripts

   > [testcases/](testcases/)  
   >  > [README.md](testcases/README.md): Instructions on how to run the provided testcases
   >  > [SodProblem/](testcases/SodProblem/): Contains the setup and postporcessing files needed to run Sod's shock tube  
   >  > [LaxProblem/](testcases/LaxProblem/): Contains the setup and postporcessing files needed to run Lax's shock tube  
   >  > [ShuOsherProblem/](testcases/ShuOsherProblem/): Contains the setup and postporcessing files needed to run Shu-Osher's shock tube  
   >  > [GrossmanCinnellaProblem/](testcases/GrossmanCinnellaProblem/): Contains the setup and postporcessing files needed to run Grossman-Cinnella's shock tube  
   >  > [Blasius/](testcases/Blasius/): Contains the setup and postporcessing files needed to run an incompressible boundary layer  
   >  > [CompressibleBL/](testcases/CompressibleBL/): Contains the setup and postporcessing files needed to run a compressible boundary layer  
   >  > [VortexAdvection2D/](testcases/VortexAdvection2D/): Contains the setup and postporcessing files needed to run the inviscid vortex advection  
   >  > [TaylorGreen2D/](testcases/TaylorGreen2D/): Contains the setup and postporcessing files needed to run the 2D Taylor-Green vortex  
   >  > [Coleman/](testcases/Coleman/): Contains the setup and postporcessing files needed to run Coleman's channel flow  
   >  > [Sciacovelli/](testcases/Sciacovelli/): Contains the setup and postporcessing files needed to run Sciacovelli's channel flow  
   >  > [Franko/](testcases/Franko/): Contains the setup and postporcessing files needed to run Franko's boundary layer  
   >  > [MultispeciesTBL/](testcases/MultispeciesTBL/): Contains the setup and postporcessing files needed to run Multispecies hypersonic boundary layer  
   >  > [scalingTest/WS](testcases/scalingTest/WS): Contains the setup and postporcessing files needed to run the weak saling test


Setup (generic)
===============

See below for instructions targeting specific systems.

### Prerequisites

* Legion (latest version)
* GCC 4.9+ (we need a working `std::regex` library)
* CUDA 7.5+
* Python 2.X

The following are automatically installed during Legion installation:

* LLVM 3.8 (for CUDA 8.0+) or 3.5 (for CUDA 7.5, and better debug info)
* GASNET (custom version)
* Terra (custom version -- we need to use LuaJIT2.1 instead of the default LuaJIT2.0, because the latter exhibits a spurious out-of-memory error when compiling large Regent programs)
* HDF5 (any recent version)

### Add to shell startup

Normally you'd need to edit file `~/.bashrc`. Replace the `???` depending on your system.

```
# Module loads (if necessary)
...
# Build config (if necessary, for Legion or Prometeo)
...
# Path setup (mandatory)
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
[export SCRATCH=???]
# CUDA config (if using CUDA code generation)
export CUDA_HOME=???
export CUDA="$CUDA_HOME"
export GPU_ARCH=???
```

### Download software

```
git clone -b htr-release https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

Replace the `?` depending on your system's capabilities.

```
cd "$LEGION_DIR"/language
USE_CUDA=? USE_OPENMP=? USE_GASNET=? USE_HDF=? MAX_DIM=4 scripts/setup_env.py --llvm-version 38 --terra-url 'https://github.com/StanfordLegion/terra.git' --terra-branch 'luajit2.1'
```

See [Elliott's instructions](https://docs.google.com/document/d/1Qkl6r-1ZIb8WyH1f_WZbKgjp3due_Q8UiWKLh_nG1ec/edit) for more help.

### Compile the HTR solver

NOTE: This step may take up to 2 hrs depending on the system

```
cd "$HTR_DIR"/src
[USE_CUDA=0] [USE_HDF=0] make
```

Running
=======

```
cd "$HTR_DIR"/src
./prometeo.sh ...
```

The [src/prometeo.sh](src/prometeo.sh) script accepts some options through the environment (see the top of that file for details), and forwards all command-line arguments to the HTR solver executable and the Legion runtime (each will ignore options it doesn't recognize).

Currently, the solver reads the following options:

* `-i <config>.json`: Provide a case configuration file, to be run as an additional sample. See [src/config_schema.lua](src/config_schema.lua) for documentation on the available options (`Config` struct).
* `-o <out_dir>`: Specify an output directory for the executable (default is current directory).

Setup (local Ubuntu machine w/o GPU)
====================================

### Add to shell startup

```
# Build config
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
```

### Download software

```
git clone -b htr-release https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
USE_CUDA=0 USE_OPENMP=1 USE_GASNET=0 USE_HDF=1 MAX_DIM=4 scripts/setup_env.py --llvm-version 38 --terra-url 'https://github.com/StanfordLegion/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
USE_CUDA=0 make
```

Setup (Sapling @ Stanford)
==========================

### Add to shell startup

```
# Module loads
module load mpi/openmpi/1.8.2
module load cuda/7.0
# Build config
export CONDUIT=ibv
export CC=gcc-4.9
export CXX=g++-4.9
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
export SCRATCH=/scratch/oldhome/`whoami`
# CUDA config
export CUDA_HOME=/usr/local/cuda-7.0
export CUDA="$CUDA_HOME"
export GPU_ARCH=fermi
```

### Download software

```
git clone -b htr-release https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
USE_CUDA=1 USE_OPENMP=1 USE_GASNET=1 USE_HDF=1 MAX_DIM=4 scripts/setup_env.py --llvm-version 35 --terra-url 'https://github.com/StanfordLegion/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make
```

Setup (Certainty @ Stanford)
============================

### Add to shell startup

```
# Module loads
module load gnu7/7.2.0
module load cuda/8.0
module load openmpi3/3.0.0
# Build config
export CONDUIT=ibv
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
# CUDA config
export CUDA_HOME=/usr/local/cuda-8.0
export CUDA="$CUDA_HOME"
export GPU_ARCH=fermi
```

### Download software

```
git clone -b htr-release https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
# Disable PMI in GASnet, because the PMI library is missing on Certainty.
git clone https://github.com/StanfordLegion/gasnet.git $LEGION_DIR/language/gasnet
cd "$LEGION_DIR"/language/gasnet
sed -i 's|$(GASNET_VERSION)/configure --prefix=|$(GASNET_VERSION)/configure --disable-pmi --prefix=|g' Makefile
make
# Rest of compilation as normal
cd "$LEGION_DIR"/language
USE_CUDA=1 USE_OPENMP=1 USE_GASNET=1 USE_HDF=1 MAX_DIM=4 scripts/setup_env.py --llvm-version 38 --terra-url 'https://github.com/StanfordLegion/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make
```

Setup (Quartz @ LLNL)
============================

### Add to shell startup

```
# Module loads
module load gcc/4.9.3
module load openmpi/2.0.0
module load python
module load paraview/5.4
# Build config
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
export SCRATCH=???
```

### Download software

```
git clone -b htr-release https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
USE_CUDA=0 USE_OPENMP=1 USE_GASNET=1 USE_HDF=1 MAX_DIM=4 scripts/setup_env.py --llvm-version 38 --terra-url 'https://github.com/StanfordLegion/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make
```

Setup (Lassen @ LLNL)
============================

### Add to shell startup

```
# Module loads
module load gcc/7.3.1
module load cuda/9.2.148
module load python
# Build config
export CC=gcc
export CXX=g++
export CONDUIT=ibv
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
export SCRATCH=???
# CUDA config
export CUDA_HOME=/usr/tce/packages/cuda/cuda-9.2.148
export CUDA="$CUDA_HOME"
export GPU_ARCH=volta

export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=4 
export TERRA_USE_PUC_LUA=1

```

### Download software

```
git clone -b htr-release https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
lalloc 1 scripts/setup_env.py --llvm-version 38 --terra-branch 'puc_lua_master'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
lalloc 1 -W 120 make
```
