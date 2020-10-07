Organization of the repository
==============================

./  
   > [prometeo.sh](prometeo.sh): Script to run a calculation  

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
   >  > [prometeo_sensor.rg](src/prometeo_sensor.rg): Module that contains the tasks that update the shock sensors  
   >  > [prometeo_sensor.h](src/prometeo_sensor.h): C headers for the tasks that update the shock sensors  
   >  > [prometeo_sensor.hpp](src/prometeo_sensor.hpp): C++ headers for the tasks that update the shock sensors  
   >  > [prometeo_sensor.inl](src/prometeo_sensor.inl): C++ implementations of the inlined tasks that update the shock sensors  
   >  > [prometeo_sensor.cc](src/prometeo_sensor.cc): C++ implementations of the tasks that update the shock sensors  
   >  > [prometeo_sensor.cu](src/prometeo_sensor.cu): CUDA kernels of the tasks that update the shock sensors  
   >  > [prometeo_rhs.rg](src/prometeo_rhs.rg): Module that contains the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_rhs.h](src/prometeo_rhs.h): C header for the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_rhs.hpp](src/prometeo_rhs.hpp): C++ headers for the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_rhs.inl](src/prometeo_rhs.inl): C++ implementations of the inlined tasks that operate on the right-hand side of the equations  
   >  > [prometeo_rhs.cc](src/prometeo_rhs.cc): C++ implementations of the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_rhs.cu](src/prometeo_rhs.cu): CUDA kernels of the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_partitioner.rg](src/prometeo_partitioner.rg): Module that contains the tasks to perform the partitioning of the fluid region  
   >  > [prometeo_operators.rg](src/prometeo_operators.rg): Module that contains the tasks corresponding to some differential operators  
   >  > [prometeo_variables.rg](src/prometeo_variables.rg): Module that contains the tasks that compute auxiliary variables from the unknowns and the other way around  
   >  > [prometeo_variables.h](src/prometeo_variables.h): C header for the tasks that compute auxiliary variables from the unknowns and the other way around  
   >  > [prometeo_variables.hpp](src/prometeo_variables.hpp): C++ header for the tasks that compute auxiliary variables from the unknowns and the other way around  
   >  > [prometeo_variables.cc](src/prometeo_variables.cc): C++ implementations of the tasks that compute auxiliary variables from the unknowns and the other way around  
   >  > [prometeo_variables.cu](src/prometeo_variables.cu): CUDA kernels of the tasks that compute auxiliary variables from the unknowns and the other way around  
   >  > [prometeo_IO.rg](src/prometeo_IO.rg): Module that contains the tasks that perform I/O operations  
   >  > [prometeo_probe.rg](src/prometeo_probe.rg): Module that contains the tasks to perform the time probes of the solution  
   >  > [prometeo_average.rg](src/prometeo_average.rg): Module that contains the tasks that perform 2D averages of the solution  
   >  > [prometeo_metric.rg](src/prometeo_metric.rg): Module that contains the tasks that compute the metric of the solver  
   >  > [prometeo_metric.h](src/prometeo_metric.h): C header for the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_metric_coeffs.h](src/prometeo_metric_coeffs.h): C header containing constant coefficients for the tasks that compute the metric of the solver  
   >  > [prometeo_metric.hpp](src/prometeo_metric.hpp): C++ header for the tasks that compute the metric of the solver  
   >  > [prometeo_metric.inl](src/prometeo_metric.inl): C++ implementations of the inlined functions that compute and apply the metric of the solver  
   >  > [prometeo_metric.cc](src/prometeo_metric.cc): C++ implementations of the tasks that compute and apply the metric of the solver  
   >  > [prometeo_metric.cu](src/prometeo_metric.cu): CUDA kernels of the tasks that compute and apply the metric of the solver  
   >  > [prometeo_mapper.h](src/prometeo_mapper.h) [prometeo_mapper.cc](src/prometeo_mapper.cc): Source files for the mapper of the solver  
   >  > [desugar.py](src/desugar.py): Script that substitutes some macros used for metaprogramming  
   >  > [Makefile](src/Makefile): Builds the solver  
   >  > [json.h](src/json.h) [json.c](src/json.c): Source files to interpret `*.json` files  
   >  > [Reaction.rg](src/Reaction.rg): Tasks and data structures related to chemical reactions  
   >  > [Reaction.h](src/Reaction.h): C header for the data structures related to chemical reactions  
   >  > [Species.rg](src/Species.rg): Tasks and data structures related to chemical species  
   >  > [Species.h](src/Species.h): C header for the data structures related to chemical species  
   >  > [Species.hpp](src/Species.hpp): C++ implementations of the tasks related to chemical species  
   >  > [MultiComponent.rg](src/MultiComponent.rg): Tasks related to a multicomponent mixtures  
   >  > [MultiComponent.hpp](src/MultiComponent.hpp): C++ implementations of the tasks related to a multicomponent mixtures  
   >  > [ConstPropMix.rg](src/ConstPropMix.rg): Tasks and data structures describing a calorically perfect gas  
   >  > [ConstPropMix.h](src/ConstPropMix.h): C header for the data structures describing a calorically perfect gas  
   >  > [ConstPropMix.hpp](src/ConstPropMix.hpp): C++ headers for the tasks describing a calorically perfect gas  
   >  > [IsentropicMix.rg](src/IsentropicMix.rg): Tasks and data structures describing isoentropic calorically perfect gas  
   >  > [IsentropicMix.h](src/IsentropicMix.h): C header for the data structures describing an isoentropic calorically perfect gas  
   >  > [IsentropicMix.hpp](src/IsentropicMix.hpp): C++ headers for the tasks describing an isoentropic calorically perfect gas  
   >  > [AirMix.rg](src/AirMix.rg): Tasks and data structures describing a non-equilibrium dissociating air  
   >  > [AirMix.h](src/AirMix.h): C header for the data structures describing a non-equilibrium dissociating air  
   >  > [AirMix.hpp](src/AirMix.hpp): C++ headers for the tasks of a non-equilibrium dissociating air  
   >  > [CH41StMix.rg](src/CH41StMix.rg): Tasks and data structures describing single step combustion mechanism for CH4  
   >  > [CH41StMix.h](src/CH41StMix.h): C header for the data structures describing single step combustion mechanism for CH4  
   >  > [CH41StMix.hpp](src/CH41StMix.hpp): C++ headers for the tasks and data structures describing single step combustion mechanism for CH4  
   >  > [config_schema.lua](src/config_schema.lua): Lua file describing the fields of the input file  
   >  > [process_schema.rg](src/process_schema.rg): Interpreter for [config_schema.lua](src/config_schema.lua)  
   >  > [util.rg ](src/util.rg): Various Regent tasks and Lua functions deployed throughout the solver  
   >  > [math_utils.rg](src/math_utils.rg): Basic mathematical operations deployed in the solver  
   >  > [math_utils.hpp](src/math_utils.hpp): C++ implementations of basic mathematical operations deployed in the solver  
   >  > [cuda_utils.hpp](src/cuda_utils.hpp): Utilities for CUDA kernels  
   >  > [hdf_helper.rg](src/hdf_helper.rg): Scripts to read and write HDF5 files  

   > [jobscripts/](jobscripts/)  
   >  > [blacklist](jobscripts/blacklist): Folder containing potential blacklists of nodes that should not be used
   >  > [run.sh](jobscripts/run.sh): Script called by [prometeo.sh](prometeo.sh) to generate the execution command (modify this script using the provided templates to add a new machine)  
   >  > [jobscript_shared.sh](jobscripts/jobscript_shared.sh): Script called by [run.sh](src/run.sh)  
   >  > [yellowstone.slurm](jobscripts/yellowstone.slurm): Submission script for Yellowstone (@ Stanford) (use as a template script for slurm system)  
   >  > [armstrong.slurm](jobscripts/armstrong.slurm): Submission script for Armstrong (@ Stanford) (use as a template script for slurm system)  
   >  > [quartz.slurm](jobscripts/quartz.slurm): Submission script for Quartz (@ LLNL) (use as a template script for slurm system)  
   >  > [lassen.lsf](jobscripts/lassen.lsf): Submission script for Lassen (@ LLNL) (use as a template script for IBM Spectrum LSF system)  
   >  > [galileo.slurm](jobscripts/galileo.slurm): Submission script for Galileo (@ Cineca) (use as a template script for slurm system)  
   >  > [m100.slurm](jobscripts/m100.slurm): Submission script for Marconi100 (@ Cineca) (use as a template script for slurm system)  

   > [scripts/](scripts/)  
   >  > [viz_fluid.py](scripts/viz_fluid.py): Script to produce Xdmf files for visualization  
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
   >  > [RecycleBoundary/](testcases/RecycleBoundary/): Contains the setup and postporcessing files needed to run a compressible boundary layer with recycle-rescaling boundary condition  
   >  > [VortexAdvection2D/](testcases/VortexAdvection2D/): Contains the setup and postporcessing files needed to run the inviscid vortex advection  
   >  > [TaylorGreen2D/](testcases/TaylorGreen2D/): Contains the setup and postporcessing files needed to run the 2D Taylor-Green vortex  
   >  > [Coleman/](testcases/Coleman/): Contains the setup and postporcessing files needed to run Coleman's channel flow  
   >  > [Sciacovelli/](testcases/Sciacovelli/): Contains the setup and postporcessing files needed to run Sciacovelli's channel flow  
   >  > [Franko/](testcases/Franko/): Contains the setup and postporcessing files needed to run Franko's boundary layer  
   >  > [MultispeciesTBL/](testcases/MultispeciesTBL/): Contains the setup and postporcessing files needed to run Multispecies hypersonic boundary layer  
   >  > [scalingTest/WeakScaling](testcases/scalingTest/WeakScaling): Contains the setup and postporcessing files needed to run the weak scaling test
   >  > [scalingTest/StrongScaling](testcases/scalingTest/StrongScaling): Contains the setup and postporcessing files needed to run the strong scaling test

   > [unitTests/](unitTests/)  
   >  > [cflTest/](unitTests/cflTest/): Contains the unit test for the cfl module  
   >  > [chemTest/](unitTests/chemTest/): Contains the unit test for the chemistry module  
   >  > [configTest/](unitTests/configTest/): Contains the unit test for the config schema module  
   >  > [geometryTest/](unitTests/geometryTest/): Contains the unit test for the geometry module  
   >  > [hdfTest/](unitTests/hdfTest/): Contains the unit test for the hdf helper module  
   >  > [mathUtilsTest/](unitTests/mathUtilsTest/): Contains the unit test for the mathUtils module  
   >  > [metricTest/](unitTests/metricTest/): Contains the unit test for the metric module  
   >  > [mixTest/](unitTests/mixTest/): Contains the unit test for the mixture modules  
   >  > [variablesTest/](unitTests/variablesTest/): Contains the unit test for the variables module  
   >  > [probeTest/](unitTests/probeTest/): Contains the unit test for the probe module  
   >  > [averageTest/](unitTests/averageTest/): Contains the unit test for the average module  

   > [solverTests/](solverTests/)  
   >  > [VortexAdvection2D/](solverTests/VortexAdvection2D/): Contains the solver test for the bi-periodic 2D inviscid testcase  
   >  > [3DPeriodic/](solverTests/3DPeriodic/): Contains the solver test for the tri-periodic 3D testcase  
   >  > [3DPeriodic_Air/](solverTests/3DPeriodic_Air): Contains the solver test for the tri-periodic 3D testcase with AirMix  
   >  > [ChannelFlow/](solverTests/ChannelFlow): Contains the solver test for the bi-periodic ChannelFlow testcase  
   >  > [BoundaryLayer/](solverTests/BoundaryLayer): Contains the solver test for a compressible boundary layer  
   >  > [M2_Re4736/](solverTests/M2_Re4736): Contains the solver test for a compressible boundary layer at Mach=2 and Re=4736 with recycle-rescaling boundary condition  
   >  > [PlanarJet/](solverTests/PlanarJet): Contains the solver test for a compressible jet of methane in air  
   >  > [ShockTube/](solverTests/ShockTube): Contains the solver test for the Shu-Osher's shock tube  


Setup (generic)
===============

See below for instructions targeting specific systems.

### Prerequisites

* Legion (latest version)
* GCC 4.9+ (we need a working `std::regex` library)
* CUDA 8.0+
* Python 3.X

The following are automatically installed during Legion installation:

* LLVM 6.0
* GASNET (custom version)
* Terra (custom version)
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
# Legion setup
export USE_CUDA=?
export USE_OPENMP=?
export USE_GASNET=?
export USE_HDF=?
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

Replace the `?` depending on your system's capabilities.

```
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

See [Elliott's instructions](https://docs.google.com/document/d/1Qkl6r-1ZIb8WyH1f_WZbKgjp3due_Q8UiWKLh_nG1ec/edit) for more help.

### Compile the HTR solver

NOTE: This step may take up to 2 hrs depending on the system

```
cd "$HTR_DIR"/src
make
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
# Legion setup
export USE_CUDA=0
export USE_OPENMP=1
export USE_GASNET=0
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
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
# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make
```

Setup (Yellowstone @ Stanford w/o GPUs)

### Add to shell startup

```
# Module loads
module load gnu7/7.3.0
module load openmpi3/3.0.0
module load pmix/2.2.2
# Build config
export CONDUIT=ibv
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
# Legion setup
export USE_CUDA=0
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
# Rest of compilation as normal
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make -j
```

Setup (Yellowstone @ Stanford w/ GPUs)
============================

### Add to shell startup

```
# Module loads
module load gnu7/7.3.0
module load cuda/9.2
module load openmpi3/3.0.0
module load pmix/2.2.2
# Build config
export CONDUIT=ibv
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
# CUDA config
export CUDA_HOME=/usr/local/cuda-9.2
export CUDA="$CUDA_HOME"
export GPU_ARCH=kepler
# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
# Rest of compilation as normal
cd "$LEGION_DIR"/language
srun -p gpu scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
srun -p gpu make -j
```

Setup (Armstrong @ Stanford)
============================

### Add to shell startup

```
# Module loads
module load gnu7/7.3.0
module load openmpi3/3.1.0
module load pmix/2.2.2
# Build config
export CONDUIT=ibv
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
# Legion setup
export USE_CUDA=0
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
# Rest of compilation as normal
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
make -j
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
# Legion setup
export USE_CUDA=0
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
export GASNET_VERSION="GASNet-1.32.0"
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
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

export GROUP=???

# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3 
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
lalloc 1 scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
lalloc 1 -W 120 make
```

Setup (Galileo @ Cineca)
============================

### Add to shell startup

```
# Module loads
module load gnu
module load openmpi/3.1.1--gnu--6.1.0
module load cuda/9.0
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
export CUDA_HOME="/cineca/prod/compilers/cuda/9.0/none"
export CUDA="$CUDA_HOME"
export GPU_ARCH=kepler

export ACCOUNT=[ACCOUNT TO BE CHARGED]

# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
srun --cpus-per-task=3 --mem 30000 -p gll_usr_gpuprod -t 0:50:00 make -j &
```

Setup (Marconi100 @ Cineca)
============================

### Add to shell startup

```
# Module loads
module load profile/advanced
module load gnu
module load cuda/10.1
module load openmpi/4.0.3--gnu--8.4.0
module load cmake
module load anaconda
# Build config
export CC=mpicc
export CXX=mpic++
export CONDUIT=ibv
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
export SCRATCH=???
# CUDA config
export CUDA_HOME="/cineca/prod/opt/compilers/cuda/10.1/none"
export CUDA="$CUDA_HOME"
export GPU_ARCH=volta

export ACCOUNT=[ACCOUNT TO BE CHARGED]

# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b control_replication https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py --llvm-version 60 --terra-url 'https://github.com/mariodirenzo/terra.git' --terra-branch 'luajit2.1'
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
make -j
```

