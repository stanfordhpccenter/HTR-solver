Organization of the repository
==============================

./  
   > [prometeo.sh](prometeo.sh): Script to run a calculation  
   > [Makefile.in](Makefile.in): Basic definitions of compile time variables  
   >
   > [src/](src/)  
   >  > [prometeo.rg](src/prometeo.rg): Main tasks of the solver  
   >  > [prometeo_const.rg](src/prometeo_const.rg): Module that contains constants used by the solver  
   >  > [prometeo_macro.rg](src/prometeo_macro.rg): Module that contains simple macros used by the solver  
   >  > [prometeo_init.rg](src/prometeo_init.rg): Module that contains initialization tasks  
   >  > [prometeo_grid.rg](src/prometeo_grid.rg): Module that contains the tasks to generate the computational grid  
   >  > [prometeo_cfl.rg](src/prometeo_cfl.rg): Module that contains the tasks to compute the CFL number  
   >  > [prometeo_cfl.h](src/prometeo_cfl.h): C header for the tasks to compute the CFL number  
   >  > [prometeo_cfl.hpp](src/prometeo_cfl.hpp): C++ header for the tasks to compute the CFL number  
   >  > [prometeo_cfl.cc](src/prometeo_cfl.cc): C++ implementations of the tasks to compute the CFL number  
   >  > [prometeo_cfl.cu](src/prometeo_cfl.cu): CUDA implementations of the tasks to compute the CFL number  
   >  > [prometeo_chem.rg](src/prometeo_chem.rg): Module that contains the tasks to advance chemistry  
   >  > [prometeo_chem.h](src/prometeo_chem.h): C header for the tasks that advance chemistry  
   >  > [prometeo_chem.hpp](src/prometeo_chem.hpp): C++ header for the tasks that advance chemistry  
   >  > [prometeo_chem.cc](src/prometeo_chem.cc): C++ implementations of the tasks that advance chemistry  
   >  > [prometeo_chem.cu](src/prometeo_chem.cu): CUDA implementations of the tasks that advance chemistry  
   >  > [prometeo_bc.rg](src/prometeo_bc.rg): Module that contains the tasks that handle boundary conditions  
   >  > [prometeo_bc_types.h](src/prometeo_bc_types.h): C header for the data types used in boundary condition tasks  
   >  > [prometeo_bc.h](src/prometeo_bc.h): C header for the tasks that handle boundary conditions  
   >  > [prometeo_bc.hpp](src/prometeo_bc.hpp): C++ header for the tasks that handle boundary conditions  
   >  > [prometeo_bc.cc](src/prometeo_bc.cc): C++ implementations of the tasks that handle boundary conditions  
   >  > [prometeo_bc.cu](src/prometeo_bc.cu): CUDA implementations of the tasks that handle boundary conditions  
   >  > [prometeo_profiles.rg](src/prometeo_profiles.rg): Module that contains the tasks that handle external profiles provided to the solver  
   >  > [prometeo_rk.rg](src/prometeo_rk.rg): Module that contains the tasks for Runge-Kutta algorithm  
   >  > [prometeo_stat.rg](src/prometeo_stat.rg): Module that contains the tasks that extract integral quantities from the solution  
   >  > [prometeo_redop.inl](src/prometeo_redop.inl): Inlined file containing the reduction operations used in the solver  
   >  > [prometeo_mixture.rg](src/prometeo_mixture.rg): Module that contains the tasks that initialize the mixure data structure  
   >  > [prometeo_mixture.h](src/prometeo_mixture.h): C header for the tasks that initialize the mixure data structure  
   >  > [prometeo_mixture.hpp](src/prometeo_mixture.hpp): C++ header for the tasks that initialize the mixure data structure  
   >  > [prometeo_mixture_wrappers.hpp](src/prometeo_mixture_wrappers.hpp): C++ header containing the declarations of the C wrappers of the methods contained in the mixture data structure  
   >  > [prometeo_mixture.cc](src/prometeo_mixture.cc): C++ implementations of the tasks that initialize the mixure data structure  
   >  > [prometeo_mixture.cu](src/prometeo_mixture.cu): CUDA kernels of the tasks that initialize the mixure data structure  
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
   >  > [prometeo_average_types.h](src/prometeo_average_types.h): C header that contains the data types for the tasks that perform space and time averages of the solution  
   >  > [prometeo_average.rg](src/prometeo_average.rg): Module that contains the tasks that perform space and time averages of the solution  
   >  > [prometeo_average.h](src/prometeo_average.h): C header for the tasks that perform space and time averages of the solution  
   >  > [prometeo_average.hpp](src/prometeo_average.hpp):  C++ header for the tasks that perform space and time averages of the solution  
   >  > [prometeo_average.inl](src/prometeo_average.inl):  C++ implementations of the inlined functions that perform space and time averages of the solution  
   >  > [prometeo_average.cc](src/prometeo_average.cc): C++ implementations of tasks that perform space and time averages of the solution  
   >  > [prometeo_average.cu](src/prometeo_average.cu): CUDA implementations of tasks that perform space and time averages of the solution  
   >  > [prometeo_metric.rg](src/prometeo_metric.rg): Module that contains the tasks that compute the metric of the solver  
   >  > [prometeo_metric.h](src/prometeo_metric.h): C header for the tasks that operate on the right-hand side of the equations  
   >  > [prometeo_metric.hpp](src/prometeo_metric.hpp): C++ header for the tasks that compute the metric of the solver  
   >  > [prometeo_metric.inl](src/prometeo_metric.inl): C++ implementations of the inlined functions that compute and apply the metric of the solver  
   >  > [prometeo_metric.cc](src/prometeo_metric.cc): C++ implementations of the tasks that compute and apply the metric of the solver  
   >  > [prometeo_metric.cu](src/prometeo_metric.cu): CUDA kernels of the tasks that compute and apply the metric of the solver  
   >  > [prometeo_metric_coeffs.h](src/prometeo_metric_coeffs.h): C header that declares the constant coefficients for the tasks that compute the metric of the solver  
   >  > [prometeo_metric_coeffs.cc](src/prometeo_metric_coeffs.cc): C++ sources that define the constant coefficients for the tasks that compute the metric of the solver  
   >  > [prometeo_metric_coeffs.cu](src/prometeo_metric_coeffs.cu): CUDA sources that define the constant coefficients for the tasks that compute the metric of the solver  
   >  > [prometeo_metric_coeffs_macros.h](src/prometeo_metric_coeffs_macros.h): C header that contains helper macros for definition of the constant coefficients for the tasks that compute the metric of the solver  
   >  > [prometeo_electricField.rg](src/src/prometeo_electricField.rg): Module that contains the tasks to update the electric field and ion wind  
   >  > [prometeo_electricField.h](src/prometeo_electricField.h): C header for the tasks to update the electric field and ion wind  
   >  > [prometeo_electricField.hpp](src/prometeo_electricField.hpp): C++ header for the tasks to update the electric field and ion wind  
   >  > [prometeo_electricField.inl](src/prometeo_electricField.inl): C++ implementations of the inlined function that update the electric field and ion wind  
   >  > [prometeo_electricField.cc](src/prometeo_electricField.cc): C++ implementations of the tasks to update the electric field and ion wind  
   >  > [prometeo_electricField.cu](src/prometeo_electricField.cu): CUDA implementations of the tasks to update the electric field and ion wind  
   >  > [prometeo_laser.rg](src/prometeo_laser.rg): Module containing tasks for the laser model
   >  > [prometeo_mapper.h](src/prometeo_mapper.h) [prometeo_mapper.cc](src/prometeo_mapper.cc): Source files for the mapper of the solver  
   >  > [desugar.py](src/desugar.py): Script that substitutes some macros used for metaprogramming  
   >  > [Makefile](src/Makefile): Builds the solver  
   >  > [json.h](src/json.h) [json.c](src/json.c): Source files to interpret `*.json` files  
   >  > [config_schema.lua](src/config_schema.lua): Lua file describing the fields of the input file  
   >  > [process_schema.rg](src/process_schema.rg): Interpreter for [config_schema.lua](src/config_schema.lua)  
   >  > [util.rg ](src/util.rg): Various Regent tasks and Lua functions deployed throughout the solver  
   >  > [hdf_helper.rg](src/hdf_helper.rg): Scripts to read and write HDF5 files  
   >  > [Utils](src/Utils)  
   >  >  > [constants.h](src/Utils/constants.h): Definition of main physical and math constants used in the sover  
   >  >  > [my_array.hpp](src/Utils/my_array.hpp): Definition of array and matrix data types  
   >  >  > [task_helper.hpp](src/Utils/task_helper.hpp): Utilities for task registration  
   >  >  > [PointDomain_helper.hpp](src/Utils/PointDomain_helper.hpp): Utilities for point and domains  
   >  >  > [cuda_utils.hpp](src/Utils/cuda_utils.hpp): Utilities for CUDA kernels  
   >  >  > [cuda_utils.cu](src/Utils/cuda_utils.cu): Utilities for CUDA kernels  
   >  >  > [math_utils.h](src/Utils/math_utils.h): Definition of data types for basic mathematical operations deployed in the solver  
   >  >  > [math_utils.rg](src/Utils/math_utils.rg): Regent implementation of basic mathematical operations deployed in the solver  
   >  >  > [math_utils.hpp](src/Utils/math_utils.hpp): C++ implementations of basic mathematical operations deployed in the solver  
   >  >
   >  > [Poisson](src/Poisson)  
   >  >  > [Poisson.rg](src/Poisson/Poisson.rg): Module that contains the the tasks of the Poisson solver  
   >  >  > [Poisson.h](src/Poisson/Poisson.h): C header for the tasks of the Poisson solver  
   >  >  > [Poisson.hpp](src/Poisson/Poisson.hpp): C++ header for the tasks of the Poisson solver  
   >  >  > [Poisson.cc](src/Poisson/Poisson.cc): C++ implementations of the tasks of the Poisson solver  
   >  >  > [Poisson.cu](src/Poisson/Poisson.cu): CUDA implementations of the tasks of the Poisson solver  
   >  >
   >  > [Mixtures](src/Mixtures)  
   >  >  > [Reaction.hpp](src/Mixtures/Reaction.hpp): C++ declarations of the data structures related to chemical reactions  
   >  >  > [Reaction.inl](src/Mixtures/Reaction.inl): C++ implementations of the functions related to chemical reactions  
   >  >  > [Species.hpp](src/Mixtures/Species.hpp): C++ declarations of the data structures related to chemical species  
   >  >  > [Species.inl](src/Mixtures/Species.inl): C++ implementations of the functions related to chemical species  
   >  >  > [MultiComponent.hpp](src/Mixtures/MultiComponent.hpp): C++ declarations for multicomponent mixtures  
   >  >  > [MultiComponent.inl](src/Mixtures/MultiComponent.inl): C++ implementations of the functions related to a multicomponent mixture  
   >  >  > [ConstPropMix.hpp](src/Mixtures/ConstPropMix.hpp): C++ declarations for the data structure describing a calorically perfect gas  
   >  >  > [ConstPropMix.inl](src/Mixtures/ConstPropMix.inl): C++ implementations of the data structure describing a calorically perfect gas  
   >  >  > [IsentropicMix.hpp](src/Mixtures/IsentropicMix.hpp): C++ declarations for the data structure describing an isoentropic calorically perfect gas  
   >  >  > [IsentropicMix.inl](src/Mixtures/IsentropicMix.inl): C++ implementations of the data structure describing an isoentropic calorically perfect gas  
   >  >  > [AirMix.hpp](src/Mixtures/AirMix.hpp): C++ headers for the tasks of a non-equilibrium dissociating air  
   >  >  > [CH41StMix.hpp](src/Mixtures/CH41StMix.hpp): C++ headers for the tasks and data structures describing single step combustion mechanism for CH4  
   >  >  > [CH4_30SpMix.hpp](src/Mixtures/CH4_30SpMix.hpp): C++ headers for the tasks and data structures describing Lu and Law (2008) combustion mechanism for CH4  
   >  >  > [FFCM1Mix.hpp](src/Mixtures/FFCM1Mix.hpp): C++ headers for the tasks and data structures describing FFCM1 mechanism, (https://web.stanford.edu/group/haiwanglab/FFCM1/)  
   >  >  > [CH4_26SpIonsMix.hpp](src/Mixtures/CH4_26SpIonsMix.hpp): C++ headers for the tasks and data structures describing 26 species combustion mechanism for CH4 with ions  
   >  >  > [BoivinMix.hpp](src/Mixtures/BoivinMix.hpp): C++ headers for the tasks and data structures describing 9 species combustion mechanism for H2 from Boivin et al. PCI 2011  
   >  >  > [H2_UCSDMix.hpp](src/Mixtures/H2_UCSDMix.hpp): C++ headers for the tasks and data structures describing UCSD 9 species combustion mechanism for H2 (Saxena & Williams (2006) CnF 145))  
   >
   > [jobscripts/](jobscripts/)  
   >  > [blacklist](jobscripts/blacklist): Folder containing potential blacklists of nodes that should not be used  
   >  > [run.sh](jobscripts/run.sh): Script called by [prometeo.sh](prometeo.sh) to generate the execution command (modify this script using the provided templates to add a new machine)  
   >  > [jobscript_shared.sh](jobscripts/jobscript_shared.sh): Script called by [run.sh](src/run.sh)  
   >  > [yellowstone.slurm](jobscripts/yellowstone.slurm): Submission script for Yellowstone (@ Stanford) (use as a template script for slurm system)  
   >  > [armstrong.slurm](jobscripts/armstrong.slurm): Submission script for Armstrong (@ Stanford) (use as a template script for slurm system)  
   >  > [quartz.slurm](jobscripts/quartz.slurm): Submission script for Quartz (@ LLNL) (use as a template script for slurm system)  
   >  > [lassen.lsf](jobscripts/lassen.lsf): Submission script for Lassen (@ LLNL) (use as a template script for IBM Spectrum LSF system)  
   >  > [m100.slurm](jobscripts/m100.slurm): Submission script for Marconi100 (@ Cineca) (use as a template script for slurm system)  
   >  > [kraken.slurm](jobscripts/kraken.slurm): Submission script for Kraken (@ CERFACS)
   >
   > [scripts/](scripts/)  
   >  > [viz_fluid.py](scripts/viz_fluid.py): Script to produce Xdmf files for visualization  
   >  > [compare_hdf.py](scripts/compare_hdf.py): Script to compare two HDF5 files  
   >  > [merge.py](scripts/merge.py): Script to merge multiple HDF5 files into a single file  
   >  > [makeVirtualLayout.py](scripts/makeVirtualLayout.py): Script to merge multiple HDF5 files into a single file using a virtual layout  
   >  > [interpolate.py](scripts/interpolate.py): Script to interpolate a solution on a new grid  
   >  > [convert_output_for_viz.py](scripts/convert_output_for_viz.py): Script that automates the production of visualization files for multiple snapshots  
   >  > [modules/](scripts/modules/): Various utility modules used by the python scripts
   >
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
   >  > [Speelman/](testcases/Speelman/): Contains the setup and postporcessing files needed to run the burner stabilized flame in Speelman et al. (2015) with an applied difference of potential of 250 V  
   >  > [Speelman_DV250/](testcases/Speelman_DV250/): Contains the setup and postporcessing files needed to run the burner stabilized flame in Speelman et al. (2015) with an applied difference of potential of 250 V  
   >  > [Franko/](testcases/Franko/): Contains the setup and postporcessing files needed to run Franko's boundary layer  
   >  > [MultispeciesTBL/](testcases/MultispeciesTBL/): Contains the setup and postporcessing files needed to run Multispecies hypersonic boundary layer  
   >  > [scalingTest/WeakScaling](testcases/scalingTest/WeakScaling): Contains the setup and postporcessing files needed to run the weak scaling test
   >  > [scalingTest/StrongScaling](testcases/scalingTest/StrongScaling): Contains the setup and postporcessing files needed to run the strong scaling test
   >  > [scalingTest/LFHF](testcases/scalingTest/LFHF): Contains the setup and postporcessing files needed to run the scaling test of an ensamble run with high-fidelity and low-fidelity samples
   >
   > [unitTests/](unitTests/)  
   >  > [cflTest/](unitTests/cflTest/): Contains the unit test for the cfl module  
   >  > [chemTest/](unitTests/chemTest/): Contains the unit test for the chemistry module  
   >  > [configTest/](unitTests/configTest/): Contains the unit test for the config schema module  
   >  > [geometryTest/](unitTests/geometryTest/): Contains the unit test for the geometry module  
   >  > [hdfTest/](unitTests/hdfTest/): Contains the unit test for the hdf helper module  
   >  > [laserTest/](unitTests/laserTest/): Contains the unit test for the laser model
   >  > [mathUtilsTest/](unitTests/mathUtilsTest/): Contains the unit test for the mathUtils module  
   >  > [metricTest/](unitTests/metricTest/): Contains the unit test for the metric module  
   >  > [mixTest/](unitTests/mixTest/): Contains the unit test for the mixture modules  
   >  > [variablesTest/](unitTests/variablesTest/): Contains the unit test for the variables module  
   >  > [probeTest/](unitTests/probeTest/): Contains the unit test for the probe module  
   >  > [averageTest/](unitTests/averageTest/): Contains the unit test for the average module  
   >
   > [solverTests/](solverTests/)  
   >  > [VortexAdvection2D/](solverTests/VortexAdvection2D/): Contains the solver test for the bi-periodic 2D inviscid testcase  
   >  > [3DPeriodic/](solverTests/3DPeriodic/): Contains the solver test for the tri-periodic 3D testcase  
   >  > [3DPeriodic_Air/](solverTests/3DPeriodic_Air): Contains the solver test for the tri-periodic 3D testcase with AirMix  
   >  > [3DPeriodic_SkewSymmetric/](solverTests/3DPeriodic_SkewSymmetric): Contains the solver test for the tri-periodic 3D testcase with the skew symmetric scheme  
   >  > [3DPeriodic_TENOA/](solverTests/3DPeriodic_TENOA): Contains the solver test for the tri-periodic 3D testcase with the TENO-A scheme  
   >  > [3DPeriodic_TENOLAD/](solverTests/3DPeriodic_TENOLAD): Contains the solver test for the tri-periodic 3D testcase with the TENO-LAD scheme  
   >  > [ChannelFlow/](solverTests/ChannelFlow): Contains the solver test for the bi-periodic ChannelFlow testcase  
   >  > [BoundaryLayer/](solverTests/BoundaryLayer): Contains the solver test for a compressible boundary layer  
   >  > [RecycleBoundary/](solverTests/RecycleBoundary): Contains the solver test for a compressible boundary layer with recycle/rescaling BC  
   >  > [M2_Re4736/](solverTests/M2_Re4736): Contains the solver test for a compressible boundary layer at Mach=2 and Re=4736 with recycle-rescaling boundary condition  
   >  > [PlanarJet/](solverTests/PlanarJet): Contains the solver test for a compressible jet of methane in air  
   >  > [ShockTube/](solverTests/ShockTube): Contains the solver test for the Shu-Osher's shock tube  
   >  > [Speelman_DV250/](solverTests/Speelman_DV250): Contains the solver test for the burner stabilized flame in Speelman et al. (2015) with a voltage of 250 V (requires `ELECTRIC_FIELD=1`) 


Setup (generic)
===============

See below for instructions targeting specific systems.

### Prerequisites

* Legion (latest version)
* GCC 6.0+
* CUDA 9.0+
* Python 3.X

The following are automatically installed during Legion installation:

* LLVM 13
* GASNET (custom version)
* Terra 1.0.3
* HDF5 (any recent version)

### Optional modules include

An ion wind solver for charged species activated by the environment variable `ELECTRIC_FIELD`
This module requires FFTW (>=3.3.6) libraries built with the --enable-threads option.

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
# Optional modules
ELECTRIC_FIELD=???
# CUDA config (if using CUDA code generation)
export CUDA_HOME=???
export CUDA="$CUDA_HOME"
export GPU_ARCH=???
# FFTW config (if using electric field solver)
export FFTW_HOME=???
# Legion setup
export USE_CUDA=?
export USE_OPENMP=?
export USE_GASNET=?
export USE_HDF=?
export MAX_DIM=3
```

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

Replace the `?` depending on your system's capabilities.

```
cd "$LEGION_DIR"/language
scripts/setup_env.py
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
* `-lp <config>.json`: Provide a case configuration file to be run as an additional low-priority sample. Low-priority samples are limited to using CPUs only. See [src/config_schema.lua](src/config_schema.lua) for documentation on the available options (`Config` struct).

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
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
USE_CUDA=0 make
```

Setup (local macOS machine w/o GPU)
====================================

### Add to shell startup

```
# Build config
export INCLUDE_PATH="$(xcrun --sdk macosx --show-sdk-path)/usr/include:$INCLUDE_PATH"
export CXX_FLAGS="-std=c++11"
# Path setup
export LLVM_DIR=???
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

### Install HDF5

```
mkdir $LEGION_DIR/language/hdf
cd $LEGION_DIR/language/hdf
curl http://sapling.stanford.edu/~manolis/hdf/hdf5-1.10.1.tar.gz --out hdf5-1.10.1.tar.gz
tar -xvf hdf5-1.10.1.tar.gz
cd hdf5-1.10.1
./configure --prefix=$HDF_ROOT  --enable-threadsafe --disable-hl
make -j
make install
```

### Download LLVM

```
mkdir $LLVM_DIR
cd $LLVM_DIR
curl -O https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/clang+llvm-9.0.1-x86_64-apple-darwin.tar.xz
tar xfJ clang+llvm-9.0.1-x86_64-apple-darwin.tar.xz
```

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
CXXFLAGS='-std=c++11' $LEGION_DIR/language/install.py -j --no-clean
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make
```

Setup (Sapling @ Stanford)
==========================

### Add to shell startup

```
# Module loads
module load mpi/openmpi/4.1.0
module load slurm/20.11.4
# Build config
export CONDUIT=ibv
export CC=gcc-8
export CXX=g++-8
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
export SCRATCH=/scratch/oldhome/`whoami`
# Legion setup
export USE_CUDA=0
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
export GASNET_VERSION="GASNet-2020.3.0"
```

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
make
```

Setup (Yellowstone @ Stanford w/o GPUs)
============================

### Add to shell startup

```
# Module loads
module load gnu7/7.3.0
module load openmpi3/3.1.0
module load pmix/2.2.2
module load cmake/3.15.4
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
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
# Rest of compilation as normal
cd "$LEGION_DIR"/language
scripts/setup_env.py
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
module load cuda/11.1
module load openmpi3/3.1.0
module load pmix/2.2.2
module load cmake/3.15.4
# Build config
export CONDUIT=ibv
export CC=gcc
export CXX=g++
# Path setup
export LEGION_DIR=???
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=???
# CUDA config
export CUDA_HOME=/usr/local/cuda-11.1
export CUDA="$CUDA_HOME"
export GPU_ARCH=maxwell
# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
# Rest of compilation as normal
cd "$LEGION_DIR"/language
srun -p gpu-maxwell scripts/setup_env.py
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
srun -p gpu-maxwell make -j
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
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://gitlab.com/insieme1/htr/htr-solver.git "$HTR_DIR"
```

### Install Legion

```
# Rest of compilation as normal
cd "$LEGION_DIR"/language
scripts/setup_env.py
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
module load gcc/6.1.0
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
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py
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
module load cuda/11.1.1
module load cmake
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
export CUDA_HOME=/usr/tce/packages/cuda/cuda-11.1.1
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
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
lalloc 1 scripts/setup_env.py
```

### Compile the HTR solver

```
cd "$HTR_DIR"/src
lalloc 1 -W 120 make
```

Setup (Solo @ Sandia)
============================

### Add to shell startup

```
# Module loads
module load gnu/7.3.1
module load openmpi-gnu/3.0
module load pmix214
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
# Legion setup
export USE_CUDA=0
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
salloc -N1 --time=1:00:00 -p short --account=??? scripts/setup_env.py
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
salloc -N1 --time=1:00:00 -p short --account=??? make -j
```

Setup (Marconi100 @ Cineca)
============================

### Add to shell startup

```
# Module loads
module load profile/global
module load gnu/8.4.0
module load cuda/11.1
module load spectrum_mpi/10.4.0--binary
module load cmake
module load anaconda
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
export CUDA_HOME="/cineca/prod/opt/compilers/cuda/11.1/none"
export CUDA="$CUDA_HOME"
export GPU_ARCH=volta

export ACCOUNT=[ACCOUNT TO BE CHARGED]

# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
export REALM_NETWORKS=gasnetex
```

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
make -j
```

Setup (Kraken @ CERFACS)
============================

### Add to shell startup

```
# Module loads
module purge
module load compiler/gcc/8.3.0
module load mpi/openmpi/3.1.5_gcc83
module load nvidia/cuda/11.2
module load python/anaconda3.7
module load tools/cmake
# Build config
export CONDUIT=psm
export CC=gcc
export CXX=g++
# Path setup
export SCRATCH=/scratch/cfd/direnzo
export LEGION_DIR="$SCRATCH"/legion
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR="$SCRATCH"/htr
# CUDA config
export CUDA_HOME="/softs/nvidia/cuda-11.2"
export CUDA="$CUDA_HOME"
export GPU_ARCH=volta

# Legion setup
export USE_CUDA=1
export USE_OPENMP=1
export USE_GASNET=1
export USE_HDF=1
export MAX_DIM=3
```

# For new GPU nodes change `GPU_ARCH=ampere`

### Download software

```
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://github.com/stanfordhpccenter/HTR-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
scripts/setup_env.py
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
make -j
```

Setup (Agave @ ASU)
============================

### Add to shell startup

```
# Module loads
module purge
module load gcc/7.5.0
module load openmpi/3.0.6-gnu-7.5.0
module load cmake/3.20.3
# Build config
export CC=gcc
export CXX=g++
export CONDUIT=psm
# Path setup
export LEGION_DIR=$HOME/legion
export HDF_ROOT="$LEGION_DIR"/language/hdf/install
export HTR_DIR=$HOME/htr
export SCRATCH=/scratch/$USER
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
git clone -b HTR-1.4 https://gitlab.com/mario.direnzo/legion.git "$LEGION_DIR"
git clone https://gitlab.com/insieme1/htr/htr-solver.git "$HTR_DIR"
```

### Install Legion

```
cd "$LEGION_DIR"/language
srun -p parallel --cpu-bind=none --mpi=pmi2 -N 1 --exclusive scripts/setup_env.py
```

### Compile Prometeo

```
cd "$HTR_DIR"/src
srun -p parallel --cpu-bind=none --mpi=pmi2 -N 1 --exclusive make -j
```

