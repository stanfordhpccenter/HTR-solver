How to run the testcases
===============

### Sod shock-tube

```
cd SodProblem
bash run.sh
python postProc.py
```

### Lax shock-tube

```
cd LaxProblem
bash run.sh
python postProc.py
```

### Shu-Osher shock-tube

```
cd ShuOsherProblem
bash run.sh
python postProc.py
```

### Grossman-Cinnella shock-tube

```
cd GrossmanCinnellaProblem
bash run.sh
python postProc.py
```

### Incompressible boundary layer

```
cd Blasius
python MakeProfile.py
bash run.sh
python postProc.py
```

### Compressible boundary layer

```
cd CompressibleBL
python MakeInput.py CBL.json
bash run.sh
python postProc.py
```

### 2D Vortex advection

```
cd VortexAdvection2D
bash run.sh [Number of refinements]
python postProc.py -n [Number of refinements]
```

### 2D Taylor-Green vortex

```
cd TaylorGreen2D
bash run.sh [Number of refinements]
python postProc.py -n [Number of refinements]
```

### Coleman's channel flow

The number of nodes for the calculation is set using the `Mapping/tiles` and `Mapping/tilesPerRank` parameters in base.json.
It is strongly advised to run with one tile per GPU

```
cd Coleman
python MakeChannel.py base.json
$HTR_DIR/prometeo.sh -i ChannelFlow.json
$HTR_DIR/prometeo.sh -i ChannelFlowStats.json
python postProc.py -json ChannelFlowStats.json -in [Averages files produced by the code]
```

### Sciacovelli's channel flow

The number of nodes for the calculation is set using the `Mapping/tiles` and `Mapping/tilesPerRank` parameters in base.json.
It is strongly advised to run with one tile per GPU

```
cd Sciacovelli
python MakeChannel.py base.json
$HTR_DIR/prometeo.sh -i ChannelFlow.json
$HTR_DIR/prometeo.sh -i ChannelFlowStats.json
python postProc.py -json ChannelFlowStats.json -in [Averages files produced by the code]
```

### Franko's boundary layer

The number of nodes for the calculation is set using the `Mapping/tiles` and `Mapping/tilesPerRank` parameters in base.json.
It is strongly advised to run with one tile per GPU

```
cd Franko
python MakeInput.py base.json
$HTR_DIR/prometeo.sh -i NoStats.json
$HTR_DIR/prometeo.sh -i   Stats.json
python postProc.py -json Stats.json -in [Averages files produced by the code]
```

### Multispecies hypersonic boundary layer

The number of nodes for the calculation is set using the `Mapping/tiles` and `Mapping/tilesPerRank` parameters in base.json.
It is strongly advised to run with one tile per GPU

```
cd MultispeciesTBL
python MakeInput.py base.json
$HTR_DIR/prometeo.sh -i NoStats.json
$HTR_DIR/prometeo.sh -i   Stats.json
python postProc.py -json Stats.json -in [Averages files produced by the code]
```

### Speelman burner stabilized flame

```
cd Speelman
python Speelman.py (This is optional. It produces the reference solution using Cantera)
$HTR_DIR/prometeo.sh -i Speelman.json
python postProc.py
```

### Speelman_DV250 burner stabilized flame with applied voltage
### NB: This testacase requires the solver to be compiled with the `ELECTRIC_FIELD=1`

```
cd Speelman_DV250
python mkHTRrestart.py
$HTR_DIR/prometeo.sh -i Speelman.json
python postProc.py
```

### One-dimensional planar flame

This test case is similar to `Speelman` but provides the premixed laminar flame
speed, which is readily compared with published values.  Depending on the
system, the `wallTime` parameter in `base.json` may need to be adjusted in
order to reach a steady state.  Four cases are run at varying equivalence
ratios and compared with a Cantera-generated solution.

```
cd PlanarFlame1D
source run.sh
python3 postProc.py
```

### Weak scaling test

```
cd scalingTest/WeakScaling
python scale_up.py -n [Number of refinements] -out [output dir] base.json
python postProc.py -n [Number of refinements] -out [output dir]
```

### Strong scaling test

```
cd scalingTest/StrongScaling
python scale_up.py -n [Number of refinements] -out [output dir] base.json
python postProc.py -n [Number of refinements] -out [output dir]
```
### Laser-in-box test

This test runs a low-energy case for the `GeometricKernel` laser model, post-processes
the solution files, and checks global quantities.

```
cd LaserInBox
source run.sh
source run-postproc.sh # Run this after the simulation is done.

### High-Fidelity along with Low fidelity test

```
cd scalingTest/LFHF
python scale_up.py -n [Number of refinements] --lp lf.json -out [output dir] base.json
python postProc.py -n [Number of refinements] -out [output dir]
```
