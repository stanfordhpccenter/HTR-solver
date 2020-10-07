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

### Weak scaling test

```
cd scalingTest/WS
python scale_up.py -n [Number of refinements] -out [output dir] base.json
python postProc.py -n [Number of refinements] -out [output dir]
```

