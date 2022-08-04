import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local cos  = regentlib.cos( double)
local sin  = regentlib.sin( double)
local sqrt = regentlib.sqrt(double)

local SCHEMA = terralib.includec("config_schema.h")
local REGISTRAR = terralib.includec("registrar.h")
local UTIL = require 'util'
local CONST = require "prometeo_const"

local MAPPER = {
   SAMPLE_ID_TAG = 1234
}

--local TYPES = terralib.includec("Poisson.h", include_dirs)

local PI = CONST.PI

-------------------------------------------------------------------------------
-- Load data types
-------------------------------------------------------------------------------
local types_inc_flags = terralib.newlist({"-DEOS=IsentropicMix", "-DELECTRIC_FIELD"})
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local bBoxType = TYPES.bBoxType

-------------------------------------------------------------------------------
--External modules
-------------------------------------------------------------------------------
local MACRO = require "prometeo_macro"
local IO = (require 'prometeo_IO')(SCHEMA)
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)
local GRID = (require 'prometeo_grid')(SCHEMA, IO, Fluid_columns, bBoxType,
                                       PART.zones_partitions, PART.output_partitions)
local POISSON = (require "Poisson")(SCHEMA, MIX, TYPES, Fluid_columns,
                                    terralib.newlist({"rho"}), TYPES.TID_performDirFFTFromField,
                                    "electricPotential",
                                    "dcsi_d", "deta_d", "dzet_d", "deta_s", "nType_y")

-------------------------------------------------------------------------------
-- Test parameters
-------------------------------------------------------------------------------
local Amp = 10
local WN = 1

local Phi_bc = 100

local Ng = 1
local Npx = 32
local Npy = 256
local Npz = 32

local xO = 0.0
local yO = 0.0
local zO = 0.0

local xW = 1.0
local yW = 1.0
local zW = 1.0

local Nx = 4
local Ny = 4
local Nz = 4

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
   fill(Fluid.nType_x, 0)
   fill(Fluid.nType_y, 0)
   fill(Fluid.nType_z, 0)
   fill(Fluid.dcsi_e, 0.0)
   fill(Fluid.deta_e, 0.0)
   fill(Fluid.dzet_e, 0.0)
   fill(Fluid.dcsi_d, 0.0)
   fill(Fluid.deta_d, 0.0)
   fill(Fluid.dzet_d, 0.0)
   fill(Fluid.dcsi_s, 0.0)
   fill(Fluid.deta_s, 0.0)
   fill(Fluid.dzet_s, 0.0)
   fill(Fluid.rho, 0.0)
   fill(Fluid.electricPotential, 0.0)
end

__demand(__leaf)
task setRHS(r : region(ispace(int3d), Fluid_columns),
            A : double)
where
   reads(r.centerCoordinates),
   writes(r.rho)
do
   var Nx = r.bounds.hi.x - r.bounds.lo.x + 1
   -- Add BCs
   for c in r do
      r[c].rho = A*sin(WN*2*PI*r[c].centerCoordinates[1]/xW)
   end
end

__demand(__leaf)
task CheckSolution_noRHS(r : region(ispace(int3d), Fluid_columns))
where
   reads(r.{centerCoordinates, rho, electricPotential})
do
   for c in r do
      var err = r[c].electricPotential/Phi_bc - r[c].centerCoordinates[1]
      regentlib.assert(fabs(err) < 1e-8, "poissonTest: ERROR in solution without RHS")
   end
end

__demand(__leaf)
task CheckSolution_RHS(r : region(ispace(int3d), Fluid_columns))
where
   reads(r.{centerCoordinates, rho, electricPotential})
do
   var fact = WN*2*PI*WN*2*PI
   for c in r do
      var err = (r[c].rho + r[c].electricPotential*fact)/Amp
      --if ((c.x == 0) and (c.z == 0)) then
      --   C.printf("%10.4e %10.4e %10.4e %10.4e\n", r[c].centerCoordinates[1], r[c].rho, r[c].electricPotential, err)
      --end
      regentlib.assert(fabs(err) < 1e-4, "poissonTest: ERROR in solution with RHS")
   end
end

__demand(__leaf)
task printResults(r : region(ispace(int3d), Fluid_columns),
                  s : region(ispace(int3d), complex64))
where
   reads(r.{centerCoordinates, rho, electricPotential}),
   reads(s)
do
   C.printf("Solution:\n")
   for i=r.bounds.lo.y, r.bounds.hi.y+1 do
      var c = int3d({0,i,0})
      C.printf("%d %10.4e %10.4e %10.4e %10.4e\n", i, r[c].centerCoordinates[1], r[c].rho, r[c].electricPotential, s[c].real)
   end
end

local PoissonData = POISSON.mkDataList()
local Grid = {
   xBnum = regentlib.newsymbol(),
   yBnum = regentlib.newsymbol(),
   zBnum = regentlib.newsymbol(),
   NX = regentlib.newsymbol(),
   NY = regentlib.newsymbol(),
   NZ = regentlib.newsymbol(),
   numTiles = regentlib.newsymbol(),
}

task main()

   C.printf("poissonTest: run...\n")

   var config : SCHEMA.Config

   C.snprintf([&int8](config.Mapping.outDir), 256, "./PoissonDir")
   UTIL.createDir(config.Mapping.outDir)

   config.BC.xBCLeft.type  = SCHEMA.FlowBC_Periodic
   config.BC.xBCRight.type = SCHEMA.FlowBC_Periodic
   config.BC.yBCLeft.type  = SCHEMA.FlowBC_NSCBC_Outflow
   config.BC.yBCRight.type = SCHEMA.FlowBC_NSCBC_Outflow
   config.BC.zBCLeft.type  = SCHEMA.FlowBC_Periodic
   config.BC.zBCRight.type = SCHEMA.FlowBC_Periodic

   config.Grid.xNum = Npx
   config.Grid.yNum = Npy
   config.Grid.zNum = Npz

   config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
   config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
   config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
   config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
   config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Cosine
   config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
   config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform

   config.Efield.type = SCHEMA.EFieldStruct_Ybc

   config.Flow.mixture.type = SCHEMA.MixtureModel_IsentropicMix
   config.Flow.mixture.u.IsentropicMix.gasConstant = 1.0
   config.Flow.mixture.u.IsentropicMix.gasConstant = 1.4

   var [Grid.xBnum] = 0
   var [Grid.yBnum] = 1
   var [Grid.zBnum] = 0

   var [Grid.NX] = Nx
   var [Grid.NY] = Ny
   var [Grid.NZ] = Nz
   var [Grid.numTiles] = Nx*Ny*Nz

   -- Define the domain
   var is_Fluid = ispace(int3d, {x = Npx + 2*Grid.xBnum,
                                 y = Npy + 2*Grid.yBnum,
                                 z = Npz + 2*Grid.zBnum})
   var Fluid = region(is_Fluid, Fluid_columns)
   var Fluid_bounds = Fluid.bounds

   -- Partitioning domain
   var tiles = ispace(int3d, {Nx, Ny, Nz})

   -- Fluid Partitioning
   var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, Grid.xBnum, Grid.yBnum, Grid.zBnum)
   var Fluid_Ghost = PART.PartitionGhost(Fluid, tiles, Fluid_Zones, config)
   var {p_All} = Fluid_Zones

   -- Init the mixture
   var Mix = MIX.InitMixture(Fluid, tiles, Fluid_Zones.p_All, config);

   -- Declare Poisson solver
   [POISSON.DeclSymbols(PoissonData, Fluid, tiles, Fluid_Zones, Grid, config, MAPPER)];

   -- Fill Fluid
   InitializeCell(Fluid)

   -- Initialize operators
   METRIC.InitializeOperators(Fluid, tiles, Fluid_Zones, config,
                              Grid.xBnum, Grid.yBnum, Grid.zBnum)

   -- Initialize the geometry
   var boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)

   -- Init Metrics
   METRIC.InitializeMetric(Fluid, tiles, Fluid_Zones, Fluid_Ghost,
                           boundingBox, config);

   -- Init Poisson solver
   [POISSON.Init(PoissonData, tiles, Grid, config)];

   -- Solve without RHS
   -- SetBCs
   config.Efield.u.Ybc.Phi_bottom = 0.0
   config.Efield.u.Ybc.Phi_top = Phi_bc
   -- SetRHS
   setRHS(Fluid, 0.0);
   -- Solve Poisson without RHS
   [POISSON.Solve(PoissonData, tiles, Mix, config)];
   -- Check
   CheckSolution_noRHS(Fluid)

   -- Solve with RHS
   -- SetBCs
   config.Efield.u.Ybc.Phi_bottom = 0.0
   config.Efield.u.Ybc.Phi_top    = 0.0
   -- SetRHS
   setRHS(Fluid, Amp);
   -- Solve
   [POISSON.Solve(PoissonData, tiles, Mix, config)];
   -- Check
   CheckSolution_RHS(Fluid);

   -- Cleanup
   [POISSON.Cleanup(PoissonData, tiles, config)];

   __fence(__execution, __block)

   C.printf("poissonTest: TEST OK\n")

end

--regentlib.start(main)
regentlib.saveobj(main, "poissonTest.o", "object", REGISTRAR.register_tasks)
