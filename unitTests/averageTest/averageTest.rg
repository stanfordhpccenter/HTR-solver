import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local REGISTRAR = terralib.includec("registrar.h")
local SCHEMA = terralib.includec("config_schema.h")
local UTIL = require "util"
local format = require "std/format"

local Config = SCHEMA.Config

local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local Primitives = CONST.Primitives
local Properties = CONST.Properties

-------------------------------------------------------------------------------
-- ACTIVATE ELECTRIC FIELD SOLVER
-------------------------------------------------------------------------------

local ELECTRIC_FIELD = false
if os.getenv("ELECTRIC_FIELD") == "1" then
   ELECTRIC_FIELD = true
   print("#############################################################################")
   print("WARNING: You are compiling with electric field solver.")
   print("#############################################################################")
end

local types_inc_flags = terralib.newlist({"-DEOS="..os.getenv("EOS")})
types_inc_flags:insert("-DAVERAGE_TEST")
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)
local nSpec = MIX.nSpec
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

--External modules
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)
local IO = (require 'prometeo_IO')(SCHEMA)
local AVG = (require 'prometeo_average')(SCHEMA, MIX, TYPES, PART, ELECTRIC_FIELD)

-- Test parameters
local Npx = 16
local Npy = 16
local Npz = 16
local Nx = 2
local Ny = 2
local Nz = 2

local Nrx = 1
local Nry = 1
local Nrz = 1
local fromCellx = rexpr array(  1,   8,   1) end
local uptoCellx = rexpr array(Npx,   8, Npz) end
local fromCelly = rexpr array(  8,   1,   1) end
local uptoCelly = rexpr array(  8, Npy, Npz) end
local fromCellz = rexpr array(  8,   1,   1) end
local uptoCellz = rexpr array(  8, Npy, Npz) end

local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 5000.0 end
local Xi = rexpr array(0.4, 0.2, 0.15, 0.15, 0.1) end
local v  = rexpr array(1.0, 2.0, 3.0) end
local Tres = rexpr 500.0 end

-- Expected properties
local eRho = rexpr 6.2899525132668e-02 end
local eMu  = rexpr 1.2424432854120e-04 end
local eLam = rexpr 2.4040406505225e-01 end
local eDi  = rexpr array(2.5146942638910e-03, 2.4389378958326e-03, 2.4551032686824e-03, 3.6168376636972e-03, 3.9165741891924e-03) end
local eSoS = rexpr 1.4518745895533e+03 end

-- Expected conserved variables
local eConserved  = rexpr array(2.3342371515176e-02, 1.3331617683677e-02, 9.3760512904744e-03, 4.3766946590956e-03, 3.3329044209192e-03, 1.8268107194243e-04, 3.6536214388485e-04, 5.4804321582728e-04, 5.3385045774457e+00) end

-- Normalization quantities
local LRef = rexpr 1.0 end
local PRef = rexpr 101325.0 end
local TRef = rexpr 300.0 end
local YO2Ref = rexpr 0.22 end
local YN2Ref = rexpr 0.78 end
local MixWRef = rexpr 1.0/([YN2Ref]/28.0134e-3 + [YO2Ref]/(2*15.999e-3)) end
local rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
local eRef = rexpr [PRef]/[rhoRef] end
local uRef = rexpr sqrt([PRef]/[rhoRef]) end
local muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
local lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
local DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
   fill(Fluid.nType_x, CONST.Std_node)
   fill(Fluid.nType_y, CONST.Std_node)
   fill(Fluid.nType_z, CONST.Std_node)
   fill(Fluid.dcsi_d, 0.0)
   fill(Fluid.deta_d, 0.0)
   fill(Fluid.dzet_d, 0.0)
   fill(Fluid.pressure, [P]/[PRef])
   fill(Fluid.temperature, [T]/[TRef])
   fill(Fluid.MassFracs,  [Xi])
   fill(Fluid.MolarFracs, [Xi])
   fill(Fluid.velocity, array([v][0]/[uRef], [v][1]/[uRef], [v][2]/[uRef]))
   fill(Fluid.rho, [eRho]/[rhoRef])
   fill(Fluid.mu , [eMu]/[muRef])
   fill(Fluid.lam, [eLam]/[lamRef])
   fill(Fluid.Di , array([eDi][0]/[DiRef], [eDi][1]/[DiRef], [eDi][2]/[DiRef], [eDi][3]/[DiRef], [eDi][4]/[DiRef]))
   fill(Fluid.SoS, [eSoS]/[uRef])
   fill(Fluid.Conserved, [eConserved])
end

__demand(__inline)
task InitGeometry(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid.{dcsi_d, deta_d, dzet_d})
do
   for c in Fluid do
      Fluid[c].dcsi_d = 1.0
      Fluid[c].deta_d = 1.0/double(c.y)
      Fluid[c].dzet_d = 1.0
   end
end

local function checkAverages(YZAverages, XZAverages, XYAverages)
   return rquote
      regentlib.assert(YZAverages[int2d{0,0}].weight == 0.0, "averageTest: ERROR in YZAverages")
      regentlib.assert(YZAverages[int2d{0,0}].pressure_avg == 0.0, "averageTest: ERROR in YZAverages (Pavg)")
      regentlib.assert(YZAverages[int2d{0,0}].pressure_rms == 0.0, "averageTest: ERROR in YZAverages (Prms)")
      for i=1, Npx+1 do
         regentlib.assert(fabs(YZAverages[int2d{i,0}].weight/double(128.0) - 1.0) < 1e-9, "averageTest: ERROR in YZAverages")
         regentlib.assert(fabs(YZAverages[int2d{i,0}].pressure_avg/double([P]/[PRef]*128.0) - 1.0) < 1e-9, "averageTest: ERROR in YZAverages (Pavg)")
         regentlib.assert(fabs(YZAverages[int2d{i,0}].pressure_rms/double([P]/[PRef]*[P]/[PRef]*128.0) - 1.0) < 1e-9, "averageTest: ERROR in YZAverages (Prms)")
      end
      regentlib.assert(YZAverages[int2d{Npx+1,0}].weight == 0.0, "averageTest: ERROR in YZAverages")
      regentlib.assert(YZAverages[int2d{Npx+1,0}].pressure_avg == 0.0, "averageTest: ERROR in YZAverages (Pavg)")
      regentlib.assert(YZAverages[int2d{Npx+1,0}].pressure_rms == 0.0, "averageTest: ERROR in YZAverages (Prms)")

      regentlib.assert(XZAverages[int2d{0,0}].weight == 0.0, "averageTest: ERROR in XZAverages")
      regentlib.assert(XZAverages[int2d{0,0}].pressure_avg == 0.0, "averageTest: ERROR in XZAverages (Pavg)")
      regentlib.assert(XZAverages[int2d{0,0}].pressure_rms == 0.0, "averageTest: ERROR in XZAverages (Prms)")
      for i=1, Npy+1 do
         regentlib.assert(fabs(XZAverages[int2d{i,0}].weight/double(i*Npz) - 1.0) < 1e-9, "averageTest: ERROR in XZAverages")
         regentlib.assert(fabs(XZAverages[int2d{i,0}].pressure_avg/double([P]/[PRef]*i*Npz) - 1.0) < 1e-9, "averageTest: ERROR in XZAverages (Pavg)")
         regentlib.assert(fabs(XZAverages[int2d{i,0}].pressure_rms/double([P]/[PRef]*[P]/[PRef]*i*Npz) - 1.0) < 1e-9, "averageTest: ERROR in XZAverages (Prms)")
      end
      regentlib.assert(XZAverages[int2d{Npy+1,0}].weight == 0.0, "averageTest: ERROR in XZAverages")
      regentlib.assert(XZAverages[int2d{Npy+1,0}].pressure_avg == 0.0, "averageTest: ERROR in XZAverages (Pavg)")
      regentlib.assert(XZAverages[int2d{Npy+1,0}].pressure_rms == 0.0, "averageTest: ERROR in XZAverages (Prms)")

      regentlib.assert(XYAverages[int2d{0,0}].weight == 0.0, "averageTest: ERROR in XYAverages")
      regentlib.assert(XYAverages[int2d{0,0}].pressure_avg == 0.0, "averageTest: ERROR in XYAverages (Pavg)")
      regentlib.assert(XYAverages[int2d{0,0}].pressure_rms == 0.0, "averageTest: ERROR in XYAverages (Prms)")
      for i=1, Npz+1 do
         regentlib.assert(fabs(XYAverages[int2d{i,0}].weight/double(136.0) - 1.0) < 1e-9, "averageTest: ERROR in XYAverages")
         regentlib.assert(fabs(XYAverages[int2d{i,0}].pressure_avg/double([P]/[PRef]*136.0) - 1.0) < 1e-9, "averageTest: ERROR in XYAverages (Pavg)")
         regentlib.assert(fabs(XYAverages[int2d{i,0}].pressure_rms/double([P]/[PRef]*[P]/[PRef]*136.0) - 1.0) < 1e-9, "averageTest: ERROR in XYAverages (Prms)")
      end
      regentlib.assert(XYAverages[int2d{Npz+1,0}].weight == 0.0, "averageTest: ERROR in XYAverages")
      regentlib.assert(XYAverages[int2d{Npz+1,0}].pressure_avg == 0.0, "averageTest: ERROR in XYAverages (Pavg)")
      regentlib.assert(XYAverages[int2d{Npz+1,0}].pressure_rms == 0.0, "averageTest: ERROR in XYAverages (Prms)")
   end
end

local Averages = AVG.mkAvgList()

local MAPPER = {
   SAMPLE_ID_TAG = 1234
}

local Grid = {
   xBnum = regentlib.newsymbol(),
   yBnum = regentlib.newsymbol(),
   zBnum = regentlib.newsymbol(),
   NX = regentlib.newsymbol(),
   NY = regentlib.newsymbol(),
   NZ = regentlib.newsymbol(),
   numTiles = regentlib.newsymbol(),
   NXout = regentlib.newsymbol(),
   NYout = regentlib.newsymbol(),
   NZout = regentlib.newsymbol(),
   numTilesOut = regentlib.newsymbol(),
}

task zero() return 0.0 end

task main()
   -- Init config
   var config : Config

   config.Flow.initCase.type = SCHEMA.FlowInitCase_Restart
   format.snprint([&int8](config.Flow.initCase.u.Restart.restartDir), 256, ".")

   config.Grid.xNum = Npx
   config.Grid.yNum = Npy
   config.Grid.zNum = Npz

   config.IO.YZAverages.length = Nrx
   config.IO.XZAverages.length = Nry
   config.IO.XYAverages.length = Nrz

   config.IO.YZAverages.values[0].fromCell = [fromCellx]
   config.IO.YZAverages.values[0].uptoCell = [uptoCellx]
   config.IO.XZAverages.values[0].fromCell = [fromCelly]
   config.IO.XZAverages.values[0].uptoCell = [uptoCelly]
   config.IO.XYAverages.values[0].fromCell = [fromCellz]
   config.IO.XYAverages.values[0].uptoCell = [uptoCellz]

   config.IO.XAverages.length = 0
   config.IO.YAverages.length = 0
   config.IO.ZAverages.length = 0

   -- Init the mixture
   config.Flow.mixture.type = SCHEMA.MixtureModel_AirMix
   config.Flow.mixture.u.AirMix.LRef = [LRef]
   config.Flow.mixture.u.AirMix.TRef = [TRef]
   config.Flow.mixture.u.AirMix.PRef = [PRef]
   config.Flow.mixture.u.AirMix.XiRef.Species.length = 2
   C.snprintf([&int8](config.Flow.mixture.u.AirMix.XiRef.Species.values[0].Name), 10, "O2")
   C.snprintf([&int8](config.Flow.mixture.u.AirMix.XiRef.Species.values[1].Name), 10, "N2")
   config.Flow.mixture.u.AirMix.XiRef.Species.values[0].MolarFrac = [MixWRef]*[YO2Ref]/(2*15.999e-3)
   config.Flow.mixture.u.AirMix.XiRef.Species.values[1].MolarFrac = [MixWRef]*[YN2Ref]/28.0134e-3

   -- Define the domain
   var [Grid.xBnum] = 1
   var [Grid.yBnum] = 1
   var [Grid.zBnum] = 1

   var [Grid.NX] = Nx
   var [Grid.NY] = Ny
   var [Grid.NZ] = Nz
   var [Grid.numTiles] = Grid.NX * Grid.NY * Grid.NZ

   var [Grid.NXout] = 1
   var [Grid.NYout] = 1
   var [Grid.NZout] = 1
   var [Grid.numTilesOut] = Grid.NXout * Grid.NYout * Grid.NZout

   -- Define the domain
   var is_Fluid = ispace(int3d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                 y = config.Grid.yNum + 2*Grid.yBnum,
                                 z = config.Grid.zNum + 2*Grid.zBnum})
   var Fluid = region(is_Fluid, Fluid_columns);
   var Fluid_bounds = Fluid.bounds

   -- Partitioning domain
   var tiles = ispace(int3d, {Grid.NX, Grid.NY, Grid.NZ})

   -- Fluid Partitioning
   var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, Grid.xBnum, Grid.yBnum, Grid.zBnum)
   var Fluid_Ghost = PART.PartitionGhost(Fluid, tiles, Fluid_Zones, config)
   var {p_All} = Fluid_Zones

   -- Define mixture
   var Mix = MIX.InitMixture(Fluid, tiles, p_All, config);

   [AVG.DeclSymbols(Averages, Grid, Fluid, p_All, config, MAPPER)];

   InitializeCell(Fluid)

   InitGeometry(Fluid);

   -- Initialize averages partitions
   [AVG.InitPartitions(Averages, Grid, Fluid, p_All, config)];

   -- Initialize averages
   [AVG.InitRakesAndPlanes(Averages)];

   var dt = double(0.1)
   dt += zero() -- so it becomes a future

   for i=0, 10 do
      [AVG.AddAverages(Averages, Fluid_bounds, dt, config, Mix)];
   end

   var SpeciesNames = MIX.GetSpeciesNames(Mix)
   var dirname = [&int8](C.malloc(256))
   format.snprint(dirname, 256, '.');

   [AVG.WriteAverages(0, Averages,   tiles, dirname, IO, SpeciesNames, config)];
   [checkAverages(Averages.YZAverages, Averages.XZAverages, Averages.XYAverages)];

   __fence(__execution, __block)

   [AVG.ReadAverages(Averages, config)];
   [checkAverages(Averages.YZAverages, Averages.XZAverages, Averages.XYAverages)];

   C.free(dirname);

   __fence(__execution, __block)

   format.println("averageTest: TEST OK!")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "averageTest.o", "object", REGISTRAR.register_tasks)
