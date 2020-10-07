import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local UTIL = require "util-desugared"
local format = require "std/format"

local Config = SCHEMA.Config

local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local MIX = (require "AirMix")(SCHEMA)
local nSpec = MIX.nSpec
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

local struct Fluid_columns {
   -- Grid point
   centerCoordinates : double[3];
   cellWidth : double[3];
   -- Primitive variables
   pressure    : double;
   temperature : double;
   MassFracs   : double[nSpec];
   MolarFracs  : double[nSpec];
   velocity    : double[3];
   -- Properties
   rho  : double;
   mu   : double;
   lam  : double;
   Di   : double[nSpec];
   SoS  : double;
   -- Gradients
   velocityGradientX   : double[3];
   velocityGradientY   : double[3];
   velocityGradientZ   : double[3];
   temperatureGradient : double[3];
   -- Conserved varaibles
   Conserved       : double[nEq];
}

--External modules
local IO = (require 'prometeo_IO')(SCHEMA)
local AVG = (require 'prometeo_average')(SCHEMA, MIX, Fluid_columns)

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

--local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 5000.0 end
local Xi = rexpr array(0.4, 0.2, 0.15, 0.15, 0.1) end
local v  = rexpr array(1.0, 2.0, 3.0) end
local Tres = rexpr 500.0 end

-- Expected properties
local eRho = rexpr 6.2899871101668e-02 end
local eMu  = rexpr 1.2424467023580e-04 end
local eLam = rexpr 2.2727742147267e-01 end
local eDi  = rexpr array(2.5146873480781e-03, 2.4389311883618e-03, 2.4550965167542e-03, 3.6168277168130e-03, 3.9165634179846e-03) end
local eSoS = rexpr 1.4518705966651e+03 end

-- Expected conserved variables
local eConserved  = rexpr array(2.7311049167620e-02, 1.5598263689963e-02, 1.0970170602665e-02, 5.1208217189288e-03, 3.8995659224908e-03, 6.2899871101668e-02, 1.2579974220334e-01, 1.8869961330500e-01, 5.4092397631210e+05) end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
   fill(Fluid.cellWidth, array(0.0, 0.0, 0.0))
   fill(Fluid.pressure, [P])
   fill(Fluid.temperature, [T])
   fill(Fluid.MassFracs,  [Xi])
   fill(Fluid.MolarFracs, [Xi])
   fill(Fluid.velocity, [v])
   fill(Fluid.rho, [eRho])
   fill(Fluid.mu , [eMu])
   fill(Fluid.lam, [eLam])
   fill(Fluid.Di , [eDi])
   fill(Fluid.SoS, [eSoS])
   fill(Fluid.velocityGradientX,   array(0.0, 0.0, 0.0))
   fill(Fluid.velocityGradientY,   array(0.0, 0.0, 0.0))
   fill(Fluid.velocityGradientZ,   array(0.0, 0.0, 0.0))
   fill(Fluid.temperatureGradient, array(0.0, 0.0, 0.0))
   fill(Fluid.Conserved, [eConserved])
end

__demand(__inline)
task InitGeometry(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid.cellWidth)
do
   for c in Fluid do
      Fluid[c].cellWidth = array(1.0, double(c.y), 1.0)
   end
end

local function checkAverages(YZAverages, XZAverages, XYAverages)
   return rquote
      regentlib.assert(YZAverages[int2d{0,0}].weight == 0.0, "averageTest: ERROR in YZAverages")
      for i=1, Npx+1 do
         regentlib.assert(fabs(YZAverages[int2d{i,0}].weight/double(128.0) - 1.0) < 1e-9, "averageTest: ERROR in YZAverages")
      end
      regentlib.assert(YZAverages[int2d{Npx+1,0}].weight == 0.0, "averageTest: ERROR in YZAverages")

      regentlib.assert(XZAverages[int2d{0,0}].weight == 0.0, "averageTest: ERROR in XZAverages")
      for i=1, Npy+1 do
         regentlib.assert(fabs(XZAverages[int2d{i,0}].weight/double(i*Npz) - 1.0) < 1e-9, "averageTest: ERROR in XZAverages")
      end
      regentlib.assert(XZAverages[int2d{Npy+1,0}].weight == 0.0, "averageTest: ERROR in XZAverages")

      regentlib.assert(XYAverages[int2d{0,0}].weight == 0.0, "averageTest: ERROR in XYAverages")
      for i=1, Npz+1 do
         regentlib.assert(fabs(XYAverages[int2d{i,0}].weight/double(136.0) - 1.0) < 1e-9, "averageTest: ERROR in XYAverages")
      end
      regentlib.assert(XYAverages[int2d{Npz+1,0}].weight == 0.0, "averageTest: ERROR in XYAverages")
   end
end

local Averages = AVG.AvgList

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
   var Mix = MIX.InitMixture(config)

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

   -- Partitioning domain
   var tiles = ispace(int3d, {Grid.NX, Grid.NY, Grid.NZ})

   -- Fluid Partitioning
   var p_Fluid =
      [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_All")]
      (Fluid, tiles, int3d{Grid.xBnum,Grid.yBnum,Grid.zBnum}, int3d{0,0,0});

   [AVG.DeclSymbols(Averages,   Grid, Fluid, p_Fluid, config, MAPPER)];

   InitializeCell(Fluid)

   InitGeometry(Fluid);

   -- Initialize averages
   [AVG.InitRakesAndPlanes(Averages)];

   for i=0, 10 do
      [AVG.AddAverages(Averages, rexpr double(0.1) end, config, Mix)];
   end

   var SpeciesNames = MIX.GetSpeciesNames(Mix)
   var dirname = [&int8](C.malloc(256))
   format.snprint(dirname, 256, '.');

   [AVG.WriteAverages(Averages,   tiles, dirname, IO, SpeciesNames, config)];
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

regentlib.saveobj(main, "averageTest.o", "object")
