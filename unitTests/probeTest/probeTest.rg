import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

   local C = regentlib.c
local fabs = regentlib.fabs(double)
   local SCHEMA = terralib.includec("config_schema.h")
   local UTIL = require 'util'

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
   if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
   end
   local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
   local Fluid_columns = TYPES.Fluid_columns
   local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)
   local nSpec = MIX.nSpec
   local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

   --External modules
   local IO = (require 'prometeo_IO')(SCHEMA)
   local PROBES = (require 'prometeo_probe')(SCHEMA, MIX, IO, Fluid_columns)

   -- Test parameters
   local Npx = 16
   local Npy = 16
   local Npz = 16
   local Nx = 2
   local Ny = 2
   local Nz = 2

   local Nsam = 1
   local fromCell = rexpr array(  8,   8,   8) end
   local uptoCell = fromCell

   --local R = rexpr 8.3144598 end
   local P = rexpr 101325.0 end
   local T = rexpr 5000.0 end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.dcsi_d, 0.0)
   fill(Fluid.deta_d, 0.0)
   fill(Fluid.dzet_d, 0.0)
   fill(Fluid.pressure, [P])
   fill(Fluid.temperature, [T])
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

local function checkProbe(s)
   return rquote
      regentlib.assert(s.Volumes[0] == double(fromCell[2]), "averageTest: ERROR in probe volume")
   end
end

local Probes = PROBES.mkProbesList()

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
   -- Init config
   var config : Config

   config.Grid.xNum = Npx
   config.Grid.yNum = Npy
   config.Grid.zNum = Npz

   config.IO.probesSamplingInterval = 1
   config.IO.probes.length = Nsam
   config.IO.probes.values[0].fromCell = [fromCell]
   config.IO.probes.values[0].uptoCell = [uptoCell]

   C.snprintf([&int8](config.Mapping.outDir), 256, '.')

   -- Define the domain
   var [Grid.xBnum] = 1
   var [Grid.yBnum] = 1
   var [Grid.zBnum] = 1

   var [Grid.NX] = Nx
   var [Grid.NY] = Ny
   var [Grid.NZ] = Nz
   var [Grid.numTiles] = Grid.NX * Grid.NY * Grid.NZ

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

   ---------------------------------------------------------------------------
   -- Create probes
   ---------------------------------------------------------------------------
   [PROBES.DeclSymbols(Probes, Grid, Fluid, p_Fluid, config)];

   InitializeCell(Fluid)

   InitGeometry(Fluid);

   ---------------------------------------------------------------------------
   -- Initialize probes
   ---------------------------------------------------------------------------
   [PROBES.InitProbes(Probes, config)];

   for i=0, 10 do
      -- Write probe files
      [PROBES.WriteProbes(Probes, 1, rexpr double(0.1) end, config)];
   end

   [checkProbe(Probes)];

   __fence(__execution, __block)

   C.printf("probeTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "probeTest.o", "object")
