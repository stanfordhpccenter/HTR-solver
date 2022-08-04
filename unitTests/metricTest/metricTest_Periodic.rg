import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local REGISTRAR = terralib.includec("prometeo_metric.h")
local UTIL = require 'util'
local CONST = require "prometeo_const"

-- Reference solution
local r_e = terralib.newlist({8.568533992260184e+01, 3.608187334579468e+01, 2.163728324750048e+01, 1.613384956141754e+01, 1.324070814947565e+01, 1.160557775759335e+01, 1.069576171814133e+01, 1.028472959356542e+01, 1.028472959356546e+01, 1.069576171814136e+01, 1.160557775759337e+01, 1.324070814947574e+01, 1.613384956141772e+01, 2.163728324750069e+01, 3.608187334579547e+01, 8.568533992260457e+01})
local r_d = terralib.newlist({6.983855536601261e+01, 3.548676869874529e+01, 2.185264426707614e+01, 1.623797958722048e+01, 1.332616545314058e+01, 1.168048171072343e+01, 1.076479359670524e+01, 1.035110861574966e+01, 1.035110861574967e+01, 1.076479359670527e+01, 1.168048171072347e+01, 1.332616545314066e+01, 1.623797958722064e+01, 2.185264426707641e+01, 3.548676869874606e+01, 6.983855536601438e+01})
local r_s = terralib.newlist({5.254828473817612e+01, 2.678888324902875e+01, 1.845250371901093e+01, 1.449803914164615e+01, 1.232956880116953e+01, 1.109631876257714e+01, 1.045250371901098e+01, 1.025166179096599e+01, 1.045250371901100e+01, 1.109631876257717e+01, 1.232956880116959e+01, 1.449803914164627e+01, 1.845250371901114e+01, 2.678888324902913e+01, 5.254828473817803e+01, 1.040868689198210e+02})
local Ref_e = terralib.global(`arrayof(double, [r_e]))
local Ref_d = terralib.global(`arrayof(double, [r_d]))
local Ref_s = terralib.global(`arrayof(double, [r_s]))

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
local bBoxType = TYPES.bBoxType

--External modules
local MACRO = require "prometeo_macro"
local IO = (require 'prometeo_IO')(SCHEMA)
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)
local GRID = (require 'prometeo_grid')(SCHEMA, IO, Fluid_columns, bBoxType,
                                       PART.zones_partitions, PART.output_partitions)

-- Test parameters
local Npx = 16
local Npy = 16
local Npz = 16
local Nx = 2
local Ny = 2
local Nz = 2
local xO = 0.0
local yO = 0.0
local zO = 0.0
local xW = 1.0
local yW = 1.0
local zW = 1.0

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
end

__demand(__inline)
task checkMetric(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_e, deta_e, dzet_e}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   reads(Fluid.{dcsi_s, deta_s, dzet_s})
do
   for c in Fluid do
      -- Check x-direction
      regentlib.assert(fabs(Fluid[c].dcsi_e/Ref_e[c.x] - 1.0) < 1e-12, "metricTest: error in Internal Metric on dcsi_e")
      regentlib.assert(fabs(Fluid[c].dcsi_d/Ref_d[c.x] - 1.0) < 1e-12, "metricTest: error in Internal Metric on dcsi_d")
      regentlib.assert(fabs(Fluid[c].dcsi_s/Ref_s[c.x] - 1.0) < 1e-12, "metricTest: error in Internal Metric on dcsi_s")
      -- Check y-direction
      regentlib.assert(fabs(Fluid[c].deta_e/Ref_e[c.y] - 1.0) < 1e-12, "metricTest: error in Internal Metric on deta_e")
      regentlib.assert(fabs(Fluid[c].deta_d/Ref_d[c.y] - 1.0) < 1e-12, "metricTest: error in Internal Metric on deta_d")
      regentlib.assert(fabs(Fluid[c].deta_s/Ref_s[c.y] - 1.0) < 1e-12, "metricTest: error in Internal Metric on deta_s")
      -- Check z-direction
      regentlib.assert(fabs(Fluid[c].dzet_e/Ref_e[c.z] - 1.0) < 1e-12, "metricTest: error in Internal Metric on dzet_e")
      regentlib.assert(fabs(Fluid[c].dzet_d/Ref_d[c.z] - 1.0) < 1e-12, "metricTest: error in Internal Metric on dzet_d")
      regentlib.assert(fabs(Fluid[c].dzet_s/Ref_s[c.z] - 1.0) < 1e-12, "metricTest: error in Internal Metric on dzet_s")
   end
end

task main()

   C.printf("metricTest_Periodic: run...\n")

   var config : SCHEMA.Config

   C.snprintf([&int8](config.Mapping.outDir), 256, "./PeriodicDir")
   UTIL.createDir(config.Mapping.outDir)

   config.BC.xBCLeft.type  = SCHEMA.FlowBC_Periodic
   config.BC.xBCRight.type = SCHEMA.FlowBC_Periodic
   config.BC.yBCLeft.type  = SCHEMA.FlowBC_Periodic
   config.BC.yBCRight.type = SCHEMA.FlowBC_Periodic
   config.BC.zBCLeft.type  = SCHEMA.FlowBC_Periodic
   config.BC.zBCRight.type = SCHEMA.FlowBC_Periodic

   config.Grid.xNum = Npx
   config.Grid.yNum = Npy
   config.Grid.zNum = Npz

   config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
   config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
   config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
   config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Cosine
   config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Cosine
   config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Cosine

   -- No ghost cells
   var xBnum = 0
   var yBnum = 0
   var zBnum = 0

   -- Define the domain
   var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                 y = Npy + 2*yBnum,
                                 z = Npz + 2*zBnum})
   var Fluid = region(is_Fluid, Fluid_columns)

   -- Partitioning domain
   var tiles = ispace(int3d, {Nx, Ny, Nz})

   -- Fluid Partitioning
   var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, xBnum, yBnum, zBnum)
   var Fluid_Ghost = PART.PartitionGhost(Fluid, tiles, Fluid_Zones, config)

   InitializeCell(Fluid)

   var boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)

   METRIC.InitializeOperators(Fluid, tiles, Fluid_Zones, config,
                              xBnum, yBnum, zBnum)

   METRIC.InitializeMetric(Fluid, tiles, Fluid_Zones, Fluid_Ghost,
                           boundingBox, config)

   checkMetric(Fluid)

   __fence(__execution, __block)

   C.printf("metricTest_Periodic: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "metricTest_Periodic.o", "object", REGISTRAR.register_metric_tasks)
