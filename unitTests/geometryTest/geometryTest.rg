import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local UTIL = require 'util'

local Config = SCHEMA.Config

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
local Npx = 32
local Npy = 32
local Npz = 32
local Nx = 2
local Ny = 2
local Nz = 2
local xO = 1.0
local yO = 1.0
local zO = 1.0
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

function runPeriodic()

   local task checkGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                      config : SCHEMA.Config,
                      Grid_xBnum : int32,
                      Grid_yBnum : int32,
                      Grid_zBnum : int32)
   where
   reads(Fluid.centerCoordinates)
   do
      var Cart = config.Grid.GridInput.u.Cartesian
      regentlib.assert(config.Grid.GridInput.type == SCHEMA.GridInputStruct_Cartesian, "geometryTest: only Cartesian supported for now")
      regentlib.assert(Cart.xType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      regentlib.assert(Cart.yType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      regentlib.assert(Cart.zType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      var Grid_xNum    = config.Grid.xNum
      var Grid_yNum    = config.Grid.yNum
      var Grid_zNum    = config.Grid.zNum
      var Grid_xOrigin = Cart.origin[0]
      var Grid_yOrigin = Cart.origin[1]
      var Grid_zOrigin = Cart.origin[2]
      var Grid_xWidth  = Cart.width[0]
      var Grid_yWidth  = Cart.width[1]
      var Grid_zWidth  = Cart.width[2]

      for c in Fluid do
         var  x_exp = (c.x+0.5)/Grid_xNum*Grid_xWidth + Grid_xOrigin
         var dx_exp = Grid_xWidth/Grid_xNum
         regentlib.assert(fabs((Fluid[c].centerCoordinates[0]/(x_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometry x coordinate")

         var  y_exp = (c.y+0.5)/Grid_yNum*Grid_yWidth + Grid_yOrigin
         var dy_exp = Grid_yWidth/Grid_yNum
         regentlib.assert(fabs((Fluid[c].centerCoordinates[1]/(y_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometry y coordinate")

         var  z_exp = (c.z+0.5)/Grid_zNum*Grid_zWidth + Grid_zOrigin
         var dz_exp = Grid_zWidth/Grid_zNum
         regentlib.assert(fabs((Fluid[c].centerCoordinates[2]/(z_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometry z coordinate")
      end
   end

   return rquote

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
      config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform

      -- No ghost cells
      var xBnum = 0
      var yBnum = 0
      var zBnum = 0

      -- Define the domain
      var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                    y = Npy + 2*yBnum,
                                    z = Npz + 2*zBnum})
      var Fluid = region(is_Fluid, Fluid_columns);

      -- Partitioning domain
      var tiles = ispace(int3d, {Nx, Ny, Nz})

      -- Fluid Partitioning
      var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, xBnum, yBnum, zBnum)
      var {p_All} = Fluid_Zones

      InitializeCell(Fluid)
      var boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)
      checkGeometry(Fluid, config, xBnum, yBnum, zBnum)

      -- Check FromFile capabilities
      __fence(__execution, __block)
      C.printf("geometryTest: Periodic reading from file ...\n")

      config.Grid.GridInput.type = SCHEMA.GridInputStruct_FromFile
      C.snprintf([&int8](config.Grid.GridInput.u.FromFile.gridDir), 256, "./PeriodicDir/nodes_grid")
      C.snprintf([&int8](config.Mapping.outDir), 256, "./PeriodicDir2")
      UTIL.createDir(config.Mapping.outDir)

      InitializeCell(Fluid)
      boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)
      config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
      config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
      config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
      config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform
      checkGeometry(Fluid, config, xBnum, yBnum, zBnum)
   end
end

function runStaggeredBC()

   local task checkGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                      config : SCHEMA.Config,
                      Grid_xBnum : int32,
                      Grid_yBnum : int32,
                      Grid_zBnum : int32)
   where
   reads(Fluid.centerCoordinates)
   do
      var Cart = config.Grid.GridInput.u.Cartesian
      regentlib.assert(config.Grid.GridInput.type == SCHEMA.GridInputStruct_Cartesian, "geometryTest: only Cartesian supported for now")
      regentlib.assert(Cart.xType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      regentlib.assert(Cart.yType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      regentlib.assert(Cart.zType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      var Grid_xNum    = config.Grid.xNum
      var Grid_yNum    = config.Grid.yNum
      var Grid_zNum    = config.Grid.zNum
      var Grid_xOrigin = Cart.origin[0]
      var Grid_yOrigin = Cart.origin[1]
      var Grid_zOrigin = Cart.origin[2]
      var Grid_xWidth  = Cart.width[0]
      var Grid_yWidth  = Cart.width[1]
      var Grid_zWidth  = Cart.width[2]

      for c in Fluid do

         var xNegGhost = MACRO.is_xNegGhost(c, Grid_xBnum)
         var xPosGhost = MACRO.is_xPosGhost(c, Grid_xBnum, Grid_xNum)
         var yNegGhost = MACRO.is_yNegGhost(c, Grid_yBnum)
         var yPosGhost = MACRO.is_yPosGhost(c, Grid_yBnum, Grid_yNum)
         var zNegGhost = MACRO.is_zNegGhost(c, Grid_zBnum)
         var zPosGhost = MACRO.is_zPosGhost(c, Grid_zBnum, Grid_zNum)


         var dx_exp : double
         var  x_exp : double
         var dy_exp : double
         var  y_exp : double
         var dz_exp : double
         var  z_exp : double

         if xNegGhost then
            dx_exp = 1e-12
             x_exp = Grid_xOrigin
         elseif xPosGhost then
            dx_exp = 1e-12
             x_exp = Grid_xWidth + Grid_xOrigin
         else
            dx_exp = Grid_xWidth/Grid_xNum
             x_exp = (c.x-Grid_xBnum+0.5)/Grid_xNum*Grid_xWidth + Grid_xOrigin
         end

         if yNegGhost then
            dy_exp = 1e-12
             y_exp = Grid_yOrigin
         elseif yPosGhost then
            dy_exp = 1e-12
             y_exp = Grid_yWidth + Grid_yOrigin
         else
            dy_exp = Grid_yWidth/Grid_yNum
             y_exp = (c.y-Grid_yBnum+0.5)/Grid_yNum*Grid_yWidth + Grid_yOrigin
         end

         if zNegGhost then
            dz_exp = 1e-12
             z_exp = Grid_zOrigin
         elseif zPosGhost then
            dz_exp = 1e-12
             z_exp = Grid_zWidth + Grid_zOrigin
         else
            dz_exp = Grid_zWidth/Grid_zNum
             z_exp = (c.z-Grid_zBnum+0.5)/Grid_zNum*Grid_zWidth + Grid_zOrigin
         end

         regentlib.assert(fabs((Fluid[c].centerCoordinates[0]/(x_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometryStaggeredBC x coordinate")

         regentlib.assert(fabs((Fluid[c].centerCoordinates[1]/(y_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometryStaggeredBC y coordinate")

         regentlib.assert(fabs((Fluid[c].centerCoordinates[2]/(z_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometryStaggeredBC z coordinate")

      end
   end

   return rquote

      var config : SCHEMA.Config

      C.snprintf([&int8](config.Mapping.outDir), 256, "./StaggeredDir")
      UTIL.createDir(config.Mapping.outDir)

      config.BC.xBCLeft.type  = SCHEMA.FlowBC_Dirichlet
      config.BC.xBCRight.type = SCHEMA.FlowBC_Dirichlet
      config.BC.yBCLeft.type  = SCHEMA.FlowBC_Dirichlet
      config.BC.yBCRight.type = SCHEMA.FlowBC_Dirichlet
      config.BC.zBCLeft.type  = SCHEMA.FlowBC_Dirichlet
      config.BC.zBCRight.type = SCHEMA.FlowBC_Dirichlet

      config.Grid.xNum = Npx
      config.Grid.yNum = Npy
      config.Grid.zNum = Npz

      config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
      config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
      config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
      config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform

      -- No ghost cells
      var xBnum = 1
      var yBnum = 1
      var zBnum = 1

      -- Define the domain
      var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                    y = Npy + 2*yBnum,
                                    z = Npz + 2*zBnum})
      var Fluid = region(is_Fluid, Fluid_columns);

      -- Partitioning domain
      var tiles = ispace(int3d, {Nx, Ny, Nz})

      -- Fluid Partitioning
      var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, xBnum, yBnum, zBnum)
      var {p_All} = Fluid_Zones

      InitializeCell(Fluid)
      var boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)
      checkGeometry(Fluid, config, xBnum, yBnum, zBnum)

      -- Check FromFile capabilities
      __fence(__execution, __block)
      C.printf("geometryTest: Staggered reading from file ...\n")

      config.Grid.GridInput.type = SCHEMA.GridInputStruct_FromFile
      C.snprintf([&int8](config.Grid.GridInput.u.FromFile.gridDir), 256, "./StaggeredDir/nodes_grid")
      C.snprintf([&int8](config.Mapping.outDir), 256, "./StaggeredDir2")
      UTIL.createDir(config.Mapping.outDir)

      InitializeCell(Fluid)
      boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)
      config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
      config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
      config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
      config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform
      checkGeometry(Fluid, config, xBnum, yBnum, zBnum)

   end
end

function runCollocatedBC()

   local task checkGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                      config : SCHEMA.Config,
                      Grid_xBnum : int32,
                      Grid_yBnum : int32,
                      Grid_zBnum : int32)
   where
   reads(Fluid.centerCoordinates)
   do
      var Cart = config.Grid.GridInput.u.Cartesian
      regentlib.assert(config.Grid.GridInput.type == SCHEMA.GridInputStruct_Cartesian, "geometryTest: only Cartesian supported for now")
      regentlib.assert(Cart.xType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      regentlib.assert(Cart.yType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      regentlib.assert(Cart.zType.type == SCHEMA.GridTypes_Uniform, "geometryTest: only Uniform supported for now")
      var Grid_xNum    = config.Grid.xNum
      var Grid_yNum    = config.Grid.yNum
      var Grid_zNum    = config.Grid.zNum
      var Grid_xOrigin = Cart.origin[0]
      var Grid_yOrigin = Cart.origin[1]
      var Grid_zOrigin = Cart.origin[2]
      var Grid_xWidth  = Cart.width[0]
      var Grid_yWidth  = Cart.width[1]
      var Grid_zWidth  = Cart.width[2]

      for c in Fluid do

         var xNegGhost = MACRO.is_xNegGhost(c, Grid_xBnum)
         var xPosGhost = MACRO.is_xPosGhost(c, Grid_xBnum, Grid_xNum)
         var yNegGhost = MACRO.is_yNegGhost(c, Grid_yBnum)
         var yPosGhost = MACRO.is_yPosGhost(c, Grid_yBnum, Grid_yNum)
         var zNegGhost = MACRO.is_zNegGhost(c, Grid_zBnum)
         var zPosGhost = MACRO.is_zPosGhost(c, Grid_zBnum, Grid_zNum)


         var dx_exp : double
         var  x_exp : double
         var dy_exp : double
         var  y_exp : double
         var dz_exp : double
         var  z_exp : double

         if xNegGhost then
            dx_exp = Grid_xWidth/Grid_xNum
             x_exp = Grid_xOrigin - 0.5*dx_exp
         elseif xPosGhost then
            dx_exp = Grid_xWidth/Grid_xNum
             x_exp = Grid_xOrigin + Grid_xWidth + 0.5*dx_exp
         else
            dx_exp = Grid_xWidth/Grid_xNum
             x_exp = (c.x-Grid_xBnum+0.5)/Grid_xNum*Grid_xWidth + Grid_xOrigin
         end

         if yNegGhost then
            dy_exp = Grid_yWidth/Grid_yNum
             y_exp = Grid_yOrigin - 0.5*dy_exp
         elseif yPosGhost then
            dy_exp = Grid_yWidth/Grid_yNum
             y_exp = Grid_yOrigin + Grid_yWidth + 0.5*dy_exp
         else
            dy_exp = Grid_yWidth/Grid_yNum
             y_exp = (c.y-Grid_yBnum+0.5)/Grid_yNum*Grid_yWidth + Grid_yOrigin
         end

         if zNegGhost then
            dz_exp = Grid_zWidth/Grid_zNum
             z_exp = Grid_zOrigin - 0.5*dz_exp
         elseif zPosGhost then
            dz_exp = Grid_zWidth/Grid_zNum
             z_exp = Grid_zOrigin + Grid_zWidth + 0.5*dz_exp
         else
            dz_exp = Grid_zWidth/Grid_zNum
             z_exp = (c.z-Grid_zBnum+0.5)/Grid_zNum*Grid_zWidth + Grid_zOrigin
         end

         regentlib.assert(fabs((Fluid[c].centerCoordinates[0]/(x_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometryCollocatedBC x coordinate")

         regentlib.assert(fabs((Fluid[c].centerCoordinates[1]/(y_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometryCollocatedBC y coordinate")

         regentlib.assert(fabs((Fluid[c].centerCoordinates[2]/(z_exp)) - 1.0) < 1e-3, "geometryTest: ERROR in checkGeometryCollocatedBC z coordinate")

      end
   end

   return rquote

      var config : SCHEMA.Config

      C.snprintf([&int8](config.Mapping.outDir), 256, "./CollocatedDir")
      UTIL.createDir(config.Mapping.outDir)

      config.BC.xBCLeft.type  = SCHEMA.FlowBC_NSCBC_Inflow
      config.BC.xBCRight.type = SCHEMA.FlowBC_NSCBC_Inflow
      config.BC.yBCLeft.type  = SCHEMA.FlowBC_NSCBC_Inflow
      config.BC.yBCRight.type = SCHEMA.FlowBC_NSCBC_Inflow
      config.BC.zBCLeft.type  = SCHEMA.FlowBC_NSCBC_Inflow
      config.BC.zBCRight.type = SCHEMA.FlowBC_NSCBC_Inflow

      config.Grid.xNum = Npx
      config.Grid.yNum = Npy
      config.Grid.zNum = Npz

      config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
      config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
      config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
      config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform

      -- No ghost cells
      var xBnum = 1
      var yBnum = 1
      var zBnum = 1

      -- Define the domain
      var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                    y = Npy + 2*yBnum,
                                    z = Npz + 2*zBnum})
      var Fluid = region(is_Fluid, Fluid_columns);

      -- Partitioning domain
      var tiles = ispace(int3d, {Nx, Ny, Nz})

      -- Fluid Partitioning
      var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, xBnum, yBnum, zBnum)
      var {p_All} = Fluid_Zones

      InitializeCell(Fluid)
      var boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)
      checkGeometry(Fluid, config, xBnum, yBnum, zBnum)

      -- Check FromFile capabilities
      __fence(__execution, __block)
      C.printf("geometryTest: Collocated reading from file ...\n")

      config.Grid.GridInput.type = SCHEMA.GridInputStruct_FromFile
      C.snprintf([&int8](config.Grid.GridInput.u.FromFile.gridDir), 256, "./CollocatedDir/nodes_grid")
      C.snprintf([&int8](config.Mapping.outDir), 256, "./CollocatedDir2")
      UTIL.createDir(config.Mapping.outDir)

      InitializeCell(Fluid)
      boundingBox = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)
      config.Grid.GridInput.type = SCHEMA.GridInputStruct_Cartesian
      config.Grid.GridInput.u.Cartesian.origin = array(double(xO), double(yO), double(zO))
      config.Grid.GridInput.u.Cartesian.width  = array(double(xW), double(yW), double(zW))
      config.Grid.GridInput.u.Cartesian.xType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.yType.type = SCHEMA.GridTypes_Uniform
      config.Grid.GridInput.u.Cartesian.zType.type = SCHEMA.GridTypes_Uniform
      checkGeometry(Fluid, config, xBnum, yBnum, zBnum)
   end
end

task main()
   -- Start with a periodic case
   [runPeriodic()];

   -- Start with a staggered bc case
   [runStaggeredBC()];

   -- Start with a collocated bc case
   [runCollocatedBC()];

   __fence(__execution, __block)

   C.printf("geometryTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "geometryTest.o", "object")
