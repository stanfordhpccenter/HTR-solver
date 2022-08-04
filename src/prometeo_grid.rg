-- Copyright (c) "2019, by Stanford University
--               Developer: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
--                         HTR solver: An open-source exascale-oriented task-based
--                         multi-GPU high-order code for hypersonic aerothermodynamics.
--                         Computer Physics Communications 255, 107262"
-- All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--    * Redistributions of source code must retain the above copyright
--      notice, this list of conditions and the following disclaimer.
--    * Redistributions in binary form must reproduce the above copyright
--      notice, this list of conditions and the following disclaimer in the
--      documentation and/or other materials provided with the distribution.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
-- ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import "regent"

return function(SCHEMA, IO, Fluid_columns, bBoxType,
                zones_partitions, output_partitions) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local cos  = regentlib.cos(double)
local sinh = regentlib.sinh(double)
local cosh = regentlib.cosh(double)
local tanh = regentlib.tanh(double)

local MAPPER = terralib.includec("prometeo_mapper.h")

local UTIL = require "util"
local MACRO = require 'prometeo_macro'
local VERSION = require "version"

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"

local PI = CONST.PI

-------------------------------------------------------------------------------
-- DATA TYPES
-------------------------------------------------------------------------------

local struct nodeType {
   position : double;
}

-------------------------------------------------------------------------------
-- EXTERNAL MODULES IMPORTS
-------------------------------------------------------------------------------
local NGridVars = terralib.newlist({
   'position',
})

local CCGridVars = terralib.newlist({
   'centerCoordinates',
})

local HDF_N = (require 'hdf_helper')(int1d, int1d, nodeType,
                                                   NGridVars,
                                                   {Bnum=int,
                                                    NegStaggered=bool, PosStaggered=bool,
                                                    boundingBox=bBoxType},
                                                   {Versions={2, VERSION.Length}})

local HDF_C = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                                   CCGridVars,
                                                   {},
                                                   {Versions={2, VERSION.Length}})

-------------------------------------------------------------------------------
-- MESH ROUTINES
-------------------------------------------------------------------------------
-- Description:
--     Linear interpolation, given the line defined by the points
--     (x=alpha, y=a) and (x=beta, y=b) find the y location of the
--     point on the line (x=xi, y=?)
-- Input:
--     xi = location on x axis
--     alpha = lower point on x axis
--     beta =  upper point on x axis
--     a = lower point on y axis
--     b = upper point on y axis
-- Output:
--     y location on line at x=xi
local __demand(__inline)
task linear_interpolation(xi : double,
                          alpha : double,
                          beta  : double,
                          a     : double,
                          b     : double) : double
   return (b-a)/(beta-alpha)*(xi-alpha) + a
end


-- Description:
--     Generate the cell width of a nonuniform mesh
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum
--     Nx = number of cells between x_min and x_max
-- Output:
--     width of the non-uniform mesh cell
local __demand(__inline)
task uniform_cell_width(x_min : double,
                        x_max : double,
                        Nx    : uint64) : double
   return (x_max-x_min)/Nx
end

-- Description:
--     non-linear map point (x) on the interval (x_min, x_max) using
--     a cosine
-- Input:
--     x = location on uniform mesh
--     x_min = domain minimum
--     x_max = domain maximum
-- Output:
--     x location on a non-uniform mesh
local __demand(__inline)
task transform_uniform_to_nonuniform(x : double,
                                     x_min : double,
                                     x_max : double,
                                     Grid_Type : SCHEMA.GridTypes) : double
   var transformed : double
   if (Grid_Type.type == SCHEMA.GridTypes_Uniform) then
      transformed = x
   elseif (Grid_Type.type == SCHEMA.GridTypes_Cosine) then
      -- map x onto the interval -1 to 1
      var x_scaled_minus1_to_plus1 = linear_interpolation(x, x_min, x_max, -1.0, 1.0)
      -- map non-uniformly onto the interval -1 to 1
      var x_non_uniform_minus1_to_plus1 = -1.0*cos(PI*(x_scaled_minus1_to_plus1+1.0)/2.0)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_plus1, -1.0, 1.0, x_min, x_max)
   elseif (Grid_Type.type == SCHEMA.GridTypes_TanhMinus) then
      var Stretching = Grid_Type.u.TanhMinus.Stretching
      -- map x onto the interval -1 to 0
      var x_scaled_minus1_to_zero = linear_interpolation(x, x_min, x_max, -1.0, 0.0)
      -- map non-uniformly onto the interval -1 to 0
      var x_non_uniform_minus1_to_zero = tanh(Stretching*x_scaled_minus1_to_zero)/tanh(Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_zero, -1.0, 0.0, x_min, x_max)
   elseif (Grid_Type.type == SCHEMA.GridTypes_TanhPlus) then
      var Stretching = Grid_Type.u.TanhPlus.Stretching
      -- map x onto the interval 0 to 1
      var x_scaled_zero_to_plus1 = linear_interpolation(x, x_min, x_max, 0.0, 1.0)
      -- map non-uniformly onto the interval 0 to 1
      var x_non_uniform_zero_to_plus1 = tanh(Stretching*x_scaled_zero_to_plus1)/tanh(Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_zero_to_plus1, 0.0, 1.0, x_min, x_max)
   elseif (Grid_Type.type == SCHEMA.GridTypes_Tanh) then
      var Stretching = Grid_Type.u.Tanh.Stretching
      -- map x onto the interval -1 to 1
      var x_scaled_minus1_to_plus1 = linear_interpolation(x, x_min, x_max, -1.0, 1.0)
      -- map non-uniformly onto the interval -1 to 1
      var x_non_uniform_minus1_to_plus1 = tanh(Stretching*x_scaled_minus1_to_plus1)/tanh(Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_plus1, -1.0, 1.0, x_min, x_max)
   elseif (Grid_Type.type == SCHEMA.GridTypes_SinhMinus) then
      var Stretching = Grid_Type.u.SinhMinus.Stretching
      -- map x onto the interval 0 to 1
      var x_scaled_zero_to_plus1 = linear_interpolation(x, x_min, x_max, 0.0, 1.0)
      -- map non-uniformly onto the interval 0 to 1
      var x_non_uniform_zero_to_plus1 = sinh(Stretching*x_scaled_zero_to_plus1)/sinh(Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_zero_to_plus1, 0.0, 1.0, x_min, x_max)
   elseif (Grid_Type.type == SCHEMA.GridTypes_SinhPlus) then
      var Stretching = Grid_Type.u.SinhPlus.Stretching
      -- map x onto the interval -1 to 0
      var x_scaled_minus1_to_zero = linear_interpolation(x, x_min, x_max, -1.0, 0.0)
      -- map non-uniformly onto the interval -1 to 0
      var x_non_uniform_minus1_to_zero = sinh(Stretching*x_scaled_minus1_to_zero)/sinh(Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_zero, -1.0, 0.0, x_min, x_max)
   elseif (Grid_Type.type == SCHEMA.GridTypes_Sinh) then
      var Stretching = Grid_Type.u.Sinh.Stretching
      -- map x onto the interval -1 to 1
      var x_scaled_minus1_to_plus1 = linear_interpolation(x, x_min, x_max, -1.0, 1.0)
      -- map non-uniformly onto the interval -1 to 1
      var x_non_uniform_minus1_to_plus1 = sinh(Stretching*x_scaled_minus1_to_plus1)/sinh(Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_plus1, -1.0, 1.0, x_min, x_max)
   end
   return  transformed
end

-------------------------------------------------------------------------------
-- NODE MESH TASKS
-------------------------------------------------------------------------------
-- TODO: Regent does not like SCHEMA.GridTypes in its CUDA kernels
--       we need either to recactor this piece of code or ask for an upgrade
local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
task InitializeNodeGrid(nodes : region(ispace(int1d), nodeType),
                        origin : double,
                        width : double,
                        Type : SCHEMA.GridTypes)
where
   writes(nodes)
do
   var xmin = origin
   var xmax = origin+width
   var Nx = nodes.bounds.hi-nodes.bounds.lo
   var dx = uniform_cell_width(xmin, xmax, Nx)
   __demand(__openmp)
   for c in nodes do
      var uniform_node = xmin + double(c)*dx
      nodes[c].position = transform_uniform_to_nonuniform(uniform_node, xmin, xmax, Type)
   end
end

-------------------------------------------------------------------------------
-- CELL CENTER MESH TASKS
-------------------------------------------------------------------------------
local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task InitializeCellCenters(Fluid : region(ispace(int3d), Fluid_columns),
                           Xnode : region(ispace(int1d), nodeType),
                           Ynode : region(ispace(int1d), nodeType),
                           Znode : region(ispace(int1d), nodeType),
                           is_xNeg_Staggered : bool, is_xPos_Staggered : bool,
                           is_yNeg_Staggered : bool, is_yPos_Staggered : bool,
                           is_zNeg_Staggered : bool, is_zPos_Staggered : bool,
                           xBnum : uint64, xNum : uint64,
                           yBnum : uint64, yNum : uint64,
                           zBnum : uint64, zNum : uint64)
where
   reads(Xnode, Ynode, Znode),
   writes(Fluid.centerCoordinates)
do
   -- Find cell center coordinates and cell width of interior cells
   __demand(__openmp)
   for c in Fluid do
      var xNegGhost = MACRO.is_xNegGhost(c, xBnum)
      var xPosGhost = MACRO.is_xPosGhost(c, xBnum, xNum)
      var yNegGhost = MACRO.is_yNegGhost(c, yBnum)
      var yPosGhost = MACRO.is_yPosGhost(c, yBnum, yNum)
      var zNegGhost = MACRO.is_zNegGhost(c, zBnum)
      var zPosGhost = MACRO.is_zPosGhost(c, zBnum, zNum)

      var cc = array(0.0, 0.0, 0.0)
rescape
   local function ComputeCoordinatesExpr(dir)
      local idx
      local NegGhost
      local PosGhost
      local node
      local is_Neg_Staggered
      local is_Pos_Staggered
      local r_idx
      local Bnum
      if     (dir == "x") then
         idx = 0
         NegGhost = xNegGhost
         PosGhost = xPosGhost
         node = Xnode
         is_Neg_Staggered = is_xNeg_Staggered
         is_Pos_Staggered = is_xPos_Staggered
         r_idx = rexpr c.x end
         Bnum = xBnum
      elseif (dir == "y") then
         idx = 1
         NegGhost = yNegGhost
         PosGhost = yPosGhost
         node = Ynode
         is_Neg_Staggered = is_yNeg_Staggered
         is_Pos_Staggered = is_yPos_Staggered
         r_idx = rexpr c.y end
         Bnum = yBnum
      elseif (dir == "z") then
         idx = 2
         NegGhost = zNegGhost
         PosGhost = zPosGhost
         node = Znode
         is_Neg_Staggered = is_zNeg_Staggered
         is_Pos_Staggered = is_zPos_Staggered
         r_idx = rexpr c.z end
         Bnum = zBnum
      else assert(false) end
      return rquote
         if [NegGhost] then
            if [is_Neg_Staggered] then
               -- Staggered point
               cc[idx] = node[r_idx].position
            else
               -- Colocated point
               cc[idx] = 1.5*node[r_idx].position - 0.5*node[r_idx+1].position
            end
         elseif [PosGhost] then
            if [is_Pos_Staggered] then
               -- Staggered point
               cc[idx] = node[r_idx-1].position
            else
               -- Colocated point
               cc[idx] = 1.5*node[r_idx-1].position - 0.5*node[r_idx-2].position
            end
         else
            -- internal point
            var node_idx = [r_idx] - Bnum
            cc[idx] = 0.5*(node[node_idx+1].position + node[node_idx].position)
         end
      end
   end
   remit ComputeCoordinatesExpr("x")
   remit ComputeCoordinatesExpr("y")
   remit ComputeCoordinatesExpr("z")
end
      Fluid[c].centerCoordinates = cc
   end
end

-------------------------------------------------------------------------------
-- INLINED TASKS
-------------------------------------------------------------------------------
-- Workaround to avoid lifting to future bBoxType
local terra strip_future(x : bBoxType) return x end
strip_future.replicable = true

local __demand(__inline)
task isStaggered(BCType : SCHEMA.FlowBCType)
   return ((BCType == SCHEMA.FlowBC_Dirichlet) or
           (BCType == SCHEMA.FlowBC_AdiabaticWall) or
           (BCType == SCHEMA.FlowBC_IsothermalWall) or
           (BCType == SCHEMA.FlowBC_SuctionAndBlowingWall))
end

local __demand(__inline)
task InitializeGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                        tiles : ispace(int3d),
                        Fluid_Zones : zones_partitions(Fluid, tiles),
                        config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var { p_All } = Fluid_Zones

   -- Determine number of ghost cells in each direction
   -- 0 ghost cells if periodic and 1 otherwise
   var xBnum = 1
   var yBnum = 1
   var zBnum = 1
   if config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic then xBnum = 0 end
   if config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic then yBnum = 0 end
   if config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic then zBnum = 0 end

   -- Determine if boundary conditions are staggered or not
   var is_xNeg_Staggered = isStaggered(config.BC.xBCLeft.type )
   var is_xPos_Staggered = isStaggered(config.BC.xBCRight.type)
   var is_yNeg_Staggered = isStaggered(config.BC.yBCLeft.type )
   var is_yPos_Staggered = isStaggered(config.BC.yBCRight.type)
   var is_zNeg_Staggered = isStaggered(config.BC.zBCLeft.type )
   var is_zPos_Staggered = isStaggered(config.BC.zBCRight.type)

   -- Define the bounding box
   var bBox : bBoxType

   -- Create node regions
   var sampleId = config.Mapping.sampleId
   var is_Xnodes = ispace(int1d, config.Grid.xNum + 1)
   var is_Ynodes = ispace(int1d, config.Grid.yNum + 1)
   var is_Znodes = ispace(int1d, config.Grid.zNum + 1)
   var Xnodes = region(is_Xnodes, nodeType);
   var Ynodes = region(is_Ynodes, nodeType);
   var Znodes = region(is_Znodes, nodeType);
   [UTIL.emitRegionTagAttach(Xnodes, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   [UTIL.emitRegionTagAttach(Ynodes, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   [UTIL.emitRegionTagAttach(Znodes, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

   var Xnodes_copy = region(is_Xnodes, nodeType);
   var Ynodes_copy = region(is_Ynodes, nodeType);
   var Znodes_copy = region(is_Znodes, nodeType);
   [UTIL.emitRegionTagAttach(Xnodes_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   [UTIL.emitRegionTagAttach(Ynodes_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   [UTIL.emitRegionTagAttach(Znodes_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

   -- Generate nodes grid partitions
   -- IO (dump everything in one file for now)
   var IO_grid_tiles = ispace(int1d, 1)
   var p_Xnodes_IO      = partition(equal, Xnodes     , IO_grid_tiles)
   var p_Ynodes_IO      = partition(equal, Ynodes     , IO_grid_tiles)
   var p_Znodes_IO      = partition(equal, Znodes     , IO_grid_tiles)
   var p_Xnodes_IO_copy = partition(equal, Xnodes_copy, IO_grid_tiles)
   var p_Ynodes_IO_copy = partition(equal, Ynodes_copy, IO_grid_tiles)
   var p_Znodes_IO_copy = partition(equal, Znodes_copy, IO_grid_tiles)

   -- Initialize the nodes grid
   fill(Xnodes.position, 0.0)
   fill(Ynodes.position, 0.0)
   fill(Znodes.position, 0.0)

   if (config.Grid.GridInput.type == SCHEMA.GridInputStruct_Cartesian) then
      -- Generate a Cartesian grid based on config data
      var gridGen = config.Grid.GridInput.u.Cartesian
      InitializeNodeGrid(Xnodes, gridGen.origin[0], gridGen.width[0], gridGen.xType)
      InitializeNodeGrid(Ynodes, gridGen.origin[1], gridGen.width[1], gridGen.yType)
      InitializeNodeGrid(Znodes, gridGen.origin[2], gridGen.width[2], gridGen.zType)

      -- Initialize the bounding box
      bBox.v0 = array(gridGen.origin[0]                 , gridGen.origin[1]                 , gridGen.origin[2])
      bBox.v1 = array(gridGen.origin[0]+gridGen.width[0], gridGen.origin[1]                 , gridGen.origin[2])
      bBox.v2 = array(gridGen.origin[0]+gridGen.width[0], gridGen.origin[1]+gridGen.width[1], gridGen.origin[2])
      bBox.v3 = array(gridGen.origin[0]                 , gridGen.origin[1]+gridGen.width[1], gridGen.origin[2])
      bBox.v4 = array(gridGen.origin[0]                 , gridGen.origin[1]                 , gridGen.origin[2]+gridGen.width[2])
      bBox.v5 = array(gridGen.origin[0]+gridGen.width[0], gridGen.origin[1]                 , gridGen.origin[2]+gridGen.width[2])
      bBox.v6 = array(gridGen.origin[0]+gridGen.width[0], gridGen.origin[1]+gridGen.width[1], gridGen.origin[2]+gridGen.width[2])
      bBox.v7 = array(gridGen.origin[0]                 , gridGen.origin[1]+gridGen.width[1], gridGen.origin[2]+gridGen.width[2]);

   elseif (config.Grid.GridInput.type == SCHEMA.GridInputStruct_FromFile) then
      -- Read the node location from file
      var gridDir = config.Grid.GridInput.u.FromFile.gridDir
      var dirname = [&int8](C.malloc(256))
      -- X nodes
      C.snprintf(dirname, 256, '%s/xNodes', gridDir)
      bBox = strip_future(HDF_N.read.boundingBox(IO_grid_tiles, dirname, Xnodes, p_Xnodes_IO))
      HDF_N.load(IO_grid_tiles, dirname, Xnodes, Xnodes_copy, p_Xnodes_IO, p_Xnodes_IO_copy)
      -- Y nodes
      C.snprintf(dirname, 256, '%s/yNodes', gridDir)
      HDF_N.load(IO_grid_tiles, dirname, Ynodes, Ynodes_copy, p_Ynodes_IO, p_Ynodes_IO_copy)
      -- Z nodes
      C.snprintf(dirname, 256, '%s/zNodes', gridDir)
      HDF_N.load(IO_grid_tiles, dirname, Znodes, Znodes_copy, p_Znodes_IO, p_Znodes_IO_copy)
      C.free(dirname)

   else regentlib.assert(false, 'Unhandled GridInput type') end

   -- Dump nodes grid
   var dirname = [&int8](C.malloc(256))
   C.snprintf(dirname, 256, '%s/nodes_grid', config.Mapping.outDir)
   var _1 = IO.createDir(0, dirname)
   -- X grid
   C.snprintf(dirname, 256, '%s/nodes_grid/xNodes', config.Mapping.outDir)
   var _2 = IO.createDir(_1, dirname)
   _2 = HDF_N.dump( _2, IO_grid_tiles, dirname, Xnodes, Xnodes_copy, p_Xnodes_IO, p_Xnodes_IO_copy)
   _2 = HDF_N.write.Bnum(        _2, dirname, xBnum)
   _2 = HDF_N.write.NegStaggered(_2, dirname, is_xNeg_Staggered)
   _2 = HDF_N.write.PosStaggered(_2, dirname, is_xPos_Staggered)
   _2 = HDF_N.write.boundingBox( _2, dirname, bBox)
   _2 = HDF_N.write.Versions(    _2, dirname, array(regentlib.string([VERSION.SolverVersion]),
                                                    regentlib.string([VERSION.LegionVersion])))
   -- Y grid
   C.snprintf(dirname, 256, '%s/nodes_grid/yNodes', config.Mapping.outDir)
   var _3 = IO.createDir(_1, dirname)
   _3 = HDF_N.dump( _3, IO_grid_tiles, dirname, Ynodes, Ynodes_copy, p_Ynodes_IO, p_Ynodes_IO_copy)
   _3 = HDF_N.write.Bnum(        _3, dirname, yBnum)
   _3 = HDF_N.write.NegStaggered(_3, dirname, is_yNeg_Staggered)
   _3 = HDF_N.write.PosStaggered(_3, dirname, is_yPos_Staggered)
   _3 = HDF_N.write.boundingBox( _3, dirname, bBox)
   _3 = HDF_N.write.Versions(    _3, dirname, array(regentlib.string([VERSION.SolverVersion]),
                                                    regentlib.string([VERSION.LegionVersion])))

   -- Z grid
   C.snprintf(dirname, 256, '%s/nodes_grid/zNodes', config.Mapping.outDir)
   var _4 = IO.createDir(_1, dirname)
   _4 = HDF_N.dump( _4, IO_grid_tiles, dirname, Znodes, Znodes_copy, p_Znodes_IO, p_Znodes_IO_copy)
   _4 = HDF_N.write.Bnum(        _4, dirname, zBnum)
   _4 = HDF_N.write.NegStaggered(_4, dirname, is_zNeg_Staggered)
   _4 = HDF_N.write.PosStaggered(_4, dirname, is_zPos_Staggered)
   _4 = HDF_N.write.boundingBox( _4, dirname, bBox)
   _4 = HDF_N.write.Versions(    _4, dirname, array(regentlib.string([VERSION.SolverVersion]),
                                                    regentlib.string([VERSION.LegionVersion])))

   C.free(dirname)

   -- Initialize cell centers of standard points
   __demand(__index_launch)
   for c in tiles do
      InitializeCellCenters(p_All[c],
                         Xnodes, Ynodes, Znodes,
                         is_xNeg_Staggered, is_xPos_Staggered,
                         is_yNeg_Staggered, is_yPos_Staggered,
                         is_zNeg_Staggered, is_zPos_Staggered,
                         xBnum, config.Grid.xNum,
                         yBnum, config.Grid.yNum,
                         zBnum, config.Grid.zNum)
   end

   return bBox
end

local __demand(__inline)
task dumpCellCenterGrid(Fluid : region(ispace(int3d), Fluid_columns),
                        Fluid_copy : region(ispace(int3d), Fluid_columns),
                        tiles_output : ispace(int3d),
                        Fluid_Output      : output_partitions(Fluid,      tiles_output),
                        Fluid_Output_copy : output_partitions(Fluid_copy, tiles_output),
                        config : SCHEMA.Config)
where
   reads(Fluid),
   reads writes(Fluid_copy),
   Fluid * Fluid_copy
do
   -- Unpack the partitions that we are going to need
   var {p_Output } = Fluid_Output
   var {p_Output_copy=p_Output} = Fluid_Output_copy

   var dirname = [&int8](C.malloc(256))
   C.snprintf(dirname, 256, '%s/cellCenter_grid', config.Mapping.outDir)
   var _1 = IO.createDir(0, dirname)
   _1 = HDF_C.dump(           _1, tiles_output, dirname, Fluid, Fluid_copy, p_Output, p_Output_copy)
   _1 = HDF_C.write.Versions( _1, dirname, array(regentlib.string([VERSION.SolverVersion]),
                                                 regentlib.string([VERSION.LegionVersion])))
   C.free(dirname)
end

-------------------------------------------------------------------------------
-- EXPORTED TASKS
-------------------------------------------------------------------------------

Exports.InitializeGeometry = InitializeGeometry
Exports.dumpCellCenterGrid = dumpCellCenterGrid

return Exports end

