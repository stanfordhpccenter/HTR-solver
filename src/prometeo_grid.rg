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

return function(SCHEMA, Fluid_columns, zones_partitions) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local cos  = regentlib.cos(double)
local sinh = regentlib.sinh(double)
local cosh = regentlib.cosh(double)
local tanh = regentlib.tanh(double)

local MACRO = require 'prometeo_macro'

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"

local PI = CONST.PI

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
--     Generate the cell center on a uniform mesh
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as left face and 
--               i = Nx-1 has x_max as right face
--               so no ghost cells accounted for here
-- Output:
--     location of cell center
local __demand(__inline)
task uniform_cell_center(x_min : double,
                         x_max : double,
                         Nx    : uint64,
                         i     : uint64) : double
   var dx = uniform_cell_width(x_min, x_max, Nx)
   return x_min + i*dx + dx/2.0
end

-- Description:
--     Generate the location of the face in the negative direction
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as negative direction (left) face and 
--               i = Nx-1 has x_max as positive direction (right) face
--               so no ghost cells accounted for here
-- Output:
--     location of face in the negative direction
local __demand(__inline)
task uniform_cell_neg_face(x_min : double,
                           x_max : double,
                           Nx    : uint64,
                           i     : uint64) : double
   var dx = uniform_cell_width(x_min, x_max, Nx)
   var x_center = uniform_cell_center(x_min, x_max, Nx, i)
   return x_center - 0.5*dx
end

-- Description:
--     Generate the location of the face in the postive direction
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as negative direction (left) face and 
--               i = Nx-1 has x_max as positive direction (right) face
--               so no ghost cells accounted for here
-- Output:
--     location of face in the postive direction
local __demand(__inline)
task uniform_cell_pos_face(x_min : double,
                           x_max : double,
                           Nx    : uint64,
                           i     : uint64) : double
   var dx = uniform_cell_width(x_min, x_max, Nx)
   var x_center = uniform_cell_center(x_min, x_max, Nx, i)
   return x_center + 0.5*dx
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
                                     Grid_Type : SCHEMA.GridType,
                                     Grid_Stretching : double) : double
   var transformed : double
   if (Grid_Type == SCHEMA.GridType_Uniform) then
      transformed = x
   elseif (Grid_Type == SCHEMA.GridType_Cosine) then
      -- map x onto the interval -1 to 1
      var x_scaled_minus1_to_plus1 = linear_interpolation(x, x_min, x_max, -1.0, 1.0)
      -- map non-uniformly onto the interval -1 to 1
      var x_non_uniform_minus1_to_plus1 = -1.0*cos(PI*(x_scaled_minus1_to_plus1+1.0)/2.0)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_plus1, -1.0, 1.0, x_min, x_max)
   elseif (Grid_Type == SCHEMA.GridType_TanhMinus) then
      -- map x onto the interval -1 to 0
      var x_scaled_minus1_to_zero = linear_interpolation(x, x_min, x_max, -1.0, 0.0)
      -- map non-uniformly onto the interval -1 to 0
      var x_non_uniform_minus1_to_zero = tanh(Grid_Stretching*x_scaled_minus1_to_zero)/tanh(Grid_Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_zero, -1.0, 0.0, x_min, x_max)
   elseif (Grid_Type == SCHEMA.GridType_TanhPlus) then
      -- map x onto the interval 0 to 1
      var x_scaled_zero_to_plus1 = linear_interpolation(x, x_min, x_max, 0.0, 1.0)
      -- map non-uniformly onto the interval 0 to 1
      var x_non_uniform_zero_to_plus1 = tanh(Grid_Stretching*x_scaled_zero_to_plus1)/tanh(Grid_Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_zero_to_plus1, 0.0, 1.0, x_min, x_max)
   elseif (Grid_Type == SCHEMA.GridType_Tanh) then
      -- map x onto the interval -1 to 1
      var x_scaled_minus1_to_plus1 = linear_interpolation(x, x_min, x_max, -1.0, 1.0)
      -- map non-uniformly onto the interval -1 to 1
      var x_non_uniform_minus1_to_plus1 = tanh(Grid_Stretching*x_scaled_minus1_to_plus1)/tanh(Grid_Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_plus1, -1.0, 1.0, x_min, x_max)
   elseif (Grid_Type == SCHEMA.GridType_SinhMinus) then
      -- map x onto the interval 0 to 1
      var x_scaled_zero_to_plus1 = linear_interpolation(x, x_min, x_max, 0.0, 1.0)
      -- map non-uniformly onto the interval 0 to 1
      var x_non_uniform_zero_to_plus1 = sinh(Grid_Stretching*x_scaled_zero_to_plus1)/sinh(Grid_Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_zero_to_plus1, 0.0, 1.0, x_min, x_max)
   elseif (Grid_Type == SCHEMA.GridType_SinhPlus) then
      -- map x onto the interval -1 to 0
      var x_scaled_minus1_to_zero = linear_interpolation(x, x_min, x_max, -1.0, 0.0)
      -- map non-uniformly onto the interval -1 to 0
      var x_non_uniform_minus1_to_zero = sinh(Grid_Stretching*x_scaled_minus1_to_zero)/sinh(Grid_Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_zero, -1.0, 0.0, x_min, x_max)
   elseif (Grid_Type == SCHEMA.GridType_Sinh) then
      -- map x onto the interval -1 to 1
      var x_scaled_minus1_to_plus1 = linear_interpolation(x, x_min, x_max, -1.0, 1.0)
      -- map non-uniformly onto the interval -1 to 1
      var x_non_uniform_minus1_to_plus1 = sinh(Grid_Stretching*x_scaled_minus1_to_plus1)/sinh(Grid_Stretching)
      -- map non-uniform sample back to origional interval x_min to x_max
      transformed = linear_interpolation(x_non_uniform_minus1_to_plus1, -1.0, 1.0, x_min, x_max)
   end
   return  transformed
end

-- Description:
--     Generate the location of the face in the negative direction
--     on a non uniform mesh
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as negative direction (left) face and 
--               i = Nx-1 has x_max as positive direction (right) face
--               so no ghost cells accounted for here
-- Output:
--     location of face in the negative direction
local __demand(__inline)
task nonuniform_cell_neg_face(x_min : double,
                              x_max : double,
                              Nx    : uint64,
                              i     : uint64,
                              Grid_Type: SCHEMA.GridType,
                              Grid_Stretching: double) : double
   var x_uniform_neg_face = uniform_cell_neg_face(x_min, x_max, Nx, i)
   return transform_uniform_to_nonuniform(x_uniform_neg_face, x_min, x_max, Grid_Type, Grid_Stretching)
end

-- Description:
--     Generate the location of the face in the postive direction
--     on a non uniform mesh
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as negative direction (left) face and 
--               i = Nx-1 has x_max as positive direction (right) face
--               so no ghost cells accounted for here
-- Output:
--     location of face in the postive direction
local __demand(__inline)
task nonuniform_cell_pos_face(x_min : double,
                              x_max : double,
                              Nx    : uint64,
                              i     : uint64,
                              Grid_Type: SCHEMA.GridType,
                              Grid_Stretching: double) : double
   var x_uniform_pos_face = uniform_cell_pos_face(x_min, x_max, Nx, i)
   return transform_uniform_to_nonuniform(x_uniform_pos_face, x_min, x_max, Grid_Type, Grid_Stretching)
end

-- Description:
--     Generate the cell center of a nonuniform mesh
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as left face and 
--               i = Nx-1 has x_max as right face
--               so no ghost cells accounted for here
-- Output:
--     x location on a non-uniform mesh
local __demand(__inline)
task cell_center(x_min : double,
                 x_max : double,
                 Nx    : uint64,
                 i     : uint64,
                 Grid_Type: SCHEMA.GridType,
                 Grid_Stretching: double) : double
   var x_non_uniform_neg_face = nonuniform_cell_neg_face(x_min, x_max, Nx, i, Grid_Type, Grid_Stretching)
   var x_non_uniform_pos_face = nonuniform_cell_pos_face(x_min, x_max, Nx, i, Grid_Type, Grid_Stretching)
   return 0.5*(x_non_uniform_neg_face + x_non_uniform_pos_face)
end

-- Description:
--     Generate the cell width of a nonuniform mesh
-- Input:
--     x_min = domain minimum
--     x_max = domain maximum 
--     Nx = number of cells between x_min and x_max
--     i  = cell index between x_min and x_max
--         Note: i = 0 has x_min as left face and 
--               i = Nx-1 has x_max as right face
--               so no ghost cells accounted for here
-- Output:
--     width of the non-uniform mesh cell
local __demand(__inline)
task cell_width(x_min : double,
                x_max : double,
                Nx    : uint64,
                i     : uint64,
                Grid_Type: int,
                Grid_Stretching: double) : double
   var x_non_uniform_neg_face = nonuniform_cell_neg_face(x_min, x_max, Nx, i, Grid_Type, Grid_Stretching)
   var x_non_uniform_pos_face = nonuniform_cell_pos_face(x_min, x_max, Nx, i, Grid_Type, Grid_Stretching)
   return x_non_uniform_pos_face - x_non_uniform_neg_face 
end


__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                        Grid_xType : SCHEMA.GridType, Grid_yType : SCHEMA.GridType, Grid_zType : SCHEMA.GridType,
                        Grid_xStretching : double,    Grid_yStretching : double,    Grid_zStretching : double,
                        Grid_xBnum : int32, Grid_xNum : int32, Grid_xOrigin : double, Grid_xWidth : double,
                        Grid_yBnum : int32, Grid_yNum : int32, Grid_yOrigin : double, Grid_yWidth : double,
                        Grid_zBnum : int32, Grid_zNum : int32, Grid_zOrigin : double, Grid_zWidth : double)
where
   reads writes(Fluid.centerCoordinates),
   reads writes(Fluid.cellWidth)
do
   -- Find cell center coordinates and cell width of interior cells
   __demand(__openmp)
   for c in Fluid do
      var xNegGhost = MACRO.is_xNegGhost(c, Grid_xBnum)
      var xPosGhost = MACRO.is_xPosGhost(c, Grid_xBnum, Grid_xNum)
      var yNegGhost = MACRO.is_yNegGhost(c, Grid_yBnum)
      var yPosGhost = MACRO.is_yPosGhost(c, Grid_yBnum, Grid_yNum)
      var zNegGhost = MACRO.is_zNegGhost(c, Grid_zBnum)
      var zPosGhost = MACRO.is_zPosGhost(c, Grid_zBnum, Grid_zNum)

      if not (xNegGhost or xPosGhost) then
         var x_neg_boundary = Grid_xOrigin
         var x_pos_boundary = Grid_xOrigin + Grid_xWidth
         var x_idx_interior = c.x - Grid_xBnum
         Fluid[c].centerCoordinates[0] = cell_center(x_neg_boundary, x_pos_boundary, Grid_xNum, x_idx_interior, Grid_xType, Grid_xStretching)
         Fluid[c].cellWidth[0]         = cell_width( x_neg_boundary, x_pos_boundary, Grid_xNum, x_idx_interior, Grid_xType, Grid_xStretching)
      end

      if not (yNegGhost or yPosGhost) then
         var y_neg_boundary = Grid_yOrigin
         var y_pos_boundary = Grid_yOrigin + Grid_yWidth
         var y_idx_interior = c.y - Grid_yBnum
         Fluid[c].centerCoordinates[1] = cell_center(y_neg_boundary, y_pos_boundary, Grid_yNum, y_idx_interior, Grid_yType, Grid_yStretching)
         Fluid[c].cellWidth[1]         = cell_width( y_neg_boundary, y_pos_boundary, Grid_yNum, y_idx_interior, Grid_yType, Grid_yStretching)
      end

      if not (zNegGhost or zPosGhost) then
         var z_neg_boundary = Grid_zOrigin
         var z_pos_boundary = Grid_zOrigin + Grid_zWidth
         var z_idx_interior = c.z - Grid_zBnum
         Fluid[c].centerCoordinates[2] = cell_center(z_neg_boundary, z_pos_boundary, Grid_zNum, z_idx_interior, Grid_zType, Grid_zStretching)
         Fluid[c].cellWidth[2]         = cell_width( z_neg_boundary, z_pos_boundary, Grid_zNum, z_idx_interior, Grid_zType, Grid_zStretching)
      end
   end
end

-- NOTE: It is safe to not pass the ghost regions to this task, because we
-- always group ghost cells with their neighboring interior cells.
local mkInitializeGhostGeometry = terralib.memoize(function(sdir)
   local InitializeGhostGeometry

   local ind
   local sign
   if     sdir == "xNeg" then
      ind = 0
      sign = -1
   elseif sdir == "xPos" then
      ind = 0
      sign = 1
   elseif sdir == "yNeg" then
      ind = 1
      sign = -1
   elseif sdir == "yPos" then
      ind = 1
      sign = 1
   elseif sdir == "zNeg" then
      ind = 2
      sign = -1
   elseif sdir == "zPos" then
      ind = 2
      sign = 1
   else assert(false) end

   local mk_cint
   if sdir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif sdir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif sdir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif sdir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif sdir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif sdir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task InitializeGhostGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                                Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                BCType : int32)

   where
      reads writes(Fluid.centerCoordinates),
      reads writes(Fluid.cellWidth)
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      var isStaggered = ((BCType == SCHEMA.FlowBC_Dirichlet) or
                         (BCType == SCHEMA.FlowBC_AdiabaticWall) or
                         (BCType == SCHEMA.FlowBC_IsothermalWall) or
                         (BCType == SCHEMA.FlowBC_SuctionAndBlowingWall))

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];
         if isStaggered then
            -- Staggered BCs
            BC[c].centerCoordinates[ind] = BCst[c_int].centerCoordinates[ind]
                                           + 0.5*[sign]*BCst[c_int].cellWidth[ind]
            BC[c].cellWidth[ind] = 1e-12
         else
            BC[c].centerCoordinates[ind] = BCst[c_int].centerCoordinates[ind]
                                           + [sign]*BCst[c_int].cellWidth[ind]
            BC[c].cellWidth[ind] = BCst[c_int].cellWidth[ind]
         end
      end
   end
   return InitializeGhostGeometry
end)

__demand(__inline)
task Exports.InitializeGhostGeometry(Fluid : region(ispace(int3d), Fluid_columns),
                                     tiles : ispace(int3d),
                                     Fluid_Zones : zones_partitions(Fluid, tiles),
                                     config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All,
        p_AllxNeg, p_AllxPos, p_AllyNeg, p_AllyPos, p_AllzNeg, p_AllzPos} = Fluid_Zones

   __demand(__index_launch)
   for c in tiles do [mkInitializeGhostGeometry("xNeg")](p_All[c], p_AllxNeg[c], config.BC.xBCLeft.type ) end
   __demand(__index_launch)
   for c in tiles do [mkInitializeGhostGeometry("xPos")](p_All[c], p_AllxPos[c], config.BC.xBCRight.type) end
   __demand(__index_launch)
   for c in tiles do [mkInitializeGhostGeometry("yNeg")](p_All[c], p_AllyNeg[c], config.BC.yBCLeft.type ) end
   __demand(__index_launch)
   for c in tiles do [mkInitializeGhostGeometry("yPos")](p_All[c], p_AllyPos[c], config.BC.yBCRight.type) end
   __demand(__index_launch)
   for c in tiles do [mkInitializeGhostGeometry("zNeg")](p_All[c], p_AllzNeg[c], config.BC.zBCLeft.type ) end
   __demand(__index_launch)
   for c in tiles do [mkInitializeGhostGeometry("zPos")](p_All[c], p_AllzPos[c], config.BC.zBCRight.type) end
end

return Exports end

