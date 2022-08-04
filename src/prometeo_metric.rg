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

return function(SCHEMA, TYPES,
                zones_partitions, ghost_partitions) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local MATH = require "math_utils"
local MACRO = require "prometeo_macro"
local CONST = require "prometeo_const"

local pow  = regentlib.pow(double)
local fabs = regentlib.fabs(double)

local COEFFS = terralib.includec("prometeo_metric_coeffs.h")

-- Types
local Fluid_columns = TYPES.Fluid_columns
local bBoxType      = TYPES.bBoxType

-- Node types
local Std_node   = CONST.Std_node
local L_S_node   = CONST.L_S_node
local Lp1_S_node = CONST.Lp1_S_node
local Lp2_S_node = CONST.Lp2_S_node
local Rm3_S_node = CONST.Rm3_S_node
local Rm2_S_node = CONST.Rm2_S_node
local Rm1_S_node = CONST.Rm1_S_node
local R_S_node   = CONST.R_S_node
local L_C_node   = CONST.L_C_node
local Lp1_C_node = CONST.Lp1_C_node
local Rm2_C_node = CONST.Rm2_C_node
local Rm1_C_node = CONST.Rm1_C_node
local R_C_node   = CONST.R_C_node

-- Stencil indices
local Stencil1  = CONST.Stencil1
local Stencil2  = CONST.Stencil2
local Stencil3  = CONST.Stencil3
local Stencil4  = CONST.Stencil4
local nStencils = CONST.nStencils

-------------------------------------------
-- Assemble vectors with coefficients
-- Note: they must be in the same order as the node types
------------------------------------------

-- NOTE: DO NOT USE THESE COEFFICIENTS IN ACTUAL TASKS (unless they are CPU only)
--       THEY ARE HERE UNTIL THE TEST SUITE IS TRANSITIONED TO C++

local Cp           = COEFFS.Cp_cpu
local Recon_Plus   = COEFFS.Recon_Plus_cpu
local Recon_Minus  = COEFFS.Recon_Minus_cpu
local Coeffs_Plus  = COEFFS.Coeffs_Plus_cpu
local Coeffs_Minus = COEFFS.Coeffs_Minus_cpu
local Interp       = COEFFS.Interp_cpu
local Grad         = COEFFS.Grad_cpu
local KennedyOrder = COEFFS.KennedyOrder_cpu
local KennedyCoeff = COEFFS.KennedyCoeff_cpu

-- Helper functions
local function GetCp(dir, c, t, i, b)
   if b == nil then
          if dir=="x" then return rexpr (c + int3d({Cp[t][i],        0,        0})) end
      elseif dir=="y" then return rexpr (c + int3d({       0, Cp[t][i],        0})) end
      elseif dir=="z" then return rexpr (c + int3d({       0,        0, Cp[t][i]})) end
      else assert(0) end
   else
          if dir=="x" then return rexpr (c + int3d({Cp[t][i],        0,        0}))%b end
      elseif dir=="y" then return rexpr (c + int3d({       0, Cp[t][i],        0}))%b end
      elseif dir=="z" then return rexpr (c + int3d({       0,        0, Cp[t][i]}))%b end
      else assert(0) end
   end
end
function Exports.GetCm2(dir, c, t, b) return GetCp(dir, c, t, 0, b) end
function Exports.GetCm1(dir, c, t, b) return GetCp(dir, c, t, 1, b) end
function Exports.GetCp1(dir, c, t, b) return GetCp(dir, c, t, 2, b) end
function Exports.GetCp2(dir, c, t, b) return GetCp(dir, c, t, 3, b) end
function Exports.GetCp3(dir, c, t, b) return GetCp(dir, c, t, 4, b) end

function Exports.GetRecon_Plus( t, St, n) return rexpr Recon_Plus [t][St*6+n] end end
function Exports.GetRecon_Minus(t, St, n) return rexpr Recon_Minus[t][St*6+n] end end

function Exports.GetCoeffs_Plus( t, St) return rexpr Coeffs_Plus [t][St] end end
function Exports.GetCoeffs_Minus(t, St) return rexpr Coeffs_Minus[t][St] end end

function Exports.GetRecon(t, n) return rexpr Recon[t][n] end end

function Exports.GetInterp(t, n) return rexpr Interp[t][n] end end

function Exports.GetGrad(t, n) return rexpr Grad[t][n] end end

function Exports.GetKennedyOrder(t) return rexpr KennedyOrder[t] end end
function Exports.GetKennedyCoeff(t, n) return rexpr KennedyCoeff[t][n] end end

-------------------------------------------------------------------------------
-- OPERATORS ROUTINES
-------------------------------------------------------------------------------

local __demand(__inline)
task InitializeOperators(Fluid : region(ispace(int3d), Fluid_columns),
                                 tiles : ispace(int3d),
                                 p_All : partition(disjoint, Fluid, tiles))
where
   writes(Fluid.{nType_x, nType_y, nType_z})
do
   -- X direction
   -- Flag as in internal point
   __demand(__index_launch)
   for c in tiles do fill((p_All[c]).nType_x, Std_node) end
   -- Y direction
   -- Flag as in internal point
   for c in tiles do fill((p_All[c]).nType_y, Std_node) end
   -- Z direction
   -- Flag as in internal point
   for c in tiles do fill((p_All[c]).nType_z, Std_node) end
end

local mkCorrectGhostOperators = terralib.memoize(function(sdir)
   local CorrectGhostOperators

   local dir
   local nType
   local is_PosGhost
   local is_NegGhost
   local mk_cm2
   local mk_cm1
   local mk_cp1
   local mk_cp2
   local mk_cp3
   if sdir == "x" then
      dir = 0
      nType = "nType_x"
      is_PosGhost = MACRO.is_xPosGhost
      is_NegGhost = MACRO.is_xNegGhost
      mk_cm2 = function(c, b) return rexpr (c+int3d{-2, 0, 0})%b end end
      mk_cm1 = function(c, b) return rexpr (c+int3d{-1, 0, 0})%b end end
      mk_cp1 = function(c, b) return rexpr (c+int3d{ 1, 0, 0})%b end end
      mk_cp2 = function(c, b) return rexpr (c+int3d{ 2, 0, 0})%b end end
      mk_cp3 = function(c, b) return rexpr (c+int3d{ 3, 0, 0})%b end end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      is_PosGhost = MACRO.is_yPosGhost
      is_NegGhost = MACRO.is_yNegGhost
      mk_cm2 = function(c, b) return rexpr (c+int3d{ 0,-2, 0})%b end end
      mk_cm1 = function(c, b) return rexpr (c+int3d{ 0,-1, 0})%b end end
      mk_cp1 = function(c, b) return rexpr (c+int3d{ 0, 1, 0})%b end end
      mk_cp2 = function(c, b) return rexpr (c+int3d{ 0, 2, 0})%b end end
      mk_cp3 = function(c, b) return rexpr (c+int3d{ 0, 3, 0})%b end end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      is_PosGhost = MACRO.is_zPosGhost
      is_NegGhost = MACRO.is_zNegGhost
      mk_cm2 = function(c, b) return rexpr (c+int3d{ 0, 0,-2})%b end end
      mk_cm1 = function(c, b) return rexpr (c+int3d{ 0, 0,-1})%b end end
      mk_cp1 = function(c, b) return rexpr (c+int3d{ 0, 0, 1})%b end end
      mk_cp2 = function(c, b) return rexpr (c+int3d{ 0, 0, 2})%b end end
      mk_cp3 = function(c, b) return rexpr (c+int3d{ 0, 0, 3})%b end end
   end

   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task CorrectGhostOperators(Fluid : region(ispace(int3d), Fluid_columns),
                           Fluid_bounds : rect3d,
                           BCLeft : int32, BCRight : int32,
                           Grid_Bnum : int32, Grid_Num : int32)
   where
      reads writes(Fluid.[nType])
   do
      var isLeftStaggered = (BCLeft == SCHEMA.FlowBC_Dirichlet or
                             BCLeft == SCHEMA.FlowBC_AdiabaticWall or
                             BCLeft == SCHEMA.FlowBC_IsothermalWall or
                             BCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall)

      var isLeftCollocated = (BCLeft == SCHEMA.FlowBC_NSCBC_Inflow or
                              BCLeft == SCHEMA.FlowBC_NSCBC_Outflow or
                              BCLeft == SCHEMA.FlowBC_NSCBC_FarField or
                              BCLeft == SCHEMA.FlowBC_IncomingShock or
                              BCLeft == SCHEMA.FlowBC_RecycleRescaling)

      var isRightStaggered = (BCRight == SCHEMA.FlowBC_Dirichlet or
                              BCRight == SCHEMA.FlowBC_AdiabaticWall or
                              BCRight == SCHEMA.FlowBC_IsothermalWall or
                              BCRight == SCHEMA.FlowBC_SuctionAndBlowingWall)

      var isRightCollocated = (BCRight == SCHEMA.FlowBC_NSCBC_Inflow or
                               BCRight == SCHEMA.FlowBC_NSCBC_Outflow or
                               BCRight == SCHEMA.FlowBC_NSCBC_FarField or
                               BCRight == SCHEMA.FlowBC_IncomingShock or
                               BCRight == SCHEMA.FlowBC_RecycleRescaling)

      __demand(__openmp)
      for c in Fluid do
         var cm2 = [mk_cm2(rexpr c end, rexpr Fluid_bounds end)];
         var cm1 = [mk_cm1(rexpr c end, rexpr Fluid_bounds end)];
         var cp1 = [mk_cp1(rexpr c end, rexpr Fluid_bounds end)];
         var cp2 = [mk_cp2(rexpr c end, rexpr Fluid_bounds end)];
         var cp3 = [mk_cp3(rexpr c end, rexpr Fluid_bounds end)];

         -- At first update the node type
         -- Left side
         if is_NegGhost(c, Grid_Bnum) then
            if     isLeftStaggered  then Fluid[c].[nType] = L_S_node
            elseif isLeftCollocated then Fluid[c].[nType] = L_C_node
            end
         elseif is_NegGhost(cm1, Grid_Bnum) then
            if     isLeftStaggered  then Fluid[c].[nType] = Lp1_S_node
            elseif isLeftCollocated then Fluid[c].[nType] = Lp1_C_node
            end
         elseif is_NegGhost(cm2, Grid_Bnum) then
            if isLeftStaggered then Fluid[c].[nType] = Lp2_S_node end
         end

         -- Right side
         if is_PosGhost(c, Grid_Bnum, Grid_Num) then
            if     isRightStaggered  then Fluid[c].[nType] = R_S_node
            elseif isRightCollocated then Fluid[c].[nType] = R_C_node
            end
         elseif is_PosGhost(cp1, Grid_Bnum, Grid_Num) then
            if     isRightStaggered  then Fluid[c].[nType] = Rm1_S_node
            elseif isRightCollocated then Fluid[c].[nType] = Rm1_C_node
            end
        elseif is_PosGhost(cp2, Grid_Bnum, Grid_Num) then
            if     isRightStaggered  then Fluid[c].[nType] = Rm2_S_node
            elseif isRightCollocated then Fluid[c].[nType] = Rm2_C_node
            end
         elseif is_PosGhost(cp3, Grid_Bnum, Grid_Num) then
            if isRightStaggered then Fluid[c].[nType] = Rm3_S_node end
         end
      end
   end
   return CorrectGhostOperators
end)

__demand(__inline)
task Exports.InitializeOperators(Fluid : region(ispace(int3d), Fluid_columns),
                                 tiles : ispace(int3d),
                                 Fluid_Zones : zones_partitions(Fluid, tiles),
                                 config    : SCHEMA.Config,
                                 xBnum : int32, yBnum : int32, zBnum : int32)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All} = Fluid_Zones

   -- Initialize the internal operators
   InitializeOperators(Fluid, tiles, p_All)

   -- Enforce BCs on the operators
   __demand(__index_launch)
   for c in tiles do [mkCorrectGhostOperators("x")](p_All[c], Fluid.bounds, config.BC.xBCLeft.type, config.BC.xBCRight.type, xBnum, config.Grid.xNum) end
   __demand(__index_launch)
   for c in tiles do [mkCorrectGhostOperators("y")](p_All[c], Fluid.bounds, config.BC.yBCLeft.type, config.BC.yBCRight.type, yBnum, config.Grid.yNum) end
   __demand(__index_launch)
   for c in tiles do [mkCorrectGhostOperators("z")](p_All[c], Fluid.bounds, config.BC.zBCLeft.type, config.BC.zBCRight.type, zBnum, config.Grid.zNum) end

end

-------------------------------------------------------------------------------
-- METRIC ROUTINES
-------------------------------------------------------------------------------

local extern task InitializeMetric(MetricGhosts : region(ispace(int3d), Fluid_columns),
                                   Fluid : region(ispace(int3d), Fluid_columns),
                                   Fluid_bounds : rect3d,
                                   bBox : bBoxType)
where
   reads(MetricGhosts.centerCoordinates),
   reads(MetricGhosts.{nType_x, nType_y, nType_z}),
   writes(Fluid.{dcsi_e, deta_e, dzet_e}),
   writes(Fluid.{dcsi_d, deta_d, dzet_d}),
   writes(Fluid.{dcsi_s, deta_s, dzet_s})
end
InitializeMetric:set_task_id(TYPES.TID_InitializeMetric)
--for k, v in pairs(InitializeMetric:get_params_struct():getentries()) do
--   print(k, v)
--   for k2, v2 in pairs(v) do print(k2, v2) end
--end

local mkCorrectGhostMetric = terralib.memoize(function(sdir)
   local CorrectGhostMetric

   local nType
   local N
   if sdir == "x" then
      nType = "nType_x"
      N  = "dcsi_e"
   elseif sdir == "y" then
      nType = "nType_y"
      N  = "deta_e"
   elseif sdir == "z" then
      nType = "nType_z"
      N  = "dzet_e"
   end

   extern task CorrectGhostMetric(Fluid : region(ispace(int3d), Fluid_columns))
   where
      reads(Fluid.centerCoordinates),
      reads(Fluid.[nType]),
      reads writes(Fluid.[N])
   end
   if sdir == "x" then
      CorrectGhostMetric:set_task_id(TYPES.TID_CorrectGhostMetricX)
   elseif sdir == "y" then
      CorrectGhostMetric:set_task_id(TYPES.TID_CorrectGhostMetricY)
   elseif sdir == "z" then
      CorrectGhostMetric:set_task_id(TYPES.TID_CorrectGhostMetricZ)
   end
   return CorrectGhostMetric
end)

__demand(__inline)
task Exports.InitializeMetric(Fluid : region(ispace(int3d), Fluid_columns),
                                  tiles : ispace(int3d),
                                  Fluid_Zones : zones_partitions(Fluid, tiles),
                                  Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                  bBox   : bBoxType,
                                  config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All} = Fluid_Zones
   var {p_MetricGhosts} = Fluid_Ghost

   -- Initialize internal metrics
   __demand(__index_launch)
   for c in tiles do
      InitializeMetric(p_MetricGhosts[c],
                       p_All[c],
                       Fluid.bounds,
                       bBox)
   end

   -- Enforce BCs on the metrics
   __demand(__index_launch)
   for c in tiles do [mkCorrectGhostMetric("x")](p_All[c]) end
   __demand(__index_launch)
   for c in tiles do [mkCorrectGhostMetric("y")](p_All[c]) end
   __demand(__index_launch)
   for c in tiles do [mkCorrectGhostMetric("z")](p_All[c]) end
end

return Exports end
