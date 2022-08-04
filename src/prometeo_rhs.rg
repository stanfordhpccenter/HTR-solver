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

return function(SCHEMA, MIX, METRIC, TYPES, STAT,
                zones_partitions, ghost_partitions,
                IncomingShockParams,
                ATOMIC) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local UTIL = require 'util'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-- Math
local C = regentlib.c
local pow  = regentlib.pow(double)

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives   = CONST.Primitives
local Properties   = CONST.Properties
local ProfilesVars = CONST.ProfilesVars

-- Types
local Fluid_columns = TYPES.Fluid_columns
local bBoxType      = TYPES.bBoxType

-- Atomic switch
local Fluid = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "Fluid")
local BC = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "BC")
local coherence_mode
local coherence_mode_BC
if ATOMIC then
   coherence_mode    = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
   coherence_mode_BC = regentlib.coherence(regentlib.atomic,       BC, "Conserved_t")
else
   coherence_mode    = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
   coherence_mode_BC = regentlib.coherence(regentlib.exclusive,    BC, "Conserved_t")
end

-------------------------------------------------------------------------------
-- EULER FLUX-DIVERGENCE ROUTINES
-------------------------------------------------------------------------------

local mkUpdateUsingHybridEulerFlux = terralib.memoize(function(dir)
   local UpdateUsingHybridEulerFlux

   local shockSensor
   local nType
   local m_e
   if (dir == "x") then
      shockSensor = "shockSensorX"
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
      shockSensor = "shockSensorY"
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
      shockSensor = "shockSensorZ"
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end
   local Conserved_list = terralib.newlist({"Conserved"})
   if MIX.nSpec > 1 then
      Conserved_list:insert("Conserved_old")
   end

   extern task UpdateUsingHybridEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                          SensorGhost : region(ispace(int3d), Fluid_columns),
                                          FluxGhost : region(ispace(int3d), Fluid_columns),
                                          [Fluid],
                                          RK_coeffs    : double[2],
                                          deltaTime    : double,
                                          Fluid_bounds : rect3d,
                                          mix          : MIX.Mixture)
   where
      reads(EulerGhost.[Conserved_list]),
      reads(EulerGhost.rho),
      reads(EulerGhost.MassFracs),
      reads(EulerGhost.velocity),
      reads(EulerGhost.pressure),
      reads(EulerGhost.SoS),
      reads(SensorGhost.[shockSensor]),
      reads(SensorGhost.[m_e]),
      reads(FluxGhost.[nType]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   --for k, v in pairs(UpdateUsingFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if     (dir == "x") then
      UpdateUsingHybridEulerFlux:set_task_id(TYPES.TID_UpdateUsingHybridEulerFluxX)
   elseif (dir == "y") then
      UpdateUsingHybridEulerFlux:set_task_id(TYPES.TID_UpdateUsingHybridEulerFluxY)
   elseif (dir == "z") then
      UpdateUsingHybridEulerFlux:set_task_id(TYPES.TID_UpdateUsingHybridEulerFluxZ)
   end
   return UpdateUsingHybridEulerFlux
end)

local mkUpdateUsingTENOEulerFlux = terralib.memoize(function(dir, Op)
   local UpdateUsingTENOEulerFlux

   local nType
   local m_e
   if (dir == "x") then
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end
   local Conserved_list = terralib.newlist({"Conserved"})
   if MIX.nSpec > 1 then
      Conserved_list:insert("Conserved_old")
   end

   extern task UpdateUsingTENOEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                        DiffGhost : region(ispace(int3d), Fluid_columns),
                                        FluxGhost : region(ispace(int3d), Fluid_columns),
                                        [Fluid],
                                        RK_coeffs    : double[2],
                                        deltaTime    : double,
                                        Fluid_bounds : rect3d,
                                        mix          : MIX.Mixture)
   where
      reads(EulerGhost.[Conserved_list]),
      reads(EulerGhost.rho),
      reads(EulerGhost.velocity),
      reads(EulerGhost.pressure),
      reads(EulerGhost.SoS),
      reads(DiffGhost.MassFracs),
      reads(DiffGhost.[m_e]),
      reads(FluxGhost.[nType]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   --for k, v in pairs(UpdateUsingFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if (Op == "TENO") then
      if     (dir == "x") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOEulerFluxX)
      elseif (dir == "y") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOEulerFluxY)
      elseif (dir == "z") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOEulerFluxZ)
      else
         assert(false)
      end
   elseif (Op == "TENOA") then
      if     (dir == "x") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOAEulerFluxX)
      elseif (dir == "y") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOAEulerFluxY)
      elseif (dir == "z") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOAEulerFluxZ)
      else
         assert(false)
      end
   elseif (Op == "TENOLAD") then
      if     (dir == "x") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOLADEulerFluxX)
      elseif (dir == "y") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOLADEulerFluxY)
      elseif (dir == "z") then
         UpdateUsingTENOEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOLADEulerFluxZ)
      else
         assert(false)
      end
   else
      assert(false)
   end
   return UpdateUsingTENOEulerFlux
end)

local mkUpdateUsingSkewSymmetricEulerFlux = terralib.memoize(function(dir)
   local UpdateUsingSkewSymmetricEulerFlux

   local nType
   local m_e
   if (dir == "x") then
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end

   extern task UpdateUsingSkewSymmetricEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                                 FluxGhost : region(ispace(int3d), Fluid_columns),
                                                 [Fluid],
                                                 Fluid_bounds : rect3d,
                                                 mix : MIX.Mixture)
   where
      reads(EulerGhost.Conserved),
      reads(EulerGhost.rho),
      reads(EulerGhost.MassFracs),
      reads(EulerGhost.velocity),
      reads(EulerGhost.pressure),
      reads(FluxGhost.[nType]),
      reads(Fluid.[m_e]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   if     (dir == "x") then
      UpdateUsingSkewSymmetricEulerFlux:set_task_id(TYPES.TID_UpdateUsingSkewSymmetricEulerFluxX)
   elseif (dir == "y") then
      UpdateUsingSkewSymmetricEulerFlux:set_task_id(TYPES.TID_UpdateUsingSkewSymmetricEulerFluxY)
   elseif (dir == "z") then
      UpdateUsingSkewSymmetricEulerFlux:set_task_id(TYPES.TID_UpdateUsingSkewSymmetricEulerFluxZ)
   end
   return UpdateUsingSkewSymmetricEulerFlux
end)

__demand(__inline)
task Exports.UpdateUsingEulerFlux([Fluid],
                                  tiles : ispace(int3d),
                                  Fluid_Zones : zones_partitions(Fluid, tiles),
                                  Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                  RK_coeffs : double[2],
                                  deltaTime : double,
                                  Mix       : MIX.Mixture,
                                  config    : SCHEMA.Config)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_x_divg, p_y_divg, p_z_divg} = Fluid_Zones;
   var {p_XFluxGhosts,    p_YFluxGhosts,    p_ZFluxGhosts,
        p_XDiffGhosts,    p_YDiffGhosts,    p_ZDiffGhosts,
        p_XEulerGhosts,   p_YEulerGhosts,   p_ZEulerGhosts,
        p_XSensorGhosts2, p_YSensorGhosts2, p_ZSensorGhosts2} = Fluid_Ghost;

   if (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_Hybrid) then
      -- Call tasks with hybrid scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingHybridEulerFlux("z")](p_ZEulerGhosts[c], p_ZSensorGhosts2[c], p_ZFluxGhosts[c],
                                             p_z_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingHybridEulerFlux("y")](p_YEulerGhosts[c], p_YSensorGhosts2[c], p_YFluxGhosts[c],
                                             p_y_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingHybridEulerFlux("x")](p_XEulerGhosts[c], p_XSensorGhosts2[c], p_XFluxGhosts[c],
                                             p_x_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_TENO) then
      -- Call tasks with TENO scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("z", "TENO")](p_ZEulerGhosts[c], p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                                   p_z_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("y", "TENO")](p_YEulerGhosts[c], p_YDiffGhosts[c], p_YFluxGhosts[c],
                                                   p_y_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("x", "TENO")](p_XEulerGhosts[c], p_XDiffGhosts[c], p_XFluxGhosts[c],
                                                   p_x_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_TENOA) then
      -- Call tasks with TENO-A scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("z", "TENOA")](p_ZEulerGhosts[c], p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                                    p_z_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("y", "TENOA")](p_YEulerGhosts[c], p_YDiffGhosts[c], p_YFluxGhosts[c],
                                                    p_y_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("x", "TENOA")](p_XEulerGhosts[c], p_XDiffGhosts[c], p_XFluxGhosts[c],
                                                    p_x_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_TENOLAD) then
      -- Call tasks with TENO-A scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("z", "TENOLAD")](p_ZEulerGhosts[c], p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                                      p_z_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("y", "TENOLAD")](p_YEulerGhosts[c], p_YDiffGhosts[c], p_YFluxGhosts[c],
                                                      p_y_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOEulerFlux("x", "TENOLAD")](p_XEulerGhosts[c], p_XDiffGhosts[c], p_XFluxGhosts[c],
                                                      p_x_divg[c], RK_coeffs, deltaTime, Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_SkewSymmetric) then
      -- Call tasks with SkewSymmetric scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingSkewSymmetricEulerFlux("z")](p_ZEulerGhosts[c], p_ZFluxGhosts[c],
                                                    p_z_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingSkewSymmetricEulerFlux("y")](p_YEulerGhosts[c], p_YFluxGhosts[c],
                                                    p_y_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingSkewSymmetricEulerFlux("x")](p_XEulerGhosts[c], p_XFluxGhosts[c],
                                                    p_x_divg[c], Fluid.bounds, Mix)
      end
   end
end

-------------------------------------------------------------------------------
-- DIFFUSION FLUX-DIVERGENCE ROUTINES
-------------------------------------------------------------------------------

local mkUpdateUsingDiffusionFlux = terralib.memoize(function(dir)
   local UpdateUsingDiffusionFlux

   local nType
   local nType1
   local nType2
   local m_d
   local m_d1
   local m_d2
   local m_s
   if (dir == "x") then
      nType  = "nType_x"
      nType1 = "nType_y"
      nType2 = "nType_z"
      m_d    = "dcsi_d"
      m_d1   = "deta_d"
      m_d2   = "dzet_d"
      m_s    = "dcsi_s"
   elseif (dir == "y") then
      nType  = "nType_y"
      nType1 = "nType_x"
      nType2 = "nType_z"
      m_d    = "deta_d"
      m_d1   = "dcsi_d"
      m_d2   = "dzet_d"
      m_s    = "deta_s"
   elseif (dir == "z") then
      nType  = "nType_z"
      nType1 = "nType_x"
      nType2 = "nType_y"
      m_d    = "dzet_d"
      m_d1   = "dcsi_d"
      m_d2   = "deta_d"
      m_s    = "dzet_s"
   else assert(false) end

   extern task UpdateUsingDiffusionFlux(DiffGhost : region(ispace(int3d), Fluid_columns),
                                        DiffGradGhost : region(ispace(int3d), Fluid_columns),
                                        FluxGhost : region(ispace(int3d), Fluid_columns),
                                        [Fluid],
                                        Fluid_bounds : rect3d,
                                        mix : MIX.Mixture)
   where
      reads(DiffGhost.Conserved),
      reads(DiffGhost.{MolarFracs, temperature}),
      reads(DiffGhost.{rho, mu, lam, Di}),
      reads(DiffGhost.{[nType1], [nType2]}),
      reads(DiffGhost.{[m_d1], [m_d2]}),
      reads(DiffGradGhost.velocity),
      reads(FluxGhost.[nType]),
      reads(FluxGhost.[m_s]),
      reads(Fluid.[m_d]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   --for k, v in pairs(UpdateUsingDiffusionFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if     (dir == "x") then
      UpdateUsingDiffusionFlux:set_task_id(TYPES.TID_UpdateUsingDiffusionFluxX)
   elseif (dir == "y") then
      UpdateUsingDiffusionFlux:set_task_id(TYPES.TID_UpdateUsingDiffusionFluxY)
   elseif (dir == "z") then
      UpdateUsingDiffusionFlux:set_task_id(TYPES.TID_UpdateUsingDiffusionFluxZ)
   end
   return UpdateUsingDiffusionFlux
end)

__demand(__inline)
task Exports.UpdateUsingDiffusionFlux([Fluid],
                                      tiles : ispace(int3d),
                                      Fluid_Zones : zones_partitions(Fluid, tiles),
                                      Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                      Mix : MIX.Mixture,
                                      config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_x_divg, p_y_divg, p_z_divg} = Fluid_Zones;
   var {p_XFluxGhosts,     p_YFluxGhosts,     p_ZFluxGhosts,
        p_XDiffGhosts,     p_YDiffGhosts,     p_ZDiffGhosts,
        p_XDiffGradGhosts, p_YDiffGradGhosts, p_ZDiffGradGhosts} = Fluid_Ghost;

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingDiffusionFlux("z")](p_ZDiffGhosts[c], p_ZDiffGradGhosts[c], p_ZFluxGhosts[c],
                                        p_z_divg[c], Fluid.bounds, Mix)
   end

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingDiffusionFlux("y")](p_YDiffGhosts[c], p_YDiffGradGhosts[c], p_YFluxGhosts[c],
                                        p_y_divg[c], Fluid.bounds, Mix)
   end

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingDiffusionFlux("x")](p_XDiffGhosts[c], p_XDiffGradGhosts[c], p_XFluxGhosts[c],
                                        p_x_divg[c], Fluid.bounds, Mix)
   end
end

-------------------------------------------------------------------------------
-- NSCBC-FLUX ROUTINES
-------------------------------------------------------------------------------
-- Adds NSCBC fluxes to the inflow cells
local mkUpdateUsingFluxNSCBCInflow = terralib.memoize(function(dir)
   local UpdateUsingFluxNSCBCInflow

   local nType
   local m_d
   if     ((dir == "xNeg") or (dir == "xPos")) then
      nType = "nType_x"
      m_d = "dcsi_d"
   elseif ((dir == "yNeg") or (dir == "yPos")) then
      nType = "nType_y"
      m_d = "deta_d"
   elseif ((dir == "zNeg") or (dir == "zPos")) then
      nType = "nType_z"
      m_d = "dzet_d"
   end

   extern task UpdateUsingFluxNSCBCInflow([Fluid],
                                          [BC],
                                          mix : MIX.Mixture)
   where
      reads(Fluid.{MassFracs, pressure, velocity}),
      reads(BC.[nType]),
      reads(BC.[m_d]),
      reads(BC.{rho, SoS}),
      reads(BC.{temperature}),
      reads(BC.{dudtBoundary, dTdtBoundary}),
      reads writes(BC.Conserved_t)
   end
   if     dir == "xNeg" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowXNeg)
   elseif dir == "xPos" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowXPos)
   elseif dir == "yNeg" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowYNeg)
   elseif dir == "yPos" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowYPos)
   elseif dir == "zNeg" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowZNeg)
   elseif dir == "zPos" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowZPos)
   else assert(false) end
   return UpdateUsingFluxNSCBCInflow
end)

__demand(__inline)
task Exports.UpdateUsingNSCBCInflow([Fluid],
                                    tiles : ispace(int3d),
                                    Fluid_Zones : zones_partitions(Fluid, tiles),
                                    Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                    Mix : MIX.Mixture,
                                    config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_All,
        p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Zones;

   -- Update using NSCBC_Inflow
   if ((config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) or
       (config.BC.xBCLeft.type == SCHEMA.FlowBC_RecycleRescaling)) then
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkUpdateUsingFluxNSCBCInflow("xNeg")](p_All[c], p_xNeg[0][c], Mix)
      end
   end

   if ((config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_Inflow)) then
      __demand(__index_launch)
      for c in xPos_ispace do
         [mkUpdateUsingFluxNSCBCInflow("xPos")](p_All[c], p_xPos[0][c], Mix)
      end
   end

   if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) then
      __demand(__index_launch)
      for c in yNeg_ispace do
         [mkUpdateUsingFluxNSCBCInflow("yNeg")](p_All[c], p_yNeg[0][c], Mix)
      end
   end

   if (config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Inflow) then
      __demand(__index_launch)
      for c in yPos_ispace do
         [mkUpdateUsingFluxNSCBCInflow("yPos")](p_All[c], p_yPos[0][c], Mix)
      end
   end

   if (config.BC.zBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) then
      __demand(__index_launch)
      for c in zNeg_ispace do
         [mkUpdateUsingFluxNSCBCInflow("zNeg")](p_All[c], p_zNeg[0][c], Mix)
      end
   end

   if ((config.BC.zBCRight.type == SCHEMA.FlowBC_NSCBC_Inflow)) then
      __demand(__index_launch)
      for c in zPos_ispace do
         [mkUpdateUsingFluxNSCBCInflow("zPos")](p_All[c], p_zPos[0][c], Mix)
      end
   end
end

-- Adds NSCBC fluxes to the outflow cells
local mkUpdateUsingFluxNSCBCOutflow  = terralib.memoize(function(dir)
   local UpdateUsingFluxNSCBCOutflow

   local nType_N
   local nType_T1
   local nType_T2
   local m_d_N
   local m_d_T1
   local m_d_T2
   if     ((dir == "xNeg") or (dir == "xPos")) then
      nType_N  = "nType_x"
      nType_T1 = "nType_y"
      nType_T2 = "nType_z"
      m_d_N  = "dcsi_d"
      m_d_T1 = "deta_d"
      m_d_T2 = "dzet_d"
   elseif ((dir == "yNeg") or (dir == "yPos")) then
      nType_N  = "nType_y"
      nType_T1 = "nType_x"
      nType_T2 = "nType_z"
      m_d_N  = "deta_d"
      m_d_T1 = "dcsi_d"
      m_d_T2 = "dzet_d"
   elseif ((dir == "zNeg") or (dir == "zPos")) then
      nType_N  = "nType_z"
      nType_T1 = "nType_x"
      nType_T2 = "nType_y"
      m_d_N  = "dzet_d"
      m_d_T1 = "dcsi_d"
      m_d_T2 = "deta_d"
   end
   extern task UpdateUsingFluxNSCBCOutflow(Ghost : region(ispace(int3d), Fluid_columns),
                                           [Fluid],
                                           [BC],
                                           Fluid_bounds : rect3d,
                                           mix : MIX.Mixture,
                                           MaxMach : double,
                                           LengthScale : double,
                                           Pinf : double)
   where
      reads(Ghost.velocity),
      reads(Fluid.{[nType_N], [nType_T1], [nType_T2]}),
      reads(Fluid.{[m_d_N], [m_d_T1], [m_d_T2]}),
      reads(Fluid.{rho, mu}),
      reads(Fluid.{MassFracs, pressure}),
      reads(BC.SoS),
      reads(BC.temperature),
      reads(BC.Conserved),
      reads writes(BC.Conserved_t),
      [coherence_mode_BC]
   end
   if     dir == "xNeg" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowXNeg)
   elseif dir == "xPos" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowXPos)
   elseif dir == "yNeg" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowYNeg)
   elseif dir == "yPos" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowYPos)
   elseif dir == "zNeg" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowZNeg)
   elseif dir == "zPos" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowZPos)
   else assert(false) end
--   for k, v in pairs(UpdateUsingFluxNSCBCOutflow:get_params_struct():getentries()) do
--      print(k, v)
--      for k2, v2 in pairs(v) do print(k2, v2) end
--   end
   return UpdateUsingFluxNSCBCOutflow
end)

__demand(__inline)
task Exports.UpdateUsingNSCBCOutflow([Fluid],
                                     tiles : ispace(int3d),
                                     Fluid_Zones : zones_partitions(Fluid, tiles),
                                     Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                     Mix : MIX.Mixture,
                                     config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_All, p_Interior,
        p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Zones;
   var {p_GradientGhosts} = Fluid_Ghost;
   -- TODO: p_GradientGhosts is an overkill for this task, but we also want to avoid creating too many partitions

   if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow or config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 0)
      end
      if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in xNeg_ispace do
            [mkUpdateUsingFluxNSCBCOutflow("xNeg")](p_GradientGhosts[c], p_All[c], p_xNeg[0][c],
                                                    Fluid.bounds, Mix, MaxMach,
                                                    config.BC.xBCLeft.u.NSCBC_Outflow.LengthScale,
                                                    config.BC.xBCLeft.u.NSCBC_Outflow.P)
         end
      end
      if (config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in xPos_ispace do
            [mkUpdateUsingFluxNSCBCOutflow("xPos")](p_GradientGhosts[c], p_All[c], p_xPos[0][c],
                                                    Fluid.bounds, Mix, MaxMach,
                                                    config.BC.xBCRight.u.NSCBC_Outflow.LengthScale,
                                                    config.BC.xBCRight.u.NSCBC_Outflow.P)
         end
      end
   end

   if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow or config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 1)
      end
      if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in yNeg_ispace do
            [mkUpdateUsingFluxNSCBCOutflow("yNeg")](p_GradientGhosts[c], p_All[c], p_yNeg[0][c],
                                                    Fluid.bounds, Mix, MaxMach,
                                                    config.BC.yBCLeft.u.NSCBC_Outflow.LengthScale,
                                                    config.BC.yBCLeft.u.NSCBC_Outflow.P)
         end
      end
      if (config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in yPos_ispace do
            [mkUpdateUsingFluxNSCBCOutflow("yPos")](p_GradientGhosts[c], p_All[c], p_yPos[0][c],
                                                    Fluid.bounds, Mix, MaxMach,
                                                    config.BC.yBCRight.u.NSCBC_Outflow.LengthScale,
                                                    config.BC.yBCRight.u.NSCBC_Outflow.P)
         end
      end
   end

   if (config.BC.zBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow or config.BC.zBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 2)
      end
      if (config.BC.zBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in zNeg_ispace do
            [mkUpdateUsingFluxNSCBCOutflow("zNeg")](p_GradientGhosts[c], p_All[c], p_zNeg[0][c],
                                                    Fluid.bounds, Mix, MaxMach,
                                                    config.BC.zBCLeft.u.NSCBC_Outflow.LengthScale,
                                                    config.BC.zBCLeft.u.NSCBC_Outflow.P)
         end
      end
      if (config.BC.zBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in zPos_ispace do
            [mkUpdateUsingFluxNSCBCOutflow("zPos")](p_GradientGhosts[c], p_All[c], p_zPos[0][c],
                                                    Fluid.bounds, Mix, MaxMach,
                                                    config.BC.zBCRight.u.NSCBC_Outflow.LengthScale,
                                                    config.BC.zBCRight.u.NSCBC_Outflow.P)
         end
      end
   end
end

-- Adds NSCBC fluxes to the far field cells
local mkUpdateUsingFluxNSCBCFarField  = terralib.memoize(function(dir)
   local UpdateUsingFluxNSCBCFarField

   local nType
   local m_d
   if     ((dir == "xNeg") or (dir == "xPos")) then
      nType = "nType_x"
      m_d   = "dcsi_d"
   elseif ((dir == "yNeg") or (dir == "yPos")) then
      nType = "nType_y"
      m_d   = "deta_d"
   elseif ((dir == "zNeg") or (dir == "zPos")) then
      nType = "nType_z"
      m_d   = "dzet_d"
   end
   extern task UpdateUsingFluxNSCBCFarField([Fluid],
                                            [BC],
                                            mix : MIX.Mixture,
                                            MaxMach : double,
                                            LengthScale : double,
                                            Pinf : double)
   where
      reads(Fluid.rho),
      reads(Fluid.{MassFracs, pressure, temperature, velocity}),
      reads(BC.[m_d]),
      reads(BC.[nType]),
      reads(BC.SoS),
      reads(BC.Conserved),
      reads(BC.[ProfilesVars]),
      reads writes(BC.Conserved_t),
      [coherence_mode_BC]
   end
   if     dir == "xNeg" then
      UpdateUsingFluxNSCBCFarField:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCFarFieldXNeg)
   elseif dir == "xPos" then
      UpdateUsingFluxNSCBCFarField:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCFarFieldXPos)
   elseif dir == "yNeg" then
      UpdateUsingFluxNSCBCFarField:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCFarFieldYNeg)
   elseif dir == "yPos" then
      UpdateUsingFluxNSCBCFarField:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCFarFieldYPos)
   elseif dir == "zNeg" then
      UpdateUsingFluxNSCBCFarField:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCFarFieldZNeg)
   elseif dir == "zPos" then
      UpdateUsingFluxNSCBCFarField:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCFarFieldZPos)
   else assert(false) end
   return UpdateUsingFluxNSCBCFarField
end)

__demand(__inline)
task Exports.UpdateUsingNSCBCFarField([Fluid],
                                      tiles : ispace(int3d),
                                      Fluid_Zones : zones_partitions(Fluid, tiles),
                                      Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                      Mix : MIX.Mixture,
                                      config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_All, p_Interior,
        p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Zones;

   if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_FarField or config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_FarField) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 0)
      end
      if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_FarField) then
         __demand(__index_launch)
         for c in xNeg_ispace do
            [mkUpdateUsingFluxNSCBCFarField("xNeg")](p_All[c], p_xNeg[0][c],
                                                    Mix, MaxMach,
                                                    config.BC.xBCLeft.u.NSCBC_FarField.LengthScale,
                                                    config.BC.xBCLeft.u.NSCBC_FarField.P)
         end
      end
      if (config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_FarField) then
         __demand(__index_launch)
         for c in xPos_ispace do
            [mkUpdateUsingFluxNSCBCFarField("xPos")](p_All[c], p_xPos[0][c],
                                                    Mix, MaxMach,
                                                    config.BC.xBCRight.u.NSCBC_FarField.LengthScale,
                                                    config.BC.xBCRight.u.NSCBC_FarField.P)
         end
      end
   end

   if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_FarField or config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_FarField) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 1)
      end
      if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_FarField) then
         __demand(__index_launch)
         for c in yNeg_ispace do
            [mkUpdateUsingFluxNSCBCFarField("yNeg")](p_All[c], p_yNeg[0][c],
                                                    Mix, MaxMach,
                                                    config.BC.yBCLeft.u.NSCBC_FarField.LengthScale,
                                                    config.BC.yBCLeft.u.NSCBC_FarField.P)
         end
      end
      if (config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_FarField) then
         __demand(__index_launch)
         for c in yPos_ispace do
            [mkUpdateUsingFluxNSCBCFarField("yPos")](p_All[c], p_yPos[0][c],
                                                    Mix, MaxMach,
                                                    config.BC.yBCRight.u.NSCBC_FarField.LengthScale,
                                                    config.BC.yBCRight.u.NSCBC_FarField.P)
         end
      end
   end

   if (config.BC.zBCLeft.type == SCHEMA.FlowBC_NSCBC_FarField or config.BC.zBCRight.type == SCHEMA.FlowBC_NSCBC_FarField) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 2)
      end
      if (config.BC.zBCLeft.type == SCHEMA.FlowBC_NSCBC_FarField) then
         __demand(__index_launch)
         for c in zNeg_ispace do
            [mkUpdateUsingFluxNSCBCFarField("zNeg")](p_All[c], p_zNeg[0][c],
                                                    Mix, MaxMach,
                                                    config.BC.zBCLeft.u.NSCBC_FarField.LengthScale,
                                                    config.BC.zBCLeft.u.NSCBC_FarField.P)
         end
      end
      if (config.BC.zBCRight.type == SCHEMA.FlowBC_NSCBC_FarField) then
         __demand(__index_launch)
         for c in zPos_ispace do
            [mkUpdateUsingFluxNSCBCFarField("zPos")](p_All[c], p_zPos[0][c],
                                                    Mix, MaxMach,
                                                    config.BC.zBCRight.u.NSCBC_FarField.LengthScale,
                                                    config.BC.zBCRight.u.NSCBC_FarField.P)
         end
      end
   end
end

-- Adds NSCBC fluxes to the incoming shock cells
local mkUpdateUsingFluxIncomingShock  = terralib.memoize(function(dir)
   local UpdateUsingFluxIncomingShock

   local nType
   local m_d
   if     ((dir == "xNeg") or (dir == "xPos")) then
      nType = "nType_x"
      m_d   = "dcsi_d"
   elseif ((dir == "yNeg") or (dir == "yPos")) then
      nType = "nType_y"
      m_d   = "deta_d"
   elseif ((dir == "zNeg") or (dir == "zPos")) then
      nType = "nType_z"
      m_d   = "dzet_d"
   end
   extern task UpdateUsingFluxIncomingShock([Fluid],
                                            [BC],
                                            mix : MIX.Mixture,
                                            MaxMach : double,
                                            LengthScale : double,
                                            IncomingShock : IncomingShockParams)
   where
      reads(Fluid.rho),
      reads(Fluid.{MassFracs, pressure, temperature, velocity}),
      reads(BC.[m_d]),
      reads(BC.[nType]),
      reads(BC.SoS),
      reads(BC.Conserved),
      reads writes(BC.Conserved_t),
      [coherence_mode_BC]
   end
--   if     dir == "xNeg" then
--      UpdateUsingFluxIncomingShock:set_task_id(TYPES.TID_UpdateUsingFluxIncomingShockXNeg)
--   elseif dir == "xPos" then
--      UpdateUsingFluxIncomingShock:set_task_id(TYPES.TID_UpdateUsingFluxIncomingShockXPos)
--   elseif dir == "yNeg" then
--      UpdateUsingFluxIncomingShock:set_task_id(TYPES.TID_UpdateUsingFluxIncomingShockYNeg)
--   elseif dir == "yPos" then
   if dir == "yPos" then
      UpdateUsingFluxIncomingShock:set_task_id(TYPES.TID_UpdateUsingFluxIncomingShockYPos)
--   elseif dir == "zNeg" then
--      UpdateUsingFluxIncomingShock:set_task_id(TYPES.TID_UpdateUsingFluxIncomingShockZNeg)
--   elseif dir == "zPos" then
--      UpdateUsingFluxIncomingShock:set_task_id(TYPES.TID_UpdateUsingFluxIncomingShockZPos)
   else assert(false) end
   return UpdateUsingFluxIncomingShock
end)

__demand(__inline)
task Exports.UpdateUsingNSCBCIncomingShock([Fluid],
                                           tiles : ispace(int3d),
                                           Fluid_Zones : zones_partitions(Fluid, tiles),
                                           Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                           IncomingShock : IncomingShockParams,
                                           Mix : MIX.Mixture,
                                           config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_All, p_Interior,
        p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Zones;
   if (config.BC.yBCLeft.type == SCHEMA.FlowBC_IncomingShock or config.BC.yBCRight.type == SCHEMA.FlowBC_IncomingShock) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_Interior[c], 1)
      end
      if (config.BC.yBCRight.type == SCHEMA.FlowBC_IncomingShock) then
         __demand(__index_launch)
         for c in yPos_ispace do
            [mkUpdateUsingFluxIncomingShock("yPos")](p_All[c], p_yPos[0][c],
                                                     Mix, MaxMach,
                                                     config.BC.yBCRight.u.IncomingShock.LengthScale,
                                                     IncomingShock)
         end
      end
   end
end

-------------------------------------------------------------------------------
-- FORCING ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.AddBodyForces([Fluid],
                           Flow_bodyForce : double[3])
where
   reads(Fluid.{rho, velocity}),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do
   __demand(__openmp)
   for c in Fluid do
      for i=0, 3 do
         Fluid[c].Conserved_t[irU+i] += Fluid[c].rho*Flow_bodyForce[i]
      end
      Fluid[c].Conserved_t[irE] += Fluid[c].rho*MACRO.dot(Flow_bodyForce, Fluid[c].velocity)
   end
end

local extern task CalculateAveragePD(Ghost : region(ispace(int3d), Fluid_columns),
                                     [Fluid],
                                     Fluid_bounds : rect3d) : double
where
   reads(Ghost.velocity),
   reads(Fluid.{nType_x, nType_y, nType_z}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   reads(Fluid.pressure)
end
CalculateAveragePD:set_task_id(TYPES.TID_CalculateAveragePD)

local mkAddDissipation = terralib.memoize(function(dir)
   local AddDissipation

   local nType
   local nType1
   local nType2
   local m_d1
   local m_d2
   local m_s
   if (dir == "x") then
      nType  = "nType_x"
      nType1 = "nType_y"
      nType2 = "nType_z"
      m_d1   = "deta_d"
      m_d2   = "dzet_d"
      m_s    = "dcsi_s"
   elseif (dir == "y") then
      nType  = "nType_y"
      nType1 = "nType_x"
      nType2 = "nType_z"
      m_d1   = "dcsi_d"
      m_d2   = "dzet_d"
      m_s    = "deta_s"
   elseif (dir == "z") then
      nType  = "nType_z"
      nType1 = "nType_x"
      nType2 = "nType_y"
      m_d1   = "dcsi_d"
      m_d2   = "deta_d"
      m_s    = "dzet_s"
   else assert(false) end

   extern task AddDissipation(DiffGhost : region(ispace(int3d), Fluid_columns),
                              DiffGradGhost : region(ispace(int3d), Fluid_columns),
                              FluxGhost : region(ispace(int3d), Fluid_columns),
                              [Fluid],
                              Fluid_bounds : rect3d,
                              mix : MIX.Mixture) : double
   where
      reads(DiffGhost.mu),
      reads(DiffGhost.{[nType1], [nType2]}),
      reads(DiffGhost.{[m_d1], [m_d2]}),
      reads(DiffGradGhost.velocity),
      reads(FluxGhost.[nType]),
      reads(FluxGhost.[m_s])
   end
   --for k, v in pairs(UpdateUsingDiffusionFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if     (dir == "x") then
      AddDissipation:set_task_id(TYPES.TID_AddDissipationX)
   elseif (dir == "y") then
      AddDissipation:set_task_id(TYPES.TID_AddDissipationY)
   elseif (dir == "z") then
      AddDissipation:set_task_id(TYPES.TID_AddDissipationZ)
   end
   return AddDissipation
end)

local __demand(__cuda, __leaf)
task AddHITSource([Fluid],
                  averageDissipation : double,
                  averageKineticEnergy : double,
                  averagePressureDilatation : double,
                  turbForcing : SCHEMA.TurbForcingModel)
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   reads(Fluid.{rho, velocity}),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do
   var W = averagePressureDilatation + averageDissipation
   var G   = turbForcing.u.HIT.Gain
   var t_o = turbForcing.u.HIT.t_o
   var K_o = turbForcing.u.HIT.K_o
   var A = (-W-G*(averageKineticEnergy-K_o)/t_o) / (2.0*averageKineticEnergy)
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var force = MACRO.vs_mul(Fluid[c].velocity, Fluid[c].rho*A)
      for i=0, 3 do
         Fluid[c].Conserved_t[irU+i] += force[i]
      end
      var work = MACRO.dot(force, Fluid[c].velocity)
      Fluid[c].Conserved_t[irE] += work
      var cellVolume = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      acc += MACRO.dot(force, Fluid[c].velocity) * cellVolume
   end
   return acc
end

local __demand(__cuda, __leaf)
task AdjustHITSource([Fluid],
                     averageEnergySource : double)
where
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do
   __demand(__openmp)
   for c in Fluid do
      Fluid[c].Conserved_t[irE] -= averageEnergySource
   end
end

__demand(__inline)
task Exports.UpdateUsingHITForcing([Fluid],
                                   tiles : ispace(int3d),
                                   Fluid_Zones : zones_partitions(Fluid, tiles),
                                   Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                   Mix : MIX.Mixture,
                                   config : SCHEMA.Config,
                                   interior_volume : double)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_Interior} = Fluid_Zones;
   var {p_GradientGhosts,
        p_XFluxGhosts,     p_YFluxGhosts,     p_ZFluxGhosts,
        p_XDiffGhosts,     p_YDiffGhosts,     p_ZDiffGhosts,
        p_XDiffGradGhosts, p_YDiffGradGhosts, p_ZDiffGradGhosts} = Fluid_Ghost;

   -- Calculate pressure dilatation
   var averagePressureDilatation = 0.0
   __demand(__index_launch)
   for c in tiles do
      averagePressureDilatation += CalculateAveragePD(p_GradientGhosts[c], p_Interior[c], Fluid.bounds)
   end
   averagePressureDilatation /= interior_volume

   -- Calculate dissipation
   var averageDissipation = 0.0
   __demand(__index_launch)
   for c in tiles do
      averageDissipation += [mkAddDissipation("z")](p_ZDiffGhosts[c], p_ZDiffGradGhosts[c], p_ZFluxGhosts[c],
                                                    p_Interior[c], Fluid.bounds, Mix)
   end
   __demand(__index_launch)
   for c in tiles do
      averageDissipation += [mkAddDissipation("y")](p_YDiffGhosts[c], p_YDiffGradGhosts[c], p_YFluxGhosts[c],
                                                    p_Interior[c], Fluid.bounds, Mix)
   end
   __demand(__index_launch)
   for c in tiles do
      averageDissipation += [mkAddDissipation("x")](p_XDiffGhosts[c], p_XDiffGradGhosts[c], p_XFluxGhosts[c],
                                                    p_Interior[c], Fluid.bounds, Mix)
   end
   averageDissipation /= interior_volume

   -- Calculate kinetic energy
   var averageKineticEnergy = 0.0
   __demand(__index_launch)
   for c in tiles do
      averageKineticEnergy += STAT.CalculateAverageKineticEnergy(p_Interior[c])
   end
   averageKineticEnergy = averageKineticEnergy/interior_volume

   -- Add forcing for energy and momentum
   var averageEnergySource = 0.0
   __demand(__index_launch)
   for c in tiles do
      averageEnergySource += AddHITSource(p_Interior[c],
                                          averageDissipation,
                                          averageKineticEnergy,
                                          averagePressureDilatation,
                                          config.Flow.turbForcing)
   end
   averageEnergySource /= interior_volume

   -- Make sure that we are not adding energy to the system
   __demand(__index_launch)
   for c in tiles do
      AdjustHITSource(p_Interior[c], averageEnergySource)
   end
end

-------------------------------------------------------------------------------
-- BUFFER ZONE ROUTINES
-------------------------------------------------------------------------------

-- Buffer zone source; see Freund, AIAA (1997)
__demand(__cuda, __leaf)
task Exports.AddBufferZoneSource([Fluid],
                                 mix    : MIX.Mixture,
                                 bBox   : bBoxType,
                                 config : SCHEMA.Config)
where
   reads(Fluid.centerCoordinates),
   reads(Fluid.Conserved),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do

   -- Target mixture
   var mixtureTarget = config.BC.bufferZone.u.Basic.XiTarget
   var XiTarget : double[nSpec]
   for k = 0, mixtureTarget.Species.length do
      var Species = mixtureTarget.Species.values[k]
      XiTarget[MIX.FindSpecies(Species.Name, &mix)] = Species.MolarFrac
   end

   -- Target state (zero velocity assumed)
   var pTarget = config.BC.bufferZone.u.Basic.pTarget
   var TTarget = config.BC.bufferZone.u.Basic.TTarget
   var mixWTarget = MIX.GetMolarWeightFromXi(XiTarget, &mix)
   var rhoTarget = MIX.GetRho(pTarget,TTarget,mixWTarget, &mix)
   var YiTarget : double[nSpec]
   for k = 0,nSpec do
      YiTarget[k] =  MIX.GetSpeciesMolarWeight(k, &mix)/mixWTarget * XiTarget[k]
   end
   var rhoeTarget = rhoTarget * MIX.GetInternalEnergy(TTarget,YiTarget, &mix)

   ---- Grid extents
   var xmin = bBox.v0[0]
   var xmax = bBox.v1[0]
   var ymin = bBox.v0[1]
   var ymax = bBox.v2[1]
   var zmin = bBox.v0[2]
   var zmax = bBox.v4[2]

   -- Buffer parameters
   var xBufferLength = config.BC.bufferZone.u.Basic.xBufferLength
   var yBufferLength = config.BC.bufferZone.u.Basic.yBufferLength
   var zBufferLength = config.BC.bufferZone.u.Basic.zBufferLength
   var maxAmp = config.BC.bufferZone.u.Basic.maxAmplitude
   var n = config.BC.bufferZone.u.Basic.power
   var xLeft = xmin+xBufferLength
   var yLeft = ymin+yBufferLength
   var zLeft = zmin+zBufferLength
   var xRight = xmax-xBufferLength
   var yRight = ymax-yBufferLength
   var zRight = zmax-zBufferLength

   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0]
      var y = Fluid[c].centerCoordinates[1]
      var z = Fluid[c].centerCoordinates[2]

      -- Determine source amplitude
      var sigxLeft = pow(max(xLeft-x,0)/(xLeft-xmin),2)
      var sigxRight = pow(max(x-xRight,0)/(xmax-xRight),2)
      var sigx = max(sigxLeft,sigxRight)

      var sigyLeft = pow(max(yLeft-y,0)/(yLeft-ymin),2)
      var sigyRight = pow(max(y-yRight,0)/(ymax-yRight),2)
      var sigy = max(sigyLeft,sigyRight)

      var sigzLeft = pow(max(zLeft-z,0)/(zLeft-zmin),2)
      var sigzRight = pow(max(z-zRight,0)/(zmax-zRight),2)
      var sigz = max(sigzLeft,sigzRight)

      --if (x <= xLeft) then
         --sigx = maxAmp * (xLeft-x)/(xLeft-xmin) * (xLeft-x)/(xLeft-xmin)
      --elseif (x >= xRight) then
         --sigx = maxAmp * (x-xRight)/(xmax-xRight) * (x-xRight)/(xmax-xRight)
      --end
      --if (y <= yLeft) then
         --sigy = maxAmp * (yLeft-y)/(yLeft-ymin) * (yLeft-y)/(yLeft-ymin)
      --elseif (y >= yRight) then
         --sigy = maxAmp * (y-yRight)/(ymax-yRight) * (y-yRight)/(ymax-yRight)
      --end
      --if (z <= zLeft) then
         --sigz = maxAmp * (zLeft-z)/(zLeft-zmin) * (zLeft-z)/(zLeft-zmin)
      --elseif (z >= zRight) then
         --sigz = maxAmp * (z-zRight)/(zmax-zRight) * (z-zRight)/(zmax-zRight)
      --end
      var sig = maxAmp * max(sigx,max(sigy,sigz))

      -- Add source to RHS
      for k = 0,nSpec do
         Fluid[c].Conserved_t[k] += sig * (rhoTarget*YiTarget[k] - Fluid[c].Conserved[k])
      end
      for k = 0,3 do
         Fluid[c].Conserved_t[irU+k] += sig * (0.0 - Fluid[c].Conserved[irU+k])
      end
      Fluid[c].Conserved_t[irE] += sig * (rhoeTarget - Fluid[c].Conserved[irE])
   end
end

return Exports end

