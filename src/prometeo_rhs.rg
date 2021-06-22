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

return function(SCHEMA, MIX, METRIC, TYPES, Fluid_columns,
                zones_partitions, ghost_partitions,
                ATOMIC) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local UTIL = require 'util-desugared'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

-- Atomic switch
local Fluid = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "Fluid")
local coherence_mode
if ATOMIC then
   coherence_mode = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
else
   coherence_mode = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
end

-------------------------------------------------------------------------------
-- EULER FLUX-DIVERGENCE ROUTINES
-------------------------------------------------------------------------------

local mkUpdateUsingHybridEulerFlux = terralib.memoize(function(dir)
   local UpdateUsingHybridEulerFlux

--   local FluxC
   local shockSensor
   local nType
   local m_e
   if (dir == "x") then
--      FluxC = "FluxXCorr"
      shockSensor = "shockSensorX"
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
--      FluxC = "FluxYCorr"
      shockSensor = "shockSensorY"
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
--      FluxC = "FluxZCorr"
      shockSensor = "shockSensorZ"
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end

   extern task UpdateUsingHybridEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                          SensorGhost : region(ispace(int3d), Fluid_columns),
                                          DiffGhost : region(ispace(int3d), Fluid_columns),
                                          FluxGhost : region(ispace(int3d), Fluid_columns),
                                          [Fluid],
                                          ModCells : region(ispace(int3d), Fluid_columns),
                                          Fluid_bounds : rect3d,
                                          mix : MIX.Mixture)
   where
      reads(EulerGhost.Conserved),
      reads(EulerGhost.rho),
      reads(EulerGhost.MassFracs),
      reads(EulerGhost.velocity),
      reads(EulerGhost.pressure),
      reads(EulerGhost.SoS),
      reads(SensorGhost.[shockSensor]),
      reads(DiffGhost.temperature),
      reads(FluxGhost.[nType]),
      reads(Fluid.[m_e]),
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

local mkUpdateUsingTENOAEulerFlux = terralib.memoize(function(dir)
   local UpdateUsingTENOAEulerFlux

--   local FluxC
   local nType
   local m_e
   if (dir == "x") then
--      FluxC = "FluxXCorr"
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
--      FluxC = "FluxYCorr"
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
--      FluxC = "FluxZCorr"
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end

   extern task UpdateUsingTENOAEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                         DiffGhost : region(ispace(int3d), Fluid_columns),
                                         FluxGhost : region(ispace(int3d), Fluid_columns),
                                         [Fluid],
                                         ModCells : region(ispace(int3d), Fluid_columns),
                                         Fluid_bounds : rect3d,
                                         mix : MIX.Mixture)
   where
      reads(EulerGhost.Conserved),
      reads(EulerGhost.rho),
      reads(EulerGhost.velocity),
      reads(EulerGhost.pressure),
      reads(EulerGhost.SoS),
      reads(DiffGhost.{MassFracs, temperature}),
      reads(FluxGhost.[nType]),
      reads(Fluid.[m_e]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   --for k, v in pairs(UpdateUsingFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if     (dir == "x") then
      UpdateUsingTENOAEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOAEulerFluxX)
   elseif (dir == "y") then
      UpdateUsingTENOAEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOAEulerFluxY)
   elseif (dir == "z") then
      UpdateUsingTENOAEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOAEulerFluxZ)
   end
   return UpdateUsingTENOAEulerFlux
end)

local mkUpdateUsingTENOLADEulerFlux = terralib.memoize(function(dir)
   local UpdateUsingTENOLADEulerFlux

--   local FluxC
   local nType
   local m_e
   if (dir == "x") then
--      FluxC = "FluxXCorr"
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
--      FluxC = "FluxYCorr"
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
--      FluxC = "FluxZCorr"
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end

   extern task UpdateUsingTENOLADEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                           DiffGhost : region(ispace(int3d), Fluid_columns),
                                           FluxGhost : region(ispace(int3d), Fluid_columns),
                                           [Fluid],
                                           ModCells : region(ispace(int3d), Fluid_columns),
                                           Fluid_bounds : rect3d,
                                           mix : MIX.Mixture)
   where
      reads(EulerGhost.Conserved),
      reads(EulerGhost.rho),
      reads(EulerGhost.velocity),
      reads(EulerGhost.pressure),
      reads(EulerGhost.SoS),
      reads(DiffGhost.{MassFracs, temperature}),
      reads(FluxGhost.[nType]),
      reads(Fluid.[m_e]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   --for k, v in pairs(UpdateUsingFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if     (dir == "x") then
      UpdateUsingTENOLADEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOLADEulerFluxX)
   elseif (dir == "y") then
      UpdateUsingTENOLADEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOLADEulerFluxY)
   elseif (dir == "z") then
      UpdateUsingTENOLADEulerFlux:set_task_id(TYPES.TID_UpdateUsingTENOLADEulerFluxZ)
   end
   return UpdateUsingTENOLADEulerFlux
end)

local mkUpdateUsingSkewSymmetricEulerFlux = terralib.memoize(function(dir)
   local UpdateUsingSkewSymmetricEulerFlux

--   local FluxC
   local nType
   local m_e
   if (dir == "x") then
--      FluxC = "FluxXCorr"
      nType = "nType_x"
      m_e   = "dcsi_e"
   elseif (dir == "y") then
--      FluxC = "FluxYCorr"
      nType = "nType_y"
      m_e   = "deta_e"
   elseif (dir == "z") then
--      FluxC = "FluxZCorr"
      nType = "nType_z"
      m_e   = "dzet_e"
   else assert(false) end

   extern task UpdateUsingSkewSymmetricEulerFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                                 FluxGhost : region(ispace(int3d), Fluid_columns),
                                                 [Fluid],
                                                 ModCells : region(ispace(int3d), Fluid_columns),
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
task Exports.UpdateUsingEulerFlux(Fluid : region(ispace(int3d), Fluid_columns),
                                  tiles : ispace(int3d),
                                  Fluid_Zones : zones_partitions(Fluid, tiles),
                                  Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                  Mix : MIX.Mixture,
                                  config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All, p_x_divg, p_y_divg, p_z_divg} = Fluid_Zones;
   var {p_XFluxGhosts,     p_YFluxGhosts,   p_ZFluxGhosts,
        p_XDiffGhosts,    p_YDiffGhosts,    p_ZDiffGhosts,
        p_XEulerGhosts2,  p_YEulerGhosts2,  p_ZEulerGhosts2,
        p_XSensorGhosts2, p_YSensorGhosts2, p_ZSensorGhosts2} = Fluid_Ghost;

   if (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_Hybrid) then
      -- Call tasks with hybrid scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingHybridEulerFlux("z")](p_ZEulerGhosts2[c], p_ZSensorGhosts2[c], p_ZDiffGhosts[c],
                                             p_ZFluxGhosts[c], p_All[c], p_z_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingHybridEulerFlux("y")](p_YEulerGhosts2[c], p_YSensorGhosts2[c], p_YDiffGhosts[c],
                                             p_YFluxGhosts[c], p_All[c], p_y_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingHybridEulerFlux("x")](p_XEulerGhosts2[c], p_XSensorGhosts2[c], p_XDiffGhosts[c],
                                             p_XFluxGhosts[c], p_All[c], p_x_divg[c], Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_TENOA) then
      -- Call tasks with TENO-A scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOAEulerFlux("z")](p_ZEulerGhosts2[c], p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                            p_All[c], p_z_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOAEulerFlux("y")](p_YEulerGhosts2[c], p_YDiffGhosts[c], p_YFluxGhosts[c],
                                            p_All[c], p_y_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOAEulerFlux("x")](p_XEulerGhosts2[c], p_XDiffGhosts[c], p_XFluxGhosts[c],
                                            p_All[c], p_x_divg[c], Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_TENOLAD) then
      -- Call tasks with TENO-A scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOLADEulerFlux("z")](p_ZEulerGhosts2[c], p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                              p_All[c], p_z_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOLADEulerFlux("y")](p_YEulerGhosts2[c], p_YDiffGhosts[c], p_YFluxGhosts[c],
                                              p_All[c], p_y_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingTENOLADEulerFlux("x")](p_XEulerGhosts2[c], p_XDiffGhosts[c], p_XFluxGhosts[c],
                                              p_All[c], p_x_divg[c], Fluid.bounds, Mix)
      end

   elseif (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_SkewSymmetric) then
      -- Call tasks with SkewSymmetric scheme
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingSkewSymmetricEulerFlux("z")](p_ZEulerGhosts2[c], p_ZFluxGhosts[c],
                                                    p_All[c], p_z_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingSkewSymmetricEulerFlux("y")](p_YEulerGhosts2[c], p_YFluxGhosts[c],
                                                    p_All[c], p_y_divg[c], Fluid.bounds, Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateUsingSkewSymmetricEulerFlux("x")](p_XEulerGhosts2[c], p_XFluxGhosts[c],
                                                    p_All[c], p_x_divg[c], Fluid.bounds, Mix)
      end
   end
end

-------------------------------------------------------------------------------
-- DIFFUSION FLUX-DIVERGENCE ROUTINES
-------------------------------------------------------------------------------

local mkUpdateUsingDiffusionFlux = terralib.memoize(function(dir)
   local UpdateUsingDiffusionFlux

   local nType
   local m_d
   local m_s
   local vGrad1
   local vGrad2
   if (dir == "x") then
      nType  = "nType_x"
      m_d    = "dcsi_d"
      m_s    = "dcsi_s"
      vGrad1 = "velocityGradientY"
      vGrad2 = "velocityGradientZ"
   elseif (dir == "y") then
      nType  = "nType_y"
      m_d    = "deta_d"
      m_s    = "deta_s"
      vGrad1 = "velocityGradientX"
      vGrad2 = "velocityGradientZ"
   elseif (dir == "z") then
      nType  = "nType_z"
      m_d    = "dzet_d"
      m_s    = "dzet_s"
      vGrad1 = "velocityGradientX"
      vGrad2 = "velocityGradientY"
   else assert(false) end

   extern task UpdateUsingDiffusionFlux(DiffGhost : region(ispace(int3d), Fluid_columns),
                                        FluxGhost : region(ispace(int3d), Fluid_columns),
                                        [Fluid],
                                        ModCells : region(ispace(int3d), Fluid_columns),
                                        Fluid_bounds : rect3d,
                                        mix : MIX.Mixture)
   where
      reads(DiffGhost.Conserved),
      reads(DiffGhost.{MolarFracs, temperature, velocity}),
      reads(DiffGhost.{rho, mu, lam, Di}),
      reads(DiffGhost.{[vGrad1], [vGrad2]}),
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
task Exports.UpdateUsingDiffusionFlux(Fluid : region(ispace(int3d), Fluid_columns),
                                      tiles : ispace(int3d),
                                      Fluid_Zones : zones_partitions(Fluid, tiles),
                                      Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                      Mix : MIX.Mixture,
                                      config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_All, p_x_divg, p_y_divg, p_z_divg} = Fluid_Zones;
   var {p_XFluxGhosts,     p_YFluxGhosts,   p_ZFluxGhosts,
        p_XDiffGhosts,    p_YDiffGhosts,    p_ZDiffGhosts} = Fluid_Ghost;

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingDiffusionFlux("z")](p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                        p_All[c], p_z_divg[c], Fluid.bounds, Mix)
   end

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingDiffusionFlux("y")](p_YDiffGhosts[c], p_YFluxGhosts[c],
                                        p_All[c], p_y_divg[c], Fluid.bounds, Mix)
   end

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingDiffusionFlux("x")](p_XDiffGhosts[c], p_XFluxGhosts[c],
                                        p_All[c], p_x_divg[c], Fluid.bounds, Mix)
   end
end

-------------------------------------------------------------------------------
-- NSCBC-FLUX ROUTINES
-------------------------------------------------------------------------------
-- Adds NSCBC fluxes to the inflow cells
Exports.mkUpdateUsingFluxNSCBCInflow = terralib.memoize(function(dir)
   local UpdateUsingFluxNSCBCInflow

   local nType
   local m_d
   local vGrad
   if     dir == "xNeg" then
      nType = "nType_x"
      m_d = "dcsi_d"
      vGrad = "velocityGradientX"
   elseif dir == "yNeg" then
      nType = "nType_y"
      m_d = "deta_d"
      vGrad = "velocityGradientY"
   elseif dir == "yPos" then
      nType = "nType_y"
      m_d = "deta_d"
      vGrad = "velocityGradientY"
   end

   extern task UpdateUsingFluxNSCBCInflow(Fluid    : region(ispace(int3d), Fluid_columns),
                                          Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                          mix : MIX.Mixture)
   where
      reads(Fluid.[nType]),
      reads(Fluid.[m_d]),
      reads(Fluid.{rho, SoS}),
      reads(Fluid.{MassFracs, pressure, temperature, velocity}),
      reads(Fluid.[vGrad]),
      reads(Fluid.{dudtBoundary, dTdtBoundary}),
      reads writes(Fluid.Conserved_t)
   end

   if     dir == "xNeg" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowXNeg)
   elseif dir == "yNeg" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowYNeg)
   elseif dir == "yPos" then
      UpdateUsingFluxNSCBCInflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCInflowYPos)
   else assert(false) end

   return UpdateUsingFluxNSCBCInflow
end)

-- Adds NSCBC fluxes to the outflow cells
Exports.mkUpdateUsingFluxNSCBCOutflow  = terralib.memoize(function(dir)
   local UpdateUsingFluxNSCBCOutflow

   local nType
   local m_d
   if     dir == "xPos" then
      nType = "nType_x"
      m_d = "dcsi_d"
   elseif dir == "yNeg" then
      nType = "nType_y"
      m_d = "deta_d"
   elseif dir == "yPos" then
      nType = "nType_y"
      m_d = "deta_d"
   end
   extern task UpdateUsingFluxNSCBCOutflow([Fluid],
                                           Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                           mix : MIX.Mixture,
                                           MaxMach : double,
                                           LengthScale : double,
                                           Pinf : double)
   where
      reads(Fluid.[nType]),
      reads(Fluid.[m_d]),
      reads(Fluid.{rho, mu, SoS}),
      reads(Fluid.{MassFracs, pressure, temperature, velocity}),
      reads(Fluid.Conserved),
      reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   if     dir == "xPos" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowXPos)
   elseif dir == "yNeg" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowYNeg)
   elseif dir == "yPos" then
      UpdateUsingFluxNSCBCOutflow:set_task_id(TYPES.TID_UpdateUsingFluxNSCBCOutflowYPos)
   end
--   for k, v in pairs(UpdateUsingFluxNSCBCOutflow:get_params_struct():getentries()) do
--      print(k, v)
--      for k2, v2 in pairs(v) do print(k2, v2) end
--   end
   return UpdateUsingFluxNSCBCOutflow
end)

-------------------------------------------------------------------------------
-- FORCING ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.AddBodyForces([Fluid],
                           ModCells : region(ispace(int3d), Fluid_columns),
                           Flow_bodyForce : double[3])
where
   reads(Fluid.{rho, velocity}),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do
   __demand(__openmp)
   for c in ModCells do
      for i=0, 3 do
         Fluid[c].Conserved_t[irU+i] += Fluid[c].rho*Flow_bodyForce[i]
      end
      Fluid[c].Conserved_t[irE] += Fluid[c].rho*MACRO.dot(Flow_bodyForce, Fluid[c].velocity)
   end
end

---------------------------------------------------------------------------------
---- CORRECTION ROUTINES
---------------------------------------------------------------------------------
--
--local __demand(__inline)
--task isValid(Conserved : double[nEq],
--             mix : MIX.Mixture)
--   var valid = [UTIL.mkArrayConstant(nSpec+1, true)];
--   var rhoYi : double[nSpec]
--   for i=0, nSpec do
--      if Conserved[i] < 0.0 then valid[i] = false end
--      rhoYi[i] = Conserved[i]
--   end
--   var rho = MIX.GetRhoFromRhoYi(rhoYi)
--   var Yi = MIX.GetYi(rho, rhoYi)
--   var rhoInv = 1.0/rho
--   var velocity = array(Conserved[irU+0]*rhoInv,
--                        Conserved[irU+1]*rhoInv,
--                        Conserved[irU+2]*rhoInv)
--   var kineticEnergy = (0.5*MACRO.dot(velocity, velocity))
--   var InternalEnergy = Conserved[irE]*rhoInv - kineticEnergy
--   valid[nSpec] = MIX.isValidInternalEnergy(InternalEnergy, Yi, mix)
--   return valid
--end
--
--Exports.mkCorrectUsingFlux = terralib.memoize(function(dir)
--   local CorrectUsingFlux
--
--   local Flux
--   local nType
--   if (dir == "x") then
--      Flux = "FluxXCorr"
--      nType = "nType_z"
--   elseif (dir == "y") then
--      Flux = "FluxYCorr"
--      nType = "nType_y"
--   elseif (dir == "z") then
--      Flux = "FluxZCorr"
--      nType = "nType_z"
--   else assert(false) end
--   local cm1_d = function(r, c, b) return METRIC.GetCm1(dir, c, rexpr [r][c].[nType] end, b) end
--   local cp1_d = function(r, c, b) return METRIC.GetCp1(dir, c, rexpr [r][c].[nType] end, b) end
--
--   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
--   task CorrectUsingFlux(Ghost : region(ispace(int3d), Fluid_columns),
--                         [Fluid],
--                         ModCells : region(ispace(int3d), Fluid_columns),
--                         Fluid_bounds : rect3d,
--                         mix : MIX.Mixture)
--   where
--      reads(Fluid.[nType]),
--      reads(Ghost.Conserved_hat),
--      reads(Ghost.[Flux]),
--      reads writes(Fluid.Conserved_t),
--      [coherence_mode]
--   do
--      __demand(__openmp)
--      for c in ModCells do
--         -- Stencil
--         var cm1 = [cm1_d(rexpr Fluid end, rexpr c end, rexpr Fluid_bounds end)];
--         var cp1 = [cp1_d(rexpr Fluid end, rexpr c end, rexpr Fluid_bounds end)];
--
--         -- Do derivatives need to be corrected?
--         var correctC   = false
--         var correctCM1 = false
--
--         var valid_cm1 = isValid(Ghost[cm1].Conserved_hat, [mix])
--         var valid_c   = isValid(Ghost[c  ].Conserved_hat, [mix])
--         var valid_cp1 = isValid(Ghost[cp1].Conserved_hat, [mix])
--
--         for i=0, nSpec+1 do
--            if not (valid_cp1[i] and valid_c[i]) then correctC   = true end
--            if not (valid_cm1[i] and valid_c[i]) then correctCM1 = true end
--         end
--
--         -- Correct using Flux on i-1 face
--         if correctCM1 then
--            -- Correct time derivatives using fluxes between cm1 and c
--            if not (valid_cm1[nSpec] and valid_c[nSpec]) then
--               -- Temeperature is going south
--               -- Correct everything
--               for i=0, nEq do
--                  Fluid[c].Conserved_t[i] -= Ghost[cm1].[Flux][i]
--               end
--            else
--               for i=0, nSpec do
--                  if not (valid_cm1[i] and valid_c[i]) then
--                     -- Correct single species flux
--                     Fluid[c].Conserved_t[i] -= Ghost[cm1].[Flux][i]
--                  end
--               end
--            end
--         end
--
--         -- Correct using Flux on i face
--         if correctC  then
--            -- Correct time derivatives using fluxes between c and cp1
--            if not (valid_cp1[nSpec] and valid_c[nSpec]) then
--               -- Temeperature is going south
--               -- Correct everything
--               for i=0, nEq do
--                  Fluid[c].Conserved_t[i] += Ghost[c].[Flux][i]
--               end
--            else
--               for i=0, nSpec do
--                  if not (valid_cp1[i] and valid_c[i]) then
--                     -- Correct single species flux
--                     Fluid[c].Conserved_t[i] += Ghost[c].[Flux][i]
--                  end
--               end
--            end
--         end
--      end
--   end
--   return CorrectUsingFlux
--end)

return Exports end

