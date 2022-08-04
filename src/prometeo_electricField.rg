-- Copyright (c) "2020, by Centre Européen de Recherche et de Formation Avancée en Calcul Scientifiq
--               Developer: Mario Di Renzo
--               Affiliation: Centre Européen de Recherche et de Formation Avancée en Calcul Scientifique
--               URL: https://cerfacs.fr
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

return function(SCHEMA, MIX, TYPES,
                zones_partitions, ghost_partitions,
                ATOMIC) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c

local Fluid_columns = TYPES.Fluid_columns

-- Atomic switch
local Fluid = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "Fluid")
local coherence_mode
if ATOMIC then
   coherence_mode = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
else
   coherence_mode = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
end

-------------------------------------------------------------------------------
-- POISSON SOLVER OBJECTS
-------------------------------------------------------------------------------
local SOLVER = (require "Poisson")(SCHEMA, MIX, TYPES, Fluid_columns,
                                   terralib.newlist({"rho", "MolarFracs"}), TYPES.TID_performDirFFTFromMix,
                                   "electricPotential",
                                   "dcsi_d", "deta_d", "dzet_d", "deta_s", "nType_y")

Exports.mkDataList  = SOLVER.mkDataList
Exports.DeclSymbols = SOLVER.DeclSymbols
Exports.Init        = SOLVER.Init
Exports.Solve       = SOLVER.Solve
Exports.Cleanup     = SOLVER.Cleanup

-------------------------------------------------------------------------------
-- ELECTRIC FIELD TASK
-------------------------------------------------------------------------------
local extern task GetElectricField(Ghost : region(ispace(int3d), Fluid_columns),
                                   [Fluid],
                                   Fluid_bounds : rect3d)
where
   reads(Ghost.electricPotential),
   reads(Fluid.{nType_x, nType_y, nType_z}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   writes(Fluid.electricField)
end
GetElectricField:set_task_id(TYPES.TID_GetElectricField)

function Exports.UpdateElectricField(solverData, Fluid, Fluid_Zones, Fluid_Ghost, tiles, Fluid_bounds, Mix, config) return rquote

   -- Solve the electric potential
   [EFIELD.Solve(solverData, tiles, Mix, config)];

   -- Update the electric field
   var {p_All} = Fluid_Zones;
   var {p_GradientGhosts} = Fluid_Ghost;
   __demand(__index_launch)
   for c in tiles do
      GetElectricField(p_GradientGhosts[c], p_All[c], Fluid_bounds)
   end
end end

-------------------------------------------------------------------------------
-- ION DRIFT FLUX-DIVERGENCE ROUTINES
-------------------------------------------------------------------------------

if (MIX.nIons > 0) then
local mkUpdateUsingIonDriftFlux = terralib.memoize(function(dir)
   local UpdateUsingIonDriftFlux

   local nType
   local m_e
   local m_d
   if (dir == "x") then
      nType  = "nType_x"
      m_e    = "dcsi_e"
      m_d    = "dcsi_d"
   elseif (dir == "y") then
      nType  = "nType_y"
      m_e    = "deta_e"
      m_d    = "deta_d"
   elseif (dir == "z") then
      nType  = "nType_z"
      m_e    = "dzet_e"
      m_d    = "dzet_d"
   else assert(false) end

   extern task UpdateUsingIonDriftFlux(EulerGhost : region(ispace(int3d), Fluid_columns),
                                       DiffGhost : region(ispace(int3d), Fluid_columns),
                                       FluxGhost : region(ispace(int3d), Fluid_columns),
                                       [Fluid],
                                       Fluid_bounds : rect3d,
                                       mix : MIX.Mixture)
   where
      reads(EulerGhost.Conserved),
      reads(EulerGhost.electricField),
      reads(EulerGhost.Ki),
      reads(DiffGhost.MassFracs),
      reads(FluxGhost.[nType]),
      reads(Fluid.[m_e]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   end
   --for k, v in pairs(UpdateUsingDiffusionFlux:get_params_struct():getentries()) do
   --   print(k, v)
   --   for k2, v2 in pairs(v) do print(k2, v2) end
   --end
   if     (dir == "x") then
      UpdateUsingIonDriftFlux:set_task_id(TYPES.TID_UpdateUsingIonDriftFluxX)
   elseif (dir == "y") then
      UpdateUsingIonDriftFlux:set_task_id(TYPES.TID_UpdateUsingIonDriftFluxY)
   elseif (dir == "z") then
      UpdateUsingIonDriftFlux:set_task_id(TYPES.TID_UpdateUsingIonDriftFluxZ)
   end
   return UpdateUsingIonDriftFlux
end)

__demand(__inline)
task Exports.UpdateUsingIonDriftFlux([Fluid],
                                     tiles : ispace(int3d),
                                     Fluid_Zones : zones_partitions(Fluid, tiles),
                                     Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                     Mix : MIX.Mixture,
                                     config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   var {p_All, p_x_divg, p_y_divg, p_z_divg} = Fluid_Zones;
   var {p_XFluxGhosts,  p_YFluxGhosts,  p_ZFluxGhosts,
        p_XDiffGhosts,  p_YDiffGhosts,  p_ZDiffGhosts,
        p_XEulerGhosts, p_YEulerGhosts, p_ZEulerGhosts} = Fluid_Ghost;

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingIonDriftFlux("z")](p_ZEulerGhosts[c], p_ZDiffGhosts[c], p_ZFluxGhosts[c],
                                       p_z_divg[c], Fluid.bounds, Mix)
   end

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingIonDriftFlux("y")](p_YEulerGhosts[c], p_YDiffGhosts[c], p_YFluxGhosts[c],
                                       p_y_divg[c], Fluid.bounds, Mix)
   end

   __demand(__index_launch)
   for c in tiles do
      [mkUpdateUsingIonDriftFlux("x")](p_XEulerGhosts[c], p_XDiffGhosts[c], p_XFluxGhosts[c],
                                       p_x_divg[c], Fluid.bounds, Mix)
   end
end
end

-------------------------------------------------------------------------------
-- ELECTRIC SOURCE TERM TASKS
-------------------------------------------------------------------------------

local Prop = terralib.newlist({"rho", "Di"})
if (MIX.nIons > 0) then
   Prop:insert("Ki")
end
extern task Exports.AddIonWindSources(GradGhost : region(ispace(int3d), Fluid_columns),
                                      [Fluid],
                                      Fluid_bounds : rect3d,
                                      mix : MIX.Mixture)
where
   reads(GradGhost.MolarFracs),
   reads(Fluid.{nType_x, nType_y, nType_z}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   reads(Fluid.{velocity, electricField}),
   reads(Fluid.[Prop]),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
end
Exports.AddIonWindSources:set_task_id(TYPES.TID_AddIonWindSources)

return Exports end

