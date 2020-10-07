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

return function(SCHEMA, MIX, METRIC, Fluid_columns, DEBUG_OUTPUT) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local TYPES = terralib.includec("prometeo_types.h", {"-DEOS="..os.getenv("EOS")})

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

-------------------------------------------------------------------------------
-- MIXTURE PROPERTIES ROUTINES
-------------------------------------------------------------------------------

extern task Exports.UpdatePropertiesFromPrimitive(Fluid    : region(ispace(int3d), Fluid_columns),
                                                  ModCells : region(ispace(int3d), Fluid_columns),
                                                  mix : MIX.Mixture)
where
   ModCells <= Fluid,
   reads(Fluid.[Primitives]),
   writes(Fluid.MassFracs),
   writes(Fluid.[Properties])
end

Exports.UpdatePropertiesFromPrimitive:set_calling_convention(regentlib.convention.manual())
Exports.UpdatePropertiesFromPrimitive:set_task_id(TYPES.TID_UpdatePropertiesFromPrimitive)

-------------------------------------------------------------------------------
-- CONSERVED TO PRIMITIVE/PRIMITIVE TO CONSERVED ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdateConservedFromPrimitive(Fluid    : region(ispace(int3d), Fluid_columns),
                                          ModCells : region(ispace(int3d), Fluid_columns),
                                          mix : MIX.Mixture)
where
   reads(Fluid.{MassFracs, temperature, velocity}),
   reads(Fluid.rho),
   writes(Fluid.Conserved)
do
   __demand(__openmp)
   for c in ModCells do
      var rhoYi = MIX.GetRhoYiFromYi(Fluid[c].rho, Fluid[c].MassFracs)
      var Conserved : double[nEq]
      for i=0, nSpec do
         Conserved[i] = rhoYi[i]
      end
      for i=0, 3 do
         Conserved[i+irU] = Fluid[c].rho*Fluid[c].velocity[i]
      end
      Conserved[irE] = (Fluid[c].rho*(0.5*MACRO.dot(Fluid[c].velocity, Fluid[c].velocity)
                     + MIX.GetInternalEnergy(Fluid[c].temperature, Fluid[c].MassFracs, mix)))
      -- TODO this trick is needed because of the bug in the write privileges in regent
      Fluid[c].Conserved = Conserved
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdatePrimitiveFromConserved(Fluid    : region(ispace(int3d), Fluid_columns),
                                          ModCells : region(ispace(int3d), Fluid_columns),
                                          mix : MIX.Mixture)
where
   reads(Fluid.Conserved),
   reads(Fluid.temperature),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in ModCells do
      var rhoYi : double[nSpec]
      for i=0, nSpec do
         rhoYi[i] = Fluid[c].Conserved[i]
      end
      var rho = MIX.GetRhoFromRhoYi(rhoYi)
      var Yi = MIX.GetYi(rho, rhoYi)
      Yi = MIX.ClipYi(Yi)
      var MixW = MIX.GetMolarWeightFromYi(Yi, mix)
      Fluid[c].MolarFracs = MIX.GetMolarFractions(MixW, Yi, mix)
      var rhoInv = 1.0/rho
      var velocity = array(Fluid[c].Conserved[irU+0]*rhoInv,
                           Fluid[c].Conserved[irU+1]*rhoInv,
                           Fluid[c].Conserved[irU+2]*rhoInv)
      Fluid[c].velocity = velocity
      var kineticEnergy = (0.5*MACRO.dot(velocity, velocity))
      var InternalEnergy = Fluid[c].Conserved[irE]*rhoInv - kineticEnergy
      Fluid[c].temperature = MIX.GetTFromInternalEnergy(InternalEnergy, Fluid[c].temperature, Yi, mix)
      Fluid[c].pressure    = MIX.GetPFromRhoAndT(rho, MixW, Fluid[c].temperature)
   end
end

-------------------------------------------------------------------------------
-- GRADIENT ROUTINES
-------------------------------------------------------------------------------

extern task Exports.GetVelocityGradients(Ghost : region(ispace(int3d), Fluid_columns),
                                         Fluid : region(ispace(int3d), Fluid_columns),
                                         Fluid_bounds : rect3d)
where
   reads(Ghost.velocity),
   reads(Fluid.{nType_x, nType_y, nType_z}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   writes(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ})
end
Exports.GetVelocityGradients:set_calling_convention(regentlib.convention.manual())
Exports.GetVelocityGradients:set_task_id(TYPES.TID_GetVelocityGradients)

extern task Exports.GetTemperatureGradients(Ghost : region(ispace(int3d), Fluid_columns),
                                            Fluid : region(ispace(int3d), Fluid_columns),
                                            Fluid_bounds : rect3d)
where
   reads(Ghost.temperature),
   reads(Fluid.{nType_x, nType_y, nType_z}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   writes(Fluid.temperatureGradient)
end
Exports.GetTemperatureGradients:set_calling_convention(regentlib.convention.manual())
Exports.GetTemperatureGradients:set_task_id(TYPES.TID_GetTemperatureGradient)

return Exports end

