-- Copyright (c) "2019, by Stanford University
--               Developer: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
--                         HTR solver: An open-source exascale-oriented task-based
--                         multi-GPU high-order code for hypersonic aerothermodynamics.
--                         Computer Physics Communications (In Press), 107262"
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

return function(SCHEMA, MIX, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local OP    = require "prometeo_operators"

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

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdatePropertiesFromPrimitive(Fluid    : region(ispace(int3d), Fluid_columns),
                                           ModCells : region(ispace(int3d), Fluid_columns),
                                           mix : MIX.Mixture)
where
   ModCells <= Fluid,
   reads(Fluid.[Primitives]),
   writes(Fluid.[Properties])
do
   __demand(__openmp)
   for c in ModCells do
      var MixW     = MIX.GetMolarWeightFromXi(Fluid[c].MolarFracs, mix)
      Fluid[c].rho = MIX.GetRho(Fluid[c].pressure, Fluid[c].temperature, MixW, mix)
      Fluid[c].mu  = MIX.GetViscosity(       Fluid[c].temperature,       Fluid[c].MolarFracs, mix)
      Fluid[c].lam = MIX.GetHeatConductivity(Fluid[c].temperature,       Fluid[c].MolarFracs, mix)
      Fluid[c].Di  = MIX.GetDiffusivity(Fluid[c].pressure, Fluid[c].temperature, MixW, Fluid[c].MolarFracs, mix)
      var Yi = MIX.GetMassFractions(MixW, Fluid[c].MolarFracs, mix)
      var gamma = MIX.GetGamma(Fluid[c].temperature, MixW, Yi, mix)
      Fluid[c].SoS = MIX.GetSpeedOfSound(Fluid[c].temperature, gamma, MixW, mix)
   end
end

-------------------------------------------------------------------------------
-- CONSERVED TO PRIMITIVE/PRIMITIVE TO CONSERVED ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdateConservedFromPrimitive(Fluid    : region(ispace(int3d), Fluid_columns),
                                          ModCells : region(ispace(int3d), Fluid_columns),
                                          mix : MIX.Mixture)
where
   reads(Fluid.[Primitives]),
   reads(Fluid.rho),
   writes(Fluid.Conserved)
do
   __demand(__openmp)
   for c in ModCells do
      var MixW  = MIX.GetMolarWeightFromXi(Fluid[c].MolarFracs, mix)
      var Yi    = MIX.GetMassFractions(MixW, Fluid[c].MolarFracs, mix)
      var rhoYi = MIX.GetRhoYiFromYi(Fluid[c].rho, Yi)
      var Conserved : double[nEq]
      for i=0, nSpec do
         Conserved[i] = rhoYi[i]
      end
      for i=0, 3 do
         Conserved[i+irU] = Fluid[c].rho*Fluid[c].velocity[i]
      end
      Conserved[irE] = (Fluid[c].rho*(0.5*MACRO.dot(Fluid[c].velocity, Fluid[c].velocity)
                     + MIX.GetInternalEnergy(Fluid[c].temperature, Yi, mix)))
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

__demand(__parallel, __cuda, __leaf)
task Exports.GetVelocityGradients(Fluid : region(ispace(int3d), Fluid_columns),
                                  Fluid_bounds : rect3d)
where
   reads(Fluid.{gradX, gradY, gradZ}),
   reads(Fluid.velocity),
   writes(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ})
do
   __demand(__openmp)
   for c in Fluid do

      -- X direction
      var cm1_x = (c+{-1, 0, 0}) % Fluid_bounds
      var cp1_x = (c+{ 1, 0, 0}) % Fluid_bounds
      Fluid[c].velocityGradientX = MACRO.vv_add(MACRO.vs_mul(MACRO.vv_sub(Fluid[c    ].velocity, Fluid[cm1_x].velocity), Fluid[c].gradX[0]),
                                                MACRO.vs_mul(MACRO.vv_sub(Fluid[cp1_x].velocity, Fluid[c    ].velocity), Fluid[c].gradX[1]))

      -- Y direction
      var cm1_y = (c+{ 0,-1, 0}) % Fluid_bounds
      var cp1_y = (c+{ 0, 1, 0}) % Fluid_bounds
      Fluid[c].velocityGradientY = MACRO.vv_add(MACRO.vs_mul(MACRO.vv_sub(Fluid[c    ].velocity, Fluid[cm1_y].velocity), Fluid[c].gradY[0]),
                                                MACRO.vs_mul(MACRO.vv_sub(Fluid[cp1_y].velocity, Fluid[c    ].velocity), Fluid[c].gradY[1]))


      -- Z direction
      var cm1_z = (c+{ 0, 0,-1}) % Fluid_bounds
      var cp1_z = (c+{ 0, 0, 1}) % Fluid_bounds
      Fluid[c].velocityGradientZ = MACRO.vv_add(MACRO.vs_mul(MACRO.vv_sub(Fluid[c    ].velocity, Fluid[cm1_z].velocity), Fluid[c].gradZ[0]),
                                                MACRO.vs_mul(MACRO.vv_sub(Fluid[cp1_z].velocity, Fluid[c    ].velocity), Fluid[c].gradZ[1]))
  end
end

__demand(__parallel, __cuda, __leaf)
task Exports.GetTemperatureGradients(Fluid : region(ispace(int3d), Fluid_columns),
                                     Fluid_bounds : rect3d)
where
   reads(Fluid.{gradX, gradY, gradZ}),
   reads(Fluid.temperature),
   writes(Fluid.temperatureGradient)
do
   __demand(__openmp)
   for c in Fluid do
      Fluid[c].temperatureGradient = array([OP.emitXderiv(rexpr Fluid end, 'temperature', c, rexpr Fluid_bounds end)],
                                           [OP.emitYderiv(rexpr Fluid end, 'temperature', c, rexpr Fluid_bounds end)],
                                           [OP.emitZderiv(rexpr Fluid end, 'temperature', c, rexpr Fluid_bounds end)])
  end
end

return Exports end

