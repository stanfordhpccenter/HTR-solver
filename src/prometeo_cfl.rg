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

return function(MIX, Fluid_columns) local Exports = {}

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local fabs = regentlib.fabs(double)

-------------------------------------------------------------------------------
-- STABILITY CONDITIONS ROUTINES
-------------------------------------------------------------------------------

local __demand(__inline)
task CalculateConvectiveSpectralRadius(Fluid : region(ispace(int3d), Fluid_columns), c : int3d)
where
   reads(Fluid.{cellWidth, velocity, SoS})
do
   return (max(max(((fabs(Fluid[c].velocity[0])+Fluid[c].SoS)/Fluid[c].cellWidth[0]),
                   ((fabs(Fluid[c].velocity[1])+Fluid[c].SoS)/Fluid[c].cellWidth[1])),
                   ((fabs(Fluid[c].velocity[2])+Fluid[c].SoS)/Fluid[c].cellWidth[2])))
end

local __demand(__inline)
task CalculateViscousSpectralRadius(Fluid : region(ispace(int3d), Fluid_columns), c : int3d)
where
   reads(Fluid.{cellWidth, rho, mu})
do
   var nu = Fluid[c].mu/Fluid[c].rho
   return ((max(max((nu/(Fluid[c].cellWidth[0]*Fluid[c].cellWidth[0])),
                    (nu/(Fluid[c].cellWidth[1]*Fluid[c].cellWidth[1]))),
                    (nu/(Fluid[c].cellWidth[2]*Fluid[c].cellWidth[2]))))*4.0)
end

local __demand(__inline)
task CalculateHeatConductionSpectralRadius(Fluid : region(ispace(int3d), Fluid_columns),
                                           c : int3d,
                                           mix : MIX.Mixture)
where
   reads(Fluid.cellWidth),
   reads(Fluid.{MassFracs, temperature}),
   reads(Fluid.{rho, lam})
do
   var cp   = MIX.GetHeatCapacity(Fluid[c].temperature, Fluid[c].MassFracs, mix)
   var DifT = (Fluid[c].lam/(cp*Fluid[c].rho))
   return ((max(max((DifT/(Fluid[c].cellWidth[0]*Fluid[c].cellWidth[0])),
                    (DifT/(Fluid[c].cellWidth[1]*Fluid[c].cellWidth[1]))),
                    (DifT/(Fluid[c].cellWidth[2]*Fluid[c].cellWidth[2]))))*4.0)
end

local __demand(__inline)
task CalculateSpeciesDiffusionSpectralRadius(Fluid : region(ispace(int3d), Fluid_columns), c : int3d)
where
   reads(Fluid.{cellWidth, Di})
do
   var acc = -math.huge
   for i=0, nSpec do
      acc max= ((max(max((Fluid[c].Di[i]/(Fluid[c].cellWidth[0]*Fluid[c].cellWidth[0])),
                         (Fluid[c].Di[i]/(Fluid[c].cellWidth[1]*Fluid[c].cellWidth[1]))),
                         (Fluid[c].Di[i]/(Fluid[c].cellWidth[2]*Fluid[c].cellWidth[2]))))*4.0)
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateMaxSpectralRadius(Fluid : region(ispace(int3d), Fluid_columns),
                                        ModCells : region(ispace(int3d), Fluid_columns),
                                        mix : MIX.Mixture)
where
   reads(Fluid.cellWidth),
   reads(Fluid.{velocity, SoS}),
   reads(Fluid.{rho, mu}),
   reads(Fluid.{MassFracs, temperature, lam}),
   reads(Fluid.Di)
do
   var acc = -math.huge
   __demand(__openmp)
   for c in ModCells do
      -- Advection
      acc max= CalculateConvectiveSpectralRadius(Fluid, c)
      -- Momentum diffusion
      acc max= CalculateViscousSpectralRadius(Fluid, c)
      -- Heat Conduction
      acc max= CalculateHeatConductionSpectralRadius(Fluid, c, mix)
      -- Species diffusion
      acc max= CalculateSpeciesDiffusionSpectralRadius(Fluid, c)
   end
   return acc
end

return Exports end

