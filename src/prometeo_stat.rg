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

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

-------------------------------------------------------------------------------
-- AVERAGING ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateInteriorVolume(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      acc += 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateAveragePressure(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d, pressure})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var cellVolume = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      acc += Fluid[c].pressure*cellVolume
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateAverageTemperature(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d, temperature})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var cellVolume = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      acc += Fluid[c].temperature*cellVolume
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateAverageKineticEnergy(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d, rho, velocity})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var cellVolume = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      var kineticEnergy = 0.5*Fluid[c].rho*MACRO.dot(Fluid[c].velocity, Fluid[c].velocity)
      acc += kineticEnergy*cellVolume
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateAverageTotalEnergy(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d, Conserved})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var cellVolume = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      acc += Fluid[c].Conserved[irE]*cellVolume
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateAverageRhoU(Fluid : region(ispace(int3d), Fluid_columns),
                                  dir : int)
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d, Conserved})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var cellVolume = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      acc += Fluid[c].Conserved[irU+dir]*cellVolume
   end
   return acc
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateMaxSpeed(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.velocity)
do
   var Umax = 0.0
   __demand(__openmp)
   for c in Fluid do
      Umax max= sqrt(MACRO.dot(Fluid[c].velocity,Fluid[c].velocity))
   end
   return Umax
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateMaxDensity(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.rho)
do
   var rhomax = 0.0
   __demand(__openmp)
   for c in Fluid do
      rhomax max= Fluid[c].rho
   end
   return rhomax
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateMaxPressure(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.pressure)
do
   var Pmax = 0.0
   __demand(__openmp)
   for c in Fluid do
      Pmax max= Fluid[c].pressure
   end
   return Pmax
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateMaxTemperature(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.temperature)
do
   var Tmax = 0.0
   __demand(__openmp)
   for c in Fluid do
      Tmax max= Fluid[c].temperature
   end
   return Tmax
end

--__demand(__parallel, __cuda)
--task CalculateMinTemperature(Fluid : region(ispace(int3d), Fluid_columns),
--                             Grid_xBnum : int32, Grid_xNum : int32,
--                             Grid_yBnum : int32, Grid_yNum : int32,
--                             Grid_zBnum : int32, Grid_zNum : int32)
--where
--  reads(Fluid.temperature)
--do
--  var acc = math.huge
--  __demand(__openmp)
--  for c in Fluid do
--    acc min= Fluid[c].temperature
--  end
--  return acc
--end
--
--__demand(__parallel, __cuda)
--task CalculateMaxTemperature(Fluid : region(ispace(int3d), Fluid_columns),
--                             Grid_xBnum : int32, Grid_xNum : int32,
--                             Grid_yBnum : int32, Grid_yNum : int32,
--                             Grid_zBnum : int32, Grid_zNum : int32)
--where
--  reads(Fluid.temperature)
--do
--  var acc = -math.huge
--  __demand(__openmp)
--  for c in Fluid do
--    acc max= Fluid[c].temperature
--  end
--  return acc
--end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.CalculateMaxMachNumber(Fluid    : region(ispace(int3d), Fluid_columns),
                                    dir : int)
where
   reads(Fluid.{velocity, SoS})
do
   var acc = -math.huge
   __demand(__openmp)
   for c in Fluid do
      acc max= fabs(Fluid[c].velocity[dir])/Fluid[c].SoS
   end
   return acc
end

return Exports end

