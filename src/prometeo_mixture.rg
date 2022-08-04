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

return function(SCHEMA, TYPES) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require 'util'

-- Store Mix data structure
Exports.Mixture = TYPES.Mix
Exports.nSpec = TYPES.nSpec
Exports.nIons = TYPES.nIons

-- Load C wrappers
local MIX = terralib.includec("prometeo_mixture_wrappers.hpp", {"-DEOS="..os.getenv("EOS")})

MIX.InitMixture.replicable = true
MIX.GetSpeciesName.replicable = true
MIX.FindSpecies.replicable = true
MIX.CheckMixture.replicable = true
MIX.ClipYi.replicable = true
MIX.GetMolarWeightFromXi.replicable = true
MIX.GetMolarWeightFromYi.replicable = true
MIX.GetMolarFractions.replicable = true
MIX.GetMassFractions.replicable = true
MIX.GetRhoFromRhoYi.replicable = true
MIX.GetRhoYiFromYi.replicable = true
MIX.GetYi.replicable = true
MIX.GetRho.replicable = true
MIX.GetHeatCapacity.replicable = true
MIX.GetEnthalpy.replicable = true
MIX.GetSpeciesEnthalpy.replicable = true
MIX.GetSpeciesMolarWeight.replicable = true
MIX.GetInternalEnergy.replicable = true
MIX.GetSpecificInternalEnergy.replicable = true
MIX.GetTFromInternalEnergy.replicable = true
MIX.isValidInternalEnergy.replicable = true
MIX.GetTFromRhoAndP.replicable = true
MIX.GetPFromRhoAndT.replicable = true
MIX.GetViscosity.replicable = true
MIX.GetHeatConductivity.replicable = true
MIX.GetGamma.replicable = true
MIX.GetSpeedOfSound.replicable = true
MIX.GetDiffusivity.replicable = true
if Exports.nIons > 0 then
   MIX.GetElectricMobility.replicable = true
end
MIX.GetPartialElectricChargeDensity.replicable = true
MIX.GetElectricChargeDensity.replicable = true
MIX.GetSpeciesChargeNumber.replicable = true
MIX.GetDielectricPermittivity.replicable = true
MIX.GetProductionRates.replicable = true
MIX.Getdpde.replicable = true
MIX.Getdpdrhoi.replicable = true

Exports.InitMixtureStruct               = MIX.InitMixture
Exports.FindSpecies                     = MIX.FindSpecies
Exports.CheckMixture                    = MIX.CheckMixture
Exports.ClipYi                          = MIX.ClipYi
Exports.GetMolarWeightFromXi            = MIX.GetMolarWeightFromXi
Exports.GetMolarWeightFromYi            = MIX.GetMolarWeightFromYi
Exports.GetMolarFractions               = MIX.GetMolarFractions
Exports.GetMassFractions                = MIX.GetMassFractions
Exports.GetRhoFromRhoYi                 = MIX.GetRhoFromRhoYi
Exports.GetRhoYiFromYi                  = MIX.GetRhoYiFromYi
Exports.GetYi                           = MIX.GetYi
Exports.GetRho                          = MIX.GetRho
Exports.GetHeatCapacity                 = MIX.GetHeatCapacity
Exports.GetEnthalpy                     = MIX.GetEnthalpy
Exports.GetSpeciesEnthalpy              = MIX.GetSpeciesEnthalpy
Exports.GetSpeciesMolarWeight           = MIX.GetSpeciesMolarWeight
Exports.GetInternalEnergy               = MIX.GetInternalEnergy
Exports.GetSpecificInternalEnergy       = MIX.GetSpecificInternalEnergy
Exports.GetTFromInternalEnergy          = MIX.GetTFromInternalEnergy
Exports.isValidInternalEnergy           = MIX.isValidInternalEnergy
Exports.GetTFromRhoAndP                 = MIX.GetTFromRhoAndP
Exports.GetPFromRhoAndT                 = MIX.GetPFromRhoAndT
Exports.GetViscosity                    = MIX.GetViscosity
Exports.GetHeatConductivity             = MIX.GetHeatConductivity
Exports.GetGamma                        = MIX.GetGamma
Exports.GetSpeedOfSound                 = MIX.GetSpeedOfSound
Exports.GetDiffusivity                  = MIX.GetDiffusivity
if Exports.nIons > 0 then
   Exports.GetElectricMobility          = MIX.GetElectricMobility
end
Exports.GetPartialElectricChargeDensity = MIX.GetPartialElectricChargeDensity
Exports.GetElectricChargeDensity        = MIX.GetElectricChargeDensity
Exports.GetSpeciesChargeNumber          = MIX.GetSpeciesChargeNumber
Exports.GetDielectricPermittivity       = MIX.GetDielectricPermittivity
Exports.GetProductionRates              = MIX.GetProductionRates
Exports.Getdpde                         = MIX.Getdpde
Exports.Getdpdrhoi                      = MIX.Getdpdrhoi

-------------------------------------------------------------------------------
-- GENERATES A VECTRO WITH ALL THE SPECIES NAMES
-------------------------------------------------------------------------------

__demand(__inline)
task Exports.GetSpeciesNames(mix : TYPES.Mix)
   var Names : regentlib.string[TYPES.nSpec]
   for i = 0, TYPES.nSpec do
      Names[i] = MIX.GetSpeciesName(i, &mix)
   end
   return Names
end

-------------------------------------------------------------------------------
-- PARSES A COMPOSITION SPECIFIED IN THE INPUT FILE
-------------------------------------------------------------------------------

__demand(__inline)
task Exports.ParseConfigMixture(Mixture : SCHEMA.Mixture, mix : TYPES.Mix)
   var initMolarFracs = [UTIL.mkArrayConstant(TYPES.nSpec, rexpr 1.0e-60 end)]
   for i=0, Mixture.Species.length do
      var Species = Mixture.Species.values[i]
      initMolarFracs[MIX.FindSpecies(Species.Name, &mix)] = Species.MolarFrac
   end
   regentlib.assert(Exports.CheckMixture(initMolarFracs, &mix),
      "Molar fractions specified in the input file do not add to one")
   return initMolarFracs
end

-------------------------------------------------------------------------------
-- LOAD MIXTURE TASK
-------------------------------------------------------------------------------

local extern task LoadMixture(Fluid : region(ispace(int3d), TYPES.Fluid_columns),
                              mix : TYPES.Mix)
LoadMixture:set_task_id(TYPES.TID_LoadMixture)

__demand(__inline)
task Exports.InitMixture(Fluid : region(ispace(int3d), TYPES.Fluid_columns),
                         tiles : ispace(int3d),
                         p_all : partition(disjoint, Fluid, tiles),
                         config : SCHEMA.Config)
   var Mix = MIX.InitMixture(&config)
   __demand(__index_launch)
   for c in tiles do LoadMixture(p_all[c], Mix) end
   -- Make sure that the mixurte is loaded everywhere
   __fence(__execution, __block)
   return Mix
end

return Exports end

