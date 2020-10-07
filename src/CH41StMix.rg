-- Copyright (c) "2019, by Stanford University
--               Contributors: Mario Di Renzo
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

return function(SCHEMA) local Exports = {}

-- Utility functions
local    C = regentlib.c
local fabs = regentlib.fabs(double)
local pow  = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)
local format = require("std/format")

-- Constants
local ATom  =  1e-10          -- Angstrom to meter
local DToCm =  3.33564e-30    -- Debye to Coulomb meter
local RGAS = 8.3144598        -- [J/(mol K)]

Exports.nSpec = 4
Exports.nReac = 1

local SPECIES  = require 'Species'
local REACTION = (require 'Reaction')(Exports.nSpec, 2, 0)

struct Exports.Mixture {
   species : SPECIES.Species[Exports.nSpec]
   reactions : REACTION.Reaction[1]
   -- Max an min acceptable temeperatures
   TMax : double
   TMin : double
}

local MultiComponent = (require 'MultiComponent')(SPECIES, REACTION, Exports.Mixture, Exports.nSpec, Exports.nReac)

__demand(__inline)
task Exports.InitMixture(config : SCHEMA.Config)
   regentlib.assert(config.Flow.mixture.type == SCHEMA.MixtureModel_CH41StMix,
                    "This executable is expecting CH41StMix in the input file");

   var Mix : Exports.Mixture

--------------------------------------
-- Set Species
--------------------------------------
   -- N2
   var iCH4 = 0
   format.snprint([&int8](Mix.species[iCH4].Name), 10, "CH4")
   Mix.species[iCH4].W = 12.0107e-3 + 4.0*1.00784e-3
   Mix.species[iCH4].Geom = SPECIES.SpeciesGeom_NonLinear
   Mix.species[iCH4].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iCH4].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iCH4].cpCoeff.TMin     = 0200.0000
   Mix.species[iCH4].cpCoeff.TMax     = 6000.0007
   Mix.species[iCH4].cpCoeff.cpH      = array( 3.730042760e+06,-1.383501485e+04, 2.049107091e+01,-1.961974759e-03, 4.727313040e-07,-3.728814690e-11, 1.623737207e-15, 7.532066910e+04,-1.219124889e+02)
   Mix.species[iCH4].cpCoeff.cpM      = array( 3.730042760e+06,-1.383501485e+04, 2.049107091e+01,-1.961974759e-03, 4.727313040e-07,-3.728814690e-11, 1.623737207e-15, 7.532066910e+04,-1.219124889e+02)
   Mix.species[iCH4].cpCoeff.cpL      = array(-1.766850998e+05, 2.786181020e+03,-1.202577850e+01, 3.917619290e-02,-3.619054430e-05, 2.026853043e-08,-4.976705490e-12,-2.331314360e+04, 8.904322750e+01 )
   Mix.species[iCH4].DiffCoeff.sigma   = 3.746*ATom
   Mix.species[iCH4].DiffCoeff.kbOveps = 1.0/141.4
   Mix.species[iCH4].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iCH4].DiffCoeff.alpha   = 2.600*ATom
   Mix.species[iCH4].DiffCoeff.Z298    = 13.000
   -- O2
   var iO2 = 1
   format.snprint([&int8](Mix.species[iO2].Name), 10, "O2")
   Mix.species[iO2].W = 2*15.9994e-3
   Mix.species[iO2].Geom = SPECIES.SpeciesGeom_Linear
   Mix.species[iO2].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iO2].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iO2].cpCoeff.TMin     = 0200.0000
   Mix.species[iO2].cpCoeff.TMax     = 20000.0007
   Mix.species[iO2].cpCoeff.cpH      = array( 4.975294300e+08,-2.866106874e+05, 6.690352250e+01,-6.169959020e-03, 3.016396027e-07,-7.421416600e-12, 7.278175770e-17, 2.293554027e+06,-5.530621610e+02 )
   Mix.species[iO2].cpCoeff.cpM      = array(-1.037939022e+06, 2.344830282e+03, 1.819732036e+00, 1.267847582e-03,-2.188067988e-07, 2.053719572e-11,-8.193467050e-16,-1.689010929e+04, 1.738716506e+01 )
   Mix.species[iO2].cpCoeff.cpL      = array(-3.425563420e+04, 4.847000970e+02, 1.119010961e+00, 4.293889240e-03,-6.836300520e-07,-2.023372700e-09, 1.039040018e-12,-3.391454870e+03, 1.849699470e+01 )
   Mix.species[iO2].DiffCoeff.sigma   = 3.458*ATom
   Mix.species[iO2].DiffCoeff.kbOveps = 1.0/107.40
   Mix.species[iO2].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iO2].DiffCoeff.alpha   = 1.600*ATom
   Mix.species[iO2].DiffCoeff.Z298    = 3.800
   -- NO
   var iCO2 = 2
   format.snprint([&int8](Mix.species[iCO2].Name), 10, "CO2")
   Mix.species[iCO2].W = 12.0107e-3+2.0*15.9994e-3
   Mix.species[iCO2].Geom = SPECIES.SpeciesGeom_Linear
   Mix.species[iCO2].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iCO2].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iCO2].cpCoeff.TMin     = 0200.0000
   Mix.species[iCO2].cpCoeff.TMax     = 20000.0007
   Mix.species[iCO2].cpCoeff.cpH      = array(-1.544423287e+09, 1.016847056e+06,-2.561405230e+02, 3.369401080e-02,-2.181184337e-06, 6.991420840e-11,-8.842351500e-16,-8.043214510e+06, 2.254177493e+03 )
   Mix.species[iCO2].cpCoeff.cpM      = array( 1.176962419e+05,-1.788791477e+03, 8.291523190e+00,-9.223156780e-05, 4.863676880e-09,-1.891053312e-12, 6.330036590e-16,-3.908350590e+04,-2.652669281e+01 )
   Mix.species[iCO2].cpCoeff.cpL      = array( 4.943650540e+04,-6.264116010e+02, 5.301725240e+00, 2.503813816e-03,-2.127308728e-07,-7.689988780e-10, 2.849677801e-13,-4.528198460e+04,-7.048279440e+00 )
   Mix.species[iCO2].DiffCoeff.sigma   = 3.763*ATom
   Mix.species[iCO2].DiffCoeff.kbOveps = 1.0/244.0
   Mix.species[iCO2].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iCO2].DiffCoeff.alpha   = 2.650*ATom
   Mix.species[iCO2].DiffCoeff.Z298    = 2.100
   -- N
   var iH2O = 3
   format.snprint([&int8](Mix.species[iH2O].Name), 10, "H2O")
   Mix.species[iH2O].W = 2.0*1.00784e-3 + 15.9994e-3
   Mix.species[iH2O].Geom = SPECIES.SpeciesGeom_NonLinear
   Mix.species[iH2O].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iH2O].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iH2O].cpCoeff.TMin     = 0200.0000
   Mix.species[iH2O].cpCoeff.TMax     = 6000.0007
   Mix.species[iH2O].cpCoeff.cpH      = array( 1.034972096e+06,-2.412698562e+03, 4.646110780e+00, 2.291998307e-03,-6.836830480e-07, 9.426468930e-11,-4.822380530e-15,-1.384286509e+04,-7.978148510e+00 )
   Mix.species[iH2O].cpCoeff.cpM      = array( 1.034972096e+06,-2.412698562e+03, 4.646110780e+00, 2.291998307e-03,-6.836830480e-07, 9.426468930e-11,-4.822380530e-15,-1.384286509e+04,-7.978148510e+00 )
   Mix.species[iH2O].cpCoeff.cpL      = array(-3.947960830e+04, 5.755731020e+02, 9.317826530e-01, 7.222712860e-03,-7.342557370e-06, 4.955043490e-09,-1.336933246e-12,-3.303974310e+04, 1.724205775e+01 )
   Mix.species[iH2O].DiffCoeff.sigma   = 2.605*ATom
   Mix.species[iH2O].DiffCoeff.kbOveps = 1.0/572.4
   Mix.species[iH2O].DiffCoeff.mu      = 1.844*DToCm
   Mix.species[iH2O].DiffCoeff.alpha   = 0.000*ATom
   Mix.species[iH2O].DiffCoeff.Z298    = 4.000

   var i = 0
   -- Oxygen dissociation (CH4 + 2 O2 -> 2 H2O + CO2)
   Mix.reactions[i].A    = 1.1e7
   Mix.reactions[i].n    = 0.0
   Mix.reactions[i].EovR = 20000*4.184/RGAS
   Mix.reactions[i].has_backward = false
   -- Educts
   Mix.reactions[i].Neducts = 0
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i], iCH4, 1.0, 1.0)
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i],  iO2, 2.0, 0.5)
   -- Products
   Mix.reactions[i].Npducts = 0
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i], iH2O, 2.0, 2.0)
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i], iCO2, 1.0, 1.0)
   -- Colliders
   Mix.reactions[i].Nthirdb = 0

   regentlib.assert(i+1 == Exports.nReac, "Something wrong with number of reactions in InitMixture")

   -- Set maximum and minimum temperature
   Mix.TMax = math.huge
   Mix.TMin = 0.0
   for i = 0, Exports.nSpec do
      Mix.TMax min= Mix.species[i].cpCoeff.TMax
      Mix.TMin max= Mix.species[i].cpCoeff.TMin
   end

   return Mix
end

-- Copy all elements form MultiComponent
for k, t in pairs(MultiComponent) do
   local v = k
   Exports[v] = t
end

return Exports end
