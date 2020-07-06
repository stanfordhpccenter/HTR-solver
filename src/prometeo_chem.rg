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

return function(SCHEMA, MIX, Fluid_columns, ATOMIC) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require 'util-desugared'
local MATH = require 'math_utils'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local fabs = regentlib.fabs(double)

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local ImplicitVars = terralib.newlist({
   'Conserved',
   'Conserved_t_old',
   'temperature'
})

-- Atomic switch
local Fluid = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "Fluid")
local coherence_mode
if ATOMIC then
   coherence_mode = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
else
   coherence_mode = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
end

-------------------------------------------------------------------------------
-- CHEMISTRY ROUTINES
-------------------------------------------------------------------------------

-- Parse input mixture
__demand(__inline)
task Exports.ParseConfigMixture(Mixture : SCHEMA.Mixture, mix : MIX.Mixture)
   var initMolarFracs = [UTIL.mkArrayConstant(nSpec, rexpr 1.0e-60 end)] 
   for i=0, Mixture.Species.length do
      var Species = Mixture.Species.values[i]
      initMolarFracs[MIX.FindSpecies(Species.Name, mix)] = Species.MolarFrac
   end
   return initMolarFracs
end

-- Reset mixture
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.ResetMixture(Fluid    : region(ispace(int3d), Fluid_columns),
                          ModCells : region(ispace(int3d), Fluid_columns),
                          initMolarFracs : double[nSpec])
where
   writes(Fluid.MolarFracs)
do
   __demand(__openmp)
   for c in ModCells do
      Fluid[c].MolarFracs = initMolarFracs
   end
end

-- RHS function for the implicit solver
local __demand(__inline)
task rhsChem(Fluid : region(ispace(int3d), Fluid_columns),
                 c : int3d,
               mix : MIX.Mixture)
where
   reads writes(Fluid.[ImplicitVars])
do
   var f : double[nEq]

   var rhoYi : double[nSpec]
   for i = 0, nSpec do
      rhoYi[i] = Fluid[c].Conserved[i]
   end
   var rho = MIX.GetRhoFromRhoYi(rhoYi)
   var Yi = MIX.GetYi(rho, rhoYi)
   Yi = MIX.ClipYi(Yi)
   var MixW = MIX.GetMolarWeightFromYi(Yi, mix)

   var rhoInv = 1.0/rho
   var velocity = array(Fluid[c].Conserved[irU+0]*rhoInv,
                        Fluid[c].Conserved[irU+1]*rhoInv,
                        Fluid[c].Conserved[irU+2]*rhoInv)

   var kineticEnergy = (0.5*MACRO.dot(velocity, velocity))
   var InternalEnergy = Fluid[c].Conserved[irE]*rhoInv - kineticEnergy
   Fluid[c].temperature = MIX.GetTFromInternalEnergy(InternalEnergy, Fluid[c].temperature, Yi, mix)
   var P = MIX.GetPFromRhoAndT(rho, MixW, Fluid[c].temperature)

   var w  = MIX.GetProductionRates(rho, P, Fluid[c].temperature, Yi, mix)

   for i = 0, nSpec do
      f[i] = w[i] + Fluid[c].Conserved_t_old[i]
   end
   for i = nSpec, nEq do
      f[i] = Fluid[c].Conserved_t_old[i]
   end
   return f
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdateChemistry(Fluid    : region(ispace(int3d), Fluid_columns),
                             ModCells : region(ispace(int3d), Fluid_columns),
                             Integrator_deltaTime : double,
                             mix : MIX.Mixture)
where
   reads(Fluid.Conserved_t),
   reads writes(Fluid.[ImplicitVars])
do
   var err = 0
   __demand(__openmp)
   for c in ModCells do
      Fluid[c].Conserved_t_old = Fluid[c].Conserved_t
      err += [MATH.mkRosenbrock(nEq, Fluid_columns, ImplicitVars, "Conserved", MIX.Mixture, rhsChem)]
             (Fluid, c, Integrator_deltaTime, Integrator_deltaTime, mix)
   end
   regentlib.assert(err==0, "Something wrong in UpdateChemistry")
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.AddChemistrySources([Fluid],
                                 ModCells : region(ispace(int3d), Fluid_columns),
                                 mix : MIX.Mixture)
where
   reads(Fluid.{rho, MolarFracs, pressure, temperature}),
   reads writes (Fluid.Conserved_t),
   [coherence_mode]
do
   __demand(__openmp)
   for c in ModCells do
      var MixW = MIX.GetMolarWeightFromXi(Fluid[c].MolarFracs, mix)
      var Yi   = MIX.GetMassFractions(MixW, Fluid[c].MolarFracs, mix)
      var w    = MIX.GetProductionRates(Fluid[c].rho, Fluid[c].pressure, Fluid[c].temperature, Yi, mix)
      for i = 0, nSpec do
         Fluid[c].Conserved_t[i] += w[i]
      end
   end
end

return Exports end

