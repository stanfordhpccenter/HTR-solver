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

return function(SCHEMA, MIX, TYPES, ATOMIC) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require 'util'
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
local Fluid = regentlib.newsymbol(region(ispace(int3d), TYPES.Fluid_columns), "Fluid")
local coherence_mode
if ATOMIC then
   coherence_mode = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
else
   coherence_mode = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
end

-------------------------------------------------------------------------------
-- CHEMISTRY ROUTINES
-------------------------------------------------------------------------------

-- Reset mixture
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.ResetMixture(Fluid    : region(ispace(int3d), TYPES.Fluid_columns),
                          initMolarFracs : double[nSpec])
where
   writes(Fluid.MolarFracs)
do
   __demand(__openmp)
   for c in Fluid do
      Fluid[c].MolarFracs = initMolarFracs
   end
end

extern task Exports.UpdateChemistry(Fluid    : region(ispace(int3d), TYPES.Fluid_columns),
                                    Integrator_deltaTime : double,
                                    mix : MIX.Mixture)
where
   reads(Fluid.Conserved_t),
   reads writes(Fluid.[ImplicitVars])
end
Exports.UpdateChemistry:set_task_id(TYPES.TID_UpdateChemistry)

extern task Exports.AddChemistrySources([Fluid],
                                        mix : MIX.Mixture)
where
   reads(Fluid.{rho, MassFracs, pressure, temperature}),
   reads writes (Fluid.Conserved_t),
   [coherence_mode]
end
Exports.AddChemistrySources:set_task_id(TYPES.TID_AddChemistrySources)

return Exports end

