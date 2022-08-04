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

return function(nEq, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local UTIL = require 'util'
local CONST = require "prometeo_const"

-- Runge-Kutta coeffients 
local RK_C = CONST.RK_C

-------------------------------------------------------------------------------
-- RK ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeTemporaries(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.Conserved),
   writes(Fluid.Conserved_old)
do
  __demand(__openmp)
  for c in Fluid do
      Fluid[c].Conserved_old = Fluid[c].Conserved
  end
end

Exports.mkInitializeTimeDerivatives = terralib.memoize(function(Integrator_implicitChemistry)
   local InitializeTimeDerivatives
   if Integrator_implicitChemistry then
      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task InitializeTimeDerivatives(Fluid : region(ispace(int3d), Fluid_columns))
      where
         reads(Fluid.Conserved_t_old),
         writes(Fluid.Conserved_t)
      do
         __demand(__openmp)
         for c in Fluid do
            Fluid[c].Conserved_t = (-Fluid[c].Conserved_t_old)
         end
      end
   else
      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task InitializeTimeDerivatives(Fluid : region(ispace(int3d), Fluid_columns))
      where
         writes(Fluid.Conserved_t)
      do
         __demand(__openmp)
         for c in Fluid do
            Fluid[c].Conserved_t = [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)]
         end
      end
   end
   return InitializeTimeDerivatives
end)

Exports.mkUpdateVars = terralib.memoize(function(STAGE)
   local UpdateVars
   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task UpdateVars(Fluid : region(ispace(int3d), Fluid_columns),
                   Integrator_deltaTime : double,
                   Integrator_implicitChemistry : bool)
   where
      reads(Fluid.Conserved_old),
      reads(Fluid.Conserved_t),
      reads writes(Fluid.Conserved)
   do
      var dt = Integrator_deltaTime
      if Integrator_implicitChemistry then
         dt *= 0.5
      end
      __demand(__openmp)
      for c in Fluid do
         -- Set values for next substep
         for i=0, nEq do
            Fluid[c].Conserved[i] =  [RK_C[STAGE][1]] * Fluid[c].Conserved_old[i]
                                   + [RK_C[STAGE][2]] * Fluid[c].Conserved[i]
                                   + [RK_C[STAGE][3]] * Fluid[c].Conserved_t[i] * dt
         end
      end
   end
   return UpdateVars
end)

return Exports end

