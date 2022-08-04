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

return function(SCHEMA, MIX, METRIC, TYPES,
                ELECTRIC_FIELD) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local nIons = MIX.nIons       -- Number of ions in the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

local Fluid_columns = TYPES.Fluid_columns

-------------------------------------------------------------------------------
-- MIXTURE PROPERTIES ROUTINES
-------------------------------------------------------------------------------

if (ELECTRIC_FIELD and (MIX.nIons > 0)) then
   Properties:insert("Ki")
end

extern task Exports.UpdatePropertiesFromPrimitive(Fluid    : region(ispace(int3d), Fluid_columns),
                                                  mix : MIX.Mixture)
where
   reads(Fluid.[Primitives]),
   writes(Fluid.MassFracs),
   writes(Fluid.[Properties])
end

Exports.UpdatePropertiesFromPrimitive:set_task_id(TYPES.TID_UpdatePropertiesFromPrimitive)

-------------------------------------------------------------------------------
-- CONSERVED TO PRIMITIVE/PRIMITIVE TO CONSERVED ROUTINES
-------------------------------------------------------------------------------

extern task Exports.UpdateConservedFromPrimitive(Fluid    : region(ispace(int3d), Fluid_columns),
                                                 mix : MIX.Mixture)
where
   reads(Fluid.{MassFracs, temperature, velocity}),
   reads(Fluid.rho),
   writes(Fluid.Conserved)
end
Exports.UpdateConservedFromPrimitive:set_task_id(TYPES.TID_UpdateConservedFromPrimitive)

extern task Exports.UpdatePrimitiveFromConserved(Fluid    : region(ispace(int3d), Fluid_columns),
                                                 mix : MIX.Mixture)
where
   reads(Fluid.Conserved),
   reads(Fluid.temperature),
   writes(Fluid.[Primitives])
end
Exports.UpdatePrimitiveFromConserved:set_task_id(TYPES.TID_UpdatePrimitiveFromConserved)

return Exports end

