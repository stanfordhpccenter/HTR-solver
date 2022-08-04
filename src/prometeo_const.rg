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

local Exports = {}

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------

Exports.RGAS = 8.3144598        -- [J/(mol K)]
Exports.Na   = 6.02214086e23    -- [1/mol]
Exports.kb   = 1.38064852e-23   -- [m^2 kg /( s^2 K)]
Exports.PI   = 3.1415926535898

-- Stencil indices
Exports.Stencil1 = 0
Exports.Stencil2 = 1
Exports.Stencil3 = 2
Exports.Stencil4 = 3
Exports.nStencils = 4

-- SSP Runge-Kutta method Gottlieb et al. (2001)
-- RK_C[*][1] -- Coefficinets for solution at time n
-- RK_C[*][2] -- Coefficinets for intermediate solution
-- RK_C[*][3] -- Coefficinets for delta t
Exports.RK_C = {
   [1] = {    1.0,     0.0,     1.0},
   [2] = {3.0/4.0, 1.0/4.0, 1.0/4.0},
   [3] = {1.0/3.0, 2.0/3.0, 2.0/3.0},
}

Exports.RK_T = {
   [1] = 1.0,
   [2] = 0.5,
   [3] = 1.0,
}

-- Common groups of variables
Exports.Primitives = terralib.newlist({
   'pressure',
   'temperature',
   'MolarFracs',
   'velocity'
})

Exports.Properties = terralib.newlist({
   'rho',
   'mu',
   'lam',
   'Di',
   'SoS'
})

Exports.ProfilesVars = terralib.newlist({
   'MolarFracs_profile',
   'velocity_profile',
   'temperature_profile'
})

Exports.RecycleVars = terralib.newlist({
   'temperature_recycle',
   'MolarFracs_recycle',
   'velocity_recycle'
})

-- Variable indices
function Exports.GetirU(MIX) return MIX.nSpec   end
function Exports.GetirE(MIX) return MIX.nSpec+3 end
function Exports.GetnEq(MIX) return MIX.nSpec+4 end

-- Node types
Exports.Std_node   = 0  -- Node with standard stencil
Exports.L_S_node   = 1  -- Left node on staggered bc
Exports.Lp1_S_node = 2  -- Left plus one node on staggered bc
Exports.Lp2_S_node = 3  -- Left plus two node on staggered bc
Exports.Rm3_S_node = 4  -- Right minus three node on staggered bc
Exports.Rm2_S_node = 5  -- Right minus two node on staggered bc
Exports.Rm1_S_node = 6  -- Right minus one node on staggered bc
Exports.R_S_node   = 7  -- Right node on staggered bc
Exports.L_C_node   = 8  -- Left node on collocated bc
Exports.Lp1_C_node = 9  -- Left plus one node on collocated bc
Exports.Rm2_C_node = 10 -- Right minus two node on collocated bc
Exports.Rm1_C_node = 11 -- Right minus one node on collocated bc
Exports.R_C_node   = 12 -- Right node on collocated bc

return Exports

