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

return function(nSpec, MAX_NUM_REACTANTS, MAX_NUM_TB) local Exports = {}

-- Utility functions
local log  = regentlib.log(double)
local exp  = regentlib.exp(double)
local pow  = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-- Constants
local RGAS = 8.3144598        -- [J/(mol K)]

local struct Reactant {
   ind : int,     -- Index in the species vector
   nu  : double   -- Stoichiometric coefficient
}

-- Species structure
struct Exports.Reaction {
   A         : double         -- Arrhenius pre-exponential factor [m^{3*(o-1)}/(mol^(o-1) s)] where o is the order fo the reaction
   n         : double         -- Arrhenius temperature exponent
   EovR      : double         -- Arrhenius activation energy [K]
   has_backward : bool        -- Self-explenatory

   Neducts : int              -- number of reactants
   Npducts : int              -- number of products
   Nthirdb : int              -- number of third bodies

   educts : Reactant[MAX_NUM_REACTANTS]   -- List of reactants and stoichiometric coefficients
   pducts : Reactant[MAX_NUM_REACTANTS]   -- List of products and stoichiometric coefficients
   thirdb : Reactant[MAX_NUM_TB]          -- List of third bodies and efficiencies

--   vector<pair<double,shared_ptr<Species>>> educts   -- List of reactants and stoichiometric coefficient
--   vector<pair<double,shared_ptr<Species>>> pducts   -- List of products and stoichiometric coefficient
--   vector<pair<double,shared_ptr<Species>>> thirdb   -- List of third bodies and efficiency
}

__demand(__inline)
task Exports.AddEduct( r : Exports.Reaction, index : int, nu : double )
   regentlib.assert(r.Neducts < MAX_NUM_REACTANTS, "Increase MAX_NUM_REACTANTS")
   r.educts[r.Neducts].ind = index
   r.educts[r.Neducts].nu  = nu
   r.Neducts += 1
   return r
end

__demand(__inline)
task Exports.AddPduct( r : Exports.Reaction, index : int, nu : double )
   regentlib.assert(r.Npducts < MAX_NUM_REACTANTS, "Increase MAX_NUM_REACTANTS")
   r.pducts[r.Npducts].ind = index
   r.pducts[r.Npducts].nu  = nu
   r.Npducts += 1
   return r
end

__demand(__inline)
task Exports.AddThirdb( r : Exports.Reaction, index : int, nu : double )
   regentlib.assert(r.Nthirdb < MAX_NUM_TB, "Increase MAX_NUM_TB")
   r.thirdb[r.Nthirdb].ind = index
   r.thirdb[r.Nthirdb].nu  = nu
   r.Nthirdb += 1
   return r
end

local __demand(__inline)
task CompRateCoeff( r : Exports.Reaction, T : double )
   var Kf = r.A
   if ( r.n ~= 0.0 ) then
      Kf *= pow( T, r.n )
   end
   if ( r.EovR > 1e-5 ) then
      Kf *= exp( -r.EovR / T )
   end
   return Kf
end

local __demand(__inline)
task CompBackwardRateCoeff( r : Exports.Reaction, Kf : double, P : double, T : double, G : double[nSpec] )
   var sumNu = 0.0
   var sumNuG = 0.0
   for i = 0, r.Neducts do
      sumNu  -= r.educts[i].nu
      sumNuG -= r.educts[i].nu*G[r.educts[i].ind]
   end
   for i = 0, r.Npducts do
      sumNu  += r.pducts[i].nu
      sumNuG += r.pducts[i].nu*G[r.pducts[i].ind]
   end
   var lnKc = - sumNuG - sumNu * ( log(T) + log(RGAS/P) )
   return Kf * exp(-lnKc)
end

local __demand(__inline)
task GetReactionRate( r : Exports.Reaction, P : double, T : double, C : double[nSpec], G : double[nSpec] )
   -- Forward reaction rate
   var Kf = CompRateCoeff(r, T)
   var a = 1.0
   for i = 0, r.Neducts do
      var ind = r.educts[i].ind
      a *= pow(C[ind],r.educts[i].nu)
   end
   -- Backward reaction rate
   var Kb = 0.0
   var b = 1.0
   if ( r.has_backward ) then
      Kb = CompBackwardRateCoeff(r, Kf, P, T, G)
      for i = 0, r.Npducts do
         var ind = r.pducts[i].ind
         b *= pow(C[ind],r.pducts[i].nu)
      end
   end
   -- Third body efficiency
   var c = 1.0
   if (r.Nthirdb ~= 0) then
      c = 0.0
      for i = 0, r.Nthirdb do
         var ind = r.thirdb[i].ind
         c += C[ind]*r.thirdb[i].nu
      end
   end
   -- Compute reaction rate
   return c*(Kf*a - Kb*b)
end

__demand(__inline)
task Exports.AddProductionRates( r : Exports.Reaction, P : double, T : double, C : double[nSpec], G : double[nSpec], w : double[nSpec] )
   var R = GetReactionRate(r, P, T, C, G)
   for i = 0, r.Neducts do
      var ind = r.educts[i].ind
      w[ind] -= r.educts[i].nu*R
   end
   for i = 0, r.Npducts do
      var ind = r.pducts[i].ind
      w[ind] += r.pducts[i].nu*R
   end
   return w
end

return Exports end
