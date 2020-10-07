-- Copyright (c) "2019, by Stanford University
--               Contributors: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: "
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

return function(SPECIES, REACTION, Mixture, nSpec, nReac) local Exports = {}

-- Utility functions
local    C = regentlib.c
local fabs = regentlib.fabs(double)
local pow  = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-- Constants
local RGAS = 8.3144598        -- [J/(mol K)]
local Na   = 6.02214086e23    -- [1/mol]
local kb   = 1.38064852e-23   -- [m^2 kg /( s^2 K)]
local PI   = 3.1415926535898

__demand(__inline)
task Exports.GetSpeciesNames(Mix : Mixture)
   var Names : regentlib.string[nSpec]
   for i = 0, nSpec do
      Names[i] = [&int8](Mix.species[i].Name)
   end
   return Names
end

__demand(__inline)
task Exports.FindSpecies(name : int8[10], Mix : Mixture)
   var iSpec = -1
   for i = 0, nSpec do
      if C.strcmp(Mix.species[i].Name, name) == 0 then
         iSpec = i
         break
      end
   end
   regentlib.assert(iSpec > -1, "Species not found");
   return iSpec
end

__demand(__inline)
task Exports.CheckMixture(Yi : double[nSpec], Mix : Mixture)
   var tmp = 0.0
   for i = 0, nSpec do
      tmp += Yi[i]
   end
   tmp -= 1.0
-- TODO: the assert is not yet supported by the cuda compiler 
--       at the moment we return something in place of the assertion
--   regentlib.assert(fabs(tmp)<1e-3, "Sum of Yi exceeded unit value");
   var err = 0
--   if (fabs(tmp)>1e-2) then err = 1 end
   return err
end

__demand(__inline)
task Exports.ClipYi(Yi : double[nSpec])
   for i = 0, nSpec do
      Yi[i] max= 1.0e-60
      Yi[i] min= 1.0
   end
   return Yi
end

__demand(__inline)
task Exports.GetMolarWeightFromYi(Yi : double[nSpec], Mix : Mixture)
   var MixW = 0.0
   for i = 0, nSpec do
      MixW += Yi[i] / Mix.species[i].W
   end
   return 1.0/MixW
end

__demand(__inline)
task Exports.GetMolarWeightFromXi(Xi : double[nSpec], Mix : Mixture)
   var MixW = 0.0
   for i = 0, nSpec do
      MixW += Xi[i] * Mix.species[i].W
   end
   return MixW
end

__demand(__inline)
task Exports.GetMolarFractions(MixW : double, Yi : double[nSpec], Mix : Mixture)
   for i = 0, nSpec do
      Yi[i] *= MixW/Mix.species[i].W
   end
   return Yi
end

__demand(__inline)
task Exports.GetMassFractions(MixW : double, Xi : double[nSpec], Mix : Mixture)
   for i = 0, nSpec do
      Xi[i] *= Mix.species[i].W/MixW
   end
   return Xi
end

__demand(__inline)
task Exports.GetRhoFromRhoYi(rhoYi : double[nSpec])
   var rho = 0.0
   for i = 0, nSpec do
      rho += rhoYi[i]
   end
   return rho
end

__demand(__inline)
task Exports.GetYi(rho : double, rhoYi : double[nSpec])
   var rhoInv = 1.0/rho
   for i = 0, nSpec do
      rhoYi[i] *= rhoInv
   end
   return rhoYi
end

__demand(__inline)
task Exports.GetRhoYiFromYi(rho : double, Yi : double[nSpec])
   for i = 0, nSpec do
      Yi[i] *= rho
   end
   return Yi
end

__demand(__inline)
task Exports.GetRho(P : double, T : double, MixW : double, Mix : Mixture)
   return P * MixW / (RGAS * T)
end

__demand(__inline)
task Exports.GetHeatCapacity(T : double, Yi : double[nSpec], Mix : Mixture)
   var cp = 0.0
   for i = 0, nSpec do
      cp += Yi[i]*SPECIES.GetCp(Mix.species[i], T)
   end
   return cp
end

__demand(__inline)
task Exports.GetEnthalpy(T : double, Yi : double[nSpec], Mix : Mixture)
   var Enth = 0.0
   for i = 0, nSpec do
      Enth += Yi[i]*SPECIES.GetEnthalpy(Mix.species[i], T)
   end
   return Enth
end

__demand(__inline)
task Exports.GetSpeciesEnthalpy(i : int, T : double,  Mix : Mixture)
   return SPECIES.GetEnthalpy(Mix.species[i], T)
end

__demand(__inline)
task Exports.GetSpeciesMolarWeight(i : int, Mix : Mixture)
   return Mix.species[i].W
end

__demand(__inline)
task Exports.GetInternalEnergy(T : double, Yi : double[nSpec], Mix : Mixture)
   var e = 0.0
   for i = 0, nSpec do
      e += Yi[i]*(SPECIES.GetEnthalpy(Mix.species[i], T) - RGAS*T/Mix.species[i].W)
   end
   return e
end

__demand(__inline)
task Exports.GetSpecificInternalEnergy(i : int, T : double,  Mix : Mixture)
   return SPECIES.GetEnthalpy(Mix.species[i], T) - RGAS*T/Mix.species[i].W
end

__demand(__inline)
task Exports.GetTFromInternalEnergy(e0 : double, T : double, Yi : double[nSpec], Mix : Mixture)
   var MAXITS = 1000
   var TOL = 1e-5
   var dfdT = 1.0
   for j = 0, MAXITS do
      var f = e0 - Exports.GetInternalEnergy(T, Yi, Mix)
      if (fabs(f/dfdT) < TOL) then break end
      dfdT = 0.0
      for i = 0, nSpec do
         dfdT += Yi[i]*(SPECIES.GetCp(Mix.species[i], T) - RGAS/Mix.species[i].W)
      end
      T += f/dfdT
-- TODO: the assert is not yet supported by the cuda compiler 
--      regentlib.assert(j~=MAXITS, "GetTFromInternalEnergy did not converge")
   end
   return T
end

__demand(__inline)
task Exports.isValidInternalEnergy(e : double, Yi : double[nSpec], Mix : Mixture)
   var valid = true
   if     e < Exports.GetInternalEnergy(Mix.TMin, Yi, Mix) then valid = false
   elseif e > Exports.GetInternalEnergy(Mix.TMax, Yi, Mix) then valid = false
   end
   return valid
end

__demand(__inline)
task Exports.GetTFromRhoAndP(rho: double, MixW : double, P : double)
   return P*MixW/(rho*RGAS)
end

__demand(__inline)
task Exports.GetPFromRhoAndT(rho: double, MixW : double, T : double)
   return rho*RGAS*T/MixW
end

__demand(__inline)
task Exports.GetViscosity(T : double, Xi : double[nSpec], Mix : Mixture)
   var muk : double[nSpec]
   for i = 0, nSpec do
      muk[i] = SPECIES.GetMu(Mix.species[i], T)
   end

   var mu = 0.0
   for i = 0, nSpec do
      var den = 0.0;
      for j = 0, nSpec do
         var Phi = pow(1 + sqrt(muk[i]/muk[j]) * pow(Mix.species[j].W/Mix.species[i].W, 0.25) , 2);
         Phi /= sqrt(8*(1 + Mix.species[i].W/Mix.species[j].W));
         den += Xi[j]*Phi;
      end
      mu += Xi[i]*muk[i]/den;
   end
   return mu;
end

__demand(__inline)
task Exports.GetHeatConductivity(T : double, Xi : double[nSpec], Mix : Mixture)
   var a = 0.0
   var b = 0.0
   for i = 0, nSpec do
      var lami = SPECIES.GetLam(Mix.species[i], T)
      a += Xi[i]*lami
      b += Xi[i]/lami
   end
   return 0.5*(a + 1.0/b)
end

__demand(__inline)
task Exports.GetGamma(T : double, MixW : double, Yi : double[nSpec], Mix : Mixture)
   var cp = Exports.GetHeatCapacity(T, Yi, Mix)
   return cp/(cp - RGAS/MixW)
end

__demand(__inline)
task Exports.GetSpeedOfSound(T : double, gamma : double, MixW : double, Mix : Mixture)
   return sqrt(gamma * RGAS * T / MixW)
end

__demand(__inline)
task Exports.GetDiffusivity(P: double, T : double, MixW : double, Xi : double[nSpec], Mix : Mixture)
   var invDi : double[nSpec*nSpec]
   var Di    : double[nSpec]
   for i = 0, nSpec do
      invDi[i*nSpec+i] = 0.0
      for j = 0, i do
         invDi[j*nSpec+i] = 1.0/SPECIES.GetDif(Mix.species[i], Mix.species[j], P, T)
         invDi[i*nSpec+j] = invDi[j*nSpec+i]
      end
   end

   for i = 0, nSpec do
      var num = 0.0
      var den = 0.0
      for j = 0, i do
         num += Xi[j]*Mix.species[j].W;
         den += Xi[j]*invDi[i*nSpec+j];
      end
      for j = i+1, nSpec do
         num += Xi[j]*Mix.species[j].W
         den += Xi[j]*invDi[i*nSpec+j]
      end
      Di[i] = num/(MixW*den)
   end
   return Di
end

__demand(__inline)
task Exports.GetProductionRates(rho : double, P : double, T : double, Yi : double[nSpec], Mix : Mixture)
   var G : double[nSpec]
   var C : double[nSpec]
   var w : double[nSpec]
   for i = 0, nSpec do
      w[i] = 0.0;
      C[i] = Yi[i]*rho/Mix.species[i].W
      G[i] = SPECIES.GetFreeEnthalpy(Mix.species[i], T)
   end

   for i = 0, nReac do
      w = REACTION.AddProductionRates(Mix.reactions[i], P, T, C, G, w)
   end

   -- From [mol/(s m^3)] to [kg/(s m^3)]
   for i = 0, nSpec do
      w[i] *= Mix.species[i].W
   end
   return w
end

__demand(__inline)
task Exports.Getdpde(rho : double, gamma : double, Mix : Mixture)
   return rho*(gamma - 1)
end

__demand(__inline)
task Exports.Getdpdrhoi(gamma : double, T : double, Yi : double[nSpec], Mix : Mixture)
   var e = 0.0
   var ei : double[nSpec]
   for i = 0, nSpec do
      ei[i] = (SPECIES.GetEnthalpy(Mix.species[i], T) - RGAS*T/Mix.species[i].W)
      e += Yi[i]*ei[i]
   end
   var dpdrhoi : double[nSpec]
   for i = 0, nSpec do
      dpdrhoi[i] = RGAS*T/Mix.species[i].W + (gamma - 1)*(e - ei[i])
   end
   return dpdrhoi
end

return Exports end
