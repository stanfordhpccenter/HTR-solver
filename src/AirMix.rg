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

return function(SCHEMA) local Exports = {}

-- Utility functions
local    C = regentlib.c
local fabs = regentlib.fabs(double)
local pow  = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-- Constants

local ATom  =  1e-10          -- Angstrom to meter
local DToCm =  3.33564e-30    -- Debye to Coulomb meter

local RGAS = 8.3144598        -- [J/(mol K)]
local Na   = 6.02214086e23    -- [1/mol]
local kb   = 1.38064852e-23   -- [m^2 kg /( s^2 K)]
local PI   = 3.1415926535898

Exports.nSpec = 5

local SPECIES  = require 'Species'
local REACTION = (require 'Reaction')(Exports.nSpec, 2, 5)

struct Exports.Mixture {
   nSpec : int
   nReac : int
   species : SPECIES.Species[5]
   reactions : REACTION.Reaction[5]
   -- Max an min acceptable temeperatures
   TMax : double
   TMin : double
}

__demand(__inline)
task Exports.InitMixture(config : SCHEMA.Config)
   var Mix : Exports.Mixture
   Mix.nSpec = Exports.nSpec
--------------------------------------
-- Set Species
--------------------------------------
   -- N2
   var iN2 = 0
   Mix.species[iN2].Name = "N2"
   Mix.species[iN2].W = 2*14.0067e-3
   Mix.species[iN2].Geom = 1
   Mix.species[iN2].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iN2].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iN2].cpCoeff.TMin     = 0200.0000
   Mix.species[iN2].cpCoeff.TMax     = 20000.0007
   Mix.species[iN2].cpCoeff.cpH      = array( 8.310139160e+08,-6.420733540e+05, 2.020264635e+02,-3.065092046e-02, 2.486903333e-06,-9.705954110e-11, 1.437538881e-15, 4.938707040e+06,-1.672099740e+03 )
   Mix.species[iN2].cpCoeff.cpM      = array( 5.877124060e+05,-2.239249073e+03, 6.066949220e+00,-6.139685500e-04, 1.491806679e-07,-1.923105485e-11, 1.061954386e-15, 1.283210415e+04,-1.586640027e+01 )
   Mix.species[iN2].cpCoeff.cpL      = array( 2.210371497e+04,-3.818461820e+02, 6.082738360e+00,-8.530914410e-03, 1.384646189e-05,-9.625793620e-09, 2.519705809e-12, 7.108460860e+02,-1.076003744e+01 )
   Mix.species[iN2].DiffCoeff.sigma   = 3.621*ATom
   Mix.species[iN2].DiffCoeff.kbOveps = 1.0/97.530
   Mix.species[iN2].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iN2].DiffCoeff.alpha   = 1.760*ATom
   Mix.species[iN2].DiffCoeff.Z298    = 4.000
   -- O2
   var iO2 = 1
   Mix.species[iO2].Name = "O2"
   Mix.species[iO2].W = 2*15.9994e-3
   Mix.species[iO2].Geom = 1
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
   var iNO = 2
   Mix.species[iNO].Name = "NO"
   Mix.species[iNO].W = 14.0067e-3+15.9994e-3
   Mix.species[iNO].Geom = 1
   Mix.species[iNO].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iNO].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iNO].cpCoeff.TMin     = 0200.0000
   Mix.species[iNO].cpCoeff.TMax     = 20000.0007
   Mix.species[iNO].cpCoeff.cpH      = array(-9.575303540e+08, 5.912434480e+05,-1.384566826e+02, 1.694339403e-02,-1.007351096e-06, 2.912584076e-11,-3.295109350e-16,-4.677501240e+06, 1.242081216e+03 )
   Mix.species[iNO].cpCoeff.cpM      = array( 2.239018716e+05,-1.289651623e+03, 5.433936030e+00,-3.656034900e-04, 9.880966450e-08,-1.416076856e-11, 9.380184620e-16, 1.750317656e+04,-8.501669090e+00 )
   Mix.species[iNO].cpCoeff.cpL      = array(-1.143916503e+04, 1.536467592e+02, 3.431468730e+00,-2.668592368e-03, 8.481399120e-06,-7.685111050e-09, 2.386797655e-12, 9.098214410e+03, 6.728725490e+00 )
   Mix.species[iNO].DiffCoeff.sigma   = 3.621*ATom
   Mix.species[iNO].DiffCoeff.kbOveps = 1.0/97.530
   Mix.species[iNO].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iNO].DiffCoeff.alpha   = 1.760*ATom
   Mix.species[iNO].DiffCoeff.Z298    = 4.000
   -- N
   var iN = 3
   Mix.species[iN].Name = "N"
   Mix.species[iN].W = 14.0067e-3
   Mix.species[iN].Geom = 0
   Mix.species[iN].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iN].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iN].cpCoeff.TMin     = 0200.0000
   Mix.species[iN].cpCoeff.TMax     = 20000.0007
   Mix.species[iN].cpCoeff.cpH      = array( 5.475181050e+08,-3.107574980e+05, 6.916782740e+01,-6.847988130e-03, 3.827572400e-07,-1.098367709e-11, 1.277986024e-16, 2.550585618e+06,-5.848769753e+02 )
   Mix.species[iN].cpCoeff.cpM      = array( 8.876501380e+04,-1.071231500e+02, 2.362188287e+00, 2.916720081e-04,-1.729515100e-07, 4.012657880e-11,-2.677227571e-15, 5.697351330e+04, 4.865231506e+00 )
   Mix.species[iN].cpCoeff.cpL      = array( 0.000000000e+00, 0.000000000e+00, 2.500000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.610463780e+04, 4.193905036e+00 )
   Mix.species[iN].DiffCoeff.sigma   = 3.298*ATom
   Mix.species[iN].DiffCoeff.kbOveps = 1.0/71.400
   Mix.species[iN].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iN].DiffCoeff.alpha   = 0.000*ATom
   Mix.species[iN].DiffCoeff.Z298    = 0.000
   -- O
   var iO = 4
   Mix.species[iO].Name = "O"
   Mix.species[iO].W = 15.9994e-3
   Mix.species[iO].Geom = 0
   Mix.species[iO].cpCoeff.TSwitch1 = 1000.0007
   Mix.species[iO].cpCoeff.TSwitch2 = 6000.0007
   Mix.species[iO].cpCoeff.TMin     = 0200.0000
   Mix.species[iO].cpCoeff.TMax     = 20000.0007
   Mix.species[iO].cpCoeff.cpH      = array( 1.779004264e+08,-1.082328257e+05, 2.810778365e+01,-2.975232262e-03, 1.854997534e-07,-5.796231540e-12, 7.191720164e-17, 8.890942630e+05,-2.181728151e+02 )
   Mix.species[iO].cpCoeff.cpM      = array( 2.619020262e+05,-7.298722030e+02, 3.317177270e+00,-4.281334360e-04, 1.036104594e-07,-9.438304330e-12, 2.725038297e-16, 3.392428060e+04,-6.679585350e-01 )
   Mix.species[iO].cpCoeff.cpL      = array(-7.953611300e+03, 1.607177787e+02, 1.966226438e+00, 1.013670310e-03,-1.110415423e-06, 6.517507500e-10,-1.584779251e-13, 2.840362437e+04, 8.404241820e+00 )
   Mix.species[iO].DiffCoeff.sigma   = 2.750*ATom
   Mix.species[iO].DiffCoeff.kbOveps = 1.0/80.000
   Mix.species[iO].DiffCoeff.mu      = 0.000*DToCm
   Mix.species[iO].DiffCoeff.alpha   = 0.000*ATom
   Mix.species[iO].DiffCoeff.Z298    = 0.000


   Mix.nReac = 5
   var i = 0
   -- Oxygen dissociation (O2 + X -> 2O + X)
   Mix.reactions[i].A    = 2.0e15
   Mix.reactions[i].n    =-1.5
   Mix.reactions[i].EovR = 59500
   Mix.reactions[i].has_backward = true
   -- Educts
   Mix.reactions[i].Neducts = 0
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i], iO2, 1.0)
   -- Products
   Mix.reactions[i].Npducts = 0
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i],  iO, 2.0)
   -- Colliders
   Mix.reactions[i].Nthirdb = 0
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iO2, 1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iNO, 1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iN2, 1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i],  iO, 5.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i],  iN, 5.0)

   i += 1
   -- NO dissociation (NO + X -> N + O + X)
   Mix.reactions[i].A    = 5e9
   Mix.reactions[i].n    = 0.0
   Mix.reactions[i].EovR = 75500
   Mix.reactions[i].has_backward = true
   -- Educts
   Mix.reactions[i].Neducts = 0
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i], iNO,  1.0)
   -- Products
   Mix.reactions[i].Npducts = 0
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i],  iO,  1.0)
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i],  iN,  1.0)
   -- Colliders
   Mix.reactions[i].Nthirdb = 0
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iO2,  1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iNO, 22.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iN2,  1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i],  iO, 22.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i],  iN, 22.0)

   i += 1
   -- N2 dissociation (N2 + X -> 2N + X)
   Mix.reactions[i].A    = 7e15
   Mix.reactions[i].n    =-1.6
   Mix.reactions[i].EovR = 113200
   Mix.reactions[i].has_backward = true
   -- Educts
   Mix.reactions[i].Neducts = 0
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i], iN2,  1.0)
   -- Products
   Mix.reactions[i].Npducts = 0
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i],  iN,  2.0)
   -- Colliders
   Mix.reactions[i].Nthirdb = 0
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iO2,  1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iNO,  1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i], iN2,  1.0)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i],  iO, 30.0/7)
   Mix.reactions[i] = REACTION.AddThirdb(Mix.reactions[i],  iN, 30.0/7)

   i += 1
   -- Zeldovich 1 (N2 + O -> NO + N)
   Mix.reactions[i].A    = 6.4e11
   Mix.reactions[i].n    =-1.0
   Mix.reactions[i].EovR = 38400
   Mix.reactions[i].has_backward = true
   -- Educts
   Mix.reactions[i].Neducts = 0
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i], iN2,  1.0)
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i],  iO,  1.0)
   -- Products
   Mix.reactions[i].Npducts = 0
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i], iNO,  1.0)
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i],  iN,  1.0)
   -- Colliders
   Mix.reactions[i].Nthirdb = 0

   i += 1
   -- Zeldovich 2 (NO + O -> O2 + N)
   Mix.reactions[i].A    = 8.4e6
   Mix.reactions[i].n    = 0.0
   Mix.reactions[i].EovR = 19400
   Mix.reactions[i].has_backward = true
   -- Educts
   Mix.reactions[i].Neducts = 0
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i], iNO,  1.0)
   Mix.reactions[i] = REACTION.AddEduct(Mix.reactions[i],  iO,  1.0)
   -- Products
   Mix.reactions[i].Npducts = 0
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i], iO2,  1.0)
   Mix.reactions[i] = REACTION.AddPduct(Mix.reactions[i],  iN,  1.0)
   -- Colliders
   Mix.reactions[i].Nthirdb = 0

   regentlib.assert(i+1 == Mix.nReac, "Something wrong with number of reactions in InitMixture")

   -- Set maximum and minimum temperature
   Mix.TMax = math.huge
   Mix.TMin = 0.0
   for i = 0, Exports.nSpec do
      Mix.TMax min= Mix.species[i].cpCoeff.TMax
      Mix.TMin max= Mix.species[i].cpCoeff.TMin
   end

   return Mix
end

__demand(__inline)
task Exports.GetSpeciesNames(Mix : Exports.Mixture)
   var Names : regentlib.string[Exports.nSpec]
   for i = 0, Exports.nSpec do
      Names[i] = Mix.species[i].Name
   end
   return Names
end

__demand(__inline)
task Exports.FindSpecies(name : int8[10], Mix : Exports.Mixture)
   var iSpec = -1
   for i = 0, Exports.nSpec do
      if C.strcmp(Mix.species[i].Name, name) == 0 then
         iSpec = i
         break
      end
   end
   regentlib.assert(iSpec > -1, "Species not found");
   return iSpec
end

__demand(__inline)
task Exports.ClipYi(Yi : double[Exports.nSpec])
   for i = 0, Exports.nSpec do
      Yi[i] max= 1.0e-60
      Yi[i] min= 1.0
   end
   return Yi
end

__demand(__inline)
task Exports.GetMolarWeightFromYi(Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var MixW = 0.0
   for i = 0, Exports.nSpec do
      MixW += Yi[i] / Mix.species[i].W
   end
   return 1.0/MixW
end

__demand(__inline)
task Exports.GetMolarWeightFromXi(Xi : double[Exports.nSpec], Mix : Exports.Mixture)
   var MixW = 0.0
   for i = 0, Exports.nSpec do
      MixW += Xi[i] * Mix.species[i].W
   end
   return MixW
end

__demand(__inline)
task Exports.GetMolarFractions(MixW : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   for i = 0, Exports.nSpec do
      Yi[i] *= MixW/Mix.species[i].W
   end
   return Yi
end

__demand(__inline)
task Exports.GetMassFractions(MixW : double, Xi : double[Exports.nSpec], Mix : Exports.Mixture)
   for i = 0, Exports.nSpec do
      Xi[i] *= Mix.species[i].W/MixW
   end
   return Xi
end

__demand(__inline)
task Exports.GetRhoFromRhoYi(rhoYi : double[Exports.nSpec])
   var rho = 0.0
   for i = 0, Exports.nSpec do
      rho += rhoYi[i]
   end
   return rho
end

__demand(__inline)
task Exports.GetYi(rho : double, rhoYi : double[Exports.nSpec])
   var rhoInv = 1.0/rho
   for i = 0, Exports.nSpec do
      rhoYi[i] *= rhoInv
   end
   return rhoYi
end

__demand(__inline)
task Exports.GetRhoYiFromYi(rho : double, Yi : double[Exports.nSpec])
   for i = 0, Exports.nSpec do
      Yi[i] *= rho
   end
   return Yi
end

__demand(__inline)
task Exports.GetRho(P : double, T : double, MixW : double, Mix : Exports.Mixture)
   return P * MixW / (RGAS * T)
end

__demand(__inline)
task Exports.GetHeatCapacity(T : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var cp = 0.0
   for i = 0, Exports.nSpec do
      cp += Yi[i]*SPECIES.GetCp(Mix.species[i], T)
   end
   return cp
end

__demand(__inline)
task Exports.GetEnthalpy(T : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var Enth = 0.0
   for i = 0, Exports.nSpec do
      Enth += Yi[i]*SPECIES.GetEnthalpy(Mix.species[i], T)
   end
   return Enth
end

__demand(__inline)
task Exports.GetSpeciesEnthalpy(i : int, T : double,  Mix : Exports.Mixture)
   return SPECIES.GetEnthalpy(Mix.species[i], T)
end

__demand(__inline)
task Exports.GetSpeciesMolarWeight(i : int, Mix : Exports.Mixture)
   return Mix.species[i].W
end

__demand(__inline)
task Exports.GetInternalEnergy(T : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var e = 0.0
   for i = 0, Exports.nSpec do
      e += Yi[i]*(SPECIES.GetEnthalpy(Mix.species[i], T) - RGAS*T/Mix.species[i].W)
   end
   return e
end

__demand(__inline)
task Exports.GetSpecificInternalEnergy(i : int, T : double,  Mix : Exports.Mixture)
   return SPECIES.GetEnthalpy(Mix.species[i], T) - RGAS*T/Mix.species[i].W
end

__demand(__inline)
task Exports.GetTFromInternalEnergy(e0 : double, T : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var MAXITS = 1000
   var TOL = 1e-5
   var dfdT = 1.0
   for j = 0, MAXITS do
      var f = e0 - Exports.GetInternalEnergy(T, Yi, Mix)
      if (fabs(f/dfdT) < TOL) then break end
      dfdT = 0.0
      for i = 0, Exports.nSpec do
         dfdT += Yi[i]*(SPECIES.GetCp(Mix.species[i], T) - RGAS/Mix.species[i].W)
      end
      T += f/dfdT
-- TODO: the assert is not yet supported by the cuda compiler 
--      regentlib.assert(j~=MAXITS, "GetTFromInternalEnergy did not converge")
   end
   return T
end

__demand(__inline)
task Exports.isValidInternalEnergy(e : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
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
task Exports.GetViscosity(T : double, Xi : double[Exports.nSpec], Mix : Exports.Mixture)
   var muk : double[Exports.nSpec]
   for i = 0, Exports.nSpec do
      muk[i] = SPECIES.GetMu(Mix.species[i], T)
   end

   var mu = 0.0
   for i = 0, Exports.nSpec do
      var den = 0.0;
      for j = 0, Exports.nSpec do
         var Phi = pow(1 + sqrt(muk[i]/muk[j]) * pow(Mix.species[j].W/Mix.species[i].W, 0.25) , 2);
         Phi /= sqrt(8*(1 + Mix.species[i].W/Mix.species[j].W));
         den += Xi[j]*Phi;
      end
      mu += Xi[i]*muk[i]/den;
   end
   return mu;
end

__demand(__inline)
task Exports.GetHeatConductivity(T : double, Xi : double[Exports.nSpec], Mix : Exports.Mixture)
   var a = 0.0
   var b = 0.0
   for i = 0, Exports.nSpec do
      var lami = SPECIES.GetLam(Mix.species[i], T)
      a += Xi[i]*lami
      b += Xi[i]/lami
   end
   return 0.5*(a + 1/b)
end

__demand(__inline)
task Exports.GetGamma(T : double, MixW : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var cp = Exports.GetHeatCapacity(T, Yi, Mix)
   return cp/(cp - RGAS/MixW)
end

__demand(__inline)
task Exports.GetSpeedOfSound(T : double, gamma : double, MixW : double, Mix : Exports.Mixture)
   return sqrt(gamma * RGAS * T / MixW)
end

__demand(__inline)
task Exports.GetDiffusivity(P: double, T : double, MixW : double, Xi : double[Exports.nSpec], Mix : Exports.Mixture)
   var invDi : double[Exports.nSpec*Exports.nSpec]
   var Di    : double[Exports.nSpec]
   for i = 0, Exports.nSpec do
      invDi[i*Exports.nSpec+i] = 0.0
      for j = 0, i do
         invDi[j*Exports.nSpec+i] = 1.0/SPECIES.GetDif(Mix.species[i], Mix.species[j], P, T)
         invDi[i*Exports.nSpec+j] = invDi[j*Exports.nSpec+i]
      end
   end

   for i = 0, Exports.nSpec do
      var num = 0.0
      var den = 0.0
      for j = 0, i do
         num += Xi[j]*Mix.species[j].W;
         den += Xi[j]*invDi[i*Exports.nSpec+j];
      end
      for j = i+1, Exports.nSpec do
         num += Xi[j]*Mix.species[j].W
         den += Xi[j]*invDi[i*Exports.nSpec+j]
      end
      Di[i] = num/(MixW*den)
   end
   return Di
end

__demand(__inline)
task Exports.GetProductionRates(rho : double, P : double, T : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var G : double[Exports.nSpec]
   var C : double[Exports.nSpec]
   var w : double[Exports.nSpec]
   for i = 0, Exports.nSpec do
      w[i] = 0.0;
      C[i] = Yi[i]*rho/Mix.species[i].W
      G[i] = SPECIES.GetFreeEnthalpy(Mix.species[i], T)
   end

   for i = 0, Mix.nReac do
      w = REACTION.AddProductionRates(Mix.reactions[i], P, T, C, G, w)
   end

   -- From [mol/(s m^3)] to [kg/(s m^3)]
   for i = 0, Exports.nSpec do
      w[i] *= Mix.species[i].W
   end
   return w
end

__demand(__inline)
task Exports.Getdpde(rho : double, gamma : double, Mix : Exports.Mixture)
   return rho*(gamma - 1)
end

__demand(__inline)
task Exports.Getdpdrhoi(gamma : double, T : double, Yi : double[Exports.nSpec], Mix : Exports.Mixture)
   var e = 0.0
   var ei : double[Exports.nSpec]
   for i = 0, Exports.nSpec do
      ei[i] = (SPECIES.GetEnthalpy(Mix.species[i], T) - RGAS*T/Mix.species[i].W)
      e += Yi[i]*ei[i]
   end
   var dpdrhoi : double[Exports.nSpec]
   for i = 0, Exports.nSpec do
      dpdrhoi[i] = RGAS*T/Mix.species[i].W + (gamma - 1)*(e - ei[i])
   end
   return dpdrhoi
end

return Exports end
