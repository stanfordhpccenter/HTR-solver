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

-- Utility functions
local log  = regentlib.log(double)
local pow  = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-- Constants
local CONST = require "prometeo_const"
local RGAS = CONST.RGAS  -- [J/(mol K)]
local Na   = CONST.Na    -- [1/mol]
local kb   = CONST.kb    -- [m^2 kg /( s^2 K)]
local PI   = CONST.PI

-- Species geometries
--Mixture.SpeciesGeom = Enum( 'Atom', 'Linear', 'NonLinear' )
local SpeciesGeom_Atom      = 0
local SpeciesGeom_Linear    = 1
local SpeciesGeom_NonLinear = 2

Exports.SpeciesGeom_Atom      = SpeciesGeom_Atom
Exports.SpeciesGeom_Linear    = SpeciesGeom_Linear
Exports.SpeciesGeom_NonLinear = SpeciesGeom_NonLinear

-- NASA polynomials data structure
local struct cpCoefficients
{
   TSwitch1 : double     -- Switch  temperature between Low and Mid  temperature polynomials
   TSwitch2 : double     -- Switch  temperature between Mid and High temperature polynomials
   TMin     : double     -- Minimum temperature
   TMax     : double     -- Maximum temperature
   cpH      : double[9]  -- High temperature polynomials
   cpM      : double[9]  -- Mid  temperature polynomials
   cpL      : double[9]  -- Low  temperature polynomials
}

-- Coefficinets for diffusivity
local struct DiffCoefficients
{
   sigma   : double   -- Lennard-Jones collision diameter [m]
   kbOveps : double   -- Boltzmann constant divided by Lennard-Jones potential well depth [1/K]
   mu      : double   -- Dipole moment [C*m]
   alpha   : double   -- Polarizabilty [m]
   Z298    : double   -- Rotational relaxation collision number
}

-- Species structure
struct Exports.Species {
   Name      : int8[10]         -- regentlib.string -- Name of the species
   W         : double           -- Molar weight [kg/mol]
   inx       : int              -- Index in the species vector
   Geom      : int              -- = 0 (Atom), = 1 (Linear), = 2 (Non Linear)
   cpCoeff   : cpCoefficients
   DiffCoeff : DiffCoefficients
}

--  omega_mu() returns the collision integral for mu given dimensionless temperature t/(eps/k).
--  TODO: These come from FlameMaster.
--        At a certain point, verify these implementations.
local __demand(__inline)
task omega_mu( T: double )
   var m1 = 3.3530622607
   var m2 = 2.53272006
   var m3 = 2.9024238575
   var m4 = 0.11186138893
   var m5 = 0.8662326188       -- = -0.1337673812 + 1.0
   var m6 = 1.3913958626
   var m7 = 3.158490576
   var m8 = 0.18973411754
   var m9 = 0.00018682962894

   var num = m1 + T*(m2 + T*(m3 + T*m4))
   var den = m5 + T*(m6 + T*(m7 + T*(m8 + T*m9)))
   return num / den
end

-- omega_D() returns the Stossintegral for a given dimensionless temperature t/(eps/k)
local __demand(__inline)
task omega_D( T : double )
   var m1 = 6.8728271691
   var m2 = 9.4122316321
   var m3 = 7.7442359037
   var m4 = 0.23424661229
   var m5 = 1.45337701568         -- = 1.0 + 0.45337701568
   var m6 = 5.2269794238
   var m7 = 9.7108519575
   var m8 = 0.46539437353
   var m9 = 0.00041908394781

   return (m1 + T * (m2 + T * (m3 + T * m4))) / (m5 + T * (m6 + T * (m7 + T * (m8 + T * m9))))
end

__demand(__inline)
task Exports.GetCp( s : Exports.Species, T : double )
-- TODO: the assert is not yet supported by the cuda compiler
--   regentlib.assert(T < s.cpCoeff.TMax, "Exceeded maximum temeperature")
--   regentlib.assert(T > s.cpCoeff.TMin, "Exceeded minimum temeperature")

   var rOvW = RGAS/s.W
   var Tinv = 1.0/T
   var cpCoeff : double[9]
   if ( T > s.cpCoeff.TSwitch2 ) then
      cpCoeff = s.cpCoeff.cpH
   elseif ( T > s.cpCoeff.TSwitch1 ) then
      cpCoeff = s.cpCoeff.cpM
   else
      cpCoeff = s.cpCoeff.cpL
   end
   var cp = rOvW*( cpCoeff[0]*Tinv*Tinv + cpCoeff[1]*Tinv + cpCoeff[2] + T*
                                                          ( cpCoeff[3] + T*
                                                          ( cpCoeff[4] + T*
                                                          ( cpCoeff[5] + T*cpCoeff[6]))))
   return cp
end

__demand(__inline)
task Exports.GetFreeEnthalpy( s : Exports.Species, T : double )
   -- This is (H/(RT) - S/R)
-- TODO: the assert is not yet supported by the cuda compiler
--   regentlib.assert(T < s.cpCoeff.TMax, "Exceeded maximum temeperature")
--   regentlib.assert(T > s.cpCoeff.TMin, "Exceeded minimum temeperature")

   var Tinv = 1.0/T
   var logT = log(T)
   var cpCoeff : double[9]
   if ( T > s.cpCoeff.TSwitch2 ) then
      cpCoeff = s.cpCoeff.cpH
   elseif ( T > s.cpCoeff.TSwitch1 ) then
      cpCoeff = s.cpCoeff.cpM
   else
      cpCoeff = s.cpCoeff.cpL
   end

   var G = -0.5*cpCoeff[0]*Tinv*Tinv + cpCoeff[1]*Tinv*(1.0 + logT) + cpCoeff[2]*(1.0 - logT) + cpCoeff[7]*Tinv - cpCoeff[8]
   G -= 0.5*T*( cpCoeff[3]   + T*
              ( cpCoeff[4]/3 + T*
              ( cpCoeff[5]/6 + 0.1*T*cpCoeff[6] )))

   return G
end

__demand(__inline)
task Exports.GetEnthalpy( s : Exports.Species, T : double )
-- TODO: the assert is not yet supported by the cuda compiler
   --regentlib.assert(T < s.cpCoeff.TMax, "Exceeded maximum temeperature")
   --regentlib.assert(T > s.cpCoeff.TMin, "Exceeded minimum temeperature")

   var rOvW = RGAS/s.W
   var Tinv = 1.0/T
   var cpCoeff : double[9]
   if ( T > s.cpCoeff.TSwitch2 ) then
      cpCoeff = s.cpCoeff.cpH
   elseif ( T > s.cpCoeff.TSwitch1 ) then
      cpCoeff = s.cpCoeff.cpM
   else
      cpCoeff = s.cpCoeff.cpL
   end

   var E = -cpCoeff[0]/T + cpCoeff[1]*log(T) + cpCoeff[7]      + T*
                                             ( cpCoeff[2]      + T*
                                             ( cpCoeff[3]*0.50 + T*
                                             ( cpCoeff[4]/3    + T*
                                             ( cpCoeff[5]*0.25 + cpCoeff[6]/5*T))))
   return E*rOvW
end

__demand(__inline)
task Exports.GetMu( s : Exports.Species, T : double )
   var num = 5 * sqrt(PI * s.W/Na * kb * T)
   var den = 16 * PI * pow(s.DiffCoeff.sigma,2) * omega_mu( T * s.DiffCoeff.kbOveps )
   return num/den
end

__demand(__inline)
task Exports.GetDif( s1 : Exports.Species, s2 : Exports.Species,
                     P  : double, T  : double )
   var xi = 1.0
   if ( (s1.DiffCoeff.mu*s2.DiffCoeff.mu == 0.0) and 
        (s1.DiffCoeff.mu+s2.DiffCoeff.mu ~= 0.0) ) then
      -- If I have a polar to non-polar molecule interaction
      var mup : double
      var alp : double
      var epr : double
      if (s1.DiffCoeff.mu ~= 0.0) then
         mup = s1.DiffCoeff.mu/sqrt(pow(s1.DiffCoeff.sigma,3)*kb/s1.DiffCoeff.kbOveps)
         alp = s1.DiffCoeff.alpha/s1.DiffCoeff.sigma
         epr = sqrt(s2.DiffCoeff.kbOveps/s1.DiffCoeff.kbOveps)
      else 
         mup = s2.DiffCoeff.mu/sqrt(pow(s2.DiffCoeff.sigma,3)*kb/s2.DiffCoeff.kbOveps)
         alp = s2.DiffCoeff.alpha/s2.DiffCoeff.sigma
         epr = sqrt(s1.DiffCoeff.kbOveps/s2.DiffCoeff.kbOveps)
      end
      xi = 1 + 0.25*mup*alp*epr
   end
   var invWij = (s1.W + s2.W)/(s1.W*s2.W)
   var kboEpsij = sqrt(s1.DiffCoeff.kbOveps * s2.DiffCoeff.kbOveps)/(xi*xi)
   var sigmaij = 0.5*(s1.DiffCoeff.sigma + s2.DiffCoeff.sigma)*pow(xi,1./6)
   var num = 3*sqrt(2*PI*pow(kb,3)*pow(T,3)*Na*invWij)
   var den = 16*PI*P*sigmaij*sigmaij*omega_D(T * kboEpsij)
   return num/den
end

local __demand(__inline)
task GetSelfDiffusion( s : Exports.Species, T : double )
   -- Already multiplied by partial density
   var num = 3*sqrt( PI*kb*T*s.W/Na )
   var den = 8*PI*pow(s.DiffCoeff.sigma,2)*omega_D(T * s.DiffCoeff.kbOveps)
   return num/den
end

local __demand(__inline)
task GetFZrot( s : Exports.Species, T : double )
   var tmp = 1/(s.DiffCoeff.kbOveps*T)
   return 1 + 0.5*pow(PI,1.5)*sqrt(tmp)
            + (2 + 0.25*PI*PI)*tmp
            + pow(PI,1.5)*pow(tmp,1.5)
end

local __demand(__inline)
task GetLamAtom( s : Exports.Species, T : double )
   var mu = Exports.GetMu(s, T)
   return 15.0/4*mu*RGAS/s.W
end


local __demand(__inline)
task GetLamLinear( s : Exports.Species, T : double )
   var CvTOvR = 1.5
   var CvROvR = 1.0

   var CvT = CvTOvR*RGAS
   var CvR = CvROvR*RGAS
   var CvV = Exports.GetCp(s, T)*s.W - 3.5*RGAS

   var Dkk = GetSelfDiffusion(s, T)
   var mu = Exports.GetMu(s,T)

   var fV = Dkk/mu

   var Zrot = s.DiffCoeff.Z298*GetFZrot(s, 298)/GetFZrot(s, T)

   var A = 2.5 - fV
   var B = Zrot + 2/PI*(5./3*CvROvR+fV)

   var fT = 2.5 * (1. - 2*CvR*A/(PI*CvT*B))
   var fR = fV*(1. + 2*A/(PI*B))

   return mu/s.W*(fT*CvT + fR*CvR + fV*CvV)
end

local __demand(__inline)
task GetLamNonLinear( s : Exports.Species, T : double )
   var CvTOvR = 1.5
   var CvROvR = 1.5

   var CvT = CvTOvR*RGAS
   var CvR = CvROvR*RGAS
   var CvV = Exports.GetCp(s, T)*s.W - 4.0*RGAS

   var Dkk = GetSelfDiffusion(s, T)
   var mu  = Exports.GetMu(s, T)

   var fV = Dkk/mu

   var Zrot = s.DiffCoeff.Z298*GetFZrot(s, 298.0)/GetFZrot(s, T)

   var A = 2.5 - fV
   var B = Zrot + 2/PI*(5./3*CvROvR+fV)

   var fT = 2.5 * (1. - 2*CvR*A/(PI*CvT*B))
   var fR = fV*(1. + 2*A/(PI*B))

   return mu/s.W*(fT*CvT + fR*CvR + fV*CvV)
end

__demand(__inline)
task Exports.GetLam( s : Exports.Species, T : double )
   var lam : double
   if      (s.Geom == SpeciesGeom_Atom)      then lam = GetLamAtom(s, T)
   elseif  (s.Geom == SpeciesGeom_Linear)    then lam = GetLamLinear(s, T)
   elseif  (s.Geom == SpeciesGeom_NonLinear) then lam = GetLamNonLinear(s, T) end
   return lam
end

return Exports
