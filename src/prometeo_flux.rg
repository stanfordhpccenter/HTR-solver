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

return function(SCHEMA, MIX, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require 'util-desugared'
local MATH = require 'math_utils'
local MACRO = require "prometeo_macro"
local CONST = require "prometeo_const"

local fabs = regentlib.fabs(double)
local pow  = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

-- Stencil indices
local Stencil1  = CONST.Stencil1
local Stencil2  = CONST.Stencil2
local Stencil3  = CONST.Stencil3
local Stencil4  = CONST.Stencil4
local nStencils = CONST.nStencils

local Primitives = CONST.Primitives
local Properties = CONST.Properties

local LUdec, ludcmp, lubksb = unpack(MATH.mkLUdec(nEq))

-------------------------------------------------------------------------------
-- EULER FLUXES ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.GetEulerFlux(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.Conserved),
   reads(Fluid.[Primitives]),
   writes(Fluid.{EulerFluxX, EulerFluxY, EulerFluxZ})
do
   __demand(__openmp)
   for c in Fluid do
      var Eflux : double[nEq]

      -- X direction
      for i=0, nEq do
         Eflux[i] = Fluid[c].Conserved[i]*Fluid[c].velocity[0]
      end
      Eflux[0+irU] += Fluid[c].pressure
      Eflux[  irE] += Fluid[c].pressure*Fluid[c].velocity[0]
      Fluid[c].EulerFluxX = Eflux

      -- Y direction
      for i=0, nEq do
         Eflux[i] = Fluid[c].Conserved[i]*Fluid[c].velocity[1]
      end
      Eflux[1+irU] += Fluid[c].pressure
      Eflux[  irE] += Fluid[c].pressure*Fluid[c].velocity[1]
      Fluid[c].EulerFluxY = Eflux

      -- Z direction
      for i=0, nEq do
         Eflux[i] = Fluid[c].Conserved[i]*Fluid[c].velocity[2]
      end
      Eflux[2+irU] += Fluid[c].pressure
      Eflux[  irE] += Fluid[c].pressure*Fluid[c].velocity[2]
      Fluid[c].EulerFluxZ = Eflux
   end
end

local mkGetCharacteristicMapping = terralib.memoize(function(dir)
   local GetCharacteristicMapping

   local ind
   if (dir == "x") then
      ind = 0
   elseif (dir == "y") then
      ind = 1
   elseif (dir == "z") then
      ind = 2
   else assert(false) end

   local __demand(__inline)
   task GetCharacteristicMapping(Fluid : region(ispace(int3d), Fluid_columns),
                                     c : int3d,
                                   cp1 : int3d,
                                   mix : MIX.Mixture) 
   where
      reads(Fluid.rho),
      reads(Fluid.Conserved),
      reads(Fluid.[Primitives])
   do
      -- Compute quantities on the left (L) and right (R) states
      var MixWL = MIX.GetMolarWeightFromXi(Fluid[c  ].MolarFracs, mix)
      var MixWR = MIX.GetMolarWeightFromXi(Fluid[cp1].MolarFracs, mix)
   
      var MassFracsL = MIX.GetMassFractions(MixWL, Fluid[c  ].MolarFracs, mix)
      var MassFracsR = MIX.GetMassFractions(MixWR, Fluid[cp1].MolarFracs, mix)
   
      var gammaL = MIX.GetGamma(Fluid[c  ].temperature, MixWL, MassFracsL, mix)
      var gammaR = MIX.GetGamma(Fluid[cp1].temperature, MixWR, MassFracsR, mix)
   
      var dpdrhoiL = MIX.Getdpdrhoi(gammaL, Fluid[c  ].temperature, MassFracsL, mix)
      var dpdrhoiR = MIX.Getdpdrhoi(gammaR, Fluid[cp1].temperature, MassFracsR, mix)
   
      var dpdeL = MIX.Getdpde(Fluid[c  ].rho, gammaL, mix)
      var dpdeR = MIX.Getdpde(Fluid[cp1].rho, gammaR, mix)
   
      var TotalEnergyL = Fluid[c  ].Conserved[irE]/Fluid[c  ].rho
      var TotalEnergyR = Fluid[cp1].Conserved[irE]/Fluid[cp1].rho
   
      var TotalEnthalpyL = TotalEnergyL + Fluid[c  ].pressure/Fluid[c  ].rho
      var TotalEnthalpyR = TotalEnergyR + Fluid[cp1].pressure/Fluid[cp1].rho
   
      -- Compute Roe averaged state
      var RoeFactorL = sqrt(Fluid[c  ].rho)/(sqrt(Fluid[c].rho) + sqrt(Fluid[cp1].rho))
      var RoeFactorR = sqrt(Fluid[cp1].rho)/(sqrt(Fluid[c].rho) + sqrt(Fluid[cp1].rho))
   
      var rhoRoe = sqrt(Fluid[c].rho*Fluid[cp1].rho)
   
      var YiRoe : double[nSpec]
      for i=0, nSpec do
         YiRoe[i] = MassFracsL[i]*RoeFactorL + MassFracsR[i]*RoeFactorR
      end
   
      var uRoe = MACRO.vv_add( MACRO.vs_mul(Fluid[c  ].velocity, RoeFactorL),
                               MACRO.vs_mul(Fluid[cp1].velocity, RoeFactorR))
      var HRoe =  TotalEnthalpyL*RoeFactorL +  TotalEnthalpyR*RoeFactorR
      var ERoe =    TotalEnergyL*RoeFactorL +    TotalEnergyR*RoeFactorR
   
      var dpdrhoiRoe : double[nSpec]
      for i=0, nSpec do
         dpdrhoiRoe[i] = dpdrhoiL[i]*RoeFactorL + dpdrhoiR[i]*RoeFactorR
      end
      var dpdeRoe =            dpdeL*RoeFactorL +       dpdeR*RoeFactorR
   
      -- correct the pressure derivatives in order to satisfy the pressure jump condition
      -- using the procedure in Shuen, Liou and Leer (1990)
      var dp = Fluid[cp1].pressure - Fluid[c].pressure
      var de = TotalEnergyR - 0.5*MACRO.dot(Fluid[cp1].velocity, Fluid[cp1].velocity)
      de    -= TotalEnergyL - 0.5*MACRO.dot(Fluid[c  ].velocity, Fluid[c  ].velocity)
      var drhoi : double[nSpec]
      for i=0, nSpec do
         drhoi[i] = Fluid[cp1].Conserved[i] - Fluid[c].Conserved[i]
      end
   
      -- find the error in the pressure jump due to Roe averages
      var dpError = dp - de*dpdeRoe
      var fact = pow(de*dpdeRoe, 2)
      for i=0, nSpec do
         dpError -=  drhoi[i]*dpdrhoiRoe[i]
         fact += pow(drhoi[i]*dpdrhoiRoe[i], 2)
      end
   
      -- correct pressure derivatives
      -- this threshold should not be affect the solution since fact is zero when all the jumps are zero
      fact = 1.0/max(fact, 1e-6)
      var dpdrhoiHat : double[nSpec]
      for i=0, nSpec do
         dpdrhoiHat[i] = dpdrhoiRoe[i]*(1.0 + dpdrhoiRoe[i]*drhoi[i]*dpError*fact)
      end
      var dpdeHat = dpdeRoe*(1.0 + dpdeRoe*de*dpError*fact)
   
      -- compute the Roe averaged speed of sound
      var PovRhoRoe = HRoe - ERoe
      var aRoe = PovRhoRoe/rhoRoe*dpdeHat 
      for i=0, nSpec do
         aRoe += YiRoe[i]*dpdrhoiHat[i]
      end
      aRoe = sqrt(aRoe)
   
      -- build the eigenvector matrix
      var K = [UTIL.mkArrayConstant(nEq*nEq, rexpr 0.0 end)]
   
      -- identify face normal and tangential directions
      var iN  = (ind+0)%3
      var iT1 = (ind+1)%3
      var iT2 = (ind+2)%3
   
      -- fill face normal
      var n : double[3]
      n[iN ] = 1.0
      n[iT1] = 0.0
      n[iT2] = 0.0
   
      for i=0, nSpec do
         K[i*nEq +       0] = YiRoe[i]
         K[i*nEq +     i+1] = 1.0
         K[i*nEq + nSpec+3] = YiRoe[i]
      end
   
      K[(nSpec+0)*nEq +       0] = uRoe[0] - aRoe*n[0]
      for i=0, nSpec do
         K[(nSpec+0)*nEq +  i+1] = uRoe[0]
      end
      K[(nSpec+0)*nEq + nSpec+1] = n[1] + n[2]
      K[(nSpec+0)*nEq + nSpec+2] = 0.0
      K[(nSpec+0)*nEq + nSpec+3] = uRoe[0] + aRoe*n[0]
   
      K[(nSpec+1)*nEq +       0] = uRoe[1] - aRoe*n[1]
      for i=0, nSpec do
         K[(nSpec+1)*nEq +  i+1] = uRoe[1]
      end
      K[(nSpec+1)*nEq + nSpec+1] = n[0]
      K[(nSpec+1)*nEq + nSpec+2] = n[2]
      K[(nSpec+1)*nEq + nSpec+3] = uRoe[1] + aRoe*n[1]
   
      K[(nSpec+2)*nEq +       0] = uRoe[2] - aRoe*n[2]
      for i=0, nSpec do
         K[(nSpec+2)*nEq +  i+1] = uRoe[2]
      end
      K[(nSpec+2)*nEq + nSpec+1] = 0.0
      K[(nSpec+2)*nEq + nSpec+2] = n[0] + n[1]
      K[(nSpec+2)*nEq + nSpec+3] = uRoe[2] + aRoe*n[2]
   
      K[(nSpec+3)*nEq +       0] = HRoe - uRoe[iN]*aRoe
      for i=0, nSpec do
         K[(nSpec+3)*nEq +  i+1] = ERoe - rhoRoe*dpdrhoiHat[i]/dpdeHat
      end
      K[(nSpec+3)*nEq + nSpec+1] = uRoe[1]*n[0] + uRoe[0]*n[1] + uRoe[0]*n[2]
      K[(nSpec+3)*nEq + nSpec+2] = uRoe[2]*n[0] + uRoe[2]*n[1] + uRoe[1]*n[2]
      K[(nSpec+3)*nEq + nSpec+3] = HRoe + uRoe[iN]*aRoe
   
      return K
   end
   return GetCharacteristicMapping
end)

local __demand(__inline)
task GetPlusFlux( F : double[nEq],
                  Q : double[nEq],
                lam : double[nEq])
   for i=0, nEq do
      F[i] += lam[i]*Q[i]
   end
   return F
end

local __demand(__inline)
task GetMinusFlux( F : double[nEq],
                   Q : double[nEq],
                 lam : double[nEq])
   for i=0, nEq do
      F[i] -= lam[i]*Q[i]
   end
   return F
end

local mkReconstructEulerFlux = terralib.memoize(function(dir)
   local ReconstructEulerFlux

   local ind
   local FluxC
   local EulerFlux
   local reconFacePlus
   local reconFaceMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if (dir == "x") then
      ind = 0
      FluxC  = "FluxXCorr"
      EulerFlux  = "EulerFluxX"
      reconFacePlus  = "reconXFacePlus"
      reconFaceMinus = "reconXFaceMinus"
      TENOCoeffsPlus  = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif (dir == "y") then
      ind = 1
      FluxC  = "FluxYCorr"
      EulerFlux  = "EulerFluxY"
      reconFacePlus  = "reconYFacePlus"
      reconFaceMinus = "reconYFaceMinus"
      TENOCoeffsPlus  = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif (dir == "z") then
      ind = 2
      FluxC  = "FluxZCorr"
      EulerFlux  = "EulerFluxZ"
      reconFacePlus  = "reconZFacePlus"
      reconFaceMinus = "reconZFaceMinus"
      TENOCoeffsPlus  = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   else assert(false) end

   -- TENO coefficients
   local r_c = 0.17
   local kethe = 1.e-8
   local cut_off_bc = 1.0e-3
   local incom_int = 9.5
   local diff_int = incom_int - 6.0

   local __demand(__inline)
   task TENOReconstructPlus(ym2 : double, ym1 : double, y : double, yp1 : double, yp2 : double, yp3 : double, ReconCoeffs : double[nStencils*6], TENOCoeffs : double[nStencils], BCStencil : bool)
   
      var s1 = 13.0 /12.0*(ym1 - 2.0*y   + yp1)*(ym1 - 2.0*y   + yp1) + 3.0/12.0*(ym1   - yp1)*(ym1 - yp1)
      var s2 = 13.0 /12.0*(y   - 2.0*yp1 + yp2)*(y   - 2.0*yp1 + yp2) + 3.0/12.0*(3.0*y - 4.0*yp1 + yp2)*(3.0*y - 4.0*yp1 + yp2)
      var s3 = 13.0 /12.0*(ym2 - 2.0*ym1 + y  )*(ym2 - 2.0*ym1 + y  ) + 3.0/12.0*(ym2   - 4.0*ym1 + 3.0*y)*(ym2 - 4.0*ym1 + 3.0*y)
      var s4 = 1.0/ 240.0*fabs((2107.0*y*y - 9402.0*y*yp1 + 11003.0*yp1*yp1 + 7042.0*y*yp2 - 17246.0*yp1*yp2 + 7043.0*yp2*yp2 - 1854.0*y*yp3 + 4642.0*yp1*yp3 - 3882.0*yp2*yp3 + 547.0*yp3*yp3))
   
      var s64 = 1.0 / 12.0*fabs(139633.0*yp3*yp3 - 1429976.0* yp2*yp3 + 3824847.0*yp2*yp2 + 2863984.0*yp1*yp3 - 15880404.0*yp1*yp2 + 17195652.0*yp1*yp1 - 2792660.0* y  *yp3
                            - 35817664.0*y  *yp1 + 19510972.0*y  *y   + 1325006.0*ym1*yp3 - 7727988.0*ym1*yp2 + 17905032.0*ym1*yp1 - 20427884.0*ym1*y   + 5653317.0* ym1*ym1
                            - 245620.0*  ym2*yp3 + 1458762.0* ym2*yp2 - 3462252.0*ym2*yp1 + 4086352.0*ym2*y   - 2380800.0* ym2*ym1 + 271779.0*  ym2*ym2 + 15929912.0*y  *yp2) / 10080.0
   
      var tau6 = fabs(s64 - (s3 + s2 + 4.0*s1) / 6.0)
   
      if BCStencil then tau6 = 1.0 end
   
      -- not recommend to rescale the small number
      var a1 = pow(1.0 + tau6 / (s1 + 1.0e-8), 6.0)
      var a2 = pow(1.0 + tau6 / (s2 + 1.0e-8), 6.0)
      var a3 = pow(1.0 + tau6 / (s3 + 1.0e-8), 6.0)
      var a4 = pow(1.0 + tau6 / (s4 + 1.0e-8), 6.0)
   
      if (fabs(TENOCoeffs[Stencil1]) < 1e-10) then a1 = 0.0 end 
      if (fabs(TENOCoeffs[Stencil2]) < 1e-10) then a2 = 0.0 end
      if (fabs(TENOCoeffs[Stencil3]) < 1e-10) then a3 = 0.0 end
      if (fabs(TENOCoeffs[Stencil4]) < 1e-10) then a4 = 0.0 end
   
      var a = 1.0/(a1 + a2 + a3 + a4)
      var b1 = a1*a
      var b2 = a2*a
      var b3 = a3*a
      var b4 = a4*a
   
      var var2 = fabs(ym1 - ym2)
      var var3 = fabs(y   - ym1)
      var var4 = fabs(yp1 - y  )
      var var5 = fabs(yp2 - yp1)
      var var6 = fabs(yp3 - yp2)
   
      -- the parameter r_c can be tuned to be around 0.2 if necessary
      var eps = 0.9*r_c*kethe*kethe / (1. - 0.9*r_c)
   
      var r = min((2.0*var2*var3 + eps) / (var2*var2 + var3*var3 + eps),
                  (2.0*var3*var4 + eps) / (var3*var3 + var4*var4 + eps))
      r min=      (2.0*var4*var5 + eps) / (var4*var4 + var5*var5 + eps)
      r min=      (2.0*var5*var6 + eps) / (var5*var5 + var6*var6 + eps)
   
      var delta = 1.0 - min(r / r_c, 1.0)
   
      var decay = pow((1.0 - delta), 4.0) * (1.0 + 4.0 * delta)
      var power_int = incom_int - diff_int * (1.0 - decay)
      var cut_off = pow(10.0, (-power_int))
   
      if BCStencil then cut_off = cut_off_bc end
   
      if (b1 < cut_off) then b1 = 0.0 else b1 = 1.0 end
      if (b2 < cut_off) then b2 = 0.0 else b2 = 1.0 end
      if (b3 < cut_off) then b3 = 0.0 else b3 = 1.0 end
      if (b4 < cut_off) then b4 = 0.0 else b4 = 1.0 end
   
      var Variation1 = ym2*ReconCoeffs[Stencil1*6+0] +
                       ym1*ReconCoeffs[Stencil1*6+1] +
                       y  *ReconCoeffs[Stencil1*6+2] +
                       yp1*ReconCoeffs[Stencil1*6+3] +
                       yp2*ReconCoeffs[Stencil1*6+4] +
                       yp3*ReconCoeffs[Stencil1*6+5] - y
   
      var Variation2 = ym2*ReconCoeffs[Stencil2*6+0] +
                       ym1*ReconCoeffs[Stencil2*6+1] +
                       y  *ReconCoeffs[Stencil2*6+2] +
                       yp1*ReconCoeffs[Stencil2*6+3] +
                       yp2*ReconCoeffs[Stencil2*6+4] +
                       yp3*ReconCoeffs[Stencil2*6+5] - y
   
      var Variation3 = ym2*ReconCoeffs[Stencil3*6+0] +
                       ym1*ReconCoeffs[Stencil3*6+1] +
                       y  *ReconCoeffs[Stencil3*6+2] +
                       yp1*ReconCoeffs[Stencil3*6+3] +
                       yp2*ReconCoeffs[Stencil3*6+4] +
                       yp3*ReconCoeffs[Stencil3*6+5] - y
   
      var Variation4 = ym2*ReconCoeffs[Stencil4*6+0] +
                       ym1*ReconCoeffs[Stencil4*6+1] +
                       y  *ReconCoeffs[Stencil4*6+2] +
                       yp1*ReconCoeffs[Stencil4*6+3] +
                       yp2*ReconCoeffs[Stencil4*6+4] +
                       yp3*ReconCoeffs[Stencil4*6+5] - y
   
      -- this is 6-point 6th-order reconstruction TENO6-A
      a1 = TENOCoeffs[Stencil1] * b1
      a2 = TENOCoeffs[Stencil2] * b2
      a3 = TENOCoeffs[Stencil3] * b3
      a4 = TENOCoeffs[Stencil4] * b4
   
      a = 1.0/(a1 + a2 + a3 + a4)
      var w1 = a1*a
      var w2 = a2*a
      var w3 = a3*a
      var w4 = a4*a
   
      return y + w1*Variation1 + w2*Variation2 + w3*Variation3 + w4*Variation4
   end
   
   local __demand(__inline)
   task TENOReconstructMinus(ym2 : double, ym1 : double, y : double, yp1 : double, yp2 : double, yp3 : double, ReconCoeffs : double[nStencils*6], TENOCoeffs : double[nStencils], BCStencil : bool)
   
      var s1 = 13.0/12.0*(yp2 - 2.0*yp1 + y  )*(yp2 - 2.0*yp1 + y  ) + 3.0 / 12.0*(yp2 - y)*(yp2 - y)
      var s2 = 13.0/12.0*(yp1 - 2.0*y   + ym1)*(yp1 - 2.0*y   + ym1) + 3.0 / 12.0*(3.0*yp1 - 4.0*y + ym1)*(3.0*yp1 - 4.0*y + ym1)
      var s3 = 13.0/12.0*(yp3 - 2.0*yp2 + yp1)*(yp3 - 2.0*yp2 + yp1) + 3.0 / 12.0*(yp3 - 4.0*yp2 + 3.0*yp1)*(yp3 - 4.0*yp2 + 3.0*yp1)
      var s4 = 1.0/240.0*fabs((2107.0*yp1*yp1 - 9402.0*yp1*y + 11003.0*y*y + 7042.0*yp1*ym1 - 17246.0*y*ym1 + 7043.0*ym1*ym1 - 1854.0*yp1*ym2 + 4642.0*y*ym2 - 3882.0*ym1*ym2 + 547.0*ym2*ym2))
   
      var s64 = 1.0 / 12.0*fabs(139633.0*ym2*ym2 - 1429976.0* ym1*ym2 + 3824847.0*ym1*ym1 + 2863984.0*y  *ym2 - 15880404.0*y  *ym1 + 17195652.0*y  *y   - 2792660.0* yp1*ym2
   	                      - 35817664.0*yp1*y   + 19510972.0*yp1*yp1 + 1325006.0*yp2*ym2 - 7727988.0*yp2*ym1 + 17905032.0*yp2*y   - 20427884.0*yp2*yp1 + 5653317.0* yp2*yp2
   	                      - 245620.0*  yp3*ym2 + 1458762.0* yp3*ym1 - 3462252.0*yp3*y   + 4086352.0*yp3*yp1 - 2380800.0* yp3*yp2 + 271779.0*  yp3*yp3 + 15929912.0*yp1*ym1) / 10080.0
   
      var tau6 = fabs(s64 - (s3 + s2 + 4.0*s1) / 6.0)
   
      if BCStencil then tau6 = 1.0 end
   
      -- not recommend to rescale the small number
      var a1 = pow(1.0 + tau6 / (s1 + 1.0e-8), 6.0)
      var a2 = pow(1.0 + tau6 / (s2 + 1.0e-8), 6.0)
      var a3 = pow(1.0 + tau6 / (s3 + 1.0e-8), 6.0)
      var a4 = pow(1.0 + tau6 / (s4 + 1.0e-8), 6.0)
   
      if (fabs(TENOCoeffs[Stencil1]) < 1e-10) then a1 = 0.0 end 
      if (fabs(TENOCoeffs[Stencil2]) < 1e-10) then a2 = 0.0 end
      if (fabs(TENOCoeffs[Stencil3]) < 1e-10) then a3 = 0.0 end
      if (fabs(TENOCoeffs[Stencil4]) < 1e-10) then a4 = 0.0 end
   
      var a = 1.0/(a1 + a2 + a3 + a4)
      var b1 = a1*a
      var b2 = a2*a
      var b3 = a3*a
      var b4 = a4*a
   
      var var2 = fabs(yp2 - yp3)
      var var3 = fabs(yp1 - yp2)
      var var4 = fabs(y   - yp1)
      var var5 = fabs(ym1 - y)
      var var6 = fabs(ym2 - ym1)
   
      -- the parameter r_c can be tuned to be around 0.2 if necessary
      var eps = 0.9*r_c*kethe*kethe / (1.0 - 0.9*r_c)
   
      var r = min((2.0*var2*var3 + eps) / (var2*var2 + var3*var3 + eps),
                  (2.0*var3*var4 + eps) / (var3*var3 + var4*var4 + eps))
      r min=      (2.0*var4*var5 + eps) / (var4*var4 + var5*var5 + eps)
      r min=      (2.0*var5*var6 + eps) / (var5*var5 + var6*var6 + eps)
   
      var delta = 1.0 - min(r / r_c, 1.0)
   
      var decay = pow((1.0 - delta), 4.0) * (1.0 + 4.0 * delta)
      var power_int = incom_int - diff_int * (1.0 - decay)
      var cut_off = pow(10.0, (-power_int))
   
      if BCStencil then cut_off = cut_off_bc end
   
      if (b1 < cut_off) then b1 = 0.0 else b1 = 1.0 end
      if (b2 < cut_off) then b2 = 0.0 else b2 = 1.0 end
      if (b3 < cut_off) then b3 = 0.0 else b3 = 1.0 end
      if (b4 < cut_off) then b4 = 0.0 else b4 = 1.0 end
   
      var Variation1 = ym2*ReconCoeffs[Stencil1*6+0] +
                       ym1*ReconCoeffs[Stencil1*6+1] +
                       y  *ReconCoeffs[Stencil1*6+2] +
                       yp1*ReconCoeffs[Stencil1*6+3] +
                       yp2*ReconCoeffs[Stencil1*6+4] +
                       yp3*ReconCoeffs[Stencil1*6+5] - yp1
   
      var Variation2 = ym2*ReconCoeffs[Stencil2*6+0] +
                       ym1*ReconCoeffs[Stencil2*6+1] +
                       y  *ReconCoeffs[Stencil2*6+2] +
                       yp1*ReconCoeffs[Stencil2*6+3] +
                       yp2*ReconCoeffs[Stencil2*6+4] +
                       yp3*ReconCoeffs[Stencil2*6+5] - yp1
   
      var Variation3 = ym2*ReconCoeffs[Stencil3*6+0] +
                       ym1*ReconCoeffs[Stencil3*6+1] +
                       y  *ReconCoeffs[Stencil3*6+2] +
                       yp1*ReconCoeffs[Stencil3*6+3] +
                       yp2*ReconCoeffs[Stencil3*6+4] +
                       yp3*ReconCoeffs[Stencil3*6+5] - yp1
   
      var Variation4 = ym2*ReconCoeffs[Stencil4*6+0] +
                       ym1*ReconCoeffs[Stencil4*6+1] +
                       y  *ReconCoeffs[Stencil4*6+2] +
                       yp1*ReconCoeffs[Stencil4*6+3] +
                       yp2*ReconCoeffs[Stencil4*6+4] +
                       yp3*ReconCoeffs[Stencil4*6+5] - yp1
   
      -- this is 6-point 6th-order reconstruction TENO6-A
      a1 = TENOCoeffs[Stencil1] * b1
      a2 = TENOCoeffs[Stencil2] * b2
      a3 = TENOCoeffs[Stencil3] * b3
      a4 = TENOCoeffs[Stencil4] * b4
   
      a = 1.0/(a1 + a2 + a3 + a4)
      var w1 = a1*a
      var w2 = a2*a
      var w3 = a3*a
      var w4 = a4*a
   
      return yp1 + w1*Variation1 + w2*Variation2 + w3*Variation3 + w4*Variation4
   end

   local __demand(__inline)
   task ReconstructEulerFlux(Fluid : region(ispace(int3d), Fluid_columns),
                               cm2 : int3d,
                               cm1 : int3d,
                                 c : int3d,
                               cp1 : int3d,
                               cp2 : int3d,
                               cp3 : int3d,
                               mix : MIX.Mixture) 
   where
      reads(Fluid.{[reconFacePlus], [reconFaceMinus]}),
      reads(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      reads(Fluid.[BCStencil]),
      reads(Fluid.Conserved),
      reads(Fluid.[Primitives]),
      reads(Fluid.[Properties]),
      reads(Fluid.[EulerFlux]),
      writes(Fluid.[FluxC])
   do

      -- Map in the characteristic space
      var K = [mkGetCharacteristicMapping(dir)](Fluid, c, cp1, mix)
      var KInv : LUdec
      KInv.A = K
      KInv = ludcmp(KInv)

      -- Conserved varialbes in the characteristic space
      var QCM2 = lubksb(KInv, Fluid[cm2].Conserved)
      var QCM1 = lubksb(KInv, Fluid[cm1].Conserved)
      var QC   = lubksb(KInv, Fluid[c  ].Conserved)
      var QCP1 = lubksb(KInv, Fluid[cp1].Conserved)
      var QCP2 = lubksb(KInv, Fluid[cp2].Conserved)
      var QCP3 = lubksb(KInv, Fluid[cp3].Conserved)

      -- Euler fluxes in the characteristic space
      var FCM2 = lubksb(KInv, Fluid[cm2].[EulerFlux])
      var FCM1 = lubksb(KInv, Fluid[cm1].[EulerFlux])
      var FC   = lubksb(KInv, Fluid[c  ].[EulerFlux])
      var FCP1 = lubksb(KInv, Fluid[cp1].[EulerFlux])
      var FCP2 = lubksb(KInv, Fluid[cp2].[EulerFlux])
      var FCP3 = lubksb(KInv, Fluid[cp3].[EulerFlux])

      -- Vector of maximum eigenvalues
      var Lam1 = max(max(max(max(max(
                   fabs(Fluid[cm2].velocity[ind]-Fluid[cm2].SoS),
                   fabs(Fluid[cm1].velocity[ind]-Fluid[cm1].SoS)),
                   fabs(Fluid[c  ].velocity[ind]-Fluid[c  ].SoS)),
                   fabs(Fluid[cp1].velocity[ind]-Fluid[cp1].SoS)),
                   fabs(Fluid[cp2].velocity[ind]-Fluid[cp2].SoS)),
                   fabs(Fluid[cp3].velocity[ind]-Fluid[cp3].SoS))

      var Lam2 = max(max(max(max(max(
                   fabs(Fluid[cm2].velocity[ind]),
                   fabs(Fluid[cm1].velocity[ind])),
                   fabs(Fluid[c  ].velocity[ind])),
                   fabs(Fluid[cp1].velocity[ind])),
                   fabs(Fluid[cp2].velocity[ind])),
                   fabs(Fluid[cp3].velocity[ind]))

      var Lam3 = max(max(max(max(max(
                   fabs(Fluid[cm2].velocity[ind]+Fluid[cm2].SoS),
                   fabs(Fluid[cm1].velocity[ind]+Fluid[cm1].SoS)),
                   fabs(Fluid[c  ].velocity[ind]+Fluid[c  ].SoS)),
                   fabs(Fluid[cp1].velocity[ind]+Fluid[cp1].SoS)),
                   fabs(Fluid[cp2].velocity[ind]+Fluid[cp2].SoS)),
                   fabs(Fluid[cp3].velocity[ind]+Fluid[cp3].SoS))

      if Fluid[c].[BCStencil] then
         Lam1 = max(
                  fabs(Fluid[c  ].velocity[ind]-Fluid[c  ].SoS),
                  fabs(Fluid[cp1].velocity[ind]-Fluid[cp1].SoS))

         Lam2 = max(
                  fabs(Fluid[c  ].velocity[ind]),
                  fabs(Fluid[cp1].velocity[ind]))

         Lam3 = max(
                  fabs(Fluid[c  ].velocity[ind]+Fluid[c  ].SoS),
                  fabs(Fluid[cp1].velocity[ind]+Fluid[cp1].SoS))
      end

      var LamMax = [UTIL.mkArrayConstant(nEq, rexpr Lam2 end)]
      LamMax[    0] = Lam1
      LamMax[nEq-1] = Lam3

      -- Combine positive fluxes and negative fluxes
      var EFluxPlusCM2 = GetPlusFlux(FCM2, QCM2, LamMax)
      var EFluxPlusCM1 = GetPlusFlux(FCM1, QCM1, LamMax)
      var EFluxPlusC   = GetPlusFlux(FC  , QC  , LamMax)
      var EFluxPlusCP1 = GetPlusFlux(FCP1, QCP1, LamMax)
      var EFluxPlusCP2 = GetPlusFlux(FCP2, QCP2, LamMax)
      var EFluxPlusCP3 = GetPlusFlux(FCP3, QCP3, LamMax)

      var EFluxMinusCM2 = GetMinusFlux(FCM2, QCM2, LamMax)
      var EFluxMinusCM1 = GetMinusFlux(FCM1, QCM1, LamMax)
      var EFluxMinusC   = GetMinusFlux(FC  , QC  , LamMax)
      var EFluxMinusCP1 = GetMinusFlux(FCP1, QCP1, LamMax)
      var EFluxMinusCP2 = GetMinusFlux(FCP2, QCP2, LamMax)
      var EFluxMinusCP3 = GetMinusFlux(FCP3, QCP3, LamMax)

      var FluxPlus : double[nEq]
      for i=0, nEq do
         FluxPlus[i] = TENOReconstructPlus(EFluxPlusCM2[i],
                                           EFluxPlusCM1[i],
                                           EFluxPlusC  [i],
                                           EFluxPlusCP1[i],
                                           EFluxPlusCP2[i],
                                           EFluxPlusCP3[i],
                                           Fluid[c].[reconFacePlus],
                                           Fluid[c].[TENOCoeffsPlus],
                                           Fluid[c].[BCStencil])
      end

      var FluxMinus : double[nEq]
      for i=0, nEq do
         FluxMinus[i] = TENOReconstructMinus(EFluxMinusCM2[i],
                                             EFluxMinusCM1[i],
                                             EFluxMinusC  [i],
                                             EFluxMinusCP1[i],
                                             EFluxMinusCP2[i],
                                             EFluxMinusCP3[i],
                                             Fluid[c].[reconFaceMinus],
                                             Fluid[c].[TENOCoeffsMinus],
                                             Fluid[c].[BCStencil])
      end

      var Flux : double[nEq]
      for i=0, nEq do
         Flux[i] = (-0.5*(FluxPlus[i] + FluxMinus[i]))
      end

      -- Store correction to recover the first order reconstruction
      var FluxCorr : double[nEq]
      for i=0, nEq do
         FluxCorr[i] = (-0.5*(EFluxPlusC[i] + EFluxMinusCP1[i])) - Flux[i]
      end
      Fluid[c].[FluxC] = [MATH.mkMatMul(nEq)](FluxCorr, K)

      -- Map bask to the conserved variables space
      return [MATH.mkMatMul(nEq)](Flux, K)
   end

   return ReconstructEulerFlux
end)

-------------------------------------------------------------------------------
-- DIFFUSION FLUXES ROUTINES
-------------------------------------------------------------------------------
local function emitAddDiffusionFlux(dir, r, Flux, c, cp1, mix)

   local ind
   local interp
   local deriv
   if (dir == "x") then
      interp = "interpXFace"
      deriv  =  "derivXFace"
   elseif (dir == "y") then
      interp = "interpYFace"
      deriv  =  "derivYFace"
   elseif (dir == "z") then
      interp = "interpZFace"
      deriv  =  "derivZFace"
   else assert(false) end

   local sigma = regentlib.newsymbol(double[3], "sigma")
   local muFace = regentlib.newsymbol(double, "muFace")

   local computeSigma
   if (dir == "x") then
      computeSigma = rquote
         -- Velocity gradient at the face location
         var dUdXFace = [r][c].derivXFace*([r][cp1].velocity[0] - [r][c].velocity[0])
         var dVdXFace = [r][c].derivXFace*([r][cp1].velocity[1] - [r][c].velocity[1])
         var dWdXFace = [r][c].derivXFace*([r][cp1].velocity[2] - [r][c].velocity[2])
         var dUdYFace = [r][c].interpXFace[0]*[r][c].velocityGradientY[0] + [r][c].interpXFace[1]*[r][cp1].velocityGradientY[0]
         var dUdZFace = [r][c].interpXFace[0]*[r][c].velocityGradientZ[0] + [r][c].interpXFace[1]*[r][cp1].velocityGradientZ[0]
         var dVdYFace = [r][c].interpXFace[0]*[r][c].velocityGradientY[1] + [r][c].interpXFace[1]*[r][cp1].velocityGradientY[1]
         var dVdZFace = [r][c].interpXFace[0]*[r][c].velocityGradientZ[1] + [r][c].interpXFace[1]*[r][cp1].velocityGradientZ[1]
         var dWdYFace = [r][c].interpXFace[0]*[r][c].velocityGradientY[2] + [r][c].interpXFace[1]*[r][cp1].velocityGradientY[2]
         var dWdZFace = [r][c].interpXFace[0]*[r][c].velocityGradientZ[2] + [r][c].interpXFace[1]*[r][cp1].velocityGradientZ[2]

         sigma = array(muFace*(4.0*dUdXFace-2.0*dVdYFace-2.0*dWdZFace)/3.0,
                       muFace*(dVdXFace+dUdYFace),
                       muFace*(dWdXFace+dUdZFace))
      end
   elseif (dir == "y") then
      computeSigma = rquote
         -- Velocity gradient at the face location
         var dUdYFace = [r][c].derivYFace*([r][cp1].velocity[0] - [r][c].velocity[0])
         var dVdYFace = [r][c].derivYFace*([r][cp1].velocity[1] - [r][c].velocity[1])
         var dWdYFace = [r][c].derivYFace*([r][cp1].velocity[2] - [r][c].velocity[2])
         var dUdXFace = [r][c].interpYFace[0]*[r][c].velocityGradientX[0] + [r][c].interpYFace[1]*[r][cp1].velocityGradientX[0]
         var dUdZFace = [r][c].interpYFace[0]*[r][c].velocityGradientZ[0] + [r][c].interpYFace[1]*[r][cp1].velocityGradientZ[0]
         var dVdXFace = [r][c].interpYFace[0]*[r][c].velocityGradientX[1] + [r][c].interpYFace[1]*[r][cp1].velocityGradientX[1]
         var dVdZFace = [r][c].interpYFace[0]*[r][c].velocityGradientZ[1] + [r][c].interpYFace[1]*[r][cp1].velocityGradientZ[1]
         var dWdXFace = [r][c].interpYFace[0]*[r][c].velocityGradientX[2] + [r][c].interpYFace[1]*[r][cp1].velocityGradientX[2]
         var dWdZFace = [r][c].interpYFace[0]*[r][c].velocityGradientZ[2] + [r][c].interpYFace[1]*[r][cp1].velocityGradientZ[2]

         sigma = array(muFace*(dUdYFace+dVdXFace),
                       muFace*(4.0*dVdYFace-2.0*dUdXFace-2.0*dWdZFace)/3.0,
                       muFace*(dWdYFace+dVdZFace))
      end
   elseif (dir == "z") then
      computeSigma = rquote
         -- Velocity gradient at the face location
         var dUdZFace = [r][c].derivZFace*([r][cp1].velocity[0] - [r][c].velocity[0])
         var dVdZFace = [r][c].derivZFace*([r][cp1].velocity[1] - [r][c].velocity[1])
         var dWdZFace = [r][c].derivZFace*([r][cp1].velocity[2] - [r][c].velocity[2])
         var dUdXFace = [r][c].interpZFace[0]*[r][c].velocityGradientX[0] + [r][c].interpZFace[1]*[r][cp1].velocityGradientX[0]
         var dUdYFace = [r][c].interpZFace[0]*[r][c].velocityGradientY[0] + [r][c].interpZFace[1]*[r][cp1].velocityGradientY[0]
         var dVdXFace = [r][c].interpZFace[0]*[r][c].velocityGradientX[1] + [r][c].interpZFace[1]*[r][cp1].velocityGradientX[1]
         var dVdYFace = [r][c].interpZFace[0]*[r][c].velocityGradientY[1] + [r][c].interpZFace[1]*[r][cp1].velocityGradientY[1]
         var dWdXFace = [r][c].interpZFace[0]*[r][c].velocityGradientX[2] + [r][c].interpZFace[1]*[r][cp1].velocityGradientX[2]
         var dWdYFace = [r][c].interpZFace[0]*[r][c].velocityGradientY[2] + [r][c].interpZFace[1]*[r][cp1].velocityGradientY[2]

         sigma = array(muFace*(dUdZFace+dWdXFace),
                       muFace*(dVdZFace+dWdYFace),
                       muFace*(4.0*dWdZFace-2.0*dUdXFace-2.0*dVdYFace)/3.0)
      end
   else assert(false) end

   return rquote
      -- Mixture properties at the face location
      var rhoFace  = [r][c].[interp][0]*[r][c].rho    + [r][c].[interp][1]*[r][cp1].rho
      var [muFace] = [r][c].[interp][0]*[r][c].mu     + [r][c].[interp][1]*[r][cp1].mu
      var lamFace  = [r][c].[interp][0]*[r][c].lam    + [r][c].[interp][1]*[r][cp1].lam
      var MixWFace = [r][c].[interp][0]*MIX.GetMolarWeightFromXi([r][c  ].MolarFracs, mix) + 
                     [r][c].[interp][1]*MIX.GetMolarWeightFromXi([r][cp1].MolarFracs, mix)
      var DiFace : double[nSpec]
      for i=0, nSpec do
         DiFace[i] = [r][c].[interp][0]*[r][c].Di[i] + [r][c].[interp][1]*[r][cp1].Di[i]
      end

      -- Primitive and conserved variables at the face location
      var temperatureFace = [r][c].[interp][0]*[r][c  ].temperature +
                            [r][c].[interp][1]*[r][cp1].temperature
      var velocityFace    = MACRO.vv_add(MACRO.vs_mul([r][c  ].velocity, [r][c].[interp][0]),
                                         MACRO.vs_mul([r][cp1].velocity, [r][c].[interp][1]))
      var rhoYiFace : double[nSpec]
      for i=0, nSpec do
         rhoYiFace[i] = [r][c].[interp][0]*[r][c  ].Conserved[i] +
                        [r][c].[interp][1]*[r][cp1].Conserved[i]
      end

      -- Compute shear stress tensor
      var [sigma]; [computeSigma];

      -- Assemble the fluxes
      var usigma = MACRO.dot(velocityFace, sigma)
      var heatFlux = lamFace*[r][c].[deriv]*([r][c].temperature - [r][cp1].temperature)

      -- Species diffusion velocity
      var YiVi : double[nSpec]
      var ViCorr = 0.0
      for i=0, nSpec do
         YiVi[i] = DiFace[i]*[r][c].[deriv]*([r][c].MolarFracs[i] - [r][cp1].MolarFracs[i])*MIX.GetSpeciesMolarWeight(i, mix)/MixWFace
         ViCorr += YiVi[i]
      end

      -- Partial Densities Fluxes
      for i=0, nSpec do
         var rhoYiVi = rhoFace*YiVi[i] - ViCorr*rhoYiFace[i]
         Flux[i] -= rhoYiVi
         heatFlux += rhoYiVi*MIX.GetSpeciesEnthalpy(i, temperatureFace, mix)
      end

      -- Momentum Flux
      for i=0, 3 do
         Flux[irU+i] += sigma[i]
      end

      -- Energy Flux
      Flux[irE] += (usigma-heatFlux)
   end

end


-------------------------------------------------------------------------------
-- FLUXES ROUTINES
-------------------------------------------------------------------------------

Exports.mkGetFlux = terralib.memoize(function(dir)
   local GetFlux

   local Flux
   local FluxC
   local EulerFlux
   local reconFacePlus
   local reconFaceMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local vGrad1
   local vGrad2
   local mk_cm2
   local mk_cm1
   local mk_cp1
   local mk_cp2
   local mk_cp3
   if (dir == "x") then
      Flux  = "FluxX"
      FluxC = "FluxXCorr"
      EulerFlux  = "EulerFluxX"
      reconFacePlus  = "reconXFacePlus"
      reconFaceMinus = "reconXFaceMinus"
      TENOCoeffsPlus  = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv  =  "derivXFace"
      vGrad1 = "velocityGradientY"
      vGrad2 = "velocityGradientZ"
      mk_cm2 = function(c, b) return rexpr (c+{-2, 0, 0}) % b end end
      mk_cm1 = function(c, b) return rexpr (c+{-1, 0, 0}) % b end end
      mk_cp1 = function(c, b) return rexpr (c+{ 1, 0, 0}) % b end end
      mk_cp2 = function(c, b) return rexpr (c+{ 2, 0, 0}) % b end end
      mk_cp3 = function(c, b) return rexpr (c+{ 3, 0, 0}) % b end end
   elseif (dir == "y") then
      Flux  = "FluxY"
      FluxC = "FluxYCorr"
      EulerFlux  = "EulerFluxY"
      reconFacePlus  = "reconYFacePlus"
      reconFaceMinus = "reconYFaceMinus"
      TENOCoeffsPlus  = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv  =  "derivYFace"
      vGrad1 = "velocityGradientX"
      vGrad2 = "velocityGradientZ"
      mk_cm2 = function(c, b) return rexpr (c+{ 0,-2, 0}) % b end end
      mk_cm1 = function(c, b) return rexpr (c+{ 0,-1, 0}) % b end end
      mk_cp1 = function(c, b) return rexpr (c+{ 0, 1, 0}) % b end end
      mk_cp2 = function(c, b) return rexpr (c+{ 0, 2, 0}) % b end end
      mk_cp3 = function(c, b) return rexpr (c+{ 0, 3, 0}) % b end end
   elseif (dir == "z") then
      Flux  = "FluxZ"
      FluxC = "FluxZCorr"
      EulerFlux  = "EulerFluxZ"
      reconFacePlus  = "reconZFacePlus"
      reconFaceMinus = "reconZFaceMinus"
      TENOCoeffsPlus  = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv  =  "derivZFace"
      vGrad1 = "velocityGradientX"
      vGrad2 = "velocityGradientY"
      mk_cm2 = function(c, b) return rexpr (c+{ 0, 0,-2}) % b end end
      mk_cm1 = function(c, b) return rexpr (c+{ 0, 0,-1}) % b end end
      mk_cp1 = function(c, b) return rexpr (c+{ 0, 0, 1}) % b end end
      mk_cp2 = function(c, b) return rexpr (c+{ 0, 0, 2}) % b end end
      mk_cp3 = function(c, b) return rexpr (c+{ 0, 0, 3}) % b end end
   else assert(false) end

   __demand(__parallel, __cuda, __leaf)
   task GetFlux(Fluid    : region(ispace(int3d), Fluid_columns),
                ModCells : region(ispace(int3d), Fluid_columns),
                Fluid_bounds : rect3d,
                mix : MIX.Mixture)
   where
      ModCells <= Fluid,
      reads(Fluid.{[reconFacePlus], [reconFaceMinus]}),
      reads(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      reads(Fluid.[BCStencil]),
      reads(Fluid.[interp]),
      reads(Fluid.[deriv]),
      reads(Fluid.Conserved),
      reads(Fluid.[Primitives]),
      reads(Fluid.[Properties]),
      reads(Fluid.{[vGrad1], [vGrad2]}),
      reads(Fluid.[EulerFlux]),
      writes(Fluid.[Flux]),
      writes(Fluid.[FluxC])
   do
      __demand(__openmp)
      for c in ModCells do
         var cm2 = [mk_cm2(rexpr c end, rexpr Fluid_bounds end)];
         var cm1 = [mk_cm1(rexpr c end, rexpr Fluid_bounds end)];
         var cp1 = [mk_cp1(rexpr c end, rexpr Fluid_bounds end)];
         var cp2 = [mk_cp2(rexpr c end, rexpr Fluid_bounds end)];
         var cp3 = [mk_cp3(rexpr c end, rexpr Fluid_bounds end)];
   
         -- Reconstruct Euler numerical fluxes
         var F = [mkReconstructEulerFlux(dir)](Fluid, cm2, cm1, c, cp1, cp2, cp3, mix);
   
         -- Add diffusion
         [emitAddDiffusionFlux(dir, rexpr Fluid end,
                                    rexpr F     end,
                                    rexpr c     end,
                                    rexpr cp1   end,
                                    rexpr mix   end)];
   
         Fluid[c].[Flux] = F
      end
   end
   return GetFlux
end)

return Exports end

