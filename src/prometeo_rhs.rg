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
local UTIL = require 'util-desugared'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local OP    = require "prometeo_operators"

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

-- Atomic switch
local Fluid = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "Fluid")
local coherence_mode
if ATOMIC then
   coherence_mode = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
else
   coherence_mode = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
end

-------------------------------------------------------------------------------
-- FLUX-DIVERGENCE ROUTINES
-------------------------------------------------------------------------------

Exports.mkUpdateUsingFlux = terralib.memoize(function(dir)
   local UpdateUsingFlux

   local ind
   local Flux
   local stencil
   if (dir == "x") then
      ind = 0
      Flux = "FluxX"
      stencil = function(c, b) return rexpr (c + {-1, 0, 0}) % b end end
   elseif (dir == "y") then
      ind = 1
      Flux = "FluxY"
      stencil = function(c, b) return rexpr (c + { 0,-1, 0}) % b end end
   elseif (dir == "z") then
      ind = 2
      Flux = "FluxZ"
      stencil = function(c, b) return rexpr (c + { 0, 0,-1}) % b end end
   else assert(false) end

   local __demand(__parallel, __cuda, __leaf)
   task UpdateUsingFlux([Fluid],
                        ModCells : region(ispace(int3d), Fluid_columns),
                        Fluid_bounds : rect3d)
   where
      ModCells <= Fluid,
      reads(Fluid.cellWidth),
      reads(Fluid.[Flux]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   do
      __demand(__openmp)
      for c in ModCells do
         var CellWidthInv = 1.0/Fluid[c].cellWidth[ind]
         var cm1 = [stencil(rexpr c end, rexpr Fluid_bounds end)]
         for i=0, nEq do 
            Fluid[c].Conserved_t[i] += (((Fluid[c].[Flux][i] - Fluid[cm1].[Flux][i]))*CellWidthInv)
         end
      end
   end
   return UpdateUsingFlux
end)

-------------------------------------------------------------------------------
-- NSCBC-FLUX ROUTINES
-------------------------------------------------------------------------------
-- Adds NSCBC fluxes to the inflow cells
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdateUsingFluxNSCBCInflow(Fluid    : region(ispace(int3d), Fluid_columns),
                                        Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                        mix : MIX.Mixture)
where
   reads(Fluid.gradX),
   reads(Fluid.{rho, SoS}),
   reads(Fluid.[Primitives]),
   reads(Fluid.Conserved),
   reads(Fluid.velocityGradientX),
   reads(Fluid.{dudtBoundary, dTdtBoundary}),
   reads(Fluid.{pressure, MolarFracs}),
   reads writes(Fluid.Conserved_t)
do
   var BC   = Fluid_BC[0]
   var BCst = Fluid_BC[1]

   __demand(__openmp)
   for c in BC do
      if (BC[c].velocity[0] >= BC[c].SoS) then
         -- Supersonic inlet
         BC[c].Conserved_t = [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)]
      else
         -- Add in the x flux using NSCBC
         var c_int = c + int3d{1, 0, 0}

         -- Thermo-chemical quantities
         var MixW_bnd = MIX.GetMolarWeightFromXi(      BC  [c    ].MolarFracs, mix)
         var Yi_bnd   = MIX.GetMassFractions(MixW_bnd, BC  [c    ].MolarFracs, mix)
         var MixW_int = MIX.GetMolarWeightFromXi(      BCst[c_int].MolarFracs, mix)
         var Yi_int   = MIX.GetMassFractions(MixW_int, BCst[c_int].MolarFracs, mix)
         var Cp_bnd   = MIX.GetHeatCapacity(BC[c].temperature, Yi_bnd, mix)

         -- characteristic velocity leaving the domain
         var lambda_1 = BC[c].velocity[0] - BC[c].SoS
         var lambda   = BC[c].velocity[0]

         -- compute amplitudes of waves
         var dp_dx = [OP.emitderivLeftBCBase(rexpr BC  [c    ].gradX    end, 
                                             rexpr BC  [c    ].pressure end, 
                                             rexpr BCst[c_int].pressure end)]
         var du_dx = BC[c].velocityGradientX[0]
         var dY_dx : double[nSpec]
         for i=0, nSpec do
            dY_dx[i] = [OP.emitderivLeftBCBase(rexpr BC[c].gradX end,
                                               rexpr Yi_bnd[i] end,
                                               rexpr Yi_int[i] end)]
         end

         var L1 = lambda_1*(dp_dx - BC[c].rho*BC[c].SoS*du_dx)
         var LS : double[nSpec]
         for i=0, nSpec do
            LS[i] = lambda*(dY_dx[i])
         end
         var LN = L1 - 2*BC[c].rho*BC[c].SoS*BC[c].dudtBoundary[0]

         var L2 = BC[c].dTdtBoundary/BC[c].temperature
                  +(LN+L1)/(2.0*BC[c].rho*Cp_bnd*BC[c].temperature)
         for i=0, nSpec do
            L2 -= MixW_bnd/MIX.GetSpeciesMolarWeight(i, mix)*LS[i]
         end
         L2 *= -BC[c].rho*BC[c].SoS*BC[c].SoS

         -- update RHS of transport equation for boundary cell
         var d1 = (0.5*(L1+LN)-L2)/(BC[c].SoS*BC[c].SoS)
         var dS = LS

         -- Set RHS to update the density in the ghost inflow cells
         for i=0, nSpec do
            BC[c].Conserved_t[i] -= d1*Yi_bnd[i] + BC[c].rho*dS[i]
         end
      end
   end
end

-- Adds NSCBC fluxes to the outflow cells
function Exports.mkUpdateUsingFluxNSCBCOutflow(dir)
   local UpdateUsingFluxNSCBCOutflow
   local sigma = 0.25

   if dir == "xPos" then

      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task UpdateUsingFluxNSCBCOutflow([Fluid],
                                       Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                       mix : MIX.Mixture,
                                       MaxMach : double,
                                       LengthScale : double,
                                       Pinf : double)
      where
         reads(Fluid.gradX),
         reads(Fluid.{rho, mu, SoS}),
         reads(Fluid.[Primitives]),
         reads(Fluid.Conserved),
         reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
         reads(Fluid.{MolarFracs, pressure, velocity}),
         reads(Fluid.{rho, mu}),
         reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
         reads writes(Fluid.Conserved_t),
         [coherence_mode]
      do
         var BC   = Fluid_BC[0]
         var BCst = Fluid_BC[1]

         __demand(__openmp)
         for c in BC do
            -- Add in the x fluxes using NSCBC for outflow
            var c_int = c + int3d{-1, 0, 0}
      
            -- Thermo-chemical quantities
            var MixW_bnd = MIX.GetMolarWeightFromXi(      BC  [c    ].MolarFracs, mix)
            var Yi_bnd   = MIX.GetMassFractions(MixW_bnd, BC  [c    ].MolarFracs, mix)
            var MixW_int = MIX.GetMolarWeightFromXi(      BCst[c_int].MolarFracs, mix)
            var Yi_int   = MIX.GetMassFractions(MixW_int, BCst[c_int].MolarFracs, mix)
            var Cp_bnd   = MIX.GetHeatCapacity(BC[c].temperature, Yi_bnd, mix)
      
            var drho_dx = [OP.emitderivRightBCBase(rexpr BC  [c    ].gradX end, 
                                                   rexpr BCst[c_int].rho   end, 
                                                   rexpr BC  [c    ].rho   end)]
            var dp_dx   = [OP.emitderivRightBCBase(rexpr BC  [c    ].gradX    end, 
                                                   rexpr BCst[c_int].pressure end, 
                                                   rexpr BC  [c    ].pressure end)]
            var du_dx   = BC[c].velocityGradientX[0]
            var dv_dx   = BC[c].velocityGradientX[1]
            var dw_dx   = BC[c].velocityGradientX[2]
            var dY_dx : double[nSpec]
            for i=0, nSpec do
               dY_dx[i] = [OP.emitderivRightBCBase(rexpr BC[c].gradX end,
                                                   rexpr Yi_int[i] end,
                                                   rexpr Yi_bnd[i] end)]
            end
      
            var lambda_1 = BC[c].velocity[0] - BC[c].SoS
            var lambda   = BC[c].velocity[0]
            var lambda_N = BC[c].velocity[0] + BC[c].SoS
      
            var L1 : double
            if lambda_1 > 0 then
               -- We are supersonic
               L1 = lambda_1*(dp_dx - BC[c].rho*BC[c].SoS*du_dx)
            else
               -- It is either a subsonic or partially subsonic outlet
               var K = sigma*(1.0-MaxMach*MaxMach)*BC[c].SoS/LengthScale
               if MaxMach > 0.99 then
                  -- This means that MaxMach > 1.0
                  -- Use the local Mach number for partialy supersonic outflows
                  K = sigma*(BC[c].SoS-(BC[c].velocity[0]*BC[c].velocity[0])/BC[c].SoS)/LengthScale
               end
               L1 = K*(BC[c].pressure - Pinf)
            end
      
            var L2 = lambda*(dp_dx - BC[c].SoS*BC[c].SoS*drho_dx)
            var L3 = lambda*(dv_dx)
            var L4 = lambda*(dw_dx)
            var LS : double[nSpec]
            for i=0, nSpec do
               LS[i] = lambda*(dY_dx[i])
            end
            var LN = lambda_N*(dp_dx + BC[c].rho*BC[c].SoS*du_dx)
      
            var d1 = (0.5*(LN + L1) - L2)/(BC[c].SoS*BC[c].SoS)
            var d2 = (LN - L1)/(2.0*BC[c].rho*BC[c].SoS)
            var d3 = L3
            var d4 = L4
            var dS = LS
            var dN = L2/(BC[c].SoS*BC[c].SoS)
      
            var tau11_bnd = BC[c].mu*(4.0*BC[c].velocityGradientX[0] - 2.0*BC[c].velocityGradientY[1] - 2.0*BC[c].velocityGradientZ[2])/3.0
            var tau21_bnd = BC[c].mu*(BC[c].velocityGradientX[1] + BC[c].velocityGradientY[0])
            var tau31_bnd = BC[c].mu*(BC[c].velocityGradientX[2] + BC[c].velocityGradientZ[0])
      
            var tau11_int = BCst[c_int].mu*(4.0*BCst[c_int].velocityGradientX[0] - 2.0*BCst[c_int].velocityGradientY[1] - 2.0*BCst[c_int].velocityGradientZ[2])/3.0
      
            -- Diffusion in momentum equations
            var dtau11_dx = [OP.emitderivRightBCBase(rexpr BC[c].gradX end, rexpr tau11_int end, rexpr tau11_bnd end)]
      
            -- Source in energy equation
            var energy_term_x = [OP.emitderivRightBCBase(rexpr BC[c].gradX end, 
                                                         rexpr BCst[c_int].velocity[0]*tau11_int end,
                                                         rexpr BC  [c    ].velocity[0]*tau11_bnd end)]
                                + BC[c].velocityGradientX[1]*tau21_bnd
                                + BC[c].velocityGradientX[2]*tau31_bnd
      
            -- Update the RHS of conservation equations with x fluxes
            for i=0, nSpec do
               BC[c].Conserved_t[i] -= d1*Yi_bnd[i] + BC[c].rho*dS[i]
            end
            BC[c].Conserved_t[irU+0] -= BC[c].velocity[0]*d1 + BC[c].rho*d2 - dtau11_dx
            BC[c].Conserved_t[irU+1] -= BC[c].velocity[1]*d1 + BC[c].rho*d3
            BC[c].Conserved_t[irU+2] -= BC[c].velocity[2]*d1 + BC[c].rho*d4
            BC[c].Conserved_t[irE] -= (BC[c].Conserved[irE] + BC[c].pressure)*d1/BC[c].rho
                                     + BC[c].Conserved[irU+0]*d2
                                     + BC[c].Conserved[irU+1]*d3
                                     + BC[c].Conserved[irU+2]*d4
                                     + Cp_bnd*BC[c].temperature*dN
                                     - energy_term_x
            for i=0, nSpec do
               var dedYi = MIX.GetSpecificInternalEnergy(i, BC[c].temperature, mix)
               BC[c].Conserved_t[irE] -= BC[c].rho*dedYi*dS[i]
            end
         end
      end

   elseif dir == "yNeg" then

      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task UpdateUsingFluxNSCBCOutflow([Fluid],
                                       Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                       mix : MIX.Mixture,
                                       MaxMach : double,
                                       LengthScale : double,
                                       Pinf : double)
      where
         reads(Fluid.gradY),
         reads(Fluid.{rho, mu, SoS}),
         reads(Fluid.[Primitives]),
         reads(Fluid.Conserved),
         reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
         reads(Fluid.{MolarFracs, pressure, velocity}),
         reads(Fluid.{rho, mu}),
         reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
         reads writes(Fluid.Conserved_t),
         [coherence_mode]
      do
         var BC   = Fluid_BC[0]
         var BCst = Fluid_BC[1]

         __demand(__openmp)
         for c in BC do
            var c_int = c + int3d{ 0, 1, 0}

            -- Thermo-chemical quantities
            var MixW_bnd = MIX.GetMolarWeightFromXi(      BC  [c    ].MolarFracs, mix)
            var Yi_bnd   = MIX.GetMassFractions(MixW_bnd, BC  [c    ].MolarFracs, mix)
            var MixW_int = MIX.GetMolarWeightFromXi(      BCst[c_int].MolarFracs, mix)
            var Yi_int   = MIX.GetMassFractions(MixW_int, BCst[c_int].MolarFracs, mix)
            var Cp_bnd   = MIX.GetHeatCapacity(BC[c].temperature, Yi_bnd, mix)

            var drho_dy = [OP.emitderivLeftBCBase(rexpr BC  [c    ].gradY end, 
                                                  rexpr BC  [c    ].rho   end, 
                                                  rexpr BCst[c_int].rho   end)]
            var dp_dy   = [OP.emitderivLeftBCBase(rexpr BC  [c    ].gradY    end, 
                                                  rexpr BC  [c    ].pressure end, 
                                                  rexpr BCst[c_int].pressure end)]
            var du_dy   = BC[c].velocityGradientY[0]
            var dv_dy   = BC[c].velocityGradientY[1]
            var dw_dy   = BC[c].velocityGradientY[2]
            var dY_dy : double[nSpec]
            for i=0, nSpec do
               dY_dy[i] = [OP.emitderivLeftBCBase(rexpr BC[c].gradY end,
                                                  rexpr Yi_bnd[i] end,
                                                  rexpr Yi_int[i] end)]
            end

            var lambda_1 = BC[c].velocity[1] - BC[c].SoS
            var lambda   = BC[c].velocity[1]
            var lambda_N = BC[c].velocity[1] + BC[c].SoS


            var L1 = lambda_1*(dp_dy - BC[c].rho*BC[c].SoS*dv_dy)
            var L2 = lambda*(du_dy)
            var L3 = lambda*(dp_dy - BC[c].SoS*BC[c].SoS*drho_dy)
            var L4 = lambda*(dw_dy)
            var LS : double[nSpec]
            for i=0, nSpec do
               LS[i] = lambda*(dY_dy[i])
            end

            var LN : double
            if lambda_N > 0 then
               -- We are supersonic
               LN = lambda_N*(dp_dy + BC[c].rho*BC[c].SoS*dv_dy)
            else
               -- It is either a subsonic or partially subsonic outlet
               var K = sigma*(1.0-MaxMach*MaxMach)*BC[c].SoS/LengthScale
               if MaxMach > 0.99 then
                  -- This means that MaxMach[0] > 1.0
                  -- Use the local Mach number for partialy supersonic outflows
                  K = sigma*(BC[c].SoS-(BC[c].velocity[1]*BC[c].velocity[1])/BC[c].SoS)/LengthScale
               end
               LN = K*(BC[c].pressure - Pinf)
            end

            var d1 = (0.5*(LN + L1) - L3)/(BC[c].SoS*BC[c].SoS)
            var d2 = L2
            var d3 = (LN - L1)/(2.0*BC[c].rho*BC[c].SoS)
            var d4 = L4
            var dS = LS
            var dN = L3/(BC[c].SoS*BC[c].SoS)

            var tau12_bnd = BC[c].mu*(BC[c].velocityGradientY[0] + BC[c].velocityGradientX[1])
            var tau22_bnd = BC[c].mu*(4.0*BC[c].velocityGradientY[1] - 2.0*BC[c].velocityGradientX[0] - 2.0*BC[c].velocityGradientZ[2])/3.0
            var tau32_bnd = BC[c].mu*(BC[c].velocityGradientY[2] + BC[c].velocityGradientZ[1])

            var tau22_int = BCst[c_int].mu*(4.0*BCst[c_int].velocityGradientY[1] - 2.0*BCst[c_int].velocityGradientX[0] - 2.0*BCst[c_int].velocityGradientZ[2])/3.0

            -- Diffusion in momentum equations
            var dtau22_dy = [OP.emitderivLeftBCBase(rexpr BC[c].gradY end,
                                                    rexpr tau22_bnd   end, 
                                                    rexpr tau22_int   end)]

            -- Source in energy equation
            var energy_term_y = [OP.emitderivLeftBCBase(rexpr BC[c].gradY end, 
                                                        rexpr BC  [c    ].velocity[1]*tau22_bnd end,
                                                        rexpr BCst[c_int].velocity[1]*tau22_int end)]
                                + BC[c].velocityGradientY[0]*tau12_bnd 
                                + BC[c].velocityGradientY[2]*tau32_bnd

            -- Update the RHS of conservation equations with x fluxes
            for i=0, nSpec do
               BC[c].Conserved_t[i] -= d1*Yi_bnd[i] + BC[c].rho*dS[i]
            end
            BC[c].Conserved_t[irU+0] -= BC[c].velocity[0]*d1 + BC[c].rho*d2
            BC[c].Conserved_t[irU+1] -= BC[c].velocity[1]*d1 + BC[c].rho*d3 - dtau22_dy
            BC[c].Conserved_t[irU+2] -= BC[c].velocity[2]*d1 + BC[c].rho*d4
            BC[c].Conserved_t[irE] -= (BC[c].Conserved[irE] + BC[c].pressure)*d1/BC[c].rho
                                     + BC[c].Conserved[irU+0]*d2
                                     + BC[c].Conserved[irU+1]*d3
                                     + BC[c].Conserved[irU+2]*d4
                                     + Cp_bnd*BC[c].temperature*dN
                                     - energy_term_y
            for i=0, nSpec do
               var dedYi = MIX.GetSpecificInternalEnergy(i, BC[c].temperature, mix)
               BC[c].Conserved_t[irE] -= BC[c].rho*dedYi*dS[i]
            end
         end
      end

   elseif dir == "yPos" then

      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task UpdateUsingFluxNSCBCOutflow([Fluid],
                                       Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                       mix : MIX.Mixture,
                                       MaxMach : double,
                                       LengthScale : double,
                                       Pinf : double)
      where
         reads(Fluid.gradY),
         reads(Fluid.{rho, mu, SoS}),
         reads(Fluid.[Primitives]),
         reads(Fluid.Conserved),
         reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
         reads(Fluid.{MolarFracs, pressure, velocity}),
         reads(Fluid.{rho, mu}),
         reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
         reads writes(Fluid.Conserved_t),
         [coherence_mode]
      do
         var BC   = Fluid_BC[0]
         var BCst = Fluid_BC[1]

         __demand(__openmp)
         for c in BC do
            var c_int = c + int3d{ 0,-1, 0}

            -- Thermo-chemical quantities
            var MixW_bnd = MIX.GetMolarWeightFromXi(      BC  [c    ].MolarFracs, mix)
            var Yi_bnd   = MIX.GetMassFractions(MixW_bnd, BC  [c    ].MolarFracs, mix)
            var MixW_int = MIX.GetMolarWeightFromXi(      BCst[c_int].MolarFracs, mix)
            var Yi_int   = MIX.GetMassFractions(MixW_int, BCst[c_int].MolarFracs, mix)
            var Cp_bnd   = MIX.GetHeatCapacity(BC[c].temperature, Yi_bnd, mix)

            var drho_dy = [OP.emitderivRightBCBase(rexpr BC  [c    ].gradY end, 
                                                   rexpr BCst[c_int].rho   end, 
                                                   rexpr BC  [c    ].rho   end)]
            var dp_dy   = [OP.emitderivRightBCBase(rexpr BC  [c    ].gradY    end, 
                                                   rexpr BCst[c_int].pressure end, 
                                                   rexpr BC  [c    ].pressure end)]
            var du_dy   = BC[c].velocityGradientY[0]
            var dv_dy   = BC[c].velocityGradientY[1]
            var dw_dy   = BC[c].velocityGradientY[2]
            var dY_dy : double[nSpec]
            for i=0, nSpec do
               dY_dy[i] = [OP.emitderivRightBCBase(rexpr BC[c].gradY end,
                                                   rexpr Yi_int[i] end,
                                                   rexpr Yi_bnd[i] end)]
            end

            var lambda_1 = BC[c].velocity[1] - BC[c].SoS
            var lambda   = BC[c].velocity[1]
            var lambda_N = BC[c].velocity[1] + BC[c].SoS

            var L1 : double
            if lambda_1 > 0 then
               -- We are supersonic
               L1 = lambda_1*(dp_dy - BC[c].rho*BC[c].SoS*dv_dy)
            else
               -- It is either a subsonic or partially subsonic outlet
               var K = sigma*(1.0-MaxMach*MaxMach)*BC[c].SoS/LengthScale
               if MaxMach > 0.99 then
                  -- This means that MaxMach[0] > 1.0
                  -- Use the local Mach number for partialy supersonic outflows
                  K = sigma*(BC[c].SoS-(BC[c].velocity[1]*BC[c].velocity[1])/BC[c].SoS)/LengthScale
               end
               L1 = K*(BC[c].pressure - Pinf)
            end

            var L2 = lambda*(du_dy)
            var L3 = lambda*(dp_dy - BC[c].SoS*BC[c].SoS*drho_dy)
            var L4 = lambda*(dw_dy)
            var LS : double[nSpec]
            for i=0, nSpec do
               LS[i] = lambda*(dY_dy[i])
            end
            var LN = lambda_N*(dp_dy + BC[c].rho*BC[c].SoS*dv_dy)

            var d1 = (0.5*(LN + L1) - L3)/(BC[c].SoS*BC[c].SoS)
            var d2 = L2
            var d3 = (LN - L1)/(2.0*BC[c].rho*BC[c].SoS)
            var d4 = L4
            var dS = LS
            var dN = L3/(BC[c].SoS*BC[c].SoS)

            var tau12_bnd = BC[c].mu*(BC[c].velocityGradientY[0] + BC[c].velocityGradientX[1])
            var tau22_bnd = BC[c].mu*(4.0*BC[c].velocityGradientY[1] - 2.0*BC[c].velocityGradientX[0] - 2.0*BC[c].velocityGradientZ[2])/3.0
            var tau32_bnd = BC[c].mu*(BC[c].velocityGradientY[2] + BC[c].velocityGradientZ[1])

            var tau22_int = BCst[c_int].mu*(4.0*BCst[c_int].velocityGradientY[1] - 2.0*BCst[c_int].velocityGradientX[0] - 2.0*BCst[c_int].velocityGradientZ[2])/3.0

            -- Diffusion in momentum equations
            var dtau22_dy = [OP.emitderivRightBCBase(rexpr BC[c].gradY end, rexpr tau22_int end, rexpr tau22_bnd end)]

            -- Source in energy equation
            var energy_term_y = [OP.emitderivRightBCBase(rexpr BC[c].gradY end, 
                                                         rexpr BCst[c_int].velocity[1]*tau22_int end,
                                                         rexpr BC  [c    ].velocity[1]*tau22_bnd end)]
                                + BC[c].velocityGradientY[0]*tau12_bnd 
                                + BC[c].velocityGradientY[2]*tau32_bnd

            -- Update the RHS of conservation equations with x fluxes
            for i=0, nSpec do
               BC[c].Conserved_t[i] -= d1*Yi_bnd[i] + BC[c].rho*dS[i]
            end
            BC[c].Conserved_t[irU+0] -= BC[c].velocity[0]*d1 + BC[c].rho*d2
            BC[c].Conserved_t[irU+1] -= BC[c].velocity[1]*d1 + BC[c].rho*d3 - dtau22_dy
            BC[c].Conserved_t[irU+2] -= BC[c].velocity[2]*d1 + BC[c].rho*d4
            BC[c].Conserved_t[irE] -= (BC[c].Conserved[irE] + BC[c].pressure)*d1/BC[c].rho
                                     + BC[c].Conserved[irU+0]*d2
                                     + BC[c].Conserved[irU+1]*d3
                                     + BC[c].Conserved[irU+2]*d4
                                     + Cp_bnd*BC[c].temperature*dN
                                     - energy_term_y
            for i=0, nSpec do
               var dedYi = MIX.GetSpecificInternalEnergy(i, BC[c].temperature, mix)
               BC[c].Conserved_t[irE] -= BC[c].rho*dedYi*dS[i]
            end
         end
      end

   end
   return UpdateUsingFluxNSCBCOutflow
end

-------------------------------------------------------------------------------
-- FORCING ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.AddBodyForces([Fluid],
                           ModCells : region(ispace(int3d), Fluid_columns),
                           Flow_bodyForce : double[3])
where
   reads(Fluid.{rho, velocity}),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do
   __demand(__openmp)
   for c in ModCells do
      for i=0, 3 do 
         Fluid[c].Conserved_t[irU+i] += Fluid[c].rho*Flow_bodyForce[i]
      end
      Fluid[c].Conserved_t[irE] += Fluid[c].rho*MACRO.dot(Flow_bodyForce, Fluid[c].velocity)
   end
end

-------------------------------------------------------------------------------
-- CORRECTION ROUTINES
-------------------------------------------------------------------------------

local __demand(__inline)
task isValid(Conserved : double[nEq],
             mix : MIX.Mixture)
   var valid = [UTIL.mkArrayConstant(nSpec+1, true)];
   var rhoYi : double[nSpec]
   for i=0, nSpec do
      if Conserved[i] < 0.0 then valid[i] = false end
      rhoYi[i] = Conserved[i]
   end
   var rho = MIX.GetRhoFromRhoYi(rhoYi)
   var Yi = MIX.GetYi(rho, rhoYi)
   var rhoInv = 1.0/rho
   var velocity = array(Conserved[irU+0]*rhoInv,
                        Conserved[irU+1]*rhoInv,
                        Conserved[irU+2]*rhoInv)
   var kineticEnergy = (0.5*MACRO.dot(velocity, velocity))
   var InternalEnergy = Conserved[irE]*rhoInv - kineticEnergy
   valid[nSpec] = MIX.isValidInternalEnergy(InternalEnergy, Yi, mix)
   return valid
end

Exports.mkCorrectUsingFlux = terralib.memoize(function(dir)
   local CorrectUsingFlux

   local ind
   local Flux
   local mk_cm1
   local mk_cp1
   if (dir == "x") then
      ind = 0
      Flux = "FluxXCorr"
      mk_cm1 = function(c, b) return rexpr (c + {-1, 0, 0}) % b end end
      mk_cp1 = function(c, b) return rexpr (c + { 1, 0, 0}) % b end end
   elseif (dir == "y") then
      ind = 1
      Flux = "FluxYCorr"
      mk_cm1 = function(c, b) return rexpr (c + { 0,-1, 0}) % b end end
      mk_cp1 = function(c, b) return rexpr (c + { 0, 1, 0}) % b end end
   elseif (dir == "z") then
      ind = 2
      Flux = "FluxZCorr"
      mk_cm1 = function(c, b) return rexpr (c + { 0, 0,-1}) % b end end
      mk_cp1 = function(c, b) return rexpr (c + { 0, 0, 1}) % b end end
   else assert(false) end

   __demand(__parallel, __cuda, __leaf)
   task CorrectUsingFlux([Fluid],
                         ModCells : region(ispace(int3d), Fluid_columns),
                         Fluid_bounds : rect3d,
                         mix : MIX.Mixture)
   where
      ModCells <= Fluid,
      reads(Fluid.cellWidth),
      reads(Fluid.Conserved_hat),
      reads(Fluid.[Flux]),
      reads writes(Fluid.Conserved_t),
      [coherence_mode]
   do
      __demand(__openmp)
      for c in ModCells do
         -- Stencil
         var cm1 = [mk_cm1(rexpr c end, rexpr Fluid_bounds end)];
         var cp1 = [mk_cp1(rexpr c end, rexpr Fluid_bounds end)];

         -- Do derivatives need to be corrected?
         var correctC   = false
         var correctCM1 = false

         var valid_cm1 = isValid(Fluid[cm1].Conserved_hat, [mix])
         var valid_c   = isValid(Fluid[c  ].Conserved_hat, [mix])
         var valid_cp1 = isValid(Fluid[cp1].Conserved_hat, [mix])

         for i=0, nSpec+1 do
            if not (valid_cp1[i] and valid_c[i]) then correctC   = true end
            if not (valid_cm1[i] and valid_c[i]) then correctCM1 = true end
         end

         -- Correct using Flux on i-1 face
         if correctCM1 then
            -- Correct time derivatives using fluxes between cm1 and c
            var CellWidthInv = 1.0/Fluid[c].cellWidth[ind]
            if not (valid_cm1[nSpec] and valid_c[nSpec]) then
               -- Temeperature is going south
               -- Correct everything
               for i=0, nEq do
                  Fluid[c].Conserved_t[i] -= Fluid[cm1].[Flux][i]*CellWidthInv
               end
            else
               for i=0, nSpec do
                  if not (valid_cm1[i] and valid_c[i]) then
                     -- Correct single species flux
                     Fluid[c].Conserved_t[i] -= Fluid[cm1].[Flux][i]*CellWidthInv
                  end
               end
            end
         end

         -- Correct using Flux on i face
         if correctC  then
            -- Correct time derivatives using fluxes between c and cp1
            var CellWidthInv = 1.0/Fluid[c].cellWidth[ind]
            if not (valid_cp1[nSpec] and valid_c[nSpec]) then
               -- Temeperature is going south
               -- Correct everything
               for i=0, nEq do
                  Fluid[c].Conserved_t[i] += Fluid[c].[Flux][i]*CellWidthInv
               end
            else
               for i=0, nSpec do
                  if not (valid_cp1[i] and valid_c[i]) then
                     -- Correct single species flux
                     Fluid[c].Conserved_t[i] += Fluid[c].[Flux][i]*CellWidthInv
                  end
               end
            end
         end
      end
   end
   return CorrectUsingFlux
end)

return Exports end

