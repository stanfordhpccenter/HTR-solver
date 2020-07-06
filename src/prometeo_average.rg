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
local sqrt = regentlib.sqrt(double)
local UTIL = require 'util-desugared'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

local struct Averages_columns {
   weight : double;
   -- Primitive variables
   pressure_avg : double;
   pressure_rms : double;
   temperature_avg : double;
   temperature_rms : double;
   MolarFracs_avg : double[nSpec];
   MolarFracs_rms : double[nSpec];
   MassFracs_avg  : double[nSpec];
   MassFracs_rms  : double[nSpec];
   velocity_avg : double[3];
   velocity_rms : double[3];
   velocity_rey : double[3];
   -- Properties
   rho_avg : double;
   rho_rms : double;
   mu_avg  : double;
   lam_avg : double;
   Di_avg  : double[nSpec];
   SoS_avg : double;
   cp_avg  : double;
   Ent_avg : double;
   -- Chemical production rates
   ProductionRates_avg : double[nSpec];
   ProductionRates_rms : double[nSpec];
   HeatReleaseRate_avg : double;
   HeatReleaseRate_rms : double;
   -- Favre averaged primitives
   pressure_favg : double;
   pressure_frms : double;
   temperature_favg : double;
   temperature_frms : double;
   MolarFracs_favg : double[nSpec];
   MolarFracs_frms : double[nSpec];
   MassFracs_favg  : double[nSpec];
   MassFracs_frms  : double[nSpec];
   velocity_favg : double[3];
   velocity_frms : double[3];
   velocity_frey : double[3];
   -- Favre averaged properties
   mu_favg  : double;
   lam_favg : double;
   Di_favg  : double[nSpec];
   SoS_favg : double;
   cp_favg  : double;
   Ent_favg : double;
   -- Kinetic energy budgets (y is the inhomogeneous direction)
   rhoUUv   : double[3];
   Up       : double[3];
   tau      : double[6];
   utau_y   : double[3];
   tauGradU : double[3];
   pGradU   : double[3];
   -- Fluxes
   q        : double[3];
   -- Dimensionless numbers
   Pr       : double;
   Pr_rms   : double;
   Ec       : double;
   Ec_rms   : double;
   Ma       : double;
   Sc       : double[nSpec];
   -- Correlations
   uT_avg   : double[3];
   uT_favg  : double[3];
   uYi_avg  : double[nSpec];
   vYi_avg  : double[nSpec];
   wYi_avg  : double[nSpec];
   uYi_favg : double[nSpec];
   vYi_favg : double[nSpec];
   wYi_favg : double[nSpec];

}

local AveragesVars = terralib.newlist({
   'weight',
   -- Primitive variables
   'pressure_avg',
   'pressure_rms',
   'temperature_avg',
   'temperature_rms',
   'MolarFracs_avg',
   'MolarFracs_rms',
   'MassFracs_avg',
   'MassFracs_rms',
   'velocity_avg',
   'velocity_rms',
   'velocity_rey',
   -- Properties
   'rho_avg',
   'rho_rms',
   'mu_avg',
   'lam_avg',
   'Di_avg',
   'SoS_avg',
   'cp_avg',
   'Ent_avg',
   -- Favre averaged properties
   'mu_favg',
   'lam_favg',
   'Di_favg',
   'SoS_favg',
   'cp_favg',
   'Ent_favg',
   -- Chemical production rates
   'ProductionRates_avg',
   'ProductionRates_rms',
   'HeatReleaseRate_avg',
   'HeatReleaseRate_rms',
   -- Favre averages
   'pressure_favg',
   'pressure_frms',
   'temperature_favg',
   'temperature_frms',
   'MolarFracs_favg',
   'MolarFracs_frms',
   'MassFracs_favg',
   'MassFracs_frms',
   'velocity_favg',
   'velocity_frms',
   'velocity_frey',
   -- Kinetic energy budgets (y is the inhomogeneous direction)
   'rhoUUv',
   'Up',
   'tau',
   'utau_y',
   'tauGradU',
   'pGradU',
   -- Fluxes
   'q',
   -- Dimensionless numbers
   'Pr',
   'Pr_rms',
   'Ec',
   'Ec_rms',
   'Ma',
   'Sc',
   -- Correlations
   'uT_avg',
   'uT_favg',
   'uYi_avg',
   'vYi_avg',
   'wYi_avg',
   'uYi_favg',
   'vYi_favg',
   'wYi_favg'
})

local HDF = (require 'hdf_helper')(int2d, int2d, Averages_columns,
                                                 AveragesVars,
                                                 {},
                                                 {SpeciesNames={nSpec,20}})

Exports.AvgList = {
   XAverages = regentlib.newsymbol(),
   YAverages = regentlib.newsymbol(),
   ZAverages = regentlib.newsymbol(),
   XAverages_copy = regentlib.newsymbol(),
   YAverages_copy = regentlib.newsymbol(),
   ZAverages_copy = regentlib.newsymbol(),
   is_Xrakes = regentlib.newsymbol(),
   is_Yrakes = regentlib.newsymbol(),
   is_Zrakes = regentlib.newsymbol(),
   Xrakes = regentlib.newsymbol(),
   Yrakes = regentlib.newsymbol(),
   Zrakes = regentlib.newsymbol(),
   Xrakes_copy = regentlib.newsymbol(),
   Yrakes_copy = regentlib.newsymbol(),
   Zrakes_copy = regentlib.newsymbol(),

   XAverages_local = regentlib.newsymbol(),
   YAverages_local = regentlib.newsymbol(),
   ZAverages_local = regentlib.newsymbol(),
   is_Xrakes_local = regentlib.newsymbol(),
   is_Yrakes_local = regentlib.newsymbol(),
   is_Zrakes_local = regentlib.newsymbol(),
   p_Xrakes_local = regentlib.newsymbol(),
   p_Yrakes_local = regentlib.newsymbol(),
   p_Zrakes_local = regentlib.newsymbol(),
}

-------------------------------------------------------------------------------
-- AVERAGES ROUTINES
-------------------------------------------------------------------------------
local function mkInitializeAverages(nd)
   local InitializeAverages
   __demand(__inline)
   task InitializeAverages(Averages : region(ispace(nd), Averages_columns))
   where
      writes(Averages)
   do
      fill(Averages.weight, 0.0)
      -- Primitive variables
      fill(Averages.pressure_avg, 0.0)
      fill(Averages.pressure_rms, 0.0)
      fill(Averages.temperature_avg, 0.0)
      fill(Averages.temperature_rms, 0.0)
      fill(Averages.MolarFracs_avg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MolarFracs_rms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MassFracs_avg,  [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MassFracs_rms,  [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.velocity_avg, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_rms, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_rey, array(0.0, 0.0, 0.0))
      -- Properties
      fill(Averages.rho_avg, 0.0)
      fill(Averages.rho_rms, 0.0)
      fill(Averages.mu_avg,  0.0)
      fill(Averages.lam_avg, 0.0)
      fill(Averages.Di_avg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.SoS_avg, 0.0)
      fill(Averages.cp_avg,  0.0)
      fill(Averages.Ent_avg, 0.0)
      -- Chemical production rates
      fill(Averages.ProductionRates_avg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.ProductionRates_rms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.HeatReleaseRate_avg, 0.0)
      fill(Averages.HeatReleaseRate_rms, 0.0)
      -- Favre averaged primitives
      fill(Averages.pressure_favg, 0.0)
      fill(Averages.pressure_frms, 0.0)
      fill(Averages.temperature_favg, 0.0)
      fill(Averages.temperature_frms, 0.0)
      fill(Averages.MolarFracs_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MolarFracs_frms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MassFracs_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MassFracs_frms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.velocity_favg, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_frms, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_frey, array(0.0, 0.0, 0.0))
      -- Favre averaged properties
      fill(Averages.mu_favg,  0.0)
      fill(Averages.lam_favg, 0.0)
      fill(Averages.Di_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.SoS_favg,  0.0)
      fill(Averages.cp_favg,   0.0)
      fill(Averages.Ent_favg,  0.0)
      -- Kinetic energy budgets (y is the inhomogeneous direction)
      fill(Averages.rhoUUv,   array(0.0, 0.0, 0.0))
      fill(Averages.Up,       array(0.0, 0.0, 0.0))
      fill(Averages.tau, [UTIL.mkArrayConstant(6, rexpr 0.0 end)])
      fill(Averages.utau_y,   array(0.0, 0.0, 0.0))
      fill(Averages.tauGradU, array(0.0, 0.0, 0.0))
      fill(Averages.pGradU,   array(0.0, 0.0, 0.0))
      -- Fluxes
      fill(Averages.q, array(0.0, 0.0, 0.0))
      -- Dimensionless numbers
      fill(Averages.Pr,     0.0)
      fill(Averages.Pr_rms, 0.0)
      fill(Averages.Ec,     0.0)
      fill(Averages.Ec_rms, 0.0)
      fill(Averages.Ma,     0.0)
      fill(Averages.Sc, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      -- Correlations
      fill(Averages.uT_avg,  array(0.0, 0.0, 0.0))
      fill(Averages.uT_favg, array(0.0, 0.0, 0.0))
      fill(Averages.uYi_avg,  [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.vYi_avg,  [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.wYi_avg,  [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.uYi_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.vYi_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.wYi_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   end
   return InitializeAverages
end

local function mkAddAverages(dir)
   local AddAverages
   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task AddAverages(Fluid : region(ispace(int3d), Fluid_columns),
                    Averages : region(ispace(int4d), Averages_columns),
                    mix : MIX.Mixture,
                    rake : SCHEMA.Volume,
                    Integrator_deltaTime : double)
   where
      reads(Fluid.cellWidth),
      reads(Fluid.[Primitives]),
      reads(Fluid.[Properties]),
      reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
      reads(Fluid.temperatureGradient),
      reads writes(Averages.[AveragesVars])
   do
      var fromCell = rake.fromCell
      var uptoCell = rake.uptoCell
      __demand(__openmp)
      for c in Fluid do
         if fromCell[0] <= c.x and c.x <= uptoCell[0] and
            fromCell[1] <= c.y and c.y <= uptoCell[1] and
            fromCell[2] <= c.z and c.z <= uptoCell[2] then

            var c_avg = int4d{c.[dir], Averages.bounds.lo.y, Averages.bounds.lo.z, Averages.bounds.lo.w}
            var weight = Fluid[c].cellWidth[0]*Fluid[c].cellWidth[1]*Fluid[c].cellWidth[2]*Integrator_deltaTime
            var rhoWeight = weight*Fluid[c].rho

            Averages[c_avg].weight += weight

            -- Primitive variables
            Averages[c_avg].pressure_avg    += weight*Fluid[c].pressure
            Averages[c_avg].pressure_rms    += weight*Fluid[c].pressure*Fluid[c].pressure
            Averages[c_avg].temperature_avg += weight*Fluid[c].temperature
            Averages[c_avg].temperature_rms += weight*Fluid[c].temperature*Fluid[c].temperature
            Averages[c_avg].MolarFracs_avg += [UTIL.mkArrayConstant(nSpec, weight)]*Fluid[c].MolarFracs
            Averages[c_avg].MolarFracs_rms += [UTIL.mkArrayConstant(nSpec, weight)]*Fluid[c].MolarFracs*Fluid[c].MolarFracs
            Averages[c_avg].velocity_avg += [UTIL.mkArrayConstant(3, weight)]*Fluid[c].velocity
            Averages[c_avg].velocity_rms += [UTIL.mkArrayConstant(3, weight)]*Fluid[c].velocity*Fluid[c].velocity
            Averages[c_avg].velocity_rey += array(Fluid[c].velocity[0]*Fluid[c].velocity[1]*weight,
                                                  Fluid[c].velocity[0]*Fluid[c].velocity[2]*weight,
                                                  Fluid[c].velocity[1]*Fluid[c].velocity[2]*weight)

            -- Favre averaged primitives
            Averages[c_avg].pressure_favg    += rhoWeight*Fluid[c].pressure
            Averages[c_avg].pressure_frms    += rhoWeight*Fluid[c].pressure*Fluid[c].pressure
            Averages[c_avg].temperature_favg += rhoWeight*Fluid[c].temperature
            Averages[c_avg].temperature_frms += rhoWeight*Fluid[c].temperature*Fluid[c].temperature
            Averages[c_avg].MolarFracs_favg  += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*Fluid[c].MolarFracs
            Averages[c_avg].MolarFracs_frms  += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*Fluid[c].MolarFracs*Fluid[c].MolarFracs
            Averages[c_avg].velocity_favg += [UTIL.mkArrayConstant(3, rhoWeight)]*Fluid[c].velocity
            Averages[c_avg].velocity_frms += [UTIL.mkArrayConstant(3, rhoWeight)]*Fluid[c].velocity*Fluid[c].velocity
            Averages[c_avg].velocity_frey += array(Fluid[c].velocity[0]*Fluid[c].velocity[1]*rhoWeight,
                                                   Fluid[c].velocity[0]*Fluid[c].velocity[2]*rhoWeight,
                                                   Fluid[c].velocity[1]*Fluid[c].velocity[2]*rhoWeight)

            -- Kinetic energy budgets (y is the inhomogeneous direction)
            var tau_xx = Fluid[c].mu*(4.0*Fluid[c].velocityGradientX[0] - 2.0*Fluid[c].velocityGradientY[1] - 2.0*Fluid[c].velocityGradientZ[2])/3.0
            var tau_yy = Fluid[c].mu*(4.0*Fluid[c].velocityGradientY[1] - 2.0*Fluid[c].velocityGradientX[0] - 2.0*Fluid[c].velocityGradientZ[2])/3.0
            var tau_zz = Fluid[c].mu*(4.0*Fluid[c].velocityGradientZ[2] - 2.0*Fluid[c].velocityGradientX[0] - 2.0-Fluid[c].velocityGradientY[1])/3.0
            var tau_xy = Fluid[c].mu*(Fluid[c].velocityGradientX[1] + Fluid[c].velocityGradientY[0])
            var tau_yz = Fluid[c].mu*(Fluid[c].velocityGradientY[2] + Fluid[c].velocityGradientZ[1])
            var tau_xz = Fluid[c].mu*(Fluid[c].velocityGradientZ[0] + Fluid[c].velocityGradientX[2])

            Averages[c_avg].rhoUUv += array(Fluid[c].rho*Fluid[c].velocity[0]*Fluid[c].velocity[0]*Fluid[c].velocity[1]*weight,
                                            Fluid[c].rho*Fluid[c].velocity[1]*Fluid[c].velocity[1]*Fluid[c].velocity[1]*weight,
                                            Fluid[c].rho*Fluid[c].velocity[2]*Fluid[c].velocity[2]*Fluid[c].velocity[1]*weight)
            Averages[c_avg].Up += array(Fluid[c].velocity[0]*Fluid[c].pressure*weight,
                                        Fluid[c].velocity[1]*Fluid[c].pressure*weight,
                                        Fluid[c].velocity[2]*Fluid[c].pressure*weight)
            Averages[c_avg].tau += [UTIL.mkArrayConstant(6, weight)]*array(tau_xx, tau_yy, tau_zz, tau_xy, tau_yz, tau_xz)
            Averages[c_avg].utau_y += array(Fluid[c].velocity[0]*tau_xy*weight,
                                            Fluid[c].velocity[1]*tau_yy*weight,
                                            Fluid[c].velocity[2]*tau_yz*weight)
            Averages[c_avg].tauGradU += array((tau_xx*Fluid[c].velocityGradientX[0] + tau_xy*Fluid[c].velocityGradientY[0] + tau_xz*Fluid[c].velocityGradientZ[0])*weight,
                                              (tau_xy*Fluid[c].velocityGradientX[1] + tau_yy*Fluid[c].velocityGradientY[1] + tau_yz*Fluid[c].velocityGradientZ[1])*weight,
                                              (tau_xz*Fluid[c].velocityGradientX[2] + tau_yz*Fluid[c].velocityGradientY[2] + tau_zz*Fluid[c].velocityGradientZ[2])*weight)
            Averages[c_avg].pGradU += array(Fluid[c].pressure*Fluid[c].velocityGradientX[0]*weight,
                                            Fluid[c].pressure*Fluid[c].velocityGradientY[1]*weight,
                                            Fluid[c].pressure*Fluid[c].velocityGradientZ[2]*weight)

            -- Fluxes
            Averages[c_avg].q += array( -Fluid[c].lam*Fluid[c].temperatureGradient[0]*weight, 
                                        -Fluid[c].lam*Fluid[c].temperatureGradient[1]*weight,
                                        -Fluid[c].lam*Fluid[c].temperatureGradient[2]*weight)

         end
      end

      __demand(__openmp)
      for c in Fluid do
         if fromCell[0] <= c.x and c.x <= uptoCell[0] and
            fromCell[1] <= c.y and c.y <= uptoCell[1] and
            fromCell[2] <= c.z and c.z <= uptoCell[2] then

            var c_avg = int4d{c.[dir], Averages.bounds.lo.y, Averages.bounds.lo.z, Averages.bounds.lo.w}
            var weight = Fluid[c].cellWidth[0]*Fluid[c].cellWidth[1]*Fluid[c].cellWidth[2]*Integrator_deltaTime
            var rhoWeight = weight*Fluid[c].rho

            -- Properties
            var MixW = MIX.GetMolarWeightFromXi(Fluid[c].MolarFracs, mix)
            var Yi   = MIX.GetMassFractions(MixW, Fluid[c].MolarFracs, mix)
            var cp   = MIX.GetHeatCapacity(Fluid[c].temperature, Yi, mix)
            var hi : double[nSpec]
            var Ent  = 0.0
            for i=0, nSpec do
               hi[i] = MIX.GetSpeciesEnthalpy(i, Fluid[c].temperature, mix)
               Ent += Yi[i]*hi[i]
            end
            Averages[c_avg].rho_avg += weight*Fluid[c].rho
            Averages[c_avg].rho_rms += weight*Fluid[c].rho*Fluid[c].rho
            Averages[c_avg].mu_avg  += weight*Fluid[c].mu
            Averages[c_avg].lam_avg += weight*Fluid[c].lam
            Averages[c_avg].Di_avg  += [UTIL.mkArrayConstant(nSpec, weight)]*Fluid[c].Di
            Averages[c_avg].SoS_avg += weight*Fluid[c].SoS
            Averages[c_avg].cp_avg  += weight*cp
            Averages[c_avg].Ent_avg += weight*Ent

            -- Mass fractions
            Averages[c_avg].MassFracs_avg  += [UTIL.mkArrayConstant(nSpec, weight)]*Yi
            Averages[c_avg].MassFracs_rms  += [UTIL.mkArrayConstant(nSpec, weight)]*Yi*Yi

            -- Favre averaged properties
            Averages[c_avg].mu_favg  += rhoWeight*Fluid[c].mu
            Averages[c_avg].lam_favg += rhoWeight*Fluid[c].lam
            Averages[c_avg].Di_favg  += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*Fluid[c].Di
            Averages[c_avg].SoS_favg += rhoWeight*Fluid[c].SoS
            Averages[c_avg].cp_favg  += rhoWeight*cp
            Averages[c_avg].Ent_favg += rhoWeight*Ent

            -- Favre averaged mass fractions
            Averages[c_avg].MassFracs_favg += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*Yi
            Averages[c_avg].MassFracs_frms += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*Yi*Yi

            -- Chemical production rates
            var w    = MIX.GetProductionRates(Fluid[c].rho, Fluid[c].pressure, Fluid[c].temperature, Yi, mix)
            var HR = 0.0
            for i=0, nSpec do
               HR -= w[i]*hi[i]
            end
            Averages[c_avg].ProductionRates_avg += [UTIL.mkArrayConstant(nSpec, weight)]*w
            Averages[c_avg].ProductionRates_rms += [UTIL.mkArrayConstant(nSpec, weight)]*w*w
            Averages[c_avg].HeatReleaseRate_avg += weight*HR
            Averages[c_avg].HeatReleaseRate_rms += weight*HR*HR

            -- Dimensionless numbers
            var u2 = MACRO.dot(Fluid[c].velocity, Fluid[c].velocity)
            var Pr = cp*Fluid[c].mu/Fluid[c].lam
            var Ec = u2/(cp*Fluid[c].temperature)
            var nu = Fluid[c].mu/Fluid[c].rho
            var Sc : double[nSpec]
            for i=0, nSpec do
               Sc[i] = nu/Fluid[c].Di[i]
            end
            Averages[c_avg].Pr     += weight*Pr
            Averages[c_avg].Pr_rms += weight*Pr*Pr
            Averages[c_avg].Ec     += weight*Ec
            Averages[c_avg].Ec_rms += weight*Ec*Ec
            Averages[c_avg].Ma     += weight*sqrt(u2)/Fluid[c].SoS
            Averages[c_avg].Sc     += [UTIL.mkArrayConstant(nSpec, weight)]*Sc

            -- Correlations
            var weightU    = weight*Fluid[c].velocity[0]
            var weightV    = weight*Fluid[c].velocity[1]
            var weightW    = weight*Fluid[c].velocity[2]
            var weightRhoU = weight*Fluid[c].velocity[0]*Fluid[c].rho
            var weightRhoV = weight*Fluid[c].velocity[1]*Fluid[c].rho
            var weightRhoW = weight*Fluid[c].velocity[2]*Fluid[c].rho

            Averages[c_avg].uT_avg  += array(   weightU*Fluid[c].temperature,
                                                weightV*Fluid[c].temperature,
                                                weightW*Fluid[c].temperature)
            Averages[c_avg].uT_favg += array(weightRhoU*Fluid[c].temperature,
                                             weightRhoV*Fluid[c].temperature,
                                             weightRhoW*Fluid[c].temperature)

            Averages[c_avg].uYi_avg  += [UTIL.mkArrayConstant(nSpec, weightU)]*Yi
            Averages[c_avg].vYi_avg  += [UTIL.mkArrayConstant(nSpec, weightV)]*Yi
            Averages[c_avg].wYi_avg  += [UTIL.mkArrayConstant(nSpec, weightW)]*Yi
            Averages[c_avg].uYi_favg += [UTIL.mkArrayConstant(nSpec, weightRhoU)]*Yi
            Averages[c_avg].vYi_favg += [UTIL.mkArrayConstant(nSpec, weightRhoV)]*Yi
            Averages[c_avg].wYi_favg += [UTIL.mkArrayConstant(nSpec, weightRhoW)]*Yi

         end
      end
   end
   return AddAverages
end

local __demand(__leaf)
task DummyAverages(Averages : region(ispace(int2d), Averages_columns))
where
   reads writes(Averages)
do
   -- Nothing
   -- It is just to avoid the bug of HDF libraries with parallel reduction
end

local function mkReduceAverages(dir)
   local dir1
   local dir2
   if dir=='x' then
      dir1 = 'y'
      dir2 = 'z'
   elseif dir=='y' then
      dir1 = 'x'
      dir2 = 'z'
   elseif dir=='z' then
      dir1 = 'x'
      dir2 = 'y'
   end
   local ReduceAverages
   __demand(__leaf)
   task ReduceAverages(Averages       : region(ispace(int2d), Averages_columns),
                       Averages_local : region(ispace(int4d), Averages_columns),
                       tiles : ispace(int3d))
   where
      reads(Averages_local),
      reads writes(Averages)
   do
      for t in tiles do
         __demand(__openmp)
         for c in Averages do
            var c_buf = int4d{c.x, c.y, t.[dir1], t.[dir2]}

            Averages[c].weight += Averages_local[c_buf].weight

            -- Primitive variables
            Averages[c].pressure_avg    += Averages_local[c_buf].pressure_avg    
            Averages[c].pressure_rms    += Averages_local[c_buf].pressure_rms    
            Averages[c].temperature_avg += Averages_local[c_buf].temperature_avg 
            Averages[c].temperature_rms += Averages_local[c_buf].temperature_rms 
            Averages[c].MolarFracs_avg  += Averages_local[c_buf].MolarFracs_avg  
            Averages[c].MolarFracs_rms  += Averages_local[c_buf].MolarFracs_rms  
            Averages[c].MassFracs_avg   += Averages_local[c_buf].MassFracs_avg
            Averages[c].MassFracs_rms   += Averages_local[c_buf].MassFracs_rms
            Averages[c].velocity_avg    += Averages_local[c_buf].velocity_avg    
            Averages[c].velocity_rms    += Averages_local[c_buf].velocity_rms    
            Averages[c].velocity_rey    += Averages_local[c_buf].velocity_rey    

            -- Properties
            Averages[c].rho_avg  += Averages_local[c_buf].rho_avg 
            Averages[c].rho_rms  += Averages_local[c_buf].rho_rms 
            Averages[c].mu_avg   += Averages_local[c_buf].mu_avg  
            Averages[c].lam_avg  += Averages_local[c_buf].lam_avg 
            Averages[c].Di_avg   += Averages_local[c_buf].Di_avg  
            Averages[c].SoS_avg  += Averages_local[c_buf].SoS_avg 
            Averages[c].cp_avg   += Averages_local[c_buf].cp_avg  
            Averages[c].Ent_avg  += Averages_local[c_buf].Ent_avg 

            -- Chemical production rates
            Averages[c].ProductionRates_avg += Averages_local[c_buf].ProductionRates_avg
            Averages[c].ProductionRates_rms += Averages_local[c_buf].ProductionRates_rms
            Averages[c].HeatReleaseRate_avg += Averages_local[c_buf].HeatReleaseRate_avg
            Averages[c].HeatReleaseRate_rms += Averages_local[c_buf].HeatReleaseRate_rms

            -- Favre averaged primitives
            Averages[c].pressure_favg    += Averages_local[c_buf].pressure_favg
            Averages[c].pressure_frms    += Averages_local[c_buf].pressure_frms
            Averages[c].temperature_favg += Averages_local[c_buf].temperature_favg
            Averages[c].temperature_frms += Averages_local[c_buf].temperature_frms
            Averages[c].MolarFracs_favg  += Averages_local[c_buf].MolarFracs_favg
            Averages[c].MolarFracs_frms  += Averages_local[c_buf].MolarFracs_frms
            Averages[c].MassFracs_favg   += Averages_local[c_buf].MassFracs_favg
            Averages[c].MassFracs_frms   += Averages_local[c_buf].MassFracs_frms
            Averages[c].velocity_favg    += Averages_local[c_buf].velocity_favg
            Averages[c].velocity_frms    += Averages_local[c_buf].velocity_frms
            Averages[c].velocity_frey    += Averages_local[c_buf].velocity_frey

            -- Favre averaged properties
            Averages[c].mu_favg  += Averages_local[c_buf].mu_favg
            Averages[c].lam_favg += Averages_local[c_buf].lam_favg
            Averages[c].Di_favg  += Averages_local[c_buf].Di_favg
            Averages[c].SoS_favg += Averages_local[c_buf].SoS_favg
            Averages[c].cp_favg  += Averages_local[c_buf].cp_favg
            Averages[c].Ent_favg += Averages_local[c_buf].Ent_favg

            -- Kinetic energy budgets (y is the inhomogeneous direction)
            Averages[c].rhoUUv   += Averages_local[c_buf].rhoUUv
            Averages[c].Up       += Averages_local[c_buf].Up
            Averages[c].tau      += Averages_local[c_buf].tau
            Averages[c].utau_y   += Averages_local[c_buf].utau_y
            Averages[c].tauGradU += Averages_local[c_buf].tauGradU
            Averages[c].pGradU   += Averages_local[c_buf].pGradU

            -- Fluxes
            Averages[c].q += Averages_local[c_buf].q
   
            -- Dimensionless numbers
            Averages[c].Pr     += Averages_local[c_buf].Pr
            Averages[c].Pr_rms += Averages_local[c_buf].Pr_rms
            Averages[c].Ec     += Averages_local[c_buf].Ec
            Averages[c].Ec_rms += Averages_local[c_buf].Ec_rms
            Averages[c].Ma     += Averages_local[c_buf].Ma
            Averages[c].Sc     += Averages_local[c_buf].Sc

            -- Correlations
            Averages[c].uT_avg   += Averages_local[c_buf].uT_avg
            Averages[c].uYi_avg  += Averages_local[c_buf].uYi_avg
            Averages[c].vYi_avg  += Averages_local[c_buf].vYi_avg
            Averages[c].wYi_avg  += Averages_local[c_buf].wYi_avg

            Averages[c].uT_favg  += Averages_local[c_buf].uT_favg
            Averages[c].uYi_favg += Averages_local[c_buf].uYi_favg
            Averages[c].vYi_favg += Averages_local[c_buf].vYi_favg
            Averages[c].wYi_favg += Averages_local[c_buf].wYi_favg

         end
      end
   end
   return ReduceAverages
end

-------------------------------------------------------------------------------
-- EXPORTED ROUTINES
-------------------------------------------------------------------------------
function Exports.DeclSymbols(s, Grid, config, MAPPER)
   return rquote

      var sampleId = config.Mapping.sampleId

      -- Create averages regions
      var is_XAverages = ispace(int2d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                        y = config.IO.YZAverages.length    })

      var is_YAverages = ispace(int2d, {x = config.Grid.yNum + 2*Grid.yBnum,
                                        y = config.IO.XZAverages.length    })

      var is_ZAverages = ispace(int2d, {x = config.Grid.zNum + 2*Grid.zBnum,
                                        y = config.IO.XYAverages.length    })

      var [s.XAverages] = region(is_XAverages, Averages_columns)
      var [s.YAverages] = region(is_YAverages, Averages_columns)
      var [s.ZAverages] = region(is_ZAverages, Averages_columns)
      var [s.XAverages_copy] = region(is_XAverages, Averages_columns)
      var [s.YAverages_copy] = region(is_YAverages, Averages_columns)
      var [s.ZAverages_copy] = region(is_ZAverages, Averages_columns);

      [UTIL.emitRegionTagAttach(s.XAverages,      MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.YAverages,      MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.ZAverages,      MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.XAverages_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.YAverages_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.ZAverages_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

      -- Partitioning averages in rakes
      var [s.is_Xrakes] = ispace(int2d, {1, max(config.IO.YZAverages.length, 1)})
      var [s.is_Yrakes] = ispace(int2d, {1, max(config.IO.XZAverages.length, 1)})
      var [s.is_Zrakes] = ispace(int2d, {1, max(config.IO.XYAverages.length, 1)})

      var [s.Xrakes] = partition(equal, s.XAverages, s.is_Xrakes)
      var [s.Yrakes] = partition(equal, s.YAverages, s.is_Yrakes)
      var [s.Zrakes] = partition(equal, s.ZAverages, s.is_Zrakes)

      var [s.Xrakes_copy] = partition(equal, s.XAverages_copy, s.is_Xrakes)
      var [s.Yrakes_copy] = partition(equal, s.YAverages_copy, s.is_Yrakes)
      var [s.Zrakes_copy] = partition(equal, s.ZAverages_copy, s.is_Zrakes)

--      -- TODO: in the future we might want to partition these also along the rakes
--      var is_XrakesTiles = ispace(int2d, {Grid.NX, config.IO.YZAverages.length})
--      var is_YrakesTiles = ispace(int2d, {Grid.NY, config.IO.XZAverages.length})
--      var is_ZrakesTiles = ispace(int2d, {Grid.NZ, config.IO.XYAverages.length});
--
--      var [s.p_Xrakes] = [UTIL.mkPartitionByTile(int2d, int2d, Averages_columns)]
--                         (s.Xrakes, is_XrakesTiles, int2d{Grid.xBnum,0}, int2d{0,0})
--      var [s.p_Yrakes] = [UTIL.mkPartitionByTile(int2d, int2d, Averages_columns)]
--                         (s.Yrakes, is_YrakesTiles, int2d{Grid.yBnum,0}, int2d{0,0})
--      var [s.p_Zrakes] = [UTIL.mkPartitionByTile(int2d, int2d, Averages_columns)]
--                         (s.Zrakes, is_ZrakesTiles, int2d{Grid.zBnum,0}, int2d{0,0})

      -- Create local buffers for averages regions
      var is_XAverages_local = ispace(int4d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                              y = config.IO.YZAverages.length    ,
                                              z = Grid.NY                        ,
                                              w = Grid.NZ                         })

      var is_YAverages_local = ispace(int4d, {x = config.Grid.yNum + 2*Grid.yBnum,
                                              y = config.IO.XZAverages.length    ,
                                              z = Grid.NX                        ,
                                              w = Grid.NZ                        })

      var is_ZAverages_local = ispace(int4d, {x = config.Grid.zNum + 2*Grid.zBnum,
                                              y = config.IO.XYAverages.length    ,
                                              z = Grid.NX                        ,
                                              w = Grid.NY                        })

      var [s.XAverages_local] = region(is_XAverages_local, Averages_columns)
      var [s.YAverages_local] = region(is_YAverages_local, Averages_columns)
      var [s.ZAverages_local] = region(is_ZAverages_local, Averages_columns);

      [UTIL.emitRegionTagAttach(s.XAverages_local, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.YAverages_local, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.ZAverages_local, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

      -- Partitioning local buffer in rakes
      var [s.is_Xrakes_local] = ispace(int4d, {Grid.NX, max(config.IO.YZAverages.length, 1), Grid.NY, Grid.NZ})
      var [s.is_Yrakes_local] = ispace(int4d, {Grid.NY, max(config.IO.XZAverages.length, 1), Grid.NX, Grid.NZ})
      var [s.is_Zrakes_local] = ispace(int4d, {Grid.NZ, max(config.IO.XYAverages.length, 1), Grid.NX, Grid.NY})

      var [s.p_Xrakes_local] = [UTIL.mkPartitionByTile(int4d, int4d, Averages_columns, "p_Xrakes_local")]
                               (s.XAverages_local, s.is_Xrakes_local, int4d{Grid.xBnum,0,0,0}, int4d{0,0,0,0})
      var [s.p_Yrakes_local] = [UTIL.mkPartitionByTile(int4d, int4d, Averages_columns, "p_Yrakes_local")]
                               (s.YAverages_local, s.is_Yrakes_local, int4d{Grid.yBnum,0,0,0}, int4d{0,0,0,0})
      var [s.p_Zrakes_local] = [UTIL.mkPartitionByTile(int4d, int4d, Averages_columns, "p_Zrakes_local")]
                               (s.ZAverages_local, s.is_Zrakes_local, int4d{Grid.zBnum,0,0,0}, int4d{0,0,0,0})

   end
end

function Exports.InitRakes(s)
   return rquote
      [mkInitializeAverages(int2d)](s.XAverages);
      [mkInitializeAverages(int2d)](s.YAverages);
      [mkInitializeAverages(int2d)](s.ZAverages);
   end
end

function Exports.InitBuffers(s)
   return rquote
      [mkInitializeAverages(int4d)](s.XAverages_local);
      [mkInitializeAverages(int4d)](s.YAverages_local);
      [mkInitializeAverages(int4d)](s.ZAverages_local);
   end
end

function Exports.ReadAverages(s, config)
   return rquote
      if not config.IO.ResetAverages then
         var dirname = [&int8](C.malloc(256))
         if config.IO.YZAverages.length ~= 0 then
            C.snprintf(dirname, 256, '%s/YZAverages', config.Flow.restartDir)
            HDF.load(0, s.is_Xrakes, dirname, s.XAverages, s.XAverages_copy, s.Xrakes, s.Xrakes_copy)
         end
         if config.IO.XZAverages.length ~= 0 then
            C.snprintf(dirname, 256, '%s/XZAverages', config.Flow.restartDir)
            HDF.load(0, s.is_Yrakes, dirname, s.YAverages, s.YAverages_copy, s.Yrakes, s.Yrakes_copy)
         end
         if config.IO.XYAverages.length ~= 0 then
            C.snprintf(dirname, 256, '%s/XYAverages', config.Flow.restartDir)
            HDF.load(0, s.is_Zrakes, dirname, s.ZAverages, s.ZAverages_copy, s.Zrakes, s.Zrakes_copy)
         end
         C.free(dirname)
      end
   end
end

function Exports.AddAverages(s, tiles, p_All, deltaTime, config, Mix)
   return rquote
      for rake=0, config.IO.YZAverages.length do
         __demand(__index_launch)
         for c in tiles do
            [mkAddAverages('x')](p_All[c], s.p_Xrakes_local[int4d{c.x,rake,c.y,c.z}], Mix,
                                 config.IO.YZAverages.values[rake], deltaTime)
         end
      end
      for rake=0, config.IO.XZAverages.length do
         __demand(__index_launch)
         for c in tiles do
            [mkAddAverages('y')](p_All[c], s.p_Yrakes_local[int4d{c.y,rake,c.x,c.z}], Mix,
                                 config.IO.XZAverages.values[rake], deltaTime)
         end
      end
      for rake=0, config.IO.XYAverages.length do
         __demand(__index_launch)
         for c in tiles do
            [mkAddAverages('z')](p_All[c], s.p_Zrakes_local[int4d{c.z,rake,c.x,c.y}], Mix,
                                 config.IO.XYAverages.values[rake], deltaTime)
         end
      end
   end
end

function Exports.WriteAverages(s, tiles, dirname, IO, SpeciesNames, config)
   return rquote
      if config.IO.YZAverages.length ~= 0 then
--         DummyAverages(s.XAverages)
         -- Reduce from reduction buffers
         [mkReduceAverages('x')](s.XAverages, s.XAverages_local, tiles);
         -- Reinitialize reduction buffers
         [mkInitializeAverages(int4d)](s.XAverages_local)

         var Avgdirname = [&int8](C.malloc(256))
         C.snprintf(Avgdirname, 256, '%s/YZAverages', dirname)
         var _1 = IO.createDir(Avgdirname)
         _1 = HDF.dump(               _1, s.is_Xrakes, Avgdirname, s.XAverages, s.XAverages_copy, s.Xrakes, s.Xrakes_copy)
         _1 = HDF.write.SpeciesNames( _1, s.is_Xrakes, Avgdirname, s.XAverages, s.Xrakes, SpeciesNames)
         C.free(Avgdirname)
      end
      if config.IO.XZAverages.length ~= 0 then
--         DummyAverages(s.YAverages)
         -- Reduce from reduction buffers
         [mkReduceAverages('y')](s.YAverages, s.YAverages_local, tiles);
         -- Reinitialize reduction buffers
         [mkInitializeAverages(int4d)](s.YAverages_local)

         var Avgdirname = [&int8](C.malloc(256))
         C.snprintf(Avgdirname, 256, '%s/XZAverages', dirname)
         var _1 = IO.createDir(Avgdirname)
         _1 = HDF.dump(               _1, s.is_Yrakes, Avgdirname, s.YAverages, s.YAverages_copy, s.Yrakes, s.Yrakes_copy)
         _1 = HDF.write.SpeciesNames( _1, s.is_Yrakes, Avgdirname, s.YAverages, s.Yrakes, SpeciesNames)
         C.free(Avgdirname)
      end
      if config.IO.XYAverages.length ~= 0 then
--         DummyAverages(s.ZAverages)
         -- Reduce from reduction buffers
         [mkReduceAverages('z')](s.ZAverages, s.ZAverages_local, tiles);
         -- Reinitialize reduction buffers
         [mkInitializeAverages(int4d)](s.ZAverages_local)

         var Avgdirname = [&int8](C.malloc(256))
         C.snprintf(Avgdirname, 256, '%s/XYAverages', dirname)
         var _1 = IO.createDir(Avgdirname)
         _1 = HDF.dump(               _1, s.is_Zrakes, Avgdirname, s.ZAverages, s.ZAverages_copy, s.Zrakes, s.Zrakes_copy)
         _1 = HDF.write.SpeciesNames( _1, s.is_Zrakes, Avgdirname, s.ZAverages, s.Zrakes, SpeciesNames)
         C.free(Avgdirname)
      end
   end
end

return Exports end

