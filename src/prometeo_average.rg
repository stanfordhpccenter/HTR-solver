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

return function(SCHEMA, MIX, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local sqrt = regentlib.sqrt(double)
local UTIL = require 'util-desugared'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local format = require "std/format"

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
   -- Grid point
   centerCoordinates : double[3];
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
   -- Grid point
   'centerCoordinates',
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

local HDF_RAKES = (require 'hdf_helper')(int2d, int2d, Averages_columns,
                                                       AveragesVars,
                                                       {},
                                                       {SpeciesNames={nSpec,20}})

local HDF_PLANES = (require 'hdf_helper')(int3d, int3d, Averages_columns,
                                                        AveragesVars,
                                                        {},
                                                        {SpeciesNames={nSpec,20}})

Exports.AvgList = {
   -- 2D averages
   YZAverages = regentlib.newsymbol(),
   XZAverages = regentlib.newsymbol(),
   XYAverages = regentlib.newsymbol(),
   YZAverages_copy = regentlib.newsymbol(),
   XZAverages_copy = regentlib.newsymbol(),
   XYAverages_copy = regentlib.newsymbol(),
   -- partitions for IO
   is_Xrakes = regentlib.newsymbol(),
   is_Yrakes = regentlib.newsymbol(),
   is_Zrakes = regentlib.newsymbol(),
   Xrakes = regentlib.newsymbol(),
   Yrakes = regentlib.newsymbol(),
   Zrakes = regentlib.newsymbol(),
   Xrakes_copy = regentlib.newsymbol(),
   Yrakes_copy = regentlib.newsymbol(),
   Zrakes_copy = regentlib.newsymbol(),
   -- partitions for average collection
   p_Xrakes = regentlib.newsymbol(),
   p_Yrakes = regentlib.newsymbol(),
   p_Zrakes = regentlib.newsymbol(),
   -- considered partitions of the Fluid domain
   p_Fluid_YZAvg = regentlib.newsymbol("p_Fluid_YZAvg"),
   p_Fluid_XZAvg = regentlib.newsymbol("p_Fluid_XZAvg"),
   p_Fluid_XYAvg = regentlib.newsymbol("p_Fluid_XYAvg"),
   -- tiles of Fluid where the average kernels will be launched
   YZAvg_tiles = regentlib.newsymbol(),
   XZAvg_tiles = regentlib.newsymbol(),
   XYAvg_tiles = regentlib.newsymbol(),

   -- 1D averages
   XAverages = regentlib.newsymbol(),
   YAverages = regentlib.newsymbol(),
   ZAverages = regentlib.newsymbol(),
   XAverages_copy = regentlib.newsymbol(),
   YAverages_copy = regentlib.newsymbol(),
   ZAverages_copy = regentlib.newsymbol(),
   -- partitions for average collection
   YZplanes = regentlib.newsymbol(),
   XZplanes = regentlib.newsymbol(),
   XYplanes = regentlib.newsymbol(),
   -- partitions for IO
   is_IO_YZplanes = regentlib.newsymbol(),
   is_IO_XZplanes = regentlib.newsymbol(),
   is_IO_XYplanes = regentlib.newsymbol(),
   IO_YZplanes = regentlib.newsymbol(),
   IO_XZplanes = regentlib.newsymbol(),
   IO_XYplanes = regentlib.newsymbol(),
   IO_YZplanes_copy = regentlib.newsymbol(),
   IO_XZplanes_copy = regentlib.newsymbol(),
   IO_XYplanes_copy = regentlib.newsymbol(),
   -- considered partitions of the Fluid domain
   p_Fluid_XAvg = regentlib.newsymbol("p_Fluid_XAvg"),
   p_Fluid_YAvg = regentlib.newsymbol("p_Fluid_YAvg"),
   p_Fluid_ZAvg = regentlib.newsymbol("p_Fluid_ZAvg"),
   -- tiles of Fluid where the average kernels will be launched
   XAvg_tiles = regentlib.newsymbol(),
   YAvg_tiles = regentlib.newsymbol(),
   ZAvg_tiles = regentlib.newsymbol()
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
      -- Grid point
      fill(Averages.centerCoordinates, array(0.0, 0.0, 0.0))
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

local function emitAddAvg1(r, c, avg, c_avg, Integrator_deltaTime) return rquote

   var weight = r[c].cellWidth[0]*r[c].cellWidth[1]*r[c].cellWidth[2]*Integrator_deltaTime
   var rhoWeight = weight*r[c].rho

   avg[c_avg].weight += weight

   -- Grid point
   avg[c_avg].centerCoordinates += [UTIL.mkArrayConstant(3, weight)]*r[c].centerCoordinates

   -- Primitive variables
   avg[c_avg].pressure_avg    += weight*r[c].pressure
   avg[c_avg].pressure_rms    += weight*r[c].pressure*r[c].pressure
   avg[c_avg].temperature_avg += weight*r[c].temperature
   avg[c_avg].temperature_rms += weight*r[c].temperature*r[c].temperature
   avg[c_avg].MolarFracs_avg += [UTIL.mkArrayConstant(nSpec, weight)]*r[c].MolarFracs
   avg[c_avg].MolarFracs_rms += [UTIL.mkArrayConstant(nSpec, weight)]*r[c].MolarFracs*r[c].MolarFracs
   avg[c_avg].velocity_avg += [UTIL.mkArrayConstant(3, weight)]*r[c].velocity
   avg[c_avg].velocity_rms += [UTIL.mkArrayConstant(3, weight)]*r[c].velocity*r[c].velocity
   avg[c_avg].velocity_rey += array(r[c].velocity[0]*r[c].velocity[1]*weight,
                                    r[c].velocity[0]*r[c].velocity[2]*weight,
                                    r[c].velocity[1]*r[c].velocity[2]*weight)

   -- Favre averaged primitives
   avg[c_avg].pressure_favg    += rhoWeight*r[c].pressure
   avg[c_avg].pressure_frms    += rhoWeight*r[c].pressure*r[c].pressure
   avg[c_avg].temperature_favg += rhoWeight*r[c].temperature
   avg[c_avg].temperature_frms += rhoWeight*r[c].temperature*r[c].temperature
   avg[c_avg].MolarFracs_favg  += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*r[c].MolarFracs
   avg[c_avg].MolarFracs_frms  += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*r[c].MolarFracs*r[c].MolarFracs
   avg[c_avg].velocity_favg += [UTIL.mkArrayConstant(3, rhoWeight)]*r[c].velocity
   avg[c_avg].velocity_frms += [UTIL.mkArrayConstant(3, rhoWeight)]*r[c].velocity*r[c].velocity
   avg[c_avg].velocity_frey += array(r[c].velocity[0]*r[c].velocity[1]*rhoWeight,
                                          r[c].velocity[0]*r[c].velocity[2]*rhoWeight,
                                          r[c].velocity[1]*r[c].velocity[2]*rhoWeight)

   -- Kinetic energy budgets (y is the inhomogeneous direction)
   var tau_xx = r[c].mu*(4.0*r[c].velocityGradientX[0] - 2.0*r[c].velocityGradientY[1] - 2.0*r[c].velocityGradientZ[2])/3.0
   var tau_yy = r[c].mu*(4.0*r[c].velocityGradientY[1] - 2.0*r[c].velocityGradientX[0] - 2.0*r[c].velocityGradientZ[2])/3.0
   var tau_zz = r[c].mu*(4.0*r[c].velocityGradientZ[2] - 2.0*r[c].velocityGradientX[0] - 2.0-r[c].velocityGradientY[1])/3.0
   var tau_xy = r[c].mu*(r[c].velocityGradientX[1] + r[c].velocityGradientY[0])
   var tau_yz = r[c].mu*(r[c].velocityGradientY[2] + r[c].velocityGradientZ[1])
   var tau_xz = r[c].mu*(r[c].velocityGradientZ[0] + r[c].velocityGradientX[2])

   avg[c_avg].rhoUUv += array(r[c].rho*r[c].velocity[0]*r[c].velocity[0]*r[c].velocity[1]*weight,
                              r[c].rho*r[c].velocity[1]*r[c].velocity[1]*r[c].velocity[1]*weight,
                              r[c].rho*r[c].velocity[2]*r[c].velocity[2]*r[c].velocity[1]*weight)
   avg[c_avg].Up += array(r[c].velocity[0]*r[c].pressure*weight,
                          r[c].velocity[1]*r[c].pressure*weight,
                          r[c].velocity[2]*r[c].pressure*weight)
   avg[c_avg].tau += [UTIL.mkArrayConstant(6, weight)]*array(tau_xx, tau_yy, tau_zz, tau_xy, tau_yz, tau_xz)
   avg[c_avg].utau_y += array(r[c].velocity[0]*tau_xy*weight,
                              r[c].velocity[1]*tau_yy*weight,
                              r[c].velocity[2]*tau_yz*weight)
   avg[c_avg].tauGradU += array((tau_xx*r[c].velocityGradientX[0] + tau_xy*r[c].velocityGradientY[0] + tau_xz*r[c].velocityGradientZ[0])*weight,
                                (tau_xy*r[c].velocityGradientX[1] + tau_yy*r[c].velocityGradientY[1] + tau_yz*r[c].velocityGradientZ[1])*weight,
                                (tau_xz*r[c].velocityGradientX[2] + tau_yz*r[c].velocityGradientY[2] + tau_zz*r[c].velocityGradientZ[2])*weight)
   avg[c_avg].pGradU += array(r[c].pressure*r[c].velocityGradientX[0]*weight,
                              r[c].pressure*r[c].velocityGradientY[1]*weight,
                              r[c].pressure*r[c].velocityGradientZ[2]*weight)

   -- Fluxes
   avg[c_avg].q += array( -r[c].lam*r[c].temperatureGradient[0]*weight,
                          -r[c].lam*r[c].temperatureGradient[1]*weight,
                          -r[c].lam*r[c].temperatureGradient[2]*weight)
end end

local function emitAddAvg2(r, c, avg, c_avg, Integrator_deltaTime, mix) return rquote

   var weight = r[c].cellWidth[0]*r[c].cellWidth[1]*r[c].cellWidth[2]*Integrator_deltaTime
   var rhoWeight = weight*r[c].rho

   -- Properties
   var cp   = MIX.GetHeatCapacity(r[c].temperature, r[c].MassFracs, mix)
   var hi : double[nSpec]
   var Ent  = 0.0
   for i=0, nSpec do
      hi[i] = MIX.GetSpeciesEnthalpy(i, r[c].temperature, mix)
      Ent += r[c].MassFracs[i]*hi[i]
   end
   avg[c_avg].rho_avg += weight*r[c].rho
   avg[c_avg].rho_rms += weight*r[c].rho*r[c].rho
   avg[c_avg].mu_avg  += weight*r[c].mu
   avg[c_avg].lam_avg += weight*r[c].lam
   avg[c_avg].Di_avg  += [UTIL.mkArrayConstant(nSpec, weight)]*r[c].Di
   avg[c_avg].SoS_avg += weight*r[c].SoS
   avg[c_avg].cp_avg  += weight*cp
   avg[c_avg].Ent_avg += weight*Ent

   -- Mass fractions
   avg[c_avg].MassFracs_avg  += [UTIL.mkArrayConstant(nSpec, weight)]*r[c].MassFracs
   avg[c_avg].MassFracs_rms  += [UTIL.mkArrayConstant(nSpec, weight)]*r[c].MassFracs*r[c].MassFracs

   -- Favre averaged properties
   avg[c_avg].mu_favg  += rhoWeight*r[c].mu
   avg[c_avg].lam_favg += rhoWeight*r[c].lam
   avg[c_avg].Di_favg  += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*r[c].Di
   avg[c_avg].SoS_favg += rhoWeight*r[c].SoS
   avg[c_avg].cp_favg  += rhoWeight*cp
   avg[c_avg].Ent_favg += rhoWeight*Ent

   -- Favre averaged mass fractions
   avg[c_avg].MassFracs_favg += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*r[c].MassFracs
   avg[c_avg].MassFracs_frms += [UTIL.mkArrayConstant(nSpec, rhoWeight)]*r[c].MassFracs*r[c].MassFracs

   -- Chemical production rates
   var w    = MIX.GetProductionRates(r[c].rho, r[c].pressure, r[c].temperature, r[c].MassFracs, mix)
   var HR = 0.0
   for i=0, nSpec do
      HR -= w[i]*hi[i]
   end
   avg[c_avg].ProductionRates_avg += [UTIL.mkArrayConstant(nSpec, weight)]*w
   avg[c_avg].ProductionRates_rms += [UTIL.mkArrayConstant(nSpec, weight)]*w*w
   avg[c_avg].HeatReleaseRate_avg += weight*HR
   avg[c_avg].HeatReleaseRate_rms += weight*HR*HR

   -- Dimensionless numbers
   var u2 = MACRO.dot(r[c].velocity, r[c].velocity)
   var Pr = cp*r[c].mu/r[c].lam
   var Ec = u2/(cp*r[c].temperature)
   var nu = r[c].mu/r[c].rho
   var Sc : double[nSpec]
   for i=0, nSpec do
      Sc[i] = nu/r[c].Di[i]
   end
   avg[c_avg].Pr     += weight*Pr
   avg[c_avg].Pr_rms += weight*Pr*Pr
   avg[c_avg].Ec     += weight*Ec
   avg[c_avg].Ec_rms += weight*Ec*Ec
   avg[c_avg].Ma     += weight*sqrt(u2)/r[c].SoS
   avg[c_avg].Sc     += [UTIL.mkArrayConstant(nSpec, weight)]*Sc

   -- Correlations
   var weightU    = weight*r[c].velocity[0]
   var weightV    = weight*r[c].velocity[1]
   var weightW    = weight*r[c].velocity[2]
   var weightRhoU = weight*r[c].velocity[0]*r[c].rho
   var weightRhoV = weight*r[c].velocity[1]*r[c].rho
   var weightRhoW = weight*r[c].velocity[2]*r[c].rho

   avg[c_avg].uT_avg  += array(   weightU*r[c].temperature,
                                  weightV*r[c].temperature,
                                  weightW*r[c].temperature)
   avg[c_avg].uT_favg += array(weightRhoU*r[c].temperature,
                               weightRhoV*r[c].temperature,
                               weightRhoW*r[c].temperature)

   avg[c_avg].uYi_avg  += [UTIL.mkArrayConstant(nSpec, weightU)]*r[c].MassFracs
   avg[c_avg].vYi_avg  += [UTIL.mkArrayConstant(nSpec, weightV)]*r[c].MassFracs
   avg[c_avg].wYi_avg  += [UTIL.mkArrayConstant(nSpec, weightW)]*r[c].MassFracs
   avg[c_avg].uYi_favg += [UTIL.mkArrayConstant(nSpec, weightRhoU)]*r[c].MassFracs
   avg[c_avg].vYi_favg += [UTIL.mkArrayConstant(nSpec, weightRhoV)]*r[c].MassFracs
   avg[c_avg].wYi_favg += [UTIL.mkArrayConstant(nSpec, weightRhoW)]*r[c].MassFracs

end end

local function mkAddAverages(dir)
   local AddAverages
   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task AddAverages(Fluid : region(ispace(int3d), Fluid_columns),
                    Averages : region(ispace(int2d), Averages_columns),
                    mix : MIX.Mixture,
                    Integrator_deltaTime : double)
   where
      reads(Fluid.centerCoordinates),
      reads(Fluid.cellWidth),
      reads(Fluid.[Primitives]),
      reads(Fluid.MassFracs),
      reads(Fluid.[Properties]),
      reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
      reads(Fluid.temperatureGradient),
      reduces+(Averages.[AveragesVars])
   do
      __demand(__openmp)
      for c in Fluid do
         var c_avg = int2d{c.[dir], Averages.bounds.lo.y};
         [emitAddAvg1(Fluid, c, Averages, c_avg, Integrator_deltaTime)]
      end

      __demand(__openmp)
      for c in Fluid do
         var c_avg = int2d{c.[dir], Averages.bounds.lo.y};
         [emitAddAvg2(Fluid, c, Averages, c_avg, Integrator_deltaTime, mix)]
      end
   end
   return AddAverages
end

local function mkAdd1DAverages(dir)
   local Add1DAverages
   local t1
   local t2
   if dir == "x" then
      t1 = "y"
      t2 = "z"
   elseif dir == "y" then
      t1 = "x"
      t2 = "z"
   elseif dir == "z" then
      t1 = "x"
      t2 = "y"
   else assert(false) end

   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task Add1DAverages(Fluid : region(ispace(int3d), Fluid_columns),
                      Averages : region(ispace(int3d), Averages_columns),
                      mix : MIX.Mixture,
                      Integrator_deltaTime : double)
   where
      reads(Fluid.centerCoordinates),
      reads(Fluid.cellWidth),
      reads(Fluid.[Primitives]),
      reads(Fluid.MassFracs),
      reads(Fluid.[Properties]),
      reads(Fluid.{velocityGradientX, velocityGradientY, velocityGradientZ}),
      reads(Fluid.temperatureGradient),
      reduces+(Averages.[AveragesVars])
   do
      __demand(__openmp)
      for c in Fluid do
         var c_avg = int3d{c.[t1], c.[t2], Averages.bounds.lo.z};
         [emitAddAvg1(Fluid, c, Averages, c_avg, Integrator_deltaTime)]
      end

      __demand(__openmp)
      for c in Fluid do
         var c_avg = int3d{c.[t1], c.[t2], Averages.bounds.lo.z};
         [emitAddAvg2(Fluid, c, Averages, c_avg, Integrator_deltaTime, mix)]
      end
   end
   return Add1DAverages
end

local function mkDummyAverages(nd)
   local DummyAverages
   __demand(__leaf)
   task DummyAverages(Averages : region(ispace(nd), Averages_columns))
   where
      reads writes(Averages)
   do
      -- Nothing
      -- It is just to avoid the bug of HDF libraries with parallel reduction
   end
   return DummyAverages
end

-------------------------------------------------------------------------------
-- EXPORTED ROUTINES
-------------------------------------------------------------------------------
function Exports.DeclSymbols(s, Grid, Fluid, p_All, config, MAPPER)

   local function ColorFluid(inp, color) return rquote
      for p=0, [inp].length do
         -- Clip rectangles from the input
         var vol = [inp].values[p]
         vol.fromCell[0] max= 0
         vol.fromCell[1] max= 0
         vol.fromCell[2] max= 0
         vol.uptoCell[0] min= config.Grid.xNum + 2*Grid.xBnum
         vol.uptoCell[1] min= config.Grid.yNum + 2*Grid.yBnum
         vol.uptoCell[2] min= config.Grid.zNum + 2*Grid.zBnum
         -- add to the coloring
         var rect = rect3d{
            lo = int3d{vol.fromCell[0], vol.fromCell[1], vol.fromCell[2]},
            hi = int3d{vol.uptoCell[0], vol.uptoCell[1], vol.uptoCell[2]}}
         regentlib.c.legion_domain_point_coloring_color_domain(color, int1d(p), rect)
      end
      -- Add one point to avoid errors
      if [inp].length == 0 then regentlib.c.legion_domain_point_coloring_color_domain(color, int1d(0), rect3d{lo = int3d{0,0,0}, hi = int3d{0,0,0}}) end
   end end

   return rquote

      var sampleId = config.Mapping.sampleId

      -------------------------------------------------------------------------
      -- 2D Averages
      -------------------------------------------------------------------------

      -- Create averages regions
      var is_YZAverages = ispace(int2d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                         y = config.IO.YZAverages.length    })

      var is_XZAverages = ispace(int2d, {x = config.Grid.yNum + 2*Grid.yBnum,
                                         y = config.IO.XZAverages.length    })

      var is_XYAverages = ispace(int2d, {x = config.Grid.zNum + 2*Grid.zBnum,
                                         y = config.IO.XYAverages.length    })

      var [s.YZAverages] = region(is_YZAverages, Averages_columns)
      var [s.XZAverages] = region(is_XZAverages, Averages_columns)
      var [s.XYAverages] = region(is_XYAverages, Averages_columns)
      var [s.YZAverages_copy] = region(is_YZAverages, Averages_columns)
      var [s.XZAverages_copy] = region(is_XZAverages, Averages_columns)
      var [s.XYAverages_copy] = region(is_XYAverages, Averages_columns);

      [UTIL.emitRegionTagAttach(s.YZAverages,      MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.XZAverages,      MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.XYAverages,      MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.YZAverages_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.XZAverages_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      [UTIL.emitRegionTagAttach(s.XYAverages_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

      -- Partitioning averages in rakes for IO
      var [s.is_Xrakes] = ispace(int2d, {1, max(config.IO.YZAverages.length, 1)})
      var [s.is_Yrakes] = ispace(int2d, {1, max(config.IO.XZAverages.length, 1)})
      var [s.is_Zrakes] = ispace(int2d, {1, max(config.IO.XYAverages.length, 1)})

      var [s.Xrakes] = partition(equal, s.YZAverages, s.is_Xrakes)
      var [s.Yrakes] = partition(equal, s.XZAverages, s.is_Yrakes)
      var [s.Zrakes] = partition(equal, s.XYAverages, s.is_Zrakes)

      var [s.Xrakes_copy] = partition(equal, s.YZAverages_copy, s.is_Xrakes)
      var [s.Yrakes_copy] = partition(equal, s.XZAverages_copy, s.is_Yrakes)
      var [s.Zrakes_copy] = partition(equal, s.XYAverages_copy, s.is_Zrakes)

      -- Partitioning averages in rakes for kernels
      var is_XrakesTiles = ispace(int2d, {Grid.NX, max(config.IO.YZAverages.length, 1)})
      var is_YrakesTiles = ispace(int2d, {Grid.NY, max(config.IO.XZAverages.length, 1)})
      var is_ZrakesTiles = ispace(int2d, {Grid.NZ, max(config.IO.XYAverages.length, 1)});

      var [s.p_Xrakes] = [UTIL.mkPartitionByTile(int2d, int2d, Averages_columns)]
                         (s.YZAverages, is_XrakesTiles, int2d{Grid.xBnum,0}, int2d{0,0})
      var [s.p_Yrakes] = [UTIL.mkPartitionByTile(int2d, int2d, Averages_columns)]
                         (s.XZAverages, is_YrakesTiles, int2d{Grid.yBnum,0}, int2d{0,0})
      var [s.p_Zrakes] = [UTIL.mkPartitionByTile(int2d, int2d, Averages_columns)]
                         (s.XYAverages, is_ZrakesTiles, int2d{Grid.zBnum,0}, int2d{0,0})

      -- Partition the Fluid region based on the specified regions
      -- One color for each type of rakes
      var p_YZAvg_coloring = regentlib.c.legion_domain_point_coloring_create()
      var p_XZAvg_coloring = regentlib.c.legion_domain_point_coloring_create()
      var p_XYAvg_coloring = regentlib.c.legion_domain_point_coloring_create();

      -- Color X rakes
      [ColorFluid(rexpr config.IO.YZAverages end, p_YZAvg_coloring)];
      -- Color Y rakes
      [ColorFluid(rexpr config.IO.XZAverages end, p_XZAvg_coloring)];
      -- Color Z rakes
      [ColorFluid(rexpr config.IO.XYAverages end, p_XYAvg_coloring)];

      -- Make partions of Fluid
      var Fluid_YZAvg = partition(aliased, Fluid, p_YZAvg_coloring, ispace(int1d, max(config.IO.YZAverages.length, 1)))
      var Fluid_XZAvg = partition(aliased, Fluid, p_XZAvg_coloring, ispace(int1d, max(config.IO.XZAverages.length, 1)))
      var Fluid_XYAvg = partition(aliased, Fluid, p_XYAvg_coloring, ispace(int1d, max(config.IO.XYAverages.length, 1)))

      -- Split over tiles
      var [s.p_Fluid_YZAvg] = cross_product(Fluid_YZAvg, p_All)
      var [s.p_Fluid_XZAvg] = cross_product(Fluid_XZAvg, p_All)
      var [s.p_Fluid_XYAvg] = cross_product(Fluid_XYAvg, p_All)

      -- Attach names for mapping
      for r=0, config.IO.YZAverages.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Fluid_YZAvg[r] end, "p_Fluid_YZAvg")];
      end
      for r=0, config.IO.XZAverages.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Fluid_XZAvg[r] end, "p_Fluid_XZAvg")];
      end
      for r=0, config.IO.XYAverages.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Fluid_XYAvg[r] end, "p_Fluid_XYAvg")];
      end

      -- Destroy colors
      regentlib.c.legion_domain_point_coloring_destroy(p_YZAvg_coloring)
      regentlib.c.legion_domain_point_coloring_destroy(p_XZAvg_coloring)
      regentlib.c.legion_domain_point_coloring_destroy(p_XYAvg_coloring)

      -- Extract relevant index spaces
      var aux = region(p_All.colors, bool)
      var [s.YZAvg_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_YZAvg, p_All, s.p_Fluid_YZAvg, aux)
      var [s.XZAvg_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_XZAvg, p_All, s.p_Fluid_XZAvg, aux)
      var [s.XYAvg_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_XYAvg, p_All, s.p_Fluid_XYAvg, aux)

      -------------------------------------------------------------------------
      -- 1D Averages
      -------------------------------------------------------------------------

      -- Create averages regions
      var is_XAverages = ispace(int3d, {x = config.Grid.yNum + 2*Grid.yBnum,
                                        y = config.Grid.zNum + 2*Grid.zBnum,
                                        z = config.IO.XAverages.length    })

      var is_YAverages = ispace(int3d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                        y = config.Grid.zNum + 2*Grid.zBnum,
                                        z = config.IO.YAverages.length    })

      var is_ZAverages = ispace(int3d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                        y = config.Grid.yNum + 2*Grid.yBnum,
                                        z = config.IO.ZAverages.length    })

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

      -- Partitioning averages in planes for calculations
      var is_YZplanes = ispace(int3d, {Grid.NY, Grid.NZ, max(config.IO.XAverages.length, 1)})
      var is_XZplanes = ispace(int3d, {Grid.NX, Grid.NZ, max(config.IO.YAverages.length, 1)})
      var is_XYplanes = ispace(int3d, {Grid.NX, Grid.NY, max(config.IO.ZAverages.length, 1)})

      var [s.YZplanes] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "YZplanes")]
                               (s.XAverages, is_YZplanes, int3d{Grid.yBnum,Grid.zBnum,0}, int3d{0,0,0})
      var [s.XZplanes] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "XZplanes")]
                               (s.YAverages, is_XZplanes, int3d{Grid.xBnum,Grid.zBnum,0}, int3d{0,0,0})
      var [s.XYplanes] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "XYplanes")]
                               (s.ZAverages, is_XYplanes, int3d{Grid.xBnum,Grid.yBnum,0}, int3d{0,0,0})

      -- Partitioning averages in planes for IO
      var [s.is_IO_YZplanes] = ispace(int3d, {Grid.NYout, Grid.NZout, max(config.IO.XAverages.length, 1)})
      var [s.is_IO_XZplanes] = ispace(int3d, {Grid.NXout, Grid.NZout, max(config.IO.YAverages.length, 1)})
      var [s.is_IO_XYplanes] = ispace(int3d, {Grid.NXout, Grid.NYout, max(config.IO.ZAverages.length, 1)})

      var [s.IO_YZplanes] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "IO_YZplanes")]
                               (s.XAverages, s.is_IO_YZplanes, int3d{Grid.yBnum,Grid.zBnum,0}, int3d{0,0,0})
      var [s.IO_XZplanes] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "IO_XZplanes")]
                               (s.YAverages, s.is_IO_XZplanes, int3d{Grid.xBnum,Grid.zBnum,0}, int3d{0,0,0})
      var [s.IO_XYplanes] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "IO_XYplanes")]
                               (s.ZAverages, s.is_IO_XYplanes, int3d{Grid.xBnum,Grid.yBnum,0}, int3d{0,0,0})

      var [s.IO_YZplanes_copy] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "IO_YZplanes_copy")]
                               (s.XAverages_copy, s.is_IO_YZplanes, int3d{Grid.yBnum,Grid.zBnum,0}, int3d{0,0,0})
      var [s.IO_XZplanes_copy] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "IO_XZplanes_copy")]
                               (s.YAverages_copy, s.is_IO_XZplanes, int3d{Grid.xBnum,Grid.zBnum,0}, int3d{0,0,0})
      var [s.IO_XYplanes_copy] = [UTIL.mkPartitionByTile(int3d, int3d, Averages_columns, "IO_XYplanes_copy")]
                               (s.ZAverages_copy, s.is_IO_XYplanes, int3d{Grid.xBnum,Grid.yBnum,0}, int3d{0,0,0})

      -- Partition the Fluid region based on the specified regions
      -- One color for each type of rakes
      var p_XAvg_coloring = regentlib.c.legion_domain_point_coloring_create()
      var p_YAvg_coloring = regentlib.c.legion_domain_point_coloring_create()
      var p_ZAvg_coloring = regentlib.c.legion_domain_point_coloring_create();

      -- Color X planes
      [ColorFluid(rexpr config.IO.XAverages end, p_XAvg_coloring)];
      -- Color Y planes
      [ColorFluid(rexpr config.IO.YAverages end, p_YAvg_coloring)];
      -- Color Z rakes
      [ColorFluid(rexpr config.IO.ZAverages end, p_ZAvg_coloring)];

      -- Make partions of Fluid
      var Fluid_XAvg = partition(aliased, Fluid, p_XAvg_coloring, ispace(int1d, max(config.IO.XAverages.length, 1)))
      var Fluid_YAvg = partition(aliased, Fluid, p_YAvg_coloring, ispace(int1d, max(config.IO.YAverages.length, 1)))
      var Fluid_ZAvg = partition(aliased, Fluid, p_ZAvg_coloring, ispace(int1d, max(config.IO.ZAverages.length, 1)))

      -- Split over tiles
      var [s.p_Fluid_XAvg] = cross_product(Fluid_XAvg, p_All)
      var [s.p_Fluid_YAvg] = cross_product(Fluid_YAvg, p_All)
      var [s.p_Fluid_ZAvg] = cross_product(Fluid_ZAvg, p_All)

      -- Attach names for mapping
      for r=0, config.IO.YZAverages.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Fluid_XAvg[r] end, "p_Fluid_XAvg")];
      end
      for r=0, config.IO.XZAverages.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Fluid_YAvg[r] end, "p_Fluid_YAvg")];
      end
      for r=0, config.IO.XYAverages.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Fluid_ZAvg[r] end, "p_Fluid_ZAvg")];
      end

      -- Destroy colors
      regentlib.c.legion_domain_point_coloring_destroy(p_XAvg_coloring)
      regentlib.c.legion_domain_point_coloring_destroy(p_YAvg_coloring)
      regentlib.c.legion_domain_point_coloring_destroy(p_ZAvg_coloring)

      -- Extract relevant index spaces
      --var aux = region(p_All.colors, bool) -- aux is defined earlier
      var [s.XAvg_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_XAvg, p_All, s.p_Fluid_XAvg, aux)
      var [s.YAvg_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_YAvg, p_All, s.p_Fluid_YAvg, aux)
      var [s.ZAvg_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_ZAvg, p_All, s.p_Fluid_ZAvg, aux)
   end
end

function Exports.InitRakesAndPlanes(s)
   return rquote
      [mkInitializeAverages(int2d)](s.YZAverages);
      [mkInitializeAverages(int2d)](s.XZAverages);
      [mkInitializeAverages(int2d)](s.XYAverages);
      [mkInitializeAverages(int3d)](s.XAverages);
      [mkInitializeAverages(int3d)](s.YAverages);
      [mkInitializeAverages(int3d)](s.ZAverages);
   end
end

function Exports.ReadAverages(s, config)
   local function ReadAvg(avg, dirname)
      local p
      local HDF
      if     avg == "YZAverages" then p = "Xrakes"; HDF = HDF_RAKES
      elseif avg == "XZAverages" then p = "Yrakes"; HDF = HDF_RAKES
      elseif avg == "XYAverages" then p = "Zrakes"; HDF = HDF_RAKES
      elseif avg == "XAverages" then p = "IO_YZplanes"; HDF = HDF_PLANES
      elseif avg == "YAverages" then p = "IO_XZplanes"; HDF = HDF_PLANES
      elseif avg == "ZAverages" then p = "IO_XYplanes"; HDF = HDF_PLANES
      else assert(false) end
      local is = "is_"..p
      local acopy = avg.."_copy"
      local pcopy = p.."_copy"
      return rquote
         if config.IO.[avg].length ~= 0 then
            var restartDir = config.Flow.initCase.u.Restart.restartDir
            format.snprint(dirname, 256, "{}/{}", [&int8](restartDir), [avg])
            HDF.load(0, s.[is], dirname, s.[avg], s.[acopy], s.[p], s.[pcopy])
         end
      end
   end
   return rquote
      if not config.IO.ResetAverages then
         regentlib.assert(config.Flow.initCase.type == SCHEMA.FlowInitCase_Restart,
                          "Flow.initCase needs to be equal to Restart in order to read some averages")
         var dirname = [&int8](C.malloc(256));
         -- 2D averages
         [ReadAvg("YZAverages", dirname)];
         [ReadAvg("XZAverages", dirname)];
         [ReadAvg("XYAverages", dirname)];
         -- 1D averages
         [ReadAvg("XAverages", dirname)];
         [ReadAvg("YAverages", dirname)];
         [ReadAvg("ZAverages", dirname)];
         C.free(dirname)
      end
   end
end

function Exports.AddAverages(s, deltaTime, config, Mix)
   local function Add2DAvg(dir)
      local avg
      local p1
      local p2
      local mk_c
      if     dir == "x" then
         avg = "YZAverages"
         p1 = "p_Xrakes"
         p2 = "YZAvg"
         mk_c = function(c, rake) return rexpr int2d{c.x,rake} end end
      elseif dir == "y" then
         avg = "XZAverages"
         p1 = "p_Yrakes"
         p2 = "XZAvg"
         mk_c = function(c, rake) return rexpr int2d{c.y,rake} end end
      elseif dir == "z" then
         avg = "XYAverages"
         p1 = "p_Zrakes"
         p2 = "XYAvg"
         mk_c = function(c, rake) return rexpr int2d{c.z,rake} end end
      else assert(false) end
      local fp = "p_Fluid_"..p2
      local t = p2.."_tiles"
      return rquote
         for rake=0, config.IO.[avg].length do
            var cs = s.[t][rake].ispace
            __demand(__index_launch)
            for c in cs do
               [mkAddAverages(dir)](s.[fp][rake][c], s.[p1][ [mk_c(c, rake)] ],
                                    Mix, deltaTime)
            end
         end
      end
   end
   local function Add1DAvg(dir)
      local avg
      local p1
      local p2
      local mk_c
      if     dir == "x" then
         avg = "XAverages"
         p1 = "YZplanes"
         p2 = "XAvg"
         mk_c = function(c, plane) return rexpr int3d{c.y, c.z, plane} end end
      elseif dir == "y" then
         avg = "YAverages"
         p1 = "XZplanes"
         p2 = "YAvg"
         mk_c = function(c, plane) return rexpr int3d{c.x, c.z, plane} end end
      elseif dir == "z" then
         avg = "ZAverages"
         p1 = "XYplanes"
         p2 = "ZAvg"
         mk_c = function(c, plane) return rexpr int3d{c.x, c.y, plane} end end
      else assert(false) end
      local fp = "p_Fluid_"..p2
      local t = p2.."_tiles"
      return rquote
         for plane=0, config.IO.[avg].length do
            var cs = s.[t][plane].ispace
            __demand(__index_launch)
            for c in cs do
               [mkAdd1DAverages(dir)](s.[fp][plane][c], s.[p1][ [mk_c(c, plane)] ],
                                    Mix, deltaTime)
            end
         end
      end
   end
   return rquote
      -- 2D averages
      [Add2DAvg("x")];
      [Add2DAvg("y")];
      [Add2DAvg("z")];
      -- 1D averages
      [Add1DAvg("x")];
      [Add1DAvg("y")];
      [Add1DAvg("z")];
   end
end

function Exports.WriteAverages(s, tiles, dirname, IO, SpeciesNames, config)
   local function write2DAvg(dir)
      local avg
      local p
      if     dir == "x" then avg = "YZAverages" p = "Xrakes"
      elseif dir == "y" then avg = "XZAverages" p = "Yrakes"
      elseif dir == "z" then avg = "XYAverages" p = "Zrakes"
      else assert(false) end
      local is = "is_"..p
      local alocal = avg.."_local"
      local acopy = avg.."_copy"
      local pcopy = p.."_copy"
      return rquote
         if config.IO.[avg].length ~= 0 then
            ---------------------------------
            -- Workaroud to Legion issue #521
            [mkDummyAverages(int2d)](s.[avg])
            ---------------------------------
            var Avgdirname = [&int8](C.malloc(256))
            format.snprint(Avgdirname, 256, "{}/{}", dirname, [avg])
            var _1 = IO.createDir(Avgdirname)
            _1 = HDF_RAKES.dump(               _1, s.[is], Avgdirname, s.[avg], s.[acopy], s.[p], s.[pcopy])
            _1 = HDF_RAKES.write.SpeciesNames( _1, s.[is], Avgdirname, s.[avg], s.[p], SpeciesNames)
            C.free(Avgdirname)
         end
      end
   end
   local function write1DAvg(dir)
      local avg
      local p
      local p2
      local mk_c
      local mk_c1
      if     dir == "x" then avg = "XAverages" p = "YZplanes" p2 = "XAvg"
      elseif dir == "y" then avg = "YAverages" p = "XZplanes" p2 = "YAvg"
      elseif dir == "z" then avg = "ZAverages" p = "XYplanes" p2 = "ZAvg"
      else assert(false) end
      local acopy = avg.."_copy"
      local iop = "IO_"..p
      local iopcopy = iop.."_copy"
      local is = "is_"..iop
      local t = p2.."_tiles"
      return rquote
         if config.IO.[avg].length ~= 0 then
            ----------------------------------------------
            -- Add a dummy task to avoid Legion issue #521
            [mkDummyAverages(int3d)](s.[avg])
            ----------------------------------------------
            var Avgdirname = [&int8](C.malloc(256))
            format.snprint(Avgdirname, 256, "{}/{}", dirname, [avg])
            var _1 = IO.createDir(Avgdirname)
            _1 = HDF_PLANES.dump(               _1, s.[is], Avgdirname, s.[avg], s.[acopy], s.[iop], s.[iopcopy])
            _1 = HDF_PLANES.write.SpeciesNames( _1, s.[is], Avgdirname, s.[avg], s.[iop], SpeciesNames)
            C.free(Avgdirname)
         end
      end
   end
   return rquote
      -- 2D averages
      [write2DAvg("x")];
      [write2DAvg("y")];
      [write2DAvg("z")];
      -- 1D averages
      [write1DAvg("x")];
      [write1DAvg("y")];
      [write1DAvg("z")];
   end
end

return Exports end

