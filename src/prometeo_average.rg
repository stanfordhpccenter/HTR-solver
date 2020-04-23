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

struct Exports.Averages_columns {
   weight : double;
   -- Primitive variables
   pressure_avg : double;
   pressure_rms : double;
   temperature_avg : double;
   temperature_rms : double;
   MolarFracs_avg : double[nSpec];
   MolarFracs_rms : double[nSpec];
   velocity_avg : double[3];
   velocity_rms : double[3];
   velocity_rey : double[3];
   -- Properties
   rho_avg  : double;
   rho_rms  : double;
   mu_avg   : double;
   mu_rms   : double;
   lam_avg  : double;
   lam_rms  : double;
   Di_avg   : double[nSpec];
   Di_rms   : double[nSpec];
   SoS_avg  : double;
   SoS_rms  : double;
   cp_avg   : double;
   cp_rms   : double;
   MixW_avg : double;
   MixW_rms : double;
   -- Chemical production rates
   ProductionRates_avg : double[nSpec];
   ProductionRates_rms : double[nSpec];
   HeatReleaseRate_avg : double;
   HeatReleaseRate_rms : double;
   -- Favre averages
   temperature_favg : double;
   temperature_frms : double;
   MolarFracs_favg : double[nSpec];
   MolarFracs_frms : double[nSpec];
   velocity_favg : double[3];
   velocity_frms : double[3];
   velocity_frey : double[3];
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
   'velocity_avg',
   'velocity_rms',
   'velocity_rey',
   -- Properties
   'rho_avg',
   'rho_rms',
   'mu_avg',
   'mu_rms',
   'lam_avg',
   'lam_rms',
   'Di_avg',
   'Di_rms',
   'SoS_avg',
   'SoS_rms',
   'cp_avg',
   'cp_rms',
   'MixW_avg',
   'MixW_rms',
   -- Chemical production rates
   'ProductionRates_avg',
   'ProductionRates_rms',
   'HeatReleaseRate_avg',
   'HeatReleaseRate_rms',
   -- Favre averages
   'temperature_favg',
   'temperature_frms',
   'MolarFracs_favg',
   'MolarFracs_frms',
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
   'Ec_rms'
})

Exports.HDF = (require 'hdf_helper')(int2d, int2d, Exports.Averages_columns,
                                                   AveragesVars,
                                                   {},
                                                   {SpeciesNames=nSpec})

-------------------------------------------------------------------------------
-- AVERAGES ROUTINES
-------------------------------------------------------------------------------
function Exports.mkInitializeAverages(nd)
   local InitializeAverages
   __demand(__inline)
   task InitializeAverages(Averages : region(ispace(nd), Exports.Averages_columns))
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
      fill(Averages.velocity_avg, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_rms, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_rey, array(0.0, 0.0, 0.0))
      -- Properties
      fill(Averages.rho_avg, 0.0)
      fill(Averages.rho_rms, 0.0)
      fill(Averages.mu_avg,  0.0)
      fill(Averages.mu_rms,  0.0)
      fill(Averages.lam_avg, 0.0)
      fill(Averages.lam_rms, 0.0)
      fill(Averages.Di_avg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.Di_rms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.SoS_avg,  0.0)
      fill(Averages.SoS_rms,  0.0)
      fill(Averages.cp_avg,   0.0)
      fill(Averages.cp_rms,   0.0)
      fill(Averages.MixW_avg, 0.0)
      fill(Averages.MixW_rms, 0.0)
      -- Chemical production rates
      fill(Averages.ProductionRates_avg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.ProductionRates_rms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.HeatReleaseRate_avg, 0.0)
      fill(Averages.HeatReleaseRate_rms, 0.0)
      -- Favre averages
      fill(Averages.temperature_favg, 0.0)
      fill(Averages.temperature_frms, 0.0)
      fill(Averages.MolarFracs_favg, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.MolarFracs_frms, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
      fill(Averages.velocity_favg, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_frms, array(0.0, 0.0, 0.0))
      fill(Averages.velocity_frey, array(0.0, 0.0, 0.0))
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
   
   end
   return InitializeAverages
end

function Exports.mkAddAverages(dir)
   local AddAverages
   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task AddAverages(Fluid : region(ispace(int3d), Fluid_columns),
                    Averages : region(ispace(int4d), Exports.Averages_columns),
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

            -- Favre averages
            Averages[c_avg].temperature_favg += weight*Fluid[c].rho*Fluid[c].temperature
            Averages[c_avg].temperature_frms += weight*Fluid[c].rho*Fluid[c].temperature*Fluid[c].temperature
            Averages[c_avg].MolarFracs_favg  += [UTIL.mkArrayConstant(nSpec, rexpr weight*Fluid[c].rho end)]*Fluid[c].MolarFracs
            Averages[c_avg].MolarFracs_frms  += [UTIL.mkArrayConstant(nSpec, rexpr weight*Fluid[c].rho end)]*Fluid[c].MolarFracs*Fluid[c].MolarFracs
            Averages[c_avg].velocity_favg += [UTIL.mkArrayConstant(3, rexpr weight*Fluid[c].rho end)]*Fluid[c].velocity   
            Averages[c_avg].velocity_frms += [UTIL.mkArrayConstant(3, rexpr weight*Fluid[c].rho end)]*Fluid[c].velocity*Fluid[c].velocity
            Averages[c_avg].velocity_frey += array(Fluid[c].rho*Fluid[c].velocity[0]*Fluid[c].velocity[1]*weight,
                                                   Fluid[c].rho*Fluid[c].velocity[0]*Fluid[c].velocity[2]*weight,
                                                   Fluid[c].rho*Fluid[c].velocity[1]*Fluid[c].velocity[2]*weight)

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

            -- Properties
            var MixW = MIX.GetMolarWeightFromXi(Fluid[c].MolarFracs, mix)
            var Yi   = MIX.GetMassFractions(MixW, Fluid[c].MolarFracs, mix)
            var cp   = MIX.GetHeatCapacity(Fluid[c].temperature, Yi, mix)
            Averages[c_avg].rho_avg += weight*Fluid[c].rho
            Averages[c_avg].rho_rms += weight*Fluid[c].rho*Fluid[c].rho
            Averages[c_avg].mu_avg  += weight*Fluid[c].mu
            Averages[c_avg].mu_rms  += weight*Fluid[c].mu*Fluid[c].mu
            Averages[c_avg].lam_avg += weight*Fluid[c].lam
            Averages[c_avg].lam_rms += weight*Fluid[c].lam*Fluid[c].lam
            Averages[c_avg].Di_avg  += [UTIL.mkArrayConstant(nSpec, weight)]*Fluid[c].Di
            Averages[c_avg].Di_rms  += [UTIL.mkArrayConstant(nSpec, weight)]*Fluid[c].Di*Fluid[c].Di
            Averages[c_avg].SoS_avg += weight*Fluid[c].SoS
            Averages[c_avg].SoS_rms += weight*Fluid[c].SoS*Fluid[c].SoS
            Averages[c_avg].cp_avg += weight*cp
            Averages[c_avg].cp_rms += weight*cp*cp
            Averages[c_avg].MixW_avg += weight*MixW
            Averages[c_avg].MixW_rms += weight*MixW*MixW

            -- Chemical production rates
            var w    = MIX.GetProductionRates(Fluid[c].rho, Fluid[c].pressure, Fluid[c].temperature, Yi, mix)
            var HR = 0.0
            for i=0, nSpec do
               HR -= w[i]*MIX.GetSpeciesEnthalpy(i, Fluid[c].temperature, mix)
            end
            Averages[c_avg].ProductionRates_avg += [UTIL.mkArrayConstant(nSpec, weight)]*w
            Averages[c_avg].ProductionRates_rms += [UTIL.mkArrayConstant(nSpec, weight)]*w*w
            Averages[c_avg].HeatReleaseRate_avg += weight*HR
            Averages[c_avg].HeatReleaseRate_rms += weight*HR*HR

            -- Dimensionless numbers
            var u2 = MACRO.dot(Fluid[c].velocity, Fluid[c].velocity)
            var Hi = MIX.GetEnthalpy(Fluid[c].temperature, Yi, mix) + 0.5*u2
            var Pr = cp*Fluid[c].mu/Fluid[c].lam
            var Ec = u2/Hi
            Averages[c_avg].Pr     += weight*Pr
            Averages[c_avg].Pr_rms += weight*Pr*Pr
            Averages[c_avg].Ec     += weight*Ec
            Averages[c_avg].Ec_rms += weight*Ec*Ec

         end
      end
   end
   return AddAverages
end

__demand(__leaf)
task Exports.DummyAverages(Averages : region(ispace(int2d), Exports.Averages_columns))
where
   reads writes(Averages)
do
   -- Nothing
   -- It is just to avoid the bug of HDF libraries with parallel reduction
end

function Exports.mkReduceAverages(dir)
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
   task ReduceAverages(Averages       : region(ispace(int2d), Exports.Averages_columns),
                       Averages_local : region(ispace(int4d), Exports.Averages_columns),
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
            Averages[c].velocity_avg    += Averages_local[c_buf].velocity_avg    
            Averages[c].velocity_rms    += Averages_local[c_buf].velocity_rms    
            Averages[c].velocity_rey    += Averages_local[c_buf].velocity_rey    

            -- Properties
            Averages[c].rho_avg  += Averages_local[c_buf].rho_avg 
            Averages[c].rho_rms  += Averages_local[c_buf].rho_rms 
            Averages[c].mu_avg   += Averages_local[c_buf].mu_avg  
            Averages[c].mu_rms   += Averages_local[c_buf].mu_rms  
            Averages[c].lam_avg  += Averages_local[c_buf].lam_avg 
            Averages[c].lam_rms  += Averages_local[c_buf].lam_rms 
            Averages[c].Di_avg   += Averages_local[c_buf].Di_avg  
            Averages[c].Di_rms   += Averages_local[c_buf].Di_rms  
            Averages[c].SoS_avg  += Averages_local[c_buf].SoS_avg 
            Averages[c].SoS_rms  += Averages_local[c_buf].SoS_rms 
            Averages[c].cp_avg   += Averages_local[c_buf].cp_avg  
            Averages[c].cp_rms   += Averages_local[c_buf].cp_rms  
            Averages[c].MixW_avg += Averages_local[c_buf].MixW_avg
            Averages[c].MixW_rms += Averages_local[c_buf].MixW_rms

            -- Chemical production rates
            Averages[c].ProductionRates_avg += Averages_local[c_buf].ProductionRates_avg
            Averages[c].ProductionRates_rms += Averages_local[c_buf].ProductionRates_rms
            Averages[c].HeatReleaseRate_avg += Averages_local[c_buf].HeatReleaseRate_avg
            Averages[c].HeatReleaseRate_rms += Averages_local[c_buf].HeatReleaseRate_rms

            -- Favre averages
            Averages[c].temperature_favg += Averages_local[c_buf].temperature_favg
            Averages[c].temperature_frms += Averages_local[c_buf].temperature_frms
            Averages[c].MolarFracs_favg  += Averages_local[c_buf].MolarFracs_favg
            Averages[c].MolarFracs_frms  += Averages_local[c_buf].MolarFracs_frms
            Averages[c].velocity_favg    += Averages_local[c_buf].velocity_favg
            Averages[c].velocity_frms    += Averages_local[c_buf].velocity_frms
            Averages[c].velocity_frey    += Averages_local[c_buf].velocity_frey

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
         end
      end
   end
   return ReduceAverages
end

return Exports end

