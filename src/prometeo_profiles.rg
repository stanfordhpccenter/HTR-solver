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
local UTIL = require 'util'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local ProfilesVars = CONST.ProfilesVars

Exports.HDF = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                     ProfilesVars,
                                     {},
                                     {})

-------------------------------------------------------------------------------
-- PROFILES ROUTINES
-------------------------------------------------------------------------------

function Exports.mkInitializeProfilesField(Side)
  local InitializeProfilesField
  __demand(__inline)
  task InitializeProfilesField(BC : region(ispace(int3d), Fluid_columns),
                               config : SCHEMA.Config,
                               mix : MIX.Mixture)
  where
     reads writes(BC.[ProfilesVars])
  do
     if (config.BC.[Side].type == SCHEMA.FlowBC_Dirichlet) then
        var Dirichlet = config.BC.[Side].u.Dirichlet
        if Dirichlet.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
           var BC_Mixture = MIX.ParseConfigMixture(Dirichlet.MixtureProfile.u.Constant.Mixture,  mix)
           fill(BC.MolarFracs_profile, BC_Mixture)
        end
        if Dirichlet.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
           fill(BC.velocity_profile, Dirichlet.VelocityProfile.u.Constant.velocity)
        end
        if Dirichlet.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
           fill(BC.temperature_profile, Dirichlet.TemperatureProfile.u.Constant.temperature)
        end

     elseif (config.BC.[Side].type == SCHEMA.FlowBC_NSCBC_Inflow) then
         var NSCBC_Inflow = config.BC.[Side].u.NSCBC_Inflow
         if NSCBC_Inflow.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
            var BC_Mixture = MIX.ParseConfigMixture(NSCBC_Inflow.MixtureProfile.u.Constant.Mixture,  mix)
            fill(BC.MolarFracs_profile, BC_Mixture)
         end
         if NSCBC_Inflow.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
            fill(BC.velocity_profile, NSCBC_Inflow.VelocityProfile.u.Constant.velocity)
         end
         if NSCBC_Inflow.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
            fill(BC.temperature_profile, NSCBC_Inflow.TemperatureProfile.u.Constant.temperature)
         end

     elseif (config.BC.[Side].type == SCHEMA.FlowBC_NSCBC_FarField) then
         var NSCBC_FarField = config.BC.[Side].u.NSCBC_FarField
         if NSCBC_FarField.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
            var BC_Mixture = MIX.ParseConfigMixture(NSCBC_FarField.MixtureProfile.u.Constant.Mixture,  mix)
            fill(BC.MolarFracs_profile, BC_Mixture)
         end
         if NSCBC_FarField.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
            fill(BC.velocity_profile, NSCBC_FarField.VelocityProfile.u.Constant.velocity)
         end
         if NSCBC_FarField.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
            fill(BC.temperature_profile, NSCBC_FarField.TemperatureProfile.u.Constant.temperature)
         end

      elseif (config.BC.[Side].type == SCHEMA.FlowBC_IsothermalWall) then
         var IsothermalWall = config.BC.[Side].u.IsothermalWall
         if IsothermalWall.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
            fill(BC.temperature_profile, IsothermalWall.TemperatureProfile.u.Constant.temperature)
         end

      elseif (config.BC.[Side].type == SCHEMA.FlowBC_SuctionAndBlowingWall) then
         var SuctionAndBlowingWall = config.BC.[Side].u.SuctionAndBlowingWall
         if SuctionAndBlowingWall.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
            fill(BC.temperature_profile, SuctionAndBlowingWall.TemperatureProfile.u.Constant.temperature)
         end

     elseif (config.BC.[Side].type == SCHEMA.FlowBC_RecycleRescaling) then
         var RecycleRescaling = config.BC.[Side].u.RecycleRescaling
         if RecycleRescaling.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
            var BC_Mixture = MIX.ParseConfigMixture(RecycleRescaling.MixtureProfile.u.Constant.Mixture,  mix)
            fill(BC.MolarFracs_profile, BC_Mixture)
         end
         if RecycleRescaling.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
            fill(BC.velocity_profile, RecycleRescaling.VelocityProfile.u.Constant.velocity)
         end
         if RecycleRescaling.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
            fill(BC.temperature_profile, RecycleRescaling.TemperatureProfile.u.Constant.temperature)
         end

      end
   end
   return InitializeProfilesField
end

return Exports end

