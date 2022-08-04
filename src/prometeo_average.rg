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

return function(SCHEMA, MIX, TYPES, PART,
                ELECTRIC_FIELD) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local sqrt = regentlib.sqrt(double)
local UTIL = require 'util'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local format = require "std/format"

local types_inc_flags = terralib.newlist({"-DEOS="..os.getenv("EOS")})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local AVE_TYPES = terralib.includec("prometeo_average_types.h", types_inc_flags)

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

local Fluid_columns = TYPES.Fluid_columns

local Averages_columns = AVE_TYPES.Averages_columns

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

-- Add electric varaibles to the lists
local additionalVars = terralib.newlist({})
if ELECTRIC_FIELD then
   AveragesVars:insert("electricPotential_avg")
   AveragesVars:insert("chargeDensity_avg")
   additionalVars:insert("electricPotential")
   additionalVars:insert("electricField")
   if (MIX.nIons > 0) then
      additionalVars:insert("Ki")
   end
end

local HDF_RAKES = (require 'hdf_helper')(int2d, int2d, Averages_columns,
                                                       AveragesVars,
                                                       {},
                                                       {SpeciesNames={nSpec,20}})

local HDF_PLANES = (require 'hdf_helper')(int3d, int3d, Averages_columns,
                                                        AveragesVars,
                                                        {},
                                                        {SpeciesNames={nSpec,20}})

function Exports.mkAvgList()
   return {
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
      -- considered partitions of the Fluid domain that provide support to gradient stencil
      p_Gradient_YZAvg = regentlib.newsymbol("p_Gradient_YZAvg"),
      p_Gradient_XZAvg = regentlib.newsymbol("p_Gradient_XZAvg"),
      p_Gradient_XYAvg = regentlib.newsymbol("p_Gradient_XYAvg"),
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
      -- considered partitions of the Fluid domain that provide support to gradient stencil
      p_Gradient_XAvg = regentlib.newsymbol("p_Gradient_YZAvg"),
      p_Gradient_YAvg = regentlib.newsymbol("p_Gradient_XZAvg"),
      p_Gradient_ZAvg = regentlib.newsymbol("p_Gradient_XYAvg"),
      -- tiles of Fluid where the average kernels will be launched
      XAvg_tiles = regentlib.newsymbol(),
      YAvg_tiles = regentlib.newsymbol(),
      ZAvg_tiles = regentlib.newsymbol()
   }
end

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
      fill(Averages.velocity_rey, array(0.0, 0.0, 0.0));
[(function() local __quotes = terralib.newlist()
if ELECTRIC_FIELD then __quotes:insert(rquote
      -- Electric quantities
      fill(Averages.electricPotential_avg, 0.0)
      fill(Averages.chargeDensity_avg, 0.0)
end) end return __quotes end)()];
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
   local Add2DAverages
   extern task Add2DAverages(Ghost : region(ispace(int3d), Fluid_columns),
                             Fluid : region(ispace(int3d), Fluid_columns),
                             Averages : region(ispace(int2d), Averages_columns),
                             mix : MIX.Mixture,
                             Fluid_bounds : rect3d,
                             Integrator_deltaTime : double)
   where
      reads(Ghost.{temperature, MolarFracs, velocity}),
      reads(Fluid.centerCoordinates),
      reads(Fluid.{nType_x, nType_y, nType_z}),
      reads(Fluid.{dcsi_d, deta_d, dzet_d}),
      reads(Fluid.{pressure, MassFracs}),
      reads(Fluid.[Properties]),
      reads(Fluid.[additionalVars]),
      reduces+(Averages.[AveragesVars])
   end
   if dir == "x" then
      Add2DAverages:set_task_id(TYPES.TID_Add2DAveragesX)
   elseif dir == "y" then
      Add2DAverages:set_task_id(TYPES.TID_Add2DAveragesY)
   elseif dir == "z" then
      Add2DAverages:set_task_id(TYPES.TID_Add2DAveragesZ)
   else assert(false) end
   return Add2DAverages
end

local function mkAdd1DAverages(dir)
   local Add1DAverages
   extern task Add1DAverages(Ghost : region(ispace(int3d), Fluid_columns),
                             Fluid : region(ispace(int3d), Fluid_columns),
                             Averages : region(ispace(int3d), Averages_columns),
                             mix : MIX.Mixture,
                             Fluid_bounds : rect3d,
                             Integrator_deltaTime : double)
   where
      reads(Ghost.{temperature, MolarFracs, velocity}),
      reads(Fluid.centerCoordinates),
      reads(Fluid.{nType_x, nType_y, nType_z}),
      reads(Fluid.{dcsi_d, deta_d, dzet_d}),
      reads(Fluid.{pressure, MassFracs}),
      reads(Fluid.[Properties]),
      reads(Fluid.[additionalVars]),
      reduces+(Averages.[AveragesVars])
   end
   if dir == "x" then
      Add1DAverages:set_task_id(TYPES.TID_Add1DAveragesX)
   elseif dir == "y" then
      Add1DAverages:set_task_id(TYPES.TID_Add1DAveragesY)
   elseif dir == "z" then
      Add1DAverages:set_task_id(TYPES.TID_Add1DAveragesZ)
   else assert(false) end
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
function Exports.DeclSymbols(s, Grid, Fluid, Fluid_Zones, config, MAPPER)
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
   end
end

function Exports.InitPartitions(s, Grid, Fluid, p_All, config)

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
      -------------------------------------------------------------------------
      -- 2D Averages
      -------------------------------------------------------------------------
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

      -- Determine partitions for gradient operations
      var [s.p_Gradient_YZAvg] = PART.PartitionAverageGhost(Fluid, p_All, Fluid_YZAvg, s.p_Fluid_YZAvg, config.IO.YZAverages.length, config)
      var [s.p_Gradient_XZAvg] = PART.PartitionAverageGhost(Fluid, p_All, Fluid_XZAvg, s.p_Fluid_XZAvg, config.IO.XZAverages.length, config)
      var [s.p_Gradient_XYAvg] = PART.PartitionAverageGhost(Fluid, p_All, Fluid_XYAvg, s.p_Fluid_XYAvg, config.IO.XYAverages.length, config)

      -------------------------------------------------------------------------
      -- 1D Averages
      -------------------------------------------------------------------------
      -- Partition the Fluid region based on the specified regions
      -- One color for each type of planes
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

      -- Determine partitions for gradient operations of 2D averages
      var [s.p_Gradient_XAvg] = PART.PartitionAverageGhost(Fluid, p_All, Fluid_XAvg, s.p_Fluid_XAvg, config.IO.XAverages.length, config)
      var [s.p_Gradient_YAvg] = PART.PartitionAverageGhost(Fluid, p_All, Fluid_YAvg, s.p_Fluid_YAvg, config.IO.YAverages.length, config)
      var [s.p_Gradient_ZAvg] = PART.PartitionAverageGhost(Fluid, p_All, Fluid_ZAvg, s.p_Fluid_ZAvg, config.IO.ZAverages.length, config)
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
            HDF.load(s.[is], dirname, s.[avg], s.[acopy], s.[p], s.[pcopy])
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

function Exports.AddAverages(s, Fluid_bounds, deltaTime, config, Mix)
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
      local gp = "p_Gradient_"..p2
      local t = p2.."_tiles"
      return rquote
         for rake=0, config.IO.[avg].length do
            var cs = s.[t][rake].ispace
            var { p_GradientGhosts } = s.[gp][rake]
            __demand(__index_launch)
            for c in cs do
               [mkAddAverages(dir)](p_GradientGhosts[c], s.[fp][rake][c],
                                    s.[p1][ [mk_c(c, rake)] ], Mix,
                                    Fluid_bounds, deltaTime)
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
      local gp = "p_Gradient_"..p2
      local t = p2.."_tiles"
      return rquote
         for plane=0, config.IO.[avg].length do
            var cs = s.[t][plane].ispace
            var { p_GradientGhosts } = s.[gp][plane]
            __demand(__index_launch)
            for c in cs do
               [mkAdd1DAverages(dir)](p_GradientGhosts[c], s.[fp][plane][c],
                                      s.[p1][ [mk_c(c, plane)] ], Mix,
                                      Fluid_bounds, deltaTime)
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

function Exports.WriteAverages(_, s, tiles, dirname, IO, SpeciesNames, config)
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
            var _1 = IO.createDir(_, Avgdirname)
            _1 = HDF_RAKES.dump(               _1, s.[is], Avgdirname, s.[avg], s.[acopy], s.[p], s.[pcopy])
            _1 = HDF_RAKES.write.SpeciesNames( _1, Avgdirname, SpeciesNames)
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
            var _1 = IO.createDir(_, Avgdirname)
            _1 = HDF_PLANES.dump(               _1, s.[is], Avgdirname, s.[avg], s.[acopy], s.[iop], s.[iopcopy])
            _1 = HDF_PLANES.write.SpeciesNames( _1, Avgdirname, SpeciesNames)
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

