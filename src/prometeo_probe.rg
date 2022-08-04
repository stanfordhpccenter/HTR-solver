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

return function(SCHEMA, MIX, IO, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local sqrt = regentlib.sqrt(double)
local UTIL = require 'util'
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

function Exports.mkProbesList()
   return {
      probes_tiles = regentlib.newsymbol(),
      p_Probes = regentlib.newsymbol("p_Probes"),
      Volumes = regentlib.newsymbol(double[5], "Volumes") -- Match the up to in config_schema.lua
   }
end

-------------------------------------------------------------------------------
-- PROBE ROUTINES
-------------------------------------------------------------------------------

local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
task ProbeVolume(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d})
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      acc += 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
   end
   return acc
end

local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
task ProbeTemperature(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   reads(Fluid.temperature)
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
      var vol = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
      acc += Fluid[c].temperature*vol
   end
   return acc
end

local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
task ProbePressure(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   reads(Fluid.pressure)
do
   var acc = 0.0
   __demand(__openmp)
   for c in Fluid do
		var vol = 1.0/(Fluid[c].dcsi_d*Fluid[c].deta_d*Fluid[c].dzet_d)
		acc += Fluid[c].pressure*vol
   end
   return acc
end

-------------------------------------------------------------------------------
-- EXPORTED ROUTINES
-------------------------------------------------------------------------------
function Exports.DeclSymbols(s, Grid, Fluid, p_All, config)
   return rquote

      -- Declare array of volumes
      var [s.Volumes];

      -- Write probe file headers
      __forbid(__index_launch)
      for probeId = 0, config.IO.probes.length do
         IO.Probe_WriteHeader([&int8](config.Mapping.outDir), probeId)
      end

      -- Partition the Fluid region based on the specified regions
      var probe_coloring = regentlib.c.legion_domain_point_coloring_create()

      for p=0, config.IO.probes.length do
         -- Clip rectangles from the input
         var sample = config.IO.probes.values[p]
         sample.fromCell[0] max= 0
         sample.fromCell[1] max= 0
         sample.fromCell[2] max= 0
         sample.uptoCell[0] min= config.Grid.xNum + 2*Grid.xBnum
         sample.uptoCell[1] min= config.Grid.yNum + 2*Grid.yBnum
         sample.uptoCell[2] min= config.Grid.zNum + 2*Grid.zBnum
         -- add to the coloring
         var rect = rect3d{
            lo = int3d{sample.fromCell[0], sample.fromCell[1], sample.fromCell[2]},
            hi = int3d{sample.uptoCell[0], sample.uptoCell[1], sample.uptoCell[2]}}
         regentlib.c.legion_domain_point_coloring_color_domain(probe_coloring, int1d(p), rect)
      end
      -- Add one point to avoid errors
      if config.IO.probes.length == 0 then
         regentlib.c.legion_domain_point_coloring_color_domain(probe_coloring, int1d(0), rect3d{lo = int3d{0,0,0}, hi = int3d{0,0,0}})
      end

      -- Make partitions of Fluid
      var Fluid_Probes = partition(aliased, Fluid, probe_coloring, ispace(int1d, max(config.IO.probes.length, 1)))

      -- Ensure that probes volumes are not empty
      for p=0, config.IO.probes.length do
         regentlib.assert(Fluid_Probes[p].volume ~= 0, "Found a probe with empty volume")
      end

      -- Split over tiles
      var [s.p_Probes] = cross_product(Fluid_Probes, p_All)

      -- Attach names for mapping
      for p=0, config.IO.probes.length do
         [UTIL.emitPartitionNameAttach(rexpr s.p_Probes[p] end, "p_Probes")];
      end

      -- Destroy color
      regentlib.c.legion_domain_point_coloring_destroy(probe_coloring)

      -- Extract relevant index space
      var aux = region(p_All.colors, bool)
      var [s.probes_tiles] = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns)]
                              (Fluid, Fluid_Probes, p_All, s.p_Probes, aux)

   end
end

function Exports.InitProbes(s, config)
   return rquote
      -- Store volume of each probe
      for p=0, config.IO.probes.length do
         s.Volumes[p] = 0.0
         var cs = s.probes_tiles[p].ispace
         __demand(__index_launch)
         for c in cs do
            s.Volumes[p] += ProbeVolume(s.p_Probes[p][c])
         end
      end
   end
end

function Exports.WriteProbes(s, Integrator_timeStep, Integrator_simTime, config)
   return rquote
      if (Integrator_timeStep % config.IO.probesSamplingInterval == 0) then
         for p=0, config.IO.probes.length do
            var cs = s.probes_tiles[p].ispace
            var avgTemperature = 0.0
            var avgPressure = 0.0
            __demand(__index_launch)
            for c in cs do
               avgTemperature += ProbeTemperature(s.p_Probes[p][c])
            end
            __demand(__index_launch)
            for c in cs do
               avgPressure += ProbePressure(s.p_Probes[p][c])
            end
            avgTemperature /= s.Volumes[p]
            avgPressure    /= s.Volumes[p]
            IO.Probe_Write([&int8](config.Mapping.outDir), p,
                           Integrator_timeStep, Integrator_simTime, avgTemperature, avgPressure)
         end
      end
   end
end

return Exports end

