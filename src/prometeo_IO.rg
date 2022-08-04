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

return function(SCHEMA) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local MACRO = require 'prometeo_macro'
local UTIL = require 'util'

-------------------------------------------------------------------------------
-- I/O RUTINES
-------------------------------------------------------------------------------

-- regentlib.rexpr, regentlib.rexpr, regentlib.rexpr* -> regentlib.rquote
local function emitConsoleWrite(outDir, format, ...)
   local args = terralib.newlist{...}
   return rquote
      var consoleFile = [&int8](C.malloc(256))
      C.snprintf(consoleFile, 256, '%s/console.txt', outDir)
      var console = UTIL.openFile(consoleFile, 'a')
      C.free(consoleFile)
      C.fprintf(console, format, [args])
      C.fflush(console)
      C.fclose(console)
   end
end

local -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Console_WriteHeader(outDir : regentlib.string)
   [emitConsoleWrite(outDir, '%10s'..
                             '%15s'..
                             '%13s'..
                             '%15s'..
                             '%15s'..
                             '%15s'..
                             '%15s'..
                             --'%15s'..
                             --'%15s'..
                             --'%15s'..
                             --'%15s'..
                             '%15s\n',
                             'Iter',
                             'Sim Time',
                             'Wall t',
                             'Delta Time',
                             'Avg Press',
                             'Avg Temp',
                             'Avg KE',
                             --'MaxSpeed',
                             --'MaxDensity',
                             --'MaxPressure',
                             --'MaxTemperature',
                             'Avg TotalE'
                             )];
end

local -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Console_Write(outDir : regentlib.string,
                   Integrator_timeStep : int,
                   Integrator_simTime : double,
                   startTime : int64,
                   Integrator_deltaTime : double,
                   Flow_averagePressure : double,
                   Flow_averageTemperature : double,
                   Flow_averageKineticEnergy : double,
                   --Flow_maxSpeed : double,
                   --Flow_maxDensity : double,
                   --Flow_maxPressure : double,
                   --Flow_maxTemperature : double,
                   Flow_totalEnergy : double)
   var deltaTime = (C.legion_get_current_time_in_micros() - startTime) / 1000;
   [emitConsoleWrite(outDir, '%10d'..
                             '%15.7e'..
                             '%9llu.%03llu'..
                             '%15.7e'..
                             '%15.7e'..
                             '%15.7e'..
                             '%15.7e'..
                             --'%15.7e'..
                             --'%15.7e'..
                             --'%15.7e'..
                             --'%15.7e'..
                             '%15.7e\n',
                     Integrator_timeStep,
                     Integrator_simTime,
                     rexpr deltaTime / 1000 end,
                     rexpr deltaTime % 1000 end,
                     Integrator_deltaTime,
                     Flow_averagePressure,
                     Flow_averageTemperature,
                     Flow_averageKineticEnergy,
                     --Flow_maxSpeed,
                     --Flow_maxDensity,
                     --Flow_maxPressure,
                     --Flow_maxTemperature,
                     Flow_totalEnergy)];
end

local -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Console_WriteFooter(outDir : regentlib.string,
                         startTime : int64)
   var deltaTime = (C.legion_get_current_time_in_micros() - startTime) / 1000;
   [emitConsoleWrite(outDir,
                     'Total time: %llu.%03llu seconds\n',
                     rexpr deltaTime / 1000 end,
                     rexpr deltaTime % 1000 end)];
end

-- regentlib.rexpr, regentlib.rexpr, regentlib.rexpr, regentlib.rexpr*
--   -> regentlib.rquote
local function emitProbeWrite(outDir, probeId, format, ...)
  local args = terralib.newlist{...}
  return rquote
    var filename = [&int8](C.malloc(256))
    C.snprintf(filename, 256, '%s/probe%d.csv', outDir, probeId)
    var file = UTIL.openFile(filename, 'a')
    C.free(filename)
    C.fprintf(file, format, [args])
    C.fflush(file)
    C.fclose(file)
  end
end

local -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Probe_WriteHeader(outDir : regentlib.string,
                       probeId : int)
   [emitProbeWrite(outDir, probeId, '%10s'..
                                    '%15s' ..
                                    '%15s' ..
                                    '%15s\n',
                                    'Iter',
                                    'Time',
                                    'Temperature',
                                    'Pressure')];
end

local -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Probe_Write(outDir : regentlib.string,
                 probeId : int,
                 Integrator_timeStep : int,
                 Integrator_simTime : double,
                 avgTemperature : double,
                 avgPressure : double)
   [emitProbeWrite(outDir, probeId, '%10d'..
                                    '%15.7e' ..
                                    '%15.7e' ..
                                    '%15.7e\n',
                                    Integrator_timeStep,
                                    Integrator_simTime,
                                    avgTemperature,
                                    avgPressure)];
end

local -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task createDir(_ : int, dirname : regentlib.string)
   UTIL.createDir(dirname)
   return _
end

--------------------------------------------------------------------------------
-- EXPORTED SYMBOLS
--------------------------------------------------------------------------------
Exports.Console_WriteHeader = Console_WriteHeader
Exports.Console_Write       = Console_Write
Exports.Console_WriteFooter = Console_WriteFooter

Exports.Probe_WriteHeader   = Probe_WriteHeader
Exports.Probe_Write         = Probe_Write

Exports.createDir           = createDir

return Exports end

