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

return function(SCHEMA, MIX, Fluid_columns, zones_partitions, ghost_partitions) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local TYPES = terralib.includec("prometeo_types.h", {"-DEOS="..os.getenv("EOS")})

-------------------------------------------------------------------------------
-- SHOCK-SENSOR ROUTINES
-------------------------------------------------------------------------------
local mkUpdateShockSensor = terralib.memoize(function(dir)
   local UpdateShockSensor

   local nType
   local shockSensor
   if (dir == "x") then
      nType = "nType_x"
      shockSensor = "shockSensorX"
   elseif (dir == "y") then
      nType = "nType_y"
      shockSensor = "shockSensorY"
   elseif (dir == "z") then
      nType = "nType_z"
      shockSensor = "shockSensorZ"
   else assert(false) end
   extern task UpdateShockSensor(Ghost : region(ispace(int3d), Fluid_columns),
                                 Fluid : region(ispace(int3d), Fluid_columns),
                                 ModCells : region(ispace(int3d), Fluid_columns),
                                 Fluid_bounds : rect3d,
                                 vorticityScale : double)
   where
      reads(Ghost.Conserved),
      reads(Ghost.{velocityGradientX, velocityGradientY, velocityGradientZ}),
      reads(Fluid.[nType]),
      writes(Fluid.[shockSensor])
   end
   UpdateShockSensor:set_calling_convention(regentlib.convention.manual())
   if     (dir == "x") then
      UpdateShockSensor:set_task_id(TYPES.TID_UpdateShockSensorX)
   elseif (dir == "y") then
      UpdateShockSensor:set_task_id(TYPES.TID_UpdateShockSensorY)
   elseif (dir == "z") then
      UpdateShockSensor:set_task_id(TYPES.TID_UpdateShockSensorZ)
   end
   return UpdateShockSensor
end)

__demand(__inline)
task Exports.UpdateShockSensors(Fluid : region(ispace(int3d), Fluid_columns),
                                tiles : ispace(int3d),
                                Fluid_Zones : zones_partitions(Fluid, tiles),
                                Fluid_Ghost : ghost_partitions(Fluid, tiles),
                                Mix : MIX.Mixture,
                                config : SCHEMA.Config)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All, p_Interior, p_AllBCs,
        p_x_faces, p_y_faces, p_z_faces } = Fluid_Zones;
   var {p_XEulerGhosts,  p_YEulerGhosts,  p_ZEulerGhosts,
        p_XSensorGhosts, p_YSensorGhosts, p_ZSensorGhosts} = Fluid_Ghost;

   -- Compute the sensors only if we are running with the hybrid scheme
   if config.Integrator.hybridScheme then
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateShockSensor("x")](p_XEulerGhosts[c], p_All[c], p_x_faces[c],
                                    Fluid.bounds, config.Integrator.vorticityScale)
      end
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateShockSensor("y")](p_YEulerGhosts[c], p_All[c], p_y_faces[c],
                                    Fluid.bounds, config.Integrator.vorticityScale)
      end
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateShockSensor("z")](p_ZEulerGhosts[c], p_All[c], p_z_faces[c],
                                    Fluid.bounds, config.Integrator.vorticityScale)
      end
   end
end

return Exports end

