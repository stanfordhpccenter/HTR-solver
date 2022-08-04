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

return function(SCHEMA, MIX, TYPES, Fluid_columns,
                zones_partitions, ghost_partitions) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c

-------------------------------------------------------------------------------
-- SHOCK-SENSOR ROUTINES
-------------------------------------------------------------------------------
local extern task UpdateDucrosSensor(Ghost : region(ispace(int3d), Fluid_columns),
                                     Fluid : region(ispace(int3d), Fluid_columns),
                                     Fluid_bounds : rect3d,
                                     vorticityScale : double)
where
   reads(Ghost.velocity),
   reads(Fluid.{nType_x, nType_y, nType_z}),
   reads(Fluid.{dcsi_d, deta_d, dzet_d}),
   writes(Fluid.DucrosSensor)
end
UpdateDucrosSensor:set_task_id(TYPES.TID_UpdateDucrosSensor)

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
                                 Fluid_bounds : rect3d)
   where
      reads(Ghost.Conserved),
      reads(Ghost.DucrosSensor),
      reads(Fluid.[nType]),
      writes(Fluid.[shockSensor])
   end
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
   var {p_All, p_x_faces, p_y_faces, p_z_faces } = Fluid_Zones;
   var {p_XSensorGhosts, p_YSensorGhosts, p_ZSensorGhosts, p_GradientGhosts} = Fluid_Ghost;

   -- Compute the sensors only if we are running with the hybrid scheme
   if (config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_Hybrid) then
      __demand(__index_launch)
      for c in tiles do
         UpdateDucrosSensor(p_GradientGhosts[c], p_All[c],
                            Fluid.bounds, config.Integrator.EulerScheme.u.Hybrid.vorticityScale)
      end

      __demand(__index_launch)
      for c in tiles do
         [mkUpdateShockSensor("x")](p_XSensorGhosts[c], p_x_faces[c], Fluid.bounds)
      end
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateShockSensor("y")](p_YSensorGhosts[c], p_y_faces[c], Fluid.bounds)
      end
      __demand(__index_launch)
      for c in tiles do
         [mkUpdateShockSensor("z")](p_ZSensorGhosts[c], p_z_faces[c], Fluid.bounds)
      end
   end
end

return Exports end

