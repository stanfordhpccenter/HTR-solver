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

local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c

-------------------------------------------------------------------------------
-- CHECK IF WE ARE ON MACOS
-------------------------------------------------------------------------------
function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end
Exports.DARWIN = (os.capture("uname") == "Darwin")

-------------------------------------------------------------------------------
-- MACROS
-------------------------------------------------------------------------------

__demand(__inline)
task Exports.is_xNegGhost(c : int3d, Grid_xBnum : int)
  return c.x < Grid_xBnum
end

__demand(__inline)
task Exports.is_yNegGhost(c : int3d, Grid_yBnum : int)
  return c.y < Grid_yBnum
end

__demand(__inline)
task Exports.is_zNegGhost(c : int3d, Grid_zBnum : int)
  return c.z < Grid_zBnum
end

__demand(__inline)
task Exports.is_xPosGhost(c : int3d, Grid_xBnum : int, Grid_xNum : int)
  return c.x >= Grid_xNum + Grid_xBnum
end

__demand(__inline)
task Exports.is_yPosGhost(c : int3d, Grid_yBnum : int, Grid_yNum : int)
  return c.y >= Grid_yNum + Grid_yBnum
end

__demand(__inline)
task Exports.is_zPosGhost(c : int3d, Grid_zBnum : int, Grid_zNum : int)
  return c.z >= Grid_zNum + Grid_zBnum
end

__demand(__inline)
task Exports.in_interior(c : int3d,
                 Grid_xBnum : int, Grid_xNum : int,
                 Grid_yBnum : int, Grid_yNum : int,
                 Grid_zBnum : int, Grid_zNum : int)
  return
    Grid_xBnum <= c.x and c.x < Grid_xNum + Grid_xBnum and
    Grid_yBnum <= c.y and c.y < Grid_yNum + Grid_yBnum and
    Grid_zBnum <= c.z and c.z < Grid_zNum + Grid_zBnum
end

__demand(__inline)
task Exports.vs_mul(a : double[3], b : double)
  return array(a[0] * b, a[1] * b, a[2] * b)
end

__demand(__inline)
task Exports.vs_div(a : double[3], b : double)
  return array(a[0] / b, a[1] / b, a[2] / b)
end

__demand(__inline)
task Exports.dot(a : double[3], b : double[3])
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
end

__demand(__inline)
task Exports.vv_add(a : double[3], b : double[3])
  return array(a[0] + b[0], a[1] + b[1], a[2] + b[2])
end

__demand(__inline)
task Exports.vv_sub(a : double[3], b : double[3])
  return array(a[0] - b[0], a[1] - b[1], a[2] - b[2])
end

__demand(__inline)
task Exports.vv_mul(a : double[3], b : double[3])
  return array(a[0] * b[0], a[1] * b[1], a[2] * b[2])
end

__demand(__inline)
task Exports.vv_div(a : double[3], b : double[3])
  return array(a[0] / b[0], a[1] / b[1], a[2] / b[2])
end

return Exports

