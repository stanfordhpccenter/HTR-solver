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

return function(SCHEMA, MIX, Fluid_columns, bBoxType) local Exports = {}

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local sin  = regentlib.sin(double)
local cos  = regentlib.cos(double)
local sinh = regentlib.sinh(double)
local cosh = regentlib.cosh(double)
local exp  = regentlib.exp(double)
local pow  = regentlib.pow(double)

local UTIL  = require "util"
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local PI = CONST.PI
local Primitives = CONST.Primitives

local rand = UTIL.mkRand()

-------------------------------------------------------------------------------
-- INITIALIZATION ROUTINES
-------------------------------------------------------------------------------

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeUniform(Fluid : region(ispace(int3d), Fluid_columns),
                               initPressure : double,
                               initTemperature : double,
                               initVelocity : double[3],
                               initMolarFracs : double[nSpec])
where
   writes(Fluid.pressure),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in Fluid do
      var i = 0
      Fluid[c].pressure    = initPressure
      Fluid[c].temperature = initTemperature
      Fluid[c].velocity    = initVelocity
      Fluid[c].MolarFracs  = initMolarFracs
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeRandom(Fluid : region(ispace(int3d), Fluid_columns),
                              initPressure : double,
                              initTemperature : double,
                              magnitude : double,
                              initMolarFracs : double[nSpec])
where
   writes(Fluid.[Primitives])
do
   var randSeed = C.legion_get_current_time_in_nanos()
   var xsize = Fluid.bounds.hi.x - Fluid.bounds.lo.x + 1
   var ysize = Fluid.bounds.hi.y - Fluid.bounds.lo.y + 1
   __demand(__openmp)
   for c in Fluid do
      var ctr1 = 3*(c.x + xsize*(c.y + ysize*c.z))
      var ctr2 = 3*(c.x + xsize*(c.y + ysize*c.z)) + 1
      var ctr3 = 3*(c.x + xsize*(c.y + ysize*c.z)) + 2
      Fluid[c].MolarFracs  = initMolarFracs
      Fluid[c].pressure    = initPressure
      Fluid[c].temperature = initTemperature
      Fluid[c].velocity = array(2 * (rand(randSeed, ctr1) - 0.5) * magnitude,
                                2 * (rand(randSeed, ctr2) - 0.5) * magnitude,
                                2 * (rand(randSeed, ctr3) - 0.5) * magnitude)
   end
end

__demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Exports.InitializeTaylorGreen2D(Fluid : region(ispace(int3d), Fluid_columns),
                             taylorGreenPressure : double,
                             taylorGreenTemperature : double,
                             taylorGreenVelocity : double,
                             taylorGreenMolarFracs : double[nSpec],
                             mix : MIX.Mixture)
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   MIX.ClipYi(taylorGreenMolarFracs, &mix)
   var MixW = MIX.GetMolarWeightFromXi(taylorGreenMolarFracs, &mix)
   var taylorGreenDensity = MIX.GetRho(taylorGreenPressure, taylorGreenTemperature, MixW, &mix)
   for c in Fluid do
      var xy = Fluid[c].centerCoordinates
      Fluid[c].temperature = taylorGreenTemperature
      Fluid[c].MolarFracs = taylorGreenMolarFracs
      Fluid[c].velocity = MACRO.vs_mul([double[3]](array(((sin(xy[0])*cos(xy[1]))), (((-cos(xy[0]))*sin(xy[1]))), 0.0)), taylorGreenVelocity)
      var factor = (cos((2.0*xy[0]))+cos((2.0*xy[1])))
      var pressure = (taylorGreenPressure+(((taylorGreenDensity*pow(taylorGreenVelocity, 2.0))/4.0)*factor))
      Fluid[c].pressure = pressure
      Fluid[c].temperature = MIX.GetTFromRhoAndP(taylorGreenDensity, MixW, pressure, &mix)
   end
end

__demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task Exports.InitializeTaylorGreen3D(Fluid : region(ispace(int3d), Fluid_columns),
                             taylorGreenPressure : double,
                             taylorGreenTemperature : double,
                             taylorGreenVelocity : double,
                             taylorGreenMolarFracs : double[nSpec],
                             mix : MIX.Mixture)
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   MIX.ClipYi(taylorGreenMolarFracs, &mix)
   var MixW = MIX.GetMolarWeightFromXi(taylorGreenMolarFracs, &mix)
   var taylorGreenDensity = MIX.GetRho(taylorGreenPressure, taylorGreenTemperature, MixW, &mix)
   for c in Fluid do
      var xy = Fluid[c].centerCoordinates
      Fluid[c].temperature = taylorGreenDensity
      Fluid[c].MolarFracs = taylorGreenMolarFracs
      Fluid[c].velocity = MACRO.vs_mul([double[3]](array(((sin(xy[0])*cos(xy[1]))*cos(xy[2])), (((-cos(xy[0]))*sin(xy[1]))*cos(xy[2])), 0.0)), taylorGreenVelocity)
      var factorA = (cos((2.0*xy[2]))+2.0)
      var factorB = (cos((2.0*xy[0]))+cos((2.0*xy[1])))
      var pressure = (taylorGreenPressure+((((taylorGreenDensity*pow(taylorGreenVelocity, 2.0))/16.0)*factorA)*factorB))
      Fluid[c].pressure = pressure
      Fluid[c].temperature = MIX.GetTFromRhoAndP(taylorGreenDensity, MixW, pressure, &mix)
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializePerturbed(Fluid : region(ispace(int3d), Fluid_columns),
                                 initPressure : double,
                                 initTemperature : double,
                                 initVelocity : double[3],
                                 initMolarFracs : double[nSpec])
where
   writes(Fluid.[Primitives])
do
   var randSeed = C.legion_get_current_time_in_nanos()
   var xsize = Fluid.bounds.hi.x - Fluid.bounds.lo.x + 1
   var ysize = Fluid.bounds.hi.y - Fluid.bounds.lo.y + 1
   __demand(__openmp)
   for c in Fluid do
      var ctr1 = 3*(c.x + xsize*(c.y + ysize*c.z))
      var ctr2 = 3*(c.x + xsize*(c.y + ysize*c.z)) + 1
      var ctr3 = 3*(c.x + xsize*(c.y + ysize*c.z)) + 2
      Fluid[c].MolarFracs  = initMolarFracs
      Fluid[c].pressure    = initPressure
      Fluid[c].temperature = initTemperature
      Fluid[c].velocity = array(initVelocity[0] + (rand(randSeed, ctr1)-0.5)*10.0,
                                initVelocity[1] + (rand(randSeed, ctr2)-0.5)*10.0,
                                initVelocity[2] + (rand(randSeed, ctr3)-0.5)*10.0)
   end
end

-- Test 1 in Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2013)
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeRiemannTestOne(Fluid : region(ispace(int3d), Fluid_columns),
                                      initMolarFracs : double[nSpec])
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0]
      Fluid[c].MolarFracs = initMolarFracs
      if (x < 0.3) then
         Fluid[c].velocity = array(0.75, 0.0, 0.0)
         Fluid[c].pressure = 1.0
         Fluid[c].temperature = 1.0
      else
         Fluid[c].velocity = array(0.0, 0.0, 0.0)
         Fluid[c].pressure = 0.1
         Fluid[c].temperature = 0.8
      end
   end
end

-- Test 4 in Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2013)
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeRiemannTestTwo(Fluid : region(ispace(int3d), Fluid_columns),
                                      initMolarFracs : double[nSpec])
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0]
      Fluid[c].MolarFracs = initMolarFracs
      if (x < 0.4) then
         Fluid[c].velocity = array(19.5975, 0.0, 0.0)
         Fluid[c].pressure = 460.894
         Fluid[c].temperature = 76.8253978837
      else
         Fluid[c].velocity = array(-6.19633, 0.0, 0.0)
         Fluid[c].pressure = 46.0950
         Fluid[c].temperature = 7.6922178352
      end
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeSodProblem(Fluid : region(ispace(int3d), Fluid_columns),
                                  initMolarFracs : double[nSpec])
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0]
      Fluid[c].MolarFracs = initMolarFracs
      if (x < 0.5) then
         Fluid[c].velocity = array(0.0, 0.0, 0.0)
         Fluid[c].pressure = 1.0
         Fluid[c].temperature = 1.0
      else
         Fluid[c].velocity = array(0.0, 0.0, 0.0)
         Fluid[c].pressure = 0.1
         Fluid[c].temperature = 0.8
      end
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeLaxProblem(Fluid : region(ispace(int3d), Fluid_columns),
                                  initMolarFracs : double[nSpec])
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0]
      Fluid[c].MolarFracs = initMolarFracs
      if (x < 0.5) then
         Fluid[c].velocity = array(0.698, 0.0, 0.0)
         Fluid[c].pressure = 3.528
         Fluid[c].temperature = 7.92808988764
      else
         Fluid[c].velocity = array(0.0, 0.0, 0.0)
         Fluid[c].pressure = 0.5710
         Fluid[c].temperature = 1.142
      end
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeShuOsherProblem(Fluid : region(ispace(int3d), Fluid_columns),
                                       initMolarFracs : double[nSpec])
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0]
      Fluid[c].MolarFracs = initMolarFracs
      if (x < 1.0) then
         Fluid[c].velocity = array(2.629, 0.0, 0.0)
         Fluid[c].pressure = 10.333
         Fluid[c].temperature = 2.67902514908
      else
         Fluid[c].velocity = array(0.0, 0.0, 0.0)
         Fluid[c].pressure = 1.0
         Fluid[c].temperature = 1.0/(1.0 + 0.2*sin(5*(x-5)))
      end
   end
end

__demand(__leaf, __cuda) -- MANUALLY PARALLELIZED
task Exports.InitializeVortexAdvection2D(Fluid : region(ispace(int3d), Fluid_columns),
                                         VortexPressure : double,
                                         VortexTemperature : double,
                                         VortexXVelocity : double,
                                         VortexYVelocity : double,
                                         VortexMolarFracs : double[nSpec],
                                         mix : MIX.Mixture)
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   var MixW = MIX.GetMolarWeightFromXi(VortexMolarFracs, &mix)
   var Yi : double[nSpec]; MIX.GetMassFractions(Yi, MixW, VortexMolarFracs, &mix)
   var gamma = MIX.GetGamma(VortexTemperature, MixW, Yi, &mix)
   __demand(__openmp)
   for c in Fluid do
      var Beta = 5.0
      var x0 = 0.0
      var y0 = 0.0
      var xy = Fluid[c].centerCoordinates
      var rx = xy[0] - x0
      var ry = xy[1] - y0
      var r2 = rx*rx + ry*ry

      var T = VortexTemperature*(1.0 - (gamma - 1.0)*Beta*Beta/(8*PI*PI*gamma)*exp(1.0 - r2))
      var P = VortexPressure*pow(T, gamma/(gamma - 1.0))

      Fluid[c].pressure = P
      Fluid[c].temperature = T
      Fluid[c].MolarFracs = VortexMolarFracs
      Fluid[c].velocity = MACRO.vv_add(array((-Beta/(2.0*PI)*exp(0.5*(1.0 - r2))*(ry)),
                                             ( Beta/(2.0*PI)*exp(0.5*(1.0 - r2))*(rx)), 0.0),
                                       array(VortexXVelocity, VortexYVelocity, 0.0))
   end
end

__demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
task Exports.InitializeGrossmanCinnellaProblem(Fluid : region(ispace(int3d), Fluid_columns),
                                               mix : MIX.Mixture)
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   -- Left-hand side mixture
   var leftMix : SCHEMA.Mixture
   leftMix.Species.length = 5
   C.snprintf([&int8](leftMix.Species.values[0].Name), 10, "N2")
   leftMix.Species.values[0].MolarFrac = 1.279631e-02

   C.snprintf([&int8](leftMix.Species.values[1].Name), 10, "O2")
   leftMix.Species.values[1].MolarFrac = 3.695112e-06

   C.snprintf([&int8](leftMix.Species.values[2].Name), 10, "NO")
   leftMix.Species.values[2].MolarFrac = 2.694521e-04

   C.snprintf([&int8](leftMix.Species.values[3].Name), 10, "N")
   leftMix.Species.values[3].MolarFrac = 7.743854e-01

   C.snprintf([&int8](leftMix.Species.values[4].Name), 10, "O")
   leftMix.Species.values[4].MolarFrac = 2.125451e-01
   var LeftMolarFracs = MIX.ParseConfigMixture(leftMix, mix)

   -- Right-hand side mixture
   var rightMix : SCHEMA.Mixture
   rightMix.Species.length = 2
   C.snprintf([&int8](rightMix.Species.values[0].Name), 10, "N2")
   rightMix.Species.values[0].MolarFrac = 0.790000e+00

   C.snprintf([&int8](rightMix.Species.values[1].Name), 10, "O2")
   rightMix.Species.values[1].MolarFrac = 0.210000e+00
   var RightMolarFracs = MIX.ParseConfigMixture(rightMix, mix)

   __demand(__openmp)
   for c in Fluid do
       var x = Fluid[c].centerCoordinates[0]
       if (x < 0.5) then
          Fluid[c].MolarFracs = LeftMolarFracs
          Fluid[c].velocity = array(0.0, 0.0, 0.0)
          Fluid[c].pressure = 1.95256e1
          Fluid[c].temperature = 30.0
       else
          Fluid[c].MolarFracs = RightMolarFracs
          Fluid[c].velocity = array(0.0, 0.0, 0.0)
          Fluid[c].pressure = 1.0
          Fluid[c].temperature = 1.0
       end
   end
end

__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeChannelFlow(Fluid : region(ispace(int3d), Fluid_columns),
                                   bulkPressure : double,
                                   bulkTemperature : double,
                                   bulkVelocity : double,
                                   StreaksIntensity : double,
                                   RandomIntensity : double,
                                   initMolarFracs : double[nSpec],
                                   mix : MIX.Mixture,
                                   bBox : bBoxType)
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.[Primitives])
do
   -- Initializes the channel with a streamwise velocity profile ~y^4
   -- streaks and random noise are used for the spanwise and wall-normal directions
   -- the fluid composition is uniform
   MIX.ClipYi(initMolarFracs, &mix)

   var randSeed = C.legion_get_current_time_in_nanos()
   var xsize = Fluid.bounds.hi.x - Fluid.bounds.lo.x + 1
   var ysize = Fluid.bounds.hi.y - Fluid.bounds.lo.y + 1

   var Grid_yOrigin = bBox.v0[1]
   var Grid_zOrigin = bBox.v0[2]
   var Grid_yWidth  = bBox.v3[1] - bBox.v0[1]
   var Grid_zWidth  = bBox.v4[2] - bBox.v0[2]

   __demand(__openmp)
   for c in Fluid do
      Fluid[c].pressure = bulkPressure
      Fluid[c].temperature = bulkTemperature
      Fluid[c].MolarFracs = initMolarFracs
      var xyz = Fluid[c].centerCoordinates
      -- normalize wall-normal and spanwise coordinates
      xyz[1] = (xyz[1] - Grid_yOrigin - Grid_yWidth*0.5)*2.0/Grid_yWidth
      xyz[2] = (xyz[2] - Grid_zOrigin                  )*2.0/Grid_zWidth*PI

      var velocity = array (0.0, 0.0, 0.0)
      -- define an offset from the walls for the streaks
      var off = 0.05
      if xyz[1] > 0.0 then off *= -1.0 end

      velocity[0] = 1.25*bulkVelocity*(1.0 - pow(xyz[1], 4))
      velocity[1] = StreaksIntensity*1.25*bulkVelocity*4.0*0.9*sin(2.0*xyz[2])/
         (cosh(0.9*(xyz[1] + off))*(0.9*0.9*pow(cos(2.0*xyz[2]), 2)/pow(cosh(0.9*(xyz[1] + off)), 2) - 1.0))
      velocity[2] = StreaksIntensity*1.25*bulkVelocity*2.0*0.9*0.9*sinh(0.9*(xyz[1] + off))*cos(2.0*xyz[2])/
         (pow(cosh(0.9*(xyz[1] + off)), 2)*(0.9*0.9*pow(cos(2.0 * xyz[2]), 2)/pow(cosh(0.9 * (xyz[1] + off)), 2) - 1.0))
      -- add the random noise
      var ctr1 = 3*(c.x + xsize*(c.y + ysize*c.z))
      var ctr2 = 3*(c.x + xsize*(c.y + ysize*c.z)) + 1
      var ctr3 = 3*(c.x + xsize*(c.y + ysize*c.z)) + 2
      velocity[0] += RandomIntensity*bulkVelocity*(rand(randSeed, ctr1)-0.5)
      velocity[1] += RandomIntensity*bulkVelocity*(rand(randSeed, ctr2)-0.5)
      velocity[2] += RandomIntensity*bulkVelocity*(rand(randSeed, ctr3)-0.5)
      Fluid[c].velocity = velocity
   end
end

return Exports end

