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

return function(SCHEMA, MIX, CHEM, Fluid_columns, zones_partitions, DEBUG_OUTPUT) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local MAPPER = terralib.includec("prometeo_mapper.h")
local UTIL = require 'util-desugared'
local MATH = require 'math_utils'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local sin  = regentlib.sin(double)
local cos  = regentlib.cos(double)
local tan  = regentlib.tan(double)
local exp  = regentlib.exp(double)
local pow  = regentlib.pow(double)
local asin = regentlib.asin(double)
local tanh = regentlib.tanh(double)
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local atan = regentlib.atan(double)

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local PI           = CONST.PI
local Primitives   = CONST.Primitives
local ProfilesVars = CONST.ProfilesVars
local RecycleVars  = CONST.RecycleVars

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

local struct IncomingShockParams {
   -- Index where the shock will be injected
   iShock  : int;
   -- Constant mixture composition
   MolarFracs : double[nSpec];
   -- Primitive variables upstream of the shock
   pressure0    : double;
   temperature0 : double;
   velocity0    : double[3];
   -- Primitive variables downstream of the shock
   pressure1    : double;
   temperature1 : double;
   velocity1    : double[3];
}

local struct RecycleAverageType {
   -- Average weight
   w : double;
   -- Distance from the wall
   y : double;
   -- Properties
   rho  : double;
   -- Primitive variables
   temperature : double;
   MolarFracs  : double[nSpec];
   velocity    : double[3];
}

-- Load fast interpolation tool
local FIData, FIType,
      FIInitData, FIInitRegion,
      FIFindIndex, FIGetWeight = unpack(MATH.mkFastInterp(RecycleAverageType, "y"))

-- Boundary layer data
local struct BLDataType {
   Uinf : double;
   aVD : double;
   bVD : double;
   QVD : double;
   Ueq : double;
}

-- Rescaling data
local struct RescalingDataType {
   -- Data for outer scaling
   delta99VD : double;
   -- Data for inner scaling
   rhow : double;
   uTau : double;
   deltaNu : double;
}

local fspace RecycleRescalingParams(Fluid : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   -- Recycling plane
   BCPlane      : partition(disjoint, Fluid, tiles),
   RecyclePlane : partition(disjoint, Fluid, tiles),
   -- Interpolation partition on boundary
   interp_tiles : ispace(int1d),
   BC_interp : partition(disjoint, Fluid, interp_tiles),
   -- Fast interpolation tools
   FIdata : FIData,
   -- Boundary layer data
   BLData : BLDataType,
   -- Rescaling data
   RescalingData : RescalingDataType
}

local fspace BCParamsStruct(Fluid : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   IncomingShock : IncomingShockParams,
   RecycleRescaling : RecycleRescalingParams(Fluid, tiles)
}

Exports.RecycleAverageType = RecycleAverageType
Exports.RecycleAverageFIType = FIType
Exports.BCParamsStruct = BCParamsStruct

Exports.BCDataList = {
   readProfiles = regentlib.newsymbol(bool),
   ProfilesDir = regentlib.newsymbol(&int8),
   BCParams = regentlib.newsymbol(),
   RecycleAverage = regentlib.newsymbol(),
   RecycleAverageFI = regentlib.newsymbol()
}

-------------------------------------------------------------------------------
-- RANKINE-HUGONIOT FOR OBLIQUE SHOCKS
-------------------------------------------------------------------------------

-- Computes the angle of the velocity vector in the x-y plane
local __demand(__inline)
task getThetaFromU(u : double[3])
   return atan(u[1]/u[0])
end

-- Computes the angle of the shock-normal velocity
local __demand(__inline)
task getUnFromU(U : double[3], beta : double, theta : double)
   return sqrt(U[0]*U[0] + U[1]*U[1])*sin(beta+theta)
end

-- Computes the angle of the shock-normal velocity
local __demand(__inline)
task getUFromUn(Un : double, beta : double, theta : double)
   var Umag = Un/sin(beta+theta)
   return array(Umag*cos(theta), Umag*sin(theta), 0.0)
end

-- Computes the post-shock density
local __demand(__inline)
task RHgetrho1(nu : double, rho0 : double) return rho0/nu end

-- Computes the post-shock normal velocity
local __demand(__inline)
task RHgetUn1(nu : double, Un0 : double) return Un0*nu end

-- Computes the post-shock pressure
local __demand(__inline)
task RHgetP1(nu : double, p0 : double , Un0 : double, rho0 : double)
   return p0 + rho0*Un0*Un0*(1.0 - nu)
end

-- Computes the post-shock static enthalpy
local __demand(__inline)
task RHgetH1(nu : double, h0 : double , Un0 : double)
   return h0 + 0.5*Un0*Un0*(1.0 - nu*nu)
end

-- Computes the post-shock velocity angle in the x-y plane
local __demand(__inline)
task RHgetTheta1(nu : double, beta : double, theta0 : double)
   var betaEff = beta+theta0
   var tanB = tan(betaEff)
   return atan(tanB*(nu-1.0)/(1.0 + nu*tanB*tanB))+theta0
end

-- This assumes constant composition across the shock
local __demand(__inline)
task RHresidual(nu : double,
                un0 : double, rho0 : double, P0 : double, h0 : double,
                Yi : double[nSpec], MixW : double, Mix : MIX.Mixture)
   var un1 = RHgetUn1(nu, un0)
   var rho1 = RHgetrho1(nu, rho0)
   var P1 = RHgetP1(nu, P0, un0, rho0)
   var T1 = MIX.GetTFromRhoAndP(rho1, MixW, P1)
   return RHgetH1(nu, h0, un0) - MIX.GetEnthalpy(T1, Yi, Mix)
end

local __demand(__inline)
task InitIncomingShockParams(config : SCHEMA.Config, Mix : MIX.Mixture)

   var input = config.BC.yBCRight.u.IncomingShock
   var params : IncomingShockParams

   -- Index where the shock will be injected
   params.iShock = input.iShock

   -- Shock angle
   input.beta *= PI/180
   params.MolarFracs = CHEM.ParseConfigMixture(input.Mixture, Mix)

   -- Primitive variables upstream of the shock
   params.pressure0    = input.pressure0
   params.temperature0 = input.temperature0
   params.velocity0    = input.velocity0

   var MixW = MIX.GetMolarWeightFromXi(params.MolarFracs, Mix)
   var Yi = MIX.GetMassFractions(MixW, params.MolarFracs, Mix)
   var h0 = MIX.GetEnthalpy(params.temperature0, Yi, Mix)
   var rho0 = MIX.GetRho(params.pressure0, params.temperature0, MixW, Mix)
   var theta0 = getThetaFromU(params.velocity0)
   var un0 = getUnFromU(params.velocity0, input.beta, theta0)

   -- Compute primitive variables downstream of the shock
   -- Use a Newton method
   var i = 0
   var nu = 1e-3
   while true do
      var f  = RHresidual(nu,   un0, rho0, params.pressure0, h0, Yi, MixW, Mix)
      var h = nu*1e-3
      var f1 = RHresidual(nu+h, un0, rho0, params.pressure0, h0, Yi, MixW, Mix)
      var fp = (f1-f)/h
      nu -= f/fp
      if (fabs(f/fp) < 1e-8) then break end
      regentlib.assert(i ~= 100, "Too many iterations in Rankine-Hugoniot non-linear solver")
      i += 1
   end

   var rho1 = RHgetrho1(nu, rho0)
   var un1 = RHgetUn1(nu, un0)
   params.pressure1 = RHgetP1(nu, params.pressure0, un0, rho0)
   params.temperature1 = MIX.GetTFromRhoAndP(rho1, MixW, params.pressure1)
   var theta1 = RHgetTheta1(nu, input.beta, theta0)
   var u1 = getUFromUn(un1, input.beta, theta1)
   params.velocity1 = array(u1[0], u1[1], params.velocity0[2])

   return params
end

-------------------------------------------------------------------------------
-- RECYCLE-RESCALING FUNCTIONS
-------------------------------------------------------------------------------
local __demand(__inline)
task initRecycleRescalingParams(Fluid : region(ispace(int3d), Fluid_columns),
                                tiles : ispace(int3d),
                                p_All : partition(disjoint, Fluid, tiles),
                                BC    : partition(disjoint, Fluid, ispace(int1d)),
                                p_BC  : cross_product(p_All, BC),
                                BCinput : SCHEMA.FlowBC,
                                Grid_xBnum : int32, Grid_yBnum : int32, Grid_zBnum : int32)

   regentlib.assert(BCinput.type == SCHEMA.FlowBC_RecycleRescaling,
      "Wrong BC got into RecycleRescaling data structure initialization")

   -- Extract Recycle and BC plane from fluid
   var BCcoloring = regentlib.c.legion_domain_point_coloring_create()
   var PLcoloring = regentlib.c.legion_domain_point_coloring_create()
   for c in tiles do
      var rect = p_BC[c][0].bounds
      -- Include ghost
      if c.y == 0                 then rect.lo.y = Fluid.bounds.lo.y end
      if c.z == 0                 then rect.lo.z = Fluid.bounds.lo.z end
      if c.y == tiles.bounds.hi.y then rect.hi.y = Fluid.bounds.hi.y end
      if c.z == tiles.bounds.hi.z then rect.hi.z = Fluid.bounds.hi.z end
      regentlib.c.legion_domain_point_coloring_color_domain(BCcoloring, c, rect)
      rect = rect + int3d{BCinput.u.RecycleRescaling.iRecycle, 0, 0}
      regentlib.c.legion_domain_point_coloring_color_domain(PLcoloring, c, rect)
   end
   var BCPlane = partition(disjoint, Fluid, BCcoloring, tiles);
   [UTIL.emitPartitionNameAttach(BCPlane, "BCPlane")];
   var RecyclePlane = partition(disjoint, Fluid, PLcoloring, tiles);
   [UTIL.emitPartitionNameAttach(RecyclePlane, "RecyclePlane")];

   regentlib.c.legion_domain_point_coloring_destroy(BCcoloring)
   regentlib.c.legion_domain_point_coloring_destroy(PLcoloring)

   -- Create tiles of the BC region that support the entire interpolation
   var interp_tiles = ispace(int1d, {tiles.bounds.hi.z + 1})
   var InterpColoring = regentlib.c.legion_multi_domain_point_coloring_create()
   for c in tiles do
      var c_interp = int1d{c.z}
      var rect = p_BC[c][0].bounds
      -- Include ghost
      if c.y == 0                 then rect.lo.y = Fluid.bounds.lo.y end
      if c.y == tiles.bounds.hi.y then rect.hi.y = Fluid.bounds.hi.y end
      regentlib.c.legion_multi_domain_point_coloring_color_domain(InterpColoring, c_interp, rect)
   end
   var BC_interp = partition(disjoint, Fluid, InterpColoring, interp_tiles);
   [UTIL.emitPartitionNameAttach(BC_interp, "BC_interp")];

   regentlib.c.legion_multi_domain_point_coloring_destroy(InterpColoring)

   -- Create a dummy structures to be filled in the initialization phase
   var dummy  : FIData
   var BLdata : BLDataType
   var Rdata  : RescalingDataType

   return [RecycleRescalingParams(Fluid, tiles)] {
      BCPlane = BCPlane,
      RecyclePlane = RecyclePlane,
      interp_tiles = interp_tiles,
      BC_interp = BC_interp,
      FIdata = dummy,
      BLData = BLdata,
      RescalingData = Rdata
   }
end

local mkInitializeRecycleAverage = terralib.memoize(function(init)
   local InitializeRecycleAverage
   if (init == "all") then
      __demand(__inline)
      task InitializeRecycleAverage(avg : region(ispace(int1d), RecycleAverageType))
      where
         writes(avg)
      do
         fill(avg.w, 0.0)
         fill(avg.y, 0.0)
         fill(avg.rho, 0.0)
         fill(avg.temperature, 0.0)
         fill(avg.MolarFracs, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
         fill(avg.velocity, array(0.0, 0.0, 0.0))
      end
   elseif (init == "std") then
      __demand(__inline)
      task InitializeRecycleAverage(avg : region(ispace(int1d), RecycleAverageType))
      where
         writes(avg.{rho, temperature, MolarFracs, velocity})
      do
         fill(avg.rho, 0.0)
         fill(avg.temperature, 0.0)
         fill(avg.MolarFracs, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
         fill(avg.velocity, array(0.0, 0.0, 0.0))
      end
   else assert(false) end
   return InitializeRecycleAverage
end)

local __demand(__inline)
task getVDvelocity(u : double, data : BLDataType)
   var Uinf = data.Uinf
   var a = data.aVD
   var b = data.bVD
   var Q = data.QVD
   return Uinf/a*(asin((2.0*a*a*u/Uinf - b)/Q) + asin(b/Q))
end

local __demand(__leaf)
task InitializeBoundarLayerData(avg : region(ispace(int1d), RecycleAverageType),
                                mix : MIX.Mixture)
where
   reads(avg.{y, w, rho, temperature, MolarFracs, velocity})
do
   var data : BLDataType

   -- Conditions at the infinity
   var cinf = avg.bounds.hi
   var Uinf = avg[cinf].velocity[0]/avg[cinf].rho
   var Tinf = avg[cinf].temperature/avg[cinf].rho
   var rhoinf = avg[cinf].rho/avg[cinf].w
   var MolarFracsinf = avg[cinf].MolarFracs
   for i=0, nSpec do MolarFracsinf[i] /= avg[cinf].rho end
   var MixWinf = MIX.GetMolarWeightFromXi(MolarFracsinf, mix)
   var Yiinf = MIX.GetMassFractions(MixWinf, MolarFracsinf, mix)
   var gammainf = MIX.GetGamma(Tinf, MixWinf, Yiinf, mix)
   var Mainf = Uinf/MIX.GetSpeedOfSound(Tinf, gammainf, MixWinf, mix)
   var muinf  = MIX.GetViscosity(       Tinf, MolarFracsinf, mix)
   var laminf = MIX.GetHeatConductivity(Tinf, MolarFracsinf, mix)
   var cpinf  = MIX.GetHeatCapacity(Tinf, Yiinf, mix)

   -- Conditions at the wall
   var cw = avg.bounds.lo
   var Tw = avg[cw].temperature/avg[cw].rho
   var rhow = avg[cw].rho/avg[cw].w
   var MolarFracsw = avg[cw].MolarFracs
   for i=0, nSpec do MolarFracsinf[i] /= avg[cw].rho end
   var muw  = MIX.GetViscosity(Tw, MolarFracsw, mix)

   -- Useful numbers
   var Pr = cpinf*muinf/laminf
   var r = pow(Pr, 1.0/3.0)
   var Taw = Tinf*(1.0 + r*0.5*(gammainf - 1.0)*Mainf*Mainf)

   -- Data for outer scaling based on
   -- Van Driest equivalent velocity (see page 545 White's book)
   data.Uinf = Uinf
   data.aVD = sqrt(r*0.5*(gammainf - 1.0)*Mainf*Mainf*Tinf/Tw)
   data.bVD = Taw/Tw - 1.0
   data.QVD = sqrt(data.bVD*data.bVD + 4.0*data.aVD*data.aVD)
   data.Ueq = getVDvelocity(Uinf, data)

--   C.printf("Ma_inf = %10.5e Pr = %10.5e\n", Mainf, Pr)
--   C.printf("T_w = %10.5e Taw = %10.5e\n", Tw, Taw)
--   C.printf("Ueq = %10.5e, delta99VD = %10.5e\n", data.Ueq, data.delta99VD)
   return data
end

local __demand(__inline)
task getdelta99VD(avg : region(ispace(int1d), RecycleAverageType),
                  data : BLDataType)
where
   reads(avg.{y, rho, velocity})
do
   var c99 = int(avg.bounds.hi)
   var Ue = 0.99*data.Ueq
   __demand(__openmp)
   for c in avg do
      var uVD = getVDvelocity(avg[c].velocity[0]/avg[c].rho, data)
      if (uVD > Ue) then c99 min= int(c) end
   end
   c99 max= 1
   var cm1 = c99 - 1
   var up = getVDvelocity(avg[int1d(c99)].velocity[0]/avg[int1d(c99)].rho, data)
   var um = getVDvelocity(avg[int1d(cm1)].velocity[0]/avg[int1d(cm1)].rho, data)
   return avg[int1d(cm1)].y + (avg[int1d(c99)].y - avg[int1d(cm1)].y)*(Ue - um)/(up - um)
end

local __demand(__leaf)
task GetRescalingData(avg : region(ispace(int1d), RecycleAverageType),
                      BLdata : BLDataType,
                      mix : MIX.Mixture)
where
   reads(avg.{y, w, rho, temperature, MolarFracs, velocity})
do
   var data : RescalingDataType

   -- Conditions at the wall
   var c = avg.bounds.lo
   var T = avg[c].temperature/avg[c].rho
   var rho = avg[c].rho/avg[c].w
   var MolarFracs = avg[c].MolarFracs
   for i=0, nSpec do MolarFracs[i] /= avg[c].rho end
   var mu = MIX.GetViscosity(T, MolarFracs, mix)

   -- Data for outer scaling
   data.delta99VD = getdelta99VD(avg, BLdata)

   -- Data for inner scaling
   var cp1 = min(c+1, avg.bounds.hi)
   var tauw = mu*(avg[cp1].velocity[0]/avg[cp1].rho -
                  avg[c  ].velocity[0]/avg[c  ].rho)
   data.rhow = rho
   data.uTau = sqrt(fabs(tauw)/rho)
   data.deltaNu = mu/(rho*data.uTau)

   return data
end

local mkAddRecycleAverage = terralib.memoize(function(op)
   local AddRecycleAverage
   if (op == "pos") then
      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task AddRecycleAverage(plane : region(ispace(int3d), Fluid_columns),
                             avg   : region(ispace(int1d), RecycleAverageType))
      where
         reads(plane.centerCoordinates),
         reads(plane.cellWidth),
         reduces+(avg.{w, y})
      do
         __demand(__openmp)
         for c in plane do
            var c_avg = int1d(c.y)
            var vol = plane[c].cellWidth[0]*
                      plane[c].cellWidth[1]*
                      plane[c].cellWidth[2]
            avg[c_avg].w += vol
            avg[c_avg].y += plane[c].centerCoordinates[1]*vol
         end
      end
   elseif (op == "std") then
      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task AddRecycleAverage(plane : region(ispace(int3d), Fluid_columns),
                             avg   : region(ispace(int1d), RecycleAverageType))
      where
         reads(plane.cellWidth),
         reads(plane.rho),
         reads(plane.{temperature, MolarFracs, velocity}),
         reduces+(avg.rho),
         reduces+(avg.{temperature, MolarFracs, velocity})
      do
         __demand(__openmp)
         for c in plane do
            var c_avg = int1d(c.y)
            var vol = (plane[c].cellWidth[0]*
                       plane[c].cellWidth[1]*
                       plane[c].cellWidth[2])
            var rvol = vol*plane[c].rho
            avg[c_avg].rho         += plane[c].rho*vol
            avg[c_avg].temperature += plane[c].temperature*rvol
            avg[c_avg].MolarFracs  += plane[c].MolarFracs *[UTIL.mkArrayConstant(nSpec, rvol)]
            avg[c_avg].velocity    += plane[c].velocity   *[UTIL.mkArrayConstant(    3, rvol)]
         end
      end
   elseif (op == "BC") then
      __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
      task AddRecycleAverage(plane : region(ispace(int3d), Fluid_columns),
                             avg   : region(ispace(int1d), RecycleAverageType),
                             mix : MIX.Mixture,
                             Pbc : double)
      where
         reads(plane.cellWidth),
         reads(plane.[ProfilesVars]),
         reduces+(avg.rho),
         reduces+(avg.{temperature, MolarFracs, velocity})
      do
         __demand(__openmp)
         for c in plane do
            var c_avg = int1d(c.y)
            var vol = (plane[c].cellWidth[0]*
                       plane[c].cellWidth[1]*
                       plane[c].cellWidth[2])
            var MixW = MIX.GetMolarWeightFromXi(plane[c].MolarFracs_profile, mix)
            var rho = MIX.GetRho(Pbc, plane[c].temperature_profile, MixW, mix)
            var rvol = vol*rho
            avg[c_avg].rho         += rho*vol
            avg[c_avg].temperature += plane[c].temperature_profile*rvol
            avg[c_avg].MolarFracs  += plane[c].MolarFracs_profile *[UTIL.mkArrayConstant(nSpec, rvol)]
            avg[c_avg].velocity    += plane[c].velocity_profile   *[UTIL.mkArrayConstant(    3, rvol)]
         end
      end
   else assert(false) end
   return AddRecycleAverage
end)

local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task ComputeRecycleAveragePosition(avg : region(ispace(int1d), RecycleAverageType))
where
   reads(avg.w),
   reads writes(avg.y)
do
   __demand(__openmp)
   for c in avg do
      avg[c].y /= avg[c].w
   end
end

local mkUpdateRecycleVariables = terralib.memoize(function(dir)
   local UpdateRecycleVariables
   local idx
   if dir == "x" then
      idx = 0
--   elseif dir == "y" then
--      idx = 1
--   elseif dir == "z" then
--      idx = 2
   else assert(false) end

   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task UpdateRecycleVariables(Fluid    : region(ispace(int3d), Fluid_columns),
                               Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                               avg      : region(ispace(int1d), RecycleAverageType),
                               plane    : region(ispace(int3d), Fluid_columns),
                               iRecycle : int)
   where
      reads(plane.{temperature, MolarFracs, velocity}),
      reads(avg.{rho, temperature, MolarFracs, velocity}),
      writes(Fluid.[RecycleVars])
   do
      var BC   = Fluid_BC[0]
      __demand(__openmp)
      for c in BC do
         var cPlane = int3d({iRecycle, c.y, c.z})
         var cAvg = int1d(c.y)
         -- Complete averaging over the plane
         var iw = 1.0/avg[cAvg].rho
         var AvgTemperature = avg[cAvg].temperature*iw
         var AvgMolarFracs  = avg[cAvg].MolarFracs
         var AvgVelocity    = avg[cAvg].velocity
         for i=0, nSpec do AvgMolarFracs[i] *= iw end
         for i=0,     3 do AvgVelocity[i]   *= iw end
         -- Compute the fluctuations
         var MolarFracs_recycle = plane[cPlane].MolarFracs
         var velocity_recycle   = plane[cPlane].velocity
         for i=0, nSpec do MolarFracs_recycle[i]  -= AvgMolarFracs[i] end
         for i=0,     3 do velocity_recycle[i]    -= AvgVelocity[i]   end
         BC[c].temperature_recycle = plane[cPlane].temperature - AvgTemperature
         BC[c].MolarFracs_recycle  = MolarFracs_recycle
         BC[c].velocity_recycle    = velocity_recycle
      end
   end
   return UpdateRecycleVariables
end)

-------------------------------------------------------------------------------
-- CHECK FUNCTIONS
-------------------------------------------------------------------------------

local function CheckNSCBC_Inflow(BC, NSCBC_Inflow)
   return rquote
      -- Check velocity profile
      if NSCBC_Inflow.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
         -- Do nothing
      elseif NSCBC_Inflow.VelocityProfile.type == SCHEMA.InflowProfile_File then
         BC.readProfiles = true
         BC.ProfilesDir = NSCBC_Inflow.VelocityProfile.u.File.FileDir
      elseif NSCBC_Inflow.VelocityProfile.type == SCHEMA.InflowProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming InflowProfile not supported')
      else regentlib.assert(false, 'Unhandled case in InflowProfile switch') end

      -- Check temperature profile
      if NSCBC_Inflow.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
         -- Do nothing
      elseif NSCBC_Inflow.TemperatureProfile.type == SCHEMA.TempProfile_File then
         if (BC.readProfiles) then
            regentlib.assert(C.strcmp(BC.ProfilesDir, NSCBC_Inflow.TemperatureProfile.u.File.FileDir) == 0, 'Only one file is allowed for profiles')
         else
            BC.readProfiles = true
            BC.ProfilesDir = NSCBC_Inflow.TemperatureProfile.u.File.FileDir
         end
      elseif NSCBC_Inflow.TemperatureProfile.type == SCHEMA.TempProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming heat model not supported')
      else regentlib.assert(false, 'Unhandled case in TempProfile switch') end

      -- Check mixture profile
      if NSCBC_Inflow.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
         -- Do nothing
      elseif NSCBC_Inflow.MixtureProfile.type == SCHEMA.MixtureProfile_File then
         if (BC.readProfiles) then
            regentlib.assert(C.strcmp(BC.ProfilesDir, NSCBC_Inflow.MixtureProfile.u.File.FileDir) == 0, 'Only one file is allowed for profiles')
         else
            BC.readProfiles = true
            BC.ProfilesDir = NSCBC_Inflow.MixtureProfile.u.File.FileDir
         end
      elseif NSCBC_Inflow.MixtureProfile.type == SCHEMA.MixtureProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming mixture model not supported')

      else regentlib.assert(false, 'Unhandled case in MixtureProfile switch') end
   end
end

local function CheckDirichlet(BC, Dirichlet)
   return rquote
      -- Check velocity profile
      if Dirichlet.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
         -- Do nothing
      elseif Dirichlet.VelocityProfile.type == SCHEMA.InflowProfile_File then
         BC.readProfiles = true
         BC.ProfilesDir = Dirichlet.VelocityProfile.u.File.FileDir
      elseif Dirichlet.VelocityProfile.type == SCHEMA.InflowProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming InflowProfile not supported')
      else regentlib.assert(false, 'Unhandled case in InflowProfile switch') end

      -- Check temperature profile
      if Dirichlet.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
         -- Do nothing
      elseif Dirichlet.TemperatureProfile.type == SCHEMA.TempProfile_File then
         if (BC.readProfiles) then
            regentlib.assert(C.strcmp(BC.ProfilesDir, Dirichlet.TemperatureProfile.u.File.FileDir) == 0, 'Only one file is allowed for profiles')
         else
            BC.readProfiles = true
            BC.ProfilesDir = Dirichlet.TemperatureProfile.u.File.FileDir
         end
      elseif Dirichlet.TemperatureProfile.type == SCHEMA.TempProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming heat model not supported')
      else regentlib.assert(false, 'Unhandled case in TempProfile switch') end

      -- Check mixture profile
      if Dirichlet.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
         -- Do nothing
      elseif Dirichlet.MixtureProfile.type == SCHEMA.MixtureProfile_File then
         if (BC.readProfiles) then
            regentlib.assert(C.strcmp(BC.ProfilesDir, Dirichlet.MixtureProfile.u.File.FileDir) == 0, 'Only one file is allowed for profiles')
         else
            BC.readProfiles = true
            BC.ProfilesDir = Dirichlet.MixtureProfile.u.File.FileDir
         end
      elseif Dirichlet.MixtureProfile.type == SCHEMA.MixtureProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming mixture model not supported')

      else regentlib.assert(false, 'Unhandled case in MixtureProfile switch') end
   end
end

local function CheckIsothermalWall(IsothermalWall)
   return rquote
      if IsothermalWall.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
         -- Do nothing
      else
         regentlib.assert(false, 'Only constant heat model supported')
      end
   end
end

local function CheckSuctionAndBlowingWall(SuctionAndBlowingWall)
   return rquote
      regentlib.assert(SuctionAndBlowingWall.A.length == SuctionAndBlowingWall.omega.length,
                      "Equal number of amplitudes and frequencies must be specified")

      regentlib.assert(SuctionAndBlowingWall.A.length == SuctionAndBlowingWall.beta.length,
                      "Equal number of amplitudes and spanwise wave numbers must be specified")

      if SuctionAndBlowingWall.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
         -- Do nothing
      else
         regentlib.assert(false, 'Only constant heat model supported')
      end
   end
end

local function CheckRecycleRescaling(BC, RecycleRescaling, config, initialized)
   return rquote
      -- We currently support only one RecycleRescaling BC
      regentlib.assert(not initialized, "Only one RecycleRescaling BC is supported for now")

      -- Check that we are asking to recycle an existing plane
      var vaidIRecycle = ((RecycleRescaling.iRecycle >= 0) and
                          (RecycleRescaling.iRecycle <= config.Grid.xNum))
      regentlib.assert(vaidIRecycle, 'iRecycle does not fall within the domain bounds')

      -- Check velocity profile
      if RecycleRescaling.VelocityProfile.type == SCHEMA.InflowProfile_Constant then
         -- Do nothing
      elseif RecycleRescaling.VelocityProfile.type == SCHEMA.InflowProfile_File then
         BC.readProfiles = true
         BC.ProfilesDir = RecycleRescaling.VelocityProfile.u.File.FileDir
      elseif RecycleRescaling.VelocityProfile.type == SCHEMA.InflowProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming InflowProfile not supported')
      else regentlib.assert(false, 'Unhandled case in InflowProfile switch') end

      -- Check temperature profile
      if RecycleRescaling.TemperatureProfile.type == SCHEMA.TempProfile_Constant then
         -- Do nothing
      elseif RecycleRescaling.TemperatureProfile.type == SCHEMA.TempProfile_File then
         if (BC.readProfiles) then
            regentlib.assert(C.strcmp(BC.ProfilesDir, RecycleRescaling.TemperatureProfile.u.File.FileDir) == 0, 'Only one file is allowed for profiles')
         else
            BC.readProfiles = true
            BC.ProfilesDir = RecycleRescaling.TemperatureProfile.u.File.FileDir
         end
      elseif RecycleRescaling.TemperatureProfile.type == SCHEMA.TempProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming heat model not supported')
      else regentlib.assert(false, 'Unhandled case in TempProfile switch') end

      -- Check mixture profile
      if RecycleRescaling.MixtureProfile.type == SCHEMA.MixtureProfile_Constant then
         -- Do nothing
      elseif RecycleRescaling.MixtureProfile.type == SCHEMA.MixtureProfile_File then
         if (BC.readProfiles) then
            regentlib.assert(C.strcmp(BC.ProfilesDir, RecycleRescaling.MixtureProfile.u.File.FileDir) == 0, 'Only one file is allowed for profiles')
         else
            BC.readProfiles = true
            BC.ProfilesDir = RecycleRescaling.MixtureProfile.u.File.FileDir
         end
      elseif RecycleRescaling.MixtureProfile.type == SCHEMA.MixtureProfile_Incoming then
         -- Do nothing
         regentlib.assert(false, 'Incoming mixture model not supported')

      else regentlib.assert(false, 'Unhandled case in MixtureProfile switch') end
   end
end

function Exports.CheckInput(BC, config) return rquote

   var [BC.readProfiles] = false
   var [BC.ProfilesDir] = ''

   var RecycleRescalingInitialized = false

   -- Set up flow BC's in x direction
   if (not((config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.xBCRight.type == SCHEMA.FlowBC_Periodic))) then

      if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) then
         [CheckNSCBC_Inflow(BC, rexpr config.BC.xBCLeft.u.NSCBC_Inflow end)];

      elseif (config.BC.xBCLeft.type == SCHEMA.FlowBC_Dirichlet) then
         [CheckDirichlet(BC, rexpr config.BC.xBCLeft.u.Dirichlet end)];

      elseif (config.BC.xBCLeft.type == SCHEMA.FlowBC_RecycleRescaling) then
         [CheckRecycleRescaling(BC, rexpr config.BC.xBCLeft.u.RecycleRescaling end, config, RecycleRescalingInitialized)];

      elseif (config.BC.xBCLeft.type == SCHEMA.FlowBC_AdiabaticWall) then
         -- Do nothing

      else
         regentlib.assert(false, "Boundary conditions in xBCLeft not implemented")
      end

      if (config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         -- Do nothing

      elseif (config.BC.xBCRight.type == SCHEMA.FlowBC_Dirichlet) then
         [CheckDirichlet(BC, rexpr config.BC.xBCRight.u.Dirichlet end)];

      elseif (config.BC.xBCRight.type == SCHEMA.FlowBC_AdiabaticWall) then
         -- Do nothing

      else
         regentlib.assert(false, "Boundary conditions in xBCRight not implemented")
      end
   end

   -- Set up flow BC's in y direction
   if (not((config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.yBCRight.type == SCHEMA.FlowBC_Periodic))) then

      if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         -- Do nothing

      elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_Dirichlet) then
         [CheckDirichlet(BC, rexpr config.BC.yBCLeft.u.Dirichlet end)];

--   if (config.BC.yBCLeft.type == SCHEMA.FlowBC_Symmetry) then
      elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_AdiabaticWall) then
         -- Do nothing

      elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_IsothermalWall) then
         [CheckIsothermalWall(rexpr config.BC.yBCLeft.u.IsothermalWall end)];

      elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_SuctionAndBlowingWall) then
         [CheckSuctionAndBlowingWall(rexpr config.BC.yBCLeft.u.SuctionAndBlowingWall end)];

      else
         regentlib.assert(false, "Boundary conditions in yBCLeft not implemented")
      end

      if (config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         -- Do nothing

      elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_Dirichlet) then
         [CheckDirichlet(BC, rexpr config.BC.yBCRight.u.Dirichlet end)];

--   if (config.BC.yBCRight.type == SCHEMA.FlowBC_Symmetry) then
      elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_AdiabaticWall) then
         -- Do nothing

      elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_IsothermalWall) then
         [CheckIsothermalWall(rexpr config.BC.yBCRight.u.IsothermalWall end)];

      elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_IncomingShock) then
         var vaidIShock = ((config.BC.yBCRight.u.IncomingShock.iShock >= 0) and
                           (config.BC.yBCRight.u.IncomingShock.iShock <= config.Grid.xNum))
         regentlib.assert(vaidIShock, 'iShock does not fall within the domain bounds')

      else
         regentlib.assert(false, "Boundary conditions in yBCRight not implemented")
      end
   end

   -- Set up flow BC's in z direction
   if (not((config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.zBCRight.type == SCHEMA.FlowBC_Periodic))) then

      if (config.BC.zBCLeft.type == SCHEMA.FlowBC_Dirichlet) then
         [CheckDirichlet(BC, rexpr config.BC.zBCLeft.u.Dirichlet end)];

      else
         regentlib.assert(false, "Boundary conditions in zBCLeft not implemented")
      end

      if (config.BC.zBCRight.type == SCHEMA.FlowBC_Dirichlet) then
         [CheckDirichlet(BC, rexpr config.BC.yBCRight.u.Dirichlet end)];

      else
         regentlib.assert(false, "Boundary conditions in zBCRight not implemented")
      end
   end

   -- Check if boundary conditions in each direction are either both periodic or both non-periodic
   if (not (((config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.xBCRight.type == SCHEMA.FlowBC_Periodic))
       or ((not (config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic)) and (not (config.BC.xBCRight.type == SCHEMA.FlowBC_Periodic))))) then
      regentlib.assert(false, "Boundary conditions in x should match for periodicity")
   end
   if (not (((config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.yBCRight.type == SCHEMA.FlowBC_Periodic))
      or ((not (config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic)) and (not (config.BC.yBCRight.type == SCHEMA.FlowBC_Periodic))))) then
      regentlib.assert(false, "Boundary conditions in y should match for periodicity")
   end
   if (not (((config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.zBCRight.type == SCHEMA.FlowBC_Periodic))
      or ((not (config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic)) and (not (config.BC.zBCRight.type == SCHEMA.FlowBC_Periodic))))) then
      regentlib.assert(false, "Boundary conditions in z should match for periodicity")
   end

end end

-------------------------------------------------------------------------------
-- DECLARE SYMBOLS
-------------------------------------------------------------------------------

function Exports.DeclSymbols(BC, Fluid, tiles, Fluid_Zones, Grid, config, Mix) return rquote

   -- Unpack the partitions that we are going to need
   var {p_All,
          xNeg,   xPos,   yNeg,   yPos,   zNeg,   zPos,
        p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Zones

   var sampleId = config.Mapping.sampleId

   -- IncomingShock params
   var IncomingShock : IncomingShockParams

   -- RecycleRescaling params
   var RecycleRescaling : RecycleRescalingParams(Fluid, tiles)
   var [BC.RecycleAverage] = region(ispace(int1d, config.Grid.yNum + 2*Grid.yBnum), RecycleAverageType);
   [UTIL.emitRegionTagAttach(BC.RecycleAverage, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

   -- IncomingShock parameters
   if (config.BC.yBCRight.type == SCHEMA.FlowBC_IncomingShock) then
      IncomingShock = InitIncomingShockParams(config, Mix)
   end

   -- RecycleRescaling parameters
   if (config.BC.xBCLeft.type == SCHEMA.FlowBC_RecycleRescaling) then
      RecycleRescaling = initRecycleRescalingParams(Fluid, tiles, p_All,
                                                    xNeg, p_xNeg,
                                                    config.BC.xBCLeft,
                                                    Grid.xBnum, Grid.yBnum, Grid.zBnum);
   end

   -- Initialize BC params
   var [BC.BCParams] = [BCParamsStruct(Fluid, tiles)] {
      IncomingShock = IncomingShock,
      RecycleRescaling = RecycleRescaling
   };
end end

-------------------------------------------------------------------------------
-- BC ROUTINES
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-- INITIALIZATION FUNCTIONS
-------------------------------------------------------------------------------
-- Set up stuff for RHS of NSCBC inflow
local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task InitializeGhostNSCBC(Fluid : region(ispace(int3d), Fluid_columns),
                          Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                          mix : MIX.Mixture)
where
   reads(Fluid.[Primitives]),
   writes(Fluid.{velocity_old_NSCBC, temperature_old_NSCBC})
do
   var BC   = Fluid_BC[0]

   __demand(__openmp)
   for c in BC do
      BC[c].velocity_old_NSCBC = BC[c].velocity
      BC[c].temperature_old_NSCBC = BC[c].temperature
   end
end

function Exports.InitBCs(BC, Fluid_Zones, config, Mix) return rquote
   -- Unpack the partitions that we are going to need
   var {p_All,
        p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Zones

   -- initialize ghost cells values for NSCBC
   if ((config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) or
       (config.BC.xBCLeft.type == SCHEMA.FlowBC_RecycleRescaling))then
      __demand(__index_launch)
      for c in xNeg_ispace do
         InitializeGhostNSCBC(p_All[c], p_xNeg[c], Mix)
      end
   end

   if config.BC.yBCRight.type == SCHEMA.FlowBC_IncomingShock then
      __demand(__index_launch)
      for c in yPos_ispace do
         InitializeGhostNSCBC(p_All[c], p_yPos[c], Mix)
      end
   end

   -- initialize averages for recycle-rescaling conditions
   var initFI = false
   if config.BC.xBCLeft.type == SCHEMA.FlowBC_RecycleRescaling then
      initFI = true
      -- Initialize Recycle average region
      var {BCPlane, RecyclePlane} = BC.BCParams.RecycleRescaling;
      [mkInitializeRecycleAverage("all")](BC.RecycleAverage)
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkAddRecycleAverage("pos")](RecyclePlane[c], BC.RecycleAverage)
      end
      ComputeRecycleAveragePosition(BC.RecycleAverage)

      -- Initialize parameters for rescaling parameters
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkAddRecycleAverage("BC")](BCPlane[c], BC.RecycleAverage,
                                     Mix,
                                     config.BC.xBCLeft.u.RecycleRescaling.P)
      end
      BC.BCParams.RecycleRescaling.BLData = InitializeBoundarLayerData(BC.RecycleAverage, Mix)
      BC.BCParams.RecycleRescaling.RescalingData = GetRescalingData(BC.RecycleAverage,
                                                                    BC.BCParams.RecycleRescaling.BLData,
                                                                    Mix)
   end

   -- initialize fast interpolation tool
   var nloc = 1
   if initFI then
      BC.BCParams.RecycleRescaling.FIdata = FIInitData(BC.RecycleAverage)
      nloc = BC.BCParams.RecycleRescaling.FIdata.nloc
   end
   var sampleId = config.Mapping.sampleId
   var [BC.RecycleAverageFI] = region(ispace(int1d, nloc), FIType);
   [UTIL.emitRegionTagAttach(BC.RecycleAverageFI, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   fill([BC.RecycleAverageFI].xloc, 0.0)
   fill([BC.RecycleAverageFI].iloc, 0.0)
   if initFI then
      FIInitRegion(BC.RecycleAverageFI, BC.RecycleAverage,
                   BC.BCParams.RecycleRescaling.FIdata)
   end
end end

------------------------------------------------------------------------
-- BC utils
------------------------------------------------------------------------
local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task SetDirichletBC(Fluid    : region(ispace(int3d), Fluid_columns),
                    Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                    Pbc : double)
where
   reads(Fluid.[ProfilesVars]),
   writes(Fluid.[Primitives])
do
   var BC   = Fluid_BC[0]
   __demand(__openmp)
   for c in BC do
      BC[c].MolarFracs  = BC[c].MolarFracs_profile
      BC[c].velocity    = BC[c].velocity_profile
      BC[c].temperature = BC[c].temperature_profile
      BC[c].pressure    = Pbc
   end
end

local mkSetNSCBC_InflowBC = terralib.memoize(function(dir)
   local SetNSCBC_InflowBC
   local idx
   if dir == "x" then
      idx = 0
   elseif dir == "y" then
      idx = 1
   elseif dir == "z" then
      idx = 2
   end
   -- NOTE: It is safe to not pass the ghost regions to this task, because we
   -- always group ghost cells with their neighboring interior cells.
   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetNSCBC_InflowBC(Fluid    : region(ispace(int3d), Fluid_columns),
                          Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                          mix : MIX.Mixture,
                          Pbc : double)
   where
      reads(Fluid.SoS),
      reads(Fluid.Conserved),
      reads(Fluid.[ProfilesVars]),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      __demand(__openmp)
      for c in BC do
         BC[c].MolarFracs  = BC[c].MolarFracs_profile
         BC[c].velocity    = BC[c].velocity_profile
         BC[c].temperature = BC[c].temperature_profile
         if (BC[c].velocity_profile[idx] >= BC[c].SoS) then
            -- It is supersonic, everything is imposed by the BC
            BC[c].pressure = Pbc
         else
            -- Compute pressure from NSCBC conservation equations
            var rhoYi : double[nSpec]
            for i=0, nSpec do
               rhoYi[i] = BC[c].Conserved[i]
            end
            var rho = MIX.GetRhoFromRhoYi(rhoYi)
            var MixW = MIX.GetMolarWeightFromXi(BC[c].MolarFracs_profile, mix)
            BC[c].pressure = MIX.GetPFromRhoAndT(rho, MixW, BC[c].temperature_profile)
         end

      end
   end
   return SetNSCBC_InflowBC
end)

local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task SetNSCBC_OutflowBC(Fluid    : region(ispace(int3d), Fluid_columns),
                        Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                        mix : MIX.Mixture)
where
   reads(Fluid.Conserved),
   reads(Fluid.temperature),
   writes(Fluid.[Primitives])
do
   var BC   = Fluid_BC[0]
   var BCst = Fluid_BC[1]

   __demand(__openmp)
   for c in BC do
      -- Compute values from NSCBC conservation equations
      var rhoYi : double[nSpec]
      for i=0, nSpec do
         rhoYi[i] = BC[c].Conserved[i]
      end
      var rho = MIX.GetRhoFromRhoYi(rhoYi)
      var Yi = MIX.GetYi(rho, rhoYi)
      Yi = MIX.ClipYi(Yi)
      var MixW = MIX.GetMolarWeightFromYi(Yi, mix)
      BC[c].MolarFracs = MIX.GetMolarFractions(MixW, Yi, mix)
      var rhoInv = 1.0/rho;
      var velocity = array(BC[c].Conserved[irU+0]*rhoInv,
                           BC[c].Conserved[irU+1]*rhoInv,
                           BC[c].Conserved[irU+2]*rhoInv)
      BC[c].velocity = velocity
      var kineticEnergy = (0.5*MACRO.dot(velocity, velocity))
      var InternalEnergy = BC[c].Conserved[irE]*rhoInv - kineticEnergy
      BC[c].temperature = MIX.GetTFromInternalEnergy(InternalEnergy, BC[c].temperature, Yi, mix);
      BC[c].pressure    = MIX.GetPFromRhoAndT(rho, MixW, BC[c].temperature)
   end
end

local mkSetAdiabaticWallBC = terralib.memoize(function(dir)
   local SetAdiabaticWallBC

   local mk_cint
   if dir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif dir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif dir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif dir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif dir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif dir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetAdiabaticWallBC(Fluid    : region(ispace(int3d), Fluid_columns),
                           Fluid_BC : partition(disjoint, Fluid, ispace(int1d)))
   where
      reads(Fluid.{MolarFracs, pressure, temperature}),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];
         BC[c].MolarFracs  = BCst[c_int].MolarFracs
         BC[c].pressure    = BCst[c_int].pressure
         BC[c].temperature = BCst[c_int].temperature
         BC[c].velocity    = array(0.0, 0.0, 0.0)
      end
   end
   return SetAdiabaticWallBC
end)

local mkSetIsothermalWallBC = terralib.memoize(function(dir)
   local SetIsothermalWallBC

   local mk_cint
   if dir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif dir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif dir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif dir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif dir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif dir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetIsothermalWallBC(Fluid    : region(ispace(int3d), Fluid_columns),
                            Fluid_BC : partition(disjoint, Fluid, ispace(int1d)))
   where
      reads(Fluid.{MolarFracs, pressure}),
      reads(Fluid.temperature_profile),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];
         BC[c].MolarFracs  = BCst[c_int].MolarFracs
         BC[c].pressure    = BCst[c_int].pressure
         BC[c].temperature = BC[c].temperature_profile
         BC[c].velocity    = array(0.0, 0.0, 0.0)
      end
   end
   return SetIsothermalWallBC
end)

local mkSetSuctionAndBlowingWallBC = terralib.memoize(function(dir)
   local SetSuctionAndBlowingWallBC

   local iStrm
   local iNorm
   local iSpan
   if dir == "yNeg" then
      iStrm = 0
      iNorm = 1
      iSpan = 2
   else assert(false) end

   local mk_cint
   if dir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif dir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif dir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif dir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif dir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif dir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetSuctionAndBlowingWallBC(Fluid    : region(ispace(int3d), Fluid_columns),
                                   Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                   Xmin   : double,
                                   Xmax   : double,
                                   X0     : double,
                                   sigma  : double,
                                   Zw     : double,
                                   nModes : int,
                                   A      : double[20],
                                   omega  : double[20],
                                   beta   : double[20],
                                   Width  : double,
                                   Origin : double,
                                   time   : double)
   where
      reads(Fluid.{MolarFracs, pressure}),
      reads(Fluid.centerCoordinates),
      reads(Fluid.temperature_profile),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];

         BC[c].MolarFracs  = BCst[c_int].MolarFracs
         BC[c].pressure    = BCst[c_int].pressure
         BC[c].temperature = BC[c].temperature_profile
         var velocity      = array(0.0, 0.0, 0.0)

         if ((BC[c].centerCoordinates[iStrm] > Xmin) and
             (BC[c].centerCoordinates[iStrm] < Xmax)) then

            var f = exp(-0.5*pow((BC[c].centerCoordinates[iStrm] - X0)/sigma, 2))

            var g = 1.0 + 0.1*( exp(-pow((BC[c].centerCoordinates[iSpan] - (0.5*Width+Origin) - Zw)/Zw, 2)) -
                                exp(-pow((BC[c].centerCoordinates[iSpan] - (0.5*Width+Origin) + Zw)/Zw, 2)))

            var h = 0.0
            for i=0, nModes do
               h += A[i]*sin(omega[i]*time -
                             beta[i]*BC[c].centerCoordinates[iSpan])
            end
            velocity[iNorm] = f*g*h
         end
         BC[c].velocity = velocity
      end
   end
   return SetSuctionAndBlowingWallBC
end)

local mkSetIncomingShockBC = terralib.memoize(function(dir)
   local SetIncomingShockBC

   local idx
   local sdir
   --if dir == "x" then
   --   idx = 0
   --elseif dir == "y" then
   if dir == "yPos" then
      idx = 1
      sdir = "x"
--   elseif dir == "z" then
--      idx = 2
   else assert(false) end

   -- NOTE: It is safe to not pass the ghost regions to this task, because we
   -- always group ghost cells with their neighboring interior cells.
   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetIncomingShockBC(Fluid    : region(ispace(int3d), Fluid_columns),
                           Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                           params : IncomingShockParams,
                           mix : MIX.Mixture)
   where
      reads(Fluid.SoS),
      reads(Fluid.Conserved),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var MixW = MIX.GetMolarWeightFromXi(params.MolarFracs, mix)

      __demand(__openmp)
      for c in BC do
         if (c.[sdir] < params.iShock) then
            BC[c].MolarFracs  = params.MolarFracs
            BC[c].velocity    = params.velocity0
            BC[c].temperature = params.temperature0
            BC[c].pressure    = params.pressure0

         elseif (c.[sdir] == params.iShock) then
            BC[c].MolarFracs  = params.MolarFracs
            BC[c].velocity    = params.velocity1
            BC[c].temperature = params.temperature1
            BC[c].pressure    = params.pressure1

         else
            BC[c].MolarFracs  = params.MolarFracs
            BC[c].velocity    = params.velocity1
            BC[c].temperature = params.temperature1

            if (params.velocity1[idx] <= -BC[c].SoS) then
               -- It is supersonic, everything is imposed by the BC
               BC[c].pressure = params.pressure1
            else
               -- Compute pressure from NSCBC conservation equations
               var rhoYi : double[nSpec]
               for i=0, nSpec do
                  rhoYi[i] = BC[c].Conserved[i]
               end
               var rho = MIX.GetRhoFromRhoYi(rhoYi)
               BC[c].pressure = MIX.GetPFromRhoAndT(rho, MixW, params.temperature1)
            end

         end
      end
   end
   return SetIncomingShockBC
end)

local mkSetRecycleRescaling = terralib.memoize(function(dir)
   local SetRecycleRescaling
   local idx
   if dir == "x" then
      idx = 0
--   elseif dir == "y" then
--      idx = 1
--   elseif dir == "z" then
--      idx = 2
   else assert(false) end

   local function emitInterp(r, c, cp1, fld, w, ind)
      if ind == nil then
         return rexpr
            r[c  ].[fld]*w +
            r[cp1].[fld]*(1.0-w)
         end
      else
         return rexpr
            r[c  ].[fld][ind]*w +
            r[cp1].[fld][ind]*(1.0-w)
         end
      end
   end

   local function emitInterpAll(r0, rInt, avg, FIregion, FIdata, c, yF, t, v, MF) return rquote
      var yR = r0[c].centerCoordinates[1]*yF
      var cAvg = FIFindIndex(yR, FIregion, FIdata)
      var cAp1 = cAvg+int1d{1}
      var cInt = int3d{c.x, cAvg, c.z}
      var cIp1 = cInt+int3d{0, 1, 0}
      var w = FIGetWeight(yR, avg[cAvg].y, avg[cAp1].y)
      t = [emitInterp(rInt, cInt, cIp1, "temperature_recycle", w)];
      for i=0, 3 do
         v[i] = [emitInterp(rInt, cInt, cIp1, "velocity_recycle", w, i)]
      end
      for i=0, nSpec do
         MF[i] = [emitInterp(rInt, cInt, cIp1, "MolarFracs_recycle", w, i)]
      end
   end end

   local alpha = 4.0
--   local b = 0.125 -- for Mach 2
--   local b = 0.3   -- for Mach 3
   local b = 0.4   -- Original
   local __demand(__inline)
   task weightf(x : double)
      var rnum = alpha*(x-b)
      var rden = b + (1.0-2.0*b)*x
      var weightf = 0.5*(1.0 + tanh(rnum/rden)/tanh(alpha))
      if (x > 1.0) then weightf = 1.0 end
      return weightf
   end

   local __demand(__inline)
   task bernardinidamp(x : double)
      return 0.5*(1.0 - tanh(5.0*(x-1.75)))
   end

   __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetRecycleRescaling(Fluid     : region(ispace(int3d), Fluid_columns),
                            Fluid_BC  : partition(disjoint, Fluid, ispace(int1d)),
                            avg       : region(ispace(int1d), RecycleAverageType),
                            BC_interp : region(ispace(int3d), Fluid_columns),
                            FIregion  : region(ispace(int1d), FIType),
                            FIdata    : FIData,
                            RdataIn   : RescalingDataType,
                            RdataRe   : RescalingDataType,
                            mix : MIX.Mixture,
                            Pbc : double)
   where
      reads(Fluid.centerCoordinates),
      reads(Fluid.SoS),
      reads(Fluid.Conserved),
      reads(Fluid.[ProfilesVars]),
      reads(avg.{y}),
      reads(BC_interp.[RecycleVars]),
      reads(FIregion),
      writes(Fluid.[Primitives])
   do
      -- Compute rescaling coefficients
      var yInnFact = RdataRe.deltaNu  /RdataIn.deltaNu
      var yOutFact = RdataRe.delta99VD/RdataIn.delta99VD
      var uInnFact = RdataIn.uTau/RdataRe.uTau
      var uOutFact = uInnFact*sqrt(RdataIn.rhow/RdataRe.rhow)

      var idelta99Inl = 1.0/RdataIn.delta99VD

      -- Set boundary conditions
      var BC   = Fluid_BC[0]
      __demand(__openmp)
      for c in BC do
         -- Interpolate fluctuations based on the inner scaling
         var temperatureInn : double
         var velocityInn    : double[3]
         var MolarFracsInn  : double[nSpec]
         [emitInterpAll(BC, BC_interp, avg, FIregion, FIdata, c, yInnFact,
                        temperatureInn, velocityInn, MolarFracsInn)];
         for i=0, 3 do velocityInn[i] *= uInnFact end

         -- Interpolate fluctuations based on the outer scaling
         var temperatureOut : double
         var velocityOut    : double[3]
         var MolarFracsOut  : double[nSpec]
         [emitInterpAll(BC, BC_interp, avg, FIregion, FIdata, c, yOutFact,
                        temperatureOut, velocityOut, MolarFracsOut)];
         for i=0, 3 do velocityOut[i] *= uOutFact end

         -- Blend the results, multiply by free-stream dumping, and add mean profiles
         var etaInl = BC[c].centerCoordinates[1]*idelta99Inl
         var w = weightf(etaInl)
         var damp = bernardinidamp(etaInl)
         var temperature = (temperatureInn*(1.0-w) + temperatureOut*w)*damp + BC[c].temperature_profile
         var velocity : double[3]
         for i=0, 3 do
            velocity[i] = (velocityInn[i]*(1.0-w) + velocityOut[i]*w)*damp + BC[c].velocity_profile[i]
         end
         var MolarFracs : double[nSpec]
         for i=0, nSpec do
            MolarFracs[i] = (MolarFracsInn[i]*(1.0-w) + MolarFracsOut[i]*w)*damp + BC[c].MolarFracs_profile[i]
         end

         -- Set boundary conditions
         BC[c].MolarFracs  = MolarFracs
         BC[c].velocity    = velocity
         BC[c].temperature = temperature
         if (velocity[idx] >= BC[c].SoS) then
            -- It is supersonic, everything is imposed by the BC
            BC[c].pressure = Pbc
         else
            -- Compute pressure from NSCBC conservation equations
            var rhoYi : double[nSpec]
            for i=0, nSpec do
               rhoYi[i] = BC[c].Conserved[i]
            end
            var rho = MIX.GetRhoFromRhoYi(rhoYi)
            var MixW = MIX.GetMolarWeightFromXi(MolarFracs, mix)
            BC[c].pressure = MIX.GetPFromRhoAndT(rho, MixW, temperature)
         end
      end
   end
   return SetRecycleRescaling
end)

-- Update the ghost cells to impose boundary conditions
__demand(__inline)
task Exports.UpdateGhostPrimitives(Fluid : region(ispace(int3d), Fluid_columns),
                                   tiles : ispace(int3d),
                                   Fluid_Partitions : zones_partitions(Fluid, tiles),
                                   BCParams : BCParamsStruct(Fluid, tiles),
                                   BCRecycleAverage : region(ispace(int1d), RecycleAverageType),
                                   BCRecycleAverageFI : region(ispace(int1d), FIType),
                                   config : SCHEMA.Config,
                                   Mix : MIX.Mixture,
                                   Integrator_simTime : double)
where
   reads writes(Fluid),
   reads writes(BCRecycleAverage),
   reads(BCRecycleAverageFI)
do
   var BC_xBCLeft  = config.BC.xBCLeft.type
   var BC_xBCRight = config.BC.xBCRight.type
   var BC_yBCLeft  = config.BC.yBCLeft.type
   var BC_yBCRight = config.BC.yBCRight.type
   var BC_zBCLeft  = config.BC.zBCLeft.type
   var BC_zBCRight = config.BC.zBCRight.type

   var {p_All, p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_Partitions

   -- Start updating BCs that are local
   -- xNeg BC
   if (BC_xBCLeft == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in xNeg_ispace do
         SetDirichletBC(p_All[c], p_xNeg[c], config.BC.xBCLeft.u.Dirichlet.P)
      end
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkSetNSCBC_InflowBC("x")](p_All[c], p_xNeg[c], Mix, config.BC.xBCLeft.u.NSCBC_Inflow.P)
      end
   elseif (BC_xBCLeft == SCHEMA.FlowBC_RecycleRescaling) then
      var {RecyclePlane, BC_interp} = BCParams.RecycleRescaling;
      -- collect averages on the recycling plane
      [mkInitializeRecycleAverage("std")](BCRecycleAverage)
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkAddRecycleAverage("std")](RecyclePlane[c], BCRecycleAverage)
      end
      -- complete spatial averages and put fluctuations *_recycle vars
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkUpdateRecycleVariables("x")](p_All[c], p_xNeg[c],
                                         BCRecycleAverage,
                                         RecyclePlane[c],
                                         config.BC.xBCLeft.u.RecycleRescaling.iRecycle)
      end
      -- update rescaling data
      var RescalingDataRec = GetRescalingData(BCRecycleAverage,
                                              BCParams.RecycleRescaling.BLData,
                                              Mix)
      -- update boundary condition
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkSetRecycleRescaling("x")](p_All[c], p_xNeg[c],
                                      BCRecycleAverage,
                                      BC_interp[int1d{c.z}],
                                      BCRecycleAverageFI,
                                      BCParams.RecycleRescaling.FIdata,
                                      BCParams.RecycleRescaling.RescalingData,
                                      RescalingDataRec,
                                      Mix,
                                      config.BC.xBCLeft.u.RecycleRescaling.P)
      end
   end

   -- xPos BC
   if (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in xPos_ispace do
         SetDirichletBC(p_All[c], p_xPos[c], config.BC.xBCRight.u.Dirichlet.P)
      end
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      __demand(__index_launch)
      for c in xPos_ispace do
         SetNSCBC_OutflowBC(p_All[c], p_xPos[c], Mix)
      end
   end

   -- yNeg BC
   if (BC_yBCLeft == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in yNeg_ispace do
         SetDirichletBC(p_All[c], p_yNeg[c], config.BC.yBCLeft.u.Dirichlet.P)
      end
   elseif (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      __demand(__index_launch)
      for c in yNeg_ispace do
         SetNSCBC_OutflowBC(p_All[c], p_yNeg[c], Mix)
      end
   end

   -- yPos BC
   if (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in yPos_ispace do
         SetDirichletBC(p_All[c], p_yPos[c], config.BC.yBCRight.u.Dirichlet.P)
      end
   elseif (BC_yBCRight == SCHEMA.FlowBC_IncomingShock) then
      __demand(__index_launch)
      for c in yPos_ispace do
         [mkSetIncomingShockBC("yPos")](p_All[c], p_yPos[c], BCParams.IncomingShock, Mix)
      end
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      __demand(__index_launch)
      for c in yPos_ispace do
         SetNSCBC_OutflowBC(p_All[c], p_yPos[c], Mix)
      end
   end

   -- yNeg BC
   if (BC_zBCLeft == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in zNeg_ispace do
         SetDirichletBC(p_All[c], p_zNeg[c], config.BC.zBCLeft.u.Dirichlet.P)
      end
   end

   -- yPos BC
   if (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in zPos_ispace do
         SetDirichletBC(p_All[c], p_zPos[c], config.BC.zBCRight.u.Dirichlet.P)
      end
   end

   -- BCs that depend on other cells need to be set at the end in the order Z, Y, X
   -- yNeg BC
   if (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in yNeg_ispace do
         [mkSetAdiabaticWallBC("yNeg")](p_All[c], p_yNeg[c])
      end
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      __demand(__index_launch)
      for c in yNeg_ispace do
         [mkSetIsothermalWallBC("yNeg")](p_All[c], p_yNeg[c])
      end
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      __demand(__index_launch)
      for c in yNeg_ispace do
         [mkSetSuctionAndBlowingWallBC("yNeg")](p_All[c], p_yNeg[c],
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.Xmin,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.Xmax,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.X0,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.sigma,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.Zw,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.A.length,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.A.values,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.omega.values,
                                                config.BC.yBCLeft.u.SuctionAndBlowingWall.beta.values,
                                                config.Grid.zWidth,
                                                config.Grid.origin[2],
                                                Integrator_simTime)
      end
   end

   -- yPos BC
   if (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in yPos_ispace do
         [mkSetAdiabaticWallBC("yPos")](p_All[c], p_yPos[c])
      end
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      __demand(__index_launch)
      for c in yPos_ispace do
         [mkSetIsothermalWallBC("yPos")](p_All[c], p_yPos[c])
      end
   end

   -- xNeg BC
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in xNeg_ispace do
         [mkSetAdiabaticWallBC("xNeg")](p_All[c], p_xNeg[c])
      end
   end

   -- xPos BC
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in xPos_ispace do
         [mkSetAdiabaticWallBC("xPos")](p_All[c], p_xPos[c])
      end
   end
end

------------------------------------------------------------------------
-- Post-timestep routines
------------------------------------------------------------------------

-- Update the time derivative values needed for the inflow
local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task UpdateNSCBCGhostCellTimeDerivatives(Fluid : region(ispace(int3d), Fluid_columns),
                                         Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                         Integrator_deltaTime : double)
where
  reads(Fluid.{velocity, temperature}),
  reads writes(Fluid.{dudtBoundary, dTdtBoundary}),
  reads writes(Fluid.{velocity_old_NSCBC, temperature_old_NSCBC})
do
   var BC   = Fluid_BC[0]
   __demand(__openmp)
   for c in BC do
      for i=1, 3 do
         BC[c].dudtBoundary[i] = (BC[c].velocity[i] - BC[c].velocity_old_NSCBC[i]) / Integrator_deltaTime
      end
      BC[c].dTdtBoundary = (BC[c].temperature - BC[c].temperature_old_NSCBC) / Integrator_deltaTime
      BC[c].velocity_old_NSCBC    = BC[c].velocity
      BC[c].temperature_old_NSCBC = BC[c].temperature
   end
end

-- Update time derivatives in the ghost cells
__demand(__inline)
task Exports.UpdateNSCBCGhostCellTimeDerivatives(Fluid : region(ispace(int3d), Fluid_columns),
                                                 tiles : ispace(int3d),
                                                 Fluid_zones : zones_partitions(Fluid, tiles),
                                                 config : SCHEMA.Config,
                                                 Integrator_deltaTime : double)
where
   reads writes(Fluid)
do
   var BC_xBCLeft  = config.BC.xBCLeft.type
   var BC_xBCRight = config.BC.xBCRight.type
   var BC_yBCLeft  = config.BC.yBCLeft.type
   var BC_yBCRight = config.BC.yBCRight.type
   var BC_zBCLeft  = config.BC.zBCLeft.type
   var BC_zBCRight = config.BC.zBCRight.type

   var {p_All, p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos,
        xNeg_ispace, xPos_ispace,
        yNeg_ispace, yPos_ispace,
        zNeg_ispace, zPos_ispace} = Fluid_zones

   -- Update time derivatives at boundary for NSCBC
   if ((BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) or
       (BC_xBCLeft == SCHEMA.FlowBC_RecycleRescaling)) then
      __demand(__index_launch)
      for c in xNeg_ispace do
         UpdateNSCBCGhostCellTimeDerivatives(p_All[c], p_xNeg[c], Integrator_deltaTime)
      end
   end

   -- Update time derivatives at boundary for IncomingShock
   if BC_yBCRight == SCHEMA.FlowBC_IncomingShock then
      __demand(__index_launch)
      for c in yPos_ispace do
         UpdateNSCBCGhostCellTimeDerivatives(p_All[c], p_yPos[c], Integrator_deltaTime)
      end
   end
end

return Exports end

