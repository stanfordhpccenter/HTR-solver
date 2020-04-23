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

return function(SCHEMA, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local MATH = require "math_utils"
local MACRO = require "prometeo_macro"
local CONST = require "prometeo_const"

-- Stencil indices
local Stencil1  = CONST.Stencil1
local Stencil2  = CONST.Stencil2
local Stencil3  = CONST.Stencil3
local Stencil4  = CONST.Stencil4
local nStencils = CONST.nStencils

-- LU decomposition
local LUdec, ludcmp, lubksb = unpack(MATH.mkLUdec(nStencils))

-------------------------------------------------------------------------------
-- METRIC ROUTINES
-------------------------------------------------------------------------------

-- dxm2:     |-----------------------------------------|
-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
-- dxp2:                                               |---------------------------|
-- dxp3:                                               |-----------------------------------------|
--                 c-2           c-1            c     x=0    c+1           c+2           c+3
--           |------x------|------x------|------x------|------x------|------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                           o-------------o-----> <-----o
-- 2nd:                                         o-----> <-----o-------------o
-- 3rd:             o-------------o-------------o----->
-- 4th:                                         o-----> <-----o-------------o-------------o
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o-------------o
-- 2nd:                           o-------------o-----> <-----o
-- 3rd:                                                 <-----o-------------o-------------o
-- 4th:             o-------------o-------------o-----> <-----o

local __demand(__inline)
task GetReconstructionPlus(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double)
   -- Stencil 1
   var stencil1 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1), 0.0)
   -- Stencil 2
   var stencil2 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2), 0.0)
   -- Stencil 3
   var stencil3 = [MATH.mkReconCoeff(3)](array(dxm3, dxm2, dxm1, 0.0), 0.0)
   -- Stencil 4
   var stencil4 = [MATH.mkReconCoeff(4)](array(dxm1, 0.0, dxp1, dxp2, dxp3), 0.0)

-- Cell:           c-2          c-1           c           c+1          c+2          c+3
   return array(        0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,         0.0,
                        0.0,         0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,
                stencil3[0], stencil3[1], stencil3[2],         0.0,         0.0,         0.0,
                        0.0,         0.0, stencil4[0], stencil4[1], stencil4[2], stencil4[3])
end

local __demand(__inline)
task GetReconstructionMinus(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double)
   -- Stencil 1
   var stencil1 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2), 0.0)
   -- Stencil 2
   var stencil2 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1), 0.0)
   -- Stencil 3
   var stencil3 = [MATH.mkReconCoeff(3)](array(0.0, dxp1, dxp2, dxp3), 0.0)
   -- Stencil 4
   var stencil4 = [MATH.mkReconCoeff(4)](array(dxm3, dxm2, dxm1, 0.0, dxp1), 0.0)

-- Cell:           c-2          c-1           c           c+1          c+2          c+3
   return array(        0.0,         0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,
                        0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,         0.0,
                        0.0,         0.0,         0.0, stencil3[0], stencil3[1], stencil3[2],
                stencil4[0], stencil4[1], stencil4[2], stencil4[3],         0.0,         0.0)
end

local __demand(__inline)
task GetTENOCoeffs(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double, ReconCoeffs: double[nStencils*6])

   -- Sixth order accurate stencil
   var stencil6 = [MATH.mkReconCoeff(6)](array(dxm3, dxm2, dxm1, 0.0, dxp1, dxp2, dxp3), 0.0)
   
   var LU  : LUdec
   var b : double[nStencils]
   for i=0, nStencils do
      for j=0, nStencils do
         LU.A[i*nStencils+j] = ReconCoeffs[j*6+i]
      end
      b[i] = stencil6[i]
   end
   LU = ludcmp(LU)
   return lubksb(LU, b)

end

__demand(__parallel, __cuda, __leaf)
task Exports.InitializeMetric(Fluid : region(ispace(int3d), Fluid_columns),
                      Fluid_bounds : rect3d,
                      Grid_xBnum : int32, Grid_xNum : int32,
                      Grid_yBnum : int32, Grid_yNum : int32,
                      Grid_zBnum : int32, Grid_zNum : int32)
where
   reads(Fluid.centerCoordinates),
   reads(Fluid.cellWidth),
   writes(Fluid.{reconXFacePlus, reconXFaceMinus}),
   writes(Fluid.{reconYFacePlus, reconYFaceMinus}),
   writes(Fluid.{reconZFacePlus, reconZFaceMinus}),
   writes(Fluid.{TENOCoeffsXPlus, TENOCoeffsXMinus}),
   writes(Fluid.{TENOCoeffsYPlus, TENOCoeffsYMinus}),
   writes(Fluid.{TENOCoeffsZPlus, TENOCoeffsZMinus}),
   writes(Fluid.{BCStencilX, BCStencilY, BCStencilZ}),
   writes(Fluid.{interpXFace, interpYFace, interpZFace}),
   writes(Fluid.{ derivXFace,  derivYFace,  derivZFace}),
   writes(Fluid.{  gradX,       gradY,       gradZ})
do
   __demand(__openmp)
   for c in Fluid do
      -- X direction
      var cm2_x = (c+{-2, 0, 0}) % Fluid_bounds
      var cm1_x = (c+{-1, 0, 0}) % Fluid_bounds
      var cp1_x = (c+{ 1, 0, 0}) % Fluid_bounds
      var cp2_x = (c+{ 2, 0, 0}) % Fluid_bounds
      var cp3_x = (c+{ 3, 0, 0}) % Fluid_bounds

      -- Distance of the cell faces from the face c
      var dxm3 = -Fluid[c    ].cellWidth[0] - Fluid[cm1_x].cellWidth[0] - Fluid[cm2_x].cellWidth[0]
      var dxm2 = -Fluid[c    ].cellWidth[0] - Fluid[cm1_x].cellWidth[0]
      var dxm1 = -Fluid[c    ].cellWidth[0]
      var dxp1 =  Fluid[cp1_x].cellWidth[0]
      var dxp2 =  Fluid[cp1_x].cellWidth[0] + Fluid[cp2_x].cellWidth[0]
      var dxp3 =  Fluid[cp1_x].cellWidth[0] + Fluid[cp2_x].cellWidth[0] + Fluid[cp3_x].cellWidth[0]

      -- Face reconstruction operators
      var reconFacePlus  = GetReconstructionPlus( dxm3, dxm2, dxm1, dxp1, dxp2, dxp3)
      var reconFaceMinus = GetReconstructionMinus(dxm3, dxm2, dxm1, dxp1, dxp2, dxp3)
      Fluid[c].reconXFacePlus  = reconFacePlus
      Fluid[c].reconXFaceMinus = reconFaceMinus

      -- TENO blending coefficients
      Fluid[c].TENOCoeffsXPlus  = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, dxp2, dxp3, reconFacePlus)
      Fluid[c].TENOCoeffsXMinus = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, dxp2, dxp3, reconFaceMinus)

      -- Flag as in internal point
      Fluid[c].BCStencilX = false

      -- Face interpolation operator
      var Inv2dx = 1.0/(Fluid[c].cellWidth[0] + Fluid[cp1_x].cellWidth[0])
      Fluid[c].interpXFace = array(Fluid[cp1_x].cellWidth[0]*Inv2dx, Fluid[c].cellWidth[0]*Inv2dx)

      -- Face derivative operator
      Fluid[c].derivXFace  = 2.0*Inv2dx

      -- Cell-center gradient operator
      var Inv2dxm1 = 1.0/(Fluid[c].cellWidth[0] + Fluid[cm1_x].cellWidth[0])
      Fluid[c].gradX = array(Inv2dxm1, Inv2dx)

      -- Y direction
      var cm2_y = (c+{0,-2, 0}) % Fluid_bounds
      var cm1_y = (c+{0,-1, 0}) % Fluid_bounds
      var cp1_y = (c+{0, 1, 0}) % Fluid_bounds
      var cp2_y = (c+{0, 2, 0}) % Fluid_bounds
      var cp3_y = (c+{0, 3, 0}) % Fluid_bounds

      -- Distance of the cell faces from the face c
      var dym3 = -Fluid[c    ].cellWidth[1] - Fluid[cm1_y].cellWidth[1] - Fluid[cm2_y].cellWidth[1]
      var dym2 = -Fluid[c    ].cellWidth[1] - Fluid[cm1_y].cellWidth[1]
      var dym1 = -Fluid[c    ].cellWidth[1]
      var dyp1 =  Fluid[cp1_y].cellWidth[1]
      var dyp2 =  Fluid[cp1_y].cellWidth[1] + Fluid[cp2_y].cellWidth[1]
      var dyp3 =  Fluid[cp1_y].cellWidth[1] + Fluid[cp2_y].cellWidth[1] + Fluid[cp3_y].cellWidth[1]

      -- Face reconstruction operators
      reconFacePlus  = GetReconstructionPlus( dym3, dym2, dym1, dyp1, dyp2, dyp3)
      reconFaceMinus = GetReconstructionMinus(dym3, dym2, dym1, dyp1, dyp2, dyp3)
      Fluid[c].reconYFacePlus  = reconFacePlus
      Fluid[c].reconYFaceMinus = reconFaceMinus

      -- TENO blending coefficients
      Fluid[c].TENOCoeffsYPlus  = GetTENOCoeffs(dym3, dym2, dym1, dyp1, dyp2, dyp3, reconFacePlus)
      Fluid[c].TENOCoeffsYMinus = GetTENOCoeffs(dym3, dym2, dym1, dyp1, dyp2, dyp3, reconFaceMinus)

      -- Flag as in internal point
      Fluid[c].BCStencilY = false

      -- Face interpolation operator
      var Inv2dy = 1.0/(Fluid[c].cellWidth[1] + Fluid[cp1_y].cellWidth[1])
      Fluid[c].interpYFace = array(Fluid[cp1_y].cellWidth[1]*Inv2dy, Fluid[c].cellWidth[1]*Inv2dy)

      -- Face derivative operator
      Fluid[c].derivYFace  = 2.0*Inv2dy

      -- Cell-center gradient operator
      var Inv2dym1 = 1.0/(Fluid[c].cellWidth[1] + Fluid[cm1_y].cellWidth[1])
      Fluid[c].gradY = array(Inv2dym1, Inv2dy)

      -- Z direction
      var cm2_z = (c+{0, 0,-2}) % Fluid_bounds
      var cm1_z = (c+{0, 0,-1}) % Fluid_bounds
      var cp1_z = (c+{0, 0, 1}) % Fluid_bounds
      var cp2_z = (c+{0, 0, 2}) % Fluid_bounds
      var cp3_z = (c+{0, 0, 3}) % Fluid_bounds

      -- Distance of the cell faces from the face c
      var dzm3 = -Fluid[c    ].cellWidth[2] - Fluid[cm1_z].cellWidth[2] - Fluid[cm2_z].cellWidth[2]
      var dzm2 = -Fluid[c    ].cellWidth[2] - Fluid[cm1_z].cellWidth[2]
      var dzm1 = -Fluid[c    ].cellWidth[2]
      var dzp1 =  Fluid[cp1_z].cellWidth[2]
      var dzp2 =  Fluid[cp1_z].cellWidth[2] + Fluid[cp2_z].cellWidth[2]
      var dzp3 =  Fluid[cp1_z].cellWidth[2] + Fluid[cp2_z].cellWidth[2] + Fluid[cp3_z].cellWidth[2]

      -- Face reconstruction operators
      reconFacePlus  = GetReconstructionPlus( dzm3, dzm2, dzm1, dzp1, dzp2, dzp3)
      reconFaceMinus = GetReconstructionMinus(dzm3, dzm2, dzm1, dzp1, dzp2, dzp3)
      Fluid[c].reconZFacePlus  = reconFacePlus
      Fluid[c].reconZFaceMinus = reconFaceMinus

      -- TENO blending coefficients
      Fluid[c].TENOCoeffsZPlus  = GetTENOCoeffs(dzm3, dzm2, dzm1, dzp1, dzp2, dzp3, reconFacePlus)
      Fluid[c].TENOCoeffsZMinus = GetTENOCoeffs(dzm3, dzm2, dzm1, dzp1, dzp2, dzp3, reconFaceMinus)

      -- Flag as in internal point
      Fluid[c].BCStencilZ = false

      -- Face interpolation operator
      var Inv2dz = 1.0/(Fluid[c].cellWidth[2] + Fluid[cp1_z].cellWidth[2])
      Fluid[c].interpZFace = array(Fluid[cp1_z].cellWidth[2]*Inv2dz, Fluid[c].cellWidth[2]*Inv2dz)

      -- Face derivative operator
      Fluid[c].derivZFace  = 2.0*Inv2dz

      -- Cell-center gradient operator
      var Inv2dzm1 = 1.0/(Fluid[c].cellWidth[2] + Fluid[cm1_z].cellWidth[2])
      Fluid[c].gradZ = array(Inv2dzm1, Inv2dz)
   end
end

-------------------------------------------------------------------------------
-- STAGGERED LEFT BC
-------------------------------------------------------------------------------
-- These functions generate the tasks to correct the metric for a left boundary
-- where the ghost point is staggered on the cell face

local mkCorrectMetricLeftStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricLeftStaggered

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end

   __demand(__inline)
   task CorrectMetricLeftStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                   c : int3d, cp1 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil]),
      writes(Fluid.{[interp], [deriv], [grad]})
   do
      -- Face reconstruction operators
      -- Boundary node is staggered on the face so we do not need any reconstruction
      Fluid[c].[reconPlus]  = array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
      Fluid[c].[reconMinus] = array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

      Fluid[c].[BCStencil] = true

      -- Face interpolation operator
      -- The true value comes from the ghost cell
      Fluid[c].[interp] = array(1.0, 0.0)

      -- Face derivative operator
      var Inv2dx = 1.0/Fluid[cp1].cellWidth[dir]
      Fluid[c].[deriv] = 2.0*Inv2dx

      -- Cell-center gradient operator
      Fluid[c  ].[grad] = array(    0.0, 2.0*Inv2dx)
   end
   return CorrectMetricLeftStaggered
end)

--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
-- dxp2:                                               |---------------------------|
-- dxp3:                                               |-----------------------------------------|
--                                     c-0.5    c     x=0    c+1           c+2           c+3
--                                       x------x------|------x------|------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                                  o------o-----> <-----o
-- 2nd:                                         o-----> <-----o-------------o
-- 3rd:                                         does not exist
-- 4th:                                         o-----> <-----o-------------o-------------o
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o-------------o
-- 2nd:                                  o------o-----> <-----o
-- 3rd:                                                 <-----o-------------o-------------o
-- 4th:                                         does not exist

local mkCorrectMetricLeftPlusOneStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricLeftPlusOneStaggered

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      grad = "gradZ"
   end

   local __demand(__inline)
   task GetReconstructionPlus(dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeffLeftBC(3)](array(dxm1, 0.0, dxp1)      , 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2)      , 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeffLeftBC(2)](array(dxm1, 0.0)            , 0.0)
      -- Stencil 4
      var stencil4 = [MATH.mkReconCoeff(4)](array(dxm1, 0.0, dxp1, dxp2, dxp3), 0.0)

      -- Cell:           c-2        c-1/2           c           c+1          c+2          c+3
      return array(        0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,         0.0,
                           0.0,         0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,
                           0.0, stencil3[0], stencil3[1],         0.0,         0.0,         0.0,
                           0.0,         0.0, stencil4[0], stencil4[1], stencil4[2], stencil4[3])

   end

   local __demand(__inline)
   task GetReconstructionMinus(dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2), 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeffLeftBC(3)](array(dxm1, 0.0, dxp1), 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeff(3)](array(0.0, dxp1, dxp2, dxp3), 0.0)

      -- Cell:           c-2        c-1/2           c           c+1          c+2          c+3
      return array(        0.0,         0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,
                           0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,         0.0,
                           0.0,         0.0,         0.0, stencil3[0], stencil3[1], stencil3[2],
                           0.0,         0.0,         1.0,         0.0,         0.0,         0.0)
      -- The last row is just to avoid a singular problem in the coefficient calculation
   end

   local __demand(__inline)
   task GetTENOCoeffs(dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double, ReconCoeffs: double[nStencils*6])

      -- TENO coefficinets excluding the first cell
      -- (Fifth order accurate)
      var stencil = [MATH.mkReconCoeffLeftBC(5)](array(dxm1, 0.0, dxp1, dxp2, dxp3), 0.0)

      var LU  : LUdec
      var b : double[nStencils]
      for i=0, nStencils do
         for j=0, nStencils do
            LU.A[i*nStencils+j] = ReconCoeffs[j*6+i+1]
         end
         b[i] = stencil[i]
      end
      LU = ludcmp(LU)
      return lubksb(LU, b)

   end

   __demand(__inline)
   task CorrectMetricLeftPlusOneStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                          c : int3d, cp1 : int3d, cp2 : int3d, cp3 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil]),
      writes(Fluid.{[grad]})
   do
      -- Face reconstruction operators
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]
      var dxp2 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir]
      var dxp3 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir] + Fluid[cp3].cellWidth[dir]

      Fluid[c].[reconPlus]  = GetReconstructionPlus( dxm1, dxp1, dxp2, dxp3)
      Fluid[c].[reconMinus] = GetReconstructionMinus(dxm1, dxp1, dxp2, dxp3)
      Fluid[c].[TENOCoeffsPlus]  = GetTENOCoeffs(dxm1, dxp1, dxp2, dxp3, Fluid[c].[reconPlus])
      Fluid[c].[TENOCoeffsMinus] = GetTENOCoeffs(dxm1, dxp1, dxp2, dxp3, Fluid[c].[reconMinus])

      Fluid[c].[BCStencil] = true

      -- Cell-center gradient operator
      var Inv2dx   = 1.0/ Fluid[c].cellWidth[dir]
      var Inv2dxp2 = 1.0/(Fluid[c].cellWidth[dir] + Fluid[cp1].cellWidth[dir])
      Fluid[c].[grad] = array( Inv2dx,   Inv2dxp2)
   end
   return CorrectMetricLeftPlusOneStaggered
end)

-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
-- dxp2:                                               |---------------------------|
-- dxp3:                                               |-----------------------------------------|
--                       c-1.5   c-1            c     x=0    c+1           c+2           c+3
--                         x------x------|------x------|------x------|------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                           o-------------o-----> <-----o
-- 2nd:                                         o-----> <-----o-------------o
-- 3rd:                    o------o-------------o----->
-- 4th:                                         o-----> <-----o-------------o-------------o
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o-------------o
-- 2nd:                           o-------------o-----> <-----o
-- 3rd:                                                 <-----o-------------o-------------o
-- 4th:                    o------o-------------o-----> <-----o

local mkCorrectMetricLeftPlusTwoStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricLeftPlusTwoStaggered

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   end

   local __demand(__inline)
   task GetReconstructionPlus(dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1), 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2), 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeffLeftBC(3)](array(dxm2, dxm1, 0.0), 0.0)
      -- Stencil 4
      var stencil4 = [MATH.mkReconCoeff(4)](array(dxm1, 0.0, dxp1, dxp2, dxp3), 0.0)

      -- Cell:         c-1.5        c-1           c           c+1          c+2          c+3
      return array(        0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,         0.0,
                           0.0,         0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,
                   stencil3[0], stencil3[1], stencil3[2],         0.0,         0.0,         0.0,
                           0.0,         0.0, stencil4[0], stencil4[1], stencil4[2], stencil4[3])

   end

   local __demand(__inline)
   task GetReconstructionMinus(dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2), 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1), 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeff(3)](array(0.0, dxp1, dxp2, dxp3), 0.0)
      -- Stencil 4
      var stencil4 = [MATH.mkReconCoeffLeftBC(4)](array(dxm2, dxm1, 0.0, dxp1), 0.0)

      -- Cell:         c-1.5        c-1           c           c+1          c+2          c+3
      return array(        0.0,         0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,
                           0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,         0.0,
                           0.0,         0.0,         0.0, stencil3[0], stencil3[1], stencil3[2],
                   stencil4[0], stencil4[1], stencil4[2], stencil4[3],         0.0,         0.0)

   end

   local __demand(__inline)
   task GetTENOCoeffs(dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double, ReconCoeffs: double[nStencils*6])

      -- Sixth order accurate stencil with the first point staggered on the face
      var stencil = [MATH.mkReconCoeffLeftBC(6)](array(dxm2, dxm1, 0.0, dxp1, dxp2, dxp3), 0.0)

      var LU  : LUdec
      var b : double[nStencils]
      for i=0, nStencils do
         for j=0, nStencils do
            LU.A[i*nStencils+j] = ReconCoeffs[j*6+i]
         end
         b[i] = stencil[i]
      end
      LU = ludcmp(LU)
      return lubksb(LU, b)

   end

   __demand(__inline)
   task CorrectMetricLeftPlusTwoStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                          cm1 : int3d, c : int3d, cp1 : int3d, cp2 : int3d, cp3 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil])
   do
      -- Face reconstruction operators
      var dxm2 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir]
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]
      var dxp2 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir]
      var dxp3 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir] + Fluid[cp3].cellWidth[dir]

      Fluid[c].[reconPlus]  = GetReconstructionPlus( dxm2, dxm1, dxp1, dxp2, dxp3)
      Fluid[c].[reconMinus] = GetReconstructionMinus(dxm2, dxm1, dxp1, dxp2, dxp3)
      Fluid[c].[TENOCoeffsPlus]  = GetTENOCoeffs(dxm2, dxm1, dxp1, dxp2, dxp3, Fluid[c].[reconPlus])
      Fluid[c].[TENOCoeffsMinus] = GetTENOCoeffs(dxm2, dxm1, dxp1, dxp2, dxp3, Fluid[c].[reconMinus])

      Fluid[c].[BCStencil] = true
   end
   return CorrectMetricLeftPlusTwoStaggered
end)

-------------------------------------------------------------------------------
-- COLLOCATED LEFT BC
-------------------------------------------------------------------------------
-- These functions generate the tasks to correct the metric for a left boundary
-- where the ghost point is in a cell center

--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
--                                              c     x=0    c+1
--                                       |------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                                         o----->        
-- 2nd:                                         does not exist
-- 3rd:                                         does not exist
-- 4th:                                         does not exist
--
-- Minus reconstruction:
-- 1st:                                                 <-----o
-- 2nd:                                         does not exist
-- 3rd:                                         does not exist
-- 4th:                                         does not exist

local mkCorrectMetricLeftCollocated = terralib.memoize(function(sdir) 
   local CorrectMetricLeftCollocated

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      grad = "gradZ"
   end

   __demand(__inline)
   task CorrectMetricLeftCollocated(Fluid : region(ispace(int3d), Fluid_columns),
                                    c : int3d, cp1 : int3d, cp2 : int3d, cp3 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil]),
      writes(Fluid.{[grad]})
   do
      -- Face reconstruction operators
      -- Take into account that there is no c-1 and c-2
      Fluid[c].[reconPlus][Stencil2*6+2] = 1.0
      Fluid[c].[reconPlus][Stencil2*6+3] = 0.0
      Fluid[c].[reconPlus][Stencil2*6+4] = 0.0
      Fluid[c].[reconMinus][Stencil1*6+2] = 0.0
      Fluid[c].[reconMinus][Stencil1*6+3] = 1.0
      Fluid[c].[reconMinus][Stencil1*6+4] = 0.0

      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]
      var dxp2 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir]
      var dxp3 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir] + Fluid[cp3].cellWidth[dir]

      Fluid[c].[TENOCoeffsPlus]  = array(0.0, 1.0, 0.0, 0.0)
      Fluid[c].[TENOCoeffsMinus] = array(1.0, 0.0, 0.0, 0.0)

      Fluid[c].[BCStencil] = true

      -- Cell-center gradient operator
      var Inv2dx = 1.0/(Fluid[c].cellWidth[dir] + Fluid[cp1].cellWidth[dir])
      Fluid[c].[grad] = array( 0.0, 2.0*Inv2dx)

   end
   return CorrectMetricLeftCollocated
end)

-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
-- dxp2:                                               |---------------------------|
-- dxp3:                                               |-----------------------------------------|
--                               c-1            c     x=0    c+1           c+2           c+3
--                         |------x------|------x------|------x------|------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                           o-------------o-----> <-----o
-- 2nd:                                         o-----> <-----o-------------o
-- 3rd:                                         does not exist
-- 4th:                                         does not exist
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o-------------o
-- 2nd:                           o-------------o-----> <-----o
-- 3rd:                                         does not exist
-- 4th:                                         does not exist

local mkCorrectMetricLeftPlusOneCollocated = terralib.memoize(function(sdir) 
   local CorrectMetricLeftPlusOneCollocated

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   end

   local __demand(__inline)
   task GetTENOCoeffs(dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, dxp3 : double, ReconCoeffs: double[nStencils*6])

      -- TENO coefficinets excluding the first and last cell
      -- (Fourth order accurate)
      var stencil = [MATH.mkReconCoeff(4)](array(dxm2, dxm1, 0.0, dxp1, dxp2), 0.0)
   
      var LU  : LUdec
      var b : double[nStencils]
      for i=0, nStencils do
         for j=0, nStencils do
            LU.A[i*nStencils+j] = ReconCoeffs[j*6+i+1]
         end
         b[i] = stencil[i]
      end
      LU = ludcmp(LU)
      return lubksb(LU, b)

   end

   __demand(__inline)
   task CorrectMetricLeftPlusOneCollocated(Fluid : region(ispace(int3d), Fluid_columns),
                                           cm1 : int3d, c : int3d, cp1 : int3d, cp2 : int3d, cp3 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil])
   do
      -- Face reconstruction operators
      var dxm2 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir]
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]
      var dxp2 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir]
      var dxp3 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir] + Fluid[cp3].cellWidth[dir]

      Fluid[c].[TENOCoeffsPlus]  = GetTENOCoeffs(dxm2, dxm1, dxp1, dxp2, dxp3, Fluid[c].[reconPlus])
      Fluid[c].[TENOCoeffsMinus] = GetTENOCoeffs(dxm2, dxm1, dxp1, dxp2, dxp3, Fluid[c].[reconMinus])

      Fluid[c].[BCStencil] = true
   end
   return CorrectMetricLeftPlusOneCollocated
end)

-------------------------------------------------------------------------------
-- STAGGERED RIGHT BC
-------------------------------------------------------------------------------
-- These functions generate the tasks to correct the metric for a right boundary
-- where the ghost point is staggered on the cell face

local mkCorrectMetricRightStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricRightStaggered

   local dir
   local grad
   if sdir == "x" then
      dir = 0
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      grad = "gradZ"
   end

   __demand(__inline)
   task CorrectMetricRightStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                    cm1 : int3d, c : int3d)
   where
      reads(Fluid.cellWidth),
      writes(Fluid.{[grad]})
   do
      -- Cell-center gradient operator
      var Inv2dx = 1.0/Fluid[cm1].cellWidth[dir]
      Fluid[c  ].[grad] = array(2.0*Inv2dx,    0.0)
   end
   return CorrectMetricRightStaggered
end)

local mkCorrectMetricRightMinusOneStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricRightMinusOneStaggered

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end

   __demand(__inline)
   task CorrectMetricRightMinusOneStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                            cm1 : int3d, c : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil]),
      writes(Fluid.{[interp], [deriv], [grad]})
   do
      -- Face reconstruction operators
      -- Boundary node is staggered on the face so we do not need any reconstruction
      Fluid[c].[reconPlus]  = array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
      Fluid[c].[reconMinus] = array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

      Fluid[c].[BCStencil] = true

      -- Face interpolation operator
      -- The true value comes from the ghost cell
      Fluid[c].[interp] = array(0.0, 1.0)

      -- Face derivative operator
      var Inv2dx = 1.0/Fluid[c].cellWidth[dir]
      Fluid[c].[deriv] = 2.0*Inv2dx

      -- Cell-center gradient operator
      var Inv2dxm1 = 1.0/(Fluid[c].cellWidth[dir] + Fluid[cm1].cellWidth[dir])
      Fluid[c].[grad] = array(Inv2dxm1, Inv2dx)
   end
   return CorrectMetricRightMinusOneStaggered
end)

-- dxm2:     |-----------------------------------------|
-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
--                 c-2           c-1            c     x=0    c+1   c+1.5
--           |------x------|------x------|------x------|------x------x
--
-- Plus reconstruction:
-- 1st:                           o-------------o-----> <-----o
-- 2nd:                                         o-----> <-----o------o
-- 3rd:             o-------------o-------------o----->
-- 4th:                                         does not exist
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o------o
-- 2nd:                           o-------------o-----> <-----o
-- 3rd:                                         does not exist
-- 4th:             o-------------o-------------o-----> <-----o

local mkCorrectMetricRightMinusTwoStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricRightMinusTwoStaggered

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   end

   local __demand(__inline)
   task GetReconstructionPlus(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1)      , 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeffRightBC(3)](array(dxm1, 0.0, dxp1)     , 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeff(3)](array(dxm3, dxm2, dxm1, 0.0)      , 0.0)

      -- Cell:           c-2          c-1           c           c+1        c+1.5        c+3
      return array(        0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,         0.0,
                           0.0,         0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,
                   stencil3[0], stencil3[1], stencil3[2],         0.0,         0.0,         0.0,
                           0.0,         0.0,         0.0,         1.0,         0.0,         0.0)
      -- The last row is just to avoid a singular problem in the coefficient calculation
   end

   local __demand(__inline)
   task GetReconstructionMinus(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeffRightBC(3)](array(dxm1, 0.0, dxp1)     , 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1)      , 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeffRightBC(2)](array(0.0, dxp1)           , 0.0)
      -- Stencil 4
      var stencil4 = [MATH.mkReconCoeff(4)](array(dxm3, dxm2, dxm1, 0.0, dxp1), 0.0)

      -- Cell:           c-2          c-1           c           c+1        c+1.5        c+3
      return array(        0.0,         0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,
                           0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,         0.0,
                           0.0,         0.0,         0.0, stencil3[0], stencil3[1],         0.0,
                   stencil4[0], stencil4[1], stencil4[2], stencil4[3],         0.0,         0.0)

   end

   local __demand(__inline)
   task GetTENOCoeffs(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, ReconCoeffs: double[nStencils*6])

      -- TENO coefficinets excluding the last cell
      -- (Fifth order accurate)
      var stencil = [MATH.mkReconCoeffRightBC(5)](array(dxm3, dxm2, dxm1, 0.0, dxp1), 0.0)
   
      var LU  : LUdec
      var b : double[nStencils]
      for i=0, nStencils do
         for j=0, nStencils do
            LU.A[i*nStencils+j] = ReconCoeffs[j*6+i]
         end
         b[i] = stencil[i]
      end
      LU = ludcmp(LU)
      return lubksb(LU, b)

   end

   __demand(__inline)
   task CorrectMetricRightMinusTwoStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                            cm2 : int3d, cm1 : int3d, c : int3d, cp1 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil])
   do
      -- Face reconstruction operators
      var dxm3 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir] - Fluid[cm2].cellWidth[dir]
      var dxm2 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir]
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]

      Fluid[c].[reconPlus]  = GetReconstructionPlus( dxm3, dxm2, dxm1, dxp1)
      Fluid[c].[reconMinus] = GetReconstructionMinus(dxm3, dxm2, dxm1, dxp1)
      Fluid[c].[TENOCoeffsPlus]  = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, Fluid[c].[reconPlus])
      Fluid[c].[TENOCoeffsMinus] = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, Fluid[c].[reconMinus])

      Fluid[c].[BCStencil] = true
   end
   return CorrectMetricRightMinusTwoStaggered
end)

-- dxm2:     |-----------------------------------------|
-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
-- dxp2:                                               |---------------------------|
--                 c-2           c-1            c     x=0    c+1           c+2   c+2.5
--           |------x------|------x------|------x------|------x------|------x------x
--
-- Plus reconstruction:
-- 1st:                           o-------------o-----> <-----o
-- 2nd:                                         o-----> <-----o-------------o
-- 3rd:             o-------------o-------------o----->
-- 4th:                                         o-----> <-----o-------------o------o
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o-------------o
-- 2nd:                           o-------------o-----> <-----o
-- 3rd:                                                 <-----o-------------o------o
-- 4th:             o-------------o-------------o-----> <-----o

local mkCorrectMetricRightMinusThreeStaggered = terralib.memoize(function(sdir) 
   local CorrectMetricRightMinusThreeStaggered

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   end

   local __demand(__inline)
   task GetReconstructionPlus(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1), 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2), 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeff(3)](array(dxm3, dxm2, dxm1, 0.0), 0.0)
      -- Stencil 4
      var stencil4 = [MATH.mkReconCoeffRightBC(4)](array(dxm1, 0.0, dxp1, dxp2), 0.0)

      -- Cell:           c-2          c-1           c           c+1          c+2        c+5/2
      return array(        0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,         0.0,
                           0.0,         0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,
                   stencil3[0], stencil3[1], stencil3[2],         0.0,         0.0,         0.0,
                           0.0,         0.0, stencil4[0], stencil4[1], stencil4[2], stencil4[3])

   end

   local __demand(__inline)
   task GetReconstructionMinus(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double)
      -- Stencil 1
      var stencil1 = [MATH.mkReconCoeff(3)](array(dxm1, 0.0, dxp1, dxp2)      , 0.0)
      -- Stencil 2
      var stencil2 = [MATH.mkReconCoeff(3)](array(dxm2, dxm1, 0.0, dxp1)      , 0.0)
      -- Stencil 3
      var stencil3 = [MATH.mkReconCoeffRightBC(3)](array(0.0, dxp1, dxp2)     , 0.0)
      -- Stencil 4
      var stencil4 = [MATH.mkReconCoeff(4)](array(dxm3, dxm2, dxm1, 0.0, dxp1), 0.0)

      -- Cell:           c-2          c-1           c           c+1          c+2          c+2.5
      return array(        0.0,         0.0, stencil1[0], stencil1[1], stencil1[2],         0.0,
                           0.0, stencil2[0], stencil2[1], stencil2[2],         0.0,         0.0,
                           0.0,         0.0,         0.0, stencil3[0], stencil3[1], stencil3[2],
                   stencil4[0], stencil4[1], stencil4[2], stencil4[3],         0.0,         0.0)

   end

   local __demand(__inline)
   task GetTENOCoeffs(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, ReconCoeffs: double[nStencils*6])

      -- TENO coefficinets considering that the last cell is staggered
      var stencil = [MATH.mkReconCoeffRightBC(6)](array(dxm3, dxm2, dxm1, 0.0, dxp1, dxp2), 0.0)

      var LU  : LUdec
      var b : double[nStencils]
      for i=0, nStencils do
         for j=0, nStencils do
            LU.A[i*nStencils+j] = ReconCoeffs[j*6+i]
         end
         b[i] = stencil[i]
      end
      LU = ludcmp(LU)
      return lubksb(LU, b)

   end

   __demand(__inline)
   task CorrectMetricRightMinusThreeStaggered(Fluid : region(ispace(int3d), Fluid_columns),
                                              cm2 : int3d, cm1 : int3d, c : int3d, cp1 : int3d, cp2 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil])
   do
      -- Face reconstruction operators
      var dxm3 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir] - Fluid[cm2].cellWidth[dir]
      var dxm2 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir]
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]
      var dxp2 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir]

      Fluid[c].[reconPlus]  = GetReconstructionPlus( dxm3, dxm2, dxm1, dxp1, dxp2)
      Fluid[c].[reconMinus] = GetReconstructionMinus(dxm3, dxm2, dxm1, dxp1, dxp2)
      Fluid[c].[TENOCoeffsPlus]  = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, dxp2, Fluid[c].[reconPlus])
      Fluid[c].[TENOCoeffsMinus] = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, dxp2, Fluid[c].[reconMinus])

      Fluid[c].[BCStencil] = true
   end
   return CorrectMetricRightMinusThreeStaggered
end)

-------------------------------------------------------------------------------
-- COLLOCATED RIGHT BC
-------------------------------------------------------------------------------
-- This function generates the task to correct the metric for a right boundary
-- where the ghost point is in a cell center

local mkCorrectMetricRightCollocated = terralib.memoize(function(sdir)
   local CorrectMetricRightCollocated

   local dir
   local grad
   if sdir == "x" then
      dir = 0
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      grad = "gradZ"
   end

   __demand(__inline)
   task CorrectMetricRightCollocated(Fluid : region(ispace(int3d), Fluid_columns),
                                     cm1 : int3d, c : int3d)
   where
      reads(Fluid.cellWidth),
      writes(Fluid.{[grad]})
   do
      -- Cell-center gradient operator
      var Inv2dx = 1.0/(Fluid[cm1].cellWidth[dir] + Fluid[c].cellWidth[dir])
      Fluid[c].[grad] = array(2.0*Inv2dx, 0.0)
   end
   return CorrectMetricRightCollocated
end)

-- dxm2:     |-----------------------------------------|
-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
--                 c-2           c-1            c     x=0    c+1
--           |------x------|------x------|------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                                         o----->
-- 2nd:                                         does not exist
-- 3rd:                                         does not exist
-- 4th:                                         does not exist
--
-- Minus reconstruction:
-- 1st:                                                 <-----o
-- 2nd:                                         does not exist
-- 3rd:                                         does not exist
-- 4th:                                         does not exist

local mkCorrectMetricRightMinusOneCollocated = terralib.memoize(function(sdir)
   local CorrectMetricRightMinusOneCollocated

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   end

   __demand(__inline)
   task CorrectMetricRightMinusOneCollocated(Fluid : region(ispace(int3d), Fluid_columns),
                                             cm2 : int3d, cm1 : int3d, c : int3d, cp1 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil])
   do
      -- Face reconstruction operators
      -- Take into account that there is no c+1 and c+2
      Fluid[c].[reconPlus][Stencil2*6+2] = 1.0
      Fluid[c].[reconPlus][Stencil2*6+3] = 0.0
      Fluid[c].[reconPlus][Stencil2*6+4] = 0.0
      Fluid[c].[reconMinus][Stencil1*6+2] = 0.0
      Fluid[c].[reconMinus][Stencil1*6+3] = 1.0
      Fluid[c].[reconMinus][Stencil1*6+4] = 0.0

      var dxm3 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir] - Fluid[cm2].cellWidth[dir]
      var dxm2 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir]
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]

      Fluid[c].[TENOCoeffsPlus]  = array(0.0, 1.0, 0.0, 0.0)
      Fluid[c].[TENOCoeffsMinus] = array(1.0, 0.0, 0.0, 0.0)

      Fluid[c].[BCStencil] = true
   end
   return CorrectMetricRightMinusOneCollocated
end)

-- dxm2:     |-----------------------------------------|
-- dxm1:                   |---------------------------|
--   dx:                                 |-------------|
-- dxp1:                                               |-------------|
-- dxp2:                                               |---------------------------|
--                 c-2           c-1            c     x=0    c+1           c+2
--           |------x------|------x------|------x------|------x------|------x------|
--
-- Plus reconstruction:
-- 1st:                           o-------------o-----> <-----o
-- 2nd:                                         o-----> <-----o-------------o
-- 3rd:                                         does not exist
-- 4th:                                         does not exist
--
-- Minus reconstruction:
-- 1st:                                         o-----> <-----o-------------o
-- 2nd:                           o-------------o-----> <-----o
-- 3rd:                                         does not exist
-- 4th:                                         does not exist

local mkCorrectMetricRightMinusTwoCollocated = terralib.memoize(function(sdir)
   local CorrectMetricRightMinusTwoCollocated

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
   end

   local __demand(__inline)
   task GetTENOCoeffs(dxm3 : double, dxm2 : double, dxm1 : double, dxp1 : double, dxp2 : double, ReconCoeffs: double[nStencils*6])

      -- TENO coefficinets excluding the first and last cell
      -- (Fourth order accurate)
      var stencil = [MATH.mkReconCoeff(4)](array(dxm2, dxm1, 0.0, dxp1, dxp2), 0.0)

      var LU  : LUdec
      var b : double[nStencils]
      for i=0, nStencils do
         for j=0, nStencils do
            LU.A[i*nStencils+j] = ReconCoeffs[j*6+i+1]
         end
         b[i] = stencil[i]
      end
      LU = ludcmp(LU)
      return lubksb(LU, b)

   end

   __demand(__inline)
   task CorrectMetricRightMinusTwoCollocated(Fluid : region(ispace(int3d), Fluid_columns),
                                             cm2 : int3d, cm1 : int3d, c : int3d, cp1 : int3d, cp2 : int3d)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil])
   do
      -- Face reconstruction operators
      var dxm3 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir] - Fluid[cm2].cellWidth[dir]
      var dxm2 = -Fluid[c  ].cellWidth[dir] - Fluid[cm1].cellWidth[dir]
      var dxm1 = -Fluid[c  ].cellWidth[dir]
      var dxp1 =  Fluid[cp1].cellWidth[dir]
      var dxp2 =  Fluid[cp1].cellWidth[dir] + Fluid[cp2].cellWidth[dir]

      Fluid[c].[TENOCoeffsPlus]  = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, dxp2, Fluid[c].[reconPlus])
      Fluid[c].[TENOCoeffsMinus] = GetTENOCoeffs(dxm3, dxm2, dxm1, dxp1, dxp2, Fluid[c].[reconMinus])

      Fluid[c].[BCStencil] = true
   end
   return CorrectMetricRightMinusTwoCollocated
end)

Exports.mkCorrectGhostMetric = terralib.memoize(function(sdir)
   local CorrectGhostMetric

   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   local is_PosGhost
   local is_NegGhost
   local mk_cm2
   local mk_cm1
   local mk_cp1
   local mk_cp2
   local mk_cp3
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
      is_PosGhost = MACRO.is_xPosGhost
      is_NegGhost = MACRO.is_xNegGhost
      mk_cm2 = function(c, b) return rexpr (c+{-2, 0, 0}) % b end end
      mk_cm1 = function(c, b) return rexpr (c+{-1, 0, 0}) % b end end
      mk_cp1 = function(c, b) return rexpr (c+{ 1, 0, 0}) % b end end
      mk_cp2 = function(c, b) return rexpr (c+{ 2, 0, 0}) % b end end
      mk_cp3 = function(c, b) return rexpr (c+{ 3, 0, 0}) % b end end
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
      is_PosGhost = MACRO.is_yPosGhost
      is_NegGhost = MACRO.is_yNegGhost
      mk_cm2 = function(c, b) return rexpr (c+{ 0,-2, 0}) % b end end
      mk_cm1 = function(c, b) return rexpr (c+{ 0,-1, 0}) % b end end
      mk_cp1 = function(c, b) return rexpr (c+{ 0, 1, 0}) % b end end
      mk_cp2 = function(c, b) return rexpr (c+{ 0, 2, 0}) % b end end
      mk_cp3 = function(c, b) return rexpr (c+{ 0, 3, 0}) % b end end
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
      is_PosGhost = MACRO.is_zPosGhost
      is_NegGhost = MACRO.is_zNegGhost
      mk_cm2 = function(c, b) return rexpr (c+{ 0, 0,-2}) % b end end
      mk_cm1 = function(c, b) return rexpr (c+{ 0, 0,-1}) % b end end
      mk_cp1 = function(c, b) return rexpr (c+{ 0, 0, 1}) % b end end
      mk_cp2 = function(c, b) return rexpr (c+{ 0, 0, 2}) % b end end
      mk_cp3 = function(c, b) return rexpr (c+{ 0, 0, 3}) % b end end
   end

   __demand(__parallel, __cuda, __leaf)
   task CorrectGhostMetric(Fluid : region(ispace(int3d), Fluid_columns),
                           Fluid_bounds : rect3d,
                           BCLeft : int32, BCRight : int32,
                           Grid_Bnum : int32, Grid_Num : int32)
   where
      reads(Fluid.cellWidth),
      reads writes(Fluid.{[reconPlus], [reconMinus]}),
      writes(Fluid.{[TENOCoeffsPlus], [TENOCoeffsMinus]}),
      writes(Fluid.[BCStencil]),
      writes(Fluid.[interp]),
      writes(Fluid.[deriv]),
      writes(Fluid.[grad])
   do
      __demand(__openmp)
      for c in Fluid do
         var cm2 = [mk_cm2(rexpr c end, rexpr Fluid_bounds end)];
         var cm1 = [mk_cm1(rexpr c end, rexpr Fluid_bounds end)];
         var cp1 = [mk_cp1(rexpr c end, rexpr Fluid_bounds end)];
         var cp2 = [mk_cp2(rexpr c end, rexpr Fluid_bounds end)];
         var cp3 = [mk_cp3(rexpr c end, rexpr Fluid_bounds end)];

         -- Left side
         if is_NegGhost(c, Grid_Bnum) then
            if (BCLeft == SCHEMA.FlowBC_Dirichlet or
                BCLeft == SCHEMA.FlowBC_AdiabaticWall or
                BCLeft == SCHEMA.FlowBC_IsothermalWall or
                BCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricLeftStaggered(sdir)](Fluid, c, cp1)

            elseif (BCLeft == SCHEMA.FlowBC_NSCBC_Inflow or
                    BCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then

               [mkCorrectMetricLeftCollocated(sdir)](Fluid, c, cp1, cp2, cp3)

            end
         elseif is_NegGhost(cm1, Grid_Bnum) then
            if (BCLeft == SCHEMA.FlowBC_Dirichlet or
                BCLeft == SCHEMA.FlowBC_AdiabaticWall or
                BCLeft == SCHEMA.FlowBC_IsothermalWall or
                BCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricLeftPlusOneStaggered(sdir)](Fluid, c, cp1, cp2, cp3)

            elseif (BCLeft == SCHEMA.FlowBC_NSCBC_Inflow or
                    BCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then

               [mkCorrectMetricLeftPlusOneCollocated(sdir)](Fluid, cm1, c, cp1, cp2, cp3)

            end
         elseif is_NegGhost(cm2, Grid_Bnum) then
            if (BCLeft == SCHEMA.FlowBC_Dirichlet or
                BCLeft == SCHEMA.FlowBC_AdiabaticWall or
                BCLeft == SCHEMA.FlowBC_IsothermalWall or
                BCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricLeftPlusTwoStaggered(sdir)](Fluid, cm1, c, cp1, cp2, cp3)

            end
         end

         -- Right side
         if is_PosGhost(c, Grid_Bnum, Grid_Num) then
            if (BCRight == SCHEMA.FlowBC_Dirichlet or
                BCRight == SCHEMA.FlowBC_AdiabaticWall or
                BCRight == SCHEMA.FlowBC_IsothermalWall or
                BCRight == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricRightStaggered(sdir)](Fluid, cm1, c)

            elseif (BCRight == SCHEMA.FlowBC_NSCBC_Inflow or
                    BCRight == SCHEMA.FlowBC_NSCBC_Outflow) then

               [mkCorrectMetricRightCollocated(sdir)](Fluid, cm1, c)

            end
         elseif is_PosGhost(cp1, Grid_Bnum, Grid_Num) then
            if (BCRight == SCHEMA.FlowBC_Dirichlet or
                BCRight == SCHEMA.FlowBC_AdiabaticWall or
                BCRight == SCHEMA.FlowBC_IsothermalWall or
                BCRight == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricRightMinusOneStaggered(sdir)](Fluid, cm1, c)

            elseif (BCRight == SCHEMA.FlowBC_NSCBC_Inflow or
                    BCRight == SCHEMA.FlowBC_NSCBC_Outflow) then

               [mkCorrectMetricRightMinusOneCollocated(sdir)](Fluid, cm2, cm1, c, cp1)

            end
         elseif is_PosGhost(cp2, Grid_Bnum, Grid_Num) then
            if (BCRight == SCHEMA.FlowBC_Dirichlet or
                BCRight == SCHEMA.FlowBC_AdiabaticWall or
                BCRight == SCHEMA.FlowBC_IsothermalWall or
                BCRight == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricRightMinusTwoStaggered(sdir)](Fluid, cm2, cm1, c, cp1)

            elseif (BCRight == SCHEMA.FlowBC_NSCBC_Inflow or
                    BCRight == SCHEMA.FlowBC_NSCBC_Outflow) then

               [mkCorrectMetricRightMinusTwoCollocated(sdir)](Fluid, cm2, cm1, c, cp1, cp2)

            end
         elseif is_PosGhost(cp3, Grid_Bnum, Grid_Num) then
            if (BCRight == SCHEMA.FlowBC_Dirichlet or
                BCRight == SCHEMA.FlowBC_AdiabaticWall or
                BCRight == SCHEMA.FlowBC_IsothermalWall or
                BCRight == SCHEMA.FlowBC_SuctionAndBlowingWall) then

               [mkCorrectMetricRightMinusThreeStaggered(sdir)](Fluid, cm2, cm1, c, cp1, cp2)

            end
         end
      end
   end
   return CorrectGhostMetric
end)

return Exports end

