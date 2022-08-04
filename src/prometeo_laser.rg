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

return function(SCHEMA, MIX, Fluid_columns, zones_partitions, ghost_partitions, ATOMIC) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local UTIL = require 'util'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-- Math
local C = regentlib.c
local sin  = regentlib.sin(double)
local cos  = regentlib.cos(double)
local exp  = regentlib.exp(double)
local sqrt = regentlib.sqrt(double)
local tanh = regentlib.tanh(double)
local atanh = regentlib.atanh(double)
local log = regentlib.log(double)
local fabs = regentlib.fabs(double)
local tan = regentlib.tan(double,double)
local atan2 = regentlib.atan2(double,double)
local pow  = regentlib.pow(double)
local PI = CONST.PI

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

--local IO = (require 'prometeo_IO')(SCHEMA)
local MATH = require 'math_utils'
local LUdec, ludcmp, lubksb = unpack(MATH.mkLUdec(5))
local format = require("std/format")

-- Atomic switch
local Fluid = regentlib.newsymbol(region(ispace(int3d), Fluid_columns), "Fluid")
local coherence_mode
if ATOMIC then
   coherence_mode = regentlib.coherence(regentlib.atomic,    Fluid, "Conserved_t")
else
   coherence_mode = regentlib.coherence(regentlib.exclusive, Fluid, "Conserved_t")
end

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

local fspace LaserStruct(Fluid : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   Laser_tiles : ispace(int3d),
   p_Laser : partition(disjoint, Fluid, tiles),
}
Exports.LaserStruct = LaserStruct

function Exports.mkLaserList()
   return {
      LaserData = regentlib.newsymbol()
   }
end

-------------------------------------------------------------------------------
-- ROUTINES
-------------------------------------------------------------------------------
function Exports.DeclSymbols(s, Grid, Fluid, tiles, p_All, config)
   return rquote

      -- Init partitions and tiles
      var tmp : LaserStruct(Fluid, tiles)
      var [s.LaserData] = tmp

      -- Extract parameters
      var volume : SCHEMA.Volume
      if (config.Flow.laser.type == SCHEMA.LaserModel_Algebraic) then
         volume = config.Flow.laser.u.Algebraic.volume
      elseif (config.Flow.laser.type == SCHEMA.LaserModel_GeometricKernel) then
         volume = config.Flow.laser.u.GeometricKernel.volume
      end

      -- Which model?
      if (config.Flow.laser.type == SCHEMA.LaserModel_Algebraic or
          config.Flow.laser.type == SCHEMA.LaserModel_GeometricKernel) then

         -- Partition the Fluid region based on the specified volume
         var laser_coloring = regentlib.c.legion_domain_point_coloring_create()

         -- Clip rectangle from the input
         volume.fromCell[0] max= 0
         volume.fromCell[1] max= 0
         volume.fromCell[2] max= 0
         volume.uptoCell[0] min= config.Grid.xNum + 2*Grid.xBnum
         volume.uptoCell[1] min= config.Grid.yNum + 2*Grid.yBnum
         volume.uptoCell[2] min= config.Grid.zNum + 2*Grid.zBnum
         -- add to the coloring
         var rect = rect3d{
            lo = int3d{volume.fromCell[0], volume.fromCell[1], volume.fromCell[2]},
            hi = int3d{volume.uptoCell[0], volume.uptoCell[1], volume.uptoCell[2]}}
         regentlib.c.legion_domain_point_coloring_color_domain(laser_coloring, int1d(0), rect)

         -- Make partitions of Fluid
         var Fluid_Laser = partition(disjoint, Fluid, laser_coloring, ispace(int1d, 1))[0]

         -- Split over tiles
         var p_Laser = Fluid & (Fluid_Laser & p_All);

         -- Attach name for mapping
         [UTIL.emitPartitionNameAttach(p_Laser, "p_Laser")];

         -- Destroy color
         regentlib.c.legion_domain_point_coloring_destroy(laser_coloring)

         -- Extract relevant index space
         var Laser_tiles = [UTIL.mkExtractRelevantIspace("partition", int3d, int3d, Fluid_columns)]
                                   (Fluid, p_Laser)

         s.LaserData = [LaserStruct(Fluid, tiles)] {
            Laser_tiles = Laser_tiles,
            p_Laser = p_Laser
         }
      end

   end
end

-- Algebraic laser model
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.AddLaserAlgebraic([Fluid],
                               Dimension : int32,
                               Amplitude : double,
                               Center : double[3],
                               Radius : double,
                               Delay : double,
                               Duration: double,
                               time: double)
where
   reads(Fluid.centerCoordinates),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do
   __demand(__openmp)
   for c in Fluid do
      var d0 = Fluid[c].centerCoordinates[0] - Center[0]
      var d1 = Fluid[c].centerCoordinates[1] - Center[1]
      var d2 = Fluid[c].centerCoordinates[2] - Center[2]
      var R  = Radius
      var R2 = R*R
      var D  = Duration/2.0
      var D2 = D*D
      var T  = time-Delay
      if (Dimension == 2) then
         var DSqr = d0*d0 + d1*d1
         var skernel = 1.0/R2/2.0/PI*exp(-DSqr/R2/2.0)
         var tkernel = 1.0/D/sqrt(2.0*PI)*exp(-T*T/D2/2.0)
         Fluid[c].Conserved_t[irE] += Amplitude * skernel * tkernel
      else
         var DSqr = d0*d0 + d1*d1 + d2*d2
         var skernel = 1.0/R2/R/2.0/PI/sqrt(2.0*PI)*exp(-DSqr/R2/2.0)
         var tkernel = 1.0/D/sqrt(2.0*PI)*exp(-T*T/D2/2.0)
         Fluid[c].Conserved_t[irE] += Amplitude * skernel * tkernel
      end
   end
end

-- Helper functions for AddLaserGeometricKernel
local __demand(__inline)
task tanhStep(x : double, x0 : double, w : double)
   var p = 0.9
   var s = 2.0/w*atanh(2.0*p-1.0)
   return 0.5*tanh(s*(x-x0))+0.5
end

local __demand(__inline)
task tanhBump(x : double, x1 : double, x2 : double, w1 : double, w2 : double)
   return tanhStep(x,x1,w1) + tanhStep(x,x2,-w2) - 1.0
end

local __demand(__inline)
task tanhEvenStep(x : double, x0 : double, w : double) : double
   return tanhBump(x,-x0,x0,w,w)
end

local __demand(__inline)
task kernelGeometryFunction(sol : double [5], b : double, x1 : double, x2 : double, x : double,r : double)
   var x1s = sol[0]
   var r1s = sol[1]
   var x2s = sol[2]
   var r2s = sol[3]
   var r2 = sol[4]
   var r1 = b * r2

   var F : double[5]
   F[0] = (x1s-x1)*(x1s-x1) + r1s*r1s - r1*r1
   F[1] = (x2s-x2)*(x2s-x2) + r2s*r2s - r2*r2
   F[2] = (x1-x1s)*(x2s-x1s) - r1s*(r2s-r1s)
   F[3] = (x2-x2s)*(x2s-x1s) - r2s*(r2s-r1s)
   F[4] = (x1s-x)*(r2s-r) - (x2s-x)*(r1s-r)

   return F
end

local __demand(__inline)
task kernelGeometryJacobian(sol : double [5], b : double, x1 : double, x2 : double, x : double, r : double)
   var x1s = sol[0]
   var r1s = sol[1]
   var x2s = sol[2]
   var r2s = sol[3]
   var r2 = sol[4]
   var r1 = b * r2

   var J : double[5*5]

   J[0] = 2.0*(x1s-x1) -- (1,1)
   J[1] = 2.0*r1s -- (1,2)
   J[2] = 0.0 -- (1,3)
   J[3] = 0.0 -- (1,4)
   J[4] = -2.0*b*b*r2 -- (1,5)

   J[5] = 0.0 -- (2,1)
   J[6] = 0.0 -- (2,2)
   J[7] = 2.0*(x2s-x2) -- (2,3)
   J[8] = 2.0*r2s -- (2,4)
   J[9] = -2.0*r2 -- (2,5)

   J[10] = 2.0*x1s-x2s-x1 -- (3,1)
   J[11] = 2.0*r1s-r2s -- (3,2)
   J[12] = x1-x1s -- (3,3)
   J[13] = -r1s -- (3,4)
   J[14] = 0.0 -- (3,5)

   J[15] = x2s-x2 -- (4,1)
   J[16] = r2s -- (4,2)
   J[17] = -2.0*x2s+x1s+x2 -- (4,3)
   J[18] = -2.0*r2s+r1s -- (4,4)
   J[19] = 0.0 -- (4,5)

   J[20] = r2s-r -- (5,1)
   J[21] = -x2s+x -- (5,2)
   J[22] = -r1s+r -- (5,3)
   J[23] = x1s-x -- (5,4)
   J[24] = 0.0 -- (5,5)

   return J
end

local __demand(__inline)
task newtonSolveForKernelGeometry(sol0 : double[5],
                                        maxIter : int,
                                        newtonTol : double,
                                        x1 : double,
                                        x2 : double,
                                        x : double,
                                        r : double,
                                        radiusRatio : double)
   var sol : double[5]
   var err : double[5]
   var LU : LUdec
   var F : double[5]
   var J : double[5*5]
   var dx : double[5]
   var df : double[5]
   var iter = 0
   var maxErr = 0.0
   var success = false
   sol = sol0
   while (not success) do
      F = kernelGeometryFunction(sol,radiusRatio,x1,x2,x,r)
      for i = 0,4 do
         err[i] = fabs(F[i])
         maxErr = max(err[i],maxErr)
      end
      --C.printf('inside newton, iter=%i, sol=%e %e %e %e %e, F=%e %e %e %e %e',
         --iter,sol[0],sol[1],sol[2],sol[3],sol[4],F[0],F[1],F[2],F[3],F[4])
      if (maxErr < newtonTol) then
         success = true
      else
         J = kernelGeometryJacobian(sol,radiusRatio,x1,x2,x,r)
         LU.A = array(J[0],J[1],J[2],J[3],J[4],J[5],J[6],J[7],J[8],J[9],J[10],J[11],J[12],J[13],J[14],J[15],J[16],
            J[17],J[18],J[19],J[20],J[21],J[22],J[23],J[24])
         df = array(F[0],F[1],F[2],F[3],F[4])

         --C.printf('jac=%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e',
            --J[0],J[1],J[2],J[3],J[4],J[5],J[6],J[7],J[8],J[9],J[10],J[11],J[12],J[13],J[14],J[15],J[16],
            --J[17],J[18],J[19],J[20],J[21],J[22],J[23],J[24])

         LU = ludcmp(LU)
         dx = lubksb(LU,df)
         for i = 0,4 do
            sol[i] = sol[i] - dx[i]
         end
         iter = iter + 1
      end
   end

   return sol--, err, iter
end

local __demand(__inline)
task geometricKernelProfile(x_in : double,
                            y : double,
                            z : double,
                            L : double,
                            R1_in : double,
                            R2_in : double,
                            wRatio : double)


   -- Check if kernel is "flipped" and adjust as necessary (this routine assumes R1 > R2)
   var x = x_in
   var R1 = R1_in
   var R2 = R2_in
   if (R2 > R1) then
      x = -x_in
      R1 = R2_in
      R2 = R1_in
   end

   var r = sqrt(y*y + z*z)
   var x1 = -L/2.0 + R1
   var x2 =  L/2.0 - R2
   var radiusRatio = R1/R2

   --C.printf('Start of geometricKernelProfile, x=%e, y=%e, z=%e, L=%e, R1=%e, R2=%e, wRatio=%e\n',
      --x,y,z,L,R1,R2,wRatio)

   -- Initialize
   var w : double
   var theta : double
   var xcb : double
   var rcb : double
   var q0 : double
   var q : double
   var s : double
   var smax : double

   -- Compute signed distance q from CB (contact boundary)
   if (L == 2.0*R1 or L == 2.0*R2) then -- sphere

      -- Safety -- if spherical, set R1=R2
      if (L == 2.0*R1) then
         R2 = R1
      elseif (L == 2.0*R2) then
         R1 = R2
      end

      w = wRatio*R1
      theta = atan2(r,x)
      xcb = R1*cos(theta)
      rcb = R1*sin(theta)
      q0 = R1 -- distance of CB from origin
      q = sqrt(x*x + r*r) - R1 -- <0 inside kernel, >0 outside kernel, 0 at CB

      -- Used to add tangential perturbations
      s = theta*R1
      smax = PI*R1
      --C.printf('SPHERE, x=%e, r=%e, xcb=%e, rcb=%e, q0=%e, q=%e\n',x,r,xcb,rcb,q0,q)
   elseif (R1 == R2) then -- cylindrical capsule (with spherical ends)
      w = wRatio*R1
      if (x < x1) then -- left hemisphere
        theta = atan2(r,x-x1)
        xcb = R1*cos(theta) + x1 -- location of contact boundary at current value of theta
        rcb = R1*sin(theta)      -- location of contact boundary at current value of theta
        q0 = sqrt((xcb-x1)*(xcb-x1) + rcb*rcb) -- distance of CB from (x1,0)
        s = (PI/2.0-theta)*R1 + R2*PI/2.0 + (x2-x1) -- s=[0,1]
        q = sqrt((x-x1)*(x-x1) + r*r) - R1 -- Negative inside kernel, positive outside, zero at CB
      elseif (x > x2) then -- right hemisphere
        theta = atan2(r,x-x2)
        xcb = R2*cos(theta) + x2 -- location of contact boundary at current value of theta
        rcb = R2*sin(theta)      -- location of contact boundary at current value of theta
        q0 = sqrt((xcb-x2)*(xcb-x2) + rcb*rcb) -- distance of CB from (x2,0)
        s = theta*R2 -- s=0 at rightmost point, 1 at leftmost
        q = sqrt((x-x2)*(x-x2) + r*r) - R2 -- Negative inside kernel, positive outside, zero at CB
      else -- Cylindrical section
        xcb = x
        rcb = R1
        q0 = rcb
        s = R2*PI/2.0 + fabs(x-x2)
        q = r - rcb
      end
      --smax = ((PI-theta1s)*R1 + theta2s*R2 + sqrt((r1s-r2s)*(r1s-r2s) + (x2s-x1s)*(x2s-x1s))) -- Total arclength
   else
      var reg2_R = (x2-x1)/(R1-R2) * R2/2.0
      var reg2_x = reg2_R + x2
      var reg1_R = (2.0*reg2_R + x2 - x1)/2.0
      var reg1_x = reg1_R + x1
      --var a = fabs(2./w*atanh(2.*onetenth-1))

      --C.printf('ASYM (before regions) x=%e, r=%e, reg1_x=%e, reg1_R=%e, reg2_x=%e, reg2_R=%e\n',
         --x,r,reg1_x,reg1_R,reg2_x,reg2_R)

      -- Outer region: w=R1, includes left "hemisphere" and infinity in x and r.  Note > sign.
      if ((x-reg1_x)*(x-reg1_x) + r*r > reg1_R*reg1_R) then
        w = wRatio * R1
        q = sqrt((x-x1)*(x-x1) + r*r) - R1
        q0 = R1

      -- Small inner circle: w=R2, contains right "hemisphere".  This is a finite region.
      elseif ((x-reg2_x)*(x-reg2_x) + r*r < reg2_R*reg2_R) then
        w = wRatio * R2
        q = sqrt((x-x2)*(x-x2) + r*r) - R2
        q0 = R2

      -- In-between region enclosed by two tangent circles; contains "conical" region.  Finite region, like a rainbow
      else

         var phi = 2.0 * (PI-atan2(r,x-(reg1_x+reg1_R)))
         var sol0 : double[5]
         sol0[0] = -reg1_R*cos(phi) + reg1_x
         sol0[1] = reg1_R*sin(phi)
         sol0[2] = -reg2_R*cos(phi) + reg2_x
         sol0[3] = reg2_R*sin(phi)
         sol0[4] = sqrt((sol0[2]-x2)*(sol0[2]-x2) + sol0[3]*sol0[3])

         --C.printf('ASYM (before Newton): phi=%e, sol0=%e %e %e %e %e\n',
            --phi,sol0[0],sol0[1],sol0[2],sol0[3],sol0[4])

         var sol : double[5]
         var err : double[5]
         var iter : int
         var newtonTol = 1e-10
         var maxIter = 1000
         sol = newtonSolveForKernelGeometry(sol0,maxIter,newtonTol,x1,x2,x,r,radiusRatio)

         --C.printf('ASYM (after Newton): iter=%i, sol=%e %e %e %e %e\n',iter,
            --sol[0],sol[1],sol[2],sol[3],sol[4])

         var x1s = sol[0]
         var r1s = sol[1]
         var x2s = sol[2]
         var r2s = sol[3]
         var r2 = fabs(sol[4]) -- r2 corresponding to r2 of contour we're on; each contour looks like a kernel
         if (r1s < 0.0) then -- Only in upper half-plane
            x1s = -(x1s-x1) + x1;
            r1s = -r1s;
         end
         if (r2s < 0.0) then -- Only in upper-half plane
            x2s = -(x2s-x2) + x2;
            r2s = -r2s;
         end
         var r1 = r2*radiusRatio

         -- Compute s
         -- r2 specifies which contour we're on.  s specifies tangential distance along that contour
         var s2 = atan2(r2s,x2s-x2)*r2
         var s1 = sqrt((x2s-x1s)*(x2s-x1s) + (r2s-r1s)*(r2s-r1s)) + s2
         var q1 = r1 - R1 -- r1 corresponds to the current contour; R1 corresponds to the prescribed one (CB)
         var q2 = r2 - R2 -- r2 corresponds to the current contour; R2 corresponds to the prescribed one (CB)
         s = sqrt((x-x2s)*(x-x2s) + (r-r2s)*(r-r2s)) + s2

         -- Compute w, q, q0
         -- Note: In the conical section of a variable-thickness profile, q
         -- is NOT the normal distance from CB, but rather distance along
         -- an arc that intersects the CB at a right angle.
         w = wRatio*((s-s1)/(s2-s1)*R2 + (s-s2)/(s1-s2)*R1)
         q = (s-s1)/(s2-s1)*q2 + (s-s2)/(s1-s2)*q1 -- neg inside kernel, pos outside
         q0 = (s-s1)/(s2-s1)*R2 + (s-s2)/(s1-s2)*R1
      end
   end

   -- Compute f\in[0,1], proportional to energy density
   var f = tanhEvenStep(q+q0,q0,w) -- q+q0 = q0 marks CB (i.e. q=0) -- even function of q

   --if (x > L/2.0 and f > 0.5) then
      --C.printf('x=%e, y=%e, z=%e, r=%e, q0=%e, q=%e, f=%e\n',x,y,z,r,q0,q,f)
   --end

   return f
end

-- See Wang, Buchta, Freund, JFM (2020)
__demand(__leaf)
task Exports.computeKernelProfile([Fluid],
                                  config : SCHEMA.Config)
where
   reads(Fluid.centerCoordinates),
   writes(Fluid.kernelProfile),
   [coherence_mode]
do

   -- Extract relevant shape parameters
   var focalLocation = config.Flow.laser.u.GeometricKernel.focalLocation
   var axialLength = config.Flow.laser.u.GeometricKernel.axialLength
   var nearRadius = config.Flow.laser.u.GeometricKernel.nearRadius
   var farRadius = config.Flow.laser.u.GeometricKernel.farRadius
   var dimensions = config.Flow.laser.u.GeometricKernel.dimensions
   var nx = config.Flow.laser.u.GeometricKernel.beamDirection[0]
   var ny = config.Flow.laser.u.GeometricKernel.beamDirection[1]
   var nz = config.Flow.laser.u.GeometricKernel.beamDirection[2]
   var wRatio = 1.0 -- Make this < 1 for a sharper kernel boundary

   -- Check geometry
   if (axialLength < 2.0*nearRadius or axialLength < 2.0*farRadius) then
       var stderr = C.fdopen(2, 'w')
       C.fprintf(stderr, '\n **ERROR** Invalid laser kernel geometry.  L >= 2*R1 and L >= 2*R2 are required.\n')
       C.fflush(stderr)
       C.exit(1)
   end

   -- Normalize normal vector and get rotation angles
   var norm = sqrt(nx*nx + ny*ny + nz*nz)
   nx = nx / norm
   ny = ny / norm
   nz = nz / norm
   var theta = atan2(sqrt(1.0-nx*nx),nx)
   var phi = atan2(nz,ny)

   -- Pre-compute kernel profile
   __demand(__openmp)
   for c in Fluid do
      var x = Fluid[c].centerCoordinates[0] - focalLocation[0]
      var y = Fluid[c].centerCoordinates[1] - focalLocation[1]
      var z = Fluid[c].centerCoordinates[2] - focalLocation[2]

      -- Laser-axis coords
      var xp : double
      var yp : double
      var zp : double
      if (dimensions == 2) then
         xp =  x*cos(theta) + y*sin(theta) -- along laser axis
         yp = -x*sin(theta) + y*cos(theta) -- perp to laser axis
         zp = 0.0
      elseif (dimensions == 3) then
         xp =  x*cos(theta) + y*sin(theta)*cos(phi) + z*sin(theta)*sin(phi) -- along laser axis
         yp = -x*sin(theta) + y*cos(theta)*cos(phi) + z*cos(theta)*sin(phi) -- perp to laser axis
         zp =               - y*sin(phi)            + z*cos(phi)            -- perp to laser axis
      end

      -- Kernel profile (scalar between 0 and 1)
      Fluid[c].kernelProfile = geometricKernelProfile(xp,yp,zp,axialLength,nearRadius,farRadius,wRatio)
   end
end

-- See Wang, Buchta, Freund, JFM (2020)
__demand(__cuda, __leaf)
task Exports.AddLaserGeometricKernel([Fluid],
                                  time: double,
                                  config : SCHEMA.Config)
where
   reads(Fluid.rho),
   reads(Fluid.kernelProfile),
   reads writes(Fluid.Conserved_t),
   [coherence_mode]
do

   -- Get properties
   var peakEdotPerMass = config.Flow.laser.u.GeometricKernel.peakEdotPerMass
   var pulseTime = config.Flow.laser.u.GeometricKernel.pulseTime
   var pulseFWHM = config.Flow.laser.u.GeometricKernel.pulseFWHM

   -- Compute tfac
   var t = time - pulseTime
   var sig_t = pulseFWHM/(2.0*sqrt(2.0*log(2.0)))
   var tfac = exp(-t*t/(2.0*sig_t*sig_t))
   var Edot = tfac*peakEdotPerMass

   --format.println("t={}, E={}, tL={}, fwhm={}, tfac={}, dim={}, L={}, R1={}, R2={}",
      --time, peakEdotPerMass, pulseTime, pulseFWHM, tfac, dimensions, axialLength, nearRadius, farRadius)

   __demand(__openmp)
   for c in Fluid do
      Fluid[c].Conserved_t[irE] += Fluid[c].rho*Edot*Fluid[c].kernelProfile
   end
end

return Exports end
