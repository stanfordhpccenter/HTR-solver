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

local floor = regentlib.floor(double)
local ceil  = regentlib.ceil(double)
local fabs  = regentlib.fabs(double)
local max   = regentlib.fmax
local min   = regentlib.fmin
local pow   = regentlib.pow(double)

-- Load data types
local TYPES = terralib.includec("math_utils.h")

-- Row by column multiplication
Exports.mkMatMul = terralib.memoize(function(n)
   local matMul

   local __demand (__inline)
   task matMul(x : double[n],
               A : double[n*n])
      var b : double[n]
      for i=0, n do
         b[i] = 0.0
         for j=0, n do
            b[i] += A[i*n+j]*x[j]
         end
      end
      return b
   end
   return matMul
end)

-- LU decomposition
Exports.mkLUdec = terralib.memoize(function(n)
   local LUdec, ludcmp, lubksb

   -- Data structures
   local struct LUdec {
      A    : double[n*n]
      ind  : int[n]
      sing : bool
   }

   -- Computes LU decomposition
   local __demand (__inline)
   task ludcmp(LU : LUdec)

      var TINY = 1.0e-20
      var imax = 0
      var vv : double[n]
      LU.sing = false

      for i = 0, n do
         var big = 0.0
         for j = 0, n do
            big max= fabs(LU.A[i*n+j])
         end
--    TODO : assertion is not supported by CUDA code generator
--           emit a bool for now
--         regentlib.assert(big ~= 0.0, "Singular matrix in ludcmp")
         if (big == 0.0) then
            LU.sing = true
         end
         vv[i] = 1.0/big
      end
      for j = 0, n do
         for i = 0, j do
            var sum = LU.A[i*n+j]
            for k = 0, i do
               sum -= LU.A[i*n+k]*LU.A[k*n+j]
            end
            LU.A[i*n+j] = sum
         end
         var big = 0.0
         for i = j, n do
            var sum = LU.A[i*n+j]
            for k = 0, j do
               sum -= LU.A[i*n+k]*LU.A[k*n+j]
            end
            LU.A[i*n+j] = sum
            var dum = vv[i]*fabs(sum)
            if (dum >= big) then
               big=dum
               imax=i
            end
         end
         if (j ~= imax) then
            for k = 0, n do
               var dum = LU.A[imax*n+k]
               LU.A[imax*n+k ]= LU.A[j*n+k]
               LU.A[j*n+k] = dum
            end
            vv[imax]=vv[j]
         end
         LU.ind[j] = imax
         if (LU.A[j*n+j] == 0.0) then
            LU.A[j*n+j] = TINY
         end
         if (j ~= n-1) then
            var dum=1.0/(LU.A[j*n+j])
            for i = j+1, n do
               LU.A[i*n+j] *= dum
            end
         end
      end
      return LU
   end

   -- Backsubstitutes in the LU decomposed matrix
   local __demand (__inline)
   task lubksb(LU : LUdec,
                b : double[n])
      var ii = 0
      for i = 0, n do
         var ip = LU.ind[i]
         var sum = b[ip]
         b[ip] = b[i]
         if (ii ~= 0) then
            for j = ii-1, i do
               sum -= LU.A[i*n+j]*b[j]
            end
         elseif (sum ~= 0.0) then
            ii = i+1
         end
         b[i]=sum
      end
      for i = n-1, -1, -1 do
         var sum=b[i]
         for j = i+1, n do
            sum -= LU.A[i*n+j]*b[j]
         end
         b[i] = sum/LU.A[i*n+i]
      end
      return b
   end

   return {LUdec, ludcmp, lubksb}
end)

-- Matrix inversion using Gauss elimination
local mkInverseMatrix = terralib.memoize(function(n)
   local InverseMatrix

   local __demand(__inline)
   task InverseMatrix(A : double[n*n])
      var B : double[n*n]
      for i=0, n do
         for j=0, n do
            B[i*n+j] = 0.0
         end
      end

     -- Forward elimination
      for i=0, n do
         B[i*n+i] = 1.0
         var Ainv = 1.0/A[i*n+i]
         for j=0, n do
            B[i*n+j] *= Ainv
            A[i*n+j] *= Ainv
         end
         for l=i+1, n do
            var factor = A[l*n+i]
            for j=0, n do
               B[l*n+j] -= factor*B[i*n+j]
               A[l*n+j] -= factor*A[i*n+j]
            end
         end
      end

      -- Backward substitution
      for i = n-1, -1, -1 do
         for l=i+1, n do
            var factor = A[i*n+l]
            for j=0, n do
               B[i*n+j] -= factor*B[l*n+j]
            end
         end
      end

      return B
   end
   return InverseMatrix
end)

--HO finite volume reconstruction
Exports.mkReconCoeff = terralib.memoize(function(n)
   local reconCoeff

   __demand(__inline)
   task reconCoeff(xc : double[n+1], xp : double)
      -- Form the matrix
      var A : double[n*n]
      for i=0, n do
         for j=0, n do
            A[i*n+j] = (pow(xc[i+1], n-j) - pow(xc[i], n-j))/((n-j)*(xc[i+1] - xc[i]))
         end
      end

      -- Invert it
      var B = [mkInverseMatrix(n)](A)

      -- Compute metrics
      var coeff : double[n]
      for i=0, n do
         coeff[i] = 0.0
         for j=0, n do
            coeff[i] += B[j*n+i]*pow(xp, (n-(j+1)))
         end
      end
      return coeff
   end

   return reconCoeff
end)

--HO finite volume reconstruction with left Dirichlet BC
Exports.mkReconCoeffLeftBC = terralib.memoize(function(n)
   local reconCoeff

   __demand(__inline)
   task reconCoeff(xc : double[n], xp : double)
      -- Form the matrix
      var A : double[n*n]
      for j=0, n do
         A[j] = pow(xc[0], n-j-1)
      end
      for i=1, n do
         for j=0, n do
            A[i*n+j] = (pow(xc[i], n-j) - pow(xc[i-1], n-j))/((n-j)*(xc[i] - xc[i-1]))
         end
      end

      -- Invert it
      var B = [mkInverseMatrix(n)](A)

      -- Compute metrics
      var coeff : double[n]
      for i=0, n do
         coeff[i] = 0.0
         for j=0, n do
            coeff[i] += B[j*n+i]*pow(xp, (n-(j+1)))
         end
      end
      return coeff
   end

   return reconCoeff
end)

--HO finite volume reconstruction with right Dirichlet BC
Exports.mkReconCoeffRightBC = terralib.memoize(function(n)
   local reconCoeff

   __demand(__inline)
   task reconCoeff(xc : double[n], xp : double)
      -- Form the matrix
      var A : double[n*n]
      for i=0, n-1 do
         for j=0, n do
            A[i*n+j] = (pow(xc[i+1], n-j) - pow(xc[i], n-j))/((n-j)*(xc[i+1] - xc[i]))
         end
      end
      for j=0, n do
         A[(n-1)*n+j] = pow(xc[n-1], n-j-1)
      end

      -- Invert it
      var B = [mkInverseMatrix(n)](A)

      -- Compute metrics
      var coeff : double[n]
      for i=0, n do
         coeff[i] = 0.0
         for j=0, n do
            coeff[i] += B[j*n+i]*pow(xp, (n-(j+1)))
         end
      end
      return coeff
   end

   return reconCoeff
end)

-- Implicit Rosenbrock solver
-- See Numerical recipes in c for reference
Exports.mkRosenbrock = terralib.memoize(function(nEq, Fields, Vars, Unkowns, Data, rhs)
   local Rosenbrock

   -- Algorithm paramenters

   local MAXSTP = 100000
   local MAXITS = 100
   local TOL    = 1e-10
   local SAFETY = 0.9
   local GROW   = 1.5
   local PGROW  =-0.25
   local SHRNK  = 0.5
   local PSHRNK =-1.0/3.0
   local ERRCON = 0.1296
   local GAM    = 1.0/2.0
   local A21    = 2.0
   local A31    = 48.0/25.0
   local A32    = 6.0/25.0
   local C21    =-8.0
   local C31    = 372.0/25.0
   local C32    = 12.0/5.0
   local C41    =-112.0/125.0
   local C42    =-54.0/125.0
   local C43    =-2.0/5.0
   local B1     = 19.0/9.0
   local B2     = 1.0/2.0
   local B3     = 25.0/108.0
   local B4     = 125.0/108.0
   local E1     = 17.0/54.0
   local E2     = 7.0/36.0
   local E3     = 0.0
   local E4     = 125.0/108.0
   local C1X    = 1.0/2.0
   local C2X    =-3.0/2.0
   local C3X    = 121.0/50.0
   local C4X    = 29.0/250.0
   local A2X    = 1.0
   local A3X    = 3.0/5.0

   -- Computes the jacobian with second order finite difference
   local __demand (__inline)
   task GetJacobian(Mesh : region(ispace(int3d), Fields),
                       c : int3d,
                    data : Data)
   where
      reads writes(Mesh.[Vars])
   do

      var EPS = 1.0e-6
      var DEL = 1.0e-14

      var tmp = Mesh[c].[Unkowns]

      var Jac : double[nEq*nEq]

      for j = 0, nEq do
         var h = Mesh[c].[Unkowns][j]*EPS + DEL
         Mesh[c].[Unkowns][j] = tmp[j] + h
         var hp = Mesh[c].[Unkowns][j] - tmp[j]
         var fp = rhs(Mesh, c, data)
         Mesh[c].[Unkowns][j] = tmp[j] - h
         var hm = tmp[j] - Mesh[c].[Unkowns][j]
         var fm = rhs(Mesh, c, data)
         Mesh[c].[Unkowns][j] = tmp[j]
         for i = 0, nEq do
            Jac[i*nEq+j] = (fp[i] - fm[i])/(hp + hm)
         end
      end
      return Jac
   end

   -- LU decomposition tasks
   local LUdec, ludcmp, lubksb = unpack(Exports.mkLUdec(nEq))

   __demand (__inline)
   task Rosenbrock( Mesh : region(ispace(int3d), Fields),
                       c : int3d,
                   dtTry : double,
                    DelT : double,
                    data : Data)
   where
      reads writes(Mesh.[Vars])
   do
      var err   : double[nEq]
      var g1    : double[nEq]
      var g2    : double[nEq]
      var g3    : double[nEq]
      var g4    : double[nEq]

      var finish = false

      var time = 0.0
      var dt = dtTry

      var fail = 0

      for step = 0, MAXSTP do

         var t0 = time
         var dtNext : double

         var Jac = GetJacobian(Mesh, c, data)
         var dx = rhs(Mesh, c, data)
         var xsav = Mesh[c].[Unkowns]
         var dxsav = dx

         for jtry = 0, MAXITS do
            var LU  : LUdec
            for i = 0, nEq do
               for j = 0, nEq do
                  LU.A[i*nEq+j] = -Jac[i*nEq+j]
               end
               LU.A[i*nEq+i] = LU.A[i*nEq+i] + 1.0/(GAM*dt)
            end
            LU = ludcmp(LU)
            if ( LU.sing == true ) then
               fail = 1
               break
            end
            for i = 0, nEq do
               g1[i] = dxsav[i]+dt*C1X*dx[i]
            end
            g1 = lubksb(LU, g1)
            for i = 0, nEq do
               Mesh[c].[Unkowns][i] = xsav[i]+A21*g1[i]
            end
            time = t0+A2X*dt
            dx = rhs(Mesh, c, data)
            for i = 0, nEq do
               g2[i] = dx[i]+dt*C2X*dx[i]+C21*g1[i]/dt
            end
            g2 = lubksb(LU, g2)
            for i = 0, nEq do
               Mesh[c].[Unkowns][i] = xsav[i]+A31*g1[i]+A32*g2[i]
            end
            time = t0+A3X*dt
            dx = rhs(Mesh, c, data)
            for i = 0, nEq do
               g3[i] = dx[i]+dt*C3X*dx[i]+(C31*g1[i]+C32*g2[i])/dt
            end
            g3 = lubksb(LU, g3)
            for i = 0, nEq do
               g4[i] = dx[i]+dt*C4X*dx[i]+(C41*g1[i]+C42*g2[i]+C43*g3[i])/dt
            end
            g4 = lubksb(LU ,g4)
            for i = 0, nEq do
               Mesh[c].[Unkowns][i] = xsav[i]+B1*g1[i]+B2*g2[i]+B3*g3[i]+B4*g4[i]
               err[i] = E1*g1[i]+E2*g2[i]+E3*g3[i]+E4*g4[i]
            end
            time = t0+dt
-- TODO : assertion is not supported by CUDA code generator
--        emit an int for now
--            regentlib.assert(time ~= t0, "Stepsize not significant in Rosenbrock")
            if ( time == t0 ) then
               fail = 2
               break
            end

            var errmax = 0.0
            for i = 0, nEq do
               errmax max= fabs(err[i])
            end
            errmax /= TOL
            if (errmax <= 1.0 or finish) then
               dtTry = dt
--               c.printf("Rosenbrock converged with dt = %g and errmax = %g\n", dt, errmax*TOL)
               if (errmax > ERRCON) then
                  dtNext = SAFETY*dt*pow(errmax,PGROW)
               else
                  dtNext = GROW*dt
               end
               break
            else
               dtNext = SAFETY*dt*pow(errmax,PSHRNK)
               if (dt >= 0.0 ) then
                  dt = max(dtNext,SHRNK*dt)
               else
                  dt = min(dtNext,SHRNK*dt)
               end
            end
-- TODO : assertion is not supported by CUDA code generator
--        emit an int for now
--            regentlib.assert(jtry ~= MAXITS-1, "Exceeded MAXITS in Rosenbrock")
            if (jtry == MAXITS-1) then fail = 3 end
         end
         if ( DelT == time ) then finish = true end
         if finish then break end
         if ( fail==1 ) then break end
         if ( dtNext*1.5 > DelT-time ) then
            -- Force the algorithm to integrate till DelT
            dt = DelT-time
            finish = true
         else
            dt = dtNext
         end
-- TODO : assertion is not supported by CUDA code generator
--        emit an int for now
--         regentlib.assert(step ~= MAXSTP-1, "Exceeded MAXSTP in Rosenbrock")
         if ( step == MAXSTP-1 ) then fail = 4 end
      end
      return fail
   end
   return Rosenbrock
end)

-- Fast interpolation using the integer domain
Exports.mkFastInterp = terralib.memoize(function(SrcType, xfld)
   local FastInterpData, FastInterpType
   local FastInterpInitData, FastInterpInitRegion
   local FastInterpFindIndex, FastInterpGetWeight
   local eps = 1e-6

   local FastInterpData = TYPES.FastInterpData
   local FastInterpType = TYPES.FastInterpType

   -- Initializes data structure
   local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
   task FastInterpInitData(src : region(ispace(int1d), SrcType))
   where
      reads(src.[xfld])
   do
      var data : FastInterpData
      if src.volume > 1 then
         var xmin =  math.huge
         var xmax = -math.huge
         var dxmin = math.huge
         __demand(__openmp)
         for c in src do
            xmin min= src[c].[xfld]
            xmax max= src[c].[xfld]
            if c < src.bounds.hi then
               dxmin min= src[c+1].[xfld] - src[c].[xfld]
            end
         end
         regentlib.assert(dxmin >= 0.0, "FastInterpInitData: something wrong in the input region")
         data.xmin = xmin
         data.xmax = xmax
         data.small = eps*(data.xmax - data.xmin)
         -- ensure at least 2 points per interval in src
         data.nloc = ceil(2.0*(data.xmax-data.xmin)/dxmin)
         -- Size of the uniform grid
         data.dxloc = (data.xmax-data.xmin)/(data.nloc-1)
         data.idxloc = 1.0/data.dxloc
      else
         data.xmin = src[0].[xfld]
         data.xmax = src[0].[xfld]
         data.small = eps
         data.nloc = 2
         data.dxloc = 0.0
         data.idxloc = 0.0
      end
      return data
   end

   -- Initializes region
   local --__demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
   task FastInterpInitRegion(r   : region(ispace(int1d), FastInterpType),
                             src : region(ispace(int1d), SrcType),
                             d : FastInterpData)
   where
      reads(src.[xfld]),
      reads writes(r.{xloc, iloc})
   do
      -- Initialize xloc and iloc
      __demand(__openmp)
      for c in r do
         r[c].xloc = d.xmin + float(c)*d.dxloc
         var i : int1d
         for c1 in src do
            i = c1
            var cp1 = min(c1+int1d(1), src.bounds.hi)
            if (r[c].xloc <= src[cp1].[xfld]) then break end
         end
         i min= src.bounds.hi-int1d(1)
         var ip1 = i+int1d(1)
         var w = (src[ip1].[xfld] - r[c].xloc)/(src[ip1].[xfld]-src[i].[xfld])
         r[c].iloc = w*float(i)+(1.0 - w)*float(ip1)
      end

      var dimin = math.huge
      for c in r do
         if c < r.bounds.hi then
            dimin min= r[c+1].iloc - r[c].iloc
         end
      end
      if r.volume==1 then dimin = 0 end

      -- Correct iloc to pass though src.[xfld]'s
      r[0].iloc = 0.0
--      __demand(__openmp)
      for c in src do
         if c > int1d(0) then
            var csi = int1d(0)
            for c1 in r do
               csi = c1
               var cp1 = min(c1+int1d(1), r.bounds.hi)
               if (src[c].[xfld] <= r[cp1].xloc) then break end
            end
            var cp1 = min(csi+int1d(1), r.bounds.hi)
            r[csi].iloc = float(c) + dimin*d.idxloc*(r[csi].xloc - src[c].[xfld])
            r[cp1].iloc = float(c) + dimin*d.idxloc*(r[cp1].xloc - src[c].[xfld])
         end
      end
      r[r.bounds.hi].iloc = float(src.bounds.hi)
   end

   -- Finds index of first element on the left
   __demand(__inline)
   task FastInterpFindIndex(x : double,
                            r : region(ispace(int1d), FastInterpType),
                            d : FastInterpData)
   where
      reads(r.{iloc, xloc})
   do
      x max = d.xmin+d.small
      x min = d.xmax-d.small
      var k = int1d(floor((x-d.xmin)*d.idxloc))
      var kp1 = k+int1d(1)
      return int1d(floor(r[k].iloc + (r[kp1].iloc-r[k].iloc)*
                                     (          x-r[k].xloc)*d.idxloc))
   end

   -- Compute linear interpolation weight for point on the left
   __demand(__inline)
   task FastInterpGetWeight(x : double, xm : double, xp : double)
      return (xp - x)/(xp - xm)
   end

   return {FastInterpData, FastInterpType,
           FastInterpInitData, FastInterpInitRegion,
           FastInterpFindIndex, FastInterpGetWeight}
end)

return Exports
