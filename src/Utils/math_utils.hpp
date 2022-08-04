// Copyright (c) "2019, by Stanford University
//               Developer: Mario Di Renzo
//               Affiliation: Center for Turbulence Research, Stanford University
//               URL: https://ctr.stanford.edu
//               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
//                         HTR solver: An open-source exascale-oriented task-based
//                         multi-GPU high-order code for hypersonic aerothermodynamics.
//                         Computer Physics Communications 255, 107262"
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __MATH_UTILS_HPP__
#define __MATH_UTILS_HPP__

#include "legion.h"

#include "math_utils.h"
#include "my_array.hpp"

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#ifndef __UNROLL__
#ifdef __CUDACC__
#define __UNROLL__ #pragma unroll
#else
#define __UNROLL__
#endif
#endif

#ifndef __CUDACC__

#ifdef DARWIN
#include <cmath>
#endif

using std::max;
using std::min;
#endif

__CUDA_HD__
inline double dot(const double *a, const double *b) {
   return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

template<int n>
__CUDA_HD__
inline void MatMul(const double *A, const double *x, double *r) {
// NOTE: This outer unroll increases the compile time a lot the compile time for n O(100)
//   __UNROLL__
   for (int i=0; i<n; i++) {
      r[i] = 0.0;
      __UNROLL__
      for (int j=0; j<n; j++)
         r[i] += A[i*n+j]*x[j];
   }
}

//-----------------------------------------------------------------------------
// LU decomposition
//-----------------------------------------------------------------------------
// See Numerical recipes in c for reference

template<int n>
struct LUdec {
private:
   MyMatrix<double, n, n>    A;
   MyArray<int, n>         ind;

   static constexpr double TINY = 1.0e-20;
public:
   // Initialize the matrix
   __CUDA_HD__
   inline void init(const MyMatrix<double, n, n> &in) {
      A = in;
   }

   // Provide direct access the matrix
   __CUDA_HD__
   inline double& operator()(const int i, const int j)       { return A(i, j); }
   __CUDA_HD__
   inline double  operator()(const int i, const int j) const { return A(i, j); }

   // Computes LU decomposition
   __CUDA_HD__
   inline void ludcmp(const MyMatrix<double, n, n> &in) {
      A = in;
      ludcmp();
   }

   __CUDA_HD__
   inline void ludcmp() {
      int imax = 0;
      MyArray<double, n> vv;
      for (int i=0; i<n; i++) {
         double big = 0.0;
         __UNROLL__
         for (int j=0; j<n; j++)
            big = max(fabs(A(i,j)), big);
         // Singular matrix in ludcmp
         assert(big != 0.0);
         vv[i] = 1.0/big;
      }
      for (int j=0; j<n; j++) {
         for (int i=0; i<j; i++) {
            double sum = A(i, j);
            for (int k=0; k<i; k++)
               sum -= A(i, k)*A(k, j);
            A(i, j) = sum;
         }
         double big = 0.0;
         for (int i=j; i<n; i++) {
            double sum = A(i, j);
            for (int k=0; k<j; k++)
               sum -= A(i, k)*A(k, j);
            A(i, j) = sum;
            double dum = vv[i]*fabs(sum);
            if (dum >= big) {
               big=dum;
               imax=i;
            }
         }
         if (j != imax) {
            __UNROLL__
            for (int k=0; k<n; k++) {
               const double dum = A(imax, k);
               A(imax, k) = A(j, k);
               A(j,    k) = dum;
            }
            vv[imax]=vv[j];
         }
         ind[j] = imax;
         if (A(j, j) == 0.0)
            A(j, j) = TINY;
         if (j != n-1) {
            const double dum = 1.0/(A(j, j));
            for (int i=j+1; i<n; i++)
               A(i, j) *= dum;
         }
      }
   }

   // Backsubstitutes in the LU decomposed matrix
   __CUDA_HD__
   inline void lubksb(MyArray<double, n> &b) {
      int ii = 0;
      for (int i=0; i<n; i++) {
         const int ip = ind[i];
         double sum = b[ip];
         b[ip] = b[i];
         if (ii != 0)
            for (int j=ii-1; j<i; j++)
               sum -= A(i, j)*b[j];
         else if (sum != 0.0)
            ii = i+1;
         b[i] = sum;
      }
      for (int i=n-1; i>-1; i--) {
         double sum=b[i];
         for (int j = i+1; j<n; j++)
            sum -= A(i, j)*b[j];
         b[i] = sum/A(i, i);
      }
   }
};

//-----------------------------------------------------------------------------
// Implicit Rosenbrock solver
//-----------------------------------------------------------------------------
// See Numerical recipes in c for reference

template<int n>
class Rosenbrock {

public:
   __CUDA_HD__
   virtual ~Rosenbrock() {};

   // RHS function that computes the rhs for a given x
   __CUDA_HD__
   virtual void rhs(MyArray<double, n> &r, const MyArray<double, n> &x) = 0;

   // Main body of the solver
   __CUDA_HD__
   inline void solve(MyArray<double, n> &x, const double dtTry, const double DelT) {
      bool finish = false;
      double time = 0.0;
      double dt = dtTry;

      for (int s = 0; s < MAXSTP; s++) {
         const double t0 = time;
         double dtNext;

         // Update the jacobian matrix
         GetJacobian(x);

         // Update RHS
         rhs(dx, x);
         xsav = x;
         dxsav = dx;

         for (int jtry = 0; jtry < MAXITS; jtry++) {
            // Advance solution
            time = step(x, dt, t0);

            // Check dt size
            if (itercheck(dt, dtNext, finish)) break;

            // It took too many iterations to match the dt
            assert(jtry < MAXITS-1);
         }

         // Check integration time
         if (stepcheck(dt, dtNext, DelT, time, finish)) break;

         // It took too many timesteps
         assert(s < MAXSTP-1);
      }
   };

private:

   // Computes the jacobian with second order finite difference
   __CUDA_HD__
   inline void GetJacobian(MyArray<double, n> &x) {
      static constexpr double EPS = 1.0e-6;
      static constexpr double DEL = 1.0e-14;
      // Store input solution
      MyArray<double, n> tmp = x;
      // Allocate spaces for rhs
      MyArray<double, n> fp;
      MyArray<double, n> fm;
      // Compute the Jacobian
      __UNROLL__
      for (int j=0; j<n; j++) {
         const double h = x[j]*EPS + DEL;
         x[j] = tmp[j] + h;
         const double hp = x[j] - tmp[j];
         rhs(fp, x);
         x[j] = tmp[j] - h;
         const double hm = tmp[j] - x[j];
         rhs(fm, x);
         x[j] = tmp[j];
         __UNROLL__
         for (int i=0; i<n; i++)
            Jac(i, j) = (fp[i] - fm[i])/(hp + hm);
      }
   };

   // Performs the timestep
   __CUDA_HD__
   inline double step(MyArray<double, n> &x, const double dt, const double t0) {
      MyArray<double, n>  g1;
      MyArray<double, n>  g2;
      MyArray<double, n>  g3;
      MyArray<double, n>  g4;

      for (int i=0; i<n; i++) {
         __UNROLL__
         for (int j=0; j<n; j++)
            LU(i, j) = -Jac(i, j);
         LU(i, i) += 1.0/(GAM*dt);
      }
      LU.ludcmp();

      // Sub-step 1
      __UNROLL__
      for (int i=0; i<n; i++)
         g1[i] = dxsav[i]+dt*C1X*dx[i];
      LU.lubksb(g1);
      __UNROLL__
      for (int i=0; i<n; i++)
         x[i] = xsav[i] + A21*g1[i];
      double time = t0+A2X*dt;

      // Sub-step 2
      rhs(dx, x);
      __UNROLL__
      for (int i=0; i<n; i++)
         g2[i] = dx[i] + dt*C2X*dx[i] + C21*g1[i]/dt;
      LU.lubksb(g2);
      __UNROLL__
      for (int i=0; i<n; i++)
         x[i] = xsav[i] + A31*g1[i] + A32*g2[i];
      time = t0+A3X*dt;

      // Sub-step 3
      rhs(dx, x);
      __UNROLL__
      for (int i=0; i<n; i++)
         g3[i] = dx[i] + dt*C3X*dx[i] + (C31*g1[i] + C32*g2[i])/dt;
      LU.lubksb(g3);

      // Sub-step 4
      __UNROLL__
      for (int i=0; i<n; i++)
         g4[i] = dx[i] + dt*C4X*dx[i] + (C41*g1[i] + C42*g2[i] + C43*g3[i])/dt;
      LU.lubksb(g4);
      __UNROLL__
      for (int i=0; i<n; i++)
         x[i] = xsav[i] + B1*g1[i] + B2*g2[i] + B3*g3[i] + B4*g4[i];
      __UNROLL__
      for (int i=0; i<n; i++)
         err[i] = E1*g1[i] + E2*g2[i] + E3*g3[i] + E4*g4[i];
      time = t0 + dt;

      // Stepsize not significant in Rosenbrock
      assert(time != t0);
      return time;
   }

   // checks if subiteration is converged
   __CUDA_HD__
   inline bool itercheck(double &dt, double &dtNext, const bool finish) {
      // Compute the Linfty of the error
      double errmax = 0.0;
      __UNROLL__
      for (int i=0; i<n; i++)
         errmax = max(errmax, fabs(err[i]));
      errmax *= iTOL;

      // Check convergence
      if ((errmax < 1.0) or finish) {
         // The subiterations are converged
         if (errmax > ERRCON)
            dtNext = SAFETY*dt*pow(errmax, PGROW);
         else
            dtNext = GROW*dt;
         return true;
      } else {
         // We need to take another step
         dtNext = SAFETY*dt*pow(errmax, PSHRNK);
         if (dt > 0.0 )
            dt = max(dtNext, SHRNK*dt);
         else
            dt = min(dtNext, SHRNK*dt);
         // Iterate again
         return false;
      }
   }

   // checks if subiteration is converged
   __CUDA_HD__
   inline bool stepcheck(double &dt, const double dtNext, const double DelT,
                         const double time, bool &finish) {
      const double Remainder = DelT-time;
      if (finish or (Remainder < 1e-12*DelT))
         // We are done
         return true;
      else {
         // We need to take another step
         if (dtNext*1.5 > Remainder) {
            // Force the algorithm to integrate till DelT
            dt = Remainder;
            finish = true;
         } else {
            // Take another standard step
            dt = dtNext;
            finish = false;
         }
         return false;
      }
   }

   // estimate of the integration error
   MyArray<double, n> err;

   // Auxiliary data
   MyArray<double, n> dx;
   MyArray<double, n> xsav;
   MyArray<double, n> dxsav;

   // Jacobian matrix
   MyMatrix<double, n, n> Jac;

   // LU decomposition
   LUdec<n> LU;

   // Algorithm paramenters
   static constexpr int MAXSTP = 100000;
   static constexpr int MAXITS = 100;
   static constexpr double TOL    = 1e-10;
   static constexpr double iTOL   = 1.0/TOL;
   static constexpr double SAFETY = 0.9;
   static constexpr double GROW   = 1.5;
   static constexpr double PGROW  =-0.25;
   static constexpr double SHRNK  = 0.5;
   static constexpr double PSHRNK =-1.0/3.0;
   static constexpr double ERRCON = 0.1296;
   static constexpr double GAM    = 1.0/2.0;
   static constexpr double A21    = 2.0;
   static constexpr double A31    = 48.0/25.0;
   static constexpr double A32    = 6.0/25.0;
   static constexpr double C21    =-8.0;
   static constexpr double C31    = 372.0/25.0;
   static constexpr double C32    = 12.0/5.0;
   static constexpr double C41    =-112.0/125.0;
   static constexpr double C42    =-54.0/125.0;
   static constexpr double C43    =-2.0/5.0;
   static constexpr double B1     = 19.0/9.0;
   static constexpr double B2     = 1.0/2.0;
   static constexpr double B3     = 25.0/108.0;
   static constexpr double B4     = 125.0/108.0;
   static constexpr double E1     = 17.0/54.0;
   static constexpr double E2     = 7.0/36.0;
   static constexpr double E3     = 0.0;
   static constexpr double E4     = 125.0/108.0;
   static constexpr double C1X    = 1.0/2.0;
   static constexpr double C2X    =-3.0/2.0;
   static constexpr double C3X    = 121.0/50.0;
   static constexpr double C4X    = 29.0/250.0;
   static constexpr double A2X    = 1.0;
   static constexpr double A3X    = 3.0/5.0;

};

//-----------------------------------------------------------------------------
// Fast interpolation using the integer domain
//-----------------------------------------------------------------------------

__CUDA_HD__
inline Legion::coord_t FastInterpFindIndex(double x,
                                           const Legion::FieldAccessor<READ_ONLY, float, 1, Legion::coord_t, Realm::AffineAccessor<float, 1, Legion::coord_t> > &xloc,
                                           const Legion::FieldAccessor<READ_ONLY, float, 1, Legion::coord_t, Realm::AffineAccessor<float, 1, Legion::coord_t> > &iloc,
                                           const FastInterpData &d) {

   x = max(d.xmin+d.small, x);
   x = min(d.xmax-d.small, x);
   const Legion::coord_t k   = floor((x-d.xmin)*d.idxloc);
   const Legion::coord_t kp1 = k+1;
   return floor(iloc[k] + (iloc[kp1]-iloc[k])*
                          (        x-xloc[k])*d.idxloc);
};

#endif // __MATH_UTILS_HPP__
