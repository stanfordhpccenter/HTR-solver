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

#include "prometeo_metric_coeffs.h"

#include <math.h>

#ifndef __CUDA_H__
#ifdef __CUDACC__
#define __CUDA_H__ __device__
#else
#define __CUDA_H__
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
using std::min;
#endif

__CUDA_H__
inline int offM2(const int t) { return Cp[t][0]; };
__CUDA_H__
inline int offM1(const int t) { return Cp[t][1]; };
__CUDA_H__
inline int offP1(const int t) { return Cp[t][2]; };
__CUDA_H__
inline int offP2(const int t) { return Cp[t][3]; };
__CUDA_H__
inline int offP3(const int t) { return Cp[t][4]; };

__CUDA_H__
inline double Interp2Staggered(const int t, const double x1, const double x2) { return Interp[t][0]*(x1) + Interp[t][1]*(x2); };

__CUDA_H__
inline double getDeriv(const int t, const double xm1, const double x, const double xp1, const double m) {
   return (Grad[t][0]*(x- xm1) + Grad[t][1]*(xp1 - x))*m;
};

__CUDA_H__
inline double getDerivLeftBC(const int t, const double x, const double xp1, const double m) {
   return (Grad[t][1]*(xp1 - x))*m;
};

__CUDA_H__
inline double getDerivRightBC(const int t, const double xm1, const double x, const double m) {
   return (Grad[t][0]*(x- xm1))*m;
};

// TODO: this is extremely ineficient (do not use in the actual calculation)
// Linear reconstruction based on TENO
__CUDA_H__
inline double LinearReconstruct(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType) {

   // Load coefficients
   const double *Coeffs = Coeffs_Plus[nType];
   const double *Recon  =  Recon_Plus[nType];

   if (nType == R_C_node) return 0.0;
   else if ((nType == L_C_node) || (nType == Rm1_C_node)) return 0.5*y + 0.5*yp1;
   else
      return ym2*(Recon[6*Stencil1+0]*Coeffs[Stencil1] + Recon[6*Stencil2+0]*Coeffs[Stencil2] + Recon[6*Stencil3+0]*Coeffs[Stencil3] + Recon[6*Stencil4+0]*Coeffs[Stencil4]) +
             ym1*(Recon[6*Stencil1+1]*Coeffs[Stencil1] + Recon[6*Stencil2+1]*Coeffs[Stencil2] + Recon[6*Stencil3+1]*Coeffs[Stencil3] + Recon[6*Stencil4+1]*Coeffs[Stencil4]) +
             y  *(Recon[6*Stencil1+2]*Coeffs[Stencil1] + Recon[6*Stencil2+2]*Coeffs[Stencil2] + Recon[6*Stencil3+2]*Coeffs[Stencil3] + Recon[6*Stencil4+2]*Coeffs[Stencil4]) +
             yp1*(Recon[6*Stencil1+3]*Coeffs[Stencil1] + Recon[6*Stencil2+3]*Coeffs[Stencil2] + Recon[6*Stencil3+3]*Coeffs[Stencil3] + Recon[6*Stencil4+3]*Coeffs[Stencil4]) +
             yp2*(Recon[6*Stencil1+4]*Coeffs[Stencil1] + Recon[6*Stencil2+4]*Coeffs[Stencil2] + Recon[6*Stencil3+4]*Coeffs[Stencil3] + Recon[6*Stencil4+4]*Coeffs[Stencil4]) +
             yp3*(Recon[6*Stencil1+5]*Coeffs[Stencil1] + Recon[6*Stencil2+5]*Coeffs[Stencil2] + Recon[6*Stencil3+5]*Coeffs[Stencil3] + Recon[6*Stencil4+5]*Coeffs[Stencil4]);
}

// See Eq. 16 of Pirozzoli JCP (2010)
__CUDA_H__
inline double KennedyReconstruct(const double *rho, const double *u, const double *phi, const int nType) {
   double flux;
   if (nType == L_S_node)
      // This is a staggered node
      flux = rho[2]*u[2]*phi[2];
   else if (nType == Rm1_S_node)
      // This is a staggered node
      flux = rho[3]*u[3]*phi[3];
   else {
      flux = 0.0;
      const double * Coeff = KennedyCoeff[nType];
      __UNROLL__
      for (int l = 0; l < KennedyOrder[nType]; l++) {
         const int lp = l+1;
         double acc = 0.0;
         __UNROLL__
         for (int m = 0; m < lp; m++)
            acc += ((rho[2-m] + rho[2-m+lp])*
                    (  u[2-m] +   u[2-m+lp])*
                    (phi[2-m] + phi[2-m+lp]));
         flux += Coeff[l]*acc;
      }
      flux *= 0.25;
   }
   return flux;
}

__CUDA_H__
inline double KennedyReconstruct(const double *u, const double *phi, const int nType) {
   double flux;
   if (nType == L_S_node)
      // This is a staggered node
      flux = u[2]*phi[2];
   else if (nType == Rm1_S_node)
      // This is a staggered node
      flux = u[3]*phi[3];
   else {
      flux = 0.0;
      const double * Coeff = KennedyCoeff[nType];
      __UNROLL__
      for (int l = 0; l < KennedyOrder[nType]; l++) {
         const int lp = l+1;
         double acc = 0.0;
         __UNROLL__
         for (int m = 0; m < lp; m++)
            acc += ((  u[2-m] +   u[2-m+lp])*
                    (phi[2-m] + phi[2-m+lp]));
         flux += Coeff[l]*acc;
      }
      flux *= 0.5;
   }
   return flux;
}

__CUDA_H__
inline double KennedyReconstruct(const double *phi, const int nType) {
   double flux;
   if (nType == L_S_node)
      // This is a staggered node
      flux = phi[2];
   else if (nType == Rm1_S_node)
      // This is a staggered node
      flux = phi[3];
   else {
      flux = 0.0;
      const double * Coeff = KennedyCoeff[nType];
      __UNROLL__
      for (int l = 0; l < KennedyOrder[nType]; l++) {
         const int lp = l+1;
         double acc = 0.0;
         __UNROLL__
         for (int m = 0; m < lp; m++)
            acc += (phi[2-m] + phi[2-m+lp]);
         flux += Coeff[l]*acc;
      }
   }
   return flux;
}

// Ren sensor coefficients
const __CUDA_CONST__ double Ren_r_c = 0.2;
const __CUDA_CONST__ double Ren_eps = 0.9*Ren_r_c*1e-6/(1.0 - 0.9*Ren_r_c);
const __CUDA_CONST__ double Ren_irc = 1.0/Ren_r_c;

// Constants TENO cut-off
const __CUDA_CONST__ double TENO_cut_off = 1e-6;

const __CUDA_CONST__ double TENO_Smooth_pow = 12.5;
const __CUDA_CONST__ double TENO_Diff_pow = TENO_Smooth_pow - 5.0;

// JS coefficients
const __CUDA_CONST__ double C13 =  13.0/12.0;
const __CUDA_CONST__ double C23 =   3.0/12.0;
const __CUDA_CONST__ double C14 =   1.0/36.0;
const __CUDA_CONST__ double C24 =  13.0/12.0;
const __CUDA_CONST__ double C34 = 781.0/720.0;

__CUDA_H__
inline bool TENOsensor(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType) {

   if ((nType == L_S_node) or (nType == Rm1_S_node))
      // Staggered nodes
      return true;

   else if (nType == L_C_node) {
      // Ren sensor
      const double var1 = yp1 - y;
      const double var2 = yp2 - yp1;
      const double  eta = (fabs(2.0*var1*var2) + Ren_eps)/(var1*var1 + var2*var2 + Ren_eps);
      return ((1.0 - min(1.0, Ren_irc*eta)) < 0.5);

   }
   else if (nType == Rm1_C_node) {
      // Ren sensor
      const double var1 = y   - ym1;
      const double var2 = yp1 - y;
      const double eta = (fabs(2.0*var1*var2) + Ren_eps)/(var1*var1 + var2*var2 + Ren_eps);
      return ((1.0 - min(1.0, Ren_irc*eta)) < 0.5);

   }
   else {
      // Compute smoothness factors
      double aux1; double aux2; double aux3;
      aux1 = (ym1 - 2*y   + yp1); aux2 = (        ym1 - yp1); const double s1 = C13*aux1*aux1 + C23*aux2*aux2;
      aux1 = (y   - 2*yp1 + yp2); aux2 = (3*y - 4*yp1 + yp2); const double s2 = C13*aux1*aux1 + C23*aux2*aux2;
      aux1 = (ym2 - 2*ym1 + y  ); aux2 = (3*y - 4*ym1 + ym2); const double s3 = C13*aux1*aux1 + C23*aux2*aux2;
      aux1 = (-11*y + 18*yp1 - 9*yp2 + 2*yp3);
      aux2 = (  2*y -  5*yp1 + 4*yp2 -   yp3);
      aux3 = (-   y +  3*yp1 - 3*yp2 +   yp3);
      const double s4 = C14*aux1*aux1 + C24*aux2*aux2 + C34*aux3*aux3;

      // We can do this as we know that the sensor is applied on density and density is always > 0
      double eps = (ym2 + ym1 + y + yp1 + yp2 + yp3)/6;
      eps *= eps*1e-4;
      const double tau6 = fabs(s4 - (s3 + s2 + 4*s1)/6);
      const double a1 = pow(1 + tau6/(s1 + eps), 6);
      const double a2 = pow(1 + tau6/(s2 + eps), 6);
      const double a3 = pow(1 + tau6/(s3 + eps), 6);
      const double a4 = pow(1 + tau6/(s4 + eps), 6);

      const double a = 1.0/(a1 + a2 + a3 + a4);

      return ((a1*a > TENO_cut_off) and
              (a2*a > TENO_cut_off) and
              (a3*a > TENO_cut_off) and
              (a4*a > TENO_cut_off));
   }
};

__CUDA_H__
inline bool TENOsensor(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType, const double Phi) {

   if ((nType == L_S_node) or (nType == Rm1_S_node))
      // Staggered nodes
      return true;

   else if (nType == L_C_node) {
      // Ren sensor
      const double var1 = yp1 - y;
      const double var2 = yp2 - yp1;
      const double  eta = (fabs(2.0*var1*var2) + Ren_eps)/(var1*var1 + var2*var2 + Ren_eps);
      return ((1.0 - min(1.0, Ren_irc*eta)) < 0.5);

   }
   else if (nType == Rm1_C_node) {
      // Ren sensor
      const double var1 = y   - ym1;
      const double var2 = yp1 - y;
      const double eta = (fabs(2.0*var1*var2) + Ren_eps)/(var1*var1 + var2*var2 + Ren_eps);
      return ((1.0 - min(1.0, Ren_irc*eta)) < 0.5);

   }
   else {
      // Compute smoothness factors
      double aux1; double aux2; double aux3;
      aux1 = (ym1 - 2*y   + yp1); aux2 = (        ym1 - yp1); const double s1 = C13*aux1*aux1 + C23*aux2*aux2;
      aux1 = (y   - 2*yp1 + yp2); aux2 = (3*y - 4*yp1 + yp2); const double s2 = C13*aux1*aux1 + C23*aux2*aux2;
      aux1 = (ym2 - 2*ym1 + y  ); aux2 = (3*y - 4*ym1 + ym2); const double s3 = C13*aux1*aux1 + C23*aux2*aux2;
      aux1 = (-11*y + 18*yp1 - 9*yp2 + 2*yp3);
      aux2 = (  2*y -  5*yp1 + 4*yp2 -   yp3);
      aux3 = (-   y +  3*yp1 - 3*yp2 +   yp3);
      const double s4 = C14*aux1*aux1 + C24*aux2*aux2 + C34*aux3*aux3;

      const double tau6 = fabs(s4 - (s3 + s2 + 4*s1)/6);
      const double a1 = pow(1 + tau6/(s1 + 1.0e-8), 6);
      const double a2 = pow(1 + tau6/(s2 + 1.0e-8), 6);
      const double a3 = pow(1 + tau6/(s3 + 1.0e-8), 6);
      const double a4 = pow(1 + tau6/(s4 + 1.0e-8), 6);

      const double a = 1.0/(a1 + a2 + a3 + a4);

      // Use TENO-A cutoff adaptation
      const double decay = pow((1 - Phi), 12) * (1 + 12*Phi);
      const int power = ceil(TENO_Diff_pow*(1 - decay) - TENO_Smooth_pow);
      const double cut_off = pow(10, power);

      return ((a1*a > cut_off) and
              (a2*a > cut_off) and
              (a3*a > cut_off) and
              (a4*a > cut_off));
   }
};

__CUDA_H__
inline double TENOreconstructionPlus(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType) {

   // Load coefficients
   const double *Coeffs = Coeffs_Plus[nType];
   const double *Recon  =  Recon_Plus[nType];

   // Compute smoothness factors
   double aux1; double aux2; double aux3;
   aux1 = (ym1 - 2*y   + yp1); aux2 = (        ym1 - yp1); const double s1 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (y   - 2*yp1 + yp2); aux2 = (3*y - 4*yp1 + yp2); const double s2 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (ym2 - 2*ym1 + y  ); aux2 = (3*y - 4*ym1 + ym2); const double s3 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (-11*y + 18*yp1 - 9*yp2 + 2*yp3);
   aux2 = (  2*y -  5*yp1 + 4*yp2 -   yp3);
   aux3 = (-   y +  3*yp1 - 3*yp2 +   yp3);
   const double s4 = C14*aux1*aux1 + C24*aux2*aux2 + C34*aux3*aux3;

   // not recommend to rescale the small number
   const double tau6 = fabs(s4 - (s3 + s2 + 4*s1)/6);
   double a1 = pow(1 + tau6/(s1 + 1.0e-10), 6);
   double a2 = pow(1 + tau6/(s2 + 1.0e-10), 6);
   double a3 = pow(1 + tau6/(s3 + 1.0e-10), 6);
   double a4 = pow(1 + tau6/(s4 + 1.0e-10), 6);

   if (Coeffs[Stencil1] < 1e-10) a1 = 0.0;
   if (Coeffs[Stencil2] < 1e-10) a2 = 0.0;
   if (Coeffs[Stencil3] < 1e-10) a3 = 0.0;
   if (Coeffs[Stencil4] < 1e-10) a4 = 0.0;

   double a = 1.0/(a1 + a2 + a3 + a4);
   const int b1 = (a1*a < TENO_cut_off) ? 0 : 1;
   const int b2 = (a2*a < TENO_cut_off) ? 0 : 1;
   const int b3 = (a3*a < TENO_cut_off) ? 0 : 1;
   const int b4 = (a4*a < TENO_cut_off) ? 0 : 1;

   const double Variation1 = ym2*Recon[6*Stencil1+0] +
                             ym1*Recon[6*Stencil1+1] +
                             y  *Recon[6*Stencil1+2] +
                             yp1*Recon[6*Stencil1+3] +
                             yp2*Recon[6*Stencil1+4] +
                             yp3*Recon[6*Stencil1+5] - y;

   const double Variation2 = ym2*Recon[6*Stencil2+0] +
                             ym1*Recon[6*Stencil2+1] +
                             y  *Recon[6*Stencil2+2] +
                             yp1*Recon[6*Stencil2+3] +
                             yp2*Recon[6*Stencil2+4] +
                             yp3*Recon[6*Stencil2+5] - y;

   const double Variation3 = ym2*Recon[6*Stencil3+0] +
                             ym1*Recon[6*Stencil3+1] +
                             y  *Recon[6*Stencil3+2] +
                             yp1*Recon[6*Stencil3+3] +
                             yp2*Recon[6*Stencil3+4] +
                             yp3*Recon[6*Stencil3+5] - y;

   const double Variation4 = ym2*Recon[6*Stencil4+0] +
                             ym1*Recon[6*Stencil4+1] +
                             y  *Recon[6*Stencil4+2] +
                             yp1*Recon[6*Stencil4+3] +
                             yp2*Recon[6*Stencil4+4] +
                             yp3*Recon[6*Stencil4+5] - y;

   // Assemble the operator
   a1 = Coeffs[Stencil1]*b1;
   a2 = Coeffs[Stencil2]*b2;
   a3 = Coeffs[Stencil3]*b3;
   a4 = Coeffs[Stencil4]*b4;

   a = 1.0/(a1 + a2 + a3 + a4);
   const double w1 = a1*a;
   const double w2 = a2*a;
   const double w3 = a3*a;
   const double w4 = a4*a;

   return y + w1*Variation1 + w2*Variation2 + w3*Variation3 + w4*Variation4;
}

__CUDA_H__
inline double TENOreconstructionMinus(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType) {

   // Load coefficients
   const double *Coeffs = Coeffs_Minus[nType];
   const double *Recon  =  Recon_Minus[nType];

   // Compute smoothness factors
   double aux1; double aux2; double aux3;
   aux1 = (yp2 - 2*yp1 + y  ); aux2 = (          yp2 - y  ); const double s1 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (yp1 - 2*y   + ym1); aux2 = (3*yp1 - 4*y   + ym1); const double s2 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (yp3 - 2*yp2 + yp1); aux2 = (3*yp1 - 4*yp2 + yp3); const double s3 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (-11*yp1 + 18*y - 9*ym1 + 2*ym2);
   aux2 = (  2*yp1 -  5*y + 4*ym1 -   ym2);
   aux3 = (-   yp1 +  3*y - 3*ym1 +   ym2);
   const double s4 = C14*aux1*aux1 + C24*aux2*aux2 + C34*aux3*aux3;

   // not recommend to rescale the small number
   const double tau6 = fabs(s4 - (s3 + s2 + 4*s1)/6);
   double a1 = pow(1 + tau6/(s1 + 1.0e-10), 6);
   double a2 = pow(1 + tau6/(s2 + 1.0e-10), 6);
   double a3 = pow(1 + tau6/(s3 + 1.0e-10), 6);
   double a4 = pow(1 + tau6/(s4 + 1.0e-10), 6);

   if (Coeffs[Stencil1] < 1e-10) a1 = 0.0;
   if (Coeffs[Stencil2] < 1e-10) a2 = 0.0;
   if (Coeffs[Stencil3] < 1e-10) a3 = 0.0;
   if (Coeffs[Stencil4] < 1e-10) a4 = 0.0;

   double a = 1.0/(a1 + a2 + a3 + a4);
   const int b1 = (a1*a < TENO_cut_off) ? 0 : 1;
   const int b2 = (a2*a < TENO_cut_off) ? 0 : 1;
   const int b3 = (a3*a < TENO_cut_off) ? 0 : 1;
   const int b4 = (a4*a < TENO_cut_off) ? 0 : 1;

   const double Variation1 = ym2*Recon[6*Stencil1+0] +
                             ym1*Recon[6*Stencil1+1] +
                             y  *Recon[6*Stencil1+2] +
                             yp1*Recon[6*Stencil1+3] +
                             yp2*Recon[6*Stencil1+4] +
                             yp3*Recon[6*Stencil1+5] - yp1;

   const double Variation2 = ym2*Recon[6*Stencil2+0] +
                             ym1*Recon[6*Stencil2+1] +
                             y  *Recon[6*Stencil2+2] +
                             yp1*Recon[6*Stencil2+3] +
                             yp2*Recon[6*Stencil2+4] +
                             yp3*Recon[6*Stencil2+5] - yp1;

   const double Variation3 = ym2*Recon[6*Stencil3+0] +
                             ym1*Recon[6*Stencil3+1] +
                             y  *Recon[6*Stencil3+2] +
                             yp1*Recon[6*Stencil3+3] +
                             yp2*Recon[6*Stencil3+4] +
                             yp3*Recon[6*Stencil3+5] - yp1;

   const double Variation4 = ym2*Recon[6*Stencil4+0] +
                             ym1*Recon[6*Stencil4+1] +
                             y  *Recon[6*Stencil4+2] +
                             yp1*Recon[6*Stencil4+3] +
                             yp2*Recon[6*Stencil4+4] +
                             yp3*Recon[6*Stencil4+5] - yp1;

   // Assemble the operator
   a1 = Coeffs[Stencil1]*b1;
   a2 = Coeffs[Stencil2]*b2;
   a3 = Coeffs[Stencil3]*b3;
   a4 = Coeffs[Stencil4]*b4;

   a = 1.0/(a1 + a2 + a3 + a4);
   const double w1 = a1*a;
   const double w2 = a2*a;
   const double w3 = a3*a;
   const double w4 = a4*a;

   return yp1 + w1*Variation1 + w2*Variation2 + w3*Variation3 + w4*Variation4;
}

__CUDA_H__
inline double TENOAreconstructionPlus(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType) {

   // Load coefficients
   const double *Coeffs = Coeffs_Plus[nType];
   const double *Recon  =  Recon_Plus[nType];

   // Compute smoothness factors
   double aux1; double aux2; double aux3;
   aux1 = (ym1 - 2*y   + yp1); aux2 = (        ym1 - yp1); const double s1 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (y   - 2*yp1 + yp2); aux2 = (3*y - 4*yp1 + yp2); const double s2 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (ym2 - 2*ym1 + y  ); aux2 = (3*y - 4*ym1 + ym2); const double s3 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (-11*y + 18*yp1 - 9*yp2 + 2*yp3);
   aux2 = (  2*y -  5*yp1 + 4*yp2 -   yp3);
   aux3 = (-   y +  3*yp1 - 3*yp2 +   yp3);
   const double s4 = C14*aux1*aux1 + C24*aux2*aux2 + C34*aux3*aux3;

   // not recommend to rescale the small number
   const double tau6 = fabs(s4 - (s3 + s2 + 4*s1)/6);
   double a1 = pow(1 + tau6/(s1 + 1.0e-10), 6);
   double a2 = pow(1 + tau6/(s2 + 1.0e-10), 6);
   double a3 = pow(1 + tau6/(s3 + 1.0e-10), 6);
   double a4 = pow(1 + tau6/(s4 + 1.0e-10), 6);

   if (Coeffs[Stencil1] < 1e-10) a1 = 0.0;
   if (Coeffs[Stencil2] < 1e-10) a2 = 0.0;
   if (Coeffs[Stencil3] < 1e-10) a3 = 0.0;
   if (Coeffs[Stencil4] < 1e-10) a4 = 0.0;

   // Adapt cut_off based on Ren sensor
   const double var2 = fabs(ym1 - ym2);
   const double var3 = fabs(y   - ym1);
   const double var4 = fabs(yp1 - y  );
   const double var5 = fabs(yp2 - yp1);
   const double var6 = fabs(yp3 - yp2);

   double eta = min((2*var2*var3 + Ren_eps) / (var2*var2 + var3*var3 + Ren_eps),
                    (2*var3*var4 + Ren_eps) / (var3*var3 + var4*var4 + Ren_eps));
   eta = min(eta,   (2*var4*var5 + Ren_eps) / (var4*var4 + var5*var5 + Ren_eps));
   eta = min(eta,   (2*var5*var6 + Ren_eps) / (var5*var5 + var6*var6 + Ren_eps));

   const double delta = 1 - min(eta*Ren_irc, 1.0);
   const double decay = pow((1 - delta), 8) * (1 + 8*delta);
   const int power = ceil(TENO_Diff_pow*(1 - decay) - TENO_Smooth_pow);
   const double cut_off = pow(10, power);

   // Select stencils
   double a = 1.0/(a1 + a2 + a3 + a4);
   const int b1 = (a1*a < cut_off) ? 0 : 1;
   const int b2 = (a2*a < cut_off) ? 0 : 1;
   const int b3 = (a3*a < cut_off) ? 0 : 1;
   const int b4 = (a4*a < cut_off) ? 0 : 1;

   const double Variation1 = ym2*Recon[6*Stencil1+0] +
                             ym1*Recon[6*Stencil1+1] +
                             y  *Recon[6*Stencil1+2] +
                             yp1*Recon[6*Stencil1+3] +
                             yp2*Recon[6*Stencil1+4] +
                             yp3*Recon[6*Stencil1+5] - y;

   const double Variation2 = ym2*Recon[6*Stencil2+0] +
                             ym1*Recon[6*Stencil2+1] +
                             y  *Recon[6*Stencil2+2] +
                             yp1*Recon[6*Stencil2+3] +
                             yp2*Recon[6*Stencil2+4] +
                             yp3*Recon[6*Stencil2+5] - y;

   const double Variation3 = ym2*Recon[6*Stencil3+0] +
                             ym1*Recon[6*Stencil3+1] +
                             y  *Recon[6*Stencil3+2] +
                             yp1*Recon[6*Stencil3+3] +
                             yp2*Recon[6*Stencil3+4] +
                             yp3*Recon[6*Stencil3+5] - y;

   const double Variation4 = ym2*Recon[6*Stencil4+0] +
                             ym1*Recon[6*Stencil4+1] +
                             y  *Recon[6*Stencil4+2] +
                             yp1*Recon[6*Stencil4+3] +
                             yp2*Recon[6*Stencil4+4] +
                             yp3*Recon[6*Stencil4+5] - y;

   // Assemble the operator
   a1 = Coeffs[Stencil1]*b1;
   a2 = Coeffs[Stencil2]*b2;
   a3 = Coeffs[Stencil3]*b3;
   a4 = Coeffs[Stencil4]*b4;

   a = 1.0/(a1 + a2 + a3 + a4);
   const double w1 = a1*a;
   const double w2 = a2*a;
   const double w3 = a3*a;
   const double w4 = a4*a;

   return y + w1*Variation1 + w2*Variation2 + w3*Variation3 + w4*Variation4;
}

__CUDA_H__
inline double TENOAreconstructionMinus(const double ym2, const double ym1, const double y, const double yp1, const double yp2, const double yp3, const int nType) {

   // Load coefficients
   const double *Coeffs = Coeffs_Minus[nType];
   const double *Recon  =  Recon_Minus[nType];

   // Compute smoothness factors
   double aux1; double aux2; double aux3;
   aux1 = (yp2 - 2*yp1 + y  ); aux2 = (          yp2 - y  ); const double s1 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (yp1 - 2*y   + ym1); aux2 = (3*yp1 - 4*y   + ym1); const double s2 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (yp3 - 2*yp2 + yp1); aux2 = (3*yp1 - 4*yp2 + yp3); const double s3 = C13*aux1*aux1 + C23*aux2*aux2;
   aux1 = (-11*yp1 + 18*y - 9*ym1 + 2*ym2);
   aux2 = (  2*yp1 -  5*y + 4*ym1 -   ym2);
   aux3 = (-   yp1 +  3*y - 3*ym1 +   ym2);
   const double s4 = C14*aux1*aux1 + C24*aux2*aux2 + C34*aux3*aux3;

   // not recommend to rescale the small number
   const double tau6 = fabs(s4 - (s3 + s2 + 4*s1)/6);
   double a1 = pow(1 + tau6/(s1 + 1.0e-10), 6);
   double a2 = pow(1 + tau6/(s2 + 1.0e-10), 6);
   double a3 = pow(1 + tau6/(s3 + 1.0e-10), 6);
   double a4 = pow(1 + tau6/(s4 + 1.0e-10), 6);

   if (Coeffs[Stencil1] < 1e-10) a1 = 0.0;
   if (Coeffs[Stencil2] < 1e-10) a2 = 0.0;
   if (Coeffs[Stencil3] < 1e-10) a3 = 0.0;
   if (Coeffs[Stencil4] < 1e-10) a4 = 0.0;

   // Adapt cut_off based on Ren sensor
   const double var2 = fabs(ym1 - ym2);
   const double var3 = fabs(y   - ym1);
   const double var4 = fabs(yp1 - y  );
   const double var5 = fabs(yp2 - yp1);
   const double var6 = fabs(yp3 - yp2);

   double eta = min((2*var2*var3 + Ren_eps) / (var2*var2 + var3*var3 + Ren_eps),
                    (2*var3*var4 + Ren_eps) / (var3*var3 + var4*var4 + Ren_eps));
   eta = min(eta,   (2*var4*var5 + Ren_eps) / (var4*var4 + var5*var5 + Ren_eps));
   eta = min(eta,   (2*var5*var6 + Ren_eps) / (var5*var5 + var6*var6 + Ren_eps));

   const double delta = 1 - min(eta*Ren_irc, 1.0);
   const double decay = pow((1 - delta), 8) * (1 + 8*delta);
   const int power = ceil(TENO_Diff_pow*(1 - decay) - TENO_Smooth_pow);
   const double cut_off = pow(10, power);

   // Select stencils
   double a = 1.0/(a1 + a2 + a3 + a4);
   const int b1 = (a1*a < cut_off) ? 0 : 1;
   const int b2 = (a2*a < cut_off) ? 0 : 1;
   const int b3 = (a3*a < cut_off) ? 0 : 1;
   const int b4 = (a4*a < cut_off) ? 0 : 1;

   const double Variation1 = ym2*Recon[6*Stencil1+0] +
                             ym1*Recon[6*Stencil1+1] +
                             y  *Recon[6*Stencil1+2] +
                             yp1*Recon[6*Stencil1+3] +
                             yp2*Recon[6*Stencil1+4] +
                             yp3*Recon[6*Stencil1+5] - yp1;

   const double Variation2 = ym2*Recon[6*Stencil2+0] +
                             ym1*Recon[6*Stencil2+1] +
                             y  *Recon[6*Stencil2+2] +
                             yp1*Recon[6*Stencil2+3] +
                             yp2*Recon[6*Stencil2+4] +
                             yp3*Recon[6*Stencil2+5] - yp1;

   const double Variation3 = ym2*Recon[6*Stencil3+0] +
                             ym1*Recon[6*Stencil3+1] +
                             y  *Recon[6*Stencil3+2] +
                             yp1*Recon[6*Stencil3+3] +
                             yp2*Recon[6*Stencil3+4] +
                             yp3*Recon[6*Stencil3+5] - yp1;

   const double Variation4 = ym2*Recon[6*Stencil4+0] +
                             ym1*Recon[6*Stencil4+1] +
                             y  *Recon[6*Stencil4+2] +
                             yp1*Recon[6*Stencil4+3] +
                             yp2*Recon[6*Stencil4+4] +
                             yp3*Recon[6*Stencil4+5] - yp1;

   // Assemble the operator
   a1 = Coeffs[Stencil1]*b1;
   a2 = Coeffs[Stencil2]*b2;
   a3 = Coeffs[Stencil3]*b3;
   a4 = Coeffs[Stencil4]*b4;

   a = 1.0/(a1 + a2 + a3 + a4);
   const double w1 = a1*a;
   const double w2 = a2*a;
   const double w3 = a3*a;
   const double w4 = a4*a;

   return yp1 + w1*Variation1 + w2*Variation2 + w3*Variation3 + w4*Variation4;
}
