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

#ifndef Species_HPP
#define Species_HPP

#include <stdint.h>
#include <math.h>

#include "Species.h"

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#define PI    3.1415926535898

//  omega_mu() returns the collision integral for mu given dimensionless temperature t/(eps/k).
//  TODO: These come from FlameMaster.
//        At a certain point, verify these implementations.
__CUDA_HD__
inline double omega_mu(const double T) {
   const double m1 = 3.3530622607;
   const double m2 = 2.53272006;
   const double m3 = 2.9024238575;
   const double m4 = 0.11186138893;
   const double m5 = 0.8662326188;       // = -0.1337673812 + 1.0
   const double m6 = 1.3913958626;
   const double m7 = 3.158490576;
   const double m8 = 0.18973411754;
   const double m9 = 0.00018682962894;

   const double num = m1 + T*(m2 + T*(m3 + T*m4));
   const double den = m5 + T*(m6 + T*(m7 + T*(m8 + T*m9)));
   return num / den;
};

// omega_D() returns the Stossintegral for a given dimensionless temperature t/(eps/k)
__CUDA_HD__
inline double omega_D(const double T) {
   const double m1 = 6.8728271691;
   const double m2 = 9.4122316321;
   const double m3 = 7.7442359037;
   const double m4 = 0.23424661229;
   const double m5 = 1.45337701568;         // = 1.0 + 0.45337701568
   const double m6 = 5.2269794238;
   const double m7 = 9.7108519575;
   const double m8 = 0.46539437353;
   const double m9 = 0.00041908394781;

   const double num = m1 + T * (m2 + T * (m3 + T * m4));
   const double den = m5 + T * (m6 + T * (m7 + T * (m8 + T * m9)));
   return num / den;
};

__CUDA_HD__
inline double GetCp(const Spec & s, const double T) {
   //assert(T < s.cpCoeff.TMax, "Exceeded maximum temeperature")
   //assert(T > s.cpCoeff.TMin, "Exceeded minimum temeperature")

   const double rOvW = RGAS/s.W;
   const double Tinv = 1.0/T;
   const double * cpCoeff = ( T > s.cpCoeff.TSwitch2 ) ? s.cpCoeff.cpH :
                            ( T > s.cpCoeff.TSwitch1 ) ? s.cpCoeff.cpM :
                                                         s.cpCoeff.cpL;
   return rOvW*( cpCoeff[0]*Tinv*Tinv + cpCoeff[1]*Tinv +  cpCoeff[2] + T*
                                                         ( cpCoeff[3] + T*
                                                         ( cpCoeff[4] + T*
                                                         ( cpCoeff[5] + T*cpCoeff[6]))));
};

__CUDA_HD__
inline double GetEnthalpy(const Spec & s, const double T) {
   //assert(T < s.cpCoeff.TMax, "Exceeded maximum temeperature")
   //assert(T > s.cpCoeff.TMin, "Exceeded minimum temeperature")

   const double rOvW = RGAS/s.W;
   const double Tinv = 1.0/T;
   const double * cpCoeff = ( T > s.cpCoeff.TSwitch2 ) ? s.cpCoeff.cpH :
                            ( T > s.cpCoeff.TSwitch1 ) ? s.cpCoeff.cpM :
                                                         s.cpCoeff.cpL;
   const double E = -cpCoeff[0]*Tinv + cpCoeff[1]*log(T) + cpCoeff[7]      + T*
                                                         ( cpCoeff[2]      + T*
                                                         ( cpCoeff[3]*0.50 + T*
                                                         ( cpCoeff[4]/3    + T*
                                                         ( cpCoeff[5]*0.25 + cpCoeff[6]/5*T))));
   return E*rOvW;
};

__CUDA_HD__
inline double GetMu(const Spec & s, const double T) {
   const double num = 5 * sqrt(PI * s.W/Na * kb * T);
   const double den = 16 * PI * pow(s.DiffCoeff.sigma,2) * omega_mu( T * s.DiffCoeff.kbOveps );
   return num/den;
};

__CUDA_HD__
inline double GetDif(const Spec & s1, const Spec & s2,
                     const double P, const double T) {
   double xi = 1.0;
   if ((s1.DiffCoeff.mu*s2.DiffCoeff.mu == 0.0) and
       (s1.DiffCoeff.mu+s2.DiffCoeff.mu != 0.0)) {
      // If I have a polar to non-polar molecule interaction
      double mup;
      double alp;
      double epr;
      if (s1.DiffCoeff.mu != 0.0) {
         mup = s1.DiffCoeff.mu/sqrt(pow(s1.DiffCoeff.sigma,3)*kb/s1.DiffCoeff.kbOveps);
         alp = s1.DiffCoeff.alpha/s1.DiffCoeff.sigma;
         epr = sqrt(s2.DiffCoeff.kbOveps/s1.DiffCoeff.kbOveps);
      } else {
         mup = s2.DiffCoeff.mu/sqrt(pow(s2.DiffCoeff.sigma,3)*kb/s2.DiffCoeff.kbOveps);
         alp = s2.DiffCoeff.alpha/s2.DiffCoeff.sigma;
         epr = sqrt(s1.DiffCoeff.kbOveps/s2.DiffCoeff.kbOveps);
      }
      xi = 1 + 0.25*mup*alp*epr;
   }
   const double invWij = (s1.W + s2.W)/(s1.W*s2.W);
   const double kboEpsij = sqrt(s1.DiffCoeff.kbOveps * s2.DiffCoeff.kbOveps)/(xi*xi);
   const double sigmaij = 0.5*(s1.DiffCoeff.sigma + s2.DiffCoeff.sigma)*pow(xi,1./6);
   const double num = 3*sqrt(2*PI*pow(kb,3)*pow(T,3)*Na*invWij);
   const double den = 16*PI*P*sigmaij*sigmaij*omega_D(T * kboEpsij);
   return num/den;
};

__CUDA_HD__
inline double GetSelfDiffusion(const Spec & s, const double T) {
   // Already multiplied by partial density
   const double num = 3*sqrt( PI*kb*T*s.W/Na );
   const double den = 8*PI*pow(s.DiffCoeff.sigma,2)*omega_D(T * s.DiffCoeff.kbOveps);
   return num/den;
};

__CUDA_HD__
inline double GetFZrot(const Spec & s, const double T) {
   const double tmp = 1.0/(s.DiffCoeff.kbOveps*T);
   return 1 + 0.5*pow(PI,1.5)*sqrt(tmp)
            + (2 + 0.25*PI*PI)*tmp
            + pow(PI,1.5)*pow(tmp,1.5);
};

__CUDA_HD__
inline double GetLamAtom(const Spec & s, const double T) {
   const double mu = GetMu(s, T);
   return 15.0/4*mu*RGAS/s.W;
};

__CUDA_HD__
inline double GetLamLinear(const Spec & s, const double T) {
   const double CvTOvR = 1.5;
   const double CvROvR = 1.0;

   const double CvT = CvTOvR*RGAS;
   const double CvR = CvROvR*RGAS;
   const double CvV = GetCp(s, T)*s.W - 3.5*RGAS;

   const double Dkk = GetSelfDiffusion(s, T);
   const double mu = GetMu(s, T);

   const double fV = Dkk/mu;

   const double Zrot = s.DiffCoeff.Z298*GetFZrot(s, 298)/GetFZrot(s, T);

   const double A = 2.5 - fV;
   const double B = Zrot + 2/PI*(5./3*CvROvR+fV);

   const double fT = 2.5 * (1. - 2*CvR*A/(PI*CvT*B));
   const double fR = fV*(1. + 2*A/(PI*B));

   return mu/s.W*(fT*CvT + fR*CvR + fV*CvV);
};

__CUDA_HD__
inline double GetLamNonLinear(const Spec & s, const double T) {
   const double CvTOvR = 1.5;
   const double CvROvR = 1.5;

   const double CvT = CvTOvR*RGAS;
   const double CvR = CvROvR*RGAS;
   const double CvV = GetCp(s, T)*s.W - 4.0*RGAS;

   const double Dkk = GetSelfDiffusion(s, T);
   const double mu  = GetMu(s, T);

   const double fV = Dkk/mu;

   const double Zrot = s.DiffCoeff.Z298*GetFZrot(s, 298.0)/GetFZrot(s, T);

   const double A = 2.5 - fV;
   const double B = Zrot + 2/PI*(5./3*CvROvR+fV);

   const double fT = 2.5 * (1. - 2*CvR*A/(PI*CvT*B));
   const double fR = fV*(1. + 2*A/(PI*B));

   return mu/s.W*(fT*CvT + fR*CvR + fV*CvV);
};

__CUDA_HD__
inline double GetLam(const Spec & s, const double T) {
   return ((s.Geom == SpeciesGeom_Atom)   ?      GetLamAtom(s, T)   :
          ((s.Geom == SpeciesGeom_Linear) ?      GetLamLinear(s, T) :
         /*(s.Geom == SpeciesGeom_NonLinear) ?*/ GetLamNonLinear(s, T)));
};

#endif // Species_HPP
