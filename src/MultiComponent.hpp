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

#ifndef MultiComponent_HPP
#define MultiComponent_HPP

#include "Species.hpp"

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

__CUDA_HD__
inline double GetMolarWeightFromYi(const double *Yi, const Mix &mix) {
   double MixW = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) MixW += Yi[i]/mix.species[i].W;
   return 1.0/MixW;
};

__CUDA_HD__
inline double GetMolarWeightFromXi(const double *Xi, const Mix &mix) {
   double MixW = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) MixW += Xi[i]*mix.species[i].W;
   return MixW;
};

__CUDA_HD__
inline void GetMolarFractions(double *Xi, const double MixW, const double *Yi, const Mix &mix) {
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      Xi[i] = Yi[i]*MixW/mix.species[i].W;
};

__CUDA_HD__
inline void GetMassFractions(double *Yi, const double MixW, const double *Xi, const Mix &mix) {
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      Yi[i] = Xi[i]*mix.species[i].W/MixW;
};

__CUDA_HD__
inline double GetRho(const double P, const double T, const double MixW, const Mix &mix) { return P * MixW/(RGAS * T); };

__CUDA_HD__
inline double GetHeatCapacity(const double T, const double *Yi, const Mix &mix) {
   double cp = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) cp += Yi[i]*GetCp(mix.species[i], T);
   return cp;
};

__CUDA_HD__
inline double GetSpeciesEnthalpy(const int i, const double T, const Mix &mix) { return GetEnthalpy(mix.species[i], T); };

__CUDA_HD__
inline double GetSpeciesMolarWeight(const int i, const Mix &mix) { return mix.species[i].W; };

__CUDA_HD__
inline double GetSpecificInternalEnergy(const int i, const double T, const Mix &mix) { return GetEnthalpy(mix.species[i], T) - RGAS*T/mix.species[i].W; };

__CUDA_HD__
inline double GetViscosity(const double T, const double *Xi, const Mix &mix) {
   double muk[nSpec];
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      muk[i] = GetMu(mix.species[i], T);

   double mu = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      double den = 0.0;
      for (int j = 0; j<nSpec; j++) {
         double Phi = pow(1 + sqrt(muk[i]/muk[j]) * pow(mix.species[j].W/mix.species[i].W, 0.25) , 2);
         Phi /= sqrt(8*(1 + mix.species[i].W/mix.species[j].W));
         den += Xi[j]*Phi;
      }
      mu += Xi[i]*muk[i]/den;
   }
   return mu;
};

__CUDA_HD__
inline double GetHeatConductivity(const double T, const double *Xi, const Mix &mix) {
   double a = 0.0;
   double b = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      double lami = GetLam(mix.species[i], T);
      a += Xi[i]*lami;
      b += Xi[i]/lami;
   }
   return 0.5*(a + 1.0/b);
};

__CUDA_HD__
inline double GetGamma(const double T, const double MixW, const double *Yi, const Mix &mix) {
   const double cp = GetHeatCapacity(T, Yi, mix);
   return cp/(cp - RGAS/MixW);
};

__CUDA_HD__
inline double GetSpeedOfSound(const double T, const double gamma, const double MixW, const Mix &mix) {
   return sqrt(gamma*RGAS*T/MixW);
};

__CUDA_HD__
inline void GetDiffusivity(double *Di, const double P, const double T, const double MixW, const double *Xi, const Mix &mix) {
   double invDi[nSpec*nSpec];
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      invDi[i*nSpec+i] = 0.0;
      __UNROLL__
      for (int j = 0; j<i; j++) {
         invDi[j*nSpec+i] = 1.0/GetDif(mix.species[i], mix.species[j], P, T);
         invDi[i*nSpec+j] = invDi[j*nSpec+i];
      }
   }

   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      double num = 0.0;
      double den = 0.0;
      __UNROLL__
      for (int j = 0; j<i; j++) {
         num += Xi[j]*mix.species[j].W;
         den += Xi[j]*invDi[i*nSpec+j];
      }
      __UNROLL__
      for (int j = i+1; j<nSpec; j++) {
         num += Xi[j]*mix.species[j].W;
         den += Xi[j]*invDi[i*nSpec+j];
      }
      Di[i] = num/(MixW*den);
   }
};

__CUDA_HD__
inline double Getdpde(const double rho, const double gamma, const Mix &mix) { return rho*(gamma - 1); };

__CUDA_HD__
inline void Getdpdrhoi(double *dpdrhoi, const double gamma, const double T, const double *Yi, const Mix &mix) {
   double e = 0.0;
   double ei[nSpec];
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      ei[i] = (GetEnthalpy(mix.species[i], T) - RGAS*T/mix.species[i].W);
      e += Yi[i]*ei[i];
   }
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      dpdrhoi[i] = RGAS*T/mix.species[i].W + (gamma - 1)*(e - ei[i]);
};

#endif // MultiComponent_HPP
