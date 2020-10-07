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

#ifndef IsentropicMix_HPP
#define IsentropicMix_HPP

#include "IsentropicMix.h"

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

__CUDA_HD__
inline double GetMolarWeightFromYi(const double *Yi, const Mix &mix) { return RGAS/mix.R; }

__CUDA_HD__
inline double GetMolarWeightFromXi(const double *Xi, const Mix &mix) { return RGAS/mix.R; }

__CUDA_HD__
inline void GetMolarFractions(double *Xi, const double MixW, const double *Yi, const Mix &mix) { Xi[0] = Yi[0]; }

__CUDA_HD__
inline void GetMassFractions(double *Yi, const double MixW, const double *Xi, const Mix &mix) { Yi[0] = Xi[0]; }

__CUDA_HD__
inline double GetRho(const double P, const double T, const double MixW, const Mix &mix) { return pow(T, 1.0/(mix.gamma-1)); };

__CUDA_HD__
inline double GetHeatCapacity(const double T, const double *Yi, const Mix &mix) { return mix.gamma/(mix.gamma-1)*mix.R; };

__CUDA_HD__
inline double GetSpeciesEnthalpy(const int i, const double T, const Mix &mix) { return T*mix.R*mix.gamma/(mix.gamma-1.0); };

__CUDA_HD__
inline double GetSpeciesMolarWeight(const int i, const Mix &mix) { return RGAS/mix.R; };

__CUDA_HD__
inline double GetSpecificInternalEnergy(const int i, const double T, const Mix &mix) { return T*mix.R/(mix.gamma-1.0); };

__CUDA_HD__
inline double GetViscosity(const double T, const double *Xi, const Mix &mix) { return 0.0; };

__CUDA_HD__
inline double GetHeatConductivity(const double T, const double *Xi, const Mix &mix) { return 0.0; };

__CUDA_HD__
inline double GetGamma(const double T, const double MixW, const double *Yi, const Mix &mix) { return mix.gamma; };

__CUDA_HD__
inline double GetSpeedOfSound(const double T, const double gamma, const double MixW, const Mix &mix) { return sqrt(mix.gamma*mix.R*T); };

__CUDA_HD__
inline void GetDiffusivity(double *Di, const double P, const double T, const double MixW, const double *Xi, const Mix &mix) { Di[0] = 0.0; }

__CUDA_HD__
inline double Getdpde(const double rho, const double gamma, const Mix &mix) { return rho*(mix.gamma - 1); };

__CUDA_HD__
inline void Getdpdrhoi(double *dpdrhoi, const double gamma, const double T, const double *Yi, const Mix &mix) { dpdrhoi[0] = mix.R*T; };

#endif // IsentropicMix_HPP
