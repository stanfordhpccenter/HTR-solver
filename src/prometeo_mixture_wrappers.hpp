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

#ifndef __PROMETEO_MIXTURE_WRAPPERS_HPP__
#define __PROMETEO_MIXTURE_WRAPPERS_HPP__

//-----------------------------------------------------------------------------
// LOAD PROMETEO TYPES
//-----------------------------------------------------------------------------

#include "prometeo_types.h"

//-----------------------------------------------------------------------------
// C WRAPPERS OF MIXTURE METHODS
//-----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

struct Mix InitMixture(const struct Config *config);

const char* GetSpeciesName(const int i, const struct Mix *mix);

int FindSpecies(const char *Name, const struct Mix *mix);

bool CheckMixture(const double *Yi, const struct Mix *mix);

void ClipYi(double *Yi, const struct Mix *mix);

double GetMolarWeightFromYi(const double *Yi, const struct Mix *mix);

double GetMolarWeightFromXi(const double *Xi, const struct Mix *mix);

void GetMolarFractions(double *Xi, const double MixW, const double *Yi, const struct Mix *mix);

void GetMassFractions(double *Yi, const double MixW, const double *Xi, const struct Mix *mix);

double GetRhoFromRhoYi(const double *rhoYi, const struct Mix *mix);

void GetRhoYiFromYi(double *rhoYi, const double rho, const double *Yi, const struct Mix *mix);

void GetYi(double *Yi, const double rho, const double *rhoYi, const struct Mix *mix);

double GetRho(const double P, const double T, const double MixW, const struct Mix *mix);

double GetHeatCapacity(const double T, const double *Yi, const struct Mix *mix);

double GetEnthalpy(const double T, const double *Yi, const struct Mix *mix);

double GetSpeciesEnthalpy(const int i, const double T, const struct Mix *mix);

double GetSpeciesMolarWeight(const int i, const struct Mix *mix);

double GetInternalEnergy(const double T, const double *Yi, const struct Mix *mix);

double GetSpecificInternalEnergy(const int i, const double T, const struct Mix *mix);

double GetTFromInternalEnergy(const double e0, double T, const double *Yi, const struct Mix *mix);

bool isValidInternalEnergy(const double e, const double *Yi, const struct Mix *mix);

double GetTFromRhoAndP(const double rho, const double MixW, const double P, const struct Mix *mix);

double GetPFromRhoAndT(const double rho, const double MixW, const double T, const struct Mix *mix);

double GetViscosity(const double T, const double *Xi, const struct Mix *mix);

double GetHeatConductivity(const double T, const double *Xi, const struct Mix *mix);

double GetGamma(const double T, const double MixW, const double *Yi, const struct Mix *mix);

double GetSpeedOfSound(const double T, const double gamma, const double MixW, const struct Mix *mix);

void GetDiffusivity(double *Di, const double P, const double T, const double MixW, const double *Xi, const struct Mix *mix);

#if (nIons > 0)
void GetElectricMobility(double *Ki, const double Pn, const double Tn, const double *Xi, const struct Mix *mix);
#endif

double GetPartialElectricChargeDensity(const uint8_t i, const double rhon, const double MixW, const double *Xi, const struct Mix *mix);

double GetElectricChargeDensity(const double rhon, const double MixW, const double *Xi, const struct Mix *mix);

int8_t GetSpeciesChargeNumber(const int i, const struct Mix *mix);

double GetDielectricPermittivity(const struct Mix *mix);

void GetProductionRates(double *wi, const double rhon, const double Pn, const double Tn, const double *Yi, const struct Mix *mix);

double Getdpde(const double rho, const double gamma, const struct Mix *mix);

void Getdpdrhoi(double *dpdrhoi, const double gamma, const double T, const double *Yi, const struct Mix *mix);

#ifdef __cplusplus
};
#endif

#endif // __PROMETEO_MIXTURE_WRAPPERS_HPP__
