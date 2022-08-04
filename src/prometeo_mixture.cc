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

#include "prometeo_mixture.hpp"
#include "prometeo_mixture_wrappers.hpp"

/*static*/ const char * const    LoadMixtureTask::TASK_NAME = "LoadMixture";
/*static*/ const int             LoadMixtureTask::TASK_ID = TID_LoadMixture;

void LoadMixtureTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 1);
   assert(futures.size() == 0);
   // Nothing to do
}

void register_mixture_tasks() {

   TaskHelper::register_hybrid_variants<LoadMixtureTask>();

};

//-----------------------------------------------------------------------------
// C WRAPPERS OF MIXTURE METHODS
//-----------------------------------------------------------------------------

struct Mix InitMixture(const struct Config *config) {
   return Mix(*config);
};

const char* GetSpeciesName(const int i, const struct Mix *mix) {
   return mix->GetSpeciesName(i);
};

int FindSpecies(const char *Name, const struct Mix *mix) {
   return mix->FindSpecies(Name);
};

bool CheckMixture(const double *Yi, const struct Mix *mix) {
   return mix->CheckMixture(VecNSp(Yi));
};

void ClipYi(double *Yi, const struct Mix *mix) {
   VecNSp Y(Yi);
   mix->ClipYi(Y);
   for (int i=0; i < nSpec; i++) Yi[i] = Y[i];
};

double GetMolarWeightFromYi(const double *Yi, const struct Mix *mix) {
   return mix->GetMolarWeightFromYi(VecNSp(Yi));
};

double GetMolarWeightFromXi(const double *Xi, const struct Mix *mix) {
   return mix->GetMolarWeightFromXi(VecNSp(Xi));
};

void GetMolarFractions(double *Xi, const double MixW, const double *Yi, const struct Mix *mix) {
   VecNSp X;
   mix->GetMolarFractions(X, MixW, VecNSp(Yi));
   for (int i=0; i < nSpec; i++) Xi[i] = X[i];
};

void GetMassFractions(double *Yi, const double MixW, const double *Xi, const struct Mix *mix) {
   VecNSp Y;
   mix->GetMassFractions(Y, MixW, VecNSp(Xi));
   for (int i=0; i < nSpec; i++) Yi[i] = Y[i];
};

double GetRhoFromRhoYi(const double *rhoYi, const struct Mix *mix) {
   return mix->GetRhoFromRhoYi(VecNSp(rhoYi));
};

void GetRhoYiFromYi(double *rhoYi, const double rho, const double *Yi, const struct Mix *mix) {
   VecNSp rhoY;
   mix->GetRhoYiFromYi(rhoY, rho, VecNSp(Yi));
   for (int i=0; i < nSpec; i++) rhoYi[i] = rhoY[i];
};

void GetYi(double *Yi, const double rho, const double *rhoYi, const struct Mix *mix) {
   VecNSp Y;
   mix->GetYi(Y, rho, VecNSp(rhoYi));
   for (int i=0; i < nSpec; i++) Yi[i] = Y[i];
};

double GetRho(const double P, const double T, const double MixW, const struct Mix *mix) {
   return mix->GetRho(P, T, MixW);
};

double GetHeatCapacity(const double T, const double *Yi, const struct Mix *mix) {
   return mix->GetHeatCapacity(T, VecNSp(Yi));
};

double GetEnthalpy(const double T, const double *Yi, const struct Mix *mix) {
   return mix->GetEnthalpy(T, VecNSp(Yi));
};

double GetSpeciesEnthalpy(const int i, const double T, const struct Mix *mix) {
   return mix->GetSpeciesEnthalpy(i, T);
};

double GetSpeciesMolarWeight(const int i, const struct Mix *mix) {
   return mix->GetSpeciesMolarWeight(i);
};

double GetInternalEnergy(const double T, const double *Yi, const struct Mix *mix) {
   return mix->GetInternalEnergy(T, VecNSp(Yi));
};

double GetSpecificInternalEnergy(const int i, const double T, const struct Mix *mix) {
   return mix->GetSpecificInternalEnergy(i, T);
};

double GetTFromInternalEnergy(const double e0, double T, const double *Yi, const struct Mix *mix) {
   return mix->GetTFromInternalEnergy(e0, T, VecNSp(Yi));
};

bool isValidInternalEnergy(const double e, const double *Yi, const struct Mix *mix) {
   return mix->isValidInternalEnergy(e, VecNSp(Yi));
};

double GetTFromRhoAndP(const double rho, const double MixW, const double P, const struct Mix *mix) {
   return mix->GetTFromRhoAndP(rho, MixW, P);
};

double GetPFromRhoAndT(const double rho, const double MixW, const double T, const struct Mix *mix) {
   return mix->GetPFromRhoAndT(rho, MixW, T);
};

double GetViscosity(const double T, const double *Xi, const struct Mix *mix) {
   return mix->GetViscosity(T, VecNSp(Xi));
};

double GetHeatConductivity(const double T, const double *Xi, const struct Mix *mix) {
   return mix->GetHeatConductivity(T, VecNSp(Xi));
};

double GetGamma(const double T, const double MixW, const double *Yi, const struct Mix *mix) {
   return mix->GetGamma(T, MixW, VecNSp(Yi));
};

double GetSpeedOfSound(const double T, const double gamma, const double MixW, const struct Mix *mix) {
   return mix->GetSpeedOfSound(T, gamma, MixW);
};

void GetDiffusivity(double *Di, const double P, const double T, const double MixW, const double *Xi, const struct Mix *mix) {
   VecNSp D;
   mix->GetDiffusivity(D, P, T, MixW, VecNSp(Xi));
   for (int i=0; i < nSpec; i++) Di[i] = D[i];
};

#if (nIons > 0)
void GetElectricMobility(double *Ki, const double Pn, const double Tn, const double *Xi, const struct Mix *mix) {
   VecNIo K;
   mix->GetElectricMobility(K, Pn, Tn, VecNSp(Xi));
   for (int i=0; i < nIons; i++) Ki[i] = K[i];
};
#endif

double GetPartialElectricChargeDensity(const uint8_t i, const double rhon, const double MixW, const double *Xi, const struct Mix *mix) {
   return mix->GetPartialElectricChargeDensity(i, rhon, MixW, VecNSp(Xi));
}

double GetElectricChargeDensity(const double rhon, const double MixW, const double *Xi, const struct Mix *mix) {
   return mix->GetElectricChargeDensity(rhon, MixW, VecNSp(Xi));
};

int8_t GetSpeciesChargeNumber(const int i, const struct Mix *mix) {
   return mix->GetSpeciesChargeNumber(i);
};

double GetDielectricPermittivity(const struct Mix *mix) {
   return mix->GetDielectricPermittivity();
};

void GetProductionRates(double *wi, const double rhon, const double Pn, const double Tn, const double *Yi, const struct Mix *mix) {
   VecNSp w;
   mix->GetProductionRates(w, rhon, Pn, Tn, VecNSp(Yi));
   for (int i=0; i < nSpec; i++) wi[i] = w[i];
};

double Getdpde(const double rho, const double gamma, const struct Mix *mix) {
   return mix->Getdpde(rho, gamma);
};

void Getdpdrhoi(double *dpdrhoi, const double gamma, const double T, const double *Yi, const struct Mix *mix) {
   VecNSp dpdrho;
   mix->Getdpdrhoi(dpdrho, gamma, T, VecNSp(Yi));
   for (int i=0; i < nSpec; i++) dpdrhoi[i] = dpdrho[i];
};

