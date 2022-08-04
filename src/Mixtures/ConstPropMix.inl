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

#ifndef __CUDACC__
inline Mix::Mix(const Config &config) :
   R(config.Flow.mixture.u.ConstPropMix.gasConstant),
   gamma(config.Flow.mixture.u.ConstPropMix.gamma),
   viscosityModel(config.Flow.mixture.u.ConstPropMix.viscosityModel.type),
   Prandtl(config.Flow.mixture.u.ConstPropMix.prandtl)
{
   // This executable is expecting ConstPropMix in the input file
   assert(config.Flow.mixture.type == MixtureModel_ConstPropMix);

   // Init mixture model data
   if (viscosityModel == ViscosityModel_Constant)
      constantVisc    = config.Flow.mixture.u.ConstPropMix.viscosityModel.u.Constant.Visc;
   else if (viscosityModel == ViscosityModel_PowerLaw) {
      powerlawTempRef = config.Flow.mixture.u.ConstPropMix.viscosityModel.u.PowerLaw.TempRef;
      powerlawViscRef = config.Flow.mixture.u.ConstPropMix.viscosityModel.u.PowerLaw.ViscRef;
   } else if (viscosityModel == ViscosityModel_Sutherland) {
      sutherlandSRef    = config.Flow.mixture.u.ConstPropMix.viscosityModel.u.Sutherland.SRef;
      sutherlandTempRef = config.Flow.mixture.u.ConstPropMix.viscosityModel.u.Sutherland.TempRef;
      sutherlandViscRef = config.Flow.mixture.u.ConstPropMix.viscosityModel.u.Sutherland.ViscRef;
   }
};
#endif

inline const char* Mix::GetSpeciesName(const int i) const {
   return (char*)"MIX";
};

inline int Mix::FindSpecies(const char *Name) const {
   return 0;
};

__CUDA_HD__
inline bool Mix::CheckMixture(const VecNSp &Yi) const {
#ifdef CHECK_MIX
   return (fabs(Yi[0] - 1.0) < 1e-3);
#else
   return true;
#endif
};

__CUDA_HD__
inline double Mix::GetMolarWeightFromYi(const VecNSp &Yi) const { return RGAS/R; };

__CUDA_HD__
inline double Mix::GetMolarWeightFromXi(const VecNSp &Xi) const { return RGAS/R; };

__CUDA_HD__
inline void Mix::GetMolarFractions(VecNSp &Xi, const double MixW, const VecNSp &Yi) const { Xi[0] = Yi[0]; };

__CUDA_HD__
inline void Mix::GetMassFractions(VecNSp &Yi, const double MixW, const VecNSp &Xi) const { Yi[0] = Xi[0]; };

__CUDA_HD__
inline double Mix::GetRhoFromRhoYi(const VecNSp &rhoYi) const { return rhoYi[0]; };

__CUDA_HD__
inline void Mix::GetRhoYiFromYi(VecNSp &rhoYi, const double rho, const VecNSp &Yi) const { rhoYi[0] = rho*Yi[0]; };

__CUDA_HD__
inline void Mix::GetYi(VecNSp &Yi, const double rho, const VecNSp &rhoYi) const { Yi[0] = rhoYi[0]/rho; };

__CUDA_HD__
inline double Mix::GetRho(const double P, const double T, const double MixW) const { return P/(R * T); };

__CUDA_HD__
inline double Mix::GetHeatCapacity(const double T, const VecNSp &Yi) const { return gamma/(gamma-1)*R; };

__CUDA_HD__
inline double Mix::GetEnthalpy(const double T, const VecNSp &Yi) const { return gamma/(gamma-1)*R*T; };

__CUDA_HD__
inline double Mix::GetSpeciesEnthalpy(const int i, const double T) const { return gamma/(gamma-1)*R*T; };

__CUDA_HD__
inline double Mix::GetSpeciesMolarWeight(const int i) const { return RGAS/R; };

__CUDA_HD__
inline double Mix::GetInternalEnergy(const double T, const VecNSp &Yi) const { return T*R/(gamma-1.0); };

__CUDA_HD__
inline double Mix::GetSpecificInternalEnergy(const int i, const double T) const { return T*R/(gamma-1.0); };

__CUDA_HD__
inline double Mix::GetTFromInternalEnergy(const double e0, double T, const VecNSp &Yi) const { return e0*(gamma-1.0)/R; };

__CUDA_HD__
inline double Mix::isValidInternalEnergy(const double e, const VecNSp &Yi) const { return (e > 0); };

__CUDA_HD__
inline double Mix::GetTFromRhoAndP(const double rho, const double MixW, const double P) const { return P*MixW/(rho*RGAS); };

__CUDA_HD__
inline double Mix::GetPFromRhoAndT(const double rho, const double MixW, const double T) const { return rho*RGAS*T/MixW; };

__CUDA_HD__
inline double Mix::GetViscosity(const double T, const VecNSp &Xi) const {
   return ((viscosityModel == ViscosityModel_Constant) ? constantVisc :
          ((viscosityModel == ViscosityModel_PowerLaw) ? (powerlawViscRef*pow((T/powerlawTempRef), 0.7)) :
         /*(viscosityModel == ViscosityModel_Sutherland) ? */ (sutherlandViscRef*pow((T/sutherlandTempRef), 1.5))*
                                                             ((sutherlandTempRef+sutherlandSRef)/(T+sutherlandSRef))));
};

__CUDA_HD__
inline double Mix::GetHeatConductivity(const double T, const VecNSp &Xi) const {
   const double cp = gamma/(gamma-1)*R;
   return cp/Prandtl*GetViscosity(T, Xi);
};

__CUDA_HD__
inline double Mix::GetGamma(const double T, const double MixW, const VecNSp &Yi) const { return gamma; };

__CUDA_HD__
inline double Mix::GetSpeedOfSound(const double T, const double gamma, const double MixW) const { return sqrt(gamma*R*T); };

__CUDA_HD__
inline void Mix::GetDiffusivity(VecNSp &Di, const double P, const double T, const double MixW, const VecNSp &Xi) const { Di[0] = 0.0; };

__CUDA_HD__
inline double Mix::GetPartialElectricChargeDensity(const uint8_t i, const double rhon, const double MixW, const VecNSp &Xi) const { return 0.0; };

__CUDA_HD__
inline double Mix::GetElectricChargeDensity(const double rhon, const double MixW, const VecNSp &Xi) const { return 0.0; };

__CUDA_HD__
inline int8_t Mix::GetSpeciesChargeNumber(const int i) const { return 0; };

__CUDA_HD__
inline double Mix::GetDielectricPermittivity() const { return eps_0; };

__CUDA_HD__
inline void Mix::GetProductionRates(VecNSp &w, const double rhon, const double Pn, const double Tn, const VecNSp &Yi) const { w[0] = 0.0; };

__CUDA_HD__
inline double Mix::Getdpde(const double rho, const double gamma) const { return rho*(gamma - 1); };

__CUDA_HD__
inline void Mix::Getdpdrhoi(VecNSp &dpdrhoi, const double gamma, const double T, const VecNSp &Yi) const { dpdrhoi[0] = R*T; };

