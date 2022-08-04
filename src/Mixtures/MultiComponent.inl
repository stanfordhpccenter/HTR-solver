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

inline const char* Mix::GetSpeciesName(const int i) const {
   return species[i].Name;
};

inline int Mix::FindSpecies(const char *Name) const {
   int iSpec = -1;
   for (int i=0; i < nSpec; i++)
      if (strcmp(species[i].Name, Name) == 0) {
         iSpec = i;
         break;
      }
   // Species not found
   assert(iSpec != -1);
   return iSpec;
};

__CUDA_HD__
inline bool Mix::CheckMixture(const VecNSp &Yi) const {
#ifdef CHECK_MIX
   double tmp = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) tmp += Yi[i];
   return (fabs(tmp - 1.0) < 1e-3);
#else
   return true;
#endif
};

__CUDA_HD__
inline void Mix::ClipYi(VecNSp &Yi) const {
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      Yi[i] = max(Yi[i], 1e-60);
      Yi[i] = min(Yi[i], 1.0);
   }
};

__CUDA_HD__
inline double Mix::GetMolarWeightFromYi(const VecNSp &Yi) const {
   double MixW = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) MixW += Yi[i]/species[i].W;
   return 1.0/MixW;
};

__CUDA_HD__
inline double Mix::GetMolarWeightFromXi(const VecNSp &Xi) const {
   double MixW = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) MixW += Xi[i]*species[i].W;
   return MixW;
};

__CUDA_HD__
inline void Mix::GetMolarFractions(VecNSp &Xi, const double MixW, const VecNSp &Yi) const {
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      Xi[i] = Yi[i]*MixW/species[i].W;
};

__CUDA_HD__
inline void Mix::GetMassFractions(VecNSp &Yi, const double MixW, const VecNSp &Xi) const {
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      Yi[i] = Xi[i]*species[i].W/MixW;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline void Mix::GetRhoYiFromYi(VecNSp &rhoYi, const double rho, const VecNSp &Yi) const {
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      rhoYi[i] = rho*Yi[i];
};

// The input and the outputs are in computational units
__CUDA_HD__
inline void Mix::GetYi(VecNSp &Yi, const double rho, const VecNSp &rhoYi) const {
   const double irho = 1.0/rho;
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      Yi[i] = irho*rhoYi[i];
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetRhoFromRhoYi(const VecNSp &rhoYi) const {
   double rho = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      rho += rhoYi[i];
   return rho;
};

// Returns rho in physical units
__CUDA_HD__
inline double Mix::GetRhoRef(const double P, const double T, const double MixW) const { return P * MixW/(RGAS * T); };

// Returns rho in computational units
__CUDA_HD__
inline double Mix::GetRho(const double P, const double T, const double MixW) const {
   return P * MixW*iMixWRef / T;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetHeatCapacity(const double Tn, const VecNSp &Yi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   double cp = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) cp += Yi[i]*species[i].GetCp(T);
   return cp*iCpRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetEnthalpy(const double Tn, const VecNSp &Yi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   double h = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      h += Yi[i]*species[i].GetEnthalpy(T);
   return h*ieRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetSpeciesEnthalpy(const int i, const double T) const {
   return species[i].GetEnthalpy(T*TRef)*ieRef;
};

__CUDA_HD__
inline double Mix::GetSpeciesMolarWeight(const int i) const {
   return species[i].W;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetInternalEnergy(const double Tn, const VecNSp &Yi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   double e = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) e += Yi[i]*(species[i].GetEnthalpy(T) - RGAS*T/species[i].W);
   return e*ieRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetSpecificInternalEnergy(const int i, const double Tn) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   return (species[i].GetEnthalpy(T) - RGAS*T/species[i].W)*ieRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetTFromInternalEnergy(const double e0, double T, const VecNSp &Yi) const {
   const int MAXITS = 1000;
   const double TOL = 1e-8;
   double dfdT = 1.0;
   int j = 0;
   while (j < MAXITS) {
      double f = e0 - GetInternalEnergy(T, Yi);
      if (fabs(f/dfdT) < TOL) break;
      dfdT = 0.0;
      __UNROLL__
      for (int i = 0; i<nSpec; i++) {
          dfdT += Yi[i]*(species[i].GetCp(T*TRef) - RGAS/species[i].W)*iCpRef;
      }
      T += f/dfdT;
      j++;
   }
#ifdef DEBUG_MULTICOMPONENT
   assert(j < MAXITS);
#endif
   return T;
};

// The inputs are in computational units
__CUDA_HD__
inline double Mix::isValidInternalEnergy(const double e, const VecNSp &Yi) const {
   return ((e > GetInternalEnergy(TMin, Yi)) and
           (e < GetInternalEnergy(TMax, Yi)));
};

// The inputs and the output are in computational units
__CUDA_HD__
inline double Mix::GetTFromRhoAndP(const double rho, const double MixW, const double P) const {
   return P*MixW*iMixWRef/rho;
};

// The inputs and the output are in computational units
__CUDA_HD__
inline double Mix::GetPFromRhoAndT(const double rho, const double MixW, const double T) const {
   return rho*T/(MixW*iMixWRef);
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetViscosity(const double Tn, const VecNSp &Xi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;

   VecNSp muk;
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      muk[i] = species[i].GetMu(T);

   double mu = 0.0;
// NOTE: This outer unroll increases the compile time a lot the compile time for number of speceis of O(10)
//   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      double den = 0.0;
      __UNROLL__
      for (int j = 0; j<nSpec; j++) {
         double Phi = 1 + sqrt(muk[i]/muk[j]) * pow(species[j].W/species[i].W, 0.25);
         Phi *= Phi/sqrt(8*(1 + species[i].W/species[j].W));
         den += Xi[j]*Phi;
      }
      mu += Xi[i]*muk[i]/den;
   }
   return mu*imuRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetHeatConductivity(const double Tn, const VecNSp &Xi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   double a = 0.0;
   double b = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      double lami = species[i].GetLam(T);
      a += Xi[i]*lami;
      b += Xi[i]/lami;
   }
   return 0.5*(a + 1.0/b)*ilamRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetGamma(const double T, const double MixW, const VecNSp &Yi) const {
   const double cp = GetHeatCapacity(T, Yi);
   const double Wr = MixW*iMixWRef;
   return cp/(cp - 1.0/Wr);
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetSpeedOfSound(const double T, const double gamma, const double MixW) const {
   return sqrt(gamma*T/(MixW*iMixWRef));
};

// The input and the outputs are in computational units
__CUDA_HD__
inline void Mix::GetDiffusivity(VecNSp &Di, const double Pn, const double Tn, const double MixW, const VecNSp &Xi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   const double P = Pn*PRef;
   Di.init(0.0);
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      __UNROLL__
      for (int j = 0; j<i; j++) {
         const double invDij = 1.0/species[i].GetDif(species[j], P, T);
         Di[i] += Xi[j]*invDij;
         Di[j] += Xi[i]*invDij;
      }
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      double num = 0.0;
      __UNROLL__
      for (int j = 0; j<nSpec; j++)
         if (j != i)
            num += Xi[j]*species[j].W;
      Di[i] = num/(MixW*Di[i])*iDiRef;
   }
};

#if (nIons > 0)
// The input and the outputs are in computational units
__CUDA_HD__
inline void Mix::GetElectricMobility(VecNIo &Ki, const double Pn, const double Tn, const VecNSp &Xi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   const double P = Pn*PRef;
   __UNROLL__
   for (int i = 0; i<nIons; i++) {
      uint8_t ind = ions[i];
      Ki[i] = 0.0;
      __UNROLL__
      for (int j = 0; j<nSpec; j++)
         Ki[i] += Xi[j]/species[ind].GetMob(species[j], P, T);
      Ki[i] = iKiRef/Ki[i];
   }
};
#endif

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetPartialElectricChargeDensity(const uint8_t i, const double rhon, const double MixW, const VecNSp &Xi) const {
#if (nIons > 0)
   return rhon*MixWRef/MixW*species[i].nCrg*Xi[i];
#else
   return 0;
#endif
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::GetElectricChargeDensity(const double rhon, const double MixW, const VecNSp &Xi) const {
   double rho_q = 0.0;
#if (nIons > 0)
   __UNROLL__
   for (int i = 0; i<nIons; i++)
      rho_q += species[ions[i]].nCrg*Xi[ions[i]];
#endif
   return rhon*MixWRef/MixW*rho_q;
};

__CUDA_HD__
inline int8_t Mix::GetSpeciesChargeNumber(const int i) const {
#if (nIons > 0)
   return species[i].nCrg;
#else
   return 0;
#endif
};

// The output is in computational units
__CUDA_HD__
inline double Mix::GetDielectricPermittivity() const {
   return Eps0;
};

// The inputs and the output are in computational units
__CUDA_HD__
inline void Mix::GetProductionRates(VecNSp &w, const double rhon, const double Pn, const double Tn, const VecNSp &Yi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   const double P = Pn*PRef;
   const double rho = rhon*rhoRef;

   w.init(0.0);

   VecNSp G;
   VecNSp C;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      C[i] = Yi[i]*rho/species[i].W;
      G[i] = species[i].GetFreeEnthalpy(T);
   }

#if (nReac > 0)
#if (nReac < 50)
   __UNROLL__
#endif
   for (int i = 0; i<nReac; i++)
      reactions[i].AddProductionRates(w, P, T, C, G);
#endif

#if (nTBReac > 0)
#if (nTBReac < 50)
   __UNROLL__
#endif
   for (int i = 0; i<nTBReac; i++)
      ThirdbodyReactions[i].AddProductionRates(w, P, T, C, G);
#endif

#if (nFOReac > 0)
#if (nFOReac < 50)
   __UNROLL__
#endif
   for (int i = 0; i<nFOReac; i++)
      FalloffReactions[i].AddProductionRates(w, P, T, C, G);
#endif

   // From [mol/(s m^3)] to computational units
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      w[i] *= species[i].W*iwiRef;
};

// The input and the outputs are in computational units
__CUDA_HD__
inline double Mix::Getdpde(const double rho, const double gamma) const { return rho*(gamma - 1); };

// The input and the outputs are in computational units
__CUDA_HD__
inline void Mix::Getdpdrhoi(VecNSp &dpdrhoi, const double gamma, const double Tn, const VecNSp &Yi) const {
   // Use unscaled primitive variables
   const double T = Tn*TRef;
   double e = 0.0;
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      // temporary store ei in dpdrhoi
      dpdrhoi[i] = species[i].GetEnthalpy(T) - RGAS*T/species[i].W;
      e += Yi[i]*dpdrhoi[i];
   }
   __UNROLL__
   for (int i = 0; i<nSpec; i++)
      dpdrhoi[i] = (RGAS*T/species[i].W + (gamma - 1)*(e - dpdrhoi[i]))*ieRef;
};

inline void Mix::StoreReferenceQuantities(const double PRef, const double TRef, const double LRef, const Mixture &XiRef) {
   // Check input reference quantities
   // ... reference pressure must be positive
   assert(PRef > 0);
   // ... reference length scale must be positive
   assert(LRef > 0);
   // ... reference temperature must be in the acceptable range
   // Some Tmin are very restrictive so ask for TRef > 0, for now
   assert(TRef > 0);
//   for (int i = 0; i < nSpec; i++) {
//      assert(TRef < species[i].cpCoeff.TMax);
//      assert(TRef > species[i].cpCoeff.TMin);
//   }
   // Store reference quantities...
   // ... form the input file
   this->PRef = PRef;
   this->TRef = TRef;
   for (int i = 0; i < nSpec; i++)
      this->XiRef[i] = 1.0e-60;
   double check = 0.0;
   for (unsigned int i = 0; i < XiRef.Species.length; i++) {
      Species s = XiRef.Species.values[i];
      this->XiRef[FindSpecies((char*)(s.Name))] = s.MolarFrac;
      check += s.MolarFrac;
   }
   // check that the specified mixture is physical
   assert(fabs(check - 1.0) < 1e-3);
   // ... and the derived once
   MixWRef = GetMolarWeightFromXi(this->XiRef);
   iMixWRef = 1.0/MixWRef;
   rhoRef = GetRhoRef(PRef, TRef, MixWRef);
   ieRef = rhoRef/PRef;
   iuRef = sqrt(ieRef);
   iCpRef = MixWRef/RGAS;
   imuRef  =     1.0/(LRef*sqrt(PRef*rhoRef));
   ilamRef = MixWRef/(LRef*sqrt(PRef*rhoRef)*RGAS);
   iDiRef = sqrt(rhoRef/PRef)/(LRef);
   iwiRef = LRef/sqrt(rhoRef*PRef);
   iKiRef = sqrt(PRef/rhoRef)*MixWRef/(Na*LRef*eCrg);
   Eps0 = MixWRef/(rhoRef*Na*eCrg*LRef);
   Eps0 = eps_0*PRef*Eps0*Eps0;

   // Set maximum and minimum temperature
   TMax = 1e20;
   TMin = 0.0;
   for (int i = 0; i < nSpec; i++) {
      TMax = min(TMax, species[i].cpCoeff.TMax/TRef);
      TMin = max(TMin, species[i].cpCoeff.TMin/TRef);
   }
};

