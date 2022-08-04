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

#include "prometeo_metric.inl"

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR AverageUtils
//-----------------------------------------------------------------------------

template<int N>
__CUDA_H__
inline double AverageUtils<N>::getWeight(
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const Point<3> &p,
                          const double deltaTime) {
   return deltaTime/(dcsi_d[p]*deta_d[p]*dzet_d[p]);
};

template<int N>
__CUDA_H__
inline double AverageUtils<N>::getFavreWeight(
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &rho,
                          const Point<3> &p,
                          const double deltaTime) {
   return rho[p]*deltaTime/(dcsi_d[p]*deta_d[p]*dzet_d[p]);
};

template<int N>
template<typename T>
__CUDA_H__
inline void AverageUtils<N>::Avg(
                          const T &f,
                          const AccessorSumRD<T, N> &avg,
                          const Point<N> &pA,
                          const double weight) {
   avg[pA] <<= weight*f;
};

template<int N>
template<typename T>
__CUDA_H__
inline void AverageUtils<N>::Avg(
                          const T &f,
                          const AccessorSumRD<T, N> &avg,
                          const AccessorSumRD<T, N> &rms,
                          const Point<N> &pA,
                          const double weight) {
   avg[pA] <<= weight*f;
   rms[pA] <<= weight*f*f;
};

template<int N>
template<typename T>
__CUDA_H__
inline void AverageUtils<N>::Avg(
                          const AccessorRO<T, 3> &f,
                          const AccessorSumRD<T, N> &avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   avg[pA] <<= weight*f[p];
};

template<int N>
template<typename T>
__CUDA_H__
inline void AverageUtils<N>::Avg(
                          const AccessorRO<T, 3> &f,
                          const AccessorSumRD<T, N> &avg,
                          const AccessorSumRD<T, N> &rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   avg[pA] <<= weight*f[p];
   rms[pA] <<= weight*f[p]*f[p];
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::Cor(
                          const AccessorRO<double, 3> &s,
                          const AccessorRO<  Vec3, 3> &v,
                          const AccessorSumRD<Vec3, N> &cor,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   cor[pA] <<= weight*s[p]*v[p];
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::Cor(
                          const AccessorRO<VecNSp, 3> &v1,
                          const AccessorRO<  Vec3, 3> &v2,
                          const AccessorSumRD<VecNSp, N> &cor0,
                          const AccessorSumRD<VecNSp, N> &cor1,
                          const AccessorSumRD<VecNSp, N> &cor2,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   cor0[pA] <<= weight*v1[p]*v2[p][0];
   cor1[pA] <<= weight*v1[p]*v2[p][1];
   cor2[pA] <<= weight*v1[p]*v2[p][2];
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::Avg(
                          const AccessorRO<Vec3, 3> &f,
                          const AccessorSumRD<Vec3, N> &avg,
                          const AccessorSumRD<Vec3, N> &rms,
                          const AccessorSumRD<Vec3, N> &rey,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   avg[pA] <<= weight*f[p];
   rms[pA] <<= weight*f[p]*f[p];
   Vec3 r;
   r[0] = f[p][0]*f[p][1]*weight;
   r[1] = f[p][0]*f[p][2]*weight;
   r[2] = f[p][1]*f[p][2]*weight;
   rey[pA] <<= r;
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::PositionAndWeight(
                       const AccessorRO<  Vec3, 3> &centerCoordinates,
                       const AccessorSumRD<double, N> &avg_weight,
                       const AccessorSumRD<  Vec3, N> &avg_centerCoordinates,
                       const Point<3> &p,
                       const Point<N> &pA,
                       const double weight) {
   // Average weight
   avg_weight[pA] <<= weight;
   // Grid point
   avg_centerCoordinates[pA] <<= centerCoordinates[p]*weight;
}

template<int N>
__CUDA_H__
inline void AverageUtils<N>::CpEntAvg(
                const AccessorRO<double, 3> &temperature,
                const AccessorRO<VecNSp, 3> &MassFracs,
                const AccessorSumRD<double, N> &cp_avg,
                const AccessorSumRD<double, N> &cp_favg,
                const AccessorSumRD<double, N> &Ent_avg,
                const AccessorSumRD<double, N> &Ent_favg,
                const Point<3> &p,
                const Point<N> &pA,
                const Mix &mix,
                const double weight,
                const double fweight) {
   const double cp = mix.GetHeatCapacity(temperature[p], MassFracs[p]);
   double Ent  = 0.0;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      Ent += MassFracs[p][i]*mix.GetSpeciesEnthalpy(i, temperature[p]);
   Avg(cp,  cp_avg,   pA, weight);
   Avg(cp,  cp_favg,  pA, fweight);
   Avg(Ent, Ent_avg,  pA, weight);
   Avg(Ent, Ent_favg, pA, fweight);
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::ProdRatesAvg(
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_avg,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_rms,
                          const AccessorSumRD<double, N> &HeatReleaseRate_avg,
                          const AccessorSumRD<double, N> &HeatReleaseRate_rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight) {
   VecNSp w; mix.GetProductionRates(w, rho[p], pressure[p], temperature[p], MassFracs[p]);
   double HR = 0.0;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      HR += w[i]*mix.GetSpeciesEnthalpy(i, temperature[p]);

   Avg( w, ProductionRates_avg, ProductionRates_rms, pA, weight);
   Avg(HR, HeatReleaseRate_avg, HeatReleaseRate_rms, pA, weight);
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::HeatFluxAvg(
                          const AccessorRO<   int, 3> &nType_x,
                          const AccessorRO<   int, 3> &nType_y,
                          const AccessorRO<   int, 3> &nType_z,
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                          const AccessorRO<VecNIo, 3> &Ki,
                          const AccessorRO<  Vec3, 3> &eField,
#endif
                          const AccessorSumRD<  Vec3, N> &q_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Rect<3> &Fluid_bounds,
                          const Mix &mix,
                          const double weight) {
   // Fourier heat fluxes
   q_avg[pA] <<= (-lam[p]*weight)*getGrad(temperature, p,
                                          nType_x[p], nType_y[p], nType_z[p],
                                          dcsi_d[p], deta_d[p], dzet_d[p],
                                          Fluid_bounds);
   // Species diffusion flux
   const coord_t size_x = getSize<Xdir>(Fluid_bounds);
   const coord_t size_y = getSize<Ydir>(Fluid_bounds);
   const coord_t size_z = getSize<Zdir>(Fluid_bounds);
   const Point<3> pM1_x = warpPeriodic<Xdir, Minus>(Fluid_bounds, p, size_x, offM1(nType_x[p]));
   const Point<3> pM1_y = warpPeriodic<Ydir, Minus>(Fluid_bounds, p, size_y, offM1(nType_y[p]));
   const Point<3> pM1_z = warpPeriodic<Zdir, Minus>(Fluid_bounds, p, size_z, offM1(nType_z[p]));
   const Point<3> pP1_x = warpPeriodic<Xdir, Plus >(Fluid_bounds, p, size_x, offP1(nType_x[p]));
   const Point<3> pP1_y = warpPeriodic<Ydir, Plus >(Fluid_bounds, p, size_y, offP1(nType_y[p]));
   const Point<3> pP1_z = warpPeriodic<Zdir, Plus >(Fluid_bounds, p, size_z, offP1(nType_z[p]));
   const double iMixW =  1.0/mix.GetMolarWeightFromXi(MolarFracs[p]);
   Vec3 Ji;
   Vec3 JiCorr; JiCorr.init(0);
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      Ji[0] = -getDeriv(nType_x[p], MolarFracs[pM1_x][i], MolarFracs[p][i], MolarFracs[pP1_x][i], dcsi_d[p]);
      Ji[1] = -getDeriv(nType_y[p], MolarFracs[pM1_y][i], MolarFracs[p][i], MolarFracs[pP1_y][i], deta_d[p]);
      Ji[2] = -getDeriv(nType_z[p], MolarFracs[pM1_z][i], MolarFracs[p][i], MolarFracs[pP1_z][i], dzet_d[p]);
      Ji *= (rho[p]*Di[p][i]*mix.GetSpeciesMolarWeight(i)*iMixW);
      JiCorr += Ji;
      // Add species diffusion contribution to heat flux
      q_avg[pA] <<= weight*Ji*mix.GetSpeciesEnthalpy(i, temperature[p]);
   }

#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   // add species drift
   __UNROLL__
   for (int i=0; i<nIons; i++) {
      const uint8_t ind = mix.ions[i];
      Ji = mix.GetSpeciesChargeNumber(ind)*Ki[p][i]*eField[p];
      Ji *= rho[p]*MassFracs[p][ind];
      JiCorr += Ji;
      // Add species drift contribution to heat flux
      q_avg[pA] <<= weight*Ji*mix.GetSpeciesEnthalpy(ind, temperature[p]);
   }
#endif

   __UNROLL__
   for (int i=0; i<nSpec; i++)
      // Add species diffusion correction contribution to heat flux
      q_avg[pA] <<= JiCorr*(-MassFracs[p][i]*mix.GetSpeciesEnthalpy(i, temperature[p]));
}

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgKineticEnergyBudget(
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<  Vec3, N> &rhoUUv_avg,
                          const AccessorSumRD<  Vec3, N> &Up_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   // Kinetic energy budgets (y is the inhomogeneous direction)
   rhoUUv_avg[pA] <<= velocity[p]*velocity[p]*(rho[p]*velocity[p][1]*weight);
   Up_avg[pA] <<= velocity[p]*(pressure[p]*weight);
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgKineticEnergyBudget_Tau(
                          const AccessorRO<   int, 3> &nType_x,
                          const AccessorRO<   int, 3> &nType_y,
                          const AccessorRO<   int, 3> &nType_z,
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorSumRD<TauMat, N> &tau_avg,
                          const AccessorSumRD<  Vec3, N> &utau_y_avg,
                          const AccessorSumRD<  Vec3, N> &tauGradU_avg,
                          const AccessorSumRD<  Vec3, N> &pGradU_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Rect<3> &Fluid_bounds,
                          const Mix &mix,
                          const double weight) {
   // Compute velocity gradient
   const coord_t size_x = getSize<Xdir>(Fluid_bounds);
   const coord_t size_y = getSize<Ydir>(Fluid_bounds);
   const coord_t size_z = getSize<Zdir>(Fluid_bounds);
   const Point<3> pM1_x = warpPeriodic<Xdir, Minus>(Fluid_bounds, p, size_x, offM1(nType_x[p]));
   const Point<3> pM1_y = warpPeriodic<Ydir, Minus>(Fluid_bounds, p, size_y, offM1(nType_y[p]));
   const Point<3> pM1_z = warpPeriodic<Zdir, Minus>(Fluid_bounds, p, size_z, offM1(nType_z[p]));
   const Point<3> pP1_x = warpPeriodic<Xdir, Plus >(Fluid_bounds, p, size_x, offP1(nType_x[p]));
   const Point<3> pP1_y = warpPeriodic<Ydir, Plus >(Fluid_bounds, p, size_y, offP1(nType_y[p]));
   const Point<3> pP1_z = warpPeriodic<Zdir, Plus >(Fluid_bounds, p, size_z, offP1(nType_z[p]));
   Vec3 vGradX;
   __UNROLL__
   for (int i=0; i<3; i++)
      vGradX[i] = getDeriv(nType_x[p], velocity[pM1_x][i], velocity[p][i], velocity[pP1_x][i], dcsi_d[p]);
   Vec3 vGradY;
   __UNROLL__
   for (int i=0; i<3; i++)
      vGradY[i] = getDeriv(nType_y[p], velocity[pM1_y][i], velocity[p][i], velocity[pP1_y][i], deta_d[p]);
   Vec3 vGradZ;
   __UNROLL__
   for (int i=0; i<3; i++)
      vGradZ[i] = getDeriv(nType_z[p], velocity[pM1_z][i], velocity[p][i], velocity[pP1_z][i], dzet_d[p]);

   TauMat tau;
   tau(0, 0) = mu[p]*(4.0*vGradX[0] - 2.0*vGradY[1] - 2.0*vGradZ[2])/3.0;
   tau(0, 1) = mu[p]*(vGradX[1] + vGradY[0]);
   tau(0, 2) = mu[p]*(vGradZ[0] + vGradX[2]);
   tau(1, 1) = mu[p]*(4.0*vGradY[1] - 2.0*vGradX[0] - 2.0*vGradZ[2])/3.0;
   tau(1, 2) = mu[p]*(vGradY[2] + vGradZ[1]);
   tau(2, 2) = mu[p]*(4.0*vGradZ[2] - 2.0*vGradX[0] - 2.0-vGradY[1])/3.0;

   // Kinetic energy budgets (y is the inhomogeneous direction)
   tau_avg[pA] <<= tau*weight;
   {
      Vec3 utau_y;
      utau_y[0] = velocity[p][0]*tau(0, 1);
      utau_y[1] = velocity[p][1]*tau(1, 1);
      utau_y[2] = velocity[p][2]*tau(2, 1);
      utau_y_avg[pA] <<= utau_y*weight;
   }
   {
      Vec3 tauGradU;
      __UNROLL__
      for (int i=0; i<3; i++)
         tauGradU[i] = tau(i, 0)*vGradX[i] + tau(i, 1)*vGradY[i] + tau(i, 2)*vGradZ[i];
      tauGradU_avg[pA] <<= tauGradU*weight;
   }
   {
      Vec3 pGradU;
      pGradU[0] = vGradX[0]*pressure[p];
      pGradU[1] = vGradY[1]*pressure[p];
      pGradU[2] = vGradZ[2]*pressure[p];
      pGradU_avg[pA] <<= pGradU*weight;
   }
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::PrEcAvg(
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorSumRD<double, N> &Pr_avg,
                          const AccessorSumRD<double, N> &Pr_rms,
                          const AccessorSumRD<double, N> &Ec_avg,
                          const AccessorSumRD<double, N> &Ec_rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight) {
   const double cp = mix.GetHeatCapacity(temperature[p], MassFracs[p]);
   const double u2 = velocity[p].mod2();
   const double Pr = cp*mu[p]/lam[p];
   const double Ec = u2/(cp*temperature[p]);
   Avg(Pr, Pr_avg, Pr_rms, pA, weight);
   Avg(Ec, Ec_avg, Ec_rms, pA, weight);
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::MaAvg(
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &Ma_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   const double Ma = sqrt(velocity[p].mod2())/SoS[p];
   Avg( Ma, Ma_avg, pA, weight);
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::ScAvg(
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorSumRD<VecNSp, N> &Sc_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight) {
   const double nu = mu[p]/rho[p];
   VecNSp Sc;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      Sc[i] = nu/Di[p][i];
   Avg( Sc, Sc_avg, pA, weight);
};

#ifdef ELECTRIC_FIELD
template<int N>
__CUDA_H__
inline void AverageUtils<N>::ElectricChargeAvg(
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<double, N> &Crg_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight) {
   const double MixW = mix.GetMolarWeightFromXi(MolarFracs[p]);
   Crg_avg[pA]  <<= mix.GetElectricChargeDensity(rho[p], MixW, MolarFracs[p])*weight;
};
#endif

