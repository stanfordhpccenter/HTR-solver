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
inline void AverageUtils<N>::AvgPrimitive(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<  Vec3, 3> &centerCoordinates,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorSumRD<double, N> &avg_weight,
                          const AccessorSumRD<  Vec3, N> &avg_centerCoordinates,
                          const AccessorSumRD<double, N> &pressure_avg,
                          const AccessorSumRD<double, N> &pressure_rms,
                          const AccessorSumRD<double, N> &temperature_avg,
                          const AccessorSumRD<double, N> &temperature_rms,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_avg,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_rms,
                          const AccessorSumRD<VecNSp, N> &MassFracs_avg,
                          const AccessorSumRD<VecNSp, N> &MassFracs_rms,
                          const AccessorSumRD<  Vec3, N> &velocity_avg,
                          const AccessorSumRD<  Vec3, N> &velocity_rms,
                          const AccessorSumRD<  Vec3, N> &velocity_rey,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double deltaTime) {

   const double weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

   // Average weight
   avg_weight[pA] <<= weight;

   // Grid point
   avg_centerCoordinates[pA] <<= centerCoordinates[p]*weight;

   // Primitive variables
   pressure_avg[pA]    <<= weight*pressure[p];
   pressure_rms[pA]    <<= weight*pressure[p]*pressure[p];
   temperature_avg[pA] <<= weight*temperature[p];
   temperature_rms[pA] <<= weight*temperature[p]*temperature[p];
   MolarFracs_avg[pA]  <<= weight*MolarFracs[p];
   MolarFracs_rms[pA]  <<= weight*MolarFracs[p]*MolarFracs[p];
   MassFracs_avg[pA]   <<= weight*MassFracs[p];
   MassFracs_rms[pA]   <<= weight*MassFracs[p]*MassFracs[p];
   velocity_avg[pA]    <<= weight*velocity[p];
   velocity_rms[pA]    <<= weight*velocity[p]*velocity[p];
   Vec3 rey;
   rey[0] = velocity[p][0]*velocity[p][1]*weight;
   rey[1] = velocity[p][0]*velocity[p][2]*weight;
   rey[2] = velocity[p][1]*velocity[p][2]*weight;
   velocity_rey[pA]    <<= rey;
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::FavreAvgPrimitive(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorSumRD<double, N> &pressure_favg,
                          const AccessorSumRD<double, N> &pressure_frms,
                          const AccessorSumRD<double, N> &temperature_favg,
                          const AccessorSumRD<double, N> &temperature_frms,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_favg,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_frms,
                          const AccessorSumRD<VecNSp, N> &MassFracs_favg,
                          const AccessorSumRD<VecNSp, N> &MassFracs_frms,
                          const AccessorSumRD<  Vec3, N> &velocity_favg,
                          const AccessorSumRD<  Vec3, N> &velocity_frms,
                          const AccessorSumRD<  Vec3, N> &velocity_frey,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double deltaTime) {

   const double weight = rho[p]*cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

   pressure_favg[pA]    <<= weight*pressure[p];
   pressure_frms[pA]    <<= weight*pressure[p]*pressure[p];
   temperature_favg[pA] <<= weight*temperature[p];
   temperature_frms[pA] <<= weight*temperature[p]*temperature[p];
   MolarFracs_favg[pA]  <<= weight*MolarFracs[p];
   MolarFracs_frms[pA]  <<= weight*MolarFracs[p]*MolarFracs[p];
   MassFracs_favg[pA]   <<= weight*MassFracs[p];
   MassFracs_frms[pA]   <<= weight*MassFracs[p]*MassFracs[p];
   velocity_favg[pA]    <<= weight*velocity[p];
   velocity_frms[pA]    <<= weight*velocity[p]*velocity[p];
   Vec3 rey;
   rey[0] = velocity[p][0]*velocity[p][1]*weight;
   rey[1] = velocity[p][0]*velocity[p][2]*weight;
   rey[2] = velocity[p][1]*velocity[p][2]*weight;
   velocity_frey[pA]    <<= rey;
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgProperties(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &rho_avg,
                          const AccessorSumRD<double, N> &rho_rms,
                          const AccessorSumRD<double, N> &mu_avg,
                          const AccessorSumRD<double, N> &lam_avg,
                          const AccessorSumRD<VecNSp, N> &Di_avg,
                          const AccessorSumRD<double, N> &SoS_avg,
                          const AccessorSumRD<double, N> &cp_avg,
                          const AccessorSumRD<double, N> &Ent_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime) {

   const double weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

   const double cp = mix.GetHeatCapacity(temperature[p], MassFracs[p]);
   double Ent  = 0.0;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      Ent += MassFracs[p][i]*mix.GetSpeciesEnthalpy(i, temperature[p]);

   rho_avg[pA] <<= weight*rho[p];
   rho_rms[pA] <<= weight*rho[p]*rho[p];
   mu_avg[pA]  <<= weight*mu[p];
   lam_avg[pA] <<= weight*lam[p];
   Di_avg[pA]  <<= weight*Di[p];
   SoS_avg[pA] <<= weight*SoS[p];
   cp_avg[pA]  <<= weight*cp;
   Ent_avg[pA] <<= weight*Ent;

};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::FavreAvgProperties(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &mu_favg,
                          const AccessorSumRD<double, N> &lam_favg,
                          const AccessorSumRD<VecNSp, N> &Di_favg,
                          const AccessorSumRD<double, N> &SoS_favg,
                          const AccessorSumRD<double, N> &cp_favg,
                          const AccessorSumRD<double, N> &Ent_favg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime) {

   const double weight = rho[p]*cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

   const double cp = mix.GetHeatCapacity(temperature[p], MassFracs[p]);
   double Ent  = 0.0;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      Ent += MassFracs[p][i]*mix.GetSpeciesEnthalpy(i, temperature[p]);

   mu_favg[pA]  <<= weight*mu[p];
   lam_favg[pA] <<= weight*lam[p];
   Di_favg[pA]  <<= weight*Di[p];
   SoS_favg[pA] <<= weight*SoS[p];
   cp_favg[pA]  <<= weight*cp;
   Ent_favg[pA] <<= weight*Ent;

};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgKineticEnergyBudget(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<  Vec3, 3> &vGradX,
                          const AccessorRO<  Vec3, 3> &vGradY,
                          const AccessorRO<  Vec3, 3> &vGradZ,
                          const AccessorSumRD<  Vec3, N> &rhoUUv_avg,
                          const AccessorSumRD<  Vec3, N> &Up_avg,
                          const AccessorSumRD<TauMat, N> &tau_avg,
                          const AccessorSumRD<  Vec3, N> &utau_y_avg,
                          const AccessorSumRD<  Vec3, N> &tauGradU_avg,
                          const AccessorSumRD<  Vec3, N> &pGradU_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime) {

   const double weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

   TauMat tau;
   tau(0, 0) = mu[p]*(4.0*vGradX[p][0] - 2.0*vGradY[p][1] - 2.0*vGradZ[p][2])/3.0;
   tau(0, 1) = mu[p]*(vGradX[p][1] + vGradY[p][0]);
   tau(0, 2) = mu[p]*(vGradZ[p][0] + vGradX[p][2]);
   tau(1, 1) = mu[p]*(4.0*vGradY[p][1] - 2.0*vGradX[p][0] - 2.0*vGradZ[p][2])/3.0;
   tau(1, 2) = mu[p]*(vGradY[p][2] + vGradZ[p][1]);
   tau(2, 2) = mu[p]*(4.0*vGradZ[p][2] - 2.0*vGradX[p][0] - 2.0-vGradY[p][1])/3.0;

   // Kinetic energy budgets (y is the inhomogeneous direction)
   rhoUUv_avg[pA] <<= velocity[p]*velocity[p]*(rho[p]*velocity[p][1]*weight);
   Up_avg[pA] <<= velocity[p]*(pressure[p]*weight);
   tau_avg[pA] <<= tau*weight;
   {
      Vec3 utau_y;
      utau_y[0] = velocity[p][0]*tau(0, 1)*weight;
      utau_y[1] = velocity[p][1]*tau(1, 1)*weight;
      utau_y[2] = velocity[p][2]*tau(2, 1)*weight;
      utau_y_avg[pA] <<= utau_y*weight;
   }
   {
      Vec3 tauGradU;
      __UNROLL__
      for (int i=0; i<3; i++)
         tauGradU[i] = tau(i, 0)*vGradX[p][i] + tau(i, 1)*vGradY[p][i] + tau(i, 2)*vGradZ[p][i];
      tauGradU_avg[pA] <<= tauGradU*weight;
   }
   {
      Vec3 pGradU;
      pGradU[0] = vGradX[p][0]*pressure[p];
      pGradU[1] = vGradY[p][1]*pressure[p];
      pGradU[2] = vGradZ[p][2]*pressure[p];
      pGradU_avg[pA] <<= pGradU*weight;
   }
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgFluxes_ProdRates(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<   int, 3> &nType_x,
                          const AccessorRO<   int, 3> &nType_y,
                          const AccessorRO<   int, 3> &nType_z,
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                          const AccessorRO<VecNIo, 3> &Ki,
                          const AccessorRO<  Vec3, 3> &eField,
#endif
                          const AccessorSumRD<  Vec3, N> &q_avg,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_avg,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_rms,
                          const AccessorSumRD<double, N> &HeatReleaseRate_avg,
                          const AccessorSumRD<double, N> &HeatReleaseRate_rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Rect<3> &Fluid_bounds,
                          const Mix &mix,
                          const double deltaTime) {

   const double weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

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

   // Chemical production rates
   VecNSp w; mix.GetProductionRates(w, rho[p], pressure[p], temperature[p], MassFracs[p]);
   double HR = 0.0;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      HR += w[i]*mix.GetSpeciesEnthalpy(i, temperature[p]);

   ProductionRates_avg[pA] <<= weight*w;
   ProductionRates_rms[pA] <<= weight*w*w;
   HeatReleaseRate_avg[pA] <<= weight*HR;
   HeatReleaseRate_rms[pA] <<= weight*HR*HR;
}

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgDimensionlessNumbers(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &Pr_avg,
                          const AccessorSumRD<double, N> &Pr_rms,
                          const AccessorSumRD<double, N> &Ec_avg,
                          const AccessorSumRD<double, N> &Ec_rms,
                          const AccessorSumRD<double, N> &Ma_avg,
                          const AccessorSumRD<VecNSp, N> &Sc_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime) {

   const double weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;
   const double cp = mix.GetHeatCapacity(temperature[p], MassFracs[p]);

   const double u2 = velocity[p].dot(velocity[p]);
   const double Pr = cp*mu[p]/lam[p];
   const double Ec = u2/(cp*temperature[p]);
   const double nu = mu[p]/rho[p];
   VecNSp Sc;
   __UNROLL__
   for (int i=0; i < nSpec; i++)
      Sc[i] = nu/Di[p][i];

   Pr_avg[pA] <<= weight*Pr;
   Pr_rms[pA] <<= weight*Pr*Pr;
   Ec_avg[pA] <<= weight*Ec;
   Ec_rms[pA] <<= weight*Ec*Ec;
   Ma_avg[pA] <<= weight*sqrt(u2)/SoS[p];
   Sc_avg[pA] <<= weight*Sc;
};

template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgCorrelations(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<  Vec3, N> &uT_avg,
                          const AccessorSumRD<  Vec3, N> &uT_favg,
                          const AccessorSumRD<VecNSp, N> &uYi_avg,
                          const AccessorSumRD<VecNSp, N> &vYi_avg,
                          const AccessorSumRD<VecNSp, N> &wYi_avg,
                          const AccessorSumRD<VecNSp, N> &uYi_favg,
                          const AccessorSumRD<VecNSp, N> &vYi_favg,
                          const AccessorSumRD<VecNSp, N> &wYi_favg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double deltaTime) {
   const double  weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;
   const double rweight = rho[p]*cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;

   uT_avg[pA]  <<= velocity[p]*weight;
   uT_favg[pA] <<= velocity[p]*rweight;

   uYi_avg[pA]  <<= MassFracs[p]*(velocity[p][0]*weight);
   vYi_avg[pA]  <<= MassFracs[p]*(velocity[p][1]*weight);
   wYi_avg[pA]  <<= MassFracs[p]*(velocity[p][2]*weight);
   uYi_favg[pA] <<= MassFracs[p]*(velocity[p][0]*rweight);
   vYi_favg[pA] <<= MassFracs[p]*(velocity[p][1]*rweight);
   wYi_favg[pA] <<= MassFracs[p]*(velocity[p][2]*rweight);
};

#ifdef ELECTRIC_FIELD
template<int N>
__CUDA_H__
inline void AverageUtils<N>::AvgElectricQuantities(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &ePot,
                          const AccessorSumRD<double, N> &ePot_avg,
                          const AccessorSumRD<double, N> &Crg_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime) {
   const double weight = cellWidth[p][0]*cellWidth[p][1]*cellWidth[p][2]*deltaTime;
   const double MixW = mix.GetMolarWeightFromXi(MolarFracs[p]);

   ePot_avg[pA] <<= ePot[p]*weight;
   Crg_avg[pA]  <<= mix.GetElectricChargeDensity(rho[p], MixW, MolarFracs[p])*weight;
};
#endif

