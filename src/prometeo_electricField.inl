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

#ifndef __CUDACC__
using std::max;
using std::min;
#endif

#if (nIons > 0)
//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingIonDriftFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
inline void UpdateUsingIonDriftFluxTask<dir>::GetIonDriftFlux(
                  VecNEq &Flux,
                  const AccessorRO<VecNEq, 3>       &rhoYi,
                  const AccessorRO<VecNSp, 3>          &Yi,
                  const AccessorRO<VecNIo, 3>          &Ki,
                  const AccessorRO<  Vec3, 3>      &eField,
                  const Point<3> &p,
                  const int nType,
                  const Mix &mix,
                  const coord_t dsize,
                  const Rect<3> &bounds) {

   constexpr int iN = normalIndex(dir);

   // Compute points of the stencil
   const Point<3> pts[] = {warpPeriodic<dir, Minus>(bounds, p, dsize, offM2(nType)),
                           warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType)),
                           p,
                           warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType)),
                           warpPeriodic<dir, Plus >(bounds, p, dsize, offP2(nType)),
                           warpPeriodic<dir, Plus >(bounds, p, dsize, offP3(nType))};

   // Loop on each ion
   double rhoYiViCorr = 0.0;
   double FluxP[6]; double FluxM[6];
   __UNROLL__
   for (int i=0; i<nIons; i++) {
      const uint8_t ind = mix.ions[i];
      double lam = 0;
      // Compute local fluxes
      __UNROLL__
      for (int j=0; j<6; j++) {
         const double Vd = mix.GetSpeciesChargeNumber(ind)*Ki[pts[j]][i]*eField[pts[j]][iN];
         lam = max(lam, fabs(Vd));
         FluxP[j] = FluxM[j] = rhoYi[pts[j]][ind]*Vd;
      }
      // Compute +/- fluxes
      __UNROLL__
      for (int j=0; j<6; j++) {
         FluxP[j] += rhoYi[pts[j]][ind]*lam;
         FluxM[j] -= rhoYi[pts[j]][ind]*lam;
      }
      // Reconstruct with TENO-A and combine
      Flux[ind] = -0.5*(
            TENO_Op<-60>::reconstructPlus( FluxP[0], FluxP[1], FluxP[2], FluxP[3], FluxP[4], FluxP[5], nType) +
            TENO_Op<-60>::reconstructMinus(FluxM[0], FluxM[1], FluxM[2], FluxM[3], FluxM[4], FluxM[5], nType));
      // Store correction
      rhoYiViCorr += Flux[ind];
   }

   // Apply mass conservation correction
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      Flux[i] -= Interp2Staggered(nType, Yi[pts[2]][i],  Yi[pts[3]][i])*rhoYiViCorr;
}

template<direction dir>
__CUDA_H__
void UpdateUsingIonDriftFluxTask<dir>::updateRHSSpan(
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_e,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<VecNEq, 3> &rhoYi,
                           const AccessorRO<VecNSp, 3> &Yi,
                           const AccessorRO<VecNIo, 3> &Ki,
                           const AccessorRO<  Vec3, 3> &eField,
                           const coord_t firstIndex,
                           const coord_t lastIndex,
                           const int x,
                           const int y,
                           const int z,
                           const Rect<3> &Flux_bounds,
                           const Rect<3> &Fluid_bounds,
                           const Mix &mix) {

   const coord_t size = getSize<dir>(Fluid_bounds);
   VecNEq DriftFluxM; VecNEq DriftFluxP;
   DriftFluxM.init(0.0); DriftFluxP.init(0.0);
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      GetIonDriftFlux(DriftFluxM,
                       rhoYi, Yi, Ki, eField,
                       pm1, nType[pm1], mix, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Update plus flux
      GetIonDriftFlux(DriftFluxP,
                       rhoYi, Yi, Ki, eField,
                       p, nType[p], mix, size, Fluid_bounds);

      // Update time derivative
      Conserved_t[p] += m_e[p]*(DriftFluxP - DriftFluxM);

      // Store plus flux for next point
      DriftFluxM = DriftFluxP;
   }
}
#endif

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR AddIonWindSourcesTask
//-----------------------------------------------------------------------------
__CUDA_H__
void AddIonWindSourcesTask::addIonWindSources(VecNEq &RHS,
                            const AccessorRO<double, 3>        &rho,
                            const AccessorRO<VecNSp, 3>         &Di,
#if (nIons > 0)
                            const AccessorRO<VecNIo, 3>         &Ki,
#endif
                            const AccessorRO<  Vec3, 3>   &velocity,
                            const AccessorRO<  Vec3, 3>     &eField,
                            const AccessorRO<VecNSp, 3> &MolarFracs,
                            const AccessorRO<   int, 3>    &nType_x,
                            const AccessorRO<   int, 3>    &nType_y,
                            const AccessorRO<   int, 3>    &nType_z,
                            const AccessorRO<double, 3>        &m_x,
                            const AccessorRO<double, 3>        &m_y,
                            const AccessorRO<double, 3>        &m_z,
                            const Point<3> &p,
                            const  Rect<3> &bounds,
                            const      Mix &mix) {

   const double MixW = mix.GetMolarWeightFromXi(MolarFracs[p]);

#if (nIons > 0)
   __UNROLL__
   for (uint8_t i=0; i<nIons; i++) {
      const uint8_t ind = mix.ions[i];
      const double rhoq_i = mix.GetPartialElectricChargeDensity(ind, rho[p], MixW, MolarFracs[p]);
      // Add electric force due to this ion to the momentum equation
      __UNROLL__
      for (int j=0; j<3; j++)
         RHS[irU+j] += rhoq_i*eField[p][j];
      // Add Jule heating contribution of this species to energy equation
      // TODO: neglecting the diffusion velocity correction here (its influence is O(Y_i^2))
      Vec3 Vi = mix.GetSpeciesChargeNumber(ind)*Ki[p][i]*eField[p];
      Vi -= Di[p][ind]/MolarFracs[p][ind]*getGrad(MolarFracs, p, ind,
                                  nType_x[p], nType_y[p], nType_z[p],
                                  m_x[p], m_y[p], m_z[p],
                                  bounds);
      Vi += velocity[p];
      RHS[irE] += rhoq_i*Vi.dot(eField[p]);
   }
#else
   const double rhoq = mix.GetElectricChargeDensity(rho[p], MixW, MolarFracs[p]);
   // Add electric force
   __UNROLL__
   for (int i=0; i<3; i++)
      RHS[irU+i] += rhoq*eField[p][i];
   RHS[irE] += rhoq*eField[p].dot(velocity[p]);
#endif
}

