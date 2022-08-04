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

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingEulerFluxUtils
//-----------------------------------------------------------------------------
template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(double * Sums,
                          const AccessorRO<VecNEq, 3> &Conserved,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &pressure,
                          const Point<3> &p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> &bounds) {

   // Index of normal direction
   constexpr int iN = normalIndex(dir);

   const double H = (Conserved[p][irE] + pressure[p])/rho[p];

   for (int l=0; l < KennedyNSum[nType]; l++) {
      // offset point
      const Point<3> pp = warpPeriodic<dir, Plus>(bounds, p, dsize, l+1);
      const int off_l = l*(nEq+1);

      // compute the summations
      const double rhom = rho[p] + rho[pp];
      const double vm = -(velocity[p][iN] + velocity[pp][iN]);
      __UNROLL__
      for (int i=0; i<nSpec; i++)
         Sums[off_l+i] = rhom*vm*(MassFracs[p][i] +  MassFracs[pp][i]);
      __UNROLL__
      for (int i=0; i<3; i++)
         Sums[off_l+irU+i] = rhom*vm*(velocity[p][i] + velocity[pp][i]);
      Sums[off_l+irE] = rhom*vm*(H + (Conserved[pp][irE] + pressure[pp])/rho[pp]);
      Sums[off_l+nEq] = (pressure[p] + pressure[pp]);
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(VecNEq &Flux,
                          const double *Sums,
                          const AccessorRO<VecNEq, 3> &Conserved,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &pressure,
                          const Point<3> &p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> &bounds) {

   // Index of normal direction
   constexpr int iN = normalIndex(dir);

   if (nType == L_S_node) {
      // This is a staggered node
      __UNROLL__
      for (int i=0; i<nEq; i++)
         Flux[i] = -(Conserved[p][i]*velocity[p][iN]);
      Flux[irU+iN] -= pressure[p];
      Flux[irE   ] -= pressure[p]*velocity[p][iN];

   }
   else if (nType == Rm1_S_node) {
      // This is a staggered node
      const Point<3> pp = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));
      __UNROLL__
      for (int i=0; i<nEq; i++)
         Flux[i] = -(Conserved[pp][i]*velocity[pp][iN]);
      Flux[irU+iN] -= pressure[pp];
      Flux[irE   ] -= pressure[pp]*velocity[pp][iN];

   }
   else {
      double   f[nEq+1];
      double acc[nEq+1];
      const double * Coeff = KennedyCoeff[nType];
      __UNROLL__
      for (int i=0; i<nEq+1; i++) f[i] = 0.0;
      for (int l=0; l<KennedyOrder[nType]; l++) {
         __UNROLL__
         for (int i = 0; i < nEq+1; i++) acc[i] = 0.0;
         for (int m = 0; m < l+1; m++) {
            // Sums sums is a vector of 6(nEq+1) elements rappresenting
            // a 3*3 triangular matrix whose indices are l and m
            const int off = (nEq+1)*((5-m)*m/2 + l);
            for (int i = 0; i < nEq+1; i++)
               acc[i] += Sums[off + i];
         }
         __UNROLL__
         for (int i=0; i<nEq+1; i++) f[i] += Coeff[l]*acc[i];
      }

      for (int i=0; i<nEq; i++) Flux[i] = 0.25*f[i];
      // add pressure contribution to normal momentum equation
      Flux[irU+iN] -= f[nEq];
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(VecNEq &Flux,
                          const AccessorRO<VecNEq, 3> &Conserved,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &pressure,
                          const Point<3> &p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> &bounds) {

   // Index of normal direction
   constexpr int iN = normalIndex(dir);

   // Compute points of the stencil
   const Point<3> pM2 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM2(nType));
   const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType));
   const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));
   const Point<3> pP2 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP2(nType));
   const Point<3> pP3 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP3(nType));

   // Define common arrays
   const double rhov[] = {rho[pM2], rho[pM1], rho[p  ],
                          rho[pP1], rho[pP2], rho[pP3]};
   // Put a minus here remembering that flux are on the RHS
   const double vNorm[] = {-velocity[pM2][iN], -velocity[pM1][iN], -velocity[p  ][iN],
                           -velocity[pP1][iN], -velocity[pP2][iN], -velocity[pP3][iN]};

   // Species fluxes
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double phi[] = {MassFracs[pM2][i], MassFracs[pM1][i], MassFracs[p  ][i],
                            MassFracs[pP1][i], MassFracs[pP2][i], MassFracs[pP3][i]};
      Flux[i] = KennedyReconstruct(rhov, vNorm, phi, nType);
   }
   // Momentum fluxes
   __UNROLL__
   for (int i=0; i<3; i++) {
      const double phi[] = {velocity[pM2][i], velocity[pM1][i], velocity[p  ][i],
                            velocity[pP1][i], velocity[pP2][i], velocity[pP3][i]};
      Flux[irU+i] = KennedyReconstruct(rhov, vNorm, phi, nType);
   }
   {
      const double phi[] ={pressure[pM2], pressure[pM1], pressure[p  ],
                           pressure[pP1], pressure[pP2], pressure[pP3]};
      Flux[irU+iN] -= KennedyReconstruct(phi, nType);
   }
   // Energy flux
   {
      const double phi[] = {(Conserved[pM2][irE] + pressure[pM2])/rho[pM2],
                            (Conserved[pM1][irE] + pressure[pM1])/rho[pM1],
                            (Conserved[p  ][irE] + pressure[p  ])/rho[p  ],
                            (Conserved[pP1][irE] + pressure[pP1])/rho[pP1],
                            (Conserved[pP2][irE] + pressure[pP2])/rho[pP2],
                            (Conserved[pP3][irE] + pressure[pP3])/rho[pP3]};
      Flux[irE] = KennedyReconstruct(rhov, vNorm, phi, nType);
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::ComputeRoeAverages(
                           RoeAveragesStruct  &avgs, const Mix &mix,
                           const VecNEq &ConservedL, const VecNEq &ConservedR,
                           const VecNSp        &YiL, const VecNSp        &YiR,
                           const double   pressureL, const double   pressureR,
                           const   Vec3  &velocityL, const   Vec3  &velocityR,
                           const double        rhoL, const double        rhoR) {

   // Compute quantities on the left (L) and right (R) states
   const double InternalEnergyL = ConservedL[irE]/rhoL - 0.5*velocityL.mod2();
   const double InternalEnergyR = ConservedR[irE]/rhoR - 0.5*velocityR.mod2();

   const double TotalEnthalpyL = (ConservedL[irE] + pressureL)/rhoL;
   const double TotalEnthalpyR = (ConservedR[irE] + pressureR)/rhoR;

   // Compute Roe averaged state...
   const double RoeFactorL = sqrt(rhoL)/(sqrt(rhoL) + sqrt(rhoR));
   const double RoeFactorR = sqrt(rhoR)/(sqrt(rhoL) + sqrt(rhoR));

   // ... density...
   avgs.rho = sqrt(rhoL*rhoR);

   // ... mass fractions...
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      avgs.Yi[i] = YiL[i]*RoeFactorL + YiR[i]*RoeFactorR;

   // .. velocity ...
   __UNROLL__
   for (int i=0; i<3; i++)
      avgs.velocity[i] = velocityL[i]*RoeFactorL + velocityR[i]*RoeFactorR;

   // Total enthalpy
   avgs.H =  TotalEnthalpyL*RoeFactorL +  TotalEnthalpyR*RoeFactorR;
   // Internal energy
   avgs.e = InternalEnergyL*RoeFactorL + InternalEnergyR*RoeFactorR;

   // Compute pressure derivatives based on the averaged state
   const double PovRhoRoe = avgs.H - avgs.e - 0.5*avgs.velocity.mod2();
   const double MixWRoe   = mix.GetMolarWeightFromYi(avgs.Yi);
   const double TRoe_eos  = mix.GetTFromRhoAndP(avgs.rho, MixWRoe, PovRhoRoe*avgs.rho);
   const double TRoe      = mix.GetTFromInternalEnergy(avgs.e, TRoe_eos, avgs.Yi);
   const double gammaRoe  = mix.GetGamma(TRoe, MixWRoe, avgs.Yi);
   avgs.dpde = mix.Getdpde(avgs.rho, gammaRoe);
   mix.Getdpdrhoi(avgs.dpdrhoi, gammaRoe, TRoe, avgs.Yi);

   // correct the pressure derivatives in order to satisfy the pressure jump condition
   // using the procedure in Shuen, Liou and Leer (1990)
   const double dp = pressureR - pressureL;
   const double de = InternalEnergyR - InternalEnergyL;
   VecNSp drhoi;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      drhoi[i] = rhoR*YiR[i] - rhoL*YiL[i];

   // find the error in the pressure jump due to Roe averages
   double dpError = dp - de*avgs.dpde;
   double fact = de*avgs.dpde; fact *= fact;
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double dpi = drhoi[i]*avgs.dpdrhoi[i];
      dpError -= dpi;
      fact += dpi*dpi;
   }

   // correct pressure derivatives
   // this threshold should not be affect the solution since fact is zero when all the jumps are zero
   fact = dpError/max(fact, 1e-6);
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      avgs.dpdrhoi[i] = avgs.dpdrhoi[i]*(1.0 + avgs.dpdrhoi[i]*drhoi[i]*fact);
   avgs.dpde = avgs.dpde*(1.0 + avgs.dpde*de*fact);

   // compute the Roe averaged speed of sound
   avgs.a2 = PovRhoRoe/avgs.rho*avgs.dpde;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      avgs.a2 += avgs.Yi[i]*avgs.dpdrhoi[i];
   avgs.a = sqrt(avgs.a2);
}

template<direction dir>
__CUDA_H__
inline void UpdateUsingEulerFluxUtils<dir>::computeLeftEigenvectors(MyMatrix<double, nEq, nEq> &L, const RoeAveragesStruct &avgs) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);

   // initialize L
   L.init(0);

   // Compute constants
   const double iaRoe  = 1.0/avgs.a;
   const double iaRoe2 = 1.0/avgs.a2;
   const double Coeff = (avgs.e - 0.5*avgs.velocity.mod2())*avgs.dpde/avgs.rho;
   double b[nSpec];
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      b[i] = (Coeff - avgs.dpdrhoi[i])*iaRoe2;
   const double d = avgs.dpde/(avgs.rho*avgs.a2);
   const Vec3 c(avgs.velocity*d);

   // First row
   {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L(0, j) = -0.5*(b[j] - avgs.velocity[iN]*iaRoe);
      __UNROLL__
      for (int j=0; j<3; j++)
         L(0, nSpec+j ) = -0.5*c[j];
      L(0,    nSpec+iN) -= 0.5*iaRoe;
      L(0,    nSpec+3 ) = 0.5*d;
   }

   // From 1 to nSpec
   __UNROLL__
   for (int i=1; i<nSpec+1; i++) {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L(i, j) = avgs.Yi[i-1]*b[j];
      L(i, i-1) += 1.0;
      __UNROLL__
      for (int j=0; j<3; j++)
         L(i, nSpec+j) = avgs.Yi[i-1]*c[j];
      L(i, nSpec+3) = -avgs.Yi[i-1]*d;
   }

   // nSpec + 1
   {
      const int row = (nSpec+1);
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L(row, j) = - avgs.velocity[iT1];
      // L(row, nSpec+iN) = 0.0;
      L(row, nSpec+iT1) = 1.0;
      //L(row, nSpec+iT2) = 0.0;
      //L(row, nSpec+  3) = 0.0;
   }

   // nSpec + 2
   {
      const int row = (nSpec+2);
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L(row, j) = - avgs.velocity[iT2];
      //L(row, nSpec+iN ) = 0.0;
      //L(row, nSpec+iT1) = 0.0;
      L(row, nSpec+iT2) = 1.0;
      //L(row, nSpec+  3) = 0.0;
   }

   // nSpec+3
   {
      const int row = (nSpec+3);
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         // (nEq-1)*nEq + j
         L(row, j) = -0.5*(b[j] + avgs.velocity[iN]*iaRoe);
      __UNROLL__
      for (int j=0; j<3; j++)
         L(row, nSpec+j) = -0.5*c[j];
      L(row, nSpec+iN) += 0.5*iaRoe;
      L(row, nSpec+3 ) = 0.5*d;
   }
}

template<direction dir>
__CUDA_H__
inline void UpdateUsingEulerFluxUtils<dir>::computeRightEigenvectors(MyMatrix<double, nEq, nEq> &K, const RoeAveragesStruct &avgs) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);

   // initialize K
   K.init(0);

   // fill K
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      K(i,       0) = avgs.Yi[i];
      K(i,     i+1) = 1.0;
      K(i, nSpec+3) = avgs.Yi[i];
   }

   {
      const int row = (nSpec+iN);
      K(row,       0) = avgs.velocity[iN] - avgs.a;
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K(row,  i+1) = avgs.velocity[iN];
      //K(row, nSpec+1) = 0.0;
      //K(row, nSpec+2) = 0.0;
      K(row, nSpec+3) = avgs.velocity[iN] + avgs.a;
   }

   {
      const int row = (nSpec+iT1);
      K(row,       0) = avgs.velocity[iT1];
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K(row,  i+1) = avgs.velocity[iT1];
      K(row, nSpec+1) = 1.0;
      //K(row, nSpec+2) = 0.0;
      K(row, nSpec+3) = avgs.velocity[iT1];
   }

   {
      const int row = (nSpec+iT2);
      K(row,       0) = avgs.velocity[iT2];
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K(row,  i+1) = avgs.velocity[iT2];
      //K(row, nSpec+1) = 0.0;
      K(row, nSpec+2) = 1.0;
      K(row, nSpec+3) = avgs.velocity[iT2];
   }

   {
      const int row = (nSpec+3);
      K(row,       0) = avgs.H - avgs.velocity[iN]*avgs.a;
      const double dedp = 1.0/avgs.dpde;
      const double ERoe = avgs.e + 0.5*avgs.velocity.mod2();
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K(row,  i+1) = ERoe - avgs.rho*avgs.dpdrhoi[i]*dedp;
      K(row, nSpec+1) = avgs.velocity[iT1];
      K(row, nSpec+2) = avgs.velocity[iT2];
      K(row, nSpec+3) = avgs.H + avgs.velocity[iN]*avgs.a;
   }
}

// Projects the state vector q in the characteristic space from the physiscal space
template<direction dir>
__CUDA_H__
inline void UpdateUsingEulerFluxUtils<dir>::projectToCharacteristicSpace(VecNEq &r, const VecNEq &q, const RoeAveragesStruct &avgs) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);

   // Compute constants
   const double iaRoe  = 1.0/avgs.a;
   const double iaRoe2 = 1.0/avgs.a2;
   const double Coeff = (avgs.e - 0.5*avgs.velocity.mod2())*avgs.dpde/avgs.rho;
   double sum1 = 0.0;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      sum1 += (Coeff - avgs.dpdrhoi[i])*iaRoe2*q[i];
   double sumQsp = 0.0;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      sumQsp += q[i];
   const double d = avgs.dpde/(avgs.rho*avgs.a2);
   const Vec3 c(avgs.velocity*d);

   // First row
   {
      r[0] = sumQsp*avgs.velocity[iN]*iaRoe - sum1;
      r[0] -= (c[iN] + iaRoe)*q[nSpec+iN];
      r[0] -= c[iT1]*q[nSpec+iT1];
      r[0] -= c[iT2]*q[nSpec+iT2];
      r[0] += d*q[nSpec+3];
      r[0] *= 0.5;
   }

   // From 1 to nSpec
   __UNROLL__
   for (int i=1; i<nSpec+1; i++) {
      r[i] = avgs.Yi[i-1]*sum1;
      r[i] += q[i-1];
      __UNROLL__
      for (int j=0; j<3; j++)
         r[i] += avgs.Yi[i-1]*c[j]*q[nSpec+j];
      r[i] += -avgs.Yi[i-1]*d*q[nSpec+3];
   }

   // nSpec + 1
   {
      r[nSpec+1] = -avgs.velocity[iT1]*sumQsp;
      //r[nSpec+1] += 0.0*q[nSpec+iN];
      r[nSpec+1] += q[nSpec+iT1];
      //r[nSpec+1] += 0.0*q[nSpec+iT2];
      //r[nSpec+1] += 0.0*q[nSpec+3  ];
   }

   // nSpec + 2
   {
      r[nSpec+2] = -avgs.velocity[iT2]*sumQsp;
      //r[nSpec+2] += 0.0*q[nSpec+iN ];
      //r[nSpec+2] += 0.0*q[nSpec+iT1];
      r[nSpec+2] += q[nSpec+iT2];
      //r[nSpec+2] += 0.0*q[nSpec+3  ];
   }

   // nSpec + 3
   {
      r[nSpec+3] = -sumQsp*avgs.velocity[iN]*iaRoe - sum1;
      r[nSpec+3] -= (c[iN] - iaRoe)*q[nSpec+iN];
      r[nSpec+3] -= c[iT1]*q[nSpec+iT1];
      r[nSpec+3] -= c[iT2]*q[nSpec+iT2];
      r[nSpec+3] += d*q[nSpec+3  ];
      r[nSpec+3] *= 0.5;
   }
}

// Projects the state vector q from the characteristic space to the physiscal space
template<direction dir>
__CUDA_H__
inline void UpdateUsingEulerFluxUtils<dir>::projectFromCharacteristicSpace(VecNEq &r, const VecNEq &q, const RoeAveragesStruct &avgs) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);

   // initialize r
   r.init(0);

   // Compute constants
   double sumQsp = 0.0;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      sumQsp += q[i+1];

   // First nSpec rows
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      r[i] += avgs.Yi[i]*q[0];
      r[i] += q[i+1];
      r[i] += avgs.Yi[i]*q[nSpec+3];
   }

   // nSpec+iN row
   {
      r[nSpec+iN] += (avgs.velocity[iN] - avgs.a)*q[0];
      r[nSpec+iN] += avgs.velocity[iN]*sumQsp;
      //r[nSpec+iN] += 0.0*q[nSpec+1];
      //r[nSpec+iN] += 0.0*q[nSpec+2];
      r[nSpec+iN] += (avgs.velocity[iN] + avgs.a)*q[nSpec+3];
   }

   // nSpec+iT1 row
   {
      r[nSpec+iT1] += avgs.velocity[iT1]*q[0];
      r[nSpec+iT1] += avgs.velocity[iT1]*sumQsp;
      r[nSpec+iT1] += q[nSpec+1];
      //r[nSpec+iT1] += 0.0*q[nSpec+2];
      r[nSpec+iT1] += avgs.velocity[iT1]*q[nSpec+3];
   }

   // nSpec+iT2 row
   {
      r[nSpec+iT2] += avgs.velocity[iT2]*q[0];
      r[nSpec+iT2] += avgs.velocity[iT2]*sumQsp;
      //r[nSpec+iT2] += 0.0*q[nSpec+1];
      r[nSpec+iT2] += q[nSpec+2];
      r[nSpec+iT2] += avgs.velocity[iT2]*q[nSpec+3];
   }

   // nSpec+3 row
   {
      r[nSpec+3] += (avgs.H - avgs.velocity[iN]*avgs.a)*q[0];
      const double dedp = 1.0/avgs.dpde;
      const double ERoe = avgs.e + 0.5*avgs.velocity.mod2();
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         r[nSpec+3] += (ERoe - avgs.rho*avgs.dpdrhoi[i]*dedp)*q[i+1];
      r[nSpec+3] += avgs.velocity[iT1]*q[nSpec+1];
      r[nSpec+3] += avgs.velocity[iT2]*q[nSpec+2];
      r[nSpec+3] += (avgs.H + avgs.velocity[iN]*avgs.a)*q[nSpec+3];
   }
}

// Projects the state vector q from the characteristic space to the physiscal space for one species
template<direction dir>
__CUDA_H__
inline double UpdateUsingEulerFluxUtils<dir>::projectFromCharacteristicSpace(const int i, const VecNEq &q, const RoeAveragesStruct &avgs) {
   double r = avgs.Yi[i]*q[0];
   r += q[i+1];
   r += avgs.Yi[i]*q[nSpec+3];
   return r;
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::getPlusMinusFlux(VecNEq &FluxP, VecNEq &FluxM,
                      const RoeAveragesStruct &avgs,
                      const VecNEq &Conserved,
                      const double velocity,
                      const double pressure,
                      const double Lam1,
                      const double Lam,
                      const double LamN) {

   constexpr int iN = normalIndex(dir);

   // Compute the Euler fluxes
   VecNEq Flux;
   __UNROLL__
   for (int i=0; i<nEq; i++)
      Flux[i] = Conserved[i]*velocity;
   Flux[irU+iN] += pressure;
   Flux[irE   ] += pressure*velocity;

   // Project in the characteristic space
   VecNEq Q; projectToCharacteristicSpace(Q, Conserved, avgs);
   VecNEq F; projectToCharacteristicSpace(F,      Flux, avgs);

   // Plus fluxes
   FluxP[    0] = F[    0] + Lam1*Q[    0];
   __UNROLL__
   for (int i=1; i < nEq-1; i++) FluxP[i] = F[i] + Lam*Q[i];
   FluxP[nEq-1] = F[nEq-1] + LamN*Q[nEq-1];

   // Minus fluxes
   FluxM[    0] = F[    0] - Lam1*Q[    0];
   __UNROLL__
   for (int i=1; i < nEq-1; i++) FluxM[i] = F[i] - Lam*Q[i];
   FluxM[nEq-1] = F[nEq-1] - LamN*Q[nEq-1];
}

template<direction dir>
template<class Op>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::FluxReconstruction(VecNEq &Flux,
                            const AccessorRO<VecNEq, 3> &Conserved,
#if (nSpec > 1)
                            const AccessorRO<VecNEq, 3> &Conserved_old,
#endif
                            const AccessorRO<double, 3> &SoS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const Point<3> &p,
                            const int      nType,
#if (nSpec > 1)
                            const double   RK_coeffs0,
                            const double   RK_coeffs1,
                            const double   lim_f,
#endif
                            const Mix      &mix,
                            const coord_t  dsize,
                            const Rect<3>  &bounds) {

   constexpr int iN = normalIndex(dir);

   // Compute points of the stencil
   const Point<3> pM2 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM2(nType));
   const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType));
   const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));
   const Point<3> pP2 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP2(nType));
   const Point<3> pP3 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP3(nType));

   // Compute maximum eigenvalues
   const double Lam1 = max(max(max(max(max(
                                           fabs(velocity[pM2][iN] - SoS[pM2]),
                                           fabs(velocity[pM1][iN] - SoS[pM1])),
                                           fabs(velocity[p  ][iN] - SoS[p  ])),
                                           fabs(velocity[pP1][iN] - SoS[pP1])),
                                           fabs(velocity[pP2][iN] - SoS[pP2])),
                                           fabs(velocity[pP3][iN] - SoS[pP3]));

   const double Lam  = max(max(max(max(max(
                                           fabs(velocity[pM2][iN]),
                                           fabs(velocity[pM1][iN])),
                                           fabs(velocity[p  ][iN])),
                                           fabs(velocity[pP1][iN])),
                                           fabs(velocity[pP2][iN])),
                                           fabs(velocity[pP3][iN]));

   const double LamN = max(max(max(max(max(
                                           fabs(velocity[pM2][iN] + SoS[pM2]),
                                           fabs(velocity[pM1][iN] + SoS[pM1])),
                                           fabs(velocity[p  ][iN] + SoS[p  ])),
                                           fabs(velocity[pP1][iN] + SoS[pP1])),
                                           fabs(velocity[pP2][iN] + SoS[pP2])),
                                           fabs(velocity[pP3][iN] + SoS[pP3]));

   // Compute the RoeAverages
   RoeAveragesStruct RoeAvgs;
   ComputeRoeAverages(RoeAvgs,              mix,
                 Conserved[p],   Conserved[pP1],
                 MassFracs[p],   MassFracs[pP1],
                  pressure[p],    pressure[pP1],
                  velocity[p],    velocity[pP1],
                       rho[p],         rho[pP1]);

   // Compute +/- fluxes
   VecNEq FluxPM2; VecNEq FluxMM2; getPlusMinusFlux(FluxPM2, FluxMM2, RoeAvgs, Conserved[pM2], velocity[pM2][iN], pressure[pM2], Lam1, Lam, LamN);
   VecNEq FluxPM1; VecNEq FluxMM1; getPlusMinusFlux(FluxPM1, FluxMM1, RoeAvgs, Conserved[pM1], velocity[pM1][iN], pressure[pM1], Lam1, Lam, LamN);
   VecNEq FluxP  ; VecNEq FluxM  ; getPlusMinusFlux(FluxP  , FluxM  , RoeAvgs, Conserved[p  ], velocity[p  ][iN], pressure[p  ], Lam1, Lam, LamN);
   VecNEq FluxPP1; VecNEq FluxMP1; getPlusMinusFlux(FluxPP1, FluxMP1, RoeAvgs, Conserved[pP1], velocity[pP1][iN], pressure[pP1], Lam1, Lam, LamN);
   VecNEq FluxPP2; VecNEq FluxMP2; getPlusMinusFlux(FluxPP2, FluxMP2, RoeAvgs, Conserved[pP2], velocity[pP2][iN], pressure[pP2], Lam1, Lam, LamN);
   VecNEq FluxPP3; VecNEq FluxMP3; getPlusMinusFlux(FluxPP3, FluxMP3, RoeAvgs, Conserved[pP3], velocity[pP3][iN], pressure[pP3], Lam1, Lam, LamN);

   // Reconstruct Fluxes
   VecNEq F;
   for (int i=0; i<nEq; i++) {
      const double FPlus  = Op::reconstructPlus(FluxPM2[i], FluxPM1[i], FluxP  [i],
                                                FluxPP1[i], FluxPP2[i], FluxPP3[i], nType);
      const double FMinus = Op::reconstructMinus(FluxMM2[i], FluxMM1[i], FluxM  [i],
                                                 FluxMP1[i], FluxMP2[i], FluxMP3[i], nType);
      F[i] = -0.5*(FPlus + FMinus);
   }

   // Go back to the physical space
   projectFromCharacteristicSpace(Flux, F, RoeAvgs);

#if (nSpec > 1)
   // Flux limiter (blend high order flux with first order)
   // Do we need the limiter?
   // lets allow slightly negative values before activating the limiter
   constexpr double eps_l = -1e-8; // threshold for rho
   bool limit = false;
   for (int i=0; i<nSpec; i++) {
      const double dRhoi = lim_f*Flux[i];
      if (((RK_coeffs0*Conserved_old[p  ][i] + RK_coeffs1*Conserved[p  ][i] + dRhoi) < eps_l) or
          ((RK_coeffs0*Conserved_old[pP1][i] + RK_coeffs1*Conserved[pP1][i] - dRhoi) < eps_l))
         limit = true;
   }

   if (limit) {
      constexpr double eps = 1e-60; // threshold for rho
      // We need the limiter
      // Initialize the blending parameter to 1.0
      double theta = 1.0;
      // First-order flux in the characteristic space
      F = -0.5*(FluxP + FluxMP1);
      // Let's use FluxP to store the first-order reconstructed flux in physical space
      projectFromCharacteristicSpace(FluxP, F, RoeAvgs);
      for (int i=0; i<nSpec; i++) {
         // These are the predicted increments
         const double dRhoi    = lim_f*Flux[i];
         const double dRhoi_lo = lim_f*FluxP[i];
         // Check p side
         double theta_p   = 1.0;
         const double rhoi_p = RK_coeffs0*Conserved_old[p][i] + RK_coeffs1*Conserved[p][i];
         const double rhoHO_p = rhoi_p + dRhoi;
         if (rhoHO_p < eps) {
            const double rhoLO = rhoi_p + dRhoi_lo;
            theta_p = (fabs(rhoHO_p - rhoLO) > eps) ? max(0.0, (eps - rhoLO)/(rhoHO_p - rhoLO)) : 1.0;
         }
         // Check pP1 side
         double theta_pp1 = 1.0;
         const double rhoi_pp1 = RK_coeffs0*Conserved_old[pP1][i] + RK_coeffs1*Conserved[pP1][i];
         const double rhoHO_pp1 = rhoi_pp1 - dRhoi;
         if (rhoHO_pp1 < eps) {
            const double rhoLO = rhoi_pp1 - dRhoi_lo;
            theta_pp1 = (fabs(rhoHO_pp1 - rhoLO) > eps) ? max(0.0, (eps - rhoLO)/(rhoHO_pp1 - rhoLO)) : 1.0;
         }
         // Take the minimum blending parameter
         theta = min(min(theta_p, theta_pp1), theta);
      }
      // Apply the limiter
      Flux = Flux*theta + (1-theta)*FluxP;
   }
#endif
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingHybridEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingHybridEulerFluxTask<dir>::updateRHSSpan(
                           double *KGSum,
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_e,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<  bool, 3> &shockSensor,
                           const AccessorRO<VecNEq, 3> &Conserved,
#if (nSpec > 1)
                           const AccessorRO<VecNEq, 3> &Conserved_old,
#endif
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &SoS,
                           const AccessorRO<VecNSp, 3> &MassFracs,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<double, 3> &pressure,
#if (nSpec > 1)
                           const double  RK_coeffs0,
                           const double  RK_coeffs1,
                           const double  deltaTime,
#endif
                           const coord_t firstIndex,
                           const coord_t lastIndex,
                           const int x,
                           const int y,
                           const int z,
                           const Rect<3> &Flux_bounds,
                           const Rect<3> &Fluid_bounds,
                           const Mix &mix) {

   VecNEq FluxM; VecNEq FluxP;
   const coord_t size = getSize<dir>(Fluid_bounds);
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p0 = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> p   = warpPeriodic<dir, Minus>(Fluid_bounds, p0, size, offM1(nType[p0]));
      const Point<3> pP1 = p0;
      const Point<3> pM1 = warpPeriodic<dir, Minus>(Fluid_bounds, p,  size, offM1(nType[p]));
      const Point<3> pM2 = warpPeriodic<dir, Minus>(Fluid_bounds, p,  size, offM2(nType[p]));
      // Compute KG summations (... the order is fundamental)
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[3*(nEq+1)],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    pM2, nType[p], size, Fluid_bounds);
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[2*(nEq+1)],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    pM1, nType[p], size, Fluid_bounds);
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[0],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    p, nType[p], size, Fluid_bounds);
      if (shockSensor[pM1] &&
          shockSensor[p  ] &&
          shockSensor[pP1])
         // KG reconstruction
         UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(FluxM, KGSum,
                              Conserved, velocity, pressure,
                              p, nType[p], size, Fluid_bounds);
      else
         // TENO reconstruction
         UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENO_Op<>>(
                                         FluxM,
                                         Conserved,
#if (nSpec > 1)
                                         Conserved_old,
#endif
                                         SoS, rho,
                                         velocity, pressure, MassFracs,
                                         p, nType[p],
#if (nSpec > 1)
                                         RK_coeffs0, RK_coeffs1,
                                         deltaTime*(m_e[p]+m_e[pP1]),
#endif
                                         mix, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      const Point<3> pM1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP1(nType[p]));
      // Shift and update KG summations
      __UNROLL__
      for (int l=0; l<nEq+1; l++) KGSum[5*(nEq+1) + l] = KGSum[4*(nEq+1) + l];
      __UNROLL__
      for (int l=0; l<nEq+1; l++) KGSum[4*(nEq+1) + l] = KGSum[2*(nEq+1) + l];
      __UNROLL__
      for (int l=0; l<nEq+1; l++) KGSum[3*(nEq+1) + l] = KGSum[1*(nEq+1) + l];
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[0],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    p, nType[p], size, Fluid_bounds);

      if (shockSensor[pM1] &&
          shockSensor[p  ] &&
          shockSensor[pP1])
         // KG reconstruction
         UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(FluxP, KGSum,
                              Conserved, velocity, pressure,
                              p, nType[p], size, Fluid_bounds);
      else
         // TENO reconstruction
         UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENO_Op<>>(
                                      FluxP,
                                      Conserved,
#if (nSpec > 1)
                                      Conserved_old,
#endif
                                      SoS, rho,
                                      velocity, pressure, MassFracs,
                                      p, nType[p],
#if (nSpec > 1)
                                      RK_coeffs0, RK_coeffs1,
                                      deltaTime*(m_e[p]+m_e[pP1]),
#endif
                                      mix, size, Fluid_bounds);

      // Update time derivative
      Conserved_t[p] += m_e[p]*(FluxP - FluxM);

      // Store plus flux for next point
      FluxM = FluxP;
   }
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingTENOEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir, class Op>
__CUDA_H__
void UpdateUsingTENOEulerFluxTask<dir, Op>::updateRHSSpan(
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_e,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<VecNEq, 3> &Conserved,
#if (nSpec > 1)
                           const AccessorRO<VecNEq, 3> &Conserved_old,
#endif
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &SoS,
                           const AccessorRO<VecNSp, 3> &MassFracs,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<double, 3> &pressure,
#if (nSpec > 1)
                           const double  RK_coeffs0,
                           const double  RK_coeffs1,
                           const double  deltaTime,
#endif
                           const coord_t firstIndex,
                           const coord_t lastIndex,
                           const int x,
                           const int y,
                           const int z,
                           const Rect<3> &Flux_bounds,
                           const Rect<3> &Fluid_bounds,
                           const Mix &mix) {

   VecNEq FluxM; VecNEq FluxP;
   const coord_t size = getSize<dir>(Fluid_bounds);
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<Op>(
                                      FluxM,
                                      Conserved,
#if (nSpec > 1)
                                      Conserved_old,
#endif
                                      SoS, rho,
                                      velocity, pressure, MassFracs,
                                      pm1, nType[pm1],
#if (nSpec > 1)
                                      RK_coeffs0, RK_coeffs1,
                                      deltaTime*(m_e[pm1]+m_e[p]),
#endif
                                      mix, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
#if (nSpec > 1)
      const Point<3> pP1 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP1(nType[p]));
#endif
      // Update plus flux
      UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<Op>(
                                      FluxP,
                                      Conserved,
#if (nSpec > 1)
                                      Conserved_old,
#endif
                                      SoS, rho,
                                      velocity, pressure, MassFracs,
                                      p, nType[p],
#if (nSpec > 1)
                                      RK_coeffs0, RK_coeffs1,
                                      deltaTime*(m_e[p]+m_e[pP1]),
#endif
                                      mix, size, Fluid_bounds);

      // Update time derivative
      Conserved_t[p] += m_e[p]*(FluxP - FluxM);

      // Store plus flux for next point
      FluxM = FluxP;
   }
};

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingSkewSymmetricEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingSkewSymmetricEulerFluxTask<dir>::updateRHSSpan(
                           double *KGSum,
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_e,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<VecNEq, 3> &Conserved,
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<VecNSp, 3> &MassFracs,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<double, 3> &pressure,
                           const coord_t firstIndex,
                           const coord_t lastIndex,
                           const int x,
                           const int y,
                           const int z,
                           const Rect<3> &Flux_bounds,
                           const Rect<3> &Fluid_bounds) {

   VecNEq FluxM; VecNEq FluxP;
   const coord_t size = getSize<dir>(Fluid_bounds);
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p0 = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> p   = warpPeriodic<dir, Minus>(Fluid_bounds, p0, size, offM1(nType[p0]));
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p,  size, offM1(nType[p]));
      const Point<3> pm2 = warpPeriodic<dir, Minus>(Fluid_bounds, p,  size, offM2(nType[p]));
      // Compute KG summations (... the order is fundamental)
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[3*(nEq+1)],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    pm2, nType[p], size, Fluid_bounds);
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[2*(nEq+1)],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    pm1, nType[p], size, Fluid_bounds);
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[0],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    p, nType[p], size, Fluid_bounds);

      // KG reconstruction
      UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(FluxM, KGSum,
                           Conserved, velocity, pressure,
                           p, nType[p], size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Shift and update KG summations
      __UNROLL__
      for (int l=0; l<nEq+1; l++) KGSum[5*(nEq+1) + l] = KGSum[4*(nEq+1) + l];
      __UNROLL__
      for (int l=0; l<nEq+1; l++) KGSum[4*(nEq+1) + l] = KGSum[2*(nEq+1) + l];
      __UNROLL__
      for (int l=0; l<nEq+1; l++) KGSum[3*(nEq+1) + l] = KGSum[1*(nEq+1) + l];
      UpdateUsingEulerFluxUtils<dir>::ComputeKGSums(&KGSum[0],
                    Conserved, rho, MassFracs,
                    velocity,  pressure,
                    p, nType[p], size, Fluid_bounds);

      // KG reconstruction
      UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(FluxP, KGSum,
                           Conserved, velocity, pressure,
                           p, nType[p], size, Fluid_bounds);

      // Update time derivative
      Conserved_t[p] += m_e[p]*(FluxP - FluxM);

      // Store plus flux for next point
      FluxM = FluxP;
   }
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingDiffusionFluxTask
//-----------------------------------------------------------------------------

template<>
__CUDA_H__
inline Vec3 UpdateUsingDiffusionFluxUtils<Xdir>::GetSigma(
                  const int nType, const double m_s,
                  const AccessorRO<double, 3>         &mu,
                  const AccessorRO<  Vec3, 3>   &velocity,
                  const Vec3 *vGradY,
                  const Vec3 *vGradZ,
                  const Point<3> &p, const Point<3> &pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdX_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdX_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdX_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdY_s = Interp2Staggered(nType, vGradY[0][0], vGradY[1][0]);
   const double dVdY_s = Interp2Staggered(nType, vGradY[0][1], vGradY[1][1]);
   const double dUdZ_s = Interp2Staggered(nType, vGradZ[0][0], vGradZ[1][0]);
   const double dWdZ_s = Interp2Staggered(nType, vGradZ[0][2], vGradZ[1][2]);

   Vec3 sigma;
   sigma[0] = mu_s*(4*dUdX_s - 2*dVdY_s - 2*dWdZ_s)/3;
   sigma[1] = mu_s*(dVdX_s+dUdY_s);
   sigma[2] = mu_s*(dWdX_s+dUdZ_s);
   return sigma;
}

template<>
__CUDA_H__
inline Vec3 UpdateUsingDiffusionFluxUtils<Ydir>::GetSigma(
                  const int nType, const double m_s,
                  const AccessorRO<double, 3>         &mu,
                  const AccessorRO<  Vec3, 3>   &velocity,
                  const Vec3 *vGradX,
                  const Vec3 *vGradZ,
                  const Point<3> &p, const Point<3> &pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdY_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdY_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdY_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdX_s = Interp2Staggered(nType, vGradX[0][0], vGradX[1][0]);
   const double dVdX_s = Interp2Staggered(nType, vGradX[0][1], vGradX[1][1]);
   const double dVdZ_s = Interp2Staggered(nType, vGradZ[0][1], vGradZ[1][1]);
   const double dWdZ_s = Interp2Staggered(nType, vGradZ[0][2], vGradZ[1][2]);

   Vec3 sigma;
   sigma[0] = mu_s*(dUdY_s+dVdX_s);
   sigma[1] = mu_s*(4*dVdY_s - 2*dUdX_s - 2*dWdZ_s)/3;
   sigma[2] = mu_s*(dWdY_s+dVdZ_s);
   return sigma;
}

template<>
__CUDA_H__
inline Vec3 UpdateUsingDiffusionFluxUtils<Zdir>::GetSigma(
                  const int nType, const double m_s,
                  const AccessorRO<double, 3>         &mu,
                  const AccessorRO<  Vec3, 3>   &velocity,
                  const Vec3 *vGradX,
                  const Vec3 *vGradY,
                  const Point<3> &p, const Point<3> &pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdZ_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdZ_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdZ_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdX_s = Interp2Staggered(nType, vGradX[0][0], vGradX[1][0]);
   const double dWdX_s = Interp2Staggered(nType, vGradX[0][2], vGradX[1][2]);
   const double dVdY_s = Interp2Staggered(nType, vGradY[0][1], vGradY[1][1]);
   const double dWdY_s = Interp2Staggered(nType, vGradY[0][2], vGradY[1][2]);

   Vec3 sigma;
   sigma[0] = mu_s*(dUdZ_s+dWdX_s);
   sigma[1] = mu_s*(dVdZ_s+dWdY_s);
   sigma[2] = mu_s*(4*dWdZ_s - 2*dUdX_s - 2*dVdY_s)/3;
   return sigma;
}

template<direction dir>
__CUDA_H__
inline void UpdateUsingDiffusionFluxTask<dir>::GetDiffusionFlux(
                  VecNEq &Flux, const int nType, const double m_s, const Mix &mix,
                  const AccessorRO<double, 3> &rho,
                  const AccessorRO<double, 3> &mu,
                  const AccessorRO<double, 3> &lam,
                  const AccessorRO<VecNSp, 3> &Di,
                  const AccessorRO<double, 3> &temperature,
                  const AccessorRO<  Vec3, 3> &velocity,
                  const AccessorRO<VecNSp, 3> &Xi,
                  const AccessorRO<VecNEq, 3> &rhoYi,
                  const AccessorRO<   int, 3> &nType1,
                  const AccessorRO<   int, 3> &nType2,
                  const AccessorRO<double, 3> &m_d1,
                  const AccessorRO<double, 3> &m_d2,
                  Vec3 *vGrad1,
                  Vec3 *vGrad2,
                  const Point<3> &p,
                  const coord_t size,
                  const Rect<3> &bounds) {

   // access i+1 point (warp around boundaries)
   const Point<3> pp1 = warpPeriodic<dir, Plus>(bounds, p, size, 1);

   // Update traverse gradients for pp1 (those at p must be already computed)
   vGrad1[1] = getDeriv<getT1(dir)>(velocity, pp1, nType1[pp1], m_d1[pp1], bounds);
   vGrad2[1] = getDeriv<getT2(dir)>(velocity, pp1, nType2[pp1], m_d2[pp1], bounds);

   // Mixture properties at the staggered location
   const double rho_s = Interp2Staggered(nType, rho[p], rho[pp1]);
   const double iMixW_s = 1.0/Interp2Staggered(nType, mix.GetMolarWeightFromXi(Xi[p  ]),
                                                      mix.GetMolarWeightFromXi(Xi[pp1]));

   // Primitive and conserved variables at the staggered location
   const double T_s   =  Interp2Staggered(nType, temperature[p], temperature[pp1]);

   // Assemble the fluxes
   double heatFlux = Interp2Staggered(nType, lam[p], lam[pp1])*m_s*(temperature[p] - temperature[pp1]);

   // Partial density Fluxes
   double ViCorr = 0.0;
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double YiVi = Interp2Staggered(nType, Di[p][i],  Di[pp1][i])*
                                           m_s*(Xi[p][i] - Xi[pp1][i])*
                                           mix.GetSpeciesMolarWeight(i)*iMixW_s;
      Flux[i] = -rho_s*YiVi;
      heatFlux += rho_s*YiVi*mix.GetSpeciesEnthalpy(i, T_s);
      ViCorr   += YiVi;
   }

   // Partial density fluxes correction
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double rhoYiViCorr = Interp2Staggered(nType, rhoYi[p][i], rhoYi[pp1][i])*ViCorr;
      Flux[i] += rhoYiViCorr;
      heatFlux -= rhoYiViCorr*mix.GetSpeciesEnthalpy(i, T_s);
   }

   // Momentum Flux
   const Vec3 sigma = UpdateUsingDiffusionFluxUtils<dir>::GetSigma(nType, m_s, mu, velocity, vGrad1, vGrad2, p, pp1);
   __UNROLL__
   for (int i=0; i<3; i++)
      Flux[irU+i] = sigma[i];

   // Energy Flux
   double uSigma = 0.0;
   __UNROLL__
   for (int i=0; i<3; i++)
      uSigma += Interp2Staggered(nType, velocity[p][i], velocity[pp1][i])*sigma[i];
   Flux[irE] = (uSigma - heatFlux);

   // Store traverse gradients for next point
   vGrad1[0] = vGrad1[1];
   vGrad2[0] = vGrad2[1];
}

template<direction dir>
__CUDA_H__
void UpdateUsingDiffusionFluxTask<dir>::updateRHSSpan(
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_s,
                           const AccessorRO<double, 3> &m_d,
                           const AccessorRO<double, 3> &m_d1,
                           const AccessorRO<double, 3> &m_d2,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<   int, 3> &nType1,
                           const AccessorRO<   int, 3> &nType2,
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &mu,
                           const AccessorRO<double, 3> &lam,
                           const AccessorRO<VecNSp, 3> &Di,
                           const AccessorRO<double, 3> &temperature,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<VecNSp, 3> &Xi,
                           const AccessorRO<VecNEq, 3> &rhoYi,
                           const coord_t firstIndex,
                           const coord_t lastIndex,
                           const int x,
                           const int y,
                           const int z,
                           const Rect<3> &Flux_bounds,
                           const Rect<3> &Fluid_bounds,
                           const Mix &mix) {

   const coord_t size = getSize<dir>(Fluid_bounds);
   VecNEq DiffFluxM; VecNEq DiffFluxP;
   Vec3 p_vGrad1[2], p_vGrad2[2];
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      // Compute all the traverse gradients
      p_vGrad1[0] = getDeriv<getT1(dir)>(velocity, pm1, nType1[pm1], m_d1[pm1], Fluid_bounds);
      p_vGrad2[0] = getDeriv<getT2(dir)>(velocity, pm1, nType2[pm1], m_d2[pm1], Fluid_bounds);
      // Actual flux calculation
      GetDiffusionFlux(DiffFluxM, nType[pm1], m_s[pm1], mix,
                       rho, mu, lam, Di,
                       temperature, velocity, Xi, rhoYi,
                       nType1, nType2,
                       m_d1, m_d2,
                       p_vGrad1, p_vGrad2,
                       pm1, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Update plus flux
      GetDiffusionFlux(DiffFluxP, nType[p], m_s[p], mix,
                       rho, mu, lam, Di,
                       temperature, velocity, Xi, rhoYi,
                       nType1, nType2,
                       m_d1, m_d2,
                       p_vGrad1, p_vGrad2,
                       p, size, Fluid_bounds);

      // Update time derivative
      Conserved_t[p] += m_d[p]*(DiffFluxP - DiffFluxM);

      // Store plus flux for next point
      DiffFluxM = DiffFluxP;
   }
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingFluxNSCBCInflowTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCInflowMinusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const   double SoS,
                            const   double rho,
                            const   double T,
                            const     Vec3 &dudt,
                            const   double dTdt,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);

   if (velocity[p][iN] >= SoS) {
      // Supersonic inlet
      __UNROLL__
      for (int l=0; l<nEq; l++)
         RHS[l] = 0.0;
   }
   else {
      // Subsonic inlet (add NSCBC fluxes)
      const Point<3> p_int = (dir == Xdir) ?  Point<3>(p.x+1, p.y  , p.z  ) :
                             (dir == Ydir) ?  Point<3>(p.x  , p.y+1, p.z  ) :
                           /*(dir == Zdir) ?*/Point<3>(p.x  , p.y  , p.z+1);

      // Thermo-chemical quantities
      const double MixW_bnd = mix.GetMolarWeightFromYi(MassFracs[p]);
      const double   Cp_bnd = mix.GetHeatCapacity(T,   MassFracs[p]);

      // characteristic velocity leaving the domain
      const double lambda_1 = velocity[p][iN] - SoS;

      // characteristic velocity entering the domain
      const double lambda   = velocity[p][iN];

      // compute waves amplitudes
      const double dp_dn = getDerivLeftBC(nType, pressure[p]    , pressure[p_int]    , m);
      const double du_dn = getDerivLeftBC(nType, velocity[p][iN], velocity[p_int][iN], m);

      const double L1 = lambda_1*(dp_dn - rho*SoS*du_dn);
      VecNSp LS;
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         LS[s] = lambda*getDerivLeftBC(nType, MassFracs[p][s], MassFracs[p_int][s], m);
      const double LN = L1 - 2*rho*SoS*dudt[iN];
      double L2 = dTdt/T + (LN + L1)/(2*rho*Cp_bnd*T);
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         L2 -= MixW_bnd/mix.GetSpeciesMolarWeight(s)*LS[s];
      L2 *= -rho*SoS*SoS;

      // Compute LODI fluxes
      const double d1 = (0.5*(L1 + LN) - L2)/(SoS*SoS);

      // Update the RHS
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         RHS[s] -= (d1*MassFracs[p][s] + rho*LS[s]);
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCInflowPlusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const   double SoS,
                            const   double rho,
                            const   double T,
                            const     Vec3 &dudt,
                            const   double dTdt,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);

   if (velocity[p][iN] <= -SoS) {
      // Supersonic inlet
      __UNROLL__
      for (int l=0; l<nEq; l++)
         RHS[l] = 0.0;
   }
   else {
      // Subsonic inlet (add NSCBC fluxes)
      const Point<3> p_int = (dir == Xdir) ?  Point<3>(p.x-1, p.y  , p.z  ) :
                             (dir == Ydir) ?  Point<3>(p.x  , p.y-1, p.z  ) :
                           /*(dir == Zdir) ?*/Point<3>(p.x  , p.y  , p.z-1);

      // Thermo-chemical quantities
      const double MixW_bnd = mix.GetMolarWeightFromYi(MassFracs[p]);
      const double   Cp_bnd = mix.GetHeatCapacity(T,   MassFracs[p]);

      // characteristic velocity leaving the domain
      const double lambda_N = velocity[p][iN] + SoS;

      // characteristic velocity entering the domain
      const double lambda   = velocity[p][iN];

      // compute waves amplitudes
      const double dp_dn = getDerivRightBC(nType, pressure[p_int]    , pressure[p]    , m);
      const double du_dn = getDerivRightBC(nType, velocity[p_int][iN], velocity[p][iN], m);

      const double LN = lambda_N*(dp_dn + rho*SoS*du_dn);
      VecNSp LS;
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         LS[s] = lambda*getDerivRightBC(nType, MassFracs[p_int][s], MassFracs[p][s], m);
      const double L1 = LN + 2*rho*SoS*dudt[iN];
      double L2 = dTdt/T + (LN + L1)/(2*rho*Cp_bnd*T);
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         L2 -= MixW_bnd/mix.GetSpeciesMolarWeight(s)*LS[s];
      L2 *= -rho*SoS*SoS;

      // Compute LODI fluxes
      const double d1 = (0.5*(L1 + LN) - L2)/(SoS*SoS);

      // Update the RHS
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         RHS[s] -= (d1*MassFracs[p][s] + rho*LS[s]);
   }
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingFluxNSCBCOutflowTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCOutflowMinusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<   int, 3> &nType_N,
                            const AccessorRO<   int, 3> &nType_T1,
                            const AccessorRO<   int, 3> &nType_T2,
                            const AccessorRO<double, 3> &m_d_N,
                            const AccessorRO<double, 3> &m_d_T1,
                            const AccessorRO<double, 3> &m_d_T2,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<double, 3> &mu,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const   double SoS,
                            const   double T,
                            const   VecNEq &Conserved,
                            const Point<3> &p,
                            const  Rect<3> &bounds,
                            const   double MaxMach,
                            const   double LengthScale,
                            const   double PInf,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);
   const Point<3> p_int = getPIntBC<dir, Minus>(p);

   // BC-normal pressure derivative
   const double dp_dn  = getDerivLeftBC(nType_N[p], pressure[p]     , pressure[p_int]     , m_d_N[p]);
   const double dun_dn = getDerivLeftBC(nType_N[p], velocity[p][iN] , velocity[p_int][iN] , m_d_N[p]);
   const double duT1dn = getDerivLeftBC(nType_N[p], velocity[p][iT1], velocity[p_int][iT1], m_d_N[p]);
   const double duT2dn = getDerivLeftBC(nType_N[p], velocity[p][iT2], velocity[p_int][iT2], m_d_N[p]);

   // Characteristic velocities (use a special velocity for species in case of backflow)
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_s = min(velocity[p][iN], 0.0);
   const double lambda_N = velocity[p][iN] + SoS;

   // compute waves amplitudes
   const double L1 = lambda_1*(dp_dn - rho[p]*SoS*dun_dn);
   const double LM = lambda*(dp_dn - SoS*SoS*getDerivLeftBC(nType_N[p], rho[p], rho[p_int], m_d_N[p]));
   VecNSp LS;
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      LS[s] = lambda_s*getDerivLeftBC(nType_N[p], MassFracs[p][s], MassFracs[p_int][s], m_d_N[p]);
   const double sigma = 0.25;
   /*const*/ double LN;
   if (lambda_N < 0)
      // This point is supersonic
      LN = lambda_N*(dp_dn + rho[p]*SoS*dun_dn);
   else {
      // It is either a subsonic or partially subsonic outlet
      const double K = (MaxMach < 0.99) ? sigma*(1.0-MaxMach*MaxMach)*SoS/LengthScale :
                        sigma*(SoS-(velocity[p][iN]*velocity[p][iN])/SoS)/LengthScale;
      LN = K*(pressure[p] - PInf);
   }

   // Compute LODI fluxes
   const double d1 = (0.5*(L1 + LN) - LM)/(SoS*SoS);
   Vec3 dM;
   dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
   dM[iT1] = lambda*duT1dn;
   dM[iT2] = lambda*duT2dn;
   const double dN = LM/(SoS*SoS);

   // Compute viscous terms
   const Vec3 vGradT1 = getDeriv<getT1(dir)>(velocity, p, nType_T1[p], m_d_T1[p], bounds);
   const Vec3 vGradT2 = getDeriv<getT2(dir)>(velocity, p, nType_T2[p], m_d_T2[p], bounds);
   const double tauNN_bnd = mu[p    ]*(4*dun_dn - 2*vGradT1[iT1] - 2*vGradT2[iT2])/3;
   const double tauNN_int = mu[p_int]*(4*getDeriv<      dir >(velocity, p_int, iN , nType_N[ p_int], m_d_N[ p_int], bounds)
                                     - 2*getDeriv<getT1(dir)>(velocity, p_int, iT1, nType_T1[p_int], m_d_T1[p_int], bounds)
                                     - 2*getDeriv<getT2(dir)>(velocity, p_int, iT2, nType_T2[p_int], m_d_T2[p_int], bounds))/3;
   const double dtau_dn = getDerivLeftBC(nType_N[p], tauNN_bnd, tauNN_int, m_d_N[p]);
   const double viscous_heating = getDerivLeftBC(nType_N[p], velocity[p][iN]*tauNN_bnd, velocity[p_int][iN]*tauNN_int, m_d_N[p]) +
                                  duT1dn*mu[p]*(duT1dn + vGradT1[iN]) +
                                  duT2dn*mu[p]*(duT2dn + vGradT2[iN]);

   // Update the RHS
   const double cpT = mix.GetHeatCapacity(T, MassFracs[p])*T;
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      RHS[s] -= (d1*MassFracs[p][s] + rho[p]*LS[s]);
   RHS[irU+iN ] -= (velocity[p][iN ]*d1 + rho[p]*dM[iN ] - dtau_dn);
   RHS[irU+iT1] -= (velocity[p][iT1]*d1 + rho[p]*dM[iT1]);
   RHS[irU+iT2] -= (velocity[p][iT2]*d1 + rho[p]*dM[iT2]);
   RHS[irE] -= ((Conserved[irE] + pressure[p])*d1/rho[p]
                            + Conserved[irU+iN ]*dM[iN ]
                            + Conserved[irU+iT1]*dM[iT1]
                            + Conserved[irU+iT2]*dM[iT2]
                            + cpT*dN
                            - viscous_heating);

   const double cpT_Wmix = cpT*mix.GetMolarWeightFromYi(MassFracs[p]);
   __UNROLL__
   for (int s=0; s<nSpec; s++) {
      const double dEdYi = mix.GetSpeciesEnthalpy(s, T) - cpT_Wmix/mix.GetSpeciesMolarWeight(s);
      RHS[irE] -= (rho[p]*LS[s]*dEdYi);
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<   int, 3> &nType_N,
                            const AccessorRO<   int, 3> &nType_T1,
                            const AccessorRO<   int, 3> &nType_T2,
                            const AccessorRO<double, 3> &m_d_N,
                            const AccessorRO<double, 3> &m_d_T1,
                            const AccessorRO<double, 3> &m_d_T2,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<double, 3> &mu,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const   double SoS,
                            const   double T,
                            const   VecNEq &Conserved,
                            const Point<3> &p,
                            const  Rect<3> &bounds,
                            const   double MaxMach,
                            const   double LengthScale,
                            const   double PInf,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);
   const Point<3> p_int = getPIntBC<dir, Plus>(p);

   // BC-normal pressure derivative
   const double dp_dn  = getDerivRightBC(nType_N[p], pressure[p_int]     , pressure[p]     , m_d_N[p]);
   const double dun_dn = getDerivRightBC(nType_N[p], velocity[p_int][iN] , velocity[p][iN] , m_d_N[p]);
   const double duT1dn = getDerivRightBC(nType_N[p], velocity[p_int][iT1], velocity[p][iT1], m_d_N[p]);
   const double duT2dn = getDerivRightBC(nType_N[p], velocity[p_int][iT2], velocity[p][iT2], m_d_N[p]);

   // Characteristic velocities (use a special velocity for species in case of backflow)
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_s = max(velocity[p][iN], 0.0);
   const double lambda_N = velocity[p][iN] + SoS;

   // Compute waves amplitudes
   const double sigma = 0.25;
   /*const*/ double L1;
   if (lambda_1 > 0)
      // This point is supersonic
      L1 = lambda_1*(dp_dn - rho[p]*SoS*dun_dn);
   else {
      // It is either a subsonic or partially subsonic outlet
      const double K = (MaxMach < 0.99) ? sigma*(1.0-MaxMach*MaxMach)*SoS/LengthScale :
                        sigma*(SoS-(velocity[p][iN]*velocity[p][iN])/SoS)/LengthScale;
      L1 = K*(pressure[p] - PInf);
   }
   const double LM = lambda*(dp_dn - SoS*SoS*getDerivRightBC(nType_N[p], rho[p_int], rho[p], m_d_N[p]));
   VecNSp LS;
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      LS[s] = lambda_s*getDerivRightBC(nType_N[p], MassFracs[p_int][s], MassFracs[p][s], m_d_N[p]);
   const double LN = lambda_N*(dp_dn + rho[p]*SoS*dun_dn);

   // Compute LODI fluxes
   const double d1 = (0.5*(L1 + LN) - LM)/(SoS*SoS);
   Vec3 dM;
   dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
   dM[iT1] = lambda*duT1dn;
   dM[iT2] = lambda*duT2dn;
   const double dN = LM/(SoS*SoS);

   // Compute viscous terms
   const Vec3 vGradT1 = getDeriv<getT1(dir)>(velocity, p, nType_T1[p], m_d_T1[p], bounds);
   const Vec3 vGradT2 = getDeriv<getT2(dir)>(velocity, p, nType_T2[p], m_d_T2[p], bounds);
   const double tauNN_bnd = mu[p    ]*(4*dun_dn - 2*vGradT1[iT1] - 2*vGradT2[iT2])/3;
   const double tauNN_int = mu[p_int]*(4*getDeriv<      dir >(velocity, p_int, iN , nType_N[ p_int], m_d_N[ p_int], bounds)
                                     - 2*getDeriv<getT1(dir)>(velocity, p_int, iT1, nType_T1[p_int], m_d_T1[p_int], bounds)
                                     - 2*getDeriv<getT2(dir)>(velocity, p_int, iT2, nType_T2[p_int], m_d_T2[p_int], bounds))/3;
   const double dtau_dn = getDerivRightBC(nType_N[p], tauNN_int, tauNN_bnd, m_d_N[p]);
   const double viscous_heating = getDerivRightBC(nType_N[p], velocity[p_int][iN]*tauNN_int, velocity[p][iN]*tauNN_bnd, m_d_N[p]) +
                                  duT1dn*mu[p]*(duT1dn + vGradT1[iN]) +
                                  duT2dn*mu[p]*(duT2dn + vGradT2[iN]);

   // Update the RHS
   const double cpT = mix.GetHeatCapacity(T, MassFracs[p])*T;
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      RHS[s] -= (d1*MassFracs[p][s] + rho[p]*LS[s]);
   RHS[irU+iN ] -= (velocity[p][iN ]*d1 + rho[p]*dM[iN ] - dtau_dn);
   RHS[irU+iT1] -= (velocity[p][iT1]*d1 + rho[p]*dM[iT1]);
   RHS[irU+iT2] -= (velocity[p][iT2]*d1 + rho[p]*dM[iT2]);
   RHS[irE] -= ((Conserved[irE] + pressure[p])*d1/rho[p]
                            + Conserved[irU+iN ]*dM[iN ]
                            + Conserved[irU+iT1]*dM[iT1]
                            + Conserved[irU+iT2]*dM[iT2]
                            + cpT*dN
                            - viscous_heating);
   const double cpT_Wmix = cpT*mix.GetMolarWeightFromYi(MassFracs[p]);
   __UNROLL__
   for (int s=0; s<nSpec; s++) {
      const double dEdYi = mix.GetSpeciesEnthalpy(s, T) - cpT_Wmix/mix.GetSpeciesMolarWeight(s);
      RHS[irE] -= (rho[p]*LS[s]*dEdYi);
   }
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingFluxNSCBCFarFieldTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCFarFieldMinusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const      int &nType,
                            const   double &m_d,
                            const   double &SoS,
                            const   double &temperature,
                            const   VecNEq &Conserved,
                            const   double &TInf,
                            const     Vec3 &vInf,
                            const   VecNSp &XiInf,
                            const   double PInf,
                            const   double MaxMach,
                            const   double LengthScale,
                            const Point<3> &p,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);
   const Point<3> p_int = getPIntBC<dir, Minus>(p);

   // Characteristic velocities (do not extrapolate entropy and species)
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_N = velocity[p][iN] + SoS;

   // BC-normal derivatives
   const double dp_dn  = getDerivLeftBC(nType, pressure[p]    , pressure[p_int]    , m_d);
   const double dun_dn = getDerivLeftBC(nType, velocity[p][iN], velocity[p_int][iN], m_d);

   // Thermodynamic properties
   const double MixW = mix.GetMolarWeightFromYi(MassFracs[p]);
   const double cpT = mix.GetHeatCapacity(temperature, MassFracs[p])*temperature;

   // Check if we are dealing with an inflow
   const double MaInflow = max(lambda/SoS, 0.0);

   constexpr double sigma = 0.25;
   if (MaInflow > 0) {
      // This point is an inflow
      constexpr double coeff = 39;
      const double Kp = sigma*(1 + coeff*MaInflow*MaInflow)*SoS/LengthScale;
      if (MaInflow > 1) {
         // This is a supersonic inflow (weakly impose everything)
         const double rhoTarget = mix.GetRho(PInf, TInf, MixW);
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            RHS[s] += Kp*(rhoTarget*XiInf[s]*mix.GetSpeciesMolarWeight(s)/MixW - Conserved[s]);
         __UNROLL__
         for (int i=0; i<3; i++)
            RHS[irU+i] += Kp*(rhoTarget*vInf[i] - Conserved[irU+i]);
         RHS[irE] += Kp*(rhoTarget*(0.5*vInf.mod2() + mix.GetInternalEnergy(TInf, MassFracs[p])) - Conserved[irE]);
      } else {
         // This is a subsonic inflow
         // Compute waves amplitudes
         const double K = sigma*(coeff + 1)*MaInflow*MaInflow*SoS/LengthScale;
         const double L1 = lambda_1*(dp_dn - rho[p]*SoS*dun_dn);
         const double LN = Kp*(pressure[p] - PInf);
         VecNSp LS;
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            LS[s] = K*(MassFracs[p][s] - XiInf[s]*mix.GetSpeciesMolarWeight(s)/MixW);
         double L2 = K*(TInf/temperature - 1) + (LN + L1)/(2*rho[p]*cpT);
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            L2 -= MixW/mix.GetSpeciesMolarWeight(s)*LS[s];
         L2 *= -rho[p]*SoS*SoS;

         // Compute LODI fluxes
         const double d1 = (0.5*(L1 + LN) - L2)/(SoS*SoS);
         Vec3 dM;
         dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
         dM[iT1] = K*(velocity[p][iT1] - vInf[iT1]);
         dM[iT2] = K*(velocity[p][iT2] - vInf[iT2]);
         const double dN = L2/(SoS*SoS);

         // Update the RHS
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            RHS[s] -= (d1*MassFracs[p][s] + rho[p]*LS[s]);
         RHS[irU+iN ] -= (velocity[p][iN ]*d1 + rho[p]*dM[iN ]);
         RHS[irU+iT1] -= (velocity[p][iT1]*d1 + rho[p]*dM[iT1]);
         RHS[irU+iT2] -= (velocity[p][iT2]*d1 + rho[p]*dM[iT2]);
         RHS[irE] -= ((Conserved[irE] + pressure[p])*d1/rho[p]
                                  + Conserved[irU+iN ]*dM[iN ]
                                  + Conserved[irU+iT1]*dM[iT1]
                                  + Conserved[irU+iT2]*dM[iT2]
                                  + cpT*dN);
         const double cpT_MixW = cpT*MixW;
         __UNROLL__
         for (int s=0; s<nSpec; s++) {
            const double dEdYi = mix.GetSpeciesEnthalpy(s, temperature) - cpT_MixW/mix.GetSpeciesMolarWeight(s);
            RHS[irE] -= (rho[p]*LS[s]*dEdYi);
         }
      }
   } else {
      // This point is an outflow
      // Compute waves amplitudes
      const double L1 = lambda_1*(dp_dn - rho[p]*SoS*dun_dn);
      const double L2 = lambda*(dp_dn - SoS*SoS*getDerivLeftBC(nType, rho[p], rho[p_int], m_d));
      /*const*/ VecNSp LS;
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         LS[s] = lambda*getDerivLeftBC(nType, MassFracs[p][s], MassFracs[p_int][s], m_d);
      /*const*/ double LN;
      if (lambda_N < 0)
         // This point is a supersonic outflow
         LN = lambda_N*(dp_dn + rho[p]*SoS*dun_dn);
      else {
         const double K = (MaxMach < 0.99) ? sigma*(1.0-MaxMach*MaxMach)*SoS/LengthScale :
                                             sigma*(SoS-(velocity[p][iN]*velocity[p][iN])/SoS)/LengthScale;
         LN = K*(pressure[p] - PInf);
      }

      // Compute LODI fluxes
      const double d1 = (0.5*(L1 + LN) - L2)/(SoS*SoS);
      Vec3 dM;
      dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
      dM[iT1] = lambda*getDerivLeftBC(nType, velocity[p][iT1], velocity[p_int][iT1], m_d);
      dM[iT2] = lambda*getDerivLeftBC(nType, velocity[p][iT2], velocity[p_int][iT2], m_d);
      const double dN = L2/(SoS*SoS);

      // Update the RHS
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         RHS[s] -= (d1*MassFracs[p][s] + rho[p]*LS[s]);
      RHS[irU+iN ] -= (velocity[p][iN ]*d1 + rho[p]*dM[iN ]);
      RHS[irU+iT1] -= (velocity[p][iT1]*d1 + rho[p]*dM[iT1]);
      RHS[irU+iT2] -= (velocity[p][iT2]*d1 + rho[p]*dM[iT2]);
      RHS[irE] -= ((Conserved[irE] + pressure[p])*d1/rho[p]
                               + Conserved[irU+iN ]*dM[iN ]
                               + Conserved[irU+iT1]*dM[iT1]
                               + Conserved[irU+iT2]*dM[iT2]
                               + cpT*dN);

      const double cpT_MixW = cpT*MixW;
      __UNROLL__
      for (int s=0; s<nSpec; s++) {
         const double dEdYi = mix.GetSpeciesEnthalpy(s, temperature) - cpT_MixW/mix.GetSpeciesMolarWeight(s);
         RHS[irE] -= (rho[p]*LS[s]*dEdYi);
      }
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCFarFieldPlusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const      int &nType,
                            const   double &m_d,
                            const   double &SoS,
                            const   double &temperature,
                            const   VecNEq &Conserved,
                            const   double &TInf,
                            const     Vec3 &vInf,
                            const   VecNSp &XiInf,
                            const   double PInf,
                            const   double MaxMach,
                            const   double LengthScale,
                            const Point<3> &p,
                            const      Mix &mix) {
   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);
   const Point<3> p_int = getPIntBC<dir, Plus>(p);

   // Characteristic velocities (do not extrapolate entropy and species)
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_N = velocity[p][iN] + SoS;

   // BC-normal pressure derivative
   const double dp_dn  = getDerivRightBC(nType, pressure[p_int]     , pressure[p]     , m_d);
   const double dun_dn = getDerivRightBC(nType, velocity[p_int][iN] , velocity[p][iN] , m_d);

   // Thermodynamic properties
   const double MixW = mix.GetMolarWeightFromYi(MassFracs[p]);
   const double cpT = mix.GetHeatCapacity(temperature, MassFracs[p])*temperature;

   // Check if we are dealing with an inflow
   const double MaInflow = max(-lambda/SoS, 0.0);

   constexpr double sigma = 0.25;
   if (MaInflow > 0) {
      // This point is an inflow
      constexpr double coeff = 39;
      const double Kp = sigma*(1 + coeff*MaInflow*MaInflow)*SoS/LengthScale;
      if (MaInflow > 1) {
         // This is a supersonic inflow (weakly impose everything)
         const double rhoTarget = mix.GetRho(PInf, TInf, MixW);
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            RHS[s] += Kp*(rhoTarget*XiInf[s]*mix.GetSpeciesMolarWeight(s)/MixW - Conserved[s]);
         __UNROLL__
         for (int i=0; i<3; i++)
            RHS[irU+i] += Kp*(rhoTarget*vInf[i] - Conserved[irU+i]);
         RHS[irE] += Kp*(rhoTarget*(0.5*vInf.mod2() + mix.GetInternalEnergy(TInf, MassFracs[p])) - Conserved[irE]);
      } else {
         // This is a subsonic inflow
         // Compute waves amplitudes
         const double K = sigma*(coeff + 1)*MaInflow*MaInflow*SoS/LengthScale;
         const double L1 = Kp*(pressure[p] - PInf);
         const double LN = lambda_N*(dp_dn + rho[p]*SoS*dun_dn);
         VecNSp LS;
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            LS[s] = K*(MassFracs[p][s] - XiInf[s]*mix.GetSpeciesMolarWeight(s)/MixW);
         double L2 = K*(TInf/temperature - 1) + (LN + L1)/(2*rho[p]*cpT);
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            L2 -= MixW/mix.GetSpeciesMolarWeight(s)*LS[s];
         L2 *= -rho[p]*SoS*SoS;

         // Compute LODI fluxes
         const double d1 = (0.5*(L1 + LN) - L2)/(SoS*SoS);
         Vec3 dM;
         dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
         dM[iT1] = K*(velocity[p][iT1] - vInf[iT1]);
         dM[iT2] = K*(velocity[p][iT2] - vInf[iT2]);
         const double dN = L2/(SoS*SoS);

         // Update the RHS
         __UNROLL__
         for (int s=0; s<nSpec; s++)
            RHS[s] -= (d1*MassFracs[p][s] + rho[p]*LS[s]);
         RHS[irU+iN ] -= (velocity[p][iN ]*d1 + rho[p]*dM[iN ]);
         RHS[irU+iT1] -= (velocity[p][iT1]*d1 + rho[p]*dM[iT1]);
         RHS[irU+iT2] -= (velocity[p][iT2]*d1 + rho[p]*dM[iT2]);
         RHS[irE] -= ((Conserved[irE] + pressure[p])*d1/rho[p]
                                  + Conserved[irU+iN ]*dM[iN ]
                                  + Conserved[irU+iT1]*dM[iT1]
                                  + Conserved[irU+iT2]*dM[iT2]
                                  + cpT*dN);
         const double cpT_MixW = cpT*MixW;
         __UNROLL__
         for (int s=0; s<nSpec; s++) {
            const double dEdYi = mix.GetSpeciesEnthalpy(s, temperature) - cpT_MixW/mix.GetSpeciesMolarWeight(s);
            RHS[irE] -= (rho[p]*LS[s]*dEdYi);
         }
      }
   } else {
      // This point is an outflow
      // Compute waves amplitudes
      /*const*/ double L1;
      if (lambda_1 > 0)
         // This point is supersonic outflow
         L1 = lambda_1*(dp_dn - rho[p]*SoS*dun_dn);
      else {
         const double K = (MaxMach < 0.99) ? sigma*(1.0-MaxMach*MaxMach)*SoS/LengthScale :
                                             sigma*(SoS-(velocity[p][iN]*velocity[p][iN])/SoS)/LengthScale;
         L1 = K*(pressure[p] - PInf);
      }
      const double L2 = lambda*(dp_dn - SoS*SoS*getDerivRightBC(nType, rho[p_int], rho[p], m_d));
      /*const*/ VecNSp LS;
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         LS[s] = lambda*getDerivRightBC(nType, MassFracs[p_int][s], MassFracs[p][s], m_d);
      const double LN = lambda_N*(dp_dn + rho[p]*SoS*dun_dn);

      // Compute LODI fluxes
      const double d1 = (0.5*(L1 + LN) - L2)/(SoS*SoS);
      Vec3 dM;
      dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
      dM[iT1] = lambda*getDerivRightBC(nType, velocity[p_int][iT1], velocity[p][iT1], m_d);
      dM[iT2] = lambda*getDerivRightBC(nType, velocity[p_int][iT2], velocity[p][iT2], m_d);
      const double dN = L2/(SoS*SoS);

      // Update the RHS
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         RHS[s] -= (d1*MassFracs[p][s] + rho[p]*LS[s]);
      RHS[irU+iN ] -= (velocity[p][iN ]*d1 + rho[p]*dM[iN ]);
      RHS[irU+iT1] -= (velocity[p][iT1]*d1 + rho[p]*dM[iT1]);
      RHS[irU+iT2] -= (velocity[p][iT2]*d1 + rho[p]*dM[iT2]);
      RHS[irE] -= ((Conserved[irE] + pressure[p])*d1/rho[p]
                               + Conserved[irU+iN ]*dM[iN ]
                               + Conserved[irU+iT1]*dM[iT1]
                               + Conserved[irU+iT2]*dM[iT2]
                               + cpT*dN);
      const double cpT_MixW = cpT*MixW;
      __UNROLL__
      for (int s=0; s<nSpec; s++) {
         const double dEdYi = mix.GetSpeciesEnthalpy(s, temperature) - cpT_MixW/mix.GetSpeciesMolarWeight(s);
         RHS[irE] -= (rho[p]*LS[s]*dEdYi);
      }
   }
}

//-----------------------------------------------------------------------------
// TASKS THAT UPDATE THE RHS USING TURBULENT FORCING
//-----------------------------------------------------------------------------

__CUDA_H__
double CalculateAveragePDTask::CalculatePressureDilatation(const AccessorRO<   int, 3> &nType_x,
                                                           const AccessorRO<   int, 3> &nType_y,
                                                           const AccessorRO<   int, 3> &nType_z,
                                                           const AccessorRO<double, 3> &dcsi_d,
                                                           const AccessorRO<double, 3> &deta_d,
                                                           const AccessorRO<double, 3> &dzet_d,
                                                           const AccessorRO<double, 3> &pressure,
                                                           const AccessorRO<  Vec3, 3> &velocity,
                                                           const Point<3> &p,
                                                           const Rect<3>  &Fluid_bounds) {
   const double divU = getDeriv<Xdir>(velocity, p, 0, nType_x[p], dcsi_d[p], Fluid_bounds) +
                       getDeriv<Ydir>(velocity, p, 1, nType_y[p], deta_d[p], Fluid_bounds) +
                       getDeriv<Zdir>(velocity, p, 2, nType_z[p], dzet_d[p], Fluid_bounds);
   return divU*pressure[p];
};

template<direction dir>
__CUDA_H__
inline double AddDissipationTask<dir>::GetDiffusionFlux(
                  const int nType, const double m_s, const Mix &mix,
                  const AccessorRO<double, 3> &mu,
                  const AccessorRO<  Vec3, 3> &velocity,
                  const AccessorRO<   int, 3> &nType1,
                  const AccessorRO<   int, 3> &nType2,
                  const AccessorRO<double, 3> &m_d1,
                  const AccessorRO<double, 3> &m_d2,
                  Vec3 *vGrad1,
                  Vec3 *vGrad2,
                  const Point<3> &p,
                  const coord_t size,
                  const Rect<3> &bounds) {

   // access i+1 point (warp around boundaries)
   const Point<3> pp1 = warpPeriodic<dir, Plus>(bounds, p, size, 1);

   // Update traverse gradients for pp1 (those at p must be already computed)
   vGrad1[1] = getDeriv<getT1(dir)>(velocity, pp1, nType1[pp1], m_d1[pp1], bounds);
   vGrad2[1] = getDeriv<getT2(dir)>(velocity, pp1, nType2[pp1], m_d2[pp1], bounds);

   // Momentum Flux
   const Vec3 sigma = UpdateUsingDiffusionFluxUtils<dir>::GetSigma(nType, m_s, mu, velocity, vGrad1, vGrad2, p, pp1);

   // Energy Flux
   double uSigma = 0.0;
   __UNROLL__
   for (int i=0; i<3; i++)
      uSigma += Interp2Staggered(nType, velocity[p][i], velocity[pp1][i])*sigma[i];

   // Store traverse gradients for next point
   vGrad1[0] = vGrad1[1];
   vGrad2[0] = vGrad2[1];

   return uSigma;
}

template<direction dir>
__CUDA_H__
double AddDissipationTask<dir>::AddSpan(
                           const AccessorRO<double, 3> &m_s,
                           const AccessorRO<double, 3> &m_d1,
                           const AccessorRO<double, 3> &m_d2,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<   int, 3> &nType1,
                           const AccessorRO<   int, 3> &nType2,
                           const AccessorRO<double, 3> &mu,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const coord_t firstIndex,
                           const coord_t lastIndex,
                           const int x,
                           const int y,
                           const int z,
                           const Rect<3> &Flux_bounds,
                           const Rect<3> &Fluid_bounds,
                           const Mix &mix) {

   const coord_t size = getSize<dir>(Fluid_bounds);
   double FluxM; double FluxP;
   double acc = 0.0;
   Vec3 p_vGrad1[2], p_vGrad2[2];
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      // Compute all the traverse gradients
      p_vGrad1[0] = getDeriv<getT1(dir)>(velocity, pm1, nType1[pm1], m_d1[pm1], Fluid_bounds);
      p_vGrad2[0] = getDeriv<getT2(dir)>(velocity, pm1, nType2[pm1], m_d2[pm1], Fluid_bounds);
      // Actual flux calculation
      FluxM = GetDiffusionFlux(nType[pm1], m_s[pm1], mix,
                       mu, velocity,
                       nType1, nType2,
                       m_d1, m_d2,
                       p_vGrad1, p_vGrad2,
                       pm1, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Update plus flux
      FluxP = GetDiffusionFlux(nType[p], m_s[p], mix,
                       mu, velocity,
                       nType1, nType2,
                       m_d1, m_d2,
                       p_vGrad1, p_vGrad2,
                       p, size, Fluid_bounds);

      // Update time derivative
      acc += (FluxP - FluxM)/(m_d1[p]*m_d2[p]);

      // Store plus flux for next point
      FluxM = FluxP;
   }
   return acc;
}

