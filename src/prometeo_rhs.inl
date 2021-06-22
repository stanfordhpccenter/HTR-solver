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
                           const double          TL, const double          TR,
                           const double   pressureL, const double   pressureR,
                           const   Vec3  &velocityL, const   Vec3  &velocityR,
                           const double        rhoL, const double        rhoR) {

   // Compute quantities on the left (L) and right (R) states
   const double MixWL = mix.GetMolarWeightFromYi(YiL);
   const double MixWR = mix.GetMolarWeightFromYi(YiR);

   const double gammaL = mix.GetGamma(TL, MixWL, YiL);
   const double gammaR = mix.GetGamma(TR, MixWR, YiR);

   /*const*/ VecNSp dpdrhoiL; mix.Getdpdrhoi(dpdrhoiL, gammaL, TL, YiL);
   /*const*/ VecNSp dpdrhoiR; mix.Getdpdrhoi(dpdrhoiR, gammaR, TR, YiR);

   const double dpdeL = mix.Getdpde(rhoL, gammaL);
   const double dpdeR = mix.Getdpde(rhoR, gammaR);

   const double TotalEnergyL = ConservedL[irE]/rhoL;
   const double TotalEnergyR = ConservedR[irE]/rhoR;

   const double TotalEnthalpyL = TotalEnergyL + pressureL/rhoL;
   const double TotalEnthalpyR = TotalEnergyR + pressureR/rhoR;

   // Compute Roe averaged state
   const double RoeFactorL = sqrt(rhoL)/(sqrt(rhoL) + sqrt(rhoR));
   const double RoeFactorR = sqrt(rhoR)/(sqrt(rhoL) + sqrt(rhoR));

   avgs.rho = sqrt(rhoL*rhoR);

   __UNROLL__
   for (int i=0; i<nSpec; i++)
      avgs.Yi[i] = YiL[i]*RoeFactorL + YiR[i]*RoeFactorR;

   __UNROLL__
   for (int i=0; i<3; i++)
      avgs.velocity[i] = velocityL[i]*RoeFactorL + velocityR[i]*RoeFactorR;

   avgs.H =  TotalEnthalpyL*RoeFactorL +  TotalEnthalpyR*RoeFactorR;
   avgs.E =    TotalEnergyL*RoeFactorL +    TotalEnergyR*RoeFactorR;

   double dpdrhoiRoe[nSpec];
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      dpdrhoiRoe[i] = dpdrhoiL[i]*RoeFactorL + dpdrhoiR[i]*RoeFactorR;
   const double dpdeRoe =   dpdeL*RoeFactorL +       dpdeR*RoeFactorR;

   // correct the pressure derivatives in order to satisfy the pressure jump condition
   // using the procedure in Shuen, Liou and Leer (1990)
   const double dp = pressureR - pressureL;
   const double de = TotalEnergyR - 0.5*velocityR.mod2()
                   -(TotalEnergyL - 0.5*velocityL.mod2());
   VecNSp drhoi;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      drhoi[i] = ConservedR[i] - ConservedL[i];

   // find the error in the pressure jump due to Roe averages
   double dpError = dp - de*dpdeRoe;
   double fact = de*dpdeRoe; fact *= fact;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      dpError -= drhoi[i]*dpdrhoiRoe[i];
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double dpi = drhoi[i]*dpdrhoiRoe[i];
      fact += dpi*dpi;
   }

   // correct pressure derivatives
   // this threshold should not be affect the solution since fact is zero when all the jumps are zero
   fact = dpError/max(fact, 1e-6);
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      avgs.dpdrhoi[i] = dpdrhoiRoe[i]*(1.0 + dpdrhoiRoe[i]*drhoi[i]*fact);
   avgs.dpde = dpdeRoe*(1.0 + dpdeRoe*de*fact);

   // compute the Roe averaged speed of sound
   double PovRhoRoe = avgs.H - avgs.E;
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
   const double Coeff = (avgs.E - avgs.velocity.mod2())*avgs.dpde/avgs.rho;
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
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K(row,  i+1) = avgs.E - avgs.rho*avgs.dpdrhoi[i]*dedp;
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

   // Initialize r
   r.init(0);

   // Compute constants
   const double iaRoe  = 1.0/avgs.a;
   const double iaRoe2 = 1.0/avgs.a2;
   const double Coeff = (avgs.E - avgs.velocity.mod2())*avgs.dpde/avgs.rho;
   VecNSp b;
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      b[i] = (Coeff - avgs.dpdrhoi[i])*iaRoe2;
   const double d = avgs.dpde/(avgs.rho*avgs.a2);
   const Vec3 c(avgs.velocity*d);

   // First row
   {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         r[0] -= 0.5*(b[j] - avgs.velocity[iN]*iaRoe)*q[j];
      r[0] -= (0.5*c[iN] + 0.5*iaRoe)*q[nSpec+iN];
      r[0] -= 0.5*c[iT1]*q[nSpec+iT1];
      r[0] -= 0.5*c[iT2]*q[nSpec+iT2];
      r[0] += 0.5*d*q[nSpec+3];
   }

   // From 1 to nSpec
   __UNROLL__
   for (int i=1; i<nSpec+1; i++) {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         r[i] += avgs.Yi[i-1]*b[j]*q[j];
      r[i] += q[i-1];
      __UNROLL__
      for (int j=0; j<3; j++)
         r[i] += avgs.Yi[i-1]*c[j]*q[nSpec+j];
      r[i] += -avgs.Yi[i-1]*d*q[nSpec+3];
   }

   // nSpec + 1
   {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         r[nSpec + 1] -= avgs.velocity[iT1]*q[j];
      //r[nSpec+1] += 0.0*q[nSpec+iN];
      r[nSpec+1] += q[nSpec+iT1];
      //r[nSpec+1] += 0.0*q[nSpec+iT2];
      //r[nSpec+1] += 0.0*q[nSpec+3  ];
   }

   // nSpec + 2
   {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         r[nSpec + 2] -= avgs.velocity[iT2]*q[j];
      //r[nSpec+2] += 0.0*q[nSpec+iN ];
      //r[nSpec+2] += 0.0*q[nSpec+iT1];
      r[nSpec+2] += q[nSpec+iT2];
      //r[nSpec+2] += 0.0*q[nSpec+3  ];
   }

   // nSpec + 3
   {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         r[nSpec+3] -= 0.5*(b[j] + avgs.velocity[iN]*iaRoe)*q[j];
      r[nSpec+3] -= (0.5*c[iN] - 0.5*iaRoe)*q[nSpec+iN];
      r[nSpec+3] -= 0.5*c[iT1]*q[nSpec+iT1];
      r[nSpec+3] -= 0.5*c[iT2]*q[nSpec+iT2];
      r[nSpec+3] += 0.5*d*q[nSpec+3  ];
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
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         r[nSpec+iN] += avgs.velocity[iN]*q[i+1];
      //r[nSpec+iN] += 0.0*q[nSpec+1];
      //r[nSpec+iN] += 0.0*q[nSpec+2];
      r[nSpec+iN] += (avgs.velocity[iN] + avgs.a)*q[nSpec+3];
   }

   // nSpec+iT1 row
   {
      r[nSpec+iT1] += avgs.velocity[iT1]*q[0];
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         r[nSpec+iT1] += avgs.velocity[iT1]*q[i+1];
      r[nSpec+iT1] += q[nSpec+1];
      //r[nSpec+iT1] += 0.0*q[nSpec+2];
      r[nSpec+iT1] += avgs.velocity[iT1]*q[nSpec+3];
   }

   // nSpec+iT2 row
   {
      r[nSpec+iT2] += avgs.velocity[iT2]*q[0];
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         r[nSpec+iT2] += avgs.velocity[iT2]*q[i+1];
      //r[nSpec+iT2] += 0.0*q[nSpec+1];
      r[nSpec+iT2] += q[nSpec+2];
      r[nSpec+iT2] += avgs.velocity[iT2]*q[nSpec+3];
   }

   // nSpec+3 row
   {
      r[nSpec+3] += (avgs.H - avgs.velocity[iN]*avgs.a)*q[0];
      const double dedp = 1.0/avgs.dpde;
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         r[nSpec+3] += (avgs.E - avgs.rho*avgs.dpdrhoi[i]*dedp)*q[i+1];
      r[nSpec+3] += avgs.velocity[iT1]*q[nSpec+1];
      r[nSpec+3] += avgs.velocity[iT2]*q[nSpec+2];
      r[nSpec+3] += (avgs.H + avgs.velocity[iN]*avgs.a)*q[nSpec+3];
   }
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
                            const AccessorRO<double, 3> &SoS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &temperature,
                            const Point<3> &p,
                            const int nType,
                            const Mix &mix,
                            const coord_t dsize,
                            const Rect<3> &bounds) {

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
               temperature[p], temperature[pP1],
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
   VecNEq FPlus;
   for (int i=0; i<nEq; i++)
      FPlus[i] = Op::reconstructPlus(FluxPM2[i], FluxPM1[i], FluxP  [i],
                                   FluxPP1[i], FluxPP2[i], FluxPP3[i], nType);

   VecNEq FMinus;
   for (int i=0; i<nEq; i++)
      FMinus[i] = Op::reconstructMinus(FluxMM2[i], FluxMM1[i], FluxM  [i],
                                     FluxMP1[i], FluxMP2[i], FluxMP3[i], nType);

   VecNEq F;
   __UNROLL__
   for (int i=0; i<nEq; i++)
      F[i] = -0.5*(FPlus[i] + FMinus[i]);

   // Go back to the physical space
   projectFromCharacteristicSpace(Flux, F, RoeAvgs);
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
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &SoS,
                           const AccessorRO<VecNSp, 3> &MassFracs,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<double, 3> &pressure,
                           const AccessorRO<double, 3> &temperature,
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
      const Point<3> pP1 = warpPeriodic<dir, Plus >(Fluid_bounds, p,  size, offP1(nType[p]));
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
         UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENO_Op>(
                                         FluxM,
                                         Conserved, SoS, rho, velocity,
                                         pressure, MassFracs, temperature,
                                         p, nType[p], mix, size, Fluid_bounds);
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
         UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENO_Op>(
                                      FluxP,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      p, nType[p], mix, size, Fluid_bounds);


      // Update time derivative
      Conserved_t[p] += m_e[p]*(FluxP - FluxM);

      // Store plus flux for next point
      FluxM = FluxP;
   }
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingTENOAEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingTENOAEulerFluxTask<dir>::updateRHSSpan(
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_e,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<VecNEq, 3> &Conserved,
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &SoS,
                           const AccessorRO<VecNSp, 3> &MassFracs,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<double, 3> &pressure,
                           const AccessorRO<double, 3> &temperature,
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
      UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENOA_Op>(
                                      FluxM,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      pm1, nType[pm1], mix, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Update plus flux
      UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENOA_Op>(
                                      FluxP,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      p, nType[p], mix, size, Fluid_bounds);

      // Update time derivative
      Conserved_t[p] += m_e[p]*(FluxP - FluxM);

      // Store plus flux for next point
      FluxM = FluxP;
   }
};

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingTENOLADEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingTENOLADEulerFluxTask<dir>::updateRHSSpan(
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_e,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<VecNEq, 3> &Conserved,
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &SoS,
                           const AccessorRO<VecNSp, 3> &MassFracs,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<double, 3> &pressure,
                           const AccessorRO<double, 3> &temperature,
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
      UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENOLAD_Op>(
                                      FluxM,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      pm1, nType[pm1], mix, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Update plus flux
      UpdateUsingEulerFluxUtils<dir>::template FluxReconstruction<TENOLAD_Op>(
                                      FluxP,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      p, nType[p], mix, size, Fluid_bounds);

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
inline Vec3 UpdateUsingDiffusionFluxTask<Xdir>::GetSigma(
                  const int nType, const double m_s,
                  const AccessorRO<double, 3>         &mu,
                  const AccessorRO<  Vec3, 3>   &velocity,
                  const AccessorRO<  Vec3, 3>     &vGradY,
                  const AccessorRO<  Vec3, 3>     &vGradZ,
                  const Point<3> &p, const Point<3> &pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdX_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdX_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdX_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdY_s = Interp2Staggered(nType, vGradY[p][0], vGradY[pp1][0]);
   const double dVdY_s = Interp2Staggered(nType, vGradY[p][1], vGradY[pp1][1]);
   const double dUdZ_s = Interp2Staggered(nType, vGradZ[p][0], vGradZ[pp1][0]);
   const double dWdZ_s = Interp2Staggered(nType, vGradZ[p][2], vGradZ[pp1][2]);

   Vec3 sigma;
   sigma[0] = mu_s*(4*dUdX_s - 2*dVdY_s - 2*dWdZ_s)/3;
   sigma[1] = mu_s*(dVdX_s+dUdY_s);
   sigma[2] = mu_s*(dWdX_s+dUdZ_s);
   return sigma;
}

template<>
__CUDA_H__
inline Vec3 UpdateUsingDiffusionFluxTask<Ydir>::GetSigma(
                  const int nType, const double m_s,
                  const AccessorRO<double, 3>         &mu,
                  const AccessorRO<  Vec3, 3>   &velocity,
                  const AccessorRO<  Vec3, 3>     &vGradX,
                  const AccessorRO<  Vec3, 3>     &vGradZ,
                  const Point<3> &p, const Point<3> &pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdY_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdY_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdY_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdX_s = Interp2Staggered(nType, vGradX[p][0], vGradX[pp1][0]);
   const double dVdX_s = Interp2Staggered(nType, vGradX[p][1], vGradX[pp1][1]);
   const double dVdZ_s = Interp2Staggered(nType, vGradZ[p][1], vGradZ[pp1][1]);
   const double dWdZ_s = Interp2Staggered(nType, vGradZ[p][2], vGradZ[pp1][2]);

   Vec3 sigma;
   sigma[0] = mu_s*(dUdY_s+dVdX_s);
   sigma[1] = mu_s*(4*dVdY_s - 2*dUdX_s - 2*dWdZ_s)/3;
   sigma[2] = mu_s*(dWdY_s+dVdZ_s);
   return sigma;
}

template<>
__CUDA_H__
inline Vec3 UpdateUsingDiffusionFluxTask<Zdir>::GetSigma(
                  const int nType, const double m_s,
                  const AccessorRO<double, 3>         &mu,
                  const AccessorRO<  Vec3, 3>   &velocity,
                  const AccessorRO<  Vec3, 3>     &vGradX,
                  const AccessorRO<  Vec3, 3>     &vGradY,
                  const Point<3> &p, const Point<3> &pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdZ_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdZ_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdZ_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdX_s = Interp2Staggered(nType, vGradX[p][0], vGradX[pp1][0]);
   const double dWdX_s = Interp2Staggered(nType, vGradX[p][2], vGradX[pp1][2]);
   const double dVdY_s = Interp2Staggered(nType, vGradY[p][1], vGradY[pp1][1]);
   const double dWdY_s = Interp2Staggered(nType, vGradY[p][2], vGradY[pp1][2]);

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
                  const AccessorRO<double, 3>         &rho,
                  const AccessorRO<double, 3>          &mu,
                  const AccessorRO<double, 3>         &lam,
                  const AccessorRO<VecNSp, 3>          &Di,
                  const AccessorRO<double, 3> &temperature,
                  const AccessorRO<  Vec3, 3>    &velocity,
                  const AccessorRO<VecNSp, 3>          &Xi,
                  const AccessorRO<VecNEq, 3>       &rhoYi,
                  const AccessorRO<  Vec3, 3>      &vGradY,
                  const AccessorRO<  Vec3, 3>      &vGradZ,
                  const Point<3> &p,
                  const coord_t size,
                  const Rect<3> &bounds) {

   // access i+1 point (warp around boundaries)
   const Point<3> pp1 = warpPeriodic<dir, Plus>(bounds, p, size, 1);

   // Mixture properties at the staggered location
   const double rho_s = Interp2Staggered(nType, rho[p], rho[pp1]);
   const double iMixW_s = 1.0/Interp2Staggered(nType, mix.GetMolarWeightFromXi(Xi[p  ]),
                                                      mix.GetMolarWeightFromXi(Xi[pp1]));

   // Primitive and conserved variables at the staggered location
   const double T_s   =  Interp2Staggered(nType, temperature[p], temperature[pp1]);

   // Assemble the fluxes
   double heatFlux = Interp2Staggered(nType, lam[p], lam[pp1])*m_s*(temperature[p] - temperature[pp1]);

   // Species diffusion velocity
   VecNSp YiVi;
   double ViCorr = 0.0;
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      YiVi[i] = Interp2Staggered(nType, Di[p][i],  Di[pp1][i])*
                                   m_s*(Xi[p][i] - Xi[pp1][i])*
                                   mix.GetSpeciesMolarWeight(i)*iMixW_s;
      ViCorr += YiVi[i];
   }

   // Partial Densities Fluxes
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double rhoYiVi = rho_s*YiVi[i] - ViCorr*Interp2Staggered(nType, rhoYi[p][i], rhoYi[pp1][i]);
      Flux[i] = -rhoYiVi;
      heatFlux += rhoYiVi*mix.GetSpeciesEnthalpy(i, T_s);
   }

   // Momentum Flux
   const Vec3 sigma = GetSigma(nType, m_s, mu, velocity, vGradY, vGradZ, p, pp1);
   __UNROLL__
   for (int i=0; i<3; i++)
      Flux[irU+i] = sigma[i];

   // Energy Flux
   double uSigma = 0.0;
   __UNROLL__
   for (int i=0; i<3; i++)
      uSigma += Interp2Staggered(nType, velocity[p][i], velocity[pp1][i])*sigma[i];
   Flux[irE] = (uSigma - heatFlux);
}

template<direction dir>
__CUDA_H__
void UpdateUsingDiffusionFluxTask<dir>::updateRHSSpan(
                           const AccessorRW<VecNEq, 3> &Conserved_t,
                           const AccessorRO<double, 3> &m_s,
                           const AccessorRO<double, 3> &m_d,
                           const AccessorRO<   int, 3> &nType,
                           const AccessorRO<double, 3> &rho,
                           const AccessorRO<double, 3> &mu,
                           const AccessorRO<double, 3> &lam,
                           const AccessorRO<VecNSp, 3> &Di,
                           const AccessorRO<double, 3> &temperature,
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<VecNSp, 3> &Xi,
                           const AccessorRO<VecNEq, 3> &rhoYi,
                           const AccessorRO<  Vec3, 3> &vGrad1,
                           const AccessorRO<  Vec3, 3> &vGrad2,
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
   // Compute flux of first minus inter-cell location
   {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, firstIndex, x, y, z);
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      GetDiffusionFlux(DiffFluxM, nType[pm1], m_s[pm1], mix,
                       rho, mu, lam, Di,
                       temperature, velocity, Xi,
                       rhoYi, vGrad1, vGrad2,
                       pm1, size, Fluid_bounds);
   }
   // Loop across my section of the span
   for (coord_t i = firstIndex; i < lastIndex; i++) {
      const Point<3> p = GetPointInSpan<dir>(Flux_bounds, i, x, y, z);
      // Update plus flux
      GetDiffusionFlux(DiffFluxP, nType[p], m_s[p], mix,
                       rho, mu, lam, Di,
                       temperature, velocity, Xi,
                       rhoYi, vGrad1, vGrad2,
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
                            const AccessorRO<double, 3>  &pressure,
                            const   double SoS,
                            const   double rho,
                            const   double T,
                            const     Vec3 &velocity,
                            const     Vec3 &vGrad,
                            const     Vec3 &dudt,
                            const   double dTdt,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);

   if (velocity[iN] >= SoS) {
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
      const double lambda_1 = velocity[iN] - SoS;

      // characteristic velocity entering the domain
      const double lambda   = velocity[iN];

      // compute waves amplitudes
      const double dp_dn = getDerivLeftBC(nType, pressure[p], pressure[p_int], m);
      const double du_dn = vGrad[iN];

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
                            const   double SoS,
                            const   double rho,
                            const   double T,
                            const     Vec3 &velocity,
                            const     Vec3 &vGrad,
                            const     Vec3 &dudt,
                            const   double dTdt,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);

   if (velocity[iN] <= -SoS) {
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
      const double lambda_N = velocity[iN] + SoS;

      // characteristic velocity entering the domain
      const double lambda   = velocity[iN];

      // compute waves amplitudes
      const double dp_dn = getDerivRightBC(nType, pressure[p_int], pressure[p], m);
      const double du_dn = vGrad[iN];

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
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<double, 3> &mu,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const AccessorRO<  Vec3, 3> &vGradN,
                            const AccessorRO<  Vec3, 3> &vGradT1,
                            const AccessorRO<  Vec3, 3> &vGradT2,
                            const   double SoS,
                            const   double T,
                            const   VecNEq &Conserved,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
                            const   double MaxMach,
                            const   double LengthScale,
                            const   double PInf,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);
   const Point<3> p_int = getPIntBC<dir, Minus>(p);

   // BC-normal pressure derivative
   const double dp_dn = getDerivLeftBC(nType, pressure[p], pressure[p_int], m);

   // Characteristic velocities
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_N = velocity[p][iN] + SoS;

   // compute waves amplitudes
   const double L1 = lambda_1*(dp_dn - rho[p]*SoS*vGradN[p][iN]);
   const double LM = lambda*(dp_dn - SoS*SoS*getDerivLeftBC(nType, rho[p], rho[p_int], m));
   VecNSp LS;
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      LS[s] = lambda*getDerivLeftBC(nType, MassFracs[p][s], MassFracs[p_int][s], m);
   const double sigma = 0.25;
   /*const*/ double LN;
   if (lambda_N < 0)
      // This point is supersonic
      LN = lambda_N*(dp_dn + rho[p]*SoS*vGradN[p][iN]);
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
   dM[iT1] = lambda*vGradN[p][iT1];
   dM[iT2] = lambda*vGradN[p][iT2];
   const double dN = LM/(SoS*SoS);

   // Compute viscous terms
   const double tauNN_bnd = mu[p    ]*(4*vGradN[p    ][iN] - 2*vGradT1[p    ][iT1] - 2*vGradT2[p    ][iT2])/3;
   const double tauNN_int = mu[p_int]*(4*vGradN[p_int][iN] - 2*vGradT1[p_int][iT1] - 2*vGradT2[p_int][iT2])/3;
   const double dtau_dn = getDerivLeftBC(nType, tauNN_bnd, tauNN_int, m);
   const double viscous_heating = getDerivLeftBC(nType, velocity[p][iN]*tauNN_bnd, velocity[p_int][iN]*tauNN_int, m) +
                                  vGradN[p][iT1]*mu[p]*(vGradN[p][iT1] + vGradT1[p][iN]) +
                                  vGradN[p][iT2]*mu[p]*(vGradN[p][iT2] + vGradT2[p][iN]);

   // Update the RHS
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
                            + mix.GetHeatCapacity(T, MassFracs[p])*T*dN
                            - viscous_heating);
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      RHS[irE] -= (rho[p]*mix.GetSpecificInternalEnergy(s, T)*LS[s]);
}

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<double, 3> &mu,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const AccessorRO<  Vec3, 3> &vGradN,
                            const AccessorRO<  Vec3, 3> &vGradT1,
                            const AccessorRO<  Vec3, 3> &vGradT2,
                            const   double SoS,
                            const   double T,
                            const   VecNEq &Conserved,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
                            const   double MaxMach,
                            const   double LengthScale,
                            const   double PInf,
                            const      Mix &mix) {

   constexpr int iN = normalIndex(dir);
   constexpr int iT1 = tangential1Index(dir);
   constexpr int iT2 = tangential2Index(dir);
   const Point<3> p_int = getPIntBC<dir, Plus>(p);

   // BC-normal pressure derivative
   const double dp_dn = getDerivRightBC(nType, pressure[p_int], pressure[p], m);

   // Characteristic velocities
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_N = velocity[p][iN] + SoS;

   // Compute waves amplitudes
   const double sigma = 0.25;
   /*const*/ double L1;
   if (lambda_1 > 0)
      // This point is supersonic
      L1 = lambda_1*(dp_dn - rho[p]*SoS*vGradN[p][iN]);
   else {
      // It is either a subsonic or partially subsonic outlet
      const double K = (MaxMach < 0.99) ? sigma*(1.0-MaxMach*MaxMach)*SoS/LengthScale :
                        sigma*(SoS-(velocity[p][iN]*velocity[p][iN])/SoS)/LengthScale;
      L1 = K*(pressure[p] - PInf);
   }
   const double LM = lambda*(dp_dn - SoS*SoS*getDerivRightBC(nType, rho[p_int], rho[p], m));
   VecNSp LS;
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      LS[s] = lambda*getDerivRightBC(nType, MassFracs[p_int][s], MassFracs[p][s], m);
   const double LN = lambda_N*(dp_dn + rho[p]*SoS*vGradN[p][iN]);

   // Compute LODI fluxes
   const double d1 = (0.5*(L1 + LN) - LM)/(SoS*SoS);
   Vec3 dM;
   dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
   dM[iT1] = lambda*vGradN[p][iT1];
   dM[iT2] = lambda*vGradN[p][iT2];
   const double dN = LM/(SoS*SoS);

   // Compute viscous terms
   const double tauNN_bnd = mu[p    ]*(4*vGradN[p    ][iN] - 2*vGradT1[p    ][iT1] - 2*vGradT2[p    ][iT2])/3;
   const double tauNN_int = mu[p_int]*(4*vGradN[p_int][iN] - 2*vGradT1[p_int][iT1] - 2*vGradT2[p_int][iT2])/3;
   const double dtau_dn = getDerivRightBC(nType, tauNN_int, tauNN_bnd, m);
   const double viscous_heating = getDerivRightBC(nType, velocity[p_int][iN]*tauNN_int, velocity[p][iN]*tauNN_bnd, m) +
                                  vGradN[p][iT1]*mu[p]*(vGradN[p][iT1] + vGradT1[p][iN]) +
                                  vGradN[p][iT2]*mu[p]*(vGradN[p][iT2] + vGradT2[p][iN]);

   // Update the RHS
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
                            + mix.GetHeatCapacity(T, MassFracs[p])*T*dN
                            - viscous_heating);
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      RHS[irE] -= (rho[p]*mix.GetSpecificInternalEnergy(s, T)*LS[s]);
}

