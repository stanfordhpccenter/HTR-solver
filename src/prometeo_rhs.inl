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
                          const AccessorRO<VecNEq, 3> Conserved,
                          const AccessorRO<double, 3> rho,
                          const AccessorRO<VecNSp, 3> MassFracs,
                          const AccessorRO<  Vec3, 3> velocity,
                          const AccessorRO<double, 3> pressure,
                          const Point<3> p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> bounds) {

   // TODO: implement some sort of static_if
   int iN;
   if      (dir == Xdir) iN = 0;
   else if (dir == Ydir) iN = 1;
   else if (dir == Zdir) iN = 2;

   const double H = (Conserved[p][irE] + pressure[p])/rho[p];

   for (int l=0; l < KennedyNSum[nType]; l++) {
      // offset point
      const Point<3> pp = warpPeriodic<dir, Plus>(bounds, p, dsize, l+1);
      const int off_l = l*(nEq+1);

      // compute the summations
      const double rhom = rho[p] + rho[pp];
      const double vm = -(velocity[p][iN] + velocity[pp][iN]);
      for (int i=0; i<nSpec; i++)
         Sums[off_l+i] = rhom*vm*(MassFracs[p][i] +  MassFracs[pp][i]);
      for (int i=0; i<3; i++)
         Sums[off_l+irU+i] = rhom*vm*(velocity[p][i] + velocity[pp][i]);
      Sums[off_l+irE] = rhom*vm*(H + (Conserved[pp][irE] + pressure[pp])/rho[pp]);
      Sums[off_l+nEq] = (pressure[p] + pressure[pp]);
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(double *Flux,
                          const double *Sums,
                          const AccessorRO<VecNEq, 3> Conserved,
                          const AccessorRO<  Vec3, 3> velocity,
                          const AccessorRO<double, 3> pressure,
                          const Point<3> p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> bounds) {

   // TODO: implement some sort of static_if
   int iN;
   if      (dir == Xdir) iN = 0;
   else if (dir == Ydir) iN = 1;
   else if (dir == Zdir) iN = 2;

   if (nType == L_S_node) {
      // This is a staggered node
      for (int i=0; i<nEq; i++)
         Flux[i] = -(Conserved[p][i]*velocity[p][iN]);
      Flux[irU+iN] -= pressure[p];
      Flux[irE   ] -= pressure[p]*velocity[p][iN];

   }
   else if (nType == Rm1_S_node) {
      // This is a staggered node
      const Point<3> pp = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));
      for (int i=0; i<nEq; i++)
         Flux[i] = -(Conserved[pp][i]*velocity[pp][iN]);
      Flux[irU+iN] -= pressure[pp];
      Flux[irE   ] -= pressure[pp]*velocity[pp][iN];

   }
   else {
      double   f[nEq+1];
      double acc[nEq+1];
      const double * Coeff = KennedyCoeff[nType];
      for (int i=0; i<nEq+1; i++) f[i] = 0.0;
      for (int l=0; l<KennedyOrder[nType]; l++) {
         const int off_l = l*(nEq+1);
         for (int i = 0; i < nEq+1; i++) acc[i] = 0.0;
         for (int m = 0; m < l+1; m++) {
            const int off_m = m*3*(nEq+1);
            for (int i = 0; i < nEq+1; i++)
               acc[i] += Sums[off_m + off_l + i];
         }
         for (int i=0; i<nEq+1; i++) f[i] += Coeff[l]*acc[i];
      }

      for (int i=0; i<nEq; i++) Flux[i] = 0.25*f[i];
      // add pressure contribution to normal momentum equation
      Flux[irU+iN] -= f[nEq];
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(double * Flux,
                          const AccessorRO<VecNEq, 3> Conserved,
                          const AccessorRO<double, 3> rho,
                          const AccessorRO<VecNSp, 3> MassFracs,
                          const AccessorRO<  Vec3, 3> velocity,
                          const AccessorRO<double, 3> pressure,
                          const Point<3> p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> bounds) {

   // TODO: implement some sort of static_if
   int iN;
   if      (dir == Xdir) iN = 0;
   else if (dir == Ydir) iN = 1;
   else if (dir == Zdir) iN = 2;

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
const typename UpdateUsingEulerFluxUtils<dir>::RoeAveragesStruct UpdateUsingEulerFluxUtils<dir>::ComputeRoeAverages(
                           const Mix &mix,
                           const double *ConservedL, const double *ConservedR,
                           const double        *YiL, const double        *YiR,
                           const double          TL, const double          TR,
                           const double   pressureL, const double   pressureR,
                           const double  *velocityL, const double  *velocityR,
                           const double        rhoL, const double        rhoR) {
   // Compute quantities on the left (L) and right (R) states
   const double MixWL = GetMolarWeightFromYi(YiL, mix);
   const double MixWR = GetMolarWeightFromYi(YiR, mix);

   const double gammaL = GetGamma(TL, MixWL, YiL, mix);
   const double gammaR = GetGamma(TR, MixWR, YiR, mix);

   /*const*/ double dpdrhoiL[nSpec]; Getdpdrhoi(dpdrhoiL, gammaL, TL, YiL, mix);
   /*const*/ double dpdrhoiR[nSpec]; Getdpdrhoi(dpdrhoiR, gammaR, TR, YiR, mix);

   const double dpdeL = Getdpde(rhoL, gammaL, mix);
   const double dpdeR = Getdpde(rhoR, gammaR, mix);

   const double TotalEnergyL = ConservedL[irE]/rhoL;
   const double TotalEnergyR = ConservedR[irE]/rhoR;

   const double TotalEnthalpyL = TotalEnergyL + pressureL/rhoL;
   const double TotalEnthalpyR = TotalEnergyR + pressureR/rhoR;

   // Compute Roe averaged state
   RoeAveragesStruct avgs;

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
   const double de = TotalEnergyR - 0.5*dot(velocityR, velocityR)
                   -(TotalEnergyL - 0.5*dot(velocityL, velocityL));
   double drhoi[nSpec];
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

   return avgs;
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::computeLeftEigenvectors(double *L, const RoeAveragesStruct &avgs) {

   // TODO: implement some sort of static_if
   int iN; int iT1; int iT2;
   if      (dir == Xdir) { iN = 0; iT1 = 1; iT2 = 2; }
   else if (dir == Ydir) { iN = 1; iT1 = 0; iT2 = 2; }
   else if (dir == Zdir) { iN = 2; iT1 = 0; iT2 = 1; }

   // initialize L
   __UNROLL__
   for (int i = 0; i<nEq*nEq; i++) L[i] = 0.0;

   // Compute constants
   const double iaRoe  = 1.0/avgs.a;
   const double iaRoe2 = 1.0/avgs.a2;
   const double Coeff = (avgs.E - dot(avgs.velocity, avgs.velocity))*avgs.dpde/avgs.rho;
   double b[nSpec];
   __UNROLL__
   for (int i=0; i<nSpec; i++)
      b[i] = (Coeff - avgs.dpdrhoi[i])*iaRoe2;
   const double d = avgs.dpde/(avgs.rho*avgs.a2);
   const double c[] = {avgs.velocity[0]*d,
                       avgs.velocity[1]*d,
                       avgs.velocity[2]*d};

   // First row
   {
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         // 0*nEq + j
         L[j] = -0.5*(b[j] - avgs.velocity[iN]*iaRoe);
      __UNROLL__
      for (int j=0; j<3; j++)
         L[    nSpec+j] = -0.5*c[j];
      L[    nSpec+iN] -= 0.5*iaRoe;
      L[    nSpec+3 ] = 0.5*d;
   }

   // From 1 to nSpec
   __UNROLL__
   for (int i=1; i<nSpec+1; i++) {
      const int row = i*nEq;
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L[row+j] = avgs.Yi[i-1]*b[j];
      L[row+i-1] += 1.0;
      __UNROLL__
      for (int j=0; j<3; j++)
         L[row+nSpec+j] = avgs.Yi[i-1]*c[j];
      L[row+nSpec+3] = -avgs.Yi[i-1]*d;
   }

   // nSpec + 1
   {
      const int row = (nSpec+1)*nEq;
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L[row+j] = - avgs.velocity[iT1];
      // L[row+nSpec+iN] = 0.0;
      L[row+nSpec+iT1] = 1.0;
      //L[row+nSpec+iT2] = 0.0;
      //L[row+nSpec+  3] = 0.0;
   }

   // nSpec + 2
   {
      const int row = (nSpec+2)*nEq;
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         L[row+j] = - avgs.velocity[iT2];
      //L[row+nSpec+iN] = 0.0;
      //L[row+nSpec+iT1] = 0.0;
      L[row+nSpec+iT2] = 1.0;
      //L[row+nSpec+  3] = 0.0;
   }

   // nSpec+3
   {
      const int row = (nSpec+3)*nEq;
      __UNROLL__
      for (int j=0; j<nSpec; j++)
         // (nEq-1)*nEq + j
         L[row+j] = -0.5*(b[j] + avgs.velocity[iN]*iaRoe);
      __UNROLL__
      for (int j=0; j<3; j++)
         L[row+nSpec+j] = -0.5*c[j];
      L[row+nSpec+iN] += 0.5*iaRoe;
      L[row+nSpec+3 ] = 0.5*d;
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::computeRightEigenvectors(double *K, const RoeAveragesStruct &avgs) {

   // TODO: implement some sort of static_if
   int iN; int iT1; int iT2;
   if      (dir == Xdir) { iN = 0; iT1 = 1; iT2 = 2; }
   else if (dir == Ydir) { iN = 1; iT1 = 0; iT2 = 2; }
   else if (dir == Zdir) { iN = 2; iT1 = 0; iT2 = 1; }

   // initialize K
   __UNROLL__
   for (int i = 0; i<nEq*nEq; i++) K[i] = 0.0;

   // fill K
   __UNROLL__
   for (int i = 0; i<nSpec; i++) {
      const int row = i*nEq;
      K[row +       0] = avgs.Yi[i];
      K[row +     i+1] = 1.0;
      K[row + nSpec+3] = avgs.Yi[i];
   }

   {
      const int row = (nSpec+iN)*nEq;
      K[row +       0] = avgs.velocity[iN] - avgs.a;
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K[row +  i+1] = avgs.velocity[iN];
      //K[row + nSpec+1] = 0.0;
      //K[row + nSpec+2] = 0.0;
      K[row + nSpec+3] = avgs.velocity[iN] + avgs.a;
   }

   {
      const int row = (nSpec+iT1)*nEq;
      K[row +       0] = avgs.velocity[iT1];
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K[row +  i+1] = avgs.velocity[iT1];
      K[row + nSpec+1] = 1.0;
      //K[row + nSpec+2] = 0.0;
      K[row + nSpec+3] = avgs.velocity[iT1];
   }

   {
      const int row = (nSpec+iT2)*nEq;
      K[row +       0] = avgs.velocity[iT2];
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K[row +  i+1] = avgs.velocity[iT2];
      //K[row + nSpec+1] = 0.0;
      K[row + nSpec+2] = 1.0;
      K[row + nSpec+3] = avgs.velocity[iT2];
   }

   {
      const int row = (nSpec+3)*nEq;
      K[row +       0] = avgs.H - avgs.velocity[iN]*avgs.a;
      const double dedp = 1.0/avgs.dpde;
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         K[row +  i+1] = avgs.E - avgs.rho*avgs.dpdrhoi[i]*dedp;
      K[row + nSpec+1] = avgs.velocity[iT1];
      K[row + nSpec+2] = avgs.velocity[iT2];
      K[row + nSpec+3] = avgs.H + avgs.velocity[iN]*avgs.a;
   }
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::getPlusMinusFlux(double *FluxP, double *FluxM,
                      const double *L,
                      const double *Conserved,
                      const double velocity,
                      const double pressure,
                      const double Lam1,
                      const double Lam,
                      const double LamN) {

   // TODO: implement some sort of static_if
   int iN;
   if      (dir == Xdir) iN = 0;
   else if (dir == Ydir) iN = 1;
   else if (dir == Zdir) iN = 2;

   // Compute the Euler fluxes
   double Flux[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      Flux[i] = Conserved[i]*velocity;
   Flux[irU+iN] += pressure;
   Flux[irE   ] += pressure*velocity;

   // Project in the characteristic space
   double Q[nEq]; MatMul<nEq>(L, Conserved, Q);
   double F[nEq]; MatMul<nEq>(L,      Flux, F);

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
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::TENOFluxReconstruction(double *Flux,
                            const AccessorRO<VecNEq, 3> Conserved,
                            const AccessorRO<double, 3> SoS,
                            const AccessorRO<double, 3> rho,
                            const AccessorRO<  Vec3, 3> velocity,
                            const AccessorRO<double, 3> pressure,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3> temperature,
                            const Point<3> p,
                            const int nType,
                            const Mix &mix,
                            const coord_t dsize,
                            const Rect<3> bounds) {

   // TODO: implement some sort of static_if
   int iN;
   if      (dir == Xdir) iN = 0;
   else if (dir == Ydir) iN = 1;
   else if (dir == Zdir) iN = 2;

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
   const RoeAveragesStruct RoeAvgs = ComputeRoeAverages(mix,
                           Conserved[p].v,   Conserved[pP1].v,
                           MassFracs[p].v,   MassFracs[pP1].v,
                         temperature[p]  , temperature[pP1],
                            pressure[p]  ,    pressure[pP1],
                            velocity[p].v,    velocity[pP1].v,
                                 rho[p]  ,         rho[pP1]);

   // Compute left eigenvector matrix
   double K[nEq*nEq]; computeLeftEigenvectors(K, RoeAvgs);

   // Compute +/- fluxes
   double FluxPM2[nEq]; double FluxMM2[nEq]; getPlusMinusFlux(FluxPM2, FluxMM2, K, Conserved[pM2].v, velocity[pM2][iN], pressure[pM2], Lam1, Lam, LamN);
   double FluxPM1[nEq]; double FluxMM1[nEq]; getPlusMinusFlux(FluxPM1, FluxMM1, K, Conserved[pM1].v, velocity[pM1][iN], pressure[pM1], Lam1, Lam, LamN);
   double FluxP  [nEq]; double FluxM  [nEq]; getPlusMinusFlux(FluxP  , FluxM  , K, Conserved[p  ].v, velocity[p  ][iN], pressure[p  ], Lam1, Lam, LamN);
   double FluxPP1[nEq]; double FluxMP1[nEq]; getPlusMinusFlux(FluxPP1, FluxMP1, K, Conserved[pP1].v, velocity[pP1][iN], pressure[pP1], Lam1, Lam, LamN);
   double FluxPP2[nEq]; double FluxMP2[nEq]; getPlusMinusFlux(FluxPP2, FluxMP2, K, Conserved[pP2].v, velocity[pP2][iN], pressure[pP2], Lam1, Lam, LamN);
   double FluxPP3[nEq]; double FluxMP3[nEq]; getPlusMinusFlux(FluxPP3, FluxMP3, K, Conserved[pP3].v, velocity[pP3][iN], pressure[pP3], Lam1, Lam, LamN);

   // Reconstruct Fluxes
   double FPlus[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      FPlus[i] = TENOreconstructionPlus(FluxPM2[i], FluxPM1[i], FluxP  [i],
                                        FluxPP1[i], FluxPP2[i], FluxPP3[i], nType);

   double FMinus[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      FMinus[i] = TENOreconstructionMinus(FluxMM2[i], FluxMM1[i], FluxM  [i],
                                          FluxMP1[i], FluxMP2[i], FluxMP3[i], nType);

   double F[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      F[i] = -0.5*(FPlus[i] + FMinus[i]);

   // Compute right eigenvector matrix
   computeRightEigenvectors(K, RoeAvgs);

   // Go back to the physical space
   MatMul<nEq>(K, F, Flux);
}

template<direction dir>
__CUDA_H__
void UpdateUsingEulerFluxUtils<dir>::TENOAFluxReconstruction(double *Flux,
                            const AccessorRO<VecNEq, 3> Conserved,
                            const AccessorRO<double, 3> SoS,
                            const AccessorRO<double, 3> rho,
                            const AccessorRO<  Vec3, 3> velocity,
                            const AccessorRO<double, 3> pressure,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3> temperature,
                            const Point<3> p,
                            const int nType,
                            const Mix &mix,
                            const coord_t dsize,
                            const Rect<3> bounds) {

   // TODO: implement some sort of static_if
   int iN;
   if      (dir == Xdir) iN = 0;
   else if (dir == Ydir) iN = 1;
   else if (dir == Zdir) iN = 2;

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
   const RoeAveragesStruct RoeAvgs = ComputeRoeAverages(mix,
                           Conserved[p].v,   Conserved[pP1].v,
                           MassFracs[p].v,   MassFracs[pP1].v,
                         temperature[p]  , temperature[pP1],
                            pressure[p]  ,    pressure[pP1],
                            velocity[p].v,    velocity[pP1].v,
                                 rho[p]  ,         rho[pP1]);

   // Compute left eigenvector matrix
   double K[nEq*nEq]; computeLeftEigenvectors(K, RoeAvgs);

   // Compute +/- fluxes
   double FluxPM2[nEq]; double FluxMM2[nEq]; getPlusMinusFlux(FluxPM2, FluxMM2, K, Conserved[pM2].v, velocity[pM2][iN], pressure[pM2], Lam1, Lam, LamN);
   double FluxPM1[nEq]; double FluxMM1[nEq]; getPlusMinusFlux(FluxPM1, FluxMM1, K, Conserved[pM1].v, velocity[pM1][iN], pressure[pM1], Lam1, Lam, LamN);
   double FluxP  [nEq]; double FluxM  [nEq]; getPlusMinusFlux(FluxP  , FluxM  , K, Conserved[p  ].v, velocity[p  ][iN], pressure[p  ], Lam1, Lam, LamN);
   double FluxPP1[nEq]; double FluxMP1[nEq]; getPlusMinusFlux(FluxPP1, FluxMP1, K, Conserved[pP1].v, velocity[pP1][iN], pressure[pP1], Lam1, Lam, LamN);
   double FluxPP2[nEq]; double FluxMP2[nEq]; getPlusMinusFlux(FluxPP2, FluxMP2, K, Conserved[pP2].v, velocity[pP2][iN], pressure[pP2], Lam1, Lam, LamN);
   double FluxPP3[nEq]; double FluxMP3[nEq]; getPlusMinusFlux(FluxPP3, FluxMP3, K, Conserved[pP3].v, velocity[pP3][iN], pressure[pP3], Lam1, Lam, LamN);

   // Reconstruct Fluxes
   double FPlus[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      FPlus[i] = TENOAreconstructionPlus(FluxPM2[i], FluxPM1[i], FluxP  [i],
                                         FluxPP1[i], FluxPP2[i], FluxPP3[i], nType);

   double FMinus[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      FMinus[i] = TENOAreconstructionMinus(FluxMM2[i], FluxMM1[i], FluxM  [i],
                                           FluxMP1[i], FluxMP2[i], FluxMP3[i], nType);

   double F[nEq];
   __UNROLL__
   for (int i=0; i<nEq; i++)
      F[i] = -0.5*(FPlus[i] + FMinus[i]);

   // Compute right eigenvector matrix
   computeRightEigenvectors(K, RoeAvgs);

   // Go back to the physical space
   MatMul<nEq>(K, F, Flux);
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingDiffusionFluxTask
//-----------------------------------------------------------------------------

template<>
__CUDA_H__
inline void UpdateUsingDiffusionFluxTask<Xdir>::GetSigma(
                  double *sigma, const int nType, const double m_s,
                  const AccessorRO<double, 3>          mu,
                  const AccessorRO<  Vec3, 3>    velocity,
                  const AccessorRO<  Vec3, 3>      vGradY,
                  const AccessorRO<  Vec3, 3>      vGradZ,
                  const Point<3> p, const Point<3> pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdX_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdX_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdX_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdY_s = Interp2Staggered(nType, vGradY[p][0], vGradY[pp1][0]);
   const double dVdY_s = Interp2Staggered(nType, vGradY[p][1], vGradY[pp1][1]);
   const double dUdZ_s = Interp2Staggered(nType, vGradZ[p][0], vGradZ[pp1][0]);
   const double dWdZ_s = Interp2Staggered(nType, vGradZ[p][2], vGradZ[pp1][2]);

   sigma[0] = mu_s*(4*dUdX_s - 2*dVdY_s - 2*dWdZ_s)/3;
   sigma[1] = mu_s*(dVdX_s+dUdY_s);
   sigma[2] = mu_s*(dWdX_s+dUdZ_s);
}

template<>
__CUDA_H__
inline void UpdateUsingDiffusionFluxTask<Ydir>::GetSigma(
                  double *sigma, const int nType, const double m_s,
                  const AccessorRO<double, 3>          mu,
                  const AccessorRO<  Vec3, 3>    velocity,
                  const AccessorRO<  Vec3, 3>      vGradX,
                  const AccessorRO<  Vec3, 3>      vGradZ,
                  const Point<3> p, const Point<3> pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdY_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdY_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdY_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdX_s = Interp2Staggered(nType, vGradX[p][0], vGradX[pp1][0]);
   const double dVdX_s = Interp2Staggered(nType, vGradX[p][1], vGradX[pp1][1]);
   const double dVdZ_s = Interp2Staggered(nType, vGradZ[p][1], vGradZ[pp1][1]);
   const double dWdZ_s = Interp2Staggered(nType, vGradZ[p][2], vGradZ[pp1][2]);

   sigma[0] = mu_s*(dUdY_s+dVdX_s);
   sigma[1] = mu_s*(4*dVdY_s - 2*dUdX_s - 2*dWdZ_s)/3;
   sigma[2] = mu_s*(dWdY_s+dVdZ_s);
}

template<>
__CUDA_H__
inline void UpdateUsingDiffusionFluxTask<Zdir>::GetSigma(
                  double *sigma, const int nType, const double m_s,
                  const AccessorRO<double, 3>          mu,
                  const AccessorRO<  Vec3, 3>    velocity,
                  const AccessorRO<  Vec3, 3>      vGradX,
                  const AccessorRO<  Vec3, 3>      vGradY,
                  const Point<3> p, const Point<3> pp1) {

   const double mu_s  = Interp2Staggered(nType,  mu[p],  mu[pp1]);

   const double dUdZ_s = m_s*(velocity[pp1][0] - velocity[p][0]);
   const double dVdZ_s = m_s*(velocity[pp1][1] - velocity[p][1]);
   const double dWdZ_s = m_s*(velocity[pp1][2] - velocity[p][2]);
   const double dUdX_s = Interp2Staggered(nType, vGradX[p][0], vGradX[pp1][0]);
   const double dWdX_s = Interp2Staggered(nType, vGradX[p][2], vGradX[pp1][2]);
   const double dVdY_s = Interp2Staggered(nType, vGradY[p][1], vGradY[pp1][1]);
   const double dWdY_s = Interp2Staggered(nType, vGradY[p][2], vGradY[pp1][2]);

   sigma[0] = mu_s*(dUdZ_s+dWdX_s);
   sigma[1] = mu_s*(dVdZ_s+dWdY_s);
   sigma[2] = mu_s*(4*dWdZ_s - 2*dUdX_s - 2*dVdY_s)/3;
}

template<direction dir>
__CUDA_H__
inline void UpdateUsingDiffusionFluxTask<dir>::GetDiffusionFlux(
                  double *Flux, const int nType, const double m_s, const Mix &mix,
                  const AccessorRO<double, 3>         rho,
                  const AccessorRO<double, 3>          mu,
                  const AccessorRO<double, 3>         lam,
                  const AccessorRO<VecNSp, 3>          Di,
                  const AccessorRO<double, 3> temperature,
                  const AccessorRO<  Vec3, 3>    velocity,
                  const AccessorRO<VecNSp, 3>          Xi,
                  const AccessorRO<VecNEq, 3>       rhoYi,
                  const AccessorRO<  Vec3, 3>      vGradY,
                  const AccessorRO<  Vec3, 3>      vGradZ,
                  const Point<3> p,
                  const coord_t size,
                  const Rect<3> bounds) {

   // access i+1 point (warp around boundaries)
   const Point<3> pp1 = warpPeriodic<dir, Plus>(bounds, p, size, 1);

   // Mixture properties at the staggered location
   const double rho_s = Interp2Staggered(nType, rho[p], rho[pp1]);
   const double iMixW_s = 1.0/Interp2Staggered(nType, GetMolarWeightFromXi(Xi.ptr(p  )->v, mix),
                                                      GetMolarWeightFromXi(Xi.ptr(pp1)->v, mix));

   // Primitive and conserved variables at the staggered location
   const double T_s     =  Interp2Staggered(nType, temperature[p], temperature[pp1]);
   const double vel_s[] = {Interp2Staggered(nType,    velocity[p][0], velocity[pp1][0]),
                           Interp2Staggered(nType,    velocity[p][1], velocity[pp1][1]),
                           Interp2Staggered(nType,    velocity[p][2], velocity[pp1][2])};

   // Viscous stress
   double sigma[3]; GetSigma(sigma, nType, m_s, mu, velocity, vGradY, vGradZ, p, pp1);

   // Assemble the fluxes
   const double uSigma = dot(vel_s, sigma);
   double heatFlux = Interp2Staggered(nType, lam[p], lam[pp1])*m_s*(temperature[p] - temperature[pp1]);

   // Species diffusion velocity
   double YiVi[nSpec];
   double ViCorr = 0.0;
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      YiVi[i] = Interp2Staggered(nType, Di[p][i],  Di[pp1][i])*
                                   m_s*(Xi[p][i] - Xi[pp1][i])*
                                   GetSpeciesMolarWeight(i, mix)*iMixW_s;
      ViCorr += YiVi[i];
   }

   // Partial Densities Fluxes
   __UNROLL__
   for (int i=0; i<nSpec; i++) {
      const double rhoYiVi = rho_s*YiVi[i] - ViCorr*Interp2Staggered(nType, rhoYi[p][i], rhoYi[pp1][i]);
      Flux[i] = -rhoYiVi;
      heatFlux += rhoYiVi*GetSpeciesEnthalpy(i, T_s, mix);
   }

   // Momentum Flux
   __UNROLL__
   for (int i=0; i<3; i++)
      Flux[irU+i] = sigma[i];

   // Energy Flux
   Flux[irE] = (uSigma - heatFlux);
}

//-----------------------------------------------------------------------------
// INLINE FUNCTIONS FOR UpdateUsingFluxNSCBCInflowTask
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCInflowMinusSideTask<dir>::addLODIfluxes(double *RHS,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3>  pressure,
                            const double       SoS,
                            const double       rho,
                            const double         T,
                            const double *velocity,
                            const double    *vGrad,
                            const double     *dudt,
                            const double      dTdt,
                            const Point<3> p,
                            const int  nType,
                            const double   m,
                            const Mix &mix) {

   // TODO: implement some sort of static_if
   const int iN = (dir == Xdir) ?   0 :
                  (dir == Ydir) ?   1 :
                /*(dir == Zdir) ?*/ 2;

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
                           /*(dir == Ydir) ?*/Point<3>(p.x  , p.y  , p.z+1);

      // Thermo-chemical quantities
      const double MixW_bnd = GetMolarWeightFromYi(MassFracs[p].v, mix);
      const double   Cp_bnd = GetHeatCapacity(T,   MassFracs[p].v, mix);

      // characteristic velocity leaving the domain
      const double lambda_1 = velocity[iN] - SoS;

      // characteristic velocity entering the domain
      const double lambda   = velocity[iN];

      // compute waves amplitudes
      const double dp_dn = getDerivLeftBC(nType, pressure[p], pressure[p_int], m);
      const double du_dn = vGrad[iN];
      double dY_dn[nSpec];
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         dY_dn[s] = getDerivLeftBC(nType, MassFracs[p][s], MassFracs[p_int][s], m);

      const double L1 = lambda_1*(dp_dn - rho*SoS*du_dn);
      double LS[nSpec];
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         LS[s] = lambda*dY_dn[s];
      const double LN = L1 - 2*rho*SoS*dudt[iN];
      double L2 = dTdt/T + (LN + L1)/(2*rho*Cp_bnd*T);
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         L2 -= MixW_bnd/GetSpeciesMolarWeight(s, mix)*LS[s];
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
void UpdateUsingFluxNSCBCInflowPlusSideTask<dir>::addLODIfluxes(double *RHS,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3>  pressure,
                            const double       SoS,
                            const double       rho,
                            const double         T,
                            const double *velocity,
                            const double    *vGrad,
                            const double     *dudt,
                            const double      dTdt,
                            const Point<3> p,
                            const int  nType,
                            const double   m,
                            const Mix &mix) {

   // TODO: implement some sort of static_if
   const int iN = (dir == Xdir) ?   0 :
                  (dir == Ydir) ?   1 :
                /*(dir == Zdir) ?*/ 2;

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
                           /*(dir == Ydir) ?*/Point<3>(p.x  , p.y  , p.z-1);

      // Thermo-chemical quantities
      const double MixW_bnd = GetMolarWeightFromYi(MassFracs[p].v, mix);
      const double   Cp_bnd = GetHeatCapacity(T,   MassFracs[p].v, mix);

      // characteristic velocity leaving the domain
      const double lambda_N = velocity[iN] + SoS;

      // characteristic velocity entering the domain
      const double lambda   = velocity[iN];

      // compute waves amplitudes
      const double dp_dn = getDerivRightBC(nType, pressure[p_int], pressure[p], m);
      const double du_dn = vGrad[iN];
      double dY_dn[nSpec];
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         dY_dn[s] = getDerivRightBC(nType, MassFracs[p_int][s], MassFracs[p][s], m);

      const double LN = lambda_N*(dp_dn + rho*SoS*du_dn);
      double LS[nSpec];
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         LS[s] = lambda*dY_dn[s];
      const double L1 = LN + 2*rho*SoS*dudt[iN];
      double L2 = dTdt/T + (LN + L1)/(2*rho*Cp_bnd*T);
      __UNROLL__
      for (int s=0; s<nSpec; s++)
         L2 -= MixW_bnd/GetSpeciesMolarWeight(s, mix)*LS[s];
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
void UpdateUsingFluxNSCBCOutflowMinusSideTask<dir>::addLODIfluxes(double *RHS,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3>       rho,
                            const AccessorRO<double, 3>        mu,
                            const AccessorRO<double, 3>  pressure,
                            const AccessorRO<  Vec3, 3>  velocity,
                            const AccessorRO<  Vec3, 3>    vGradN,
                            const AccessorRO<  Vec3, 3>   vGradT1,
                            const AccessorRO<  Vec3, 3>   vGradT2,
                            const double         SoS,
                            const double           T,
                            const double  *Conserved,
                            const Point<3>         p,
                            const int          nType,
                            const double           m,
                            const double     MaxMach,
                            const double LengthScale,
                            const double        PInf,
                            const Mix &mix) {

   // TODO: implement some sort of static_if
   int iN; int iT1; int iT2; Point<3> p_int;
   if      (dir == Xdir) { iN = 0; iT1 = 1; iT2 = 2; p_int = Point<3>(p.x+1, p.y  , p.z  ); }
   else if (dir == Ydir) { iN = 1; iT1 = 0; iT2 = 2; p_int = Point<3>(p.x  , p.y+1, p.z  ); }
   else if (dir == Zdir) { iN = 2; iT1 = 0; iT2 = 1; p_int = Point<3>(p.x  , p.y  , p.z+1); }

   // Thermo-chemical quantities
   const double   Cp_bnd = GetHeatCapacity(T,   MassFracs[p].v, mix);

   // BC-normal gradients
   const double   dp_dn = getDerivLeftBC(nType, pressure[p], pressure[p_int], m);
   const double drho_dn = getDerivLeftBC(nType,      rho[p],      rho[p_int], m);
   double dY_dn[nSpec];
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      dY_dn[s] = getDerivLeftBC(nType, MassFracs[p][s], MassFracs[p_int][s], m);

   // characteristic velocities
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_N = velocity[p][iN] + SoS;

   // compute waves amplitudes
   const double L1 = lambda_1*(dp_dn - rho[p]*SoS*vGradN[p][iN]);
   double LM[3];
   LM[iN ] = lambda*(dp_dn - SoS*SoS*drho_dn);
   LM[iT1] = lambda*vGradN[p][iT1];
   LM[iT2] = lambda*vGradN[p][iT2];
   double LS[nSpec];
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      LS[s] = lambda*dY_dn[s];
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
   const double d1 = (0.5*(L1 + LN) - LM[iN])/(SoS*SoS);
   double dM[3];
   dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
   dM[iT1] = LM[iT1];
   dM[iT2] = LM[iT2];
   const double dN = LM[iN]/(SoS*SoS);

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
                            + Cp_bnd*T*dN
                            - viscous_heating);
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      RHS[irE] -= (rho[p]*GetSpecificInternalEnergy(s, T, mix)*LS[s]);
}

template<direction dir>
__CUDA_H__
void UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::addLODIfluxes(double *RHS,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3>       rho,
                            const AccessorRO<double, 3>        mu,
                            const AccessorRO<double, 3>  pressure,
                            const AccessorRO<  Vec3, 3>  velocity,
                            const AccessorRO<  Vec3, 3>    vGradN,
                            const AccessorRO<  Vec3, 3>   vGradT1,
                            const AccessorRO<  Vec3, 3>   vGradT2,
                            const double         SoS,
                            const double           T,
                            const double  *Conserved,
                            const Point<3>         p,
                            const int          nType,
                            const double           m,
                            const double     MaxMach,
                            const double LengthScale,
                            const double        PInf,
                            const Mix &mix) {

   // TODO: implement some sort of static_if
   int iN; int iT1; int iT2; Point<3> p_int;
   if      (dir == Xdir) { iN = 0; iT1 = 1; iT2 = 2; p_int = Point<3>(p.x-1, p.y  , p.z  ); }
   else if (dir == Ydir) { iN = 1; iT1 = 0; iT2 = 2; p_int = Point<3>(p.x  , p.y-1, p.z  ); }
   else if (dir == Zdir) { iN = 2; iT1 = 0; iT2 = 1; p_int = Point<3>(p.x  , p.y  , p.z-1); }

   // Thermo-chemical quantities
   const double   Cp_bnd = GetHeatCapacity(T,   MassFracs[p].v, mix);

   // BC-normal gradients
   const double   dp_dn = getDerivRightBC(nType, pressure[p_int], pressure[p], m);
   const double drho_dn = getDerivRightBC(nType,      rho[p_int],      rho[p], m);
   double dY_dn[nSpec];
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      dY_dn[s] = getDerivRightBC(nType, MassFracs[p_int][s], MassFracs[p][s], m);

   // characteristic velocities
   const double lambda_1 = velocity[p][iN] - SoS;
   const double lambda   = velocity[p][iN];
   const double lambda_N = velocity[p][iN] + SoS;

   // compute waves amplitudes
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
   double LM[3];
   LM[iN ] = lambda*(dp_dn - SoS*SoS*drho_dn);
   LM[iT1] = lambda*vGradN[p][iT1];
   LM[iT2] = lambda*vGradN[p][iT2];
   double LS[nSpec];
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      LS[s] = lambda*dY_dn[s];
   const double LN = lambda_N*(dp_dn + rho[p]*SoS*vGradN[p][iN]);

   // Compute LODI fluxes
   const double d1 = (0.5*(L1 + LN) - LM[iN])/(SoS*SoS);
   double dM[3];
   dM[iN ] = (LN - L1)/(2*rho[p]*SoS);
   dM[iT1] = LM[iT1];
   dM[iT2] = LM[iT2];
   const double dN = LM[iN]/(SoS*SoS);

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
                            + Cp_bnd*T*dN
                            - viscous_heating);
   __UNROLL__
   for (int s=0; s<nSpec; s++)
      RHS[irE] -= (rho[p]*GetSpecificInternalEnergy(s, T, mix)*LS[s]);
}

