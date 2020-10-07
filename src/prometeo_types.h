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

#ifndef Prometeo_Types_H
#define Prometeo_Types_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "prometeo_const.h"

#ifndef nSpec
   #error "nSpec is undefined"
#endif

#ifndef nEq
   #error "nEq is undefined"
#endif

struct Fluid_columns {
   // Grid point
   double centerCoordinates[3];
   double cellWidth[3];
   // Node types
   int nType_x;
   int nType_y;
   int nType_z;
   // Cell-center metrics for Euler fluxes
   double dcsi_e;
   double deta_e;
   double dzet_e;
   // Cell-center metrics for diffusion fluxes
   double dcsi_d;
   double deta_d;
   double dzet_d;
   // Staggered metrics
   double dcsi_s;
   double deta_s;
   double dzet_s;
   // Properties
   double rho;
   double mu;
   double lam;
   double Di[nSpec];
   double SoS;
   // Primitive variables
   double pressure;
   double temperature;
   double MassFracs[ nSpec];
   double MolarFracs[nSpec];
   double velocity[3];
   // Gradients
   double velocityGradientX[  3];
   double velocityGradientY[  3];
   double velocityGradientZ[  3];
   double temperatureGradient[3];
   // Conserved variables
   double Conserved[      nEq];
   double Conserved_old[  nEq];
//   double Conserved_hat[  nEq];
   double Conserved_t[    nEq];
   double Conserved_t_old[nEq];
//   // Fluxes
//   double FluxXCorr[nEq];
//   double FluxYCorr[nEq];
//   double FluxZCorr[nEq];
   // Shock sensors
   bool shockSensorX;
   bool shockSensorY;
   bool shockSensorZ;
   // NSCBC variables
   double dudtBoundary[3];
   double dTdtBoundary;
   double velocity_old_NSCBC[3];
   double temperature_old_NSCBC;
   // Injected profile variables
   double MolarFracs_profile[nSpec];
   double velocity_profile[3];
   double temperature_profile;
   // Recycle variables
   double temperature_recycle;
   double MolarFracs_recycle[nSpec];
   double velocity_recycle[3];
};

// The order of this enum must match the declaration of Fluid_columns
enum FieldIDs {
   // Grid point
   FID_centerCoordinates = 101,
   FID_cellWidth,
   // Node types
   FID_nType_x,
   FID_nType_y,
   FID_nType_z,
   // Cell-center metrics for Euler fluxes
   FID_dcsi_e,
   FID_deta_e,
   FID_dzet_e,
   // Cell-center metrics for diffusion fluxes
   FID_dcsi_d,
   FID_deta_d,
   FID_dzet_d,
   // Staggered metrics
   FID_dcsi_s,
   FID_deta_s,
   FID_dzet_s,
   // Properties
   FID_rho,
   FID_mu,
   FID_lam,
   FID_Di,
   FID_SoS,
   // Primitive variables
   FID_pressure,
   FID_temperature,
   FID_MassFracs,
   FID_MolarFracs,
   FID_velocity,
   // Gradients
   FID_velocityGradientX,
   FID_velocityGradientY,
   FID_velocityGradientZ,
   FID_temperatureGradient,
   // Conserved variables
   FID_Conserved,
   FID_Conserved_old,
//   FID_Conserved_hat,
   FID_Conserved_t,
   FID_Conserved_t_old,
   // Fluxes
//   FID_FluxXCorr,
//   FID_FluxYCorr,
//   FID_FluxZCorr,
   // Shock sensors
   FID_shockSensorX,
   FID_shockSensorY,
   FID_shockSensorZ,
   // NSCBC variables
   FID_dudtBoundary,
   FID_dTdtBoundary,
   FID_velocity_old_NSCBC,
   FID_temperature_old_NSCBC,
   // Injected profile variables
   FID_MolarFracs_profile,
   FID_velocity_profile,
   FID_temperature_profile,
   // Recycle variables
   FID_temperature_recycle,
   FID_MolarFracs_recycle,
   FID_velocity_recycle,
   // keep last for counting
   FID_last
};

enum {
   TID_InitializeMetric = 1,
   TID_CorrectGhostMetricX,
   TID_CorrectGhostMetricY,
   TID_CorrectGhostMetricZ,
   TID_UpdatePropertiesFromPrimitive,
   TID_GetVelocityGradients,
   TID_GetTemperatureGradient,
   TID_UpdateShockSensorX,
   TID_UpdateShockSensorY,
   TID_UpdateShockSensorZ,
   TID_UpdateUsingHybridEulerFluxX,
   TID_UpdateUsingHybridEulerFluxY,
   TID_UpdateUsingHybridEulerFluxZ,
   TID_UpdateUsingTENOAEulerFluxX,
   TID_UpdateUsingTENOAEulerFluxY,
   TID_UpdateUsingTENOAEulerFluxZ,
   TID_UpdateUsingDiffusionFluxX,
   TID_UpdateUsingDiffusionFluxY,
   TID_UpdateUsingDiffusionFluxZ,
   TID_UpdateUsingFluxNSCBCInflowXNeg,
   TID_UpdateUsingFluxNSCBCInflowYPos,
   TID_UpdateUsingFluxNSCBCOutflowXPos,
   TID_UpdateUsingFluxNSCBCOutflowYNeg,
   TID_UpdateUsingFluxNSCBCOutflowYPos
};

#ifdef __cplusplus
}
#endif

#endif // Prometeo_Types_H
