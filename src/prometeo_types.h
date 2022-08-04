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

#include "prometeo_const.h"

#ifndef nSpec
   #error "nSpec is undefined"
#endif

#ifndef nEq
   #error "nEq is undefined"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct Fluid_columns {
   // Grid point
   double centerCoordinates[3];
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
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   double Ki[nIons];
#endif
   // Primitive variables
   double pressure;
   double temperature;
   double MassFracs[ nSpec];
   double MolarFracs[nSpec];
   double velocity[3];
   // Electric variables
#ifdef ELECTRIC_FIELD
   double electricPotential;
   double electricField[3];
#endif
   // Conserved variables
   double Conserved[      nEq];
   double Conserved_old[  nEq];
   double Conserved_t[    nEq];
   double Conserved_t_old[nEq];
   // Shock sensors
   bool shockSensorX;
   bool shockSensorY;
   bool shockSensorZ;
   double DucrosSensor;
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
   // Laser variables
   double kernelProfile;
};

// The order of this enum must match the declaration of Fluid_columns
enum FieldIDs {
   // Grid point
   FID_centerCoordinates = 101,
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
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   FID_Ki,
#endif
   // Primitive variables
   FID_pressure,
   FID_temperature,
   FID_MassFracs,
   FID_MolarFracs,
   FID_velocity,
   // Electric variables
#ifdef ELECTRIC_FIELD
   FID_electricPotential,
   FID_electricField,
#endif
   // Conserved variables
   FID_Conserved,
   FID_Conserved_old,
   FID_Conserved_t,
   FID_Conserved_t_old,
   // Shock sensors
   FID_shockSensorX,
   FID_shockSensorY,
   FID_shockSensorZ,
   FID_DucrosSensor,
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
   // Laser variables
   FID_kernelProfile,
   // keep last for counting
   FID_last
};

enum {
   TID_LoadMixture = 1,
   // Metric tasks
   TID_InitializeMetric,
   TID_CorrectGhostMetricX,
   TID_CorrectGhostMetricY,
   TID_CorrectGhostMetricZ,
   // Variables tasks
   TID_UpdatePropertiesFromPrimitive,
   TID_UpdateConservedFromPrimitive,
   TID_UpdatePrimitiveFromConserved,
   // Sensor tasks
   TID_UpdateDucrosSensor,
   TID_UpdateShockSensorX,
   TID_UpdateShockSensorY,
   TID_UpdateShockSensorZ,
   // RHS tasks
   TID_UpdateUsingHybridEulerFluxX,
   TID_UpdateUsingHybridEulerFluxY,
   TID_UpdateUsingHybridEulerFluxZ,
   TID_UpdateUsingTENOEulerFluxX,
   TID_UpdateUsingTENOEulerFluxY,
   TID_UpdateUsingTENOEulerFluxZ,
   TID_UpdateUsingTENOAEulerFluxX,
   TID_UpdateUsingTENOAEulerFluxY,
   TID_UpdateUsingTENOAEulerFluxZ,
   TID_UpdateUsingTENOLADEulerFluxX,
   TID_UpdateUsingTENOLADEulerFluxY,
   TID_UpdateUsingTENOLADEulerFluxZ,
   TID_UpdateUsingSkewSymmetricEulerFluxX,
   TID_UpdateUsingSkewSymmetricEulerFluxY,
   TID_UpdateUsingSkewSymmetricEulerFluxZ,
   TID_UpdateUsingDiffusionFluxX,
   TID_UpdateUsingDiffusionFluxY,
   TID_UpdateUsingDiffusionFluxZ,
   TID_UpdateUsingFluxNSCBCInflowXNeg,
   TID_UpdateUsingFluxNSCBCInflowXPos,
   TID_UpdateUsingFluxNSCBCInflowYNeg,
   TID_UpdateUsingFluxNSCBCInflowYPos,
   TID_UpdateUsingFluxNSCBCInflowZNeg,
   TID_UpdateUsingFluxNSCBCInflowZPos,
   TID_UpdateUsingFluxNSCBCOutflowXNeg,
   TID_UpdateUsingFluxNSCBCOutflowXPos,
   TID_UpdateUsingFluxNSCBCOutflowYNeg,
   TID_UpdateUsingFluxNSCBCOutflowYPos,
   TID_UpdateUsingFluxNSCBCOutflowZNeg,
   TID_UpdateUsingFluxNSCBCOutflowZPos,
   TID_UpdateUsingFluxNSCBCFarFieldXNeg,
   TID_UpdateUsingFluxNSCBCFarFieldXPos,
   TID_UpdateUsingFluxNSCBCFarFieldYNeg,
   TID_UpdateUsingFluxNSCBCFarFieldYPos,
   TID_UpdateUsingFluxNSCBCFarFieldZNeg,
   TID_UpdateUsingFluxNSCBCFarFieldZPos,
   TID_UpdateUsingFluxIncomingShockYPos,
   // Forcing tasks
   TID_CalculateAveragePD,
   TID_AddDissipationX,
   TID_AddDissipationY,
   TID_AddDissipationZ,
   // BC tasks
   TID_AddRecycleAverageBC,
   TID_SetNSCBC_InflowBC_X,
   TID_SetNSCBC_InflowBC_Y,
   TID_SetNSCBC_InflowBC_Z,
   TID_SetNSCBC_OutflowBC,
   TID_SetIncomingShockBC,
   TID_SetRecycleRescalingBC,
   // Chemistry tasks
   TID_UpdateChemistry,
   TID_AddChemistrySources,
   // CFL tasks
   TID_CalculateMaxSpectralRadius,
   // Average tasks
   TID_Add2DAveragesX,
   TID_Add2DAveragesY,
   TID_Add2DAveragesZ,
   TID_Add1DAveragesX,
   TID_Add1DAveragesY,
   TID_Add1DAveragesZ,
#ifdef ELECTRIC_FIELD
   // Electric field solver tasks
   TID_initFFTplans,
   TID_destroyFFTplans,
   TID_performDirFFTFromField,
   TID_performDirFFTFromMix,
   TID_performInvFFT,
   TID_solveTridiagonals,
   TID_GetElectricField,
   TID_UpdateUsingIonDriftFluxX,
   TID_UpdateUsingIonDriftFluxY,
   TID_UpdateUsingIonDriftFluxZ,
   TID_AddIonWindSources,
   TID_CorrectIonsBCXNeg,
   TID_CorrectIonsBCXPos,
   TID_CorrectIonsBCYNeg,
   TID_CorrectIonsBCYPos,
   TID_CorrectIonsBCZNeg,
   TID_CorrectIonsBCZPos,
#endif
};

enum REDOP_ID{
   REGENT_REDOP_SUM_VEC3 = 101,
   REGENT_REDOP_SUM_VECNSP,
   REGENT_REDOP_SUM_VEC6,
};

// Data type for bounding box
/* vertices order:
//
//	   7+--------+6
//	    |\         \
//	    | 4+--------+5
//	    |  |        |
//	   3+  |   2+   |
//	     \ |        |
//	      0+--------+1
*/
struct bBoxType {
   double v0[3];
   double v1[3];
   double v2[3];
   double v3[3];
   double v4[3];
   double v5[3];
   double v6[3];
   double v7[3];
};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "my_array.hpp"
// Define common tipe of arrays and matrices that we are going to use
// NOTE: Regent is not going to see this when includes this file
typedef MyArray<double,     3> Vec3;
typedef MyArray<double, nSpec> VecNSp;
typedef MyArray<double,   nEq> VecNEq;
#endif

#endif // Prometeo_Types_H
