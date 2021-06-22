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

#ifndef __PROMETEO_BC_TYPES_H__
#define __PROMETEO_BC_TYPES_H__

#include "legion.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "prometeo_const.h"

#ifndef nSpec
   #error "nSpec is undefined"
#endif

// Stores upstream and downstream impinging shock conditions
struct IncomingShockParams {
   // Index where the shock will be injected
   int iShock;
   // Constant mixture composition
   double MolarFracs[nSpec];
   // Primitive variables upstream of the shock
   double pressure0;
   double temperature0;
   double velocity0[3];
   // Primitive variables downstream of the shock
   double pressure1;
   double temperature1;
   double velocity1[3];
};

// Stores averages for Recycle/Rescaling BC
struct RecycleAverageType {
   // Average weight
   double w;
   // Distance from the wall
   double y;
   // Properties
   double rho;
   // Primitive variables
   double temperature;
   double MolarFracs[nSpec];
   double velocity[3];
};

// The order of this enum must match the declaration of RecycleAverageType
enum RA_FieldIDs {
   // Average weight
   RA_FID_w = 101,
   // Distance from the wall
   RA_FID_y,
   // Properties
   RA_FID_rho,
   // Primitive variables
   RA_FID_temperature,
   RA_FID_MolarFracs,
   RA_FID_velocity,
   // keep last for counting
   RA_FID_last
};

// Stores Van Driest provile data for Recycle/Rescaling BC
struct BLDataType {
   double Uinf;
   double aVD;
   double bVD;
   double QVD;
   double Ueq;
};

// Stores rescaling data for Recycle/Rescaling BC
struct RescalingDataType {
   // Data for outer scaling
   double delta99VD;
   // Data for inner scaling
   double rhow;
   double uTau;
   double deltaNu;
};

#ifdef __cplusplus
}
#endif

#endif // __PROMETEO_BC_TYPES_H__
