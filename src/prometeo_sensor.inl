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

#include <math.h>

#ifndef __CUDA_H__
#ifdef __CUDACC__
#define __CUDA_H__ __device__
#else
#define __CUDA_H__
#endif
#endif

#ifndef __UNROLL__
#ifdef __CUDACC__
#define __UNROLL__ #pragma unroll
#else
#define __UNROLL__
#endif
#endif

// Modified Docros sensor
__CUDA_H__
inline double UpdateDucrosSensorTask::DucrosSensor(
                           const AccessorRO<  Vec3, 3> &velocity,
                           const AccessorRO<   int, 3> &nType_csi,
                           const AccessorRO<   int, 3> &nType_eta,
                           const AccessorRO<   int, 3> &nType_zet,
                           const AccessorRO<double, 3> &dcsi_d,
                           const AccessorRO<double, 3> &deta_d,
                           const AccessorRO<double, 3> &dzet_d,
                           const Point<3> &p,
                           const Rect<3> &bounds,
                           const double eps) {
   // Compute velocity gradients
   Vec3 vGradX = getDeriv<Xdir>(velocity, p, nType_csi[p], dcsi_d[p], bounds);
   Vec3 vGradY = getDeriv<Ydir>(velocity, p, nType_eta[p], deta_d[p], bounds);
   Vec3 vGradZ = getDeriv<Zdir>(velocity, p, nType_zet[p], dzet_d[p], bounds);

   // Compute the Ducros sensor
   const double div = vGradX[0] + vGradY[1] + vGradZ[2];
   const double div2 = div*div;
   const double omz = vGradX[1] - vGradY[0];
   const double omy = vGradY[2] - vGradZ[1];
   const double omx = vGradZ[0] - vGradX[2];
   const double om2 = omx*omx + omy*omy + omz*omz;
   return div2/(div2 + om2 + eps);
}

