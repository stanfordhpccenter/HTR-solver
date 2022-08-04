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

#ifndef __PROMETEO_BC_HPP__
#define __PROMETEO_BC_HPP__

#include "legion.h"

using namespace Legion;

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "my_array.hpp"
#include "math_utils.hpp"
#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_bc.h"
#include "prometeo_bc_types.h"
#include "prometeo_redop.inl"
#include "prometeo_variables.hpp"

//-----------------------------------------------------------------------------
// TASK THAT COLLECTS THE SPATIAL AVERAGES FOR RECYCLE/RESCALING BC
//-----------------------------------------------------------------------------

class AddRecycleAverageTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    plane;
      LogicalRegion    avg;
      Mix mix;
      double Pbc;
      FieldID plane_fields   [FID_last    - 101];
      RA_FieldIDs avg_fields [RA_FID_last - 101];
   };
public:
   __CUDA_H__
   static inline void collectAverages(const AccessorRO<double, 3> &dcsi_d,
                                      const AccessorRO<double, 3> &deta_d,
                                      const AccessorRO<double, 3> &dzet_d,
                                      const AccessorRO<VecNSp, 3> &MolarFracs_profile,
                                      const AccessorRO<double, 3> &temperature_profile,
                                      const AccessorRO<  Vec3, 3> &velocity_profile,
                                      const AccessorSumRD<VecNSp, 1> &avg_MolarFracs,
                                      const AccessorSumRD<  Vec3, 1> &avg_velocity,
                                      const AccessorSumRD<double, 1> &avg_temperature,
                                      const AccessorSumRD<double, 1> &avg_rho,
                                      const double Pbc,
                                      const Point<3> &p,
                                      const Mix &mix) {
      const double vol = 1.0/(dcsi_d[p]*deta_d[p]*dzet_d[p]);
      const double MixW = mix.GetMolarWeightFromXi(MolarFracs_profile[p]);
      const double rho = mix.GetRho(Pbc, temperature_profile[p], MixW);
      double rvol = vol*rho;
      avg_rho        [p.y] <<= rvol;
      avg_temperature[p.y] <<= temperature_profile[p]*rvol;
      avg_MolarFracs [p.y] <<= MolarFracs_profile[p]*rvol;
      avg_velocity   [p.y] <<= velocity_profile[p]*rvol;
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE PRIMITIVE VARIABLES OF A NSCBC INFLOW
//-----------------------------------------------------------------------------

template<direction dir>
class SetNSCBC_InflowBCTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion BC;
      Mix mix;
      double Pbc;
      FieldID       BC_fields[FID_last - 101];
   };
public:
   __CUDA_H__
   static inline void setInflowPressure(const AccessorRO<VecNEq, 3> &Conserved,
                                        const AccessorRO<VecNSp, 3> &MolarFracs_profile,
                                        const AccessorRO<double, 3> &temperature_profile,
                                        const AccessorWO<double, 3> &pressure,
                                        const Point<3> &p,
                                        const Mix &mix) {
      VecNSp rhoYi;
      __UNROLL__
      for (int i=0; i<nSpec; i++) rhoYi[i] = Conserved[p][i];
      const double rho = mix.GetRhoFromRhoYi(rhoYi);
      const double MixW = mix.GetMolarWeightFromXi(MolarFracs_profile[p]);
      pressure[p] = mix.GetPFromRhoAndT(rho, MixW, temperature_profile[p]);
   }
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};


//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE PRIMITIVE VARIABLES OF A NSCBC OUTFLOW
//-----------------------------------------------------------------------------

class SetNSCBC_OutflowBCTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion BC;
      Mix mix;
      FieldID       BC_fields[FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE PRIMITIVE VARIABLES OF AN INCOMING SHOCK BC
//-----------------------------------------------------------------------------

class SetIncomingShockBCTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion       BC;
      IncomingShockParams params;
      Mix                 mix;
      FieldID             BC_fields[FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE PRIMITIVE VARIABLES OF AN RECYCLE RESCALING BC
//-----------------------------------------------------------------------------

class SetRecycleRescalingBCTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion     BC;
      LogicalRegion     avg;
      LogicalRegion     BC_interp;
      LogicalRegion     FIregion;
      FastInterpData    FIdata;
      RescalingDataType RdataIn;
      RescalingDataType RdataRe;
      Mix mix;
      double Pbc;
      FieldID            BC_fields [   FID_last - 101];
      RA_FieldIDs       avg_fields [RA_FID_last - 101];
      FieldID     BC_interp_fields [   FID_last - 101];
      FI_FieldIDs  FIregion_fields [FI_FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
private:
   __CUDA_H__
   static inline double interp(const double x1, const double x2, const double w) {
      return x1*w + x2*(1.0 - w);
   }

   __CUDA_H__
   static inline void interpAll(double  &t,
                                Vec3    &v,
                                VecNSp &Xi,
                                const AccessorRO<double, 3> &temperature_recycle,
                                const AccessorRO<  Vec3, 3> &velocity_recycle,
                                const AccessorRO<VecNSp, 3> &MolarFracs_recycle,
                                const AccessorRO<double, 1> &avg_y,
                                const AccessorRO< float, 1> &FI_xloc,
                                const AccessorRO< float, 1> &FI_iloc,
                                const FastInterpData &FIdata,
                                const Point<3> &p,
                                const double yR,
                                const double uFact) {
      const coord_t pAvg = FastInterpFindIndex(yR, FI_xloc, FI_iloc, FIdata);
      const coord_t pAp1 = pAvg + 1;
      const Point<3> pInt = Point<3>(p.x, pAvg, p.z);
      const Point<3> pIp1 = Point<3>(p.x, pAp1, p.z);
      const double w = (avg_y[pAp1] - yR)/(avg_y[pAp1] - avg_y[pAvg]);
      t = interp(temperature_recycle[pInt], temperature_recycle[pIp1], w);
      __UNROLL__
      for (int i=0; i<3; i++)
         v[i] = interp(velocity_recycle[pInt][i], velocity_recycle[pIp1][i], w)*uFact;
      __UNROLL__
      for (int i=0; i<nSpec; i++)
         Xi[i] = interp(MolarFracs_recycle[pInt][i], MolarFracs_recycle[pIp1][i], w);
   }

   __CUDA_H__
   static inline double weightf(const double x) {
      const double alpha = 4.0;
      //const double b = 0.125; // for Mach 2
      //const double b = 0.3;   // for Mach 3
      const double b = 0.4;   // Original
      // if we are outside the boudary layer
      if (x > 1.0) return 1.0;
      // blend otherwise
      const double rnum = alpha*(x-b);
      const double rden = b + (1.0-2.0*b)*x;
      return 0.5*(1.0 + tanh(rnum/rden)/tanh(alpha));
   }

   __CUDA_H__
   static inline double bernardinidamp(const double x) {
      return 0.5*(1.0 - tanh(5.0*(x-1.75)));
   }

public:
   __CUDA_H__
   static inline void GetRescaled(double  &t,
                                  Vec3    &v,
                                  VecNSp &Xi,
                                  const AccessorRO<  Vec3, 3> &centerCoordinates,
                                  const AccessorRO<double, 3> &temperature_recycle,
                                  const AccessorRO<  Vec3, 3> &velocity_recycle,
                                  const AccessorRO<VecNSp, 3> &MolarFracs_recycle,
                                  const AccessorRO<double, 3> &temperature_profile,
                                  const AccessorRO<  Vec3, 3> &velocity_profile,
                                  const AccessorRO<VecNSp, 3> &MolarFracs_profile,
                                  const AccessorRO<double, 1> &avg_y,
                                  const AccessorRO< float, 1> &FI_xloc,
                                  const AccessorRO< float, 1> &FI_iloc,
                                  const FastInterpData &FIdata,
                                  const Point<3> &p,
                                  const double yInnFact,
                                  const double yOutFact,
                                  const double uInnFact,
                                  const double uOutFact,
                                  const double idelta99Inl) {
      // Wall-normal distance
      const double wnDist = centerCoordinates[p][1];

      // Interpolate fluctuations based on the inner scaling
      double temperatureInn; Vec3 velocityInn; VecNSp MolarFracsInn;
      interpAll(temperatureInn, velocityInn, MolarFracsInn,
                temperature_recycle, velocity_recycle, MolarFracs_recycle,
                avg_y, FI_xloc, FI_iloc, FIdata,
                p, wnDist*yInnFact, uInnFact);

      // Interpolate fluctuations based on the inner scaling
      double temperatureOut; Vec3 velocityOut; VecNSp MolarFracsOut;
      interpAll(temperatureOut, velocityOut, MolarFracsOut,
                temperature_recycle, velocity_recycle, MolarFracs_recycle,
                avg_y, FI_xloc, FI_iloc, FIdata,
                p, wnDist*yOutFact, uOutFact);

      // Blend the results, multiply by free-stream dumping, and add mean profiles
      const double etaInl = centerCoordinates[p][1]*idelta99Inl;
      const double w = weightf(etaInl);
      const double damp = bernardinidamp(etaInl);
      t = interp(temperatureOut, temperatureInn, w)*damp + temperature_profile[p];
      __UNROLL__
      for (int i=0; i<3; i++)
         v[i] = interp(velocityOut[i], velocityInn[i], w)*damp + velocity_profile[p][i];
      __UNROLL__
      for (int i=0; i<nSpec; i++)
         Xi[i] = interp(MolarFracsOut[i], MolarFracsInn[i], w)*damp + MolarFracs_profile[p][i] ;
   }

   __CUDA_H__
   static inline double setPressure(const AccessorRO<VecNEq, 3> &Conserved,
                                    const double temperature,
                                    const VecNSp &MolarFracs,
                                    const Point<3> &p,
                                    const Mix &mix) {
      VecNSp rhoYi;
      __UNROLL__
      for (int i=0; i<nSpec; i++) rhoYi[i] = Conserved[p][i];
      const double rho = mix.GetRhoFromRhoYi(rhoYi);
      const double MixW = mix.GetMolarWeightFromXi(MolarFracs);
      return mix.GetPFromRhoAndT(rho, MixW, temperature);
   }
public:
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

#if (defined(ELECTRIC_FIELD) && (nIons > 0))
//-----------------------------------------------------------------------------
// TASK THAT CORRRECTS THE BOUNDARY CONDITIONS FOR CHARGED SPECIES
//-----------------------------------------------------------------------------

template<direction dir, side s>
class CorrectIonsBCTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    BC;
      LogicalRegion    BCst;
      Mix mix;
      FieldID   BC_fields [FID_last - 101];
      FieldID BCst_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

#endif

#endif // __PROMETEO_BC_HPP__
