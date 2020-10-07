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

#ifndef __PROMETEO_VARIABLES_HPP__
#define __PROMETEO_VARIABLES_HPP__

#include "legion.h"

using namespace Legion;

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "prometeo_types.h"
#include "task_helper.h"
#include "prometeo_variables.h"
#include "prometeo_metric.inl"

//-----------------------------------------------------------------------------
// LOAD THE EQUATION OF STATE FUNCTIONS
//-----------------------------------------------------------------------------
#define QUOTEME(M)       #M
#define INCLUDE_FILE(M)  QUOTEME(M.hpp)
#include INCLUDE_FILE( EOS )
#undef QUOTEME
#undef INCLUDE_FILE

//-----------------------------------------------------------------------------
// TASK THAT COMPUTES THE MIXTURE PROPERTIES
//-----------------------------------------------------------------------------

class UpdatePropertiesFromPrimitiveTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      LogicalRegion ModCells;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
      FieldID ModCells_fields [FID_last - 101];
   };
public:
   __CUDA_H__
   static inline void UpdateProperties(const AccessorRO<double, 3> pressure,
                                       const AccessorRO<double, 3> temperature,
                                       const AccessorRO<VecNSp, 3> MolarFracs,
                                       const AccessorRO<  Vec3, 3> velocity,
                                       const AccessorWO<VecNSp, 3> MassFracs,
                                       const AccessorWO<double, 3> rho,
                                       const AccessorWO<double, 3> mu,
                                       const AccessorWO<double, 3> lam,
                                       const AccessorWO<VecNSp, 3> Di,
                                       const AccessorWO<double, 3> SoS,
                                       const Point<3> p,
                                       const Mix &mix) {

      const double MixW = GetMolarWeightFromXi(MolarFracs[p].v, mix);
      rho[p] = GetRho(pressure[p], temperature[p], MixW, mix);
      mu[p] = GetViscosity(temperature[p], MolarFracs[p].v, mix);
      lam[p] = GetHeatConductivity(temperature[p], MolarFracs[p].v, mix);
      GetDiffusivity(Di[p].v, pressure[p], temperature[p], MixW, MolarFracs[p].v, mix);
      GetMassFractions(MassFracs[p].v, MixW, MolarFracs[p].v, mix);
      double gamma = GetGamma(temperature[p], MixW, MassFracs[p].v, mix);
      SoS[p] = GetSpeedOfSound(temperature[p], gamma, MixW, mix);
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
// TASK THAT COMPUTES THE VELOCITY GRADIENT
//-----------------------------------------------------------------------------

class GetVelocityGradientsTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Ghost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      FieldID Ghost_fields [FID_last - 101];
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   template<direction dir>
   __CUDA_H__
   static inline void computeDerivatives(const AccessorWO<Vec3, 3> vGrad,
                                         const AccessorRO<Vec3, 3> velocity,
                                         const Point<3> p,
                                         const int nType,
                                         const double  m,
                                         const coord_t dsize,
                                         const Rect<3> bounds) {
      // Compute stencil points
      const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));

      __UNROLL__
      for (int i=0; i<3; i++)
         vGrad[p][i] = getDeriv(nType, velocity[pM1][i], velocity[p  ][i], velocity[pP1][i], m);
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

//-----------------------------------------------------------------------------
// TASK THAT COMPUTES THE VELOCITY GRADIENT
//-----------------------------------------------------------------------------

class GetTemperatureGradientTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Ghost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      FieldID Ghost_fields [FID_last - 101];
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   template<direction dir>
   __CUDA_H__
   static inline double computeDerivative(const AccessorRO<double, 3> temperature,
                                          const Point<3> p,
                                          const int nType,
                                          const double  m,
                                          const coord_t dsize,
                                          const Rect<3> bounds) {
      // Compute stencil points
      const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));
      return getDeriv(nType, temperature[pM1], temperature[p], temperature[pP1], m);
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

#endif // __PROMETEO_VARIABLES_HPP__
