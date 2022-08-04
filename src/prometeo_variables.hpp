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

#include "my_array.hpp"
#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_variables.h"
#include "prometeo_metric.inl"

//-----------------------------------------------------------------------------
// TASK THAT COMPUTES THE MIXTURE PROPERTIES
//-----------------------------------------------------------------------------

class UpdatePropertiesFromPrimitiveTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   __CUDA_H__
   static inline void UpdateProperties(const AccessorRO<double, 3> &pressure,
                                       const AccessorRO<double, 3> &temperature,
                                       const AccessorRO<VecNSp, 3> &MolarFracs,
                                       const AccessorRO<  Vec3, 3> &velocity,
                                       const AccessorWO<VecNSp, 3> &MassFracs,
                                       const AccessorWO<double, 3> &rho,
                                       const AccessorWO<double, 3> &mu,
                                       const AccessorWO<double, 3> &lam,
                                       const AccessorWO<VecNSp, 3> &Di,
                                       const AccessorWO<double, 3> &SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                       const AccessorWO<VecNIo, 3> &Ki,
#endif
                                       const Point<3> &p,
                                       const Mix &mix) {

      const double MixW = mix.GetMolarWeightFromXi(MolarFracs[p]);
      rho[p] = mix.GetRho(pressure[p], temperature[p], MixW);
      mu[p] = mix.GetViscosity(temperature[p], MolarFracs[p]);
      lam[p] = mix.GetHeatConductivity(temperature[p], MolarFracs[p]);
      mix.GetDiffusivity(Di[p], pressure[p], temperature[p], MixW, MolarFracs[p]);
      mix.GetMassFractions(MassFracs[p], MixW, MolarFracs[p]);
      const double gamma = mix.GetGamma(temperature[p], MixW, MassFracs[p]);
      SoS[p] = mix.GetSpeedOfSound(temperature[p], gamma, MixW);
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
      mix.GetElectricMobility(Ki[p], pressure[p], temperature[p], MolarFracs[p]);
#endif
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
// TASK THAT COMPUTES THE CONSERVED VARIABLES FROM THE PRIMITIVE VARIABLES
//-----------------------------------------------------------------------------

class UpdateConservedFromPrimitiveTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   __CUDA_H__
   static inline void UpdateConserved(const AccessorRO<VecNSp, 3> &MassFracs,
                                      const AccessorRO<double, 3> &temperature,
                                      const AccessorRO<  Vec3, 3> &velocity,
                                      const AccessorRO<double, 3> &rho,
                                      const AccessorWO<VecNEq, 3> &Conserved,
                                      const Point<3> &p,
                                      const Mix &mix) {
      VecNSp rhoYi; mix.GetRhoYiFromYi(rhoYi, rho[p], MassFracs[p]);
      __UNROLL__
      for (int i=0; i<nSpec; i++) Conserved[p][i] = rhoYi[i];
      __UNROLL__
      for (int i=0; i<3; i++) Conserved[p][i+irU] = rho[p]*velocity[p][i];
      Conserved[p][irE] = rho[p]*(0.5*velocity[p].mod2() +
                                  mix.GetInternalEnergy(temperature[p], MassFracs[p]));
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
// TASK THAT COMPUTES THE PRIMITIVE VARIABLES FROM THE CONSERVED VARIABLES
//-----------------------------------------------------------------------------

class UpdatePrimitiveFromConservedTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   __CUDA_H__
   static inline void UpdatePrimitive(const AccessorRO<VecNEq, 3> &Conserved,
                                      const AccessorRW<double, 3> &temperature,
                                      const AccessorWO<double, 3> &pressure,
                                      const AccessorWO<VecNSp, 3> &MolarFracs,
                                      const AccessorWO<  Vec3, 3> &velocity,
                                      const Point<3> &p,
                                      const Mix &mix) {
      VecNSp rhoYi;
      __UNROLL__
      for (int i=0; i<nSpec; i++) rhoYi[i] = Conserved[p][i];
      const double rho = mix.GetRhoFromRhoYi(rhoYi);
      VecNSp Yi; mix.GetYi(Yi, rho, rhoYi);
      mix.ClipYi(Yi);
      assert(mix.CheckMixture(Yi));
      const double MixW = mix.GetMolarWeightFromYi(Yi);
      mix.GetMolarFractions(MolarFracs[p], MixW, Yi);
      const double rhoInv = 1.0/rho;
      __UNROLL__
      for (int i=0; i<3; i++) velocity[p][i] = Conserved[p][i+irU]*rhoInv;
      const double kineticEnergy = 0.5*velocity[p].mod2();
      const double InternalEnergy = Conserved[p][irE]*rhoInv - kineticEnergy;
      temperature[p] = mix.GetTFromInternalEnergy(InternalEnergy, temperature[p], Yi);
      pressure[p] = mix.GetPFromRhoAndT(rho, MixW, temperature[p]);
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

#endif // __PROMETEO_VARIABLES_HPP__
