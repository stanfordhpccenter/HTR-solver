// Copyright (c) "2020, by Centre Européen de Recherche et de Formation Avancée en Calcul Scientifiq
//               Developer: Mario Di Renzo
//               Affiliation: Centre Européen de Recherche et de Formation Avancée en Calcul Scientifique
//               URL: https://cerfacs.fr
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

#ifndef __PROMETEO_ELECTRICFIELD_HPP__
#define __PROMETEO_ELECTRICFIELD_HPP__

#include "legion.h"

using namespace Legion;

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_electricField.h"

//-----------------------------------------------------------------------------
// TASK THAT COMPUTES THE ELECTRIC FIELD
//-----------------------------------------------------------------------------
class GetElectricFieldTask {
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

#if (nIons > 0)
//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE RHS USING THE ION DRIFT FLUXES
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingIonDriftFluxTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion EulerGhost;
      LogicalRegion DiffGhost;
      LogicalRegion FluxGhost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID EulerGhost_fields[FID_last - 101];
      FieldID DiffGhost_fields [FID_last - 101];
      FieldID DivgGhost_fields [FID_last - 101];
      FieldID Fluid_fields     [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
private:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m_e;
private:
   __CUDA_H__
   static void GetIonDriftFlux(VecNEq &Flux,
                               const AccessorRO<VecNEq, 3>       &rhoYi,
                               const AccessorRO<VecNSp, 3>          &Yi,
                               const AccessorRO<VecNIo, 3>          &Ki,
                               const AccessorRO<  Vec3, 3>      &eField,
                               const Point<3> &p,
                               const int nType,
                               const Mix &mix,
                               const coord_t size,
                               const Rect<3> &bounds);
public:
   __CUDA_H__
   static void updateRHSSpan(const AccessorRW<VecNEq, 3> &Conserved_t,
                             const AccessorRO<double, 3> &m_e,
                             const AccessorRO<   int, 3> &nType,
                             const AccessorRO<VecNEq, 3> &rhoYi,
                             const AccessorRO<VecNSp, 3> &Yi,
                             const AccessorRO<VecNIo, 3> &Ki,
                             const AccessorRO<  Vec3, 3> &eField,
                             const coord_t firstIndex,
                             const coord_t lastIndex,
                             const int x,
                             const int y,
                             const int z,
                             const Rect<3> &Flux_bounds,
                             const Rect<3> &Fluid_bounds,
                             const Mix &mix);

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
#endif

//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE RHS USING THE SOURCE TERMS DUE TO ION WIND
//-----------------------------------------------------------------------------

class AddIonWindSourcesTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    GradGhost;
      LogicalRegion    Fluid;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID GradGhost_fields  [FID_last - 101];
      FieldID Fluid_fields      [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   __CUDA_H__
   static inline void addIonWindSources(VecNEq &RHS,
                            const AccessorRO<double, 3>        &rho,
                            const AccessorRO<VecNSp, 3>         &Di,
#if (nIons > 0)
                            const AccessorRO<VecNIo, 3>         &Ki,
#endif
                            const AccessorRO<  Vec3, 3>   &velocity,
                            const AccessorRO<  Vec3, 3>     &eField,
                            const AccessorRO<VecNSp, 3> &MolarFracs,
                            const AccessorRO<   int, 3>    &nType_x,
                            const AccessorRO<   int, 3>    &nType_y,
                            const AccessorRO<   int, 3>    &nType_z,
                            const AccessorRO<double, 3>        &m_x,
                            const AccessorRO<double, 3>        &m_y,
                            const AccessorRO<double, 3>        &m_z,
                            const Point<3> &p,
                            const  Rect<3> &bounds,
                            const      Mix &mix);
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

#endif // __PROMETEO_ELECTRICFIELD_HPP__
