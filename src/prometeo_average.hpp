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

#ifndef __PROMETEO_AVERAGE_HPP__
#define __PROMETEO_AVERAGE_HPP__

#include "legion.h"

using namespace Legion;

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_redop.inl"
#include "prometeo_average.h"
#include "prometeo_average_types.h"

typedef MySymMatrix<double, 3> TauMat;

//-----------------------------------------------------------------------------
// UTILITY FUNCTIONS THAT COLLECT SPATIAL AVERAGES
//-----------------------------------------------------------------------------

template<int N>
class AverageUtils {
public:
   // Averages the primitive variables
   __CUDA_H__
   static inline void AvgPrimitive(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<  Vec3, 3> &centerCoordinates,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorSumRD<double, N> &avg_weight,
                          const AccessorSumRD<  Vec3, N> &avg_centerCoordinates,
                          const AccessorSumRD<double, N> &pressure_avg,
                          const AccessorSumRD<double, N> &pressure_rms,
                          const AccessorSumRD<double, N> &temperature_avg,
                          const AccessorSumRD<double, N> &temperature_rms,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_avg,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_rms,
                          const AccessorSumRD<VecNSp, N> &MassFracs_avg,
                          const AccessorSumRD<VecNSp, N> &MassFracs_rms,
                          const AccessorSumRD<  Vec3, N> &velocity_avg,
                          const AccessorSumRD<  Vec3, N> &velocity_rms,
                          const AccessorSumRD<  Vec3, N> &velocity_rey,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double deltaTime);

   // Favre averages the primitive variables
   __CUDA_H__
   static inline void FavreAvgPrimitive(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorSumRD<double, N> &pressure_favg,
                          const AccessorSumRD<double, N> &pressure_frms,
                          const AccessorSumRD<double, N> &temperature_favg,
                          const AccessorSumRD<double, N> &temperature_frms,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_favg,
                          const AccessorSumRD<VecNSp, N> &MolarFracs_frms,
                          const AccessorSumRD<VecNSp, N> &MassFracs_favg,
                          const AccessorSumRD<VecNSp, N> &MassFracs_frms,
                          const AccessorSumRD<  Vec3, N> &velocity_favg,
                          const AccessorSumRD<  Vec3, N> &velocity_frms,
                          const AccessorSumRD<  Vec3, N> &velocity_frey,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double deltaTime);

   // Averages the properties
   __CUDA_H__
   static inline void AvgProperties(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &rho_avg,
                          const AccessorSumRD<double, N> &rho_rms,
                          const AccessorSumRD<double, N> &mu_avg,
                          const AccessorSumRD<double, N> &lam_avg,
                          const AccessorSumRD<VecNSp, N> &Di_avg,
                          const AccessorSumRD<double, N> &SoS_avg,
                          const AccessorSumRD<double, N> &cp_avg,
                          const AccessorSumRD<double, N> &Ent_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime);

   // Favre averages the properties
   __CUDA_H__
   static inline void FavreAvgProperties(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &mu_favg,
                          const AccessorSumRD<double, N> &lam_favg,
                          const AccessorSumRD<VecNSp, N> &Di_favg,
                          const AccessorSumRD<double, N> &SoS_favg,
                          const AccessorSumRD<double, N> &cp_favg,
                          const AccessorSumRD<double, N> &Ent_favg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime);

   // Averages the kinetic energy budget terms
   __CUDA_H__
   static inline void AvgKineticEnergyBudget(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<  Vec3, 3> &vGradX,
                          const AccessorRO<  Vec3, 3> &vGradY,
                          const AccessorRO<  Vec3, 3> &vGradZ,
                          const AccessorSumRD<  Vec3, N> &rhoUUv_avg,
                          const AccessorSumRD<  Vec3, N> &Up_avg,
                          const AccessorSumRD<TauMat, N> &tau_avg,
                          const AccessorSumRD<  Vec3, N> &utau_y_avg,
                          const AccessorSumRD<  Vec3, N> &tauGradU_avg,
                          const AccessorSumRD<  Vec3, N> &pGradU_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime);

   // Averages the fluxes and production rates
   __CUDA_H__
   static inline void AvgFluxes_ProdRates(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<   int, 3> &nType_x,
                          const AccessorRO<   int, 3> &nType_y,
                          const AccessorRO<   int, 3> &nType_z,
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                          const AccessorRO<VecNIo, 3> &Ki,
                          const AccessorRO<  Vec3, 3> &eField,
#endif
                          const AccessorSumRD<  Vec3, N> &q_avg,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_avg,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_rms,
                          const AccessorSumRD<double, N> &HeatReleaseRate_avg,
                          const AccessorSumRD<double, N> &HeatReleaseRate_rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Rect<3> &Fluid_bounds,
                          const Mix &mix,
                          const double deltaTime);

   // Averages dimensionless numbers
   __CUDA_H__
   static inline void AvgDimensionlessNumbers(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &Pr_avg,
                          const AccessorSumRD<double, N> &Pr_rms,
                          const AccessorSumRD<double, N> &Ec_avg,
                          const AccessorSumRD<double, N> &Ec_rms,
                          const AccessorSumRD<double, N> &Ma_avg,
                          const AccessorSumRD<VecNSp, N> &Sc_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime);

   // Averages correlations
   __CUDA_H__
   static inline void AvgCorrelations(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<  Vec3, N> &uT_avg,
                          const AccessorSumRD<  Vec3, N> &uT_favg,
                          const AccessorSumRD<VecNSp, N> &uYi_avg,
                          const AccessorSumRD<VecNSp, N> &vYi_avg,
                          const AccessorSumRD<VecNSp, N> &wYi_avg,
                          const AccessorSumRD<VecNSp, N> &uYi_favg,
                          const AccessorSumRD<VecNSp, N> &vYi_favg,
                          const AccessorSumRD<VecNSp, N> &wYi_favg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double deltaTime);

#ifdef ELECTRIC_FIELD
   // Averages electric quantities
   __CUDA_H__
   static inline void AvgElectricQuantities(
                          const AccessorRO<  Vec3, 3> &cellWidth,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &ePot,
                          const AccessorSumRD<double, N> &ePot_avg,
                          const AccessorSumRD<double, N> &Crg_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double deltaTime);
#endif

};

//-----------------------------------------------------------------------------
// TASK THAT COLLECTES TEMPORAL AND 2D SPATIAL AVERAGES
//-----------------------------------------------------------------------------

template<direction dir>
class Add2DAveragesTask : public AverageUtils<2> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Ghost;
      LogicalRegion Fluid;
      LogicalRegion Averages;
      Mix mix;
      Rect<3> Fluid_bounds;
      double Integrator_deltaTime;
      FieldID Ghost_fields      [FID_last - 101];
      FieldID Fluid_fields      [FID_last - 101];
      FieldID Averages_fields   [AVE_FID_last - 101];
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
// TASK THAT COLLECTES TEMPORAL AND 1D SPATIAL AVERAGES
//-----------------------------------------------------------------------------

template<direction dir>
class Add1DAveragesTask : public AverageUtils<3> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Ghost;
      LogicalRegion Fluid;
      LogicalRegion Averages;
      Mix mix;
      Rect<3> Fluid_bounds;
      double Integrator_deltaTime;
      FieldID Ghost_fields      [FID_last - 101];
      FieldID Fluid_fields      [FID_last - 101];
      FieldID Averages_fields   [AVE_FID_last - 101];
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

// Physical regions indices
#define iDouble 2
#define iVec3   3
#define iVecNSp 4
#define iVec6   5

#endif // __PROMETEO_AVERAGE_HPP__
