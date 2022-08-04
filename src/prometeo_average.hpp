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
// UTILITY THAT RETURNS THE POINT IN THE AVERAGE REGION
//-----------------------------------------------------------------------------
template<direction dir, int N>
__CUDA_HD__
inline Legion::Point<N> getPAvg(const Legion::Point<3> &p, const Rect<N> r_Avg);
template<>
__CUDA_HD__
inline Legion::Point<2> getPAvg<Xdir, 2>(const Legion::Point<3> &p, const Rect<2> r_Avg) { return Point<2>{p.x, r_Avg.lo.y}; };
template<>
__CUDA_HD__
inline Legion::Point<2> getPAvg<Ydir, 2>(const Legion::Point<3> &p, const Rect<2> r_Avg) { return Point<2>{p.y, r_Avg.lo.y}; };
template<>
__CUDA_HD__
inline Legion::Point<2> getPAvg<Zdir, 2>(const Legion::Point<3> &p, const Rect<2> r_Avg) { return Point<2>{p.z, r_Avg.lo.y}; };
template<>
__CUDA_HD__
inline Legion::Point<3> getPAvg<Xdir, 3>(const Legion::Point<3> &p, const Rect<3> r_Avg) { return Point<3>{p.y, p.z, r_Avg.lo.z}; };
template<>
__CUDA_HD__
inline Legion::Point<3> getPAvg<Ydir, 3>(const Legion::Point<3> &p, const Rect<3> r_Avg) { return Point<3>{p.x, p.z, r_Avg.lo.z}; };
template<>
__CUDA_HD__
inline Legion::Point<3> getPAvg<Zdir, 3>(const Legion::Point<3> &p, const Rect<3> r_Avg) { return Point<3>{p.x, p.y, r_Avg.lo.z}; };

//-----------------------------------------------------------------------------
// UTILITY FUNCTIONS THAT COLLECT SPATIAL AVERAGES
//-----------------------------------------------------------------------------
template<int N>
class AverageUtils {
public:
   // Baseline utilities
   __CUDA_H__
   static inline double getWeight(
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const Point<3> &p,
                          const double deltaTime);

   __CUDA_H__
   static inline double getFavreWeight(
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &rho,
                          const Point<3> &p,
                          const double deltaTime);

   template<typename T>
   __CUDA_H__
   static inline void Avg(const T &f,
                          const AccessorSumRD<T, N> &avg,
                          const Point<N> &pA,
                          const double weight);

   template<typename T>
   __CUDA_H__
   static inline void Avg(const T &f,
                          const AccessorSumRD<T, N> &avg,
                          const AccessorSumRD<T, N> &rms,
                          const Point<N> &pA,
                          const double weight);

   template<typename T>
   __CUDA_H__
   static inline void Avg(const AccessorRO<T, 3> &f,
                          const AccessorSumRD<T, N> &avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   template<typename T>
   __CUDA_H__
   static inline void Avg(const AccessorRO<T, 3> &f,
                          const AccessorSumRD<T, N> &avg,
                          const AccessorSumRD<T, N> &rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   __CUDA_H__
   static inline void Avg(const AccessorRO<Vec3, 3> &f,
                          const AccessorSumRD<Vec3, N> &avg,
                          const AccessorSumRD<Vec3, N> &rms,
                          const AccessorSumRD<Vec3, N> &rey,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   __CUDA_H__
   static inline void Cor(const AccessorRO<double, 3> &s,
                          const AccessorRO<  Vec3, 3> &v,
                          const AccessorSumRD<Vec3, N> &cor,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   __CUDA_H__
   static inline void Cor(const AccessorRO<VecNSp, 3> &v1,
                          const AccessorRO<  Vec3, 3> &v2,
                          const AccessorSumRD<VecNSp, N> &cor0,
                          const AccessorSumRD<VecNSp, N> &cor1,
                          const AccessorSumRD<VecNSp, N> &cor2,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   // Position and average weight
   __CUDA_H__
   static inline void PositionAndWeight(
                          const AccessorRO<  Vec3, 3> &centerCoordinates,
                          const AccessorSumRD<double, N> &avg_weight,
                          const AccessorSumRD<  Vec3, N> &avg_centerCoordinates,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   // Properties averages
   __CUDA_H__
   static inline void CpEntAvg(
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorSumRD<double, N> &cp_avg,
                          const AccessorSumRD<double, N> &cp_favg,
                          const AccessorSumRD<double, N> &Ent_avg,
                          const AccessorSumRD<double, N> &Ent_favg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight,
                          const double fweight);

  // Averages of production rates
   __CUDA_H__
   static inline void ProdRatesAvg(
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_avg,
                          const AccessorSumRD<VecNSp, N> &ProductionRates_rms,
                          const AccessorSumRD<double, N> &HeatReleaseRate_avg,
                          const AccessorSumRD<double, N> &HeatReleaseRate_rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight);

   // Heat flux average
   __CUDA_H__
   static inline void HeatFluxAvg(
                          const AccessorRO<   int, 3> &nType_x,
                          const AccessorRO<   int, 3> &nType_y,
                          const AccessorRO<   int, 3> &nType_z,
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &lam,
                          const AccessorRO<VecNSp, 3> &Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                          const AccessorRO<VecNIo, 3> &Ki,
                          const AccessorRO<  Vec3, 3> &eField,
#endif
                          const AccessorSumRD<  Vec3, N> &q_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Rect<3> &Fluid_bounds,
                          const Mix &mix,
                          const double weight);

   // Averages the kinetic energy budget terms
   __CUDA_H__
   static inline void AvgKineticEnergyBudget(
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<  Vec3, N> &rhoUUv_avg,
                          const AccessorSumRD<  Vec3, N> &Up_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   __CUDA_H__
   static inline void AvgKineticEnergyBudget_Tau(
                          const AccessorRO<   int, 3> &nType_x,
                          const AccessorRO<   int, 3> &nType_y,
                          const AccessorRO<   int, 3> &nType_z,
                          const AccessorRO<double, 3> &dcsi_d,
                          const AccessorRO<double, 3> &deta_d,
                          const AccessorRO<double, 3> &dzet_d,
                          const AccessorRO<double, 3> &pressure,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorSumRD<TauMat, N> &tau_avg,
                          const AccessorSumRD<  Vec3, N> &utau_y_avg,
                          const AccessorSumRD<  Vec3, N> &tauGradU_avg,
                          const AccessorSumRD<  Vec3, N> &pGradU_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Rect<3> &Fluid_bounds,
                          const Mix &mix,
                          const double weight);

   // Averages dimensionless numbers
   __CUDA_H__
   static inline void PrEcAvg(
                          const AccessorRO<double, 3> &temperature,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<double, 3> &lam,
                          const AccessorSumRD<double, N> &Pr_avg,
                          const AccessorSumRD<double, N> &Pr_rms,
                          const AccessorSumRD<double, N> &Ec_avg,
                          const AccessorSumRD<double, N> &Ec_rms,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight);

   __CUDA_H__
   static inline void MaAvg(
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &SoS,
                          const AccessorSumRD<double, N> &Ma_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

   __CUDA_H__
   static inline void ScAvg(
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<double, 3> &mu,
                          const AccessorRO<VecNSp, 3> &Di,
                          const AccessorSumRD<VecNSp, N> &Sc_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const double weight);

#ifdef ELECTRIC_FIELD
   // Averages electric quantities
   __CUDA_H__
   static inline void ElectricChargeAvg(
                          const AccessorRO<VecNSp, 3> &MolarFracs,
                          const AccessorRO<double, 3> &rho,
                          const AccessorSumRD<double, N> &Crg_avg,
                          const Point<3> &p,
                          const Point<N> &pA,
                          const Mix &mix,
                          const double weight);
#endif
};

//-----------------------------------------------------------------------------
// TASK THAT COLLECTES TEMPORAL AND SPATIAL AVERAGES
//-----------------------------------------------------------------------------

template<direction dir, int N>
class AddAveragesTask : public AverageUtils<N> {
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
