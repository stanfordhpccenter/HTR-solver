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

#ifndef __PROMETEO_RHS_HPP__
#define __PROMETEO_RHS_HPP__

#include "legion.h"

using namespace Legion;

#include "math.h"

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "math_utils.hpp"
#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_bc_types.h"
#include "prometeo_rhs.h"

//-----------------------------------------------------------------------------
// TASKS THAT UPDATE THE RHS USING THE EULER FLUXES
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingEulerFluxUtils {
protected:
   // Roe averages at the intercell location
   struct RoeAveragesStruct {
      double a;
      double e;
      double H;
      double a2;
      double rho;
        Vec3 velocity;
      VecNSp Yi;
      double dpde;
      VecNSp dpdrhoi;
   };

   // Computes KG summations
   __CUDA_H__
   static inline void ComputeKGSums(double * Sums,
                          const AccessorRO<VecNEq, 3> &Conserved,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &pressure,
                          const Point<3> &p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> &bounds);

   // Computes KG reconstructions
   __CUDA_H__
   static inline void KGFluxReconstruction(VecNEq &Flux,
                          const double * Sums,
                          const AccessorRO<VecNEq, 3> &Conserved,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &pressure,
                          const Point<3> &p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> &bounds);

   // Computes KG reconstructions
   __CUDA_H__
   static inline void KGFluxReconstruction(VecNEq &Flux,
                          const AccessorRO<VecNEq, 3> &Conserved,
                          const AccessorRO<double, 3> &rho,
                          const AccessorRO<VecNSp, 3> &MassFracs,
                          const AccessorRO<  Vec3, 3> &velocity,
                          const AccessorRO<double, 3> &pressure,
                          const Point<3> &p,
                          const int nType,
                          const coord_t dsize,
                          const Rect<3> &bounds);

   // Computes Roe averages across the Rieman problem
   __CUDA_H__
   static inline void ComputeRoeAverages(
                           RoeAveragesStruct  &avgs, const Mix &mix,
                           const VecNEq &ConservedL, const VecNEq &ConservedR,
                           const VecNSp        &YiL, const VecNSp        &YiR,
                           const double   pressureL, const double   pressureR,
                           const   Vec3  &velocityL, const   Vec3  &velocityR,
                           const double        rhoL, const double        rhoR);
   __CUDA_H__
   static inline void computeLeftEigenvectors( MyMatrix<double, nEq, nEq> &L, const RoeAveragesStruct &avgs);
   __CUDA_H__
   static inline void computeRightEigenvectors(MyMatrix<double, nEq, nEq> &K, const RoeAveragesStruct &avgs);

   // Projects the state vector q in the characteristic space from the physiscal space
   __CUDA_H__
   static inline void projectToCharacteristicSpace(VecNEq &r, const VecNEq &q, const RoeAveragesStruct &avgs);

   // Projects the state vector q from the characteristic space to the physiscal space
   __CUDA_H__
   static inline void projectFromCharacteristicSpace(VecNEq &r, const VecNEq &q, const RoeAveragesStruct &avgs);

   // Projects the state vector q from the characteristic space to the physiscal space for one species
   __CUDA_H__
   static inline double projectFromCharacteristicSpace(const int i, const VecNEq &q, const RoeAveragesStruct &avgs);

   // Performs the flux splitting
   __CUDA_H__
   static inline void getPlusMinusFlux(VecNEq &FluxP, VecNEq &FluxM,
                            const RoeAveragesStruct  &avgs,
                            const VecNEq &Conserved,
                            const double velocity,
                            const double pressure,
                            const double Lam1,
                            const double Lam,
                            const double LamN);

   // Performs the flux reconstruction using the Lax-Friederics splitting
   template<class Op>
   __CUDA_H__
   static inline void FluxReconstruction(VecNEq &Flux,
                            const AccessorRO<VecNEq, 3> &Conserved,
#if (nSpec > 1)
                            const AccessorRO<VecNEq, 3> &Conserved_old,
#endif
                            const AccessorRO<double, 3> &SoS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const Point<3> &p,
                            const int      nType,
#if (nSpec > 1)
                            const double   RK_coeffs0,
                            const double   RK_coeffs1,
                            const double   lim_f,
#endif
                            const Mix      &mix,
                            const coord_t  dsize,
                            const Rect<3>  &bounds);

};

template<direction dir>
class UpdateUsingHybridEulerFluxTask : private UpdateUsingEulerFluxUtils<dir> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion EulerGhost;
      LogicalRegion SensorGhost;
      LogicalRegion FluxGhost;
      LogicalRegion Fluid;
      double RK_coeffs[2];
      double deltaTime;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID EulerGhost_fields [FID_last - 101];
      FieldID SensorGhost_fields[FID_last - 101];
      FieldID FluxGhost_fields  [FID_last - 101];
      FieldID Fluid_fields      [FID_last - 101];
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
   static const FieldID FID_shockSensor;
public:
   // Direction dependent functions
   __CUDA_H__
   static inline void updateRHSSpan(double *KGSum,
                                    const AccessorRW<VecNEq, 3> &Conserved_t,
                                    const AccessorRO<double, 3> &m_e,
                                    const AccessorRO<   int, 3> &nType,
                                    const AccessorRO<  bool, 3> &shockSensor,
                                    const AccessorRO<VecNEq, 3> &Conserved,
#if (nSpec > 1)
                                    const AccessorRO<VecNEq, 3> &Conserved_old,
#endif
                                    const AccessorRO<double, 3> &rho,
                                    const AccessorRO<double, 3> &SoS,
                                    const AccessorRO<VecNSp, 3> &MassFracs,
                                    const AccessorRO<  Vec3, 3> &velocity,
                                    const AccessorRO<double, 3> &pressure,
#if (nSpec > 1)
                                    const double  RK_coeffs0,
                                    const double  RK_coeffs1,
                                    const double  deltaTime,
#endif
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

template<direction dir, class Op>
class UpdateUsingTENOEulerFluxTask: private UpdateUsingEulerFluxUtils<dir> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion EulerGhost;
      LogicalRegion DiffGhost;
      LogicalRegion FluxGhost;
      LogicalRegion Fluid;
      double RK_coeffs[2];
      double deltaTime;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID EulerGhost_fields [FID_last - 101];
      FieldID DiffGhost_fields  [FID_last - 101];
      FieldID FluxGhost_fields  [FID_last - 101];
      FieldID Fluid_fields      [FID_last - 101];
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
public:
   // Direction dependent functions
   __CUDA_H__
   static inline void updateRHSSpan(const AccessorRW<VecNEq, 3> &Conserved_t,
                                    const AccessorRO<double, 3> &m_e,
                                    const AccessorRO<   int, 3> &nType,
                                    const AccessorRO<VecNEq, 3> &Conserved,
#if (nSpec > 1)
                                    const AccessorRO<VecNEq, 3> &Conserved_old,
#endif
                                    const AccessorRO<double, 3> &rho,
                                    const AccessorRO<double, 3> &SoS,
                                    const AccessorRO<VecNSp, 3> &MassFracs,
                                    const AccessorRO<  Vec3, 3> &velocity,
                                    const AccessorRO<double, 3> &pressure,
#if (nSpec > 1)
                                    const double  RK_coeffs0,
                                    const double  RK_coeffs1,
                                    const double  deltaTime,
#endif
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

template<direction dir>
class UpdateUsingSkewSymmetricEulerFluxTask : private UpdateUsingEulerFluxUtils<dir> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion EulerGhost;
      LogicalRegion FluxGhost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID EulerGhost_fields [FID_last - 101];
      FieldID FluxGhost_fields  [FID_last - 101];
      FieldID Fluid_fields      [FID_last - 101];
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
public:
   // Direction dependent functions
   __CUDA_H__
   static inline void updateRHSSpan(double *KGSum,
                                    const AccessorRW<VecNEq, 3> &Conserved_t,
                                    const AccessorRO<double, 3> &m_e,
                                    const AccessorRO<   int, 3> &nType,
                                    const AccessorRO<VecNEq, 3> &Conserved,
                                    const AccessorRO<double, 3> &rho,
                                    const AccessorRO<VecNSp, 3> &MassFracs,
                                    const AccessorRO<  Vec3, 3> &velocity,
                                    const AccessorRO<double, 3> &pressure,
                                    const coord_t firstIndex,
                                    const coord_t lastIndex,
                                    const int x,
                                    const int y,
                                    const int z,
                                    const Rect<3> &Flux_bounds,
                                    const Rect<3> &Fluid_bounds);
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
// TASK THAT UPDATES THE RHS USING THE DIFFUSION FLUXES
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingDiffusionFluxUtils {
protected:
   __CUDA_H__
   static Vec3 GetSigma(const int nType, const double m_s,
                        const AccessorRO<double, 3>       &mu,
                        const AccessorRO<  Vec3, 3> &velocity,
                        const Vec3 *vGrad1,
                        const Vec3 *vGrad2,
                        const Point<3> &p, const Point<3> &pp1);
};

template<direction dir>
class UpdateUsingDiffusionFluxTask : private UpdateUsingDiffusionFluxUtils<dir> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion DiffGhost;
      LogicalRegion DiffGradGhost;
      LogicalRegion FluxGhost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID DiffGhost_fields    [FID_last - 101];
      FieldID DiffGradGhost_fields[FID_last - 101];
      FieldID DivgGhost_fields    [FID_last - 101];
      FieldID Fluid_fields        [FID_last - 101];
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
   static const FieldID FID_nType1;
   static const FieldID FID_nType2;
   static const FieldID FID_m_s;
   static const FieldID FID_m_d;
   static const FieldID FID_m_d1;
   static const FieldID FID_m_d2;
private:
   __CUDA_H__
   static void GetDiffusionFlux(VecNEq &Flux, const int nType, const double m_s, const Mix &mix,
                                const AccessorRO<double, 3> &rho,
                                const AccessorRO<double, 3> &mu,
                                const AccessorRO<double, 3> &lam,
                                const AccessorRO<VecNSp, 3> &Di,
                                const AccessorRO<double, 3> &temperature,
                                const AccessorRO<  Vec3, 3> &velocity,
                                const AccessorRO<VecNSp, 3> &Xi,
                                const AccessorRO<VecNEq, 3> &rhoYi,
                                const AccessorRO<   int, 3> &nType1,
                                const AccessorRO<   int, 3> &nType2,
                                const AccessorRO<double, 3> &m_d1,
                                const AccessorRO<double, 3> &m_d2,
                                Vec3 *vGrad1,
                                Vec3 *vGrad2,
                                const Point<3> &p,
                                const coord_t size,
                                const Rect<3> &bounds);
public:
   __CUDA_H__
   static void updateRHSSpan(const AccessorRW<VecNEq, 3> &Conserved_t,
                             const AccessorRO<double, 3> &m_s,
                             const AccessorRO<double, 3> &m_d,
                             const AccessorRO<double, 3> &m_d1,
                             const AccessorRO<double, 3> &m_d2,
                             const AccessorRO<   int, 3> &nType,
                             const AccessorRO<   int, 3> &nType1,
                             const AccessorRO<   int, 3> &nType2,
                             const AccessorRO<double, 3> &rho,
                             const AccessorRO<double, 3> &mu,
                             const AccessorRO<double, 3> &lam,
                             const AccessorRO<VecNSp, 3> &Di,
                             const AccessorRO<double, 3> &temperature,
                             const AccessorRO<  Vec3, 3> &velocity,
                             const AccessorRO<VecNSp, 3> &Xi,
                             const AccessorRO<VecNEq, 3> &rhoYi,
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

//-----------------------------------------------------------------------------
// TASKS THAT UPDATE THE RHS FOR AN NSCBC INFLOW
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingFluxNSCBCInflowMinusSideTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m_d;
   static const FieldID FID_vGrad;
   // Direction dependent functions
   __CUDA_H__
   static inline void addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<Vec3  , 3> &velocity,
                            const   double SoS,
                            const   double rho,
                            const   double T,
                            const     Vec3 &dudt,
                            const   double dTdt,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
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

template<direction dir>
class UpdateUsingFluxNSCBCInflowPlusSideTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m_d;
   // Direction dependent functions
   __CUDA_H__
   static inline void addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<Vec3  , 3> &velocity,
                            const   double SoS,
                            const   double rho,
                            const   double T,
                            const     Vec3 &dudt,
                            const   double dTdt,
                            const Point<3> &p,
                            const      int nType,
                            const   double m,
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

//-----------------------------------------------------------------------------
// TASKS THAT UPDATE THE RHS FOR AN NSCBC OUTFLOW
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingFluxNSCBCOutflowMinusSideTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Ghost;
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Rect<3> Fluid_bounds;
      Mix     mix;
      double  MaxMach;
      double  LengthScale;
      double  PInf;
      FieldID Ghost_fields [FID_last - 101];
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType_N;
   static const FieldID FID_nType_T1;
   static const FieldID FID_nType_T2;
   static const FieldID FID_m_d_N;
   static const FieldID FID_m_d_T1;
   static const FieldID FID_m_d_T2;
   // Direction dependent functions
   __CUDA_H__
   static inline void addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<   int, 3> &nType_N,
                            const AccessorRO<   int, 3> &nType_T1,
                            const AccessorRO<   int, 3> &nType_T2,
                            const AccessorRO<double, 3> &m_d_N,
                            const AccessorRO<double, 3> &m_d_T1,
                            const AccessorRO<double, 3> &m_d_T2,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<double, 3> &mu,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const   double SoS,
                            const   double T,
                            const   VecNEq &Conserved,
                            const Point<3> &p,
                            const  Rect<3> &bounds,
                            const   double MaxMach,
                            const   double LengthScale,
                            const   double PInf,
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

template<direction dir>
class UpdateUsingFluxNSCBCOutflowPlusSideTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Ghost;
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Rect<3> Fluid_bounds;
      Mix     mix;
      double  MaxMach;
      double  LengthScale;
      double  PInf;
      FieldID Ghost_fields [FID_last - 101];
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType_N;
   static const FieldID FID_nType_T1;
   static const FieldID FID_nType_T2;
   static const FieldID FID_m_d_N;
   static const FieldID FID_m_d_T1;
   static const FieldID FID_m_d_T2;
   // Direction dependent functions
   __CUDA_H__
   static inline void addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<   int, 3> &nType_N,
                            const AccessorRO<   int, 3> &nType_T1,
                            const AccessorRO<   int, 3> &nType_T2,
                            const AccessorRO<double, 3> &m_d_N,
                            const AccessorRO<double, 3> &m_d_T1,
                            const AccessorRO<double, 3> &m_d_T2,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<double, 3> &mu,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const   double SoS,
                            const   double T,
                            const   VecNEq &Conserved,
                            const Point<3> &p,
                            const  Rect<3> &bounds,
                            const   double MaxMach,
                            const   double LengthScale,
                            const   double PInf,
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

//-----------------------------------------------------------------------------
// TASKS THAT UPDATE THE RHS FOR AN NSCBC FARFIELD
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingFluxNSCBCFarFieldMinusSideTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Mix     mix;
      double  MaxMach;
      double  LengthScale;
      double  PInf;
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m_d;
   // Direction dependent functions
   __CUDA_H__
   static inline void addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const      int &nType,
                            const   double &m_d,
                            const   double &SoS,
                            const   double &temperature,
                            const   VecNEq &Conserved,
                            const   double &TInf,
                            const     Vec3 &vInf,
                            const   VecNSp &XiInf,
                            const   double PInf,
                            const   double MaxMach,
                            const   double LengthScale,
                            const Point<3> &p,
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

template<direction dir>
class UpdateUsingFluxNSCBCFarFieldPlusSideTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Mix     mix;
      double  MaxMach;
      double  LengthScale;
      double  PInf;
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m_d;
   // Direction dependent functions
   __CUDA_H__
   static inline void addLODIfluxes(VecNEq &RHS,
                            const AccessorRO<double, 3> &rho,
                            const AccessorRO<VecNSp, 3> &MassFracs,
                            const AccessorRO<double, 3> &pressure,
                            const AccessorRO<  Vec3, 3> &velocity,
                            const      int &nType,
                            const   double &m_d,
                            const   double &SoS,
                            const   double &temperature,
                            const   VecNEq &Conserved,
                            const   double &TInf,
                            const     Vec3 &vInf,
                            const   VecNSp &XiInf,
                            const   double PInf,
                            const   double MaxMach,
                            const   double LengthScale,
                            const Point<3> &p,
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

//-----------------------------------------------------------------------------
// TASKS THAT UPDATE THE RHS FOR AN INCOMING SHOCK BC
//-----------------------------------------------------------------------------

template<direction dir>
class UpdateUsingFluxIncomingShockTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion    Fluid;
      LogicalRegion    BC;
      Mix     mix;
      double  MaxMach;
      double  LengthScale;
      IncomingShockParams params;
      FieldID Fluid_fields [FID_last - 101];
      FieldID    BC_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m_d;
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
// TASKS THAT UPDATE THE RHS USING TURBULENT FORCING
//-----------------------------------------------------------------------------

class CalculateAveragePDTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Ghost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      FieldID Ghost_fields      [FID_last - 101];
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
   static inline double CalculatePressureDilatation(const AccessorRO<   int, 3> &nType_x,
                                                    const AccessorRO<   int, 3> &nType_y,
                                                    const AccessorRO<   int, 3> &nType_z,
                                                    const AccessorRO<double, 3> &dcsi_d,
                                                    const AccessorRO<double, 3> &deta_d,
                                                    const AccessorRO<double, 3> &dzet_d,
                                                    const AccessorRO<double, 3> &pressure,
                                                    const AccessorRO<  Vec3, 3> &velocity,
                                                    const Point<3> &p,
                                                    const Rect<3>  &Fluid_bounds);
public:
   static double cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static DeferredValue<double> gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

template<direction dir>
class AddDissipationTask : private UpdateUsingDiffusionFluxUtils<dir> {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion DiffGhost;
      LogicalRegion DiffGradGhost;
      LogicalRegion FluxGhost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      Mix mix;
      FieldID DiffGhost_fields    [FID_last - 101];
      FieldID DiffGradGhost_fields[FID_last - 101];
      FieldID DivgGhost_fields    [FID_last - 101];
      FieldID Fluid_fields        [FID_last - 101];
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
   static const FieldID FID_nType1;
   static const FieldID FID_nType2;
   static const FieldID FID_m_s;
   static const FieldID FID_m_d1;
   static const FieldID FID_m_d2;
private:
   __CUDA_H__
   static double GetDiffusionFlux(const int nType, const double m_s, const Mix &mix,
                                  const AccessorRO<double, 3> &mu,
                                  const AccessorRO<  Vec3, 3> &velocity,
                                  const AccessorRO<   int, 3> &nType1,
                                  const AccessorRO<   int, 3> &nType2,
                                  const AccessorRO<double, 3> &m_d1,
                                  const AccessorRO<double, 3> &m_d2,
                                  Vec3 *vGrad1,
                                  Vec3 *vGrad2,
                                  const Point<3> &p,
                                  const coord_t size,
                                  const Rect<3> &bounds);
public:
   __CUDA_H__
   static double AddSpan(const AccessorRO<double, 3> &m_s,
                         const AccessorRO<double, 3> &m_d1,
                         const AccessorRO<double, 3> &m_d2,
                         const AccessorRO<   int, 3> &nType,
                         const AccessorRO<   int, 3> &nType1,
                         const AccessorRO<   int, 3> &nType2,
                         const AccessorRO<double, 3> &mu,
                         const AccessorRO<  Vec3, 3> &velocity,
                         const coord_t firstIndex,
                         const coord_t lastIndex,
                         const int x,
                         const int y,
                         const int z,
                         const Rect<3> &Flux_bounds,
                         const Rect<3> &Fluid_bounds,
                         const Mix &mix);

public:
   static double cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static DeferredValue<double> gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

#endif // __PROMETEO_RHS_HPP__
