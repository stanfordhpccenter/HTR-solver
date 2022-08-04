// Copyright (c) "2019, by Stanford University
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

#include "prometeo_rhs.hpp"
#include "prometeo_rhs.inl"

#ifdef REALM_USE_OPENMP
#include <omp.h>
#endif

// CPU Implementation of UpdateUsingHybridEulerFlux
template<direction dir>
void UpdateUsingHybridEulerFluxTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 1);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved    (regions[0], FID_Conserved);
#if (nSpec > 1)
   const AccessorRO<VecNEq, 3> acc_Conserved_old(regions[0], FID_Conserved_old);
#endif
   const AccessorRO<double, 3> acc_SoS          (regions[0], FID_SoS);
   const AccessorRO<double, 3> acc_rho          (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs    (regions[0], FID_MassFracs);
   const AccessorRO<  Vec3, 3> acc_velocity     (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure     (regions[0], FID_pressure);

   // Accessors for shock sensor
   const AccessorRO<  bool, 3> acc_shockSensor  (regions[1], FID_shockSensor);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m            (regions[1], FID_m_e);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType        (regions[2], FID_nType);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t  (regions[3], FID_Conserved_t);

   // Extract execution domains
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx,
                                 regions[3].get_logical_region().get_index_space());
   const Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Allocate a buffer for the summations of the KG scheme of size (3 * (3+1)/2 * (nEq+1)) for each thread
#ifdef REALM_USE_OPENMP
   double *KGSum = new double[6*(nEq+1)*omp_get_max_threads()];
#else
   double *KGSum = new double[6*(nEq+1)];
#endif

#if (nSpec > 1)
   // Wait for the deltaTime
   const double deltaTime = futures[0].get_result<double>();
#endif

   // update RHS using Euler fluxes
   const Rect<3> cPlane = crossPlane<dir>(r_MyFluid);
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = cPlane.lo.z; k <= cPlane.hi.z; k++)
      for (int j = cPlane.lo.y; j <= cPlane.hi.y; j++)
         for (int i = cPlane.lo.x; i <= cPlane.hi.x; i++) {
#ifdef REALM_USE_OPENMP
            double *myKGSum = &KGSum[6*(nEq+1)*omp_get_thread_num()];
#else
            double *myKGSum = &KGSum[0];
#endif
            // Launch the loop for the span
            updateRHSSpan(myKGSum,
               acc_Conserved_t, acc_m, acc_nType, acc_shockSensor,
               acc_Conserved,
#if (nSpec > 1)
               acc_Conserved_old,
#endif
               acc_rho, acc_SoS,
               acc_MassFracs, acc_velocity, acc_pressure,
#if (nSpec > 1)
               args.RK_coeffs[0], args.RK_coeffs[1], deltaTime,
#endif
               0, getSize<dir>(r_MyFluid),
               i-cPlane.lo.x, j-cPlane.lo.y, k-cPlane.lo.z,
               r_MyFluid, Fluid_bounds, args.mix);
         }
   // Cleanup
   delete[] KGSum;
}

// Specielize UpdateUsingHybridEulerFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingHybridEulerFluxTask<Xdir>::TASK_NAME = "UpdateUsingHybridEulerFluxX";
template<>
/*static*/ const int             UpdateUsingHybridEulerFluxTask<Xdir>::TASK_ID = TID_UpdateUsingHybridEulerFluxX;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Xdir>::FID_m_e = FID_dcsi_e;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Xdir>::FID_shockSensor = FID_shockSensorX;

// Specielize UpdateUsingHybridEulerFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingHybridEulerFluxTask<Ydir>::TASK_NAME = "UpdateUsingHybridEulerFluxY";
template<>
/*static*/ const int             UpdateUsingHybridEulerFluxTask<Ydir>::TASK_ID = TID_UpdateUsingHybridEulerFluxY;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Ydir>::FID_m_e = FID_deta_e;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Ydir>::FID_shockSensor = FID_shockSensorY;

// Specielize UpdateUsingHybridEulerFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingHybridEulerFluxTask<Zdir>::TASK_NAME = "UpdateUsingHybridEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingHybridEulerFluxTask<Zdir>::TASK_ID = TID_UpdateUsingHybridEulerFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Zdir>::FID_m_e = FID_dzet_e;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Zdir>::FID_shockSensor = FID_shockSensorZ;

// CPU Implementation of UpdateUsingTENOEulerFluxTask
template<direction dir, class Op>
void UpdateUsingTENOEulerFluxTask<dir, Op>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 1);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved    (regions[0], FID_Conserved);
#if (nSpec > 1)
   const AccessorRO<VecNEq, 3> acc_Conserved_old(regions[0], FID_Conserved_old);
#endif
   const AccessorRO<double, 3> acc_SoS          (regions[0], FID_SoS);
   const AccessorRO<double, 3> acc_rho          (regions[0], FID_rho);
   const AccessorRO<  Vec3, 3> acc_velocity     (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure     (regions[0], FID_pressure);

   // Accessors for quantities needed for the Roe averages
   const AccessorRO<VecNSp, 3> acc_MassFracs    (regions[1], FID_MassFracs);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m            (regions[1], FID_m_e);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType        (regions[2], FID_nType);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t  (regions[3], FID_Conserved_t);

   // Extract execution domains
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx,
                                 regions[3].get_logical_region().get_index_space());
   const Rect<3> Fluid_bounds = args.Fluid_bounds;

#if (nSpec > 1)
   // Wait for the deltaTime
   const double deltaTime = futures[0].get_result<double>();
#endif

   // update RHS using Euler fluxes
   const Rect<3> cPlane = crossPlane<dir>(r_MyFluid);
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = cPlane.lo.z; k <= cPlane.hi.z; k++)
      for (int j = cPlane.lo.y; j <= cPlane.hi.y; j++)
         for (int i = cPlane.lo.x; i <= cPlane.hi.x; i++) {
            // Launch the loop for the span
            updateRHSSpan(
               acc_Conserved_t, acc_m, acc_nType,
               acc_Conserved,
#if (nSpec > 1)
               acc_Conserved_old,
#endif
               acc_rho, acc_SoS,
               acc_MassFracs, acc_velocity, acc_pressure,
#if (nSpec > 1)
               args.RK_coeffs[0], args.RK_coeffs[1], deltaTime,
#endif
               0, getSize<dir>(r_MyFluid),
               i-cPlane.lo.x, j-cPlane.lo.y, k-cPlane.lo.z,
               r_MyFluid, Fluid_bounds, args.mix);
         }
}

// Specielize UpdateUsingTENOEulerFlux for TENO and the X direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Xdir, TENO_Op<> >::TASK_NAME = "UpdateUsingTENOEulerFluxX";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Xdir, TENO_Op<> >::TASK_ID = TID_UpdateUsingTENOEulerFluxX;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Xdir, TENO_Op<> >::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Xdir, TENO_Op<> >::FID_m_e = FID_dcsi_e;

// Specielize UpdateUsingTENOEulerFlux for TENO and the Y direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Ydir, TENO_Op<> >::TASK_NAME = "UpdateUsingTENOEulerFluxY";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Ydir, TENO_Op<> >::TASK_ID = TID_UpdateUsingTENOEulerFluxY;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Ydir, TENO_Op<> >::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Ydir, TENO_Op<> >::FID_m_e = FID_deta_e;

// Specielize UpdateUsingTENOEulerFlux for TENO and the Z direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Zdir, TENO_Op<> >::TASK_NAME = "UpdateUsingTENOEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Zdir, TENO_Op<> >::TASK_ID = TID_UpdateUsingTENOEulerFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Zdir, TENO_Op<> >::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Zdir, TENO_Op<> >::FID_m_e = FID_dzet_e;

// Specielize UpdateUsingTENOEulerFlux for TENOA and the X direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Xdir, TENOA_Op<> >::TASK_NAME = "UpdateUsingTENOAEulerFluxX";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Xdir, TENOA_Op<> >::TASK_ID = TID_UpdateUsingTENOAEulerFluxX;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Xdir, TENOA_Op<> >::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Xdir, TENOA_Op<> >::FID_m_e = FID_dcsi_e;

// Specielize UpdateUsingTENOEulerFlux for TENOA and the Y direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Ydir, TENOA_Op<> >::TASK_NAME = "UpdateUsingTENOAEulerFluxY";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Ydir, TENOA_Op<> >::TASK_ID = TID_UpdateUsingTENOAEulerFluxY;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Ydir, TENOA_Op<> >::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Ydir, TENOA_Op<> >::FID_m_e = FID_deta_e;

// Specielize UpdateUsingTENOEulerFlux for TENOA and the Z direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Zdir, TENOA_Op<> >::TASK_NAME = "UpdateUsingTENOAEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Zdir, TENOA_Op<> >::TASK_ID = TID_UpdateUsingTENOAEulerFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Zdir, TENOA_Op<> >::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Zdir, TENOA_Op<> >::FID_m_e = FID_dzet_e;

// Specielize UpdateUsingTENOEulerFlux for TENOLAD and the X direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Xdir, TENOLAD_Op<> >::TASK_NAME = "UpdateUsingTENOLADEulerFluxX";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Xdir, TENOLAD_Op<> >::TASK_ID = TID_UpdateUsingTENOLADEulerFluxX;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Xdir, TENOLAD_Op<> >::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Xdir, TENOLAD_Op<> >::FID_m_e = FID_dcsi_e;

// Specielize UpdateUsingTENOEulerFlux for TENOLAD and the Y direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Ydir, TENOLAD_Op<> >::TASK_NAME = "UpdateUsingTENOLADEulerFluxY";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Ydir, TENOLAD_Op<> >::TASK_ID = TID_UpdateUsingTENOLADEulerFluxY;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Ydir, TENOLAD_Op<> >::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Ydir, TENOLAD_Op<> >::FID_m_e = FID_deta_e;

// Specielize UpdateUsingTENOEulerFlux for TENOLAD and the Z direction
template<>
/*static*/ const char * const    UpdateUsingTENOEulerFluxTask<Zdir, TENOLAD_Op<> >::TASK_NAME = "UpdateUsingTENOLADEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingTENOEulerFluxTask<Zdir, TENOLAD_Op<> >::TASK_ID = TID_UpdateUsingTENOLADEulerFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Zdir, TENOLAD_Op<> >::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingTENOEulerFluxTask<Zdir, TENOLAD_Op<> >::FID_m_e = FID_dzet_e;

// CPU Implementation of UpdateUsingSkewSymmetricEulerFlux
template<direction dir>
void UpdateUsingSkewSymmetricEulerFluxTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType      (regions[1], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m          (regions[2], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[3], FID_Conserved_t);

   // Extract execution domains
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[3].get_logical_region().get_index_space());
   const Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Allocate a buffer for the summations of the KG scheme of size (3 * (3+1)/2 * (nEq+1)) for each thread
#ifdef REALM_USE_OPENMP
   double *KGSum = new double[6*(nEq+1)*omp_get_max_threads()];
#else
   double *KGSum = new double[6*(nEq+1)];
#endif

   // update RHS using Euler fluxes
   const Rect<3> cPlane = crossPlane<dir>(r_MyFluid);
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = cPlane.lo.z; k <= cPlane.hi.z; k++)
      for (int j = cPlane.lo.y; j <= cPlane.hi.y; j++)
         for (int i = cPlane.lo.x; i <= cPlane.hi.x; i++) {
#ifdef REALM_USE_OPENMP
            double *myKGSum = &KGSum[6*(nEq+1)*omp_get_thread_num()];
#else
            double *myKGSum = &KGSum[0];
#endif
            // Launch the loop for the span
            updateRHSSpan(myKGSum,
               acc_Conserved_t, acc_m, acc_nType,
               acc_Conserved, acc_rho,
               acc_MassFracs, acc_velocity, acc_pressure,
               0, getSize<dir>(r_MyFluid),
               i-cPlane.lo.x, j-cPlane.lo.y, k-cPlane.lo.z,
               r_MyFluid, Fluid_bounds);
         }
   // Cleanup
   delete[] KGSum;
}

// Specielize UpdateUsingSkewSymmetricEulerFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingSkewSymmetricEulerFluxTask<Xdir>::TASK_NAME = "UpdateUsingSkewSymmetricEulerFluxX";
template<>
/*static*/ const int             UpdateUsingSkewSymmetricEulerFluxTask<Xdir>::TASK_ID = TID_UpdateUsingSkewSymmetricEulerFluxX;
template<>
/*static*/ const FieldID         UpdateUsingSkewSymmetricEulerFluxTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingSkewSymmetricEulerFluxTask<Xdir>::FID_m_e = FID_dcsi_e;

// Specielize UpdateUsingSkewSymmetricEulerFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingSkewSymmetricEulerFluxTask<Ydir>::TASK_NAME = "UpdateUsingSkewSymmetricEulerFluxY";
template<>
/*static*/ const int             UpdateUsingSkewSymmetricEulerFluxTask<Ydir>::TASK_ID = TID_UpdateUsingSkewSymmetricEulerFluxY;
template<>
/*static*/ const FieldID         UpdateUsingSkewSymmetricEulerFluxTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingSkewSymmetricEulerFluxTask<Ydir>::FID_m_e = FID_deta_e;

// Specielize UpdateUsingSkewSymmetricEulerFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingSkewSymmetricEulerFluxTask<Zdir>::TASK_NAME = "UpdateUsingSkewSymmetricEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingSkewSymmetricEulerFluxTask<Zdir>::TASK_ID = TID_UpdateUsingSkewSymmetricEulerFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingSkewSymmetricEulerFluxTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingSkewSymmetricEulerFluxTask<Zdir>::FID_m_e = FID_dzet_e;

// CPU Implementation of UpdateUsingDiffusionFlux
template<direction dir>
void UpdateUsingDiffusionFluxTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 0);

   // Accessors for DiffGhost region
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_temperature(regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MolarFracs (regions[0], FID_MolarFracs);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_mu         (regions[0], FID_mu);
   const AccessorRO<double, 3> acc_lam        (regions[0], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di         (regions[0], FID_Di);
   const AccessorRO<   int, 3> acc_nType1     (regions[0], FID_nType1);
   const AccessorRO<   int, 3> acc_nType2     (regions[0], FID_nType2);
   const AccessorRO<double, 3> acc_m_d1       (regions[0], FID_m_d1);
   const AccessorRO<double, 3> acc_m_d2       (regions[0], FID_m_d2);

   // Accessors for DiffGradGhost region
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[1], FID_velocity);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[2], FID_nType);
   const AccessorRO<double, 3> acc_m_s        (regions[2], FID_m_s);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_d        (regions[3], FID_m_d);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[4].get_logical_region().get_index_space());
   const Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Diffusion fluxes
   const Rect<3> cPlane = crossPlane<dir>(r_MyFluid);
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = cPlane.lo.z; k <= cPlane.hi.z; k++)
      for (int j = cPlane.lo.y; j <= cPlane.hi.y; j++)
         for (int i = cPlane.lo.x; i <= cPlane.hi.x; i++) {
            // Launch the loop for the span
            updateRHSSpan(acc_Conserved_t, acc_m_s,
                     acc_m_d, acc_m_d1, acc_m_d2,
                     acc_nType, acc_nType1, acc_nType2,
                     acc_rho, acc_mu, acc_lam, acc_Di,
                     acc_temperature, acc_velocity, acc_MolarFracs, acc_Conserved,
                     0, getSize<dir>(r_MyFluid),
                     i-cPlane.lo.x, j-cPlane.lo.y, k-cPlane.lo.z,
                     r_MyFluid, Fluid_bounds, args.mix);
         }
}

// Specielize UpdateUsingDiffusionFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingDiffusionFluxTask<Xdir>::TASK_NAME = "UpdateUsingDiffusionFluxX";
template<>
/*static*/ const int             UpdateUsingDiffusionFluxTask<Xdir>::TASK_ID = TID_UpdateUsingDiffusionFluxX;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_nType  = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_nType1 = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_nType2 = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_m_s  = FID_dcsi_s;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_m_d  = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_m_d1 = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_m_d2 = FID_dzet_d;

// Specielize UpdateUsingDiffusionFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingDiffusionFluxTask<Ydir>::TASK_NAME = "UpdateUsingDiffusionFluxY";
template<>
/*static*/ const int             UpdateUsingDiffusionFluxTask<Ydir>::TASK_ID = TID_UpdateUsingDiffusionFluxY;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_nType  = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_nType1 = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_nType2 = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_m_s  = FID_deta_s;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_m_d  = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_m_d1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_m_d2 = FID_dzet_d;

// Specielize UpdateUsingDiffusionFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingDiffusionFluxTask<Zdir>::TASK_NAME = "UpdateUsingDiffusionFluxZ";
template<>
/*static*/ const int             UpdateUsingDiffusionFluxTask<Zdir>::TASK_ID = TID_UpdateUsingDiffusionFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_nType  = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_nType1 = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_nType2 = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_m_s  = FID_dzet_s;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_m_d  = FID_dzet_d;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_m_d1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_m_d2 = FID_deta_d;

// CPU Implementation of UpdateUsingFluxNSCBCInflowMinusSide
template<direction dir>
void UpdateUsingFluxNSCBCInflowMinusSideTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for Fluid data
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);

   // Accessors for local data
   const AccessorRO<   int, 3> acc_nType      (regions[1], FID_nType);
   const AccessorRO<double, 3> acc_m_d        (regions[1], FID_m_d);
   const AccessorRO<double, 3> acc_rho        (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_SoS        (regions[1], FID_SoS);
   const AccessorRO<double, 3> acc_temperature(regions[1], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_dudt       (regions[1], FID_dudtBoundary);
   const AccessorRO<double, 3> acc_dTdt       (regions[1], FID_dTdtBoundary);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[2], FID_Conserved_t);

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[2].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addLODIfluxes(acc_Conserved_t[p],
                          acc_MassFracs, acc_pressure, acc_velocity,
                          acc_SoS[p], acc_rho[p], acc_temperature[p],
                          acc_dudt[p], acc_dTdt[p],
                          p, acc_nType[p], acc_m_d[p], args.mix);
         }
};

// Specielize UpdateUsingFluxNSCBCInflowMinusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowXNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowXNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::FID_m_d = FID_dcsi_d;

// Specielize UpdateUsingFluxNSCBCInflowMinusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowMinusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowYNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowMinusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowYNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Ydir>::FID_m_d = FID_deta_d;

// Specielize UpdateUsingFluxNSCBCInflowMinusSide for the Z direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowMinusSideTask<Zdir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowZNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowMinusSideTask<Zdir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowZNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Zdir>::FID_m_d = FID_dzet_d;

// CPU Implementation of UpdateUsingFluxNSCBCInflowPlusSide
template<direction dir>
void UpdateUsingFluxNSCBCInflowPlusSideTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for Fluid data
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);

   // Accessors for local data
   const AccessorRO<   int, 3> acc_nType      (regions[1], FID_nType);
   const AccessorRO<double, 3> acc_m_d        (regions[1], FID_m_d);
   const AccessorRO<double, 3> acc_rho        (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_SoS        (regions[1], FID_SoS);
   const AccessorRO<double, 3> acc_temperature(regions[1], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_dudt       (regions[1], FID_dudtBoundary);
   const AccessorRO<double, 3> acc_dTdt       (regions[1], FID_dTdtBoundary);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[2], FID_Conserved_t);

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[2].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addLODIfluxes(acc_Conserved_t[p],
                          acc_MassFracs, acc_pressure, acc_velocity,
                          acc_SoS[p], acc_rho[p], acc_temperature[p],
                          acc_dudt[p], acc_dTdt[p],
                          p, acc_nType[p], acc_m_d[p], args.mix);
         }
};

// Specielize UpdateUsingFluxNSCBCInflowPlusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowPlusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowXPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowPlusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowXPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Xdir>::FID_m_d = FID_dcsi_d;

// Specielize UpdateUsingFluxNSCBCInflowPlusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowYPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowYPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::FID_m_d = FID_deta_d;

// Specielize UpdateUsingFluxNSCBCInflowPlusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowPlusSideTask<Zdir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowZPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowPlusSideTask<Zdir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowZPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Zdir>::FID_m_d = FID_dzet_d;

// CPU Implementation of UpdateUsingFluxNSCBCOutflowMinusSide
template<direction dir>
void UpdateUsingFluxNSCBCOutflowMinusSideTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 1);

   // Accessors for ghost data
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);

   // Accessors for Fluid data
   const AccessorRO<   int, 3> acc_nType_N    (regions[1], FID_nType_N);
   const AccessorRO<   int, 3> acc_nType_T1   (regions[1], FID_nType_T1);
   const AccessorRO<   int, 3> acc_nType_T2   (regions[1], FID_nType_T2);
   const AccessorRO<double, 3> acc_m_d_N      (regions[1], FID_m_d_N);
   const AccessorRO<double, 3> acc_m_d_T1     (regions[1], FID_m_d_T1);
   const AccessorRO<double, 3> acc_m_d_T2     (regions[1], FID_m_d_T2);
   const AccessorRO<double, 3> acc_rho        (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_mu         (regions[1], FID_mu);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[1], FID_pressure);

   // Accessors for local data
   const AccessorRO<double, 3> acc_SoS        (regions[2], FID_SoS);
   const AccessorRO<double, 3> acc_temperature(regions[2], FID_temperature);
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[2], FID_Conserved);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[3], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[3].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addLODIfluxes(acc_Conserved_t[p],
                          acc_nType_N, acc_nType_T1, acc_nType_T2,
                          acc_m_d_N, acc_m_d_T1, acc_m_d_T2,
                          acc_MassFracs, acc_rho, acc_mu, acc_pressure, acc_velocity,
                          acc_SoS[p], acc_temperature[p], acc_Conserved[p],
                          p, args.Fluid_bounds, MaxMach, args.LengthScale, args.PInf, args.mix);
         }
};

// Specielize UpdateUsingFluxNSCBCOutflowMinusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowXNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowXNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::FID_nType_N  = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::FID_nType_T1 = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::FID_nType_T2 = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::FID_m_d_N  = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::FID_m_d_T1 = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::FID_m_d_T2 = FID_dzet_d;

// Specielize UpdateUsingFluxNSCBCOutflowMinusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowYNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowYNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_nType_N  = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_nType_T1 = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_nType_T2 = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_m_d_N  = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_m_d_T1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_m_d_T2 = FID_dzet_d;

// Specielize UpdateUsingFluxNSCBCOutflowMinusSide for the Z direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowZNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowZNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::FID_nType_N  = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::FID_nType_T1 = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::FID_nType_T2 = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::FID_m_d_N  = FID_dzet_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::FID_m_d_T1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::FID_m_d_T2 = FID_deta_d;


template<direction dir>
void UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 1);

   // Accessors for ghost data
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);

   // Accessors for Fluid data
   const AccessorRO<   int, 3> acc_nType_N    (regions[1], FID_nType_N);
   const AccessorRO<   int, 3> acc_nType_T1   (regions[1], FID_nType_T1);
   const AccessorRO<   int, 3> acc_nType_T2   (regions[1], FID_nType_T2);
   const AccessorRO<double, 3> acc_m_d_N      (regions[1], FID_m_d_N);
   const AccessorRO<double, 3> acc_m_d_T1     (regions[1], FID_m_d_T1);
   const AccessorRO<double, 3> acc_m_d_T2     (regions[1], FID_m_d_T2);
   const AccessorRO<double, 3> acc_rho        (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_mu         (regions[1], FID_mu);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[1], FID_pressure);

   // Accessors for local data
   const AccessorRO<double, 3> acc_SoS        (regions[2], FID_SoS);
   const AccessorRO<double, 3> acc_temperature(regions[2], FID_temperature);
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[2], FID_Conserved);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[3], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[3].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addLODIfluxes(acc_Conserved_t[p],
                          acc_nType_N, acc_nType_T1, acc_nType_T2,
                          acc_m_d_N, acc_m_d_T1, acc_m_d_T2,
                          acc_MassFracs, acc_rho, acc_mu, acc_pressure, acc_velocity,
                          acc_SoS[p], acc_temperature[p], acc_Conserved[p],
                          p, args.Fluid_bounds, MaxMach, args.LengthScale, args.PInf, args.mix);
         }
};

// Specielize UpdateUsingFluxNSCBCOutflowPlusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowXPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowXPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_nType_N  = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_nType_T1 = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_nType_T2 = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_m_d_N  = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_m_d_T1 = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_m_d_T2 = FID_dzet_d;

// Specielize UpdateUsingFluxNSCBCOutflowPlusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowYPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowYPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_nType_N  = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_nType_T1 = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_nType_T2 = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_m_d_N  = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_m_d_T1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_m_d_T2 = FID_dzet_d;

// Specielize UpdateUsingFluxNSCBCOutflowPlusSide for the Z direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowZPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowZPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::FID_nType_N  = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::FID_nType_T1 = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::FID_nType_T2 = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::FID_m_d_N  = FID_dzet_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::FID_m_d_T1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::FID_m_d_T2 = FID_deta_d;

// CPU Implementation of UpdateUsingFluxNSCBCFarFieldMinusSide
template<direction dir>
void UpdateUsingFluxNSCBCFarFieldMinusSideTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 1);

   // Accessors for local data
   const AccessorRO<   int, 3> acc_nType               (regions[1], FID_nType);
   const AccessorRO<double, 3> acc_m_d                 (regions[1], FID_m_d);
   const AccessorRO<double, 3> acc_rho                 (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs           (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure            (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature         (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity            (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_SoS                 (regions[1], FID_SoS);
   const AccessorRO<VecNEq, 3> acc_Conserved           (regions[1], FID_Conserved);
   const AccessorRO<VecNSp, 3> acc_MolarFracs_profile  (regions[1], FID_MolarFracs_profile);
   const AccessorRO<double, 3> acc_temperature_profile (regions[1], FID_temperature_profile);
   const AccessorRO<  Vec3, 3> acc_velocity_profile    (regions[1], FID_velocity_profile);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t         (regions[2], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[2].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addLODIfluxes(acc_Conserved_t[p],
                          acc_rho, acc_MassFracs, acc_pressure, acc_velocity,
                          acc_nType[p], acc_m_d[p],
                          acc_SoS[p], acc_temperature[p], acc_Conserved[p],
                          acc_temperature_profile[p], acc_velocity_profile[p], acc_MolarFracs_profile[p],
                          args.PInf, MaxMach, args.LengthScale, p, args.mix);
         }
};

// Specielize UpdateUsingFluxNSCBCFarFieldMinusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCFarFieldMinusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCFarFieldXNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCFarFieldMinusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCFarFieldXNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldMinusSideTask<Xdir>::FID_nType  = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldMinusSideTask<Xdir>::FID_m_d    = FID_dcsi_d;

// Specielize UpdateUsingFluxNSCBCFarFieldMinusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCFarFieldMinusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCFarFieldYNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCFarFieldMinusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCFarFieldYNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldMinusSideTask<Ydir>::FID_nType  = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldMinusSideTask<Ydir>::FID_m_d    = FID_deta_d;

// Specielize UpdateUsingFluxNSCBCFarFieldMinusSide for the Z direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCFarFieldMinusSideTask<Zdir>::TASK_NAME = "UpdateUsingFluxNSCBCFarFieldZNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCFarFieldMinusSideTask<Zdir>::TASK_ID = TID_UpdateUsingFluxNSCBCFarFieldZNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldMinusSideTask<Zdir>::FID_nType  = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldMinusSideTask<Zdir>::FID_m_d    = FID_dzet_d;

template<direction dir>
void UpdateUsingFluxNSCBCFarFieldPlusSideTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 1);

   // Accessors for local data
   const AccessorRO<   int, 3> acc_nType               (regions[1], FID_nType);
   const AccessorRO<double, 3> acc_m_d                 (regions[1], FID_m_d);
   const AccessorRO<double, 3> acc_rho                 (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs           (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure            (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature         (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity            (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_SoS                 (regions[1], FID_SoS);
   const AccessorRO<VecNEq, 3> acc_Conserved           (regions[1], FID_Conserved);
   const AccessorRO<VecNSp, 3> acc_MolarFracs_profile  (regions[1], FID_MolarFracs_profile);
   const AccessorRO<double, 3> acc_temperature_profile (regions[1], FID_temperature_profile);
   const AccessorRO<  Vec3, 3> acc_velocity_profile    (regions[1], FID_velocity_profile);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t         (regions[2], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[2].get_logical_region().get_index_space());

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addLODIfluxes(acc_Conserved_t[p],
                          acc_rho, acc_MassFracs, acc_pressure, acc_velocity,
                          acc_nType[p], acc_m_d[p],
                          acc_SoS[p], acc_temperature[p], acc_Conserved[p],
                          acc_temperature_profile[p], acc_velocity_profile[p], acc_MolarFracs_profile[p],
                          args.PInf, MaxMach, args.LengthScale, p, args.mix);
         }
};

// Specielize UpdateUsingFluxNSCBCFarFieldPlusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCFarFieldPlusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCFarFieldXPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCFarFieldPlusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCFarFieldXPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldPlusSideTask<Xdir>::FID_nType  = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldPlusSideTask<Xdir>::FID_m_d    = FID_dcsi_d;

// Specielize UpdateUsingFluxNSCBCFarFieldPlusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCFarFieldYPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCFarFieldYPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::FID_nType  = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::FID_m_d    = FID_deta_d;

// Specielize UpdateUsingFluxNSCBCFarFieldPlusSide for the Z direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCFarFieldPlusSideTask<Zdir>::TASK_NAME = "UpdateUsingFluxNSCBCFarFieldZPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCFarFieldPlusSideTask<Zdir>::TASK_ID = TID_UpdateUsingFluxNSCBCFarFieldZPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldPlusSideTask<Zdir>::FID_nType  = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCFarFieldPlusSideTask<Zdir>::FID_m_d    = FID_dzet_d;

// UpdateUsingFluxIncomingShockTask
template<>
/*static*/ const char * const    UpdateUsingFluxIncomingShockTask<Ydir>::TASK_NAME = "UpdateUsingFluxIncomingShock";
template<>
/*static*/ const int             UpdateUsingFluxIncomingShockTask<Ydir>::TASK_ID = TID_UpdateUsingFluxIncomingShockYPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxIncomingShockTask<Ydir>::FID_nType  = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxIncomingShockTask<Ydir>::FID_m_d    = FID_deta_d;

template<direction dir>
void UpdateUsingFluxIncomingShockTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 1);

   // Accessors for local data
   const AccessorRO<   int, 3> acc_nType               (regions[1], FID_nType);
   const AccessorRO<double, 3> acc_m_d                 (regions[1], FID_m_d);
   const AccessorRO<double, 3> acc_rho                 (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs           (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure            (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature         (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity            (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_SoS                 (regions[1], FID_SoS);
   const AccessorRO<VecNEq, 3> acc_Conserved           (regions[1], FID_Conserved);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t         (regions[2], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[2].get_logical_region().get_index_space());

   // Precompute the mixture averaged molecular weight
   const VecNSp MolarFracs(args.params.MolarFracs);
   const   Vec3 velocity0(args.params.velocity0);
   const   Vec3 velocity1(args.params.velocity1);

#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            if (i < args.params.iShock)
               // Set to upstream values
               UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::addLODIfluxes(
                          acc_Conserved_t[p],
                          acc_rho, acc_MassFracs, acc_pressure, acc_velocity,
                          acc_nType[p], acc_m_d[p],
                          acc_SoS[p], acc_temperature[p], acc_Conserved[p],
                          args.params.temperature0, velocity0, MolarFracs,
                          args.params.pressure0, MaxMach, args.LengthScale, p, args.mix);
            else
               // Set to downstream values
               UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::addLODIfluxes(
                          acc_Conserved_t[p],
                          acc_rho, acc_MassFracs, acc_pressure, acc_velocity,
                          acc_nType[p], acc_m_d[p],
                          acc_SoS[p], acc_temperature[p], acc_Conserved[p],
                          args.params.temperature1, velocity1, MolarFracs,
                          args.params.pressure1, MaxMach, args.LengthScale, p, args.mix);
         }
}

// CalculateAveragePDTask
/*static*/ const char * const    CalculateAveragePDTask::TASK_NAME = "CalculateAveragePD";
/*static*/ const int             CalculateAveragePDTask::TASK_ID = TID_CalculateAveragePD;

double CalculateAveragePDTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for variables in ghost region
   const AccessorRO<  Vec3, 3> acc_velocity  (regions[0], FID_velocity);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType_x   (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y   (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z   (regions[1], FID_nType_z);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_dcsi_d    (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d    (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d    (regions[1], FID_dzet_d);

   // Accessors for local primitive variables
   const AccessorRO<double, 3> acc_pressure  (regions[1], FID_pressure);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Accumulate average pressure dilatation in this buffer
   double avePD = 0.0;

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3) reduction(+:avePD)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            const double cellVolume = 1.0/(acc_dcsi_d[p]*acc_deta_d[p]*acc_dzet_d[p]);
            avePD += cellVolume*CalculatePressureDilatation(
                                 acc_nType_x, acc_nType_y, acc_nType_z,
                                 acc_dcsi_d, acc_deta_d, acc_dzet_d,
                                 acc_pressure, acc_velocity,
                                 p, Fluid_bounds);
         }
   return avePD;
}

// CPU Implementation of UpdateUsingDiffusionFlux
template<direction dir>
double AddDissipationTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Accessors for DiffGhost region
   const AccessorRO<double, 3> acc_mu         (regions[0], FID_mu);
   const AccessorRO<   int, 3> acc_nType1     (regions[0], FID_nType1);
   const AccessorRO<   int, 3> acc_nType2     (regions[0], FID_nType2);
   const AccessorRO<double, 3> acc_m_d1       (regions[0], FID_m_d1);
   const AccessorRO<double, 3> acc_m_d2       (regions[0], FID_m_d2);

   // Accessors for DiffGradGhost region
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[1], FID_velocity);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[2], FID_nType);
   const AccessorRO<double, 3> acc_m_s        (regions[2], FID_m_s);

   // Extract execution domains
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[3].get_logical_region().get_index_space());
   const Rect<3> Fluid_bounds = args.Fluid_bounds;

   // accumulate integral of diffusion fluxes
   double acc = 0.0;

   // Add contributions to integral
   const Rect<3> cPlane = crossPlane<dir>(r_MyFluid);
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3) reduction(+:acc)
#endif
   for (int k = cPlane.lo.z; k <= cPlane.hi.z; k++)
      for (int j = cPlane.lo.y; j <= cPlane.hi.y; j++)
         for (int i = cPlane.lo.x; i <= cPlane.hi.x; i++) {
            // Launch the loop for the span
            acc += AddSpan(acc_m_s, acc_m_d1, acc_m_d2,
                           acc_nType, acc_nType1, acc_nType2,
                           acc_mu, acc_velocity,
                           0, getSize<dir>(r_MyFluid),
                           i-cPlane.lo.x, j-cPlane.lo.y, k-cPlane.lo.z,
                           r_MyFluid, Fluid_bounds, args.mix);
         }

   return acc;
}

// Specielize AddDissipation for the X direction
template<>
/*static*/ const char * const    AddDissipationTask<Xdir>::TASK_NAME = "AddDissipationX";
template<>
/*static*/ const int             AddDissipationTask<Xdir>::TASK_ID = TID_AddDissipationX;
template<>
/*static*/ const FieldID         AddDissipationTask<Xdir>::FID_nType  = FID_nType_x;
template<>
/*static*/ const FieldID         AddDissipationTask<Xdir>::FID_nType1 = FID_nType_y;
template<>
/*static*/ const FieldID         AddDissipationTask<Xdir>::FID_nType2 = FID_nType_z;
template<>
/*static*/ const FieldID         AddDissipationTask<Xdir>::FID_m_s  = FID_dcsi_s;
template<>
/*static*/ const FieldID         AddDissipationTask<Xdir>::FID_m_d1 = FID_deta_d;
template<>
/*static*/ const FieldID         AddDissipationTask<Xdir>::FID_m_d2 = FID_dzet_d;

// Specielize AddDissipation for the Y direction
template<>
/*static*/ const char * const    AddDissipationTask<Ydir>::TASK_NAME = "AddDissipationY";
template<>
/*static*/ const int             AddDissipationTask<Ydir>::TASK_ID = TID_AddDissipationY;
template<>
/*static*/ const FieldID         AddDissipationTask<Ydir>::FID_nType  = FID_nType_y;
template<>
/*static*/ const FieldID         AddDissipationTask<Ydir>::FID_nType1 = FID_nType_x;
template<>
/*static*/ const FieldID         AddDissipationTask<Ydir>::FID_nType2 = FID_nType_z;
template<>
/*static*/ const FieldID         AddDissipationTask<Ydir>::FID_m_s  = FID_deta_s;
template<>
/*static*/ const FieldID         AddDissipationTask<Ydir>::FID_m_d1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         AddDissipationTask<Ydir>::FID_m_d2 = FID_dzet_d;

// Specielize AddDissipation for the Z direction
template<>
/*static*/ const char * const    AddDissipationTask<Zdir>::TASK_NAME = "AddDissipationZ";
template<>
/*static*/ const int             AddDissipationTask<Zdir>::TASK_ID = TID_AddDissipationZ;
template<>
/*static*/ const FieldID         AddDissipationTask<Zdir>::FID_nType  = FID_nType_z;
template<>
/*static*/ const FieldID         AddDissipationTask<Zdir>::FID_nType1 = FID_nType_x;
template<>
/*static*/ const FieldID         AddDissipationTask<Zdir>::FID_nType2 = FID_nType_y;
template<>
/*static*/ const FieldID         AddDissipationTask<Zdir>::FID_m_s  = FID_dzet_s;
template<>
/*static*/ const FieldID         AddDissipationTask<Zdir>::FID_m_d1 = FID_dcsi_d;
template<>
/*static*/ const FieldID         AddDissipationTask<Zdir>::FID_m_d2 = FID_deta_d;

void register_rhs_tasks() {

   TaskHelper::register_hybrid_variants<UpdateUsingHybridEulerFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingHybridEulerFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingHybridEulerFluxTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Xdir, TENO_Op<> >>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Ydir, TENO_Op<> >>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Zdir, TENO_Op<> >>();

   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Xdir, TENOA_Op<> >>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Ydir, TENOA_Op<> >>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Zdir, TENOA_Op<> >>();

   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Xdir, TENOLAD_Op<> >>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Ydir, TENOLAD_Op<> >>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOEulerFluxTask<Zdir, TENOLAD_Op<> >>();

   TaskHelper::register_hybrid_variants<UpdateUsingSkewSymmetricEulerFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingSkewSymmetricEulerFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingSkewSymmetricEulerFluxTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingDiffusionFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingDiffusionFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingDiffusionFluxTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowMinusSideTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowMinusSideTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowPlusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowPlusSideTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCFarFieldMinusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCFarFieldMinusSideTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCFarFieldMinusSideTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCFarFieldPlusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCFarFieldPlusSideTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxIncomingShockTask<Ydir>>();

   TaskHelper::register_hybrid_variants<CalculateAveragePDTask, double, DeferredValue<double>>();
   TaskHelper::register_hybrid_variants<AddDissipationTask<Xdir>, double, DeferredValue<double>>();
   TaskHelper::register_hybrid_variants<AddDissipationTask<Ydir>, double, DeferredValue<double>>();
   TaskHelper::register_hybrid_variants<AddDissipationTask<Zdir>, double, DeferredValue<double>>();

};
