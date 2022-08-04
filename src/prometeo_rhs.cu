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

#include "prometeo_rhs.hpp"
#include "prometeo_rhs.inl"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingHybridEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateRHSUsingHybridFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                     const AccessorRO<double, 3> m_e,
                                     const AccessorRO<VecNEq, 3> Conserved,
#if (nSpec > 1)
                                     const AccessorRO<VecNEq, 3> Conserved_old,
#endif
                                     const AccessorRO<double, 3> rho,
                                     const AccessorRO<double, 3> SoS,
                                     const AccessorRO<VecNSp, 3> MassFracs,
                                     const AccessorRO<  Vec3, 3> velocity,
                                     const AccessorRO<double, 3> pressure,
                                     const AccessorRO<   int, 3> nType,
                                     const AccessorRO<  bool, 3> shockSensor,
#if (nSpec > 1)
                                     const double  RK_coeffs0,
                                     const double  RK_coeffs1,
                                     const double  deltaTime,
#endif
                                     const Rect<3> Flux_bounds,
                                     const Rect<3> Fluid_bounds,
                                     const coord_t size_x,
                                     const coord_t size_y,
                                     const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if (isThreadInCrossPlane<dir>(size_x, size_y, size_z)) {
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         // Allocate a buffer for the summations of the KG scheme of size (3 * (3+1)/2 * (nEq+1)) for each thread
         double KGSum[6*(nEq+1)];
         // Launch the loop for the span
         UpdateUsingHybridEulerFluxTask<dir>::updateRHSSpan(
            KGSum,
            Conserved_t, m_e, nType, shockSensor, Conserved,
#if (nSpec > 1)
            Conserved_old,
#endif
            rho, SoS,
            MassFracs, velocity, pressure,
#if (nSpec > 1)
            RK_coeffs0, RK_coeffs1, deltaTime,
#endif
            firstIndex, lastIndex,
            x, y, z,
            Flux_bounds, Fluid_bounds, mix);
      }
   }
}


template<direction dir>
__host__
void UpdateUsingHybridEulerFluxTask<dir>::gpu_base_impl(
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
   const AccessorRO<double, 3> acc_m_e          (regions[1], FID_m_e);

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

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with hybrid scheme
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_MyFluid);
   UpdateRHSUsingHybridFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved,
#if (nSpec > 1)
                              acc_Conserved_old,
#endif
                              acc_rho, acc_SoS,
                              acc_MassFracs, acc_velocity, acc_pressure,
                              acc_nType, acc_shockSensor,
#if (nSpec > 1)
                              args.RK_coeffs[0], args.RK_coeffs[1], deltaTime,
#endif
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

// Force the compiler to instanciate these functions
template void UpdateUsingHybridEulerFluxTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingHybridEulerFluxTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingHybridEulerFluxTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingTENOEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir, class Op>
__global__
void UpdateRHSUsingTENOFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                   const AccessorRO<double, 3> m_e,
                                   const AccessorRO<VecNEq, 3> Conserved,
#if (nSpec > 1)
                                   const AccessorRO<VecNEq, 3> Conserved_old,
#endif
                                   const AccessorRO<double, 3> SoS,
                                   const AccessorRO<double, 3> rho,
                                   const AccessorRO<  Vec3, 3> velocity,
                                   const AccessorRO<double, 3> pressure,
                                   const AccessorRO<VecNSp, 3> MassFracs,
                                   const AccessorRO<   int, 3> nType,
#if (nSpec > 1)
                                   const double  RK_coeffs0,
                                   const double  RK_coeffs1,
                                   const double  deltaTime,
#endif
                                   const Rect<3> Flux_bounds,
                                   const Rect<3> Fluid_bounds,
                                   const coord_t size_x,
                                   const coord_t size_y,
                                   const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if (isThreadInCrossPlane<dir>(size_x, size_y, size_z)) {
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         // Launch the loop for the span
         UpdateUsingTENOEulerFluxTask<dir, Op>::updateRHSSpan(
            Conserved_t, m_e, nType, Conserved,
#if (nSpec > 1)
            Conserved_old,
#endif
            rho, SoS,
            MassFracs, velocity, pressure,
#if (nSpec > 1)
            RK_coeffs0, RK_coeffs1, deltaTime,
#endif
            firstIndex, lastIndex,
            x, y, z,
            Flux_bounds, Fluid_bounds, mix);
      }
   }
}

template<direction dir, class Op>
__host__
void UpdateUsingTENOEulerFluxTask<dir, Op>::gpu_base_impl(
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
   const AccessorRO<double, 3> acc_m_e          (regions[1], FID_m_e);

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

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with TENOA
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_MyFluid);
   UpdateRHSUsingTENOFlux_kernel<dir, Op><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved,
#if (nSpec > 1)
                              acc_Conserved_old,
#endif
                              acc_SoS, acc_rho,
                              acc_velocity, acc_pressure, acc_MassFracs,
                              acc_nType,
#if (nSpec > 1)
                              args.RK_coeffs[0], args.RK_coeffs[1], deltaTime,
#endif
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

// Force the compiler to instanciate these functions
template void UpdateUsingTENOEulerFluxTask<Xdir, TENO_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Ydir, TENO_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Zdir, TENO_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Xdir, TENOA_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Ydir, TENOA_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Zdir, TENOA_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Xdir, TENOLAD_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Ydir, TENOLAD_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOEulerFluxTask<Zdir, TENOLAD_Op<> >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingSkewSymmetricEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateRHSUsingKGFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                 const AccessorRO<double, 3>         m_e,
                                 const AccessorRO<VecNEq, 3>   Conserved,
                                 const AccessorRO<double, 3>         rho,
                                 const AccessorRO<VecNSp, 3>   MassFracs,
                                 const AccessorRO<  Vec3, 3>    velocity,
                                 const AccessorRO<double, 3>    pressure,
                                 const AccessorRO<   int, 3>       nType,
                                 const Rect<3> Flux_bounds,
                                 const Rect<3> Fluid_bounds,
                                 const coord_t size_x,
                                 const coord_t size_y,
                                 const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if (isThreadInCrossPlane<dir>(size_x, size_y, size_z)) {
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         // Allocate a buffer for the summations of the KG scheme of size (3 * (3+1)/2 * (nEq+1)) for each thread
         double KGSum[6*(nEq+1)];
         // Launch the loop for the span
         UpdateUsingSkewSymmetricEulerFluxTask<dir>::updateRHSSpan(
            KGSum,
            Conserved_t, m_e, nType,
            Conserved, rho,
            MassFracs, velocity, pressure,
            firstIndex, lastIndex,
            x, y, z,
            Flux_bounds, Fluid_bounds);
      }
   }
}

template<direction dir>
__host__
void UpdateUsingSkewSymmetricEulerFluxTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved(regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_rho      (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs(regions[0], FID_MassFracs);
   const AccessorRO<  Vec3, 3> acc_velocity (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure (regions[0], FID_pressure);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[1], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e(regions[2], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[3], FID_Conserved_t);

   // Extract execution domains
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[3].get_logical_region().get_index_space());
   const Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with KG
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_MyFluid);
   UpdateRHSUsingKGFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved, acc_rho, acc_MassFracs,
                              acc_velocity, acc_pressure, acc_nType,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

// Force the compiler to instanciate these functions
template void UpdateUsingSkewSymmetricEulerFluxTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingSkewSymmetricEulerFluxTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingSkewSymmetricEulerFluxTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingDiffusionFluxTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
struct UpdateRHSUsingDiffusionFlux_kernelArgs {
   const AccessorRW<VecNEq, 3> Conserved_t;
   const AccessorRO<   int, 3>       nType;
   const AccessorRO<   int, 3>      nType1;
   const AccessorRO<   int, 3>      nType2;
   const AccessorRO<double, 3>         m_s;
   const AccessorRO<double, 3>         m_d;
   const AccessorRO<double, 3>        m_d1;
   const AccessorRO<double, 3>        m_d2;
   const AccessorRO<double, 3>         rho;
   const AccessorRO<double, 3>          mu;
   const AccessorRO<double, 3>         lam;
   const AccessorRO<VecNSp, 3>          Di;
   const AccessorRO<double, 3> temperature;
   const AccessorRO<  Vec3, 3>    velocity;
   const AccessorRO<VecNSp, 3>          Xi;
   const AccessorRO<VecNEq, 3>       rhoYi;
};

template<direction dir>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void UpdateRHSUsingDiffusionFlux_kernel(const DeferredBuffer<UpdateRHSUsingDiffusionFlux_kernelArgs, 1> buffer,
                                        const Rect<3> Flux_bounds,
                                        const Rect<3> Fluid_bounds,
                                        const coord_t size_x,
                                        const coord_t size_y,
                                        const coord_t size_z)
#else
void UpdateRHSUsingDiffusionFlux_kernel(const UpdateRHSUsingDiffusionFlux_kernelArgs a,
                                        const Rect<3> Flux_bounds,
                                        const Rect<3> Fluid_bounds,
                                        const coord_t size_x,
                                        const coord_t size_y,
                                        const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   UpdateRHSUsingDiffusionFlux_kernelArgs a = buffer[0];
#endif

   if (isThreadInCrossPlane<dir>(size_x, size_y, size_z)) {
      const coord_t size = getSize<dir>(Fluid_bounds);
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         UpdateUsingDiffusionFluxTask<dir>::updateRHSSpan(
                  a.Conserved_t, a.m_s,
                  a.m_d, a.m_d1, a.m_d2,
                  a.nType, a.nType1, a.nType2,
                  a.rho, a.mu, a.lam, a.Di,
                  a.temperature, a.velocity, a.Xi, a.rhoYi,
                  firstIndex, lastIndex,
                  x, y, z,
                  Flux_bounds, Fluid_bounds, mix);
      }
   }
}

template<direction dir>
__host__
void UpdateUsingDiffusionFluxTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS using the diffusion fluxes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_MyFluid);
   const UpdateRHSUsingDiffusionFlux_kernelArgs kArgs = {
      .Conserved_t = acc_Conserved_t,
      .nType       = acc_nType,
      .nType1      = acc_nType1,
      .nType2      = acc_nType2,
      .m_s         = acc_m_s,
      .m_d         = acc_m_d,
      .m_d1        = acc_m_d1,
      .m_d2        = acc_m_d2,
      .rho         = acc_rho,
      .mu          = acc_mu,
      .lam         = acc_lam,
      .Di          = acc_Di,
      .temperature = acc_temperature,
      .velocity    = acc_velocity,
      .Xi          = acc_MolarFracs,
      .rhoYi       = acc_Conserved
   };
#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<UpdateRHSUsingDiffusionFlux_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   UpdateRHSUsingDiffusionFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(buffer,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
#else
   UpdateRHSUsingDiffusionFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(kArgs,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
#endif
}

template void UpdateUsingDiffusionFluxTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingDiffusionFluxTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingDiffusionFluxTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCInflowMinusSideTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateUsingFluxNSCBCInflowMinusSideTask_kernel(
                           const AccessorRW<VecNEq, 3> Conserved_t,
                           const AccessorRO<VecNSp, 3>   MassFracs,
                           const AccessorRO<double, 3>    pressure,
                           const AccessorRO<double, 3>         SoS,
                           const AccessorRO<double, 3>         rho,
                           const AccessorRO<double, 3> temperature,
                           const AccessorRO<  Vec3, 3>    velocity,
                           const AccessorRO<  Vec3, 3>        dudt,
                           const AccessorRO<double, 3>        dTdt,
                           const AccessorRO<   int, 3>       nType,
                           const AccessorRO<double, 3>         m_d,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      UpdateUsingFluxNSCBCInflowMinusSideTask<dir>::addLODIfluxes(
                    Conserved_t[p],
                    MassFracs, pressure, velocity,
                    SoS[p], rho[p], temperature[p],
                    dudt[p], dTdt[p],
                    p, nType[p], m_d[p], mix);

   }
}

template<direction dir>
__host__
void UpdateUsingFluxNSCBCInflowMinusSideTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateUsingFluxNSCBCInflowMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_MassFracs, acc_pressure, acc_SoS, acc_rho,
                              acc_temperature, acc_velocity,
                              acc_dudt, acc_dTdt,
                              acc_nType, acc_m_d,
                              r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

template void UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCInflowMinusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCInflowMinusSideTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCInflowPlusSideTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateUsingFluxNSCBCInflowPlusSideTask_kernel(
                           const AccessorRW<VecNEq, 3> Conserved_t,
                           const AccessorRO<VecNSp, 3>   MassFracs,
                           const AccessorRO<double, 3>    pressure,
                           const AccessorRO<double, 3>         SoS,
                           const AccessorRO<double, 3>         rho,
                           const AccessorRO<double, 3> temperature,
                           const AccessorRO<  Vec3, 3>    velocity,
                           const AccessorRO<  Vec3, 3>        dudt,
                           const AccessorRO<double, 3>        dTdt,
                           const AccessorRO<   int, 3>       nType,
                           const AccessorRO<double, 3>         m_d,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      UpdateUsingFluxNSCBCInflowPlusSideTask<dir>::addLODIfluxes(
                    Conserved_t[p],
                    MassFracs, pressure, velocity,
                    SoS[p], rho[p], temperature[p],
                    dudt[p], dTdt[p],
                    p, nType[p], m_d[p], mix);

   }
}

template<direction dir>
__host__
void UpdateUsingFluxNSCBCInflowPlusSideTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateUsingFluxNSCBCInflowPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_MassFracs, acc_pressure, acc_SoS, acc_rho,
                              acc_temperature, acc_velocity, acc_dudt, acc_dTdt,
                              acc_nType, acc_m_d,
                              r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

template void UpdateUsingFluxNSCBCInflowPlusSideTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCInflowPlusSideTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCOutflowMinusSideTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
struct UpdateUsingFluxNSCBCOutflowTask_kernelArgs {
   const AccessorRW<VecNEq, 3> Conserved_t;
   const AccessorRO<   int, 3>     nType_N;
   const AccessorRO<   int, 3>    nType_T1;
   const AccessorRO<   int, 3>    nType_T2;
   const AccessorRO<double, 3>       m_d_N;
   const AccessorRO<double, 3>      m_d_T1;
   const AccessorRO<double, 3>      m_d_T2;
   const AccessorRO<VecNSp, 3>   MassFracs;
   const AccessorRO<double, 3>    pressure;
   const AccessorRO<double, 3>         SoS;
   const AccessorRO<double, 3>         rho;
   const AccessorRO<double, 3>          mu;
   const AccessorRO<double, 3> temperature;
   const AccessorRO<  Vec3, 3>    velocity;
   const AccessorRO<VecNEq, 3>   Conserved;
};

template<direction dir>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void UpdateUsingFluxNSCBCOutflowMinusSideTask_kernel(
                           const DeferredBuffer<UpdateUsingFluxNSCBCOutflowTask_kernelArgs, 1> buffer,
                           const  double MaxMach,
                           const  double LengthScale,
                           const  double PInf,
                           const Rect<3> BC_bounds,
                           const Rect<3> Fluid_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#else
void UpdateUsingFluxNSCBCOutflowMinusSideTask_kernel(
                           const UpdateUsingFluxNSCBCOutflowTask_kernelArgs a,
                           const  double MaxMach,
                           const  double LengthScale,
                           const  double PInf,
                           const Rect<3> BC_bounds,
                           const Rect<3> Fluid_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   UpdateUsingFluxNSCBCOutflowTask_kernelArgs a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      UpdateUsingFluxNSCBCOutflowMinusSideTask<dir>::addLODIfluxes(
                    a.Conserved_t[p],
                    a.nType_N, a.nType_T1, a.nType_T2,
                    a.m_d_N, a.m_d_T1, a.m_d_T2,
                    a.MassFracs, a.rho, a.mu, a.pressure, a.velocity,
                    a.SoS[p], a.temperature[p], a.Conserved[p],
                    p, Fluid_bounds, MaxMach, LengthScale, PInf, mix);
   }
}

template<direction dir>
__host__
void UpdateUsingFluxNSCBCOutflowMinusSideTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   const UpdateUsingFluxNSCBCOutflowTask_kernelArgs kArgs = {
      .Conserved_t = acc_Conserved_t,
      .nType_N     = acc_nType_N,
      .nType_T1    = acc_nType_T1,
      .nType_T2    = acc_nType_T2,
      .m_d_N       = acc_m_d_N,
      .m_d_T1      = acc_m_d_T1,
      .m_d_T2      = acc_m_d_T2,
      .MassFracs   = acc_MassFracs,
      .pressure    = acc_pressure,
      .SoS         = acc_SoS,
      .rho         = acc_rho,
      .mu          = acc_mu,
      .temperature = acc_temperature,
      .velocity    = acc_velocity,
      .Conserved   = acc_Conserved
   };

#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<UpdateUsingFluxNSCBCOutflowTask_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   UpdateUsingFluxNSCBCOutflowMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              MaxMach, args.LengthScale, args.PInf, r_BC, args.Fluid_bounds,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#else
   UpdateUsingFluxNSCBCOutflowMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              kArgs,
                              MaxMach, args.LengthScale, args.PInf, r_BC, args.Fluid_bounds,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#endif
}

template void UpdateUsingFluxNSCBCOutflowMinusSideTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCOutflowMinusSideTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCOutflowPlusSideTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
template<direction dir>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void UpdateUsingFluxNSCBCOutflowPlusSideTask_kernel(
                           const DeferredBuffer<UpdateUsingFluxNSCBCOutflowTask_kernelArgs, 1> buffer,
                           const  double MaxMach,
                           const  double LengthScale,
                           const  double PInf,
                           const Rect<3> BC_bounds,
                           const Rect<3> Fluid_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#else
void UpdateUsingFluxNSCBCOutflowPlusSideTask_kernel(
                           const UpdateUsingFluxNSCBCOutflowTask_kernelArgs a,
                           const  double MaxMach,
                           const  double LengthScale,
                           const  double PInf,
                           const Rect<3> BC_bounds,
                           const Rect<3> Fluid_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   UpdateUsingFluxNSCBCOutflowTask_kernelArgs a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::addLODIfluxes(
                    a.Conserved_t[p],
                    a.nType_N, a.nType_T1, a.nType_T2,
                    a.m_d_N, a.m_d_T1, a.m_d_T2,
                    a.MassFracs, a.rho, a.mu, a.pressure, a.velocity,
                    a.SoS[p], a.temperature[p], a.Conserved[p],
                    p, Fluid_bounds, MaxMach, LengthScale, PInf, mix);
   }
}

template<direction dir>
__host__
void UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   const UpdateUsingFluxNSCBCOutflowTask_kernelArgs kArgs = {
      .Conserved_t = acc_Conserved_t,
      .nType_N     = acc_nType_N,
      .nType_T1    = acc_nType_T1,
      .nType_T2    = acc_nType_T2,
      .m_d_N       = acc_m_d_N,
      .m_d_T1      = acc_m_d_T1,
      .m_d_T2      = acc_m_d_T2,
      .MassFracs   = acc_MassFracs,
      .pressure    = acc_pressure,
      .SoS         = acc_SoS,
      .rho         = acc_rho,
      .mu          = acc_mu,
      .temperature = acc_temperature,
      .velocity    = acc_velocity,
      .Conserved   = acc_Conserved
   };
#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<UpdateUsingFluxNSCBCOutflowTask_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   UpdateUsingFluxNSCBCOutflowPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              MaxMach, args.LengthScale, args.PInf, r_BC, args.Fluid_bounds,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#else
   UpdateUsingFluxNSCBCOutflowPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              kArgs,
                              MaxMach, args.LengthScale, args.PInf, r_BC, args.Fluid_bounds,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#endif
}

template void UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCOutflowPlusSideTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCFarFieldMinusSideTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
struct UpdateUsingFluxNSCBCFarFieldTask_kernelArgs {
   const AccessorRW<VecNEq, 3> Conserved_t;
   const AccessorRO<   int, 3> nType;
   const AccessorRO<double, 3> m_d;
   const AccessorRO<double, 3> SoS;
   const AccessorRO<double, 3> rho;
   const AccessorRO<VecNSp, 3> MassFracs;
   const AccessorRO<double, 3> pressure;
   const AccessorRO<double, 3> temperature;
   const AccessorRO<  Vec3, 3> velocity;
   const AccessorRO<VecNEq, 3> Conserved;
   const AccessorRO<VecNSp, 3> MolarFracs_profile;
   const AccessorRO<double, 3> temperature_profile;
   const AccessorRO<  Vec3, 3> velocity_profile;
};

template<direction dir>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void UpdateUsingFluxNSCBCFarFieldMinusSideTask_kernel(
                           const DeferredBuffer<UpdateUsingFluxNSCBCFarFieldTask_kernelArgs, 1> buffer,
                           const  double PInf,
                           const  double MaxMach,
                           const  double LengthScale,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#else
void UpdateUsingFluxNSCBCFarFieldMinusSideTask_kernel(
                           const UpdateUsingFluxNSCBCFarFieldTask_kernelArgs a,
                           const  double PInf,
                           const  double MaxMach,
                           const  double LengthScale,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   UpdateUsingFluxNSCBCFarFieldTask_kernelArgs a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      UpdateUsingFluxNSCBCFarFieldMinusSideTask<dir>::addLODIfluxes(
                    a.Conserved_t[p],
                    a.rho, a.MassFracs, a.pressure, a.velocity,
                    a.nType[p], a.m_d[p],
                    a.SoS[p], a.temperature[p], a.Conserved[p],
                    a.temperature_profile[p], a.velocity_profile[p], a.MolarFracs_profile[p],
                    PInf, MaxMach, LengthScale, p, mix);
   }
}

template<direction dir>
__host__
void UpdateUsingFluxNSCBCFarFieldMinusSideTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   const UpdateUsingFluxNSCBCFarFieldTask_kernelArgs kArgs = {
      .Conserved_t         = acc_Conserved_t,
      .nType               = acc_nType,
      .m_d                 = acc_m_d,
      .SoS                 = acc_SoS,
      .rho                 = acc_rho,
      .MassFracs           = acc_MassFracs,
      .pressure            = acc_pressure,
      .temperature         = acc_temperature,
      .velocity            = acc_velocity,
      .Conserved           = acc_Conserved,
      .MolarFracs_profile  = acc_MolarFracs_profile,
      .temperature_profile = acc_temperature_profile,
      .velocity_profile    = acc_velocity_profile
   };

#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<UpdateUsingFluxNSCBCFarFieldTask_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   UpdateUsingFluxNSCBCFarFieldMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              args.PInf, MaxMach, args.LengthScale, r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#else
   UpdateUsingFluxNSCBCFarFieldMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              kArgs,
                              args.PInf, MaxMach, args.LengthScale, r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#endif
}

template void UpdateUsingFluxNSCBCFarFieldMinusSideTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCFarFieldMinusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCFarFieldMinusSideTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCFarFieldPlusSideTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
template<direction dir>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void UpdateUsingFluxNSCBCFarFieldPlusSideTask_kernel(
                           const DeferredBuffer<UpdateUsingFluxNSCBCFarFieldTask_kernelArgs, 1> buffer,
                           const  double PInf,
                           const  double MaxMach,
                           const  double LengthScale,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#else
void UpdateUsingFluxNSCBCFarFieldPlusSideTask_kernel(
                           const UpdateUsingFluxNSCBCFarFieldTask_kernelArgs a,
                           const  double PInf,
                           const  double MaxMach,
                           const  double LengthScale,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   UpdateUsingFluxNSCBCFarFieldTask_kernelArgs a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      UpdateUsingFluxNSCBCFarFieldPlusSideTask<dir>::addLODIfluxes(
                    a.Conserved_t[p],
                    a.rho, a.MassFracs, a.pressure, a.velocity,
                    a.nType[p], a.m_d[p],
                    a.SoS[p], a.temperature[p], a.Conserved[p],
                    a.temperature_profile[p], a.velocity_profile[p], a.MolarFracs_profile[p],
                    PInf, MaxMach, LengthScale, p, mix);
   }
}

template<direction dir>
__host__
void UpdateUsingFluxNSCBCFarFieldPlusSideTask<dir>::gpu_base_impl(
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

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   const UpdateUsingFluxNSCBCFarFieldTask_kernelArgs kArgs = {
      .Conserved_t         = acc_Conserved_t,
      .nType               = acc_nType,
      .m_d                 = acc_m_d,
      .SoS                 = acc_SoS,
      .rho                 = acc_rho,
      .MassFracs           = acc_MassFracs,
      .pressure            = acc_pressure,
      .temperature         = acc_temperature,
      .velocity            = acc_velocity,
      .Conserved           = acc_Conserved,
      .MolarFracs_profile  = acc_MolarFracs_profile,
      .temperature_profile = acc_temperature_profile,
      .velocity_profile    = acc_velocity_profile
   };

#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<UpdateUsingFluxNSCBCFarFieldTask_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   UpdateUsingFluxNSCBCFarFieldPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              args.PInf, MaxMach, args.LengthScale, r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#else
   UpdateUsingFluxNSCBCFarFieldPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              kArgs,
                              args.PInf, MaxMach, args.LengthScale, r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#endif
}

template void UpdateUsingFluxNSCBCFarFieldPlusSideTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCFarFieldPlusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingFluxNSCBCFarFieldPlusSideTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxIncomingShockTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
struct UpdateUsingFluxIncomingShockTask_kernelArgs {
   const AccessorRW<VecNEq, 3> Conserved_t;
   const AccessorRO<   int, 3> nType;
   const AccessorRO<double, 3> m_d;
   const AccessorRO<double, 3> SoS;
   const AccessorRO<double, 3> rho;
   const AccessorRO<VecNSp, 3> MassFracs;
   const AccessorRO<double, 3> pressure;
   const AccessorRO<double, 3> temperature;
   const AccessorRO<  Vec3, 3> velocity;
   const AccessorRO<VecNEq, 3> Conserved;
};

template<direction dir>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void UpdateUsingFluxIncomingShockTask_kernel(
                           const DeferredBuffer<UpdateUsingFluxIncomingShockTask_kernelArgs, 1> buffer,
                           const  double MaxMach,
                           const  double LengthScale,
                           const Vec3 velocity0,
                           const double temperature0,
                           const double pressure0,
                           const Vec3 velocity1,
                           const double temperature1,
                           const double pressure1,
                           const VecNSp MolarFracs0,
                           const int iShock,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#else
void UpdateUsingFluxIncomingShockTask_kernel(
                           const UpdateUsingFluxIncomingShockTask_kernelArgs a,
                           const  double MaxMach,
                           const  double LengthScale,
                           const Vec3 velocity0,
                           const double temperature0,
                           const double pressure0,
                           const Vec3 velocity1,
                           const double temperature1,
                           const double pressure1,
                           const VecNSp MolarFracs0,
                           const int iShock,
                           const Rect<3> BC_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   UpdateUsingFluxIncomingShockTask_kernelArgs a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + BC_bounds.lo.x,
                                  y + BC_bounds.lo.y,
                                  z + BC_bounds.lo.z);
      if (p.x < iShock)
         // Set to upstream values
         UpdateUsingFluxNSCBCFarFieldPlusSideTask<dir>::addLODIfluxes(
                    a.Conserved_t[p],
                    a.rho, a.MassFracs, a.pressure, a.velocity,
                    a.nType[p], a.m_d[p],
                    a.SoS[p], a.temperature[p], a.Conserved[p],
                    temperature0, velocity0, MolarFracs0,
                    pressure0, MaxMach, LengthScale, p, mix);
      else
         // Set to downstream values
         UpdateUsingFluxNSCBCFarFieldPlusSideTask<dir>::addLODIfluxes(
                    a.Conserved_t[p],
                    a.rho, a.MassFracs, a.pressure, a.velocity,
                    a.nType[p], a.m_d[p],
                    a.SoS[p], a.temperature[p], a.Conserved[p],
                    temperature1, velocity1, MolarFracs0,
                    pressure1, MaxMach, LengthScale, p, mix);
   }
}

template<direction dir>
__host__
void UpdateUsingFluxIncomingShockTask<dir>::gpu_base_impl(
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
   VecNSp MolarFracs(args.params.MolarFracs);

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   const UpdateUsingFluxIncomingShockTask_kernelArgs kArgs = {
      .Conserved_t         = acc_Conserved_t,
      .nType               = acc_nType,
      .m_d                 = acc_m_d,
      .SoS                 = acc_SoS,
      .rho                 = acc_rho,
      .MassFracs           = acc_MassFracs,
      .pressure            = acc_pressure,
      .temperature         = acc_temperature,
      .velocity            = acc_velocity,
      .Conserved           = acc_Conserved
   };

#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<UpdateUsingFluxIncomingShockTask_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   UpdateUsingFluxIncomingShockTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              MaxMach, args.LengthScale,
                              Vec3(args.params.velocity0), args.params.temperature0, args.params.pressure0,
                              Vec3(args.params.velocity1), args.params.temperature1, args.params.pressure1,
                              MolarFracs, args.params.iShock,
                              r_BC, getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#else
   UpdateUsingFluxIncomingShockTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              kArgs,
                              MaxMach, args.LengthScale,
                              Vec3(args.params.velocity0), args.params.temperature0, args.params.pressure0,
                              Vec3(args.params.velocity1), args.params.temperature1, args.params.pressure1,
                              MolarFracs, args.params.iShock,
                              r_BC, getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#endif
}

template void UpdateUsingFluxIncomingShockTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR CalculateAveragePDTask
//-----------------------------------------------------------------------------

__global__
void CalculateAveragePD_kernel(const DeferredBuffer<double, 1> buffer,
                               const AccessorRO<   int, 3> nType_x,
                               const AccessorRO<   int, 3> nType_y,
                               const AccessorRO<   int, 3> nType_z,
                               const AccessorRO<double, 3> dcsi_d,
                               const AccessorRO<double, 3> deta_d,
                               const AccessorRO<double, 3> dzet_d,
                               const AccessorRO<double, 3> pressure,
                               const AccessorRO<  Vec3, 3> velocity,
                               const Rect<3> my_bounds,
                               const Rect<3> Fluid_bounds,
                               const coord_t size_x,
                               const coord_t size_y,
                               const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   double my_PD = 0.0;
   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      const double cellVolume = 1.0/(dcsi_d[p]*deta_d[p]*dzet_d[p]);
      my_PD = cellVolume*CalculateAveragePDTask::CalculatePressureDilatation(
                          nType_x, nType_y, nType_z,
                          dcsi_d, deta_d, dzet_d,
                          pressure, velocity,
                          p, Fluid_bounds);
   }
   reduceSum(my_PD, buffer);
}

__host__
DeferredValue<double> CalculateAveragePDTask::gpu_base_impl(
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
   const Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Define thread grid
   const int threads_per_block = 256;
   dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   while (TPB_3d.x*TPB_3d.y*TPB_3d.z < 32) TPB_3d.x++;
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);

   // Store the integral per block in a deferred buffer
   const size_t total_blocks = num_blocks_3d.x*num_blocks_3d.y*num_blocks_3d.z;
   const Rect<1> bounds(Point<1>(0), Point<1>(total_blocks - 1));
   DeferredBuffer<double, 1> buffer(bounds, Memory::GPU_FB_MEM);
   CalculateAveragePD_kernel<<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              acc_nType_x, acc_nType_y, acc_nType_z,
                              acc_dcsi_d, acc_deta_d, acc_dzet_d,
                              acc_pressure, acc_velocity,
                              r_MyFluid, args.Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));

   // Reduce pressure dilatation into PD
   DeferredValue<double> PD(0.0);

   // We use at most 1024 blocks
   dim3 TPB((total_blocks > 1024) ? 1024 : total_blocks, 1, 1);
   // Round up to the nearest multiple of warps
   while ((TPB.x % 32) != 0) TPB.x++;
   const dim3 num_blocks(1, 1, 1);
   ReduceBufferSum_kernel<<<num_blocks, TPB>>>(buffer, PD, total_blocks);

   return PD;
}

//-----------------------------------------------------------------------------
// KERNELS FOR AddDissipationTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void AddDissipation_kernel(const DeferredBuffer<double, 1> buffer,
                           const AccessorRO<   int, 3>       nType,
                           const AccessorRO<   int, 3>      nType1,
                           const AccessorRO<   int, 3>      nType2,
                           const AccessorRO<double, 3>         m_s,
                           const AccessorRO<double, 3>        m_d1,
                           const AccessorRO<double, 3>        m_d2,
                           const AccessorRO<double, 3>          mu,
                           const AccessorRO<  Vec3, 3>    velocity,
                           const Rect<3> Flux_bounds,
                           const Rect<3> Fluid_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   double my_r = 0.0;
   if (isThreadInCrossPlane<dir>(size_x, size_y, size_z)) {
      const coord_t size = getSize<dir>(Fluid_bounds);
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         my_r = AddDissipationTask<dir>::AddSpan(
                  m_s, m_d1, m_d2,
                  nType, nType1, nType2,
                  mu, velocity,
                  firstIndex, lastIndex,
                  x, y, z,
                  Flux_bounds, Fluid_bounds, mix);
      }
   }
   reduceSum(my_r, buffer);
}

template<direction dir>
__host__
DeferredValue<double> AddDissipationTask<dir>::gpu_base_impl(
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
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[3].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel that stores integrated dissipation for each block in a deferred buffer
   const int threads_per_block = 256;
   dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_MyFluid);
   while (TPB_3d.x*TPB_3d.y*TPB_3d.z < 32) TPB_3d.x++;
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_MyFluid);

   // Store the integral per block in a deferred buffer
   const size_t total_blocks = num_blocks_3d.x*num_blocks_3d.y*num_blocks_3d.z;
   const Rect<1> bounds(Point<1>(0), Point<1>(total_blocks - 1));
   DeferredBuffer<double, 1> buffer(bounds, Memory::GPU_FB_MEM);
   AddDissipation_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              buffer,
                              acc_nType, acc_nType1, acc_nType2,
                              acc_m_s, acc_m_d1, acc_m_d2,
                              acc_mu, acc_velocity,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));

   // Reduce dissipation into r
   DeferredValue<double> r(0.0);

   // We use at most 1024 blocks
   dim3 TPB((total_blocks > 1024) ? 1024 : total_blocks, 1, 1);
   // Round up to the nearest multiple of warps
   while ((TPB.x % 32) != 0) TPB.x++;
   const dim3 num_blocks(1, 1, 1);
   ReduceBufferSum_kernel<<<num_blocks, TPB>>>(buffer, r, total_blocks);

   return r;
}

template DeferredValue<double> AddDissipationTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template DeferredValue<double> AddDissipationTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template DeferredValue<double> AddDissipationTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);


