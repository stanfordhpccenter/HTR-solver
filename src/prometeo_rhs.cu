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
                                     const AccessorRO<double, 3>         m_e,
                                     const AccessorRO<VecNEq, 3>   Conserved,
                                     const AccessorRO<double, 3>         rho,
                                     const AccessorRO<double, 3>         SoS,
                                     const AccessorRO<VecNSp, 3>   MassFracs,
                                     const AccessorRO<  Vec3, 3>    velocity,
                                     const AccessorRO<double, 3>    pressure,
                                     const AccessorRO<double, 3> temperature,
                                     const AccessorRO<   int, 3>       nType,
                                     const AccessorRO<  bool, 3> shockSensor,
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
      const coord_t size = getSize<dir>(Fluid_bounds);
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         // Allocate a buffer for the summations of the KG scheme of size (3 * (3+1)/2 * (nEq+1)) for each thread
         double KGSum[6*(nEq+1)];
         // Launch the loop for the span
         UpdateUsingHybridEulerFluxTask<dir>::updateRHSSpan(
            KGSum,
            Conserved_t, m_e, nType, shockSensor,
            Conserved, rho, SoS,
            MassFracs, velocity, pressure, temperature,
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
   assert(regions.size() == 7);
   assert(futures.size() == 0);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved(regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_SoS      (regions[0], FID_SoS);
   const AccessorRO<double, 3> acc_rho      (regions[0], FID_rho);
   const AccessorRO<VecNSp, 3> acc_MassFracs(regions[0], FID_MassFracs);
   const AccessorRO<  Vec3, 3> acc_velocity (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure (regions[0], FID_pressure);

   // Accessors for shock sensor
   const AccessorRO<  bool, 3> acc_shockSensor(regions[1], FID_shockSensor);

   // Accessors for quantities needed for the Roe averages
   const AccessorRO<double, 3> acc_temperature(regions[2], FID_temperature);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[3], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e(regions[4], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[5], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with hybrid scheme
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_ModCells);
   UpdateRHSUsingHybridFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved, acc_rho, acc_SoS,
                              acc_MassFracs, acc_velocity, acc_pressure, acc_temperature,
                              acc_nType, acc_shockSensor,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
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
// KERNELS FOR UpdateUsingTENOAEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateRHSUsingTENOAFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                    const AccessorRO<double, 3>         m_e,
                                    const AccessorRO<VecNEq, 3>   Conserved,
                                    const AccessorRO<double, 3>         SoS,
                                    const AccessorRO<double, 3>         rho,
                                    const AccessorRO<  Vec3, 3>    velocity,
                                    const AccessorRO<double, 3>    pressure,
                                    const AccessorRO<VecNSp, 3>   MassFracs,
                                    const AccessorRO<double, 3> temperature,
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
         // Launch the loop for the span
         UpdateUsingTENOAEulerFluxTask<dir>::updateRHSSpan(
            Conserved_t, m_e, nType,
            Conserved, rho, SoS,
            MassFracs, velocity, pressure, temperature,
            firstIndex, lastIndex,
            x, y, z,
            Flux_bounds, Fluid_bounds, mix);
      }
   }
}

template<direction dir>
__host__
void UpdateUsingTENOAEulerFluxTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 6);
   assert(futures.size() == 0);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved(regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_SoS      (regions[0], FID_SoS);
   const AccessorRO<double, 3> acc_rho      (regions[0], FID_rho);
   const AccessorRO<  Vec3, 3> acc_velocity (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure (regions[0], FID_pressure);

   // Accessors for quantities needed for the Roe averages
   const AccessorRO<double, 3> acc_temperature(regions[1], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e(regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with TENOA
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_ModCells);
   UpdateRHSUsingTENOAFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                              acc_pressure, acc_MassFracs, acc_temperature, acc_nType,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
}

// Force the compiler to instanciate these functions
template void UpdateUsingTENOAEulerFluxTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOAEulerFluxTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOAEulerFluxTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingTENOLADEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateRHSUsingTENOLADFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                      const AccessorRO<double, 3>         m_e,
                                      const AccessorRO<VecNEq, 3>   Conserved,
                                      const AccessorRO<double, 3>         SoS,
                                      const AccessorRO<double, 3>         rho,
                                      const AccessorRO<  Vec3, 3>    velocity,
                                      const AccessorRO<double, 3>    pressure,
                                      const AccessorRO<VecNSp, 3>   MassFracs,
                                      const AccessorRO<double, 3> temperature,
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
         // Launch the loop for the span
         UpdateUsingTENOLADEulerFluxTask<dir>::updateRHSSpan(
            Conserved_t, m_e, nType,
            Conserved, rho, SoS,
            MassFracs, velocity, pressure, temperature,
            firstIndex, lastIndex,
            x, y, z,
            Flux_bounds, Fluid_bounds, mix);
      }
   }
}

template<direction dir>
__host__
void UpdateUsingTENOLADEulerFluxTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 6);
   assert(futures.size() == 0);

   // Accessors for variables in the Flux stencil
   const AccessorRO<VecNEq, 3> acc_Conserved(regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_SoS      (regions[0], FID_SoS);
   const AccessorRO<double, 3> acc_rho      (regions[0], FID_rho);
   const AccessorRO<  Vec3, 3> acc_velocity (regions[0], FID_velocity);
   const AccessorRO<double, 3> acc_pressure (regions[0], FID_pressure);

   // Accessors for quantities needed for the Roe averages
   const AccessorRO<double, 3> acc_temperature(regions[1], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e(regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with TENOLAD
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_ModCells);
   UpdateRHSUsingTENOLADFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                              acc_pressure, acc_MassFracs, acc_temperature, acc_nType,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
}

// Force the compiler to instanciate these functions
template void UpdateUsingTENOLADEulerFluxTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOLADEulerFluxTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingTENOLADEulerFluxTask<Zdir>::gpu_base_impl(
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
   assert(regions.size() == 5);
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
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the Euler fluxes reconstructed with KG
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_ModCells);
   UpdateRHSUsingKGFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t, acc_m_e,
                              acc_Conserved, acc_rho, acc_MassFracs,
                              acc_velocity, acc_pressure, acc_nType,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
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

template<direction dir>
__global__
void UpdateRHSUsingDiffusionFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                        const AccessorRO<   int, 3>       nType,
                                        const AccessorRO<double, 3>         m_s,
                                        const AccessorRO<double, 3>         m_d,
                                        const AccessorRO<double, 3>         rho,
                                        const AccessorRO<double, 3>          mu,
                                        const AccessorRO<double, 3>         lam,
                                        const AccessorRO<VecNSp, 3>          Di,
                                        const AccessorRO<double, 3> temperature,
                                        const AccessorRO<  Vec3, 3>    velocity,
                                        const AccessorRO<VecNSp, 3>          Xi,
                                        const AccessorRO<VecNEq, 3>       rhoYi,
                                        const AccessorRO<  Vec3, 3>      vGrad1,
                                        const AccessorRO<  Vec3, 3>      vGrad2,
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
      const coord_t size = getSize<dir>(Fluid_bounds);
      const coord_t span_size = getSize<dir>(Flux_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex = lastIndexInSpan<dir>(span_size);
         UpdateUsingDiffusionFluxTask<dir>::updateRHSSpan(
                  Conserved_t, m_s, m_d, nType,
                  rho, mu, lam, Di,
                  temperature, velocity, Xi, rhoYi,
                  vGrad1, vGrad2,
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
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);
   const AccessorRO<VecNSp, 3> acc_MolarFracs (regions[0], FID_MolarFracs);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_mu         (regions[0], FID_mu);
   const AccessorRO<double, 3> acc_lam        (regions[0], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di         (regions[0], FID_Di);
   const AccessorRO<  Vec3, 3> acc_vGrad1     (regions[0], FID_vGrad1);
   const AccessorRO<  Vec3, 3> acc_vGrad2     (regions[0], FID_vGrad2);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[1], FID_nType);
   const AccessorRO<double, 3> acc_m_s        (regions[1], FID_m_s);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_d        (regions[2], FID_m_d);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[3], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the diffusion fluxes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_ModCells);
   UpdateRHSUsingDiffusionFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_nType, acc_m_s, acc_m_d,
                              acc_rho, acc_mu, acc_lam, acc_Di,
                              acc_temperature, acc_velocity, acc_MolarFracs,
                              acc_Conserved, acc_vGrad1, acc_vGrad2,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
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
                           const AccessorRO<  Vec3, 3>       vGrad,
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
                    MassFracs, pressure,
                    SoS[p], rho[p], temperature[p],
                    velocity[p], vGrad[p], dudt[p], dTdt[p],
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
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for Input data
   const AccessorRO<   int, 3> acc_nType      (regions[0], FID_nType);
   const AccessorRO<double, 3> acc_m_d        (regions[0], FID_m_d);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_SoS        (regions[0], FID_SoS);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature(regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);
   const AccessorRO<  Vec3, 3> acc_vGrad      (regions[0], FID_vGrad);
   const AccessorRO<  Vec3, 3> acc_dudt       (regions[0], FID_dudtBoundary);
   const AccessorRO<double, 3> acc_dTdt       (regions[0], FID_dTdtBoundary);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[1], FID_Conserved_t);

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateUsingFluxNSCBCInflowMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_MassFracs, acc_pressure, acc_SoS, acc_rho,
                              acc_temperature, acc_velocity, acc_vGrad, acc_dudt, acc_dTdt,
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
                           const AccessorRO<  Vec3, 3>       vGrad,
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
                    MassFracs, pressure,
                    SoS[p], rho[p], temperature[p],
                    velocity[p], vGrad[p], dudt[p], dTdt[p],
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
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for Input data
   const AccessorRO<   int, 3> acc_nType      (regions[0], FID_nType);
   const AccessorRO<double, 3> acc_m_d        (regions[0], FID_m_d);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_SoS        (regions[0], FID_SoS);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature(regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);
   const AccessorRO<  Vec3, 3> acc_vGrad      (regions[0], FID_vGrad);
   const AccessorRO<  Vec3, 3> acc_dudt       (regions[0], FID_dudtBoundary);
   const AccessorRO<double, 3> acc_dTdt       (regions[0], FID_dTdtBoundary);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[1], FID_Conserved_t);

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateUsingFluxNSCBCInflowPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_MassFracs, acc_pressure, acc_SoS, acc_rho,
                              acc_temperature, acc_velocity, acc_vGrad, acc_dudt, acc_dTdt,
                              acc_nType, acc_m_d,
                              r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

template void UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCOutflowMinusSideTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateUsingFluxNSCBCOutflowMinusSideTask_kernel(
                           const AccessorRW<VecNEq, 3> Conserved_t,
                           const AccessorRO<VecNSp, 3>   MassFracs,
                           const AccessorRO<double, 3>    pressure,
                           const AccessorRO<double, 3>         SoS,
                           const AccessorRO<double, 3>         rho,
                           const AccessorRO<double, 3>          mu,
                           const AccessorRO<double, 3> temperature,
                           const AccessorRO<  Vec3, 3>    velocity,
                           const AccessorRO<VecNEq, 3>   Conserved,
                           const AccessorRO<  Vec3, 3>      vGradN,
                           const AccessorRO<  Vec3, 3>     vGradT1,
                           const AccessorRO<  Vec3, 3>     vGradT2,
                           const AccessorRO<   int, 3>       nType,
                           const AccessorRO<double, 3>         m_d,
                           const double     MaxMach,
                           const double LengthScale,
                           const double        PInf,
                           const Rect<3>  BC_bounds,
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
      UpdateUsingFluxNSCBCOutflowMinusSideTask<dir>::addLODIfluxes(
                    Conserved_t[p],
                    MassFracs, rho, mu, pressure,
                    velocity, vGradN, vGradT1, vGradT2,
                    SoS[p], temperature[p], Conserved[p],
                    p, nType[p], m_d[p],
                    MaxMach, LengthScale, PInf, mix);
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
   assert(regions.size() == 2);
   assert(futures.size() == 1);

   // Accessors for Input data
   const AccessorRO<   int, 3> acc_nType      (regions[0], FID_nType);
   const AccessorRO<double, 3> acc_m_d        (regions[0], FID_m_d);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_mu         (regions[0], FID_mu);
   const AccessorRO<double, 3> acc_SoS        (regions[0], FID_SoS);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature(regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_vGradN     (regions[0], FID_vGradN);
   const AccessorRO<  Vec3, 3> acc_vGradT1    (regions[0], FID_vGradT1);
   const AccessorRO<  Vec3, 3> acc_vGradT2    (regions[0], FID_vGradT2);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[1], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateUsingFluxNSCBCOutflowMinusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_MassFracs, acc_pressure, acc_SoS, acc_rho, acc_mu,
                              acc_temperature, acc_velocity, acc_Conserved,
                              acc_vGradN, acc_vGradT1, acc_vGradT2,
                              acc_nType, acc_m_d,
                              MaxMach, args.LengthScale, args.PInf, r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

template void UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingFluxNSCBCOutflowPlusSideTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateUsingFluxNSCBCOutflowPlusSideTask_kernel(
                           const AccessorRW<VecNEq, 3> Conserved_t,
                           const AccessorRO<VecNSp, 3>   MassFracs,
                           const AccessorRO<double, 3>    pressure,
                           const AccessorRO<double, 3>         SoS,
                           const AccessorRO<double, 3>         rho,
                           const AccessorRO<double, 3>          mu,
                           const AccessorRO<double, 3> temperature,
                           const AccessorRO<  Vec3, 3>    velocity,
                           const AccessorRO<VecNEq, 3>   Conserved,
                           const AccessorRO<  Vec3, 3>      vGradN,
                           const AccessorRO<  Vec3, 3>     vGradT1,
                           const AccessorRO<  Vec3, 3>     vGradT2,
                           const AccessorRO<   int, 3>       nType,
                           const AccessorRO<double, 3>         m_d,
                           const double     MaxMach,
                           const double LengthScale,
                           const double        PInf,
                           const Rect<3>  BC_bounds,
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
      UpdateUsingFluxNSCBCOutflowPlusSideTask<dir>::addLODIfluxes(
                    Conserved_t[p],
                    MassFracs, rho, mu, pressure,
                    velocity, vGradN, vGradT1, vGradT2,
                    SoS[p], temperature[p], Conserved[p],
                    p, nType[p], m_d[p],
                    MaxMach, LengthScale, PInf, mix);
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
   assert(regions.size() == 2);
   assert(futures.size() == 1);

   // Accessors for Input data
   const AccessorRO<   int, 3> acc_nType      (regions[0], FID_nType);
   const AccessorRO<double, 3> acc_m_d        (regions[0], FID_m_d);
   const AccessorRO<double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_mu         (regions[0], FID_mu);
   const AccessorRO<double, 3> acc_SoS        (regions[0], FID_SoS);
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_pressure   (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature(regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity   (regions[0], FID_velocity);
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_vGradN     (regions[0], FID_vGradN);
   const AccessorRO<  Vec3, 3> acc_vGradT1    (regions[0], FID_vGradT1);
   const AccessorRO<  Vec3, 3> acc_vGradT2    (regions[0], FID_vGradT2);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[1], FID_Conserved_t);

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   // Launch the kernel to update the RHS NSCBC nodes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateUsingFluxNSCBCOutflowPlusSideTask_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_MassFracs, acc_pressure, acc_SoS, acc_rho, acc_mu,
                              acc_temperature, acc_velocity, acc_Conserved,
                              acc_vGradN, acc_vGradT1, acc_vGradT2,
                              acc_nType, acc_m_d,
                              MaxMach, args.LengthScale, args.PInf, r_BC,
                              getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
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

