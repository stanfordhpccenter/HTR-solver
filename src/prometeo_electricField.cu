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

#include "prometeo_electricField.hpp"
#include "prometeo_electricField.inl"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNELS FOR GetElectricFieldTask
//-----------------------------------------------------------------------------

__global__
void GetElectricField_kernel(const AccessorWO<  Vec3, 3>  eField,
                             const AccessorRO<double, 3>    ePot,
                             const AccessorRO<   int, 3> nType_x,
                             const AccessorRO<   int, 3> nType_y,
                             const AccessorRO<   int, 3> nType_z,
                             const AccessorRO<double, 3>  dcsi_d,
                             const AccessorRO<double, 3>  deta_d,
                             const AccessorRO<double, 3>  dzet_d,
                             const Rect<3> my_bounds,
                             const Rect<3> Fluid_bounds,
                             const coord_t  size_x,
                             const coord_t  size_y,
                             const coord_t  size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      eField[p] = -getGrad(ePot, p,
                   nType_x[p], nType_y[p], nType_z[p],
                   dcsi_d[p], deta_d[p], dzet_d[p],
                   Fluid_bounds);
   }
}

__host__
void GetElectricFieldTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<double, 3> acc_ePot    (regions[0], FID_electricPotential);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d  (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d  (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d  (regions[1], FID_dzet_d);

   // Accessors for gradients
   const AccessorWO<  Vec3, 3> acc_eField  (regions[2], FID_electricField);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   GetElectricField_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_eField, acc_ePot,
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        r_MyFluid, Fluid_bounds,
                        getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

#if (nIons > 0)
//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingIonDriftFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateRHSUsingIonDriftFlux_kernel(const AccessorRW<VecNEq, 3> Conserved_t,
                                        const AccessorRO<   int, 3>       nType,
                                        const AccessorRO<double, 3>         m_e,
                                        const AccessorRO<VecNEq, 3>       rhoYi,
                                        const AccessorRO<VecNSp, 3>          Yi,
                                        const AccessorRO<VecNIo, 3>          Ki,
                                        const AccessorRO<  Vec3, 3>      eField,
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
         UpdateUsingIonDriftFluxTask<dir>::updateRHSSpan(
                  Conserved_t, m_e, nType,
                  rhoYi, Yi, Ki, eField,
                  firstIndex, lastIndex,
                  x, y, z,
                  Flux_bounds, Fluid_bounds, mix);
      }
   }
}

template<direction dir>
__host__
void UpdateUsingIonDriftFluxTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 0);

   // Accessors for EulerGhost region
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_eField     (regions[0], FID_electricField);
   const AccessorRO<VecNIo, 3> acc_Ki         (regions[0], FID_Ki);

   // Accessors for DiffGhost region
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e        (regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[4].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using the diffusion fluxes
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlockSpan<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = numBlocksSpan<dir>(TPB_3d, r_MyFluid);
   UpdateRHSUsingIonDriftFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_nType, acc_m_e,
                              acc_Conserved, acc_MassFracs, acc_Ki, acc_eField,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

template void UpdateUsingIonDriftFluxTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingIonDriftFluxTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateUsingIonDriftFluxTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);
#endif

//-----------------------------------------------------------------------------
// KERNELS FOR AddIonWindSourcesTask
//-----------------------------------------------------------------------------

__global__
void AddIonWindSourcesTask_kernel(
                           const AccessorRW<VecNEq, 3> Conserved_t,
                           const AccessorRO<double, 3>         rho,
                           const AccessorRO<VecNSp, 3>          Di,
#if (nIons > 0)
                           const AccessorRO<VecNIo, 3>          Ki,
#endif
                           const AccessorRO<  Vec3, 3>    velocity,
                           const AccessorRO<  Vec3, 3>      eField,
                           const AccessorRO<VecNSp, 3>  MolarFracs,
                           const AccessorRO<   int, 3>     nType_x,
                           const AccessorRO<   int, 3>     nType_y,
                           const AccessorRO<   int, 3>     nType_z,
                           const AccessorRO<double, 3>         m_x,
                           const AccessorRO<double, 3>         m_y,
                           const AccessorRO<double, 3>         m_z,
                           const Rect<3> my_bounds,
                           const Rect<3> Fluid_bounds,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      AddIonWindSourcesTask::addIonWindSources(Conserved_t[p],
               rho, Di,
#if (nIons > 0)
               Ki,
#endif
               velocity, eField, MolarFracs,
               nType_x, nType_y, nType_z,
               m_x, m_y, m_z,
               p, Fluid_bounds, mix);
   }
}

__host__
void AddIonWindSourcesTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for stencil variables
   const AccessorRO<VecNSp, 3> acc_MolarFracs       (regions[0], FID_MolarFracs);

   // Accessors for node types and metrics
   const AccessorRO<   int, 3> acc_nType_x          (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi             (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta             (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet             (regions[1], FID_dzet_d);

   // Accessors for primitive variables
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[1], FID_velocity);
   const AccessorRO<  Vec3, 3> acc_eField           (regions[1], FID_electricField);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho              (regions[1], FID_rho);
   const AccessorRO<VecNSp, 3> acc_Di               (regions[1], FID_Di);
#if (nIons > 0)
   const AccessorRO<VecNIo, 3> acc_Ki               (regions[1], FID_Ki);
#endif

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t      (regions[2], FID_Conserved_t);

   // Extract execution domain
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Launch the kernel to update the RHS using ion wind source terms
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   AddIonWindSourcesTask_kernel<<<num_blocks_3d, TPB_3d>>>(
                              acc_Conserved_t,
                              acc_rho, acc_Di,
#if (nIons > 0)
                              acc_Ki,
#endif
                              acc_velocity, acc_eField, acc_MolarFracs,
                              acc_nType_x, acc_nType_y, acc_nType_z,
                              acc_dcsi, acc_deta, acc_dzet,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

