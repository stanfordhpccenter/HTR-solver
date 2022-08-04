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

#include "prometeo_cfl.hpp"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNEL FOR CalculateMaxSpectralRadiusTask
//-----------------------------------------------------------------------------

__global__
void CalculateMaxSpectralRadius_kernel(const DeferredBuffer<double, 1> buffer,
                                       const AccessorRO<double, 3> dcsi,
                                       const AccessorRO<double, 3> deta,
                                       const AccessorRO<double, 3> dzet,
                                       const AccessorRO<double, 3> temperature,
                                       const AccessorRO<VecNSp, 3> MassFracs,
                                       const AccessorRO<  Vec3, 3> velocity,
                                       const AccessorRO<double, 3> rho,
                                       const AccessorRO<double, 3> mu,
                                       const AccessorRO<double, 3> lam,
                                       const AccessorRO<VecNSp, 3> Di,
                                       const AccessorRO<double, 3> SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                       const AccessorRO<VecNIo, 3> Ki,
                                       const AccessorRO<  Vec3, 3> eField,
#endif
                                       const Rect<3> my_bounds,
                                       const coord_t size_x,
                                       const coord_t size_y,
                                       const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   double my_r = 0.0; // Spectral radius cannot be lower than 0
   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      my_r = CalculateMaxSpectralRadiusTask::CalculateMaxSpectralRadius(
                                    dcsi, deta, dzet,
                                    temperature, MassFracs, velocity,
                                    rho, mu, lam, Di, SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                    Ki, eField,
#endif
                                    p, mix);
   }
   reduceMax(my_r, buffer);
}

__host__
DeferredValue<double> CalculateMaxSpectralRadiusTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 1);
   assert(futures.size() == 0);

   // Accessor for metrics
   const AccessorRO<double, 3> acc_dcsi_d           (regions[0], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d           (regions[0], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d           (regions[0], FID_dzet_d);

   // Accessors for primitive variables
   const AccessorRO<VecNSp, 3> acc_MassFracs        (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho              (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_mu               (regions[0], FID_mu);
   const AccessorRO<double, 3> acc_lam              (regions[0], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di               (regions[0], FID_Di);
   const AccessorRO<double, 3> acc_SoS              (regions[0], FID_SoS);
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   const AccessorRO<VecNIo, 3> acc_Ki               (regions[0], FID_Ki);

   // Accessors for primitive variables
   const AccessorRO<  Vec3, 3> acc_eField           (regions[0], FID_electricField);
#endif

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

   // Define thread grid
   const int threads_per_block = 256;
   dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   while (TPB_3d.x*TPB_3d.y*TPB_3d.z < 32) TPB_3d.x++;
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);

   // Store the maximum value per block in a deferred buffer
   const size_t total_blocks = num_blocks_3d.x*num_blocks_3d.y*num_blocks_3d.z;
   const Rect<1> bounds(Point<1>(0), Point<1>(total_blocks - 1));
   DeferredBuffer<double, 1> buffer(bounds, Memory::GPU_FB_MEM);
   CalculateMaxSpectralRadius_kernel<<<num_blocks_3d, TPB_3d>>>(buffer,
                           acc_dcsi_d, acc_deta_d, acc_dzet_d,
                           acc_temperature, acc_MassFracs, acc_velocity,
                           acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                           acc_Ki, acc_eField,
#endif
                           r_MyFluid, getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));

   // Reduce spectral radii into r
   DeferredValue<double> r(0.0);

   // We use at most 1024 blocks
   dim3 TPB((total_blocks > 1024) ? 1024 : total_blocks, 1, 1);
   // Round up to the nearest multiple of warps
   while ((TPB.x % 32) != 0) TPB.x++;
   const dim3 num_blocks(1, 1, 1);
   ReduceBufferMax_kernel<<<num_blocks, TPB>>>(buffer, r, total_blocks);

   return r;
}

