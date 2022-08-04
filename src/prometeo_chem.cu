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

#include "prometeo_chem.hpp"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateChemistryTask
//-----------------------------------------------------------------------------

__global__
void UpdateChemistry_kernel(const AccessorRO<VecNEq, 3> Conserved_t,
                            const AccessorRW<VecNEq, 3> Conserved,
                            const AccessorRW<VecNEq, 3> Conserved_t_old,
                            const AccessorRW<double, 3> temperature,
                            const double Integrator_deltaTime,
                            const Rect<3> my_bounds,
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
      Conserved_t_old[p] = Conserved_t[p];
      ImplicitSolver s = ImplicitSolver(Conserved_t_old[p], temperature[p], mix);
      s.solve(Conserved[p], Integrator_deltaTime, Integrator_deltaTime);
   }
}

__host__
void UpdateChemistryTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 1);

   // Accessors for RHS
   const AccessorRO<VecNEq, 3> acc_Conserved_t      (regions[0], FID_Conserved_t);

   // Accessors for implicit variables
   const AccessorRW<VecNEq, 3> acc_Conserved        (regions[1], FID_Conserved);
   const AccessorRW<VecNEq, 3> acc_Conserved_t_old  (regions[1], FID_Conserved_t_old);
   const AccessorRW<double, 3> acc_temperature      (regions[1], FID_temperature);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Wait for the Integrator_deltaTime
   const double Integrator_deltaTime = futures[0].get_result<double>();

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateChemistry_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_Conserved_t, acc_Conserved, acc_Conserved_t_old, acc_temperature, Integrator_deltaTime,
                        r_MyFluid, getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}


//-----------------------------------------------------------------------------
// KERNELS FOR AddChemistrySourcesTask
//-----------------------------------------------------------------------------

__global__
void AddChemistrySources_kernel(const AccessorRO<double, 3> rho,
                                const AccessorRO<double, 3> pressure,
                                const AccessorRO<double, 3> temperature,
                                const AccessorRO<VecNSp, 3> MassFracs,
                                const AccessorRW<VecNEq, 3> Conserved_t,
                                const Rect<3> my_bounds,
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
      VecNSp w; mix.GetProductionRates(w, rho[p], pressure[p], temperature[p], MassFracs[p]);
      __UNROLL__
      for (int i = 0; i<nSpec; i++)
         Conserved_t[p][i] += w[i];
   }
}

__host__
void AddChemistrySourcesTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for primitive variables and properites
   const AccessorRO<double, 3> acc_rho              (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_pressure         (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MassFracs        (regions[0], FID_MassFracs);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t      (regions[1], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   AddChemistrySources_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_rho, acc_pressure, acc_temperature, acc_MassFracs, acc_Conserved_t,
                        r_MyFluid, getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

