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

#include "prometeo_variables.hpp"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNELS FOR UpdatePropertiesFromPrimitiveTask
//-----------------------------------------------------------------------------

__global__
void UpdatePropertiesFromPrimitive_kernel(const AccessorRO<double, 3> pressure,
                                          const AccessorRO<double, 3> temperature,
                                          const AccessorRO<VecNSp, 3> MolarFracs,
                                          const AccessorRO<  Vec3, 3> velocity,
                                          const AccessorWO<VecNSp, 3> MassFracs,
                                          const AccessorWO<double, 3> rho,
                                          const AccessorWO<double, 3> mu,
                                          const AccessorWO<double, 3> lam,
                                          const AccessorWO<VecNSp, 3> Di,
                                          const AccessorWO<double, 3> SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                          const AccessorWO<VecNIo, 3> Ki,
#endif
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
      // Mixture check
      assert(mix.CheckMixture(MolarFracs[p]));
      UpdatePropertiesFromPrimitiveTask::UpdateProperties(
                       pressure, temperature, MolarFracs, velocity,
                       MassFracs,
                       rho, mu, lam, Di, SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                       Ki,
#endif
                       p, mix);
   }
}

__host__
void UpdatePropertiesFromPrimitiveTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for primitive variables
   const AccessorRO<double, 3> acc_pressure         (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MolarFracs       (regions[0], FID_MolarFracs);
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   const AccessorWO<VecNSp, 3> acc_MassFracs        (regions[1], FID_MassFracs);

   // Accessors for properties
   const AccessorWO<double, 3> acc_rho              (regions[1], FID_rho);
   const AccessorWO<double, 3> acc_mu               (regions[1], FID_mu);
   const AccessorWO<double, 3> acc_lam              (regions[1], FID_lam);
   const AccessorWO<VecNSp, 3> acc_Di               (regions[1], FID_Di);
   const AccessorWO<double, 3> acc_SoS              (regions[1], FID_SoS);
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   const AccessorWO<VecNIo, 3> acc_Ki               (regions[1], FID_Ki);
#endif

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_ModCells) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_ModCells) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_ModCells) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdatePropertiesFromPrimitive_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_velocity, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                        acc_Ki,
#endif
                        r_ModCells, getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
}

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateConservedFromPrimitiveTask
//-----------------------------------------------------------------------------

__global__
void UpdateConservedFromPrimitive_kernel(const AccessorRO<VecNSp, 3> MassFracs,
                                         const AccessorRO<double, 3> temperature,
                                         const AccessorRO<  Vec3, 3> velocity,
                                         const AccessorRO<double, 3> rho,
                                         const AccessorWO<VecNEq, 3> Conserved,
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
      // Mixture check
      assert(mix.CheckMixture(MassFracs[p]));
      UpdateConservedFromPrimitiveTask::UpdateConserved(
                     MassFracs, temperature, velocity,
                     rho, Conserved,
                     p, mix);
   }
}

__host__
void UpdateConservedFromPrimitiveTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for primitive variables
   const AccessorRO<VecNSp, 3> acc_MassFracs        (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho              (regions[0], FID_rho);

   // Accessors for conserved variables
   const AccessorWO<VecNEq, 3> acc_Conserved        (regions[1], FID_Conserved);

   // Extract execution domains
   Domain r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());

   // Launch the kernel (launch domain might be composed by multiple rectangles)
   for (RectInDomainIterator<3> Rit(r_ModCells); Rit(); Rit++) {
      const int threads_per_block = 256;
      const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, (*Rit));
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
      UpdateConservedFromPrimitive_kernel<<<num_blocks_3d, TPB_3d>>>(
                           acc_MassFracs, acc_temperature, acc_velocity,
                           acc_rho, acc_Conserved, (*Rit),
                           getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit));
   }
}

//-----------------------------------------------------------------------------
// KERNELS FOR UpdatePrimitiveFromConservedTask
//-----------------------------------------------------------------------------

__global__
void UpdatePrimitiveFromConserved_kernel(const AccessorRO<VecNEq, 3> Conserved,
                                         const AccessorRW<double, 3> temperature,
                                         const AccessorWO<double, 3> pressure,
                                         const AccessorWO<VecNSp, 3> MolarFracs,
                                         const AccessorWO<  Vec3, 3> velocity,
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
      UpdatePrimitiveFromConservedTask::UpdatePrimitive(
                     Conserved, temperature, pressure,
                     MolarFracs, velocity,
                     p, mix);
   }
}

__host__
void UpdatePrimitiveFromConservedTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for conserved variables
   const AccessorRO<VecNEq, 3> acc_Conserved        (regions[0], FID_Conserved);

   // Accessors for temperature variables
   const AccessorRW<double, 3> acc_temperature      (regions[1], FID_temperature);

   // Accessors for primitive variables
   const AccessorWO<double, 3> acc_pressure         (regions[1], FID_pressure);
   const AccessorWO<VecNSp, 3> acc_MolarFracs       (regions[1], FID_MolarFracs);
   const AccessorWO<  Vec3, 3> acc_velocity         (regions[1], FID_velocity);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_ModCells) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_ModCells) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_ModCells) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdatePrimitiveFromConserved_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_Conserved, acc_temperature, acc_pressure,
                        acc_MolarFracs, acc_velocity, r_ModCells,
                        getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells));
}

//-----------------------------------------------------------------------------
// KERNELS FOR GetVelocityGradientsTask
//-----------------------------------------------------------------------------

__global__
void GetVelocityGradients_kernel(const AccessorWO<  Vec3, 3>   vGradX,
                                 const AccessorWO<  Vec3, 3>   vGradY,
                                 const AccessorWO<  Vec3, 3>   vGradZ,
                                 const AccessorRO<  Vec3, 3> velocity,
                                 const AccessorRO<   int, 3>  nType_x,
                                 const AccessorRO<   int, 3>  nType_y,
                                 const AccessorRO<   int, 3>  nType_z,
                                 const AccessorRO<double, 3>   dcsi_d,
                                 const AccessorRO<double, 3>   deta_d,
                                 const AccessorRO<double, 3>   dzet_d,
                                 const Rect<3> my_bounds,
                                 const Rect<3> Fluid_bounds,
                                 const coord_t  size_x,
                                 const coord_t  size_y,
                                 const coord_t  size_z,
                                 const coord_t dsize_x,
                                 const coord_t dsize_y,
                                 const coord_t dsize_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      // X gradient
      GetVelocityGradientsTask::computeDerivatives<Xdir>(vGradX, velocity, p, nType_x[p], dcsi_d[p], dsize_x, Fluid_bounds);
      // Y gradient
      GetVelocityGradientsTask::computeDerivatives<Ydir>(vGradY, velocity, p, nType_y[p], deta_d[p], dsize_y, Fluid_bounds);
      // Z gradient
      GetVelocityGradientsTask::computeDerivatives<Zdir>(vGradZ, velocity, p, nType_z[p], dzet_d[p], dsize_z, Fluid_bounds);
   }
}

__host__
void GetVelocityGradientsTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x          (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d           (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d           (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d           (regions[1], FID_dzet_d);

   // Accessors for gradients
   const AccessorWO<  Vec3, 3> acc_vGradX           (regions[2], FID_velocityGradientX);
   const AccessorWO<  Vec3, 3> acc_vGradY           (regions[2], FID_velocityGradientY);
   const AccessorWO<  Vec3, 3> acc_vGradZ           (regions[2], FID_velocityGradientZ);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Get domain sizes
   const coord_t xsize = getSize<Xdir>(Fluid_bounds);
   const coord_t ysize = getSize<Ydir>(Fluid_bounds);
   const coord_t zsize = getSize<Zdir>(Fluid_bounds);

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   GetVelocityGradients_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_vGradX, acc_vGradY, acc_vGradZ, acc_velocity,
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        r_MyFluid, Fluid_bounds,
                        getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid),
                        xsize, ysize, zsize);
}

#if 0
//-----------------------------------------------------------------------------
// KERNELS FOR GetTemperatureGradientTask
//-----------------------------------------------------------------------------

__global__
void GetTemperatureGradient_kernel(const AccessorWO<  Vec3, 3>       tGrad,
                                   const AccessorRO<double, 3> temperature,
                                   const AccessorRO<   int, 3>     nType_x,
                                   const AccessorRO<   int, 3>     nType_y,
                                   const AccessorRO<   int, 3>     nType_z,
                                   const AccessorRO<double, 3>      dcsi_d,
                                   const AccessorRO<double, 3>      deta_d,
                                   const AccessorRO<double, 3>      dzet_d,
                                   const Rect<3> my_bounds,
                                   const Rect<3> Fluid_bounds,
                                   const coord_t  size_x,
                                   const coord_t  size_y,
                                   const coord_t  size_z,
                                   const coord_t dsize_x,
                                   const coord_t dsize_y,
                                   const coord_t dsize_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      tGrad[p][0] = GetTemperatureGradientTask::computeDerivative<Xdir>(temperature, p, nType_x[p], dcsi_d[p], dsize_x, Fluid_bounds);
      tGrad[p][1] = GetTemperatureGradientTask::computeDerivative<Ydir>(temperature, p, nType_y[p], deta_d[p], dsize_y, Fluid_bounds);
      tGrad[p][2] = GetTemperatureGradientTask::computeDerivative<Zdir>(temperature, p, nType_z[p], dzet_d[p], dsize_z, Fluid_bounds);
   }
}

__host__
void GetTemperatureGradientTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x          (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d           (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d           (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d           (regions[1], FID_dzet_d);

   // Accessors for gradients
   const AccessorWO<  Vec3, 3> acc_tGrad            (regions[2], FID_temperatureGradient);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Get domain sizes
   const coord_t xsize = getSize<Xdir>(Fluid_bounds);
   const coord_t ysize = getSize<Ydir>(Fluid_bounds);
   const coord_t zsize = getSize<Zdir>(Fluid_bounds);

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   GetTemperatureGradient_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_tGrad, acc_temperature,
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        r_MyFluid, Fluid_bounds,
                        getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid),
                        xsize, ysize, zsize);
}
#endif
