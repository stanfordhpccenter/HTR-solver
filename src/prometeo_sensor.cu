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

#include "prometeo_sensor.hpp"
#include "prometeo_sensor.inl"
#include "cuda_utils.hpp"

//-----------------------------------------------------------------------------
// KERNEL FOR UpdateDucrosSensorTask
//-----------------------------------------------------------------------------

__global__
void ComputeDucrosSensor_kernel(const AccessorWO<double, 3> DucrosSensor,
                                const AccessorRO<  Vec3, 3> velocity,
                                const AccessorRO<   int, 3> nType_csi,
                                const AccessorRO<   int, 3> nType_eta,
                                const AccessorRO<   int, 3> nType_zet,
                                const AccessorRO<double, 3> dcsi_d,
                                const AccessorRO<double, 3> deta_d,
                                const AccessorRO<double, 3> dzet_d,
                                const Rect<3> my_bounds,
                                const Rect<3> Fluid_bounds,
                                const double eps,
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
      DucrosSensor[p] = UpdateDucrosSensorTask::DucrosSensor(velocity,
                                     nType_csi, nType_eta, nType_zet,
                                     dcsi_d, deta_d, dzet_d,
                                     p, Fluid_bounds, eps);

   }
}

__host__
void UpdateDucrosSensorTask::gpu_base_impl(
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

   // Accessors for shock sensor
   const AccessorWO<double, 3> acc_DucrosSensor     (regions[2], FID_DucrosSensor);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Compute vorticity scale
   const double eps = max(args.vorticityScale*args.vorticityScale, 1e-6);

   // Launch the kernel to update the Ducros sensor
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   ComputeDucrosSensor_kernel<<<num_blocks_3d, TPB_3d>>>(
                              acc_DucrosSensor, acc_velocity,
                              acc_nType_x, acc_nType_y, acc_nType_z,
                              acc_dcsi_d, acc_deta_d, acc_dzet_d,
                              r_MyFluid, Fluid_bounds, eps,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

//-----------------------------------------------------------------------------
// KERNEL FOR UpdateShockSensorTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void UpdateShockSensor_kernel(const AccessorRO<double, 3> DucrosSensor,
                              const AccessorRO<VecNEq, 3> Conserved,
                              const AccessorRO<   int, 3> nType,
                              const AccessorWO<  bool, 3> shockSensor,
                              const Rect<3> my_bounds,
                              const Rect<3> Fluid_bounds,
                              const coord_t size_x,
                              const coord_t size_y,
                              const coord_t size_z,
                              const coord_t size)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      const Point<3> pM2 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM2(nType[p]));
      const Point<3> pM1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP1(nType[p]));
      const Point<3> pP2 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP2(nType[p]));
      const Point<3> pP3 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP3(nType[p]));

      const double Phi = max(max(max(max(max(
                           DucrosSensor[pM2],
                           DucrosSensor[pM1]),
                           DucrosSensor[p  ]),
                           DucrosSensor[pP1]),
                           DucrosSensor[pP2]),
                           DucrosSensor[pP3]);

      bool sensor = true;
      #pragma unroll
      for (int i=0; i<nSpec; i++)
         sensor = sensor && TENOsensor::TENOA(Conserved[pM2][i], Conserved[pM1][i], Conserved[p  ][i],
                                              Conserved[pP1][i], Conserved[pP2][i], Conserved[pP3][i],
                                              nType[p], Phi);
      shockSensor[p] = sensor;
   }
}

template<direction dir>
__host__
void UpdateShockSensorTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<VecNEq, 3> acc_Conserved        (regions[0], FID_Conserved);
   const AccessorRO<double, 3> acc_DucrosSensor     (regions[0], FID_DucrosSensor);

   // Accessors for node type
   const AccessorRO<   int, 3> acc_nType            (regions[1], FID_nType);

   // Accessors for shock sensor
   const AccessorWO<  bool, 3> acc_shockSensor      (regions[2], FID_shockSensor);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;
   const coord_t size = getSize<dir>(Fluid_bounds);

   // Launch the kernel to update the shock sensor
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateShockSensor_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_DucrosSensor, acc_Conserved,
                              acc_nType, acc_shockSensor,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid), size);
}

template void UpdateShockSensorTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateShockSensorTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void UpdateShockSensorTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

