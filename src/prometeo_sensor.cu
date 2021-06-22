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
// KERNEL FOR UpdateShockSensorTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void ComputeDucrosSensor_kernel(const DeferredBuffer<double, 3> DucrosS,
                                const AccessorRO<  Vec3, 3>      vGradX,
                                const AccessorRO<  Vec3, 3>      vGradY,
                                const AccessorRO<  Vec3, 3>      vGradZ,
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
      DucrosS[p] = DucrosSensor(vGradX[p], vGradY[p], vGradZ[p], eps);
   }
}

template<direction dir>
__global__
void UpdateShockSensor_kernel(const DeferredBuffer<double, 3> DucrosS,
                              const AccessorRO<VecNEq, 3>   Conserved,
                              const AccessorRO<   int, 3>       nType,
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
                           DucrosS[pM2],
                           DucrosS[pM1]),
                           DucrosS[p  ]),
                           DucrosS[pP1]),
                           DucrosS[pP2]),
                           DucrosS[pP3]);

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
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<VecNEq, 3> acc_Conserved        (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_vGradX           (regions[0], FID_velocityGradientX);
   const AccessorRO<  Vec3, 3> acc_vGradY           (regions[0], FID_velocityGradientY);
   const AccessorRO<  Vec3, 3> acc_vGradZ           (regions[0], FID_velocityGradientZ);

   // Accessors for node type
   const AccessorRO<   int, 3> acc_nType            (regions[1], FID_nType);

   // Accessors for shock sensor
   const AccessorWO<  bool, 3> acc_shockSensor      (regions[2], FID_shockSensor);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;
   const coord_t size = getSize<dir>(Fluid_bounds);

   // Compute vorticity scale
   const double eps = max(args.vorticityScale*args.vorticityScale, 1e-6);

   // Store Ducros sensor in a DeferredBuffer
   Domain GhostDomain = runtime->get_index_space_domain(ctx, args.Ghost.get_index_space());
   DeferredBuffer<double, 3> DucrosSensor(Memory::GPU_FB_MEM, GhostDomain);
   {
   const int threads_per_block = 256;
   for (RectInDomainIterator<3> Rit(GhostDomain); Rit(); Rit++) {
      const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, *Rit);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
      ComputeDucrosSensor_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              DucrosSensor, acc_vGradX, acc_vGradY, acc_vGradZ,
                              (*Rit), Fluid_bounds, eps,
                              getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit));
   }
   }

   // Launch the kernel to update the shock sensor
   {
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateShockSensor_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              DucrosSensor, acc_Conserved,
                              acc_nType, acc_shockSensor,
                              r_MyFluid, Fluid_bounds,
                              getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid), size);
   }
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

