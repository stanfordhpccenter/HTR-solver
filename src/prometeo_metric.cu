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

#include "prometeo_metric.hpp"
#include "cuda_utils.hpp"

//-----------------------------------------------------------------------------
// KERNELS FOR InitializeMetricTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void ReconstructCoordinates_kernel(const DeferredBuffer<double, 3> coord,
                                   const AccessorRO<  Vec3, 3> centerCoordinates,
                                   const AccessorRO<   int, 3>             nType,
                                   const Rect<3> my_bounds,
                                   const Rect<3> Fluid_bounds,
                                   const double  width,
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
      (*coord.ptr(p)) = InitializeMetricTask::reconstructCoordEuler<dir>(centerCoordinates, p, width,
                                                                         nType[p], size, Fluid_bounds); 
   }
}

template<direction dir>
__global__
void ComputeMetrics_kernel(const DeferredBuffer<double, 3> coord,
                           const AccessorRW<double, 3>               m_e,
                           const AccessorRW<double, 3>               m_d,
                           const AccessorRW<double, 3>               m_s,
                           const AccessorRO<  Vec3, 3> centerCoordinates,
                           const AccessorRO<   int, 3>             nType,
                           const Rect<3> my_bounds,
                           const Rect<3> Fluid_bounds,
                           const double  width,
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
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
      m_e[p] = 1.0/((*coord.ptr(p)) - unwarpCoordinate<dir>((*coord.ptr(pm1)), width, -1, p, Fluid_bounds));
      InitializeMetricTask::ComputeDiffusionMetrics<dir>(m_d, m_s, centerCoordinates, p,
                                                         width, nType[p], size, Fluid_bounds);
   }
}
__host__
void InitializeMetricTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<  Vec3, 3> acc_centerCoordinates(regions[0], FID_centerCoordinates);
   const AccessorRO<   int, 3> acc_nType_x          (regions[0], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[0], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[0], FID_nType_z);

   // Accessors for metrics
   const AccessorRW<double, 3> acc_dcsi_e(regions[4], FID_dcsi_e);
   const AccessorRW<double, 3> acc_deta_e(regions[4], FID_deta_e);
   const AccessorRW<double, 3> acc_dzet_e(regions[4], FID_dzet_e);

   const AccessorRW<double, 3> acc_dcsi_d(regions[4], FID_dcsi_d);
   const AccessorRW<double, 3> acc_deta_d(regions[4], FID_deta_d);
   const AccessorRW<double, 3> acc_dzet_d(regions[4], FID_dzet_d);

   const AccessorRW<double, 3> acc_dcsi_s(regions[4], FID_dcsi_s);
   const AccessorRW<double, 3> acc_deta_s(regions[4], FID_deta_s);
   const AccessorRW<double, 3> acc_dzet_s(regions[4], FID_dzet_s);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   const int threads_per_block = 256;

   // Create one stream for each direction
   cudaStream_t Xstream;
   cudaStream_t Ystream;
   cudaStream_t Zstream;
   cudaStreamCreateWithFlags(&Xstream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&Ystream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&Zstream, cudaStreamNonBlocking);

   // X direction
   {
      const coord_t size = getSize<Xdir>(Fluid_bounds);

      Domain Ghost = runtime->get_index_space_domain(ctx, args.XGhost.get_index_space());

      // Store the reconstructed coordinate in a deferred buffer
      DeferredBuffer<double, 3> X(Memory::GPU_FB_MEM, Ghost);

      // Launch the kernel to reconstruct the coordinates
      for (RectInDomainIterator<3> Rit(Ghost); Rit(); Rit++) {
         const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, *Rit);
         const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                         (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                         (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
         ReconstructCoordinates_kernel<Xdir><<<num_blocks_3d, TPB_3d, 0, Xstream>>>(
                                       X, acc_centerCoordinates, acc_nType_x, (*Rit),
                                       Fluid_bounds, args.Grid_xWidth,
                                       getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit), size);
      }

      // Launch the kernel that computes the metric
      const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_MyFluid);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
      ComputeMetrics_kernel<Xdir><<<num_blocks_3d, TPB_3d, 0, Xstream>>>(
                           X, acc_dcsi_e, acc_dcsi_d, acc_dcsi_s,
                           acc_centerCoordinates, acc_nType_x,
                           r_MyFluid, Fluid_bounds, args.Grid_xWidth,
                           getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid), size);
   }

   // Y direction
   {
      const coord_t size = getSize<Ydir>(Fluid_bounds);

      Domain Ghost = runtime->get_index_space_domain(ctx, args.YGhost.get_index_space());

      // Store the reconstructed coordinate in a deferred buffer
      DeferredBuffer<double, 3> Y(Memory::GPU_FB_MEM, Ghost);

      // Launch the kernel to reconstruct the coordinates
      for (RectInDomainIterator<3> Rit(Ghost); Rit(); Rit++) {
         const dim3 TPB_3d = splitThreadsPerBlock<Ydir>(threads_per_block, *Rit);
         const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                         (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                         (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
         ReconstructCoordinates_kernel<Ydir><<<num_blocks_3d, TPB_3d, 0, Ystream>>>(
                                       Y, acc_centerCoordinates, acc_nType_y, (*Rit),
                                       Fluid_bounds, args.Grid_yWidth,
                                       getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit), size);
      }

      // Launch the kernel that computes the metric
      const dim3 TPB_3d = splitThreadsPerBlock<Ydir>(threads_per_block, r_MyFluid);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
      ComputeMetrics_kernel<Ydir><<<num_blocks_3d, TPB_3d, 0, Ystream>>>(
                           Y, acc_deta_e, acc_deta_d, acc_deta_s,
                           acc_centerCoordinates, acc_nType_y,
                           r_MyFluid, Fluid_bounds, args.Grid_yWidth,
                           getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid), size);
   }

   // Z direction
   {
      const coord_t size = getSize<Zdir>(Fluid_bounds);

      Domain Ghost = runtime->get_index_space_domain(ctx, args.ZGhost.get_index_space());

      // Store the reconstructed coordinate in a deferred buffer
      DeferredBuffer<double, 3> Z(Memory::GPU_FB_MEM, Ghost);

      // Launch the kernel to reconstruct the coordinates
      for (RectInDomainIterator<3> Rit(Ghost); Rit(); Rit++) {
         const dim3 TPB_3d = splitThreadsPerBlock<Zdir>(threads_per_block, *Rit);
         const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                         (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                         (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
         ReconstructCoordinates_kernel<Zdir><<<num_blocks_3d, TPB_3d, 0, Zstream>>>(
                                       Z, acc_centerCoordinates, acc_nType_z, (*Rit),
                                       Fluid_bounds, args.Grid_zWidth,
                                       getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit), size);
      }

      // Launch the kernel that computes the metric
      const dim3 TPB_3d = splitThreadsPerBlock<Zdir>(threads_per_block, r_MyFluid);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
      ComputeMetrics_kernel<Zdir><<<num_blocks_3d, TPB_3d, 0, Zstream>>>(
                           Z, acc_dzet_e, acc_dzet_d, acc_dzet_s,
                           acc_centerCoordinates, acc_nType_z,
                           r_MyFluid, Fluid_bounds, args.Grid_zWidth,
                           getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid), size);
   }

   // Cleanup the streams
   cudaStreamDestroy(Xstream);
   cudaStreamDestroy(Ystream);
   cudaStreamDestroy(Zstream);
}

//-----------------------------------------------------------------------------
// KERNELS FOR CorrectGhostMetric
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void CorrectGhostMetric_kernel(const AccessorRO<  Vec3, 3> centerCoordinates,
                               const AccessorRO<   int, 3>             nType,
                               const AccessorRW<double, 3>            metric,
                               const Rect<3> Fluid_bounds,
                               const coord_t size_x,
                               const coord_t size_y,
                               const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + Fluid_bounds.lo.x,
                                  y + Fluid_bounds.lo.y,
                                  z + Fluid_bounds.lo.z);
      if      (nType[p] == L_S_node) CorrectGhostMetricTask<dir>::CorrectLeftStaggered(  metric, centerCoordinates, p);
      else if (nType[p] == L_C_node) CorrectGhostMetricTask<dir>::CorrectLeftCollocated( metric, centerCoordinates, p);
      else if (nType[p] == R_S_node) CorrectGhostMetricTask<dir>::CorrectRightStaggered( metric, centerCoordinates, p);
      else if (nType[p] == R_C_node) CorrectGhostMetricTask<dir>::CorrectRightCollocated(metric, centerCoordinates, p);
   }
}

template<direction dir>
__host__
void CorrectGhostMetricTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<  Vec3, 3> acc_centerCoordinates(regions[0], FID_centerCoordinates);
   const AccessorRO<   int, 3> acc_nType            (regions[0], FID_nType);

   // Accessors for metrics
   const AccessorRW<double, 3> acc_m                (regions[1], FID_m);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());

   // Launch the kernel to update the ghost metric
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_MyFluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_MyFluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_MyFluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_MyFluid) + (TPB_3d.z - 1)) / TPB_3d.z);
   CorrectGhostMetric_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              acc_centerCoordinates, acc_nType, acc_m,
                              r_MyFluid, getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
}

template void CorrectGhostMetricTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectGhostMetricTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectGhostMetricTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

