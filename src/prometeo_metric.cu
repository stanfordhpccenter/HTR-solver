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
void ComputeMetrics_kernel(const AccessorWO<double, 3>               m_e,
                           const AccessorWO<double, 3>               m_d,
                           const AccessorWO<double, 3>               m_s,
                           const AccessorRO<  Vec3, 3> centerCoordinates,
                           const AccessorRO<   int, 3>             nType,
                           const Rect<3> my_bounds,
                           const Rect<3> Fluid_bounds,
                           const double  width,
                           const coord_t size_x,
                           const coord_t size_y,
                           const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if (isThreadInCrossPlane<dir>(size_x, size_y, size_z)) {
      const coord_t size = getSize<dir>(Fluid_bounds);
      const coord_t span_size = getSize<dir>(my_bounds);
      const coord_t firstIndex = firstIndexInSpan<dir>(span_size);
      if (firstIndex < span_size) {
         const coord_t lastIndex =  lastIndexInSpan<dir>(span_size);
         double coordM_e; double coordP_e;
         // Reconstruct the coordinate at -1/2 of the first point
         {
            const Point<3> p = GetPointInSpan<dir>(my_bounds, firstIndex, x, y, z);
            const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(nType[p]));
            coordM_e = InitializeMetricTask::reconstructCoordEuler<dir>(centerCoordinates, pm1, width,
                                                                        nType[pm1], size, Fluid_bounds);
            coordM_e = unwarpCoordinate<dir>(coordM_e, width, -1, p, Fluid_bounds);
         }
         // Loop across my section of the span
         for (coord_t i = firstIndex; i < lastIndex; i++) {
            const Point<3> p = GetPointInSpan<dir>(my_bounds, i, x, y, z);
            coordP_e = InitializeMetricTask::reconstructCoordEuler<dir>(centerCoordinates, p, width,
                                                                        nType[p], size, Fluid_bounds);
            // Compute the metrics
            m_e[p] = 1.0/(coordP_e - coordM_e);
            InitializeMetricTask::ComputeDiffusionMetrics<dir>(m_d, m_s, centerCoordinates, p,
                                                            width, nType[p], size, Fluid_bounds);

            // Store plus values for next point
            coordM_e = coordP_e;
         }
      }
   }
}

__host__
void InitializeMetricTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<  Vec3, 3> acc_centerCoordinates(regions[0], FID_centerCoordinates);
   const AccessorRO<   int, 3> acc_nType_x          (regions[0], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[0], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[0], FID_nType_z);

   // Accessors for metrics
   const AccessorWO<double, 3> acc_dcsi_e(regions[1], FID_dcsi_e);
   const AccessorWO<double, 3> acc_deta_e(regions[1], FID_deta_e);
   const AccessorWO<double, 3> acc_dzet_e(regions[1], FID_dzet_e);

   const AccessorWO<double, 3> acc_dcsi_d(regions[1], FID_dcsi_d);
   const AccessorWO<double, 3> acc_deta_d(regions[1], FID_deta_d);
   const AccessorWO<double, 3> acc_dzet_d(regions[1], FID_dzet_d);

   const AccessorWO<double, 3> acc_dcsi_s(regions[1], FID_dcsi_s);
   const AccessorWO<double, 3> acc_deta_s(regions[1], FID_deta_s);
   const AccessorWO<double, 3> acc_dzet_s(regions[1], FID_dzet_s);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Determine the grid width from bounding box
   const double xWidth = args.bBox.v1[0] - args.bBox.v0[0];
   const double yWidth = args.bBox.v3[1] - args.bBox.v0[1];
   const double zWidth = args.bBox.v4[2] - args.bBox.v0[2];

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
      // Launch the kernel that computes the metric
      const dim3 TPB_3d = splitThreadsPerBlockSpan<Xdir>(threads_per_block, r_MyFluid);
      const dim3 num_blocks_3d = numBlocksSpan<Xdir>(TPB_3d, r_MyFluid);
      ComputeMetrics_kernel<Xdir><<<num_blocks_3d, TPB_3d, 0, Xstream>>>(
                           acc_dcsi_e, acc_dcsi_d, acc_dcsi_s,
                           acc_centerCoordinates, acc_nType_x,
                           r_MyFluid, Fluid_bounds, xWidth,
                           getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
   }

   // Y direction
   {
      // Launch the kernel that computes the metric
      const dim3 TPB_3d = splitThreadsPerBlockSpan<Ydir>(threads_per_block, r_MyFluid);
      const dim3 num_blocks_3d = numBlocksSpan<Ydir>(TPB_3d, r_MyFluid);
      ComputeMetrics_kernel<Ydir><<<num_blocks_3d, TPB_3d, 0, Ystream>>>(
                           acc_deta_e, acc_deta_d, acc_deta_s,
                           acc_centerCoordinates, acc_nType_y,
                           r_MyFluid, Fluid_bounds, yWidth,
                           getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
   }

   // Z direction
   {
      // Launch the kernel that computes the metric
      const dim3 TPB_3d = splitThreadsPerBlockSpan<Zdir>(threads_per_block, r_MyFluid);
      const dim3 num_blocks_3d = numBlocksSpan<Zdir>(TPB_3d, r_MyFluid);
      ComputeMetrics_kernel<Zdir><<<num_blocks_3d, TPB_3d, 0, Zstream>>>(
                           acc_dzet_e, acc_dzet_d, acc_dzet_s,
                           acc_centerCoordinates, acc_nType_z,
                           r_MyFluid, Fluid_bounds, zWidth,
                           getSize<Xdir>(r_MyFluid), getSize<Ydir>(r_MyFluid), getSize<Zdir>(r_MyFluid));
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
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

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

