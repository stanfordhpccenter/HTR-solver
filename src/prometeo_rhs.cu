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

// Load thrust
#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

// Define a constant memory that will hold the Mixture struct
__device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// COMMON KERNELS
//-----------------------------------------------------------------------------
template<direction dir>
__global__
void UpdateRHSUsingFlux_kernel(const DeferredBuffer<VecNEq, 3>    Flux,
                               const AccessorRO<double, 3>           m,
                               const AccessorRW<VecNEq, 3> Conserved_t,
                               const Rect<3> Divg_bounds,
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
      const Point<3> p = Point<3>(x + Divg_bounds.lo.x,
                                  y + Divg_bounds.lo.y,
                                  z + Divg_bounds.lo.z);
      const Point<3> pm1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, -1);
      #pragma unroll
      for (int i=0; i<nEq; i++)
         Conserved_t[p][i] += m[p]*(Flux[p][i] - Flux[pm1][i]);
   }
}

//-----------------------------------------------------------------------------
// KERNELS FOR UpdateUsingHybridEulerFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
struct is_KGPoint {
   is_KGPoint(const AccessorRO<  bool, 3> shockSensor_,
              const AccessorRO<   int, 3>       nType_,
              const Rect<3> bounds_,
              const coord_t size_) :
               shockSensor(shockSensor_),
               nType(nType_),
               bounds(bounds_),
               size(size_) {};

   __device__
   bool operator()(const Point<3> p) {
      const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, size, offM1(nType[p]));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, size, offP1(nType[p]));
      return (shockSensor[pM1] &&
              shockSensor[p  ] &&
              shockSensor[pP1]);
   }
private:
   const AccessorRO<  bool, 3> shockSensor;
   const AccessorRO<   int, 3>       nType;
   const Rect<3> bounds;
   const coord_t size;
};

__global__
void storePoints_kernel(const DeferredBuffer<Point<3>, 1>  Points,
                        const Rect<3> bounds,
                        const coord_t size_x,
                        const coord_t size_y,
                        const coord_t size_z,
                        const coord_t first)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   if ((x < size_x) && (y < size_y) && (z < size_z))
      Points[Point<1>(first + x + y*size_x + z*size_x*size_y)] =
                           Point<3>(x + bounds.lo.x,
                                    y + bounds.lo.y,
                                    z + bounds.lo.z);
}

template<direction dir>
__global__
void KGFluxReconstruction_kernel(const DeferredBuffer<VecNEq, 3>      Flux,
                                 const DeferredBuffer<Point<3>, 1>  Points,
                                 const AccessorRO<VecNEq, 3>   Conserved,
                                 const AccessorRO<double, 3>         rho,
                                 const AccessorRO<VecNSp, 3>   MassFracs,
                                 const AccessorRO<  Vec3, 3>    velocity,
                                 const AccessorRO<double, 3>    pressure,
                                 const AccessorRO<   int, 3>       nType,
                                 const Rect<3> Fluid_bounds,
                                 const coord_t offset,
                                 const coord_t size)
{
   const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < offset) {
      const Point<3> p = Points[tid];
      UpdateUsingEulerFluxUtils<dir>::KGFluxReconstruction(
                           Flux[p].v,
                           Conserved, rho, MassFracs,
                           velocity,  pressure,
                           p, nType[p], size, Fluid_bounds);
   }
}

template<direction dir>
__global__
void TENOFluxReconstruction_kernel(const DeferredBuffer<VecNEq, 3>      Flux,
                                   const DeferredBuffer<Point<3>, 1>  Points,
                                   const AccessorRO<VecNEq, 3>   Conserved,
                                   const AccessorRO<double, 3>         SoS,
                                   const AccessorRO<double, 3>         rho,
                                   const AccessorRO<  Vec3, 3>    velocity,
                                   const AccessorRO<double, 3>    pressure,
                                   const AccessorRO<VecNSp, 3>   MassFracs,
                                   const AccessorRO<double, 3> temperature,
                                   const AccessorRO<   int, 3>       nType,
                                   const Rect<3> Fluid_bounds,
                                   const coord_t offset,
                                   const coord_t volume,
                                   const coord_t size)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x + offset;
   if (tid < volume) {
      const Point<3> p = Points[tid];
      UpdateUsingEulerFluxUtils<dir>::TENOFluxReconstruction(Flux[p].v,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      p, nType[p], mix, size, Fluid_bounds);
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

   // Store the Flux domain and the linear size of Fluid
   Domain FluxDomain = runtime->get_index_space_domain(ctx, args.FluxGhost.get_index_space());
   const coord_t size = getSize<dir>(Fluid_bounds);

   const int threads_per_block = 256;

   // Extract lists of points where we are going to deploy the KG or TENO reconstruction
   // Points for KG reconstruction are in PointsList[0, offset)
   // Points for TENO reconstruction are in PointsList[offset, FluxDomainVolume)
   cudaStream_t default_stream;
   cudaStreamCreate(&default_stream);
   const coord_t FluxDomainVolume = FluxDomain.get_volume();
   DeferredBuffer<Point<3>, 1> PointsList(Rect<1>(0, FluxDomainVolume), Memory::GPU_FB_MEM);
   coord_t first = 0;
   for (RectInDomainIterator<3> Rit(FluxDomain); Rit(); Rit++) {
      const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, *Rit);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
      storePoints_kernel<<<num_blocks_3d, TPB_3d, 0, default_stream>>>(PointsList, (*Rit),
                                 getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit),
                                 first);
      first += getSize<Xdir>(*Rit)*getSize<Ydir>(*Rit)*getSize<Zdir>(*Rit);
   }
   assert(first == FluxDomainVolume);
   // Partition the PointsList vector based on the shock sensor
   thrust::device_ptr<Point<3>> PointsList_vec(PointsList.ptr(Point<1>(0)));
   const coord_t offset = thrust::stable_partition(thrust::cuda::par.on(default_stream),
                                                   PointsList_vec, PointsList_vec + FluxDomainVolume,
                                                   is_KGPoint<dir>(acc_shockSensor, acc_nType, Fluid_bounds, size))
                           - PointsList_vec;
   // Record an event on the KG stream
   cudaEvent_t ListDone;
   cudaEventCreate(&ListDone);
   cudaEventRecord(ListDone, default_stream);

   // Store all diffusion fluxes in a deferred buffer
   DeferredBuffer<VecNEq, 3> Flux(Memory::GPU_FB_MEM, FluxDomain);

   // Launch the kernel to reconstruct the fluxes using KG (after the node list is built)
   cudaStream_t KGstream;
   cudaStreamCreateWithFlags(&KGstream, cudaStreamNonBlocking);
   cudaStreamWaitEvent(KGstream, ListDone, 0);
   if (offset > 0) {
      const int num_blocks = (offset + (threads_per_block-1)) / threads_per_block;
      KGFluxReconstruction_kernel<dir><<<num_blocks, threads_per_block, 0, KGstream>>>(
                                    Flux, PointsList,
                                    acc_Conserved, acc_rho, acc_MassFracs,
                                    acc_velocity, acc_pressure, acc_nType,
                                    Fluid_bounds, offset, size);
   }
   // Record an event on the KG stream
   cudaEvent_t KGend;
   cudaEventCreate(&KGend);
   cudaEventRecord(KGend, KGstream);

   // Launch the kernel to reconstruct the fluxes using TENO (after the node list is built)
   cudaStream_t TENOstream;
   cudaStreamCreateWithFlags(&TENOstream, cudaStreamNonBlocking);
   cudaStreamWaitEvent(TENOstream, ListDone, 0);
   if (FluxDomainVolume - offset > 0) {
      const int num_blocks = ((FluxDomainVolume - offset) + (threads_per_block-1)) / threads_per_block;
      TENOFluxReconstruction_kernel<dir><<<num_blocks, threads_per_block, 0, TENOstream>>>(
                                    Flux, PointsList,
                                    acc_Conserved, acc_SoS, acc_rho,
                                    acc_velocity, acc_pressure, acc_MassFracs, acc_temperature, acc_nType,
                                    Fluid_bounds, offset, FluxDomainVolume, size);
   }
   // Record an event on the TENO stream
   cudaEvent_t TENOend;
   cudaEventCreate(&TENOend);
   cudaEventRecord(TENOend, TENOstream);

   // Ensure that reconstruction kernels are done and cleanup the reconstruction streams
   cudaStreamWaitEvent(default_stream,   KGend, 0);
   cudaStreamWaitEvent(default_stream, TENOend, 0);
   cudaEventDestroy(ListDone);
   cudaEventDestroy(   KGend);
   cudaEventDestroy( TENOend);
   cudaStreamDestroy(  KGstream);
   cudaStreamDestroy(TENOstream);

   // Launch the kernel to update the RHS using the fluxes on each point of the FluxDomain
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_ModCells) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_ModCells) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_ModCells) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateRHSUsingFlux_kernel<dir><<<num_blocks_3d, TPB_3d, 0, default_stream>>>(
                              Flux, acc_m_e, acc_Conserved_t,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells), size);

   // Cleanup default stream
   cudaStreamDestroy(default_stream);
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
void TENOAFluxReconstruction_kernel(const DeferredBuffer<VecNEq, 3>      Flux,
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
                                    const coord_t size_z,
                                    const coord_t size)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + Flux_bounds.lo.x,
                                  y + Flux_bounds.lo.y,
                                  z + Flux_bounds.lo.z);
      UpdateUsingEulerFluxUtils<dir>::TENOAFluxReconstruction(Flux[p].v,
                                      Conserved, SoS, rho, velocity,
                                      pressure, MassFracs, temperature,
                                      p, nType[p], mix, size, Fluid_bounds);
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

   // Store the Flux domain and the linear size of Fluid
   Domain FluxDomain = runtime->get_index_space_domain(ctx, args.FluxGhost.get_index_space());
   const coord_t size = getSize<dir>(Fluid_bounds);

   // Store all Euler fluxes in a deferred buffer
   DeferredBuffer<VecNEq, 3> Flux(Memory::GPU_FB_MEM, FluxDomain);

   const int threads_per_block = 256;

   // Launch the kernel to compute the fluxes on each point of the FluxDomain
   for (RectInDomainIterator<3> Rit(FluxDomain); Rit(); Rit++) {
      const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, *Rit);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
      TENOAFluxReconstruction_kernel<dir><<<num_blocks_3d, TPB_3d>>>(Flux,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature, acc_nType,
                                    (*Rit), Fluid_bounds,
                                    getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit), size);
   }

   // Launch the kernel to update the RHS using the fluxes on each point of the FluxDomain
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_ModCells) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_ModCells) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_ModCells) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateRHSUsingFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              Flux, acc_m_e, acc_Conserved_t,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells), size);
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
// KERNELS FOR UpdateUsingDiffusionFluxTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void ComputeDiffusionFlux_kernel(const DeferredBuffer<VecNEq, 3>    Flux,
                                 const AccessorRO<   int, 3>       nType,
                                 const AccessorRO<double, 3>      metric,
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
                                 const coord_t size_z,
                                 const coord_t size)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + Flux_bounds.lo.x,
                                  y + Flux_bounds.lo.y,
                                  z + Flux_bounds.lo.z);
      UpdateUsingDiffusionFluxTask<dir>::GetDiffusionFlux(
                       Flux[p].v, nType[p], metric[p], mix,
                       rho, mu, lam, Di,
                       temperature, velocity, Xi,
                       rhoYi, vGrad1, vGrad2,
                       p, size, Fluid_bounds);
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

   // Store the Flux domain and the linear size of Fluid
   Domain FluxDomain = runtime->get_index_space_domain(ctx, args.FluxGhost.get_index_space());
   const coord_t size = getSize<dir>(Fluid_bounds);

   // Store all diffusion fluxes in a deferred buffer
   DeferredBuffer<VecNEq, 3> Flux(Memory::GPU_FB_MEM, FluxDomain);

   // Launch the kernel to compute the fluxes on each point of the FluxDomain
   const int threads_per_block = 256;
   for (RectInDomainIterator<3> Rit(FluxDomain); Rit(); Rit++) {
      const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, *Rit);
      const dim3 num_blocks_3d = dim3((getSize<Xdir>(*Rit) + (TPB_3d.x - 1)) / TPB_3d.x,
                                      (getSize<Ydir>(*Rit) + (TPB_3d.y - 1)) / TPB_3d.y,
                                      (getSize<Zdir>(*Rit) + (TPB_3d.z - 1)) / TPB_3d.z);
      ComputeDiffusionFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              Flux, acc_nType, acc_m_s,
                              acc_rho, acc_mu, acc_lam, acc_Di,
                              acc_temperature, acc_velocity, acc_MolarFracs,
                              acc_Conserved, acc_vGrad1, acc_vGrad2,
                              (*Rit), Fluid_bounds,
                              getSize<Xdir>(*Rit), getSize<Ydir>(*Rit), getSize<Zdir>(*Rit), size);
   }

   // Launch the kernel to update the RHS using the fluxes on each point of the FluxDomain
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_ModCells);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_ModCells) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_ModCells) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_ModCells) + (TPB_3d.z - 1)) / TPB_3d.z);
   UpdateRHSUsingFlux_kernel<dir><<<num_blocks_3d, TPB_3d>>>(
                              Flux, acc_m_d, acc_Conserved_t,
                              r_ModCells, Fluid_bounds,
                              getSize<Xdir>(r_ModCells), getSize<Ydir>(r_ModCells), getSize<Zdir>(r_ModCells), size);
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
                    Conserved_t[p].v,
                    MassFracs, pressure,
                    SoS[p], rho[p], temperature[p],
                    velocity[p].v, vGrad[p].v, dudt[p].v, dTdt[p],
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

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
      UpdateUsingFluxNSCBCInflowMinusSideTask<dir>::addLODIfluxes(
                    Conserved_t[p].v,
                    MassFracs, pressure,
                    SoS[p], rho[p], temperature[p],
                    velocity[p].v, vGrad[p].v, dudt[p].v, dTdt[p],
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

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
                    Conserved_t[p].v,
                    MassFracs, rho, mu, pressure,
                    velocity, vGradN, vGradT1, vGradT2,
                    SoS[p], temperature[p], Conserved[p].v,
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

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
                    Conserved_t[p].v,
                    MassFracs, rho, mu, pressure,
                    velocity, vGradN, vGradT1, vGradT2,
                    SoS[p], temperature[p], Conserved[p].v,
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

   // Copy the mixture to the device
   cudaMemcpyToSymbolAsync(mix, &(args.mix), sizeof(Mix));

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

