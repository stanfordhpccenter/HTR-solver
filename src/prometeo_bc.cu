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

#include "prometeo_bc.hpp"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNELS FOR AddRecycleAverageTask
//-----------------------------------------------------------------------------

__global__
void AddRecycleAverageTask_kernel(const AccessorRO<double, 3> dcsi_d,
                                  const AccessorRO<double, 3> deta_d,
                                  const AccessorRO<double, 3> dzet_d,
                                  const AccessorRO<VecNSp, 3> MolarFracs_profile,
                                  const AccessorRO<double, 3> temperature_profile,
                                  const AccessorRO<  Vec3, 3> velocity_profile,
                                  const AccessorSumRD<VecNSp, 1> avg_MolarFracs,
                                  const AccessorSumRD<  Vec3, 1> avg_velocity,
                                  const AccessorSumRD<double, 1> avg_temperature,
                                  const AccessorSumRD<double, 1> avg_rho,
                                  const double Pbc,
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
      AddRecycleAverageTask::collectAverages(dcsi_d, deta_d, dzet_d,
                     MolarFracs_profile, temperature_profile, velocity_profile,
                     avg_MolarFracs, avg_velocity, avg_temperature,
                     avg_rho, Pbc, p, mix);
   }
}

__host__
void AddRecycleAverageTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_dcsi_d              (regions[0], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d              (regions[0], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d              (regions[0], FID_dzet_d);

   // Accessors for profile variables
   const AccessorRO<VecNSp, 3> acc_MolarFracs_profile  (regions[0], FID_MolarFracs_profile);
   const AccessorRO<double, 3> acc_temperature_profile (regions[0], FID_temperature_profile);
   const AccessorRO<  Vec3, 3> acc_velocity_profile    (regions[0], FID_velocity_profile);

   // Accessors for averages
   const AccessorSumRD<double, 1> acc_avg_rho          (regions[1], RA_FID_rho,         LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 1> acc_avg_temperature  (regions[1], RA_FID_temperature, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 1> acc_avg_MolarFracs   (regions[2], RA_FID_MolarFracs,  REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, 1> acc_avg_velocity     (regions[3], RA_FID_velocity,    REGENT_REDOP_SUM_VEC3);

   // Extract execution domain
   const Rect<3> r_plane = runtime->get_index_space_domain(ctx,
                                    regions[0].get_logical_region().get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_plane);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_plane) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_plane) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_plane) + (TPB_3d.z - 1)) / TPB_3d.z);
   AddRecycleAverageTask_kernel<<<num_blocks_3d, TPB_3d>>>(acc_dcsi_d, acc_deta_d, acc_dzet_d,
                            acc_MolarFracs_profile, acc_temperature_profile, acc_velocity_profile,
                            acc_avg_MolarFracs, acc_avg_velocity, acc_avg_temperature, acc_avg_rho,
                            args.Pbc, r_plane,
                            getSize<Xdir>(r_plane), getSize<Ydir>(r_plane), getSize<Zdir>(r_plane));
}

//-----------------------------------------------------------------------------
// KERNELS FOR SetNSCBC_InflowBC
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void SetNSCBC_InflowBC_kernel(const AccessorRO<VecNEq, 3> Conserved,
                              const AccessorRO<double, 3> SoS,
                              const AccessorRO<VecNSp, 3> MolarFracs_profile,
                              const AccessorRO<double, 3> temperature_profile,
                              const AccessorRO<  Vec3, 3> velocity_profile,
                              const AccessorWO<double, 3> pressure,
                              const AccessorWO<double, 3> temperature,
                              const AccessorWO<VecNSp, 3> MolarFracs,
                              const AccessorWO<  Vec3, 3> velocity,
                              const double Pbc,
                              const Rect<3> my_bounds,
                              const coord_t  size_x,
                              const coord_t  size_y,
                              const coord_t  size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

   // Index of normal direction
   constexpr int iN = normalIndex(dir);

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      MolarFracs[p] = MolarFracs_profile[p];
      temperature[p] = temperature_profile[p];
      velocity[p] = velocity_profile[p];
      if (fabs(velocity_profile[p][iN]) >= SoS[p])
         // It is supersonic, everything is imposed by the BC
         pressure[p] = Pbc;
      else
         // Compute pressure from NSCBC conservation equations
         SetNSCBC_InflowBCTask<dir>::setInflowPressure(
                           Conserved, MolarFracs_profile, temperature_profile,
                           pressure, p, mix);
   }
}

template<direction dir>
__host__
void SetNSCBC_InflowBCTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessor for conserved variables
   const AccessorRO<VecNEq, 3> acc_Conserved           (regions[0], FID_Conserved);

   // Accessor for speed of sound
   const AccessorRO<double, 3> acc_SoS                 (regions[0], FID_SoS);

   // Accessors for profile variables
   const AccessorRO<VecNSp, 3> acc_MolarFracs_profile  (regions[0], FID_MolarFracs_profile);
   const AccessorRO<double, 3> acc_temperature_profile (regions[0], FID_temperature_profile);
   const AccessorRO<  Vec3, 3> acc_velocity_profile    (regions[0], FID_velocity_profile);

   // Accessors for primitive variables
   const AccessorWO<double, 3> acc_pressure            (regions[1], FID_pressure);
   const AccessorWO<double, 3> acc_temperature         (regions[1], FID_temperature);
   const AccessorWO<VecNSp, 3> acc_MolarFracs          (regions[1], FID_MolarFracs);
   const AccessorWO<  Vec3, 3> acc_velocity            (regions[1], FID_velocity);

   // Extract execution domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[1].get_logical_region().get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   SetNSCBC_InflowBC_kernel<dir><<<num_blocks_3d, TPB_3d>>>(acc_Conserved, acc_SoS,
                        acc_MolarFracs_profile, acc_temperature_profile, acc_velocity_profile,
                        acc_pressure, acc_temperature, acc_MolarFracs, acc_velocity,
                        args.Pbc, r_BC,
                        getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

template void SetNSCBC_InflowBCTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void SetNSCBC_InflowBCTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void SetNSCBC_InflowBCTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNELS FOR SetNSCBC_OutflowBC
//-----------------------------------------------------------------------------

__global__
void SetNSCBC_OutflowBC_kernel(const AccessorRO<VecNEq, 3> Conserved,
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
void SetNSCBC_OutflowBCTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for conserved variables
   const AccessorRO<VecNEq, 3> acc_Conserved        (regions[0], FID_Conserved);

   // Accessors for temperature
   const AccessorRW<double, 3> acc_temperature      (regions[1], FID_temperature);

   // Accessors for primitive variables
   const AccessorWO<double, 3> acc_pressure         (regions[1], FID_pressure);
   const AccessorWO<VecNSp, 3> acc_MolarFracs       (regions[1], FID_MolarFracs);
   const AccessorWO<  Vec3, 3> acc_velocity         (regions[1], FID_velocity);

   // Extract execution domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[1].get_logical_region().get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   SetNSCBC_OutflowBC_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_Conserved, acc_temperature, acc_pressure,
                        acc_MolarFracs, acc_velocity, r_BC,
                        getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

//-----------------------------------------------------------------------------
// KERNELS FOR SetIncomingShockBCTask
//-----------------------------------------------------------------------------

__global__
void SetIncomingShockBC_kernel(const AccessorRO<VecNEq, 3> Conserved,
                               const AccessorRW<double, 3> temperature,
                               const AccessorWO<double, 3> pressure,
                               const AccessorWO<VecNSp, 3> MolarFracs,
                               const AccessorWO<  Vec3, 3> velocity,
                               const Vec3 velocity0,
                               const double temperature0,
                               const double pressure0,
                               const Vec3 velocity1,
                               const double temperature1,
                               const double pressure1,
                               const VecNSp MolarFracs0,
                               const int iShock,
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
      if ((p.x < iShock - 1) or
          (p.x > iShock + 1)) {
         // Threat as an outflow
         UpdatePrimitiveFromConservedTask::UpdatePrimitive(
                         Conserved, temperature, pressure,
                         MolarFracs, velocity,
                         p, mix);
      // Inject the shock over four points
      } else if (p.x == iShock - 1) {
         MolarFracs[p]  = MolarFracs0;
         velocity[p]    = 0.75*velocity0    + 0.25*velocity1;
         temperature[p] = 0.75*temperature0 + 0.25*temperature1;
         pressure[p]    = 0.75*pressure0    + 0.25*pressure1;
      } else if (p.x == iShock) {
         MolarFracs[p]  = MolarFracs0;
         velocity[p]    = 0.25*velocity0    + 0.75*velocity1;
         temperature[p] = 0.25*temperature0 + 0.75*temperature1;
         pressure[p]    = 0.25*pressure0    + 0.75*pressure1;
      } else if (p.x == iShock + 1) {
         MolarFracs[p]  = MolarFracs0;
         velocity[p]    = velocity1;
         temperature[p] = temperature1;
         pressure[p]    = pressure1;
      }
   }
}

__host__
void SetIncomingShockBCTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessor for conserved variables
   const AccessorRO<VecNEq, 3> acc_Conserved           (regions[0], FID_Conserved);

   // Accessors for temperature
   const AccessorRW<double, 3> acc_temperature         (regions[1], FID_temperature);

   // Accessors for primitive variables
   const AccessorWO<double, 3> acc_pressure            (regions[1], FID_pressure);
   const AccessorWO<VecNSp, 3> acc_MolarFracs          (regions[1], FID_MolarFracs);
   const AccessorWO<  Vec3, 3> acc_velocity            (regions[1], FID_velocity);

   // Extract execution domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[1].get_logical_region().get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   SetIncomingShockBC_kernel<<<num_blocks_3d, TPB_3d>>>(
                        acc_Conserved,
                        acc_temperature, acc_pressure, acc_MolarFracs, acc_velocity,
                        Vec3(args.params.velocity0), args.params.temperature0, args.params.pressure0,
                        Vec3(args.params.velocity1), args.params.temperature1, args.params.pressure1,
                        VecNSp(args.params.MolarFracs), args.params.iShock,
                        r_BC, getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
}

//-----------------------------------------------------------------------------
// KERNELS FOR SetRecycleRescalingBCTask
//-----------------------------------------------------------------------------

// Workaroud for Legion issue #879
struct SetRecycleRescalingBC_kernelArgs {
   const AccessorRO<  Vec3, 3> centerCoordinates;
   const AccessorRO<VecNEq, 3> Conserved;
   const AccessorRO<double, 3> SoS;
   const AccessorWO<double, 3> temperature;
   const AccessorWO<double, 3> pressure;
   const AccessorWO<VecNSp, 3> MolarFracs;
   const AccessorWO<  Vec3, 3> velocity;
   const AccessorRO<double, 3> temperature_recycle;
   const AccessorRO<  Vec3, 3> velocity_recycle;
   const AccessorRO<VecNSp, 3> MolarFracs_recycle;
   const AccessorRO<double, 3> temperature_profile;
   const AccessorRO<  Vec3, 3> velocity_profile;
   const AccessorRO<VecNSp, 3> MolarFracs_profile;
   const AccessorRO<double, 1> avg_y;
   const AccessorRO< float, 1> FI_xloc;
   const AccessorRO< float, 1> FI_iloc;
};

__global__
#ifdef LEGION_BOUNDS_CHECKS
void SetRecycleRescalingBC_kernel(const DeferredBuffer<SetRecycleRescalingBC_kernelArgs, 1> buffer,
                                  const FastInterpData FIdata,
                                  const double Pbc,
                                  const double yInnFact,
                                  const double yOutFact,
                                  const double uInnFact,
                                  const double uOutFact,
                                  const double idelta99Inl,
                                  const Rect<3> my_bounds,
                                  const coord_t  size_x,
                                  const coord_t  size_y,
                                  const coord_t  size_z)

#else
void SetRecycleRescalingBC_kernel(const SetRecycleRescalingBC_kernelArgs a,
                                  const FastInterpData FIdata,
                                  const double Pbc,
                                  const double yInnFact,
                                  const double yOutFact,
                                  const double uInnFact,
                                  const double uOutFact,
                                  const double idelta99Inl,
                                  const Rect<3> my_bounds,
                                  const coord_t  size_x,
                                  const coord_t  size_y,
                                  const coord_t  size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   SetRecycleRescalingBC_kernelArgs a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);

      // Compute the rescaled primitive quantities
      double temperatureR; Vec3 velocityR; VecNSp MolarFracsR;
      SetRecycleRescalingBCTask::GetRescaled(
                  temperatureR, velocityR, MolarFracsR, a.centerCoordinates,
                  a.temperature_recycle, a.velocity_recycle, a.MolarFracs_recycle,
                  a.temperature_profile, a.velocity_profile, a.MolarFracs_profile,
                  a.avg_y, a.FI_xloc, a.FI_iloc, FIdata, p,
                  yInnFact, yOutFact, uInnFact, uOutFact, idelta99Inl);

      a.MolarFracs[p] = MolarFracsR;
      a.temperature[p] = temperatureR;
      a.velocity[p] = velocityR;
      if (fabs(velocityR[0]) >= a.SoS[p])
         // It is supersonic, everything is imposed by the BC
         a.pressure[p] = Pbc;
      else
         // Compute pressure from NSCBC conservation equations
         a.pressure[p] = SetRecycleRescalingBCTask::setPressure(a.Conserved, temperatureR, MolarFracsR, p, mix);
   }
}

__host__
void SetRecycleRescalingBCTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 1);

   // Accessor for speed of sound
   const AccessorRO<  Vec3, 3> acc_centerCoordinates   (regions[0], FID_centerCoordinates);

   // Accessor for conserved variables
   const AccessorRO<VecNEq, 3> acc_Conserved           (regions[0], FID_Conserved);

   // Accessor for speed of sound
   const AccessorRO<double, 3> acc_SoS                 (regions[0], FID_SoS);

   // Accessors for profile variables
   const AccessorRO<VecNSp, 3> acc_MolarFracs_profile  (regions[0], FID_MolarFracs_profile);
   const AccessorRO<double, 3> acc_temperature_profile (regions[0], FID_temperature_profile);
   const AccessorRO<  Vec3, 3> acc_velocity_profile    (regions[0], FID_velocity_profile);

   // Accessors for primitive variables
   const AccessorWO<double, 3> acc_pressure            (regions[1], FID_pressure);
   const AccessorWO<double, 3> acc_temperature         (regions[1], FID_temperature);
   const AccessorWO<VecNSp, 3> acc_MolarFracs          (regions[1], FID_MolarFracs);
   const AccessorWO<  Vec3, 3> acc_velocity            (regions[1], FID_velocity);

   // Accessors for avg wall-normal coordinate
   const AccessorRO<double, 1> acc_avg_y               (regions[2], RA_FID_y);

   // Accessors for recycle plane variables
   const AccessorRO<VecNSp, 3> acc_MolarFracs_recycle  (regions[3], FID_MolarFracs_recycle);
   const AccessorRO<double, 3> acc_temperature_recycle (regions[3], FID_temperature_recycle);
   const AccessorRO<  Vec3, 3> acc_velocity_recycle    (regions[3], FID_velocity_recycle);

   // Accessors for fast interpolation region
   const AccessorRO< float, 1> acc_FI_xloc             (regions[4], FI_FID_xloc);
   const AccessorRO< float, 1> acc_FI_iloc             (regions[4], FI_FID_iloc);

   // Extract execution domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[1].get_logical_region().get_index_space());

   // Compute rescaling coefficients
   const RescalingDataType RdataRe = futures[0].get_result<RescalingDataType>();
   const double yInnFact = RdataRe.deltaNu  /args.RdataIn.deltaNu;
   const double yOutFact = RdataRe.delta99VD/args.RdataIn.delta99VD;
   const double uInnFact = args.RdataIn.uTau/RdataRe.uTau;
   const double uOutFact = uInnFact*sqrt(args.RdataIn.rhow/RdataRe.rhow);

   const double idelta99Inl = 1.0/args.RdataIn.delta99VD;

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   struct SetRecycleRescalingBC_kernelArgs kArgs = {
      .centerCoordinates   = acc_centerCoordinates,
      .Conserved           = acc_Conserved,
      .SoS                 = acc_SoS,
      .temperature         = acc_temperature,
      .pressure            = acc_pressure,
      .MolarFracs          = acc_MolarFracs,
      .velocity            = acc_velocity,
      .temperature_recycle = acc_temperature_recycle,
      .velocity_recycle    = acc_velocity_recycle,
      .MolarFracs_recycle  = acc_MolarFracs_recycle,
      .temperature_profile = acc_temperature_profile,
      .velocity_profile    = acc_velocity_profile,
      .MolarFracs_profile  = acc_MolarFracs_profile,
      .avg_y               = acc_avg_y,
      .FI_xloc             = acc_FI_xloc,
      .FI_iloc             = acc_FI_iloc,
   };
#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<SetRecycleRescalingBC_kernelArgs, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   SetRecycleRescalingBC_kernel<<<num_blocks_3d, TPB_3d>>>(
                        buffer,
                        args.FIdata, args.Pbc,
                        yInnFact, yOutFact, uInnFact, uOutFact, idelta99Inl,
                        r_BC, getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#else
   SetRecycleRescalingBC_kernel<<<num_blocks_3d, TPB_3d>>>(
                        kArgs,
                        args.FIdata, args.Pbc,
                        yInnFact, yOutFact, uInnFact, uOutFact, idelta99Inl,
                        r_BC, getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
#endif
}

#if (defined(ELECTRIC_FIELD) && (nIons > 0))
//-----------------------------------------------------------------------------
// KERNELS FOR CorrectIonsBCTask
//-----------------------------------------------------------------------------

template<direction dir, side s>
__global__
void CorrectIonsBC_kernel(const AccessorRO<double, 3> ePotInt,
                          const AccessorRO<double, 3> ePot,
                          const AccessorRO<VecNSp, 3> MolarFracsInt,
                          const AccessorWO<VecNSp, 3> MolarFracs,
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
      const Point<3> pInt = getPIntBC<dir, s>(p);
      const double dPhi = ePotInt[pInt] - ePot[p];
      __UNROLL__
      for (int i = 0; i < nIons; i++) {
         int ind = mix.ions[i];
         if (mix.GetSpeciesChargeNumber(ind)*dPhi > 0)
            // the ion is flowing into the BC
            MolarFracs[p][ind] = MolarFracsInt[pInt][ind];
         else
            // the ion is repelled by the BC
            MolarFracs[p][ind] = 1e-60;
      }
   }
}

template<direction dir, side s>
__host__
void CorrectIonsBCTask<dir, s>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessor for BC electric potential
   const AccessorRO<double, 3> acc_ePot         (regions[0], FID_electricPotential);

   // Accessors for primitive variables
   const AccessorWO<VecNSp, 3> acc_MolarFracs   (regions[1], FID_MolarFracs);

   // Accessor for internal electric potential and molar fractions
   const AccessorRO<double, 3> acc_ePotInt      (regions[2], FID_electricPotential);
   const AccessorRO<VecNSp, 3> acc_MolarFracsInt(regions[2], FID_MolarFracs);

   // Extract execution domain
   const Rect<3> r_BC = runtime->get_index_space_domain(ctx,
                                 regions[1].get_logical_region().get_index_space());

   // Launch the kernel
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, r_BC);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_BC) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_BC) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_BC) + (TPB_3d.z - 1)) / TPB_3d.z);
   CorrectIonsBC_kernel<dir, s><<<num_blocks_3d, TPB_3d>>>(
                        acc_ePotInt, acc_ePot, acc_MolarFracsInt, acc_MolarFracs,
                        r_BC, getSize<Xdir>(r_BC), getSize<Ydir>(r_BC), getSize<Zdir>(r_BC));
};

template void CorrectIonsBCTask<Xdir, Minus>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectIonsBCTask<Xdir, Plus >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectIonsBCTask<Ydir, Minus>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectIonsBCTask<Ydir, Plus >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectIonsBCTask<Zdir, Minus>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void CorrectIonsBCTask<Zdir, Plus >::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);
#endif

