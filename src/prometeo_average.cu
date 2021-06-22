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

#ifdef BOUNDS_CHECKS
   // See Legion issue #879 for more info
   #warning "CUDA variant of average task are not compatible with BOUNDS_CHECKS. It is going to be disabled for these tasks"
   #undef BOUNDS_CHECKS
#endif

#include "prometeo_average.hpp"
#include "prometeo_average.inl"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNEL FOR Add2DAveragesTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void AvgPrimitive2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                           const AccessorRO<  Vec3, 3> centerCoordinates,
                           const AccessorRO<double, 3> pressure,
                           const AccessorRO<double, 3> temperature,
                           const AccessorRO<VecNSp, 3> MolarFracs,
                           const AccessorRO<VecNSp, 3> MassFracs,
                           const AccessorRO<  Vec3, 3> velocity,
                           const AccessorSumRD<double, 2> avg_weight,
                           const AccessorSumRD<  Vec3, 2> avg_centerCoordinates,
                           const AccessorSumRD<double, 2> pressure_avg,
                           const AccessorSumRD<double, 2> pressure_rms,
                           const AccessorSumRD<double, 2> temperature_avg,
                           const AccessorSumRD<double, 2> temperature_rms,
                           const AccessorSumRD<VecNSp, 2> MolarFracs_avg,
                           const AccessorSumRD<VecNSp, 2> MolarFracs_rms,
                           const AccessorSumRD<VecNSp, 2> MassFracs_avg,
                           const AccessorSumRD<VecNSp, 2> MassFracs_rms,
                           const AccessorSumRD<  Vec3, 2> velocity_avg,
                           const AccessorSumRD<  Vec3, 2> velocity_rms,
                           const AccessorSumRD<  Vec3, 2> velocity_rey,
                           const double deltaTime,
                           const coord_t nRake,
                           const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgPrimitive(cellWidth, centerCoordinates,
                        pressure, temperature, MolarFracs,
                        MassFracs, velocity,
                        avg_weight, avg_centerCoordinates,
                        pressure_avg,    pressure_rms,
                        temperature_avg, temperature_rms,
                        MolarFracs_avg,  MolarFracs_rms,
                        MassFracs_avg,   MassFracs_rms,
                        velocity_avg,    velocity_rms, velocity_rey,
                        p, pA, deltaTime);
   }
}

template<direction dir>
__global__
void FavreAvgPrimitive2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                const AccessorRO<double, 3> rho,
                                const AccessorRO<double, 3> pressure,
                                const AccessorRO<double, 3> temperature,
                                const AccessorRO<VecNSp, 3> MolarFracs,
                                const AccessorRO<VecNSp, 3> MassFracs,
                                const AccessorRO<  Vec3, 3> velocity,
                                const AccessorSumRD<double, 2> pressure_favg,
                                const AccessorSumRD<double, 2> pressure_frms,
                                const AccessorSumRD<double, 2> temperature_favg,
                                const AccessorSumRD<double, 2> temperature_frms,
                                const AccessorSumRD<VecNSp, 2> MolarFracs_favg,
                                const AccessorSumRD<VecNSp, 2> MolarFracs_frms,
                                const AccessorSumRD<VecNSp, 2> MassFracs_favg,
                                const AccessorSumRD<VecNSp, 2> MassFracs_frms,
                                const AccessorSumRD<  Vec3, 2> velocity_favg,
                                const AccessorSumRD<  Vec3, 2> velocity_frms,
                                const AccessorSumRD<  Vec3, 2> velocity_frey,
                                const double deltaTime,
                                const coord_t nRake,
                                const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::FavreAvgPrimitive(cellWidth, rho,
                        pressure, temperature, MolarFracs,
                        MassFracs, velocity,
                        pressure_favg,    pressure_frms,
                        temperature_favg, temperature_frms,
                        MolarFracs_favg,  MolarFracs_frms,
                        MassFracs_favg,   MassFracs_frms,
                        velocity_favg,    velocity_frms, velocity_frey,
                        p, pA, deltaTime);
   }
}

template<direction dir>
__global__
void AvgProperties2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                            const AccessorRO<double, 3> temperature,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3> rho,
                            const AccessorRO<double, 3> mu,
                            const AccessorRO<double, 3> lam,
                            const AccessorRO<VecNSp, 3> Di,
                            const AccessorRO<double, 3> SoS,
                            const AccessorSumRD<double, 2> rho_avg,
                            const AccessorSumRD<double, 2> rho_rms,
                            const AccessorSumRD<double, 2> mu_avg,
                            const AccessorSumRD<double, 2> lam_avg,
                            const AccessorSumRD<VecNSp, 2> Di_avg,
                            const AccessorSumRD<double, 2> SoS_avg,
                            const AccessorSumRD<double, 2> cp_avg,
                            const AccessorSumRD<double, 2> Ent_avg,
                            const double deltaTime,
                            const coord_t nRake,
                            const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgProperties(
                        cellWidth, temperature, MassFracs,
                        rho, mu, lam, Di, SoS,
                        rho_avg, rho_rms,
                        mu_avg, lam_avg, Di_avg,
                        SoS_avg, cp_avg, Ent_avg,
                        p, pA, mix, deltaTime);
   }
}

template<direction dir>
__global__
void FavreAvgProperties2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                 const AccessorRO<double, 3> temperature,
                                 const AccessorRO<VecNSp, 3> MassFracs,
                                 const AccessorRO<double, 3> rho,
                                 const AccessorRO<double, 3> mu,
                                 const AccessorRO<double, 3> lam,
                                 const AccessorRO<VecNSp, 3> Di,
                                 const AccessorRO<double, 3> SoS,
                                 const AccessorSumRD<double, 2> mu_favg,
                                 const AccessorSumRD<double, 2> lam_favg,
                                 const AccessorSumRD<VecNSp, 2> Di_favg,
                                 const AccessorSumRD<double, 2> SoS_favg,
                                 const AccessorSumRD<double, 2> cp_favg,
                                 const AccessorSumRD<double, 2> Ent_favg,
                                 const double deltaTime,
                                 const coord_t nRake,
                                 const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::FavreAvgProperties(
                        cellWidth, temperature, MassFracs,
                        rho, mu, lam, Di, SoS,
                        mu_favg, lam_favg, Di_favg,
                        SoS_favg, cp_favg, Ent_favg,
                        p, pA, mix, deltaTime);
   }
}

template<direction dir>
__global__
void AvgFluxes_ProdRates2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                  const AccessorRO<   int, 3> nType_x,
                                  const AccessorRO<   int, 3> nType_y,
                                  const AccessorRO<   int, 3> nType_z,
                                  const AccessorRO<double, 3> dcsi_d,
                                  const AccessorRO<double, 3> deta_d,
                                  const AccessorRO<double, 3> dzet_d,
                                  const AccessorRO<double, 3> pressure,
                                  const AccessorRO<double, 3> temperature,
                                  const AccessorRO<VecNSp, 3> MolarFracs,
                                  const AccessorRO<VecNSp, 3> MassFracs,
                                  const AccessorRO<double, 3> rho,
                                  const AccessorRO<double, 3> mu,
                                  const AccessorRO<double, 3> lam,
                                  const AccessorRO<VecNSp, 3> Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                  const AccessorRO<VecNIo, 3> Ki,
                                  const AccessorRO<  Vec3, 3> eField,
#endif
                                  const AccessorSumRD<  Vec3, 2> q_avg,
                                  const AccessorSumRD<VecNSp, 2> ProductionRates_avg,
                                  const AccessorSumRD<VecNSp, 2> ProductionRates_rms,
                                  const AccessorSumRD<double, 2> HeatReleaseRate_avg,
                                  const AccessorSumRD<double, 2> HeatReleaseRate_rms,
                                  const double deltaTime,
                                  const coord_t nRake,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgFluxes_ProdRates(cellWidth,
                        nType_x, nType_y, nType_z,
                        dcsi_d, deta_d, dzet_d,
                        pressure, temperature, MolarFracs, MassFracs,
                        rho, mu, lam, Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                        Ki, eField,
#endif
                        q_avg,
                        ProductionRates_avg, ProductionRates_rms,
                        HeatReleaseRate_avg, HeatReleaseRate_rms,
                        p, pA, Fluid_bounds, mix, deltaTime);
   }
}

template<direction dir>
__global__
void AvgKineticEnergyBudget2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                     const AccessorRO<double, 3> pressure,
                                     const AccessorRO<  Vec3, 3> velocity,
                                     const AccessorRO<double, 3> rho,
                                     const AccessorRO<double, 3> mu,
                                     const AccessorRO<  Vec3, 3> vGradX,
                                     const AccessorRO<  Vec3, 3> vGradY,
                                     const AccessorRO<  Vec3, 3> vGradZ,
                                     const AccessorSumRD<  Vec3, 2> rhoUUv_avg,
                                     const AccessorSumRD<  Vec3, 2> Up_avg,
                                     const AccessorSumRD<TauMat, 2> tau_avg,
                                     const AccessorSumRD<  Vec3, 2> utau_y_avg,
                                     const AccessorSumRD<  Vec3, 2> tauGradU_avg,
                                     const AccessorSumRD<  Vec3, 2> pGradU_avg,
                                     const double deltaTime,
                                     const coord_t nRake,
                                     const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgKineticEnergyBudget(cellWidth,
                        pressure, velocity,
                        rho, mu,
                        vGradX, vGradY, vGradZ,
                        rhoUUv_avg, Up_avg, tau_avg,
                        utau_y_avg, tauGradU_avg, pGradU_avg,
                        p, pA, mix, deltaTime);
   }
}

template<direction dir>
__global__
void AvgDimensionlessNumbers2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                      const AccessorRO<double, 3> temperature,
                                      const AccessorRO<VecNSp, 3> MassFracs,
                                      const AccessorRO<  Vec3, 3> velocity,
                                      const AccessorRO<double, 3> rho,
                                      const AccessorRO<double, 3> mu,
                                      const AccessorRO<double, 3> lam,
                                      const AccessorRO<VecNSp, 3> Di,
                                      const AccessorRO<double, 3> SoS,
                                      const AccessorSumRD<double, 2> Pr_avg,
                                      const AccessorSumRD<double, 2> Pr_rms,
                                      const AccessorSumRD<double, 2> Ec_avg,
                                      const AccessorSumRD<double, 2> Ec_rms,
                                      const AccessorSumRD<double, 2> Ma_avg,
                                      const AccessorSumRD<VecNSp, 2> Sc_avg,
                                      const double deltaTime,
                                      const coord_t nRake,
                                      const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgDimensionlessNumbers(cellWidth,
                        temperature, MassFracs, velocity,
                        rho, mu, lam, Di, SoS,
                        Pr_avg, Pr_rms,
                        Ec_avg, Ec_rms,
                        Ma_avg, Sc_avg,
                        p, pA, mix, deltaTime);
   }
}


template<direction dir>
__global__
void AvgCorrelations2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                              const AccessorRO<double, 3> temperature,
                              const AccessorRO<VecNSp, 3> MassFracs,
                              const AccessorRO<  Vec3, 3> velocity,
                              const AccessorRO<double, 3> rho,
                              const AccessorSumRD<  Vec3, 2> uT_avg,
                              const AccessorSumRD<  Vec3, 2> uT_favg,
                              const AccessorSumRD<VecNSp, 2> uYi_avg,
                              const AccessorSumRD<VecNSp, 2> vYi_avg,
                              const AccessorSumRD<VecNSp, 2> wYi_avg,
                              const AccessorSumRD<VecNSp, 2> uYi_favg,
                              const AccessorSumRD<VecNSp, 2> vYi_favg,
                              const AccessorSumRD<VecNSp, 2> wYi_favg,
                              const double deltaTime,
                              const coord_t nRake,
                              const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgCorrelations(cellWidth,
                        temperature, MassFracs, velocity, rho,
                        uT_avg, uT_favg,
                        uYi_avg,  vYi_avg,  wYi_avg,
                        uYi_favg, vYi_favg, wYi_favg,
                        p, pA, deltaTime);
   }
}

#ifdef ELECTRIC_FIELD
template<direction dir>
__global__
void AvgElectricQuantities2D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                    const AccessorRO<double, 3> temperature,
                                    const AccessorRO<VecNSp, 3> MolarFracs,
                                    const AccessorRO<double, 3> rho,
                                    const AccessorRO<double, 3> ePot,
                                    const AccessorSumRD<double, 2> ePot_avg,
                                    const AccessorSumRD<double, 2> Crg_avg,
                                    const double deltaTime,
                                    const coord_t nRake,
                                    const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<2> pA = (dir == Xdir) ? Point<2>{p.x, nRake} :
                          (dir == Ydir) ? Point<2>{p.y, nRake} :
                        /*(dir == Zdir)*/ Point<2>{p.z, nRake};
      Add2DAveragesTask<dir>::AvgElectricQuantities(cellWidth,
                        temperature, MolarFracs, rho, ePot,
                        ePot_avg, Crg_avg,
                        p, pA, mix, deltaTime);
   }
}
#endif

template<direction dir>
__host__
void Add2DAveragesTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 6);
   assert(futures.size() == 1);

   // Accessors for variables with gradient stencil access
   const AccessorRO<double, 3> acc_temperature         (regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MolarFracs          (regions[0], FID_MolarFracs);

   // Accessors for cell geometry
   const AccessorRO<  Vec3, 3> acc_centerCoordinates   (regions[1], FID_centerCoordinates);
   const AccessorRO<  Vec3, 3> acc_cellWidth           (regions[1], FID_cellWidth);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x             (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y             (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z             (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d              (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d              (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d              (regions[1], FID_dzet_d);

   // Accessors for primitive variables
   const AccessorRO<double, 3> acc_pressure            (regions[1], FID_pressure);
   const AccessorRO<VecNSp, 3> acc_MassFracs           (regions[1], FID_MassFracs);
   const AccessorRO<  Vec3, 3> acc_velocity            (regions[1], FID_velocity);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho                 (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_mu                  (regions[1], FID_mu);
   const AccessorRO<double, 3> acc_lam                 (regions[1], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di                  (regions[1], FID_Di);
   const AccessorRO<double, 3> acc_SoS                 (regions[1], FID_SoS);

   // Accessors for gradients
   const AccessorRO<  Vec3, 3> acc_vGradX              (regions[1], FID_velocityGradientX);
   const AccessorRO<  Vec3, 3> acc_vGradY              (regions[1], FID_velocityGradientY);
   const AccessorRO<  Vec3, 3> acc_vGradZ              (regions[1], FID_velocityGradientZ);

#ifdef ELECTRIC_FIELD
   // Accessors for electric variables
   const AccessorRO<double, 3> acc_ePot                (regions[1], FID_electricPotential);
#if (nIons > 0)
   const AccessorRO<  Vec3, 3> acc_eField              (regions[1], FID_electricField);
   const AccessorRO<VecNIo, 3> acc_Ki                  (regions[1], FID_Ki);
#endif
#endif

   // Accessors for averaged quantities
   //   - REGENT_REDOP_SUM_VEC3    -> iVec3
   //   - REGENT_REDOP_SUM_VECNSP  -> iVecNSp
   //   - REGENT_REDOP_SUM_VEC6    -> iVec6
   //   - LEGION_REDOP_SUM_FLOAT64 -> iDouble
   const AccessorSumRD<double, 2> acc_avg_weight            (regions[iDouble], AVE_FID_weight,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<  Vec3, 2> acc_avg_centerCoordinates (regions[iVec3  ], AVE_FID_centerCoordinates,   REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 2> acc_pressure_avg          (regions[iDouble], AVE_FID_pressure_avg,        LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_pressure_rms          (regions[iDouble], AVE_FID_pressure_rms,        LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_temperature_avg       (regions[iDouble], AVE_FID_temperature_avg,     LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_temperature_rms       (regions[iDouble], AVE_FID_temperature_rms,     LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 2> acc_MolarFracs_avg        (regions[iVecNSp], AVE_FID_MolarFracs_avg,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_MolarFracs_rms        (regions[iVecNSp], AVE_FID_MolarFracs_rms,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_MassFracs_avg         (regions[iVecNSp], AVE_FID_MassFracs_avg,       REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_MassFracs_rms         (regions[iVecNSp], AVE_FID_MassFracs_rms,       REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, 2> acc_velocity_avg          (regions[iVec3  ], AVE_FID_velocity_avg,        REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_velocity_rms          (regions[iVec3  ], AVE_FID_velocity_rms,        REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_velocity_rey          (regions[iVec3  ], AVE_FID_velocity_rey,        REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 2> acc_pressure_favg         (regions[iDouble], AVE_FID_pressure_favg,       LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_pressure_frms         (regions[iDouble], AVE_FID_pressure_frms,       LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_temperature_favg      (regions[iDouble], AVE_FID_temperature_favg,    LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_temperature_frms      (regions[iDouble], AVE_FID_temperature_frms,    LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 2> acc_MolarFracs_favg       (regions[iVecNSp], AVE_FID_MolarFracs_favg,     REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_MolarFracs_frms       (regions[iVecNSp], AVE_FID_MolarFracs_frms,     REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_MassFracs_favg        (regions[iVecNSp], AVE_FID_MassFracs_favg,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_MassFracs_frms        (regions[iVecNSp], AVE_FID_MassFracs_frms,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, 2> acc_velocity_favg         (regions[iVec3  ], AVE_FID_velocity_favg,       REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_velocity_frms         (regions[iVec3  ], AVE_FID_velocity_frms,       REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_velocity_frey         (regions[iVec3  ], AVE_FID_velocity_frey,       REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 2> acc_rho_avg               (regions[iDouble], AVE_FID_rho_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_rho_rms               (regions[iDouble], AVE_FID_rho_rms,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_mu_avg                (regions[iDouble], AVE_FID_mu_avg,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_lam_avg               (regions[iDouble], AVE_FID_lam_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 2> acc_Di_avg                (regions[iVecNSp], AVE_FID_Di_avg,              REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, 2> acc_SoS_avg               (regions[iDouble], AVE_FID_SoS_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_cp_avg                (regions[iDouble], AVE_FID_cp_avg,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Ent_avg               (regions[iDouble], AVE_FID_Ent_avg,             LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<double, 2> acc_mu_favg               (regions[iDouble], AVE_FID_mu_favg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_lam_favg              (regions[iDouble], AVE_FID_lam_favg,            LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 2> acc_Di_favg               (regions[iVecNSp], AVE_FID_Di_favg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, 2> acc_SoS_favg              (regions[iDouble], AVE_FID_SoS_favg,            LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_cp_favg               (regions[iDouble], AVE_FID_cp_favg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Ent_favg              (regions[iDouble], AVE_FID_Ent_favg,            LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<  Vec3, 2> acc_q_avg                 (regions[iVec3  ], AVE_FID_q,                   REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<VecNSp, 2> acc_ProductionRates_avg   (regions[iVecNSp], AVE_FID_ProductionRates_avg, REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_ProductionRates_rms   (regions[iVecNSp], AVE_FID_ProductionRates_rms, REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, 2> acc_HeatReleaseRate_avg   (regions[iDouble], AVE_FID_HeatReleaseRate_avg, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_HeatReleaseRate_rms   (regions[iDouble], AVE_FID_HeatReleaseRate_rms, LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<  Vec3, 2> acc_rhoUUv                (regions[iVec3  ], AVE_FID_rhoUUv,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_Up                    (regions[iVec3  ], AVE_FID_Up,                  REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<TauMat, 2> acc_tau                   (regions[iVec6  ], AVE_FID_tau,                 REGENT_REDOP_SUM_VEC6);
   const AccessorSumRD<  Vec3, 2> acc_utau_y                (regions[iVec3  ], AVE_FID_utau_y,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_tauGradU              (regions[iVec3  ], AVE_FID_tauGradU,            REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_pGradU                (regions[iVec3  ], AVE_FID_pGradU,              REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 2> acc_Pr_avg                (regions[iDouble], AVE_FID_Pr,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Pr_rms                (regions[iDouble], AVE_FID_Pr_rms,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Ec_avg                (regions[iDouble], AVE_FID_Ec,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Ec_rms                (regions[iDouble], AVE_FID_Ec_rms,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Ma_avg                (regions[iDouble], AVE_FID_Ma,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 2> acc_Sc_avg                (regions[iVecNSp], AVE_FID_Sc,                  REGENT_REDOP_SUM_VECNSP);

   const AccessorSumRD<  Vec3, 2> acc_uT_avg                (regions[iVec3  ], AVE_FID_uT_avg,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 2> acc_uT_favg               (regions[iVec3  ], AVE_FID_uT_favg,             REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<VecNSp, 2> acc_uYi_avg               (regions[iVecNSp], AVE_FID_uYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_vYi_avg               (regions[iVecNSp], AVE_FID_vYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_wYi_avg               (regions[iVecNSp], AVE_FID_wYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_uYi_favg              (regions[iVecNSp], AVE_FID_uYi_favg,            REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_vYi_favg              (regions[iVecNSp], AVE_FID_vYi_favg,            REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 2> acc_wYi_favg              (regions[iVecNSp], AVE_FID_wYi_favg,            REGENT_REDOP_SUM_VECNSP);

#ifdef ELECTRIC_FIELD
   const AccessorSumRD<double, 2> acc_ePot_avg              (regions[iDouble], AVE_FID_electricPotential_avg, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 2> acc_Crg_avg               (regions[iDouble], AVE_FID_chargeDensity_avg,     LEGION_REDOP_SUM_FLOAT64);
#endif

   // Extract execution domains
   Rect<3> r_Fluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());

   // Extract average domain
   Rect<2> r_Avg = runtime->get_index_space_domain(ctx, args.Averages.get_index_space());

   // Wait for the integrator deltaTime
   const double Integrator_deltaTime = futures[0].get_result<double>();

   // Set thread grid
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_Fluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_Fluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_Fluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_Fluid) + (TPB_3d.z - 1)) / TPB_3d.z);

   // each average kernel is going to use its own stream
   cudaStream_t            AvgPrimitiveStream;
   cudaStream_t       FavreAvgPrimitiveStream;
   cudaStream_t           AvgPropertiesStream;
   cudaStream_t      FavreAvgPropertiesStream;
   cudaStream_t     AvgFluxes_ProdRatesStream;
   cudaStream_t  AvgKineticEnergyBudgetStream;
   cudaStream_t AvgDimensionlessNumbersStream;
   cudaStream_t         AvgCorrelationsStream;
#ifdef ELECTRIC_FIELD
   cudaStream_t   AvgElectricQuantitiesStream;
#endif

   cudaStreamCreateWithFlags(           &AvgPrimitiveStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(      &FavreAvgPrimitiveStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(          &AvgPropertiesStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(     &FavreAvgPropertiesStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(    &AvgFluxes_ProdRatesStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags( &AvgKineticEnergyBudgetStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&AvgDimensionlessNumbersStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(        &AvgCorrelationsStream, cudaStreamNonBlocking);
#ifdef ELECTRIC_FIELD
   cudaStreamCreateWithFlags(  &AvgElectricQuantitiesStream, cudaStreamNonBlocking);
#endif

   // Collect averages of primitive variables
   AvgPrimitive2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgPrimitiveStream>>>(
                        acc_cellWidth, acc_centerCoordinates,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_avg_weight, acc_avg_centerCoordinates,
                        acc_pressure_avg,    acc_pressure_rms,
                        acc_temperature_avg, acc_temperature_rms,
                        acc_MolarFracs_avg,  acc_MolarFracs_rms,
                        acc_MassFracs_avg,   acc_MassFracs_rms,
                        acc_velocity_avg,    acc_velocity_rms, acc_velocity_rey,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect Favre averages of primitive variables
   FavreAvgPrimitive2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, FavreAvgPrimitiveStream>>>(
                        acc_cellWidth, acc_rho,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_pressure_favg,    acc_pressure_frms,
                        acc_temperature_favg, acc_temperature_frms,
                        acc_MolarFracs_favg,  acc_MolarFracs_frms,
                        acc_MassFracs_favg,   acc_MassFracs_frms,
                        acc_velocity_favg,    acc_velocity_frms, acc_velocity_frey,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of properties
   AvgProperties2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgPropertiesStream>>>(
                        acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_rho_avg, acc_rho_rms,
                        acc_mu_avg, acc_lam_avg, acc_Di_avg,
                        acc_SoS_avg, acc_cp_avg, acc_Ent_avg,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect Favre averages of properties
   FavreAvgProperties2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, FavreAvgPrimitiveStream>>>(
                        acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_mu_favg, acc_lam_favg, acc_Di_favg,
                        acc_SoS_favg, acc_cp_favg, acc_Ent_favg,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of fluxes and production rates
   AvgFluxes_ProdRates2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgFluxes_ProdRatesStream>>>(
                        acc_cellWidth,
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        acc_pressure, acc_temperature, acc_MolarFracs, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                        acc_Ki, acc_eField,
#endif
                        acc_q_avg,
                        acc_ProductionRates_avg, acc_ProductionRates_rms,
                        acc_HeatReleaseRate_avg, acc_HeatReleaseRate_rms,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid, args.Fluid_bounds,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of kinetic energy budget terms
   AvgKineticEnergyBudget2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgKineticEnergyBudgetStream>>>(
                        acc_cellWidth,
                        acc_pressure, acc_velocity,
                        acc_rho, acc_mu,
                        acc_vGradX, acc_vGradY, acc_vGradZ,
                        acc_rhoUUv, acc_Up, acc_tau,
                        acc_utau_y, acc_tauGradU, acc_pGradU,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of dimensionless numbers
   AvgDimensionlessNumbers2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgDimensionlessNumbersStream>>>(
                        acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_Pr_avg, acc_Pr_rms,
                        acc_Ec_avg, acc_Ec_rms,
                        acc_Ma_avg, acc_Sc_avg,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of correlations
   AvgCorrelations2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgCorrelationsStream>>>(
                        acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity, acc_rho,
                        acc_uT_avg, acc_uT_favg,
                        acc_uYi_avg, acc_vYi_avg, acc_wYi_avg,
                        acc_uYi_favg, acc_vYi_favg, acc_wYi_favg,
                        Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

#ifdef ELECTRIC_FIELD
   // Collect averages of electric quantities
   AvgElectricQuantities2D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgElectricQuantitiesStream>>>(
               acc_cellWidth,
               acc_temperature, acc_MolarFracs, acc_rho, acc_ePot,
               acc_ePot_avg, acc_Crg_avg,
               Integrator_deltaTime, r_Avg.lo.y, r_Fluid,
               getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));
#endif

   // Cleanup streams
   cudaStreamDestroy(           AvgPrimitiveStream);
   cudaStreamDestroy(      FavreAvgPrimitiveStream);
   cudaStreamDestroy(          AvgPropertiesStream);
   cudaStreamDestroy(     FavreAvgPropertiesStream);
   cudaStreamDestroy(    AvgFluxes_ProdRatesStream);
   cudaStreamDestroy( AvgKineticEnergyBudgetStream);
   cudaStreamDestroy(AvgDimensionlessNumbersStream);
   cudaStreamDestroy(        AvgCorrelationsStream);
#ifdef ELECTRIC_FIELD
   cudaStreamDestroy(  AvgElectricQuantitiesStream);
#endif
}

template void Add2DAveragesTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void Add2DAveragesTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void Add2DAveragesTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

//-----------------------------------------------------------------------------
// KERNEL FOR Add1DAveragesTask
//-----------------------------------------------------------------------------

template<direction dir>
__global__
void AvgPrimitive1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                           const AccessorRO<  Vec3, 3> centerCoordinates,
                           const AccessorRO<double, 3> pressure,
                           const AccessorRO<double, 3> temperature,
                           const AccessorRO<VecNSp, 3> MolarFracs,
                           const AccessorRO<VecNSp, 3> MassFracs,
                           const AccessorRO<  Vec3, 3> velocity,
                           const AccessorSumRD<double, 3> avg_weight,
                           const AccessorSumRD<  Vec3, 3> avg_centerCoordinates,
                           const AccessorSumRD<double, 3> pressure_avg,
                           const AccessorSumRD<double, 3> pressure_rms,
                           const AccessorSumRD<double, 3> temperature_avg,
                           const AccessorSumRD<double, 3> temperature_rms,
                           const AccessorSumRD<VecNSp, 3> MolarFracs_avg,
                           const AccessorSumRD<VecNSp, 3> MolarFracs_rms,
                           const AccessorSumRD<VecNSp, 3> MassFracs_avg,
                           const AccessorSumRD<VecNSp, 3> MassFracs_rms,
                           const AccessorSumRD<  Vec3, 3> velocity_avg,
                           const AccessorSumRD<  Vec3, 3> velocity_rms,
                           const AccessorSumRD<  Vec3, 3> velocity_rey,
                           const double deltaTime,
                           const coord_t nPlane,
                           const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgPrimitive(cellWidth, centerCoordinates,
                        pressure, temperature, MolarFracs,
                        MassFracs, velocity,
                        avg_weight, avg_centerCoordinates,
                        pressure_avg,    pressure_rms,
                        temperature_avg, temperature_rms,
                        MolarFracs_avg,  MolarFracs_rms,
                        MassFracs_avg,   MassFracs_rms,
                        velocity_avg,    velocity_rms, velocity_rey,
                        p, pA, deltaTime);
   }
}

template<direction dir>
__global__
void FavreAvgPrimitive1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                const AccessorRO<double, 3> rho,
                                const AccessorRO<double, 3> pressure,
                                const AccessorRO<double, 3> temperature,
                                const AccessorRO<VecNSp, 3> MolarFracs,
                                const AccessorRO<VecNSp, 3> MassFracs,
                                const AccessorRO<  Vec3, 3> velocity,
                                const AccessorSumRD<double, 3> pressure_favg,
                                const AccessorSumRD<double, 3> pressure_frms,
                                const AccessorSumRD<double, 3> temperature_favg,
                                const AccessorSumRD<double, 3> temperature_frms,
                                const AccessorSumRD<VecNSp, 3> MolarFracs_favg,
                                const AccessorSumRD<VecNSp, 3> MolarFracs_frms,
                                const AccessorSumRD<VecNSp, 3> MassFracs_favg,
                                const AccessorSumRD<VecNSp, 3> MassFracs_frms,
                                const AccessorSumRD<  Vec3, 3> velocity_favg,
                                const AccessorSumRD<  Vec3, 3> velocity_frms,
                                const AccessorSumRD<  Vec3, 3> velocity_frey,
                                const double deltaTime,
                                const coord_t nPlane,
                                const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::FavreAvgPrimitive(cellWidth, rho,
                        pressure, temperature, MolarFracs,
                        MassFracs, velocity,
                        pressure_favg,    pressure_frms,
                        temperature_favg, temperature_frms,
                        MolarFracs_favg,  MolarFracs_frms,
                        MassFracs_favg,   MassFracs_frms,
                        velocity_favg,    velocity_frms, velocity_frey,
                        p, pA, deltaTime);
   }
}

template<direction dir>
__global__
void AvgProperties1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                            const AccessorRO<double, 3> temperature,
                            const AccessorRO<VecNSp, 3> MassFracs,
                            const AccessorRO<double, 3> rho,
                            const AccessorRO<double, 3> mu,
                            const AccessorRO<double, 3> lam,
                            const AccessorRO<VecNSp, 3> Di,
                            const AccessorRO<double, 3> SoS,
                            const AccessorSumRD<double, 3> rho_avg,
                            const AccessorSumRD<double, 3> rho_rms,
                            const AccessorSumRD<double, 3> mu_avg,
                            const AccessorSumRD<double, 3> lam_avg,
                            const AccessorSumRD<VecNSp, 3> Di_avg,
                            const AccessorSumRD<double, 3> SoS_avg,
                            const AccessorSumRD<double, 3> cp_avg,
                            const AccessorSumRD<double, 3> Ent_avg,
                            const double deltaTime,
                            const coord_t nPlane,
                            const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgProperties(
                        cellWidth, temperature, MassFracs,
                        rho, mu, lam, Di, SoS,
                        rho_avg, rho_rms,
                        mu_avg, lam_avg, Di_avg,
                        SoS_avg, cp_avg, Ent_avg,
                        p, pA, mix, deltaTime);
   }
}

template<direction dir>
__global__
void FavreAvgProperties1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                 const AccessorRO<double, 3> temperature,
                                 const AccessorRO<VecNSp, 3> MassFracs,
                                 const AccessorRO<double, 3> rho,
                                 const AccessorRO<double, 3> mu,
                                 const AccessorRO<double, 3> lam,
                                 const AccessorRO<VecNSp, 3> Di,
                                 const AccessorRO<double, 3> SoS,
                                 const AccessorSumRD<double, 3> mu_favg,
                                 const AccessorSumRD<double, 3> lam_favg,
                                 const AccessorSumRD<VecNSp, 3> Di_favg,
                                 const AccessorSumRD<double, 3> SoS_favg,
                                 const AccessorSumRD<double, 3> cp_favg,
                                 const AccessorSumRD<double, 3> Ent_favg,
                                 const double deltaTime,
                                 const coord_t nPlane,
                                 const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::FavreAvgProperties(
                        cellWidth, temperature, MassFracs,
                        rho, mu, lam, Di, SoS,
                        mu_favg, lam_favg, Di_favg,
                        SoS_favg, cp_favg, Ent_favg,
                        p, pA, mix, deltaTime);
   }
}

template<direction dir>
__global__
void AvgFluxes_ProdRates1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                  const AccessorRO<   int, 3> nType_x,
                                  const AccessorRO<   int, 3> nType_y,
                                  const AccessorRO<   int, 3> nType_z,
                                  const AccessorRO<double, 3> dcsi_d,
                                  const AccessorRO<double, 3> deta_d,
                                  const AccessorRO<double, 3> dzet_d,
                                  const AccessorRO<double, 3> pressure,
                                  const AccessorRO<double, 3> temperature,
                                  const AccessorRO<VecNSp, 3> MolarFracs,
                                  const AccessorRO<VecNSp, 3> MassFracs,
                                  const AccessorRO<double, 3> rho,
                                  const AccessorRO<double, 3> mu,
                                  const AccessorRO<double, 3> lam,
                                  const AccessorRO<VecNSp, 3> Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                  const AccessorRO<VecNIo, 3> Ki,
                                  const AccessorRO<  Vec3, 3> eField,
#endif
                                  const AccessorSumRD<  Vec3, 3> q_avg,
                                  const AccessorSumRD<VecNSp, 3> ProductionRates_avg,
                                  const AccessorSumRD<VecNSp, 3> ProductionRates_rms,
                                  const AccessorSumRD<double, 3> HeatReleaseRate_avg,
                                  const AccessorSumRD<double, 3> HeatReleaseRate_rms,
                                  const double deltaTime,
                                  const coord_t nPlane,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgFluxes_ProdRates(cellWidth,
                        nType_x, nType_y, nType_z,
                        dcsi_d, deta_d, dzet_d,
                        pressure, temperature, MolarFracs, MassFracs,
                        rho, mu, lam, Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                        Ki, eField,
#endif
                        q_avg,
                        ProductionRates_avg, ProductionRates_rms,
                        HeatReleaseRate_avg, HeatReleaseRate_rms,
                        p, pA, Fluid_bounds, mix, deltaTime);
   }
}

template<direction dir>
__global__
void AvgKineticEnergyBudget1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                     const AccessorRO<double, 3> pressure,
                                     const AccessorRO<  Vec3, 3> velocity,
                                     const AccessorRO<double, 3> rho,
                                     const AccessorRO<double, 3> mu,
                                     const AccessorRO<  Vec3, 3> vGradX,
                                     const AccessorRO<  Vec3, 3> vGradY,
                                     const AccessorRO<  Vec3, 3> vGradZ,
                                     const AccessorSumRD<  Vec3, 3> rhoUUv_avg,
                                     const AccessorSumRD<  Vec3, 3> Up_avg,
                                     const AccessorSumRD<TauMat, 3> tau_avg,
                                     const AccessorSumRD<  Vec3, 3> utau_y_avg,
                                     const AccessorSumRD<  Vec3, 3> tauGradU_avg,
                                     const AccessorSumRD<  Vec3, 3> pGradU_avg,
                                     const double deltaTime,
                                     const coord_t nPlane,
                                     const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgKineticEnergyBudget(cellWidth,
                        pressure, velocity,
                        rho, mu,
                        vGradX, vGradY, vGradZ,
                        rhoUUv_avg, Up_avg, tau_avg,
                        utau_y_avg, tauGradU_avg, pGradU_avg,
                        p, pA, mix, deltaTime);
   }
}

template<direction dir>
__global__
void AvgDimensionlessNumbers1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                      const AccessorRO<double, 3> temperature,
                                      const AccessorRO<VecNSp, 3> MassFracs,
                                      const AccessorRO<  Vec3, 3> velocity,
                                      const AccessorRO<double, 3> rho,
                                      const AccessorRO<double, 3> mu,
                                      const AccessorRO<double, 3> lam,
                                      const AccessorRO<VecNSp, 3> Di,
                                      const AccessorRO<double, 3> SoS,
                                      const AccessorSumRD<double, 3> Pr_avg,
                                      const AccessorSumRD<double, 3> Pr_rms,
                                      const AccessorSumRD<double, 3> Ec_avg,
                                      const AccessorSumRD<double, 3> Ec_rms,
                                      const AccessorSumRD<double, 3> Ma_avg,
                                      const AccessorSumRD<VecNSp, 3> Sc_avg,
                                      const double deltaTime,
                                      const coord_t nPlane,
                                      const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgDimensionlessNumbers(cellWidth,
                        temperature, MassFracs, velocity,
                        rho, mu, lam, Di, SoS,
                        Pr_avg, Pr_rms,
                        Ec_avg, Ec_rms,
                        Ma_avg, Sc_avg,
                        p, pA, mix, deltaTime);
   }
}


template<direction dir>
__global__
void AvgCorrelations1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                              const AccessorRO<double, 3> temperature,
                              const AccessorRO<VecNSp, 3> MassFracs,
                              const AccessorRO<  Vec3, 3> velocity,
                              const AccessorRO<double, 3> rho,
                              const AccessorSumRD<  Vec3, 3> uT_avg,
                              const AccessorSumRD<  Vec3, 3> uT_favg,
                              const AccessorSumRD<VecNSp, 3> uYi_avg,
                              const AccessorSumRD<VecNSp, 3> vYi_avg,
                              const AccessorSumRD<VecNSp, 3> wYi_avg,
                              const AccessorSumRD<VecNSp, 3> uYi_favg,
                              const AccessorSumRD<VecNSp, 3> vYi_favg,
                              const AccessorSumRD<VecNSp, 3> wYi_favg,
                              const double deltaTime,
                              const coord_t nPlane,
                              const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgCorrelations(cellWidth,
                        temperature, MassFracs, velocity, rho,
                        uT_avg, uT_favg,
                        uYi_avg,  vYi_avg,  wYi_avg,
                        uYi_favg, vYi_favg, wYi_favg,
                        p, pA, deltaTime);
   }
}

#ifdef ELECTRIC_FIELD
template<direction dir>
__global__
void AvgElectricQuantities1D_kernel(const AccessorRO<  Vec3, 3> cellWidth,
                                    const AccessorRO<double, 3> temperature,
                                    const AccessorRO<VecNSp, 3> MolarFracs,
                                    const AccessorRO<double, 3> rho,
                                    const AccessorRO<double, 3> ePot,
                                    const AccessorSumRD<double, 3> ePot_avg,
                                    const AccessorSumRD<double, 3> Crg_avg,
                                    const double deltaTime,
                                    const coord_t nPlane,
                                    const Rect<3> my_bounds,
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
      // TODO: implement some sort of static_if
      const Point<3> pA = (dir == Xdir) ? Point<3>{p.y, p.z, nPlane} :
                          (dir == Ydir) ? Point<3>{p.x, p.z, nPlane} :
                        /*(dir == Zdir)*/ Point<3>{p.x, p.y, nPlane};
      Add1DAveragesTask<dir>::AvgElectricQuantities(cellWidth,
                        temperature, MolarFracs, rho, ePot,
                        ePot_avg, Crg_avg,
                        p, pA, mix, deltaTime);
   }
}
#endif

template<direction dir>
__host__
void Add1DAveragesTask<dir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 6);
   assert(futures.size() == 1);

   // Accessors for variables with gradient stencil access
   const AccessorRO<double, 3> acc_temperature         (regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MolarFracs          (regions[0], FID_MolarFracs);

   // Accessors for cell geometry
   const AccessorRO<  Vec3, 3> acc_centerCoordinates   (regions[1], FID_centerCoordinates);
   const AccessorRO<  Vec3, 3> acc_cellWidth           (regions[1], FID_cellWidth);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x             (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y             (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z             (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d              (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d              (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d              (regions[1], FID_dzet_d);

   // Accessors for primitive variables
   const AccessorRO<double, 3> acc_pressure            (regions[1], FID_pressure);
   const AccessorRO<VecNSp, 3> acc_MassFracs           (regions[1], FID_MassFracs);
   const AccessorRO<  Vec3, 3> acc_velocity            (regions[1], FID_velocity);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho                 (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_mu                  (regions[1], FID_mu);
   const AccessorRO<double, 3> acc_lam                 (regions[1], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di                  (regions[1], FID_Di);
   const AccessorRO<double, 3> acc_SoS                 (regions[1], FID_SoS);

   // Accessors for gradients
   const AccessorRO<  Vec3, 3> acc_vGradX              (regions[1], FID_velocityGradientX);
   const AccessorRO<  Vec3, 3> acc_vGradY              (regions[1], FID_velocityGradientY);
   const AccessorRO<  Vec3, 3> acc_vGradZ              (regions[1], FID_velocityGradientZ);

#ifdef ELECTRIC_FIELD
   // Accessors for electric variables
   const AccessorRO<double, 3> acc_ePot                (regions[1], FID_electricPotential);
#if (nIons > 0)
   const AccessorRO<  Vec3, 3> acc_eField              (regions[1], FID_electricField);
   const AccessorRO<VecNIo, 3> acc_Ki                  (regions[1], FID_Ki);
#endif
#endif

   // Accessors for averaged quantities
   //   - REGENT_REDOP_SUM_VEC3    -> iVec3
   //   - REGENT_REDOP_SUM_VECNSP  -> iVecNSp
   //   - REGENT_REDOP_SUM_VEC6    -> iVec6
   //   - LEGION_REDOP_SUM_FLOAT64 -> iDouble
   const AccessorSumRD<double, 3> acc_avg_weight            (regions[iDouble], AVE_FID_weight,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<  Vec3, 3> acc_avg_centerCoordinates (regions[iVec3  ], AVE_FID_centerCoordinates,   REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 3> acc_pressure_avg          (regions[iDouble], AVE_FID_pressure_avg,        LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_pressure_rms          (regions[iDouble], AVE_FID_pressure_rms,        LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_temperature_avg       (regions[iDouble], AVE_FID_temperature_avg,     LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_temperature_rms       (regions[iDouble], AVE_FID_temperature_rms,     LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 3> acc_MolarFracs_avg        (regions[iVecNSp], AVE_FID_MolarFracs_avg,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_MolarFracs_rms        (regions[iVecNSp], AVE_FID_MolarFracs_rms,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_MassFracs_avg         (regions[iVecNSp], AVE_FID_MassFracs_avg,       REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_MassFracs_rms         (regions[iVecNSp], AVE_FID_MassFracs_rms,       REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, 3> acc_velocity_avg          (regions[iVec3  ], AVE_FID_velocity_avg,        REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_velocity_rms          (regions[iVec3  ], AVE_FID_velocity_rms,        REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_velocity_rey          (regions[iVec3  ], AVE_FID_velocity_rey,        REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 3> acc_pressure_favg         (regions[iDouble], AVE_FID_pressure_favg,       LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_pressure_frms         (regions[iDouble], AVE_FID_pressure_frms,       LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_temperature_favg      (regions[iDouble], AVE_FID_temperature_favg,    LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_temperature_frms      (regions[iDouble], AVE_FID_temperature_frms,    LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 3> acc_MolarFracs_favg       (regions[iVecNSp], AVE_FID_MolarFracs_favg,     REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_MolarFracs_frms       (regions[iVecNSp], AVE_FID_MolarFracs_frms,     REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_MassFracs_favg        (regions[iVecNSp], AVE_FID_MassFracs_favg,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_MassFracs_frms        (regions[iVecNSp], AVE_FID_MassFracs_frms,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, 3> acc_velocity_favg         (regions[iVec3  ], AVE_FID_velocity_favg,       REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_velocity_frms         (regions[iVec3  ], AVE_FID_velocity_frms,       REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_velocity_frey         (regions[iVec3  ], AVE_FID_velocity_frey,       REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 3> acc_rho_avg               (regions[iDouble], AVE_FID_rho_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_rho_rms               (regions[iDouble], AVE_FID_rho_rms,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_mu_avg                (regions[iDouble], AVE_FID_mu_avg,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_lam_avg               (regions[iDouble], AVE_FID_lam_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 3> acc_Di_avg                (regions[iVecNSp], AVE_FID_Di_avg,              REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, 3> acc_SoS_avg               (regions[iDouble], AVE_FID_SoS_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_cp_avg                (regions[iDouble], AVE_FID_cp_avg,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Ent_avg               (regions[iDouble], AVE_FID_Ent_avg,             LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<double, 3> acc_mu_favg               (regions[iDouble], AVE_FID_mu_favg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_lam_favg              (regions[iDouble], AVE_FID_lam_favg,            LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 3> acc_Di_favg               (regions[iVecNSp], AVE_FID_Di_favg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, 3> acc_SoS_favg              (regions[iDouble], AVE_FID_SoS_favg,            LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_cp_favg               (regions[iDouble], AVE_FID_cp_favg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Ent_favg              (regions[iDouble], AVE_FID_Ent_favg,            LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<  Vec3, 3> acc_q_avg                 (regions[iVec3  ], AVE_FID_q,                   REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<VecNSp, 3> acc_ProductionRates_avg   (regions[iVecNSp], AVE_FID_ProductionRates_avg, REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_ProductionRates_rms   (regions[iVecNSp], AVE_FID_ProductionRates_rms, REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, 3> acc_HeatReleaseRate_avg   (regions[iDouble], AVE_FID_HeatReleaseRate_avg, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_HeatReleaseRate_rms   (regions[iDouble], AVE_FID_HeatReleaseRate_rms, LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<  Vec3, 3> acc_rhoUUv                (regions[iVec3  ], AVE_FID_rhoUUv,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_Up                    (regions[iVec3  ], AVE_FID_Up,                  REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<TauMat, 3> acc_tau                   (regions[iVec6  ], AVE_FID_tau,                 REGENT_REDOP_SUM_VEC6);
   const AccessorSumRD<  Vec3, 3> acc_utau_y                (regions[iVec3  ], AVE_FID_utau_y,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_tauGradU              (regions[iVec3  ], AVE_FID_tauGradU,            REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_pGradU                (regions[iVec3  ], AVE_FID_pGradU,              REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, 3> acc_Pr_avg                (regions[iDouble], AVE_FID_Pr,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Pr_rms                (regions[iDouble], AVE_FID_Pr_rms,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Ec_avg                (regions[iDouble], AVE_FID_Ec,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Ec_rms                (regions[iDouble], AVE_FID_Ec_rms,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Ma_avg                (regions[iDouble], AVE_FID_Ma,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, 3> acc_Sc_avg                (regions[iVecNSp], AVE_FID_Sc,                  REGENT_REDOP_SUM_VECNSP);

   const AccessorSumRD<  Vec3, 3> acc_uT_avg                (regions[iVec3  ], AVE_FID_uT_avg,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, 3> acc_uT_favg               (regions[iVec3  ], AVE_FID_uT_favg,             REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<VecNSp, 3> acc_uYi_avg               (regions[iVecNSp], AVE_FID_uYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_vYi_avg               (regions[iVecNSp], AVE_FID_vYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_wYi_avg               (regions[iVecNSp], AVE_FID_wYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_uYi_favg              (regions[iVecNSp], AVE_FID_uYi_favg,            REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_vYi_favg              (regions[iVecNSp], AVE_FID_vYi_favg,            REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, 3> acc_wYi_favg              (regions[iVecNSp], AVE_FID_wYi_favg,            REGENT_REDOP_SUM_VECNSP);

#ifdef ELECTRIC_FIELD
   const AccessorSumRD<double, 3> acc_ePot_avg              (regions[iDouble], AVE_FID_electricPotential_avg, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, 3> acc_Crg_avg               (regions[iDouble], AVE_FID_chargeDensity_avg,     LEGION_REDOP_SUM_FLOAT64);
#endif

   // Extract execution domains
   Rect<3> r_Fluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());

   // Extract average domain
   Rect<3> r_Avg = runtime->get_index_space_domain(ctx, args.Averages.get_index_space());

   // Wait for the integrator deltaTime
   const double Integrator_deltaTime = futures[0].get_result<double>();

   // Set thread grid
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_Fluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_Fluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_Fluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_Fluid) + (TPB_3d.z - 1)) / TPB_3d.z);

   // each average kernel is going to use its own stream
   cudaStream_t            AvgPrimitiveStream;
   cudaStream_t       FavreAvgPrimitiveStream;
   cudaStream_t           AvgPropertiesStream;
   cudaStream_t      FavreAvgPropertiesStream;
   cudaStream_t     AvgFluxes_ProdRatesStream;
   cudaStream_t  AvgKineticEnergyBudgetStream;
   cudaStream_t AvgDimensionlessNumbersStream;
   cudaStream_t         AvgCorrelationsStream;
#ifdef ELECTRIC_FIELD
   cudaStream_t   AvgElectricQuantitiesStream;
#endif

   cudaStreamCreateWithFlags(           &AvgPrimitiveStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(      &FavreAvgPrimitiveStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(          &AvgPropertiesStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(     &FavreAvgPropertiesStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(    &AvgFluxes_ProdRatesStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags( &AvgKineticEnergyBudgetStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(&AvgDimensionlessNumbersStream, cudaStreamNonBlocking);
   cudaStreamCreateWithFlags(        &AvgCorrelationsStream, cudaStreamNonBlocking);
#ifdef ELECTRIC_FIELD
   cudaStreamCreateWithFlags(  &AvgElectricQuantitiesStream, cudaStreamNonBlocking);
#endif

   // Collect averages of primitive variables
   AvgPrimitive1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgPrimitiveStream>>>(
                        acc_cellWidth, acc_centerCoordinates,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_avg_weight, acc_avg_centerCoordinates,
                        acc_pressure_avg,    acc_pressure_rms,
                        acc_temperature_avg, acc_temperature_rms,
                        acc_MolarFracs_avg,  acc_MolarFracs_rms,
                        acc_MassFracs_avg,   acc_MassFracs_rms,
                        acc_velocity_avg,    acc_velocity_rms, acc_velocity_rey,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect Favre averages of primitive variables
   FavreAvgPrimitive1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, FavreAvgPrimitiveStream>>>(
                        acc_cellWidth, acc_rho,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_pressure_favg,    acc_pressure_frms,
                        acc_temperature_favg, acc_temperature_frms,
                        acc_MolarFracs_favg,  acc_MolarFracs_frms,
                        acc_MassFracs_favg,   acc_MassFracs_frms,
                        acc_velocity_favg,    acc_velocity_frms, acc_velocity_frey,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of properties
   AvgProperties1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgPropertiesStream>>>(
                        acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_rho_avg, acc_rho_rms,
                        acc_mu_avg, acc_lam_avg, acc_Di_avg,
                        acc_SoS_avg, acc_cp_avg, acc_Ent_avg,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect Favre averages of properties
   FavreAvgProperties1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, FavreAvgPrimitiveStream>>>(
                        acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_mu_favg, acc_lam_favg, acc_Di_favg,
                        acc_SoS_favg, acc_cp_favg, acc_Ent_favg,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of fluxes and production rates
   AvgFluxes_ProdRates1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgFluxes_ProdRatesStream>>>(
                        acc_cellWidth,
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        acc_pressure, acc_temperature, acc_MolarFracs, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                        acc_Ki, acc_eField,
#endif
                        acc_q_avg,
                        acc_ProductionRates_avg, acc_ProductionRates_rms,
                        acc_HeatReleaseRate_avg, acc_HeatReleaseRate_rms,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid, args.Fluid_bounds,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of kinetic energy budget terms
   AvgKineticEnergyBudget1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgKineticEnergyBudgetStream>>>(
                        acc_cellWidth,
                        acc_pressure, acc_velocity,
                        acc_rho, acc_mu,
                        acc_vGradX, acc_vGradY, acc_vGradZ,
                        acc_rhoUUv, acc_Up, acc_tau,
                        acc_utau_y, acc_tauGradU, acc_pGradU,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of dimensionless numbers
   AvgDimensionlessNumbers1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgDimensionlessNumbersStream>>>(
                        acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_Pr_avg, acc_Pr_rms,
                        acc_Ec_avg, acc_Ec_rms,
                        acc_Ma_avg, acc_Sc_avg,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of correlations
   AvgCorrelations1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgCorrelationsStream>>>(
                        acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity, acc_rho,
                        acc_uT_avg, acc_uT_favg,
                        acc_uYi_avg, acc_vYi_avg, acc_wYi_avg,
                        acc_uYi_favg, acc_vYi_favg, acc_wYi_favg,
                        Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
                        getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

#ifdef ELECTRIC_FIELD
   // Collect averages of electric quantities
   AvgElectricQuantities1D_kernel<dir><<<num_blocks_3d, TPB_3d, 0, AvgElectricQuantitiesStream>>>(
               acc_cellWidth,
               acc_temperature, acc_MolarFracs, acc_rho, acc_ePot,
               acc_ePot_avg, acc_Crg_avg,
               Integrator_deltaTime, r_Avg.lo.z, r_Fluid,
               getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));
#endif

   // Cleanup streams
   cudaStreamDestroy(           AvgPrimitiveStream);
   cudaStreamDestroy(      FavreAvgPrimitiveStream);
   cudaStreamDestroy(          AvgPropertiesStream);
   cudaStreamDestroy(     FavreAvgPropertiesStream);
   cudaStreamDestroy(    AvgFluxes_ProdRatesStream);
   cudaStreamDestroy( AvgKineticEnergyBudgetStream);
   cudaStreamDestroy(AvgDimensionlessNumbersStream);
   cudaStreamDestroy(        AvgCorrelationsStream);
#ifdef ELECTRIC_FIELD
   cudaStreamDestroy(  AvgElectricQuantitiesStream);
#endif
}

template void Add1DAveragesTask<Xdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void Add1DAveragesTask<Ydir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void Add1DAveragesTask<Zdir>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);


