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

#include "prometeo_average.hpp"
#include "prometeo_average.inl"
#include "cuda_utils.hpp"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNELS FOR AddAveragesTask
//-----------------------------------------------------------------------------

template<direction dir, int N>
__global__
void PositionAndWeight_kernel(const AccessorRO<  Vec3, 3> centerCoordinates,
                         const AccessorSumRD<double, N> avg_weight,
                         const AccessorSumRD<  Vec3, N> avg_centerCoordinates,
                         const AccessorRO<double, 3> dcsi_d,
                         const AccessorRO<double, 3> deta_d,
                         const AccessorRO<double, 3> dzet_d,
                         const double deltaTime,
                         const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::PositionAndWeight(centerCoordinates, avg_weight, avg_centerCoordinates, p, pA, weight);
   }
};

template<direction dir, int N, typename T>
__global__
void Avg_kernel(const AccessorRO<T, 3> f,
                const AccessorSumRD<T, N> avg,
                const AccessorSumRD<T, N> favg,
                const AccessorRO<double, 3> rho,
                const AccessorRO<double, 3> dcsi_d,
                const AccessorRO<double, 3> deta_d,
                const AccessorRO<double, 3> dzet_d,
                const double deltaTime,
                const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Avg(f, avg, p, pA, weight);
      const double fweight = AverageUtils<N>::getFavreWeight(dcsi_d, deta_d, dzet_d, rho, p, deltaTime);
      AverageUtils<N>::Avg(f, favg, p, pA, fweight);
   }
};

template<direction dir, int N, typename T>
__global__
void Avg_kernel(const AccessorRO<T, 3> f,
                const AccessorSumRD<T, N> avg,
                const AccessorSumRD<T, N> rms,
                const AccessorSumRD<T, N> favg,
                const AccessorSumRD<T, N> frms,
                const AccessorRO<double, 3> rho,
                const AccessorRO<double, 3> dcsi_d,
                const AccessorRO<double, 3> deta_d,
                const AccessorRO<double, 3> dzet_d,
                const double deltaTime,
                const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Avg(f, avg, rms, p, pA, weight);
      const double fweight = AverageUtils<N>::getFavreWeight(dcsi_d, deta_d, dzet_d, rho, p, deltaTime);
      AverageUtils<N>::Avg(f, favg, frms, p, pA, fweight);
   }
};

template<direction dir, int N>
__global__
void Avg_kernel(const AccessorRO<Vec3, 3> f,
                const AccessorSumRD<Vec3, N> avg,
                const AccessorSumRD<Vec3, N> rms,
                const AccessorSumRD<Vec3, N> rey,
                const AccessorSumRD<Vec3, N> favg,
                const AccessorSumRD<Vec3, N> frms,
                const AccessorSumRD<Vec3, N> frey,
                const AccessorRO<double, 3> rho,
                const AccessorRO<double, 3> dcsi_d,
                const AccessorRO<double, 3> deta_d,
                const AccessorRO<double, 3> dzet_d,
                const double deltaTime,
                const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Avg(f, avg, rms, rey, p, pA, weight);
      const double fweight = AverageUtils<N>::getFavreWeight(dcsi_d, deta_d, dzet_d, rho, p, deltaTime);
      AverageUtils<N>::Avg(f, favg, frms, frey, p, pA, fweight);
   }
};

#define Avg_gpu(args...) Avg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>( \
                  args, \
                  acc_rho, \
                  acc_dcsi_d, acc_deta_d, acc_dzet_d, \
                  Integrator_deltaTime, r_Avg, r_Fluid, \
                  getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

template<direction dir, int N, typename T>
__global__
void ReynoldsAvg_kernel(const AccessorRO<T, 3> f,
                        const AccessorSumRD<T, N> avg,
                        const AccessorRO<double, 3> dcsi_d,
                        const AccessorRO<double, 3> deta_d,
                        const AccessorRO<double, 3> dzet_d,
                        const double deltaTime,
                        const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Avg(f, avg, p, pA, weight);
   }
};

template<direction dir, int N, typename T>
__global__
void ReynoldsAvg_kernel(const AccessorRO<T, 3> f,
                        const AccessorSumRD<T, N> avg,
                        const AccessorSumRD<T, N> rms,
                        const AccessorRO<double, 3> dcsi_d,
                        const AccessorRO<double, 3> deta_d,
                        const AccessorRO<double, 3> dzet_d,
                        const double deltaTime,
                        const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Avg(f, avg, rms, p, pA, weight);
   }
};

template<direction dir, int N>
__global__
void ReynoldsAvg_kernel(const AccessorRO<Vec3, 3> f,
                        const AccessorSumRD<Vec3, N> avg,
                        const AccessorSumRD<Vec3, N> rms,
                        const AccessorSumRD<Vec3, N> rey,
                        const AccessorRO<double, 3> dcsi_d,
                        const AccessorRO<double, 3> deta_d,
                        const AccessorRO<double, 3> dzet_d,
                        const double deltaTime,
                        const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Avg(f, avg, rms, rey, p, pA, weight);
   }
};

#define ReyAvg_gpu(args...) ReynoldsAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>( \
                  args, \
                  acc_dcsi_d, acc_deta_d, acc_dzet_d, \
                  Integrator_deltaTime, r_Avg, r_Fluid, \
                  getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

template<direction dir, int N>
__global__
void Cor_kernel(const AccessorRO<double, 3> s,
                const AccessorRO<  Vec3, 3> v,
                const AccessorSumRD<Vec3, N> cor,
                const AccessorSumRD<Vec3, N> fcor,
                const AccessorRO<double, 3> rho,
                const AccessorRO<double, 3> dcsi_d,
                const AccessorRO<double, 3> deta_d,
                const AccessorRO<double, 3> dzet_d,
                const double deltaTime,
                const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Cor(s, v, cor, p, pA, weight);
      const double fweight = AverageUtils<N>::getFavreWeight(dcsi_d, deta_d, dzet_d, rho, p, deltaTime);
      AverageUtils<N>::Cor(s, v, fcor, p, pA, fweight);
   }
};


template<direction dir, int N>
__global__
void Cor_kernel(const AccessorRO<VecNSp, 3> v1,
                const AccessorRO<  Vec3, 3> v2,
                const AccessorSumRD<VecNSp, N> cor0,
                const AccessorSumRD<VecNSp, N> cor1,
                const AccessorSumRD<VecNSp, N> cor2,
                const AccessorSumRD<VecNSp, N> fcor0,
                const AccessorSumRD<VecNSp, N> fcor1,
                const AccessorSumRD<VecNSp, N> fcor2,
                const AccessorRO<double, 3> rho,
                const AccessorRO<double, 3> dcsi_d,
                const AccessorRO<double, 3> deta_d,
                const AccessorRO<double, 3> dzet_d,
                const double deltaTime,
                const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::Cor(v1, v2, cor0, cor1, cor2, p, pA, weight);
      const double fweight = AverageUtils<N>::getFavreWeight(dcsi_d, deta_d, dzet_d, rho, p, deltaTime);
      AverageUtils<N>::Cor(v1, v2, fcor0, fcor1, fcor2, p, pA, weight);
   }
};

#define Cor_gpu(args...) Cor_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>( \
                  args, \
                  acc_rho, \
                  acc_dcsi_d, acc_deta_d, acc_dzet_d, \
                  Integrator_deltaTime, r_Avg, r_Fluid, \
                  getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

template<direction dir, int N>
__global__
void CpEntAvg_kernel(const AccessorRO<double, 3> temperature,
                     const AccessorRO<VecNSp, 3> MassFracs,
                     const AccessorRO<double, 3> rho,
                     const AccessorSumRD<double, N> cp_avg,
                     const AccessorSumRD<double, N> cp_favg,
                     const AccessorSumRD<double, N> Ent_avg,
                     const AccessorSumRD<double, N> Ent_favg,
                     const AccessorRO<double, 3> dcsi_d,
                     const AccessorRO<double, 3> deta_d,
                     const AccessorRO<double, 3> dzet_d,
                     const double deltaTime,
                     const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      const double fweight = AverageUtils<N>::getFavreWeight(dcsi_d, deta_d, dzet_d, rho, p, deltaTime);
      AverageUtils<N>::CpEntAvg(temperature, MassFracs, cp_avg, cp_favg, Ent_avg, Ent_favg, p, pA, mix, weight, fweight);
   }
};

template<direction dir, int N>
__global__
void ProdRatesAvg_kernel(const AccessorRO<double, 3> pressure,
                         const AccessorRO<double, 3> temperature,
                         const AccessorRO<VecNSp, 3> MassFracs,
                         const AccessorRO<double, 3> rho,
                         const AccessorSumRD<VecNSp, N> ProductionRates_avg,
                         const AccessorSumRD<VecNSp, N> ProductionRates_rms,
                         const AccessorSumRD<double, N> HeatReleaseRate_avg,
                         const AccessorSumRD<double, N> HeatReleaseRate_rms,
                         const AccessorRO<double, 3> dcsi_d,
                         const AccessorRO<double, 3> deta_d,
                         const AccessorRO<double, 3> dzet_d,
                         const double deltaTime,
                         const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::ProdRatesAvg(pressure, temperature, MassFracs, rho,
                                    ProductionRates_avg, ProductionRates_rms,
                                    HeatReleaseRate_avg, HeatReleaseRate_rms,
                                    p, pA, mix, weight);
   }
};

// Workaroud for Legion issue #879
template<int N>
struct HeatFluxAvg_kernelArgs {
   const AccessorRO<   int, 3> nType_x;
   const AccessorRO<   int, 3> nType_y;
   const AccessorRO<   int, 3> nType_z;
   const AccessorRO<double, 3> dcsi_d;
   const AccessorRO<double, 3> deta_d;
   const AccessorRO<double, 3> dzet_d;
   const AccessorRO<double, 3> temperature;
   const AccessorRO<VecNSp, 3> MolarFracs;
   const AccessorRO<VecNSp, 3> MassFracs;
   const AccessorRO<double, 3> rho;
   const AccessorRO<double, 3> lam;
   const AccessorRO<VecNSp, 3> Di;
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   const AccessorRO<VecNIo, 3> Ki;
   const AccessorRO<  Vec3, 3> eField;
#endif
   const AccessorSumRD<  Vec3, N> q_avg;
};

template<direction dir, int N>
__global__
#ifdef LEGION_BOUNDS_CHECKS
void HeatFluxAvg_kernel(const DeferredBuffer<HeatFluxAvg_kernelArgs<N>, 1> buffer,
                        const Rect<3> Fluid_bounds,
                        const double deltaTime,
                        const Rect<N> r_Avg,
                        const Rect<3> my_bounds,
                        const coord_t size_x,
                        const coord_t size_y,
                        const coord_t size_z)
#else
void HeatFluxAvg_kernel(const HeatFluxAvg_kernelArgs<N> a,
                        const Rect<3> Fluid_bounds,
                        const double deltaTime,
                        const Rect<N> r_Avg,
                        const Rect<3> my_bounds,
                        const coord_t size_x,
                        const coord_t size_y,
                        const coord_t size_z)
#endif
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef LEGION_BOUNDS_CHECKS
   const HeatFluxAvg_kernelArgs<N> a = buffer[0];
#endif

   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(a.dcsi_d, a.deta_d, a.dzet_d, p, deltaTime);
      AverageUtils<N>::HeatFluxAvg(a.nType_x, a.nType_y, a.nType_z,
                a.dcsi_d, a.deta_d, a.dzet_d,
                a.temperature, a.MolarFracs, a.MassFracs,
                a.rho, a.lam, a.Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                a.Ki, a.eField,
#endif
                a.q_avg,
                p, pA, Fluid_bounds, mix, weight);
   }
};

template<direction dir, int N>
__global__
void AvgKineticEnergyBudget_kernel(const AccessorRO<double, 3> pressure,
                                   const AccessorRO<  Vec3, 3> velocity,
                                   const AccessorRO<double, 3> rho,
                                   const AccessorSumRD<  Vec3, N> rhoUUv_avg,
                                   const AccessorSumRD<  Vec3, N> Up_avg,
                                   const AccessorRO<double, 3> dcsi_d,
                                   const AccessorRO<double, 3> deta_d,
                                   const AccessorRO<double, 3> dzet_d,
                                   const double deltaTime,
                                   const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::AvgKineticEnergyBudget(
                        pressure, velocity, rho,
                        rhoUUv_avg, Up_avg,
                        p, pA, weight);
   }
};

template<direction dir, int N>
__global__
void AvgKineticEnergyBudget_Tau_kernel(const AccessorRO<   int, 3> nType_x,
                                       const AccessorRO<   int, 3> nType_y,
                                       const AccessorRO<   int, 3> nType_z,
                                       const AccessorRO<double, 3> dcsi_d,
                                       const AccessorRO<double, 3> deta_d,
                                       const AccessorRO<double, 3> dzet_d,
                                       const AccessorRO<double, 3> pressure,
                                       const AccessorRO<  Vec3, 3> velocity,
                                       const AccessorRO<double, 3> rho,
                                       const AccessorRO<double, 3> mu,
                                       const AccessorSumRD<TauMat, N> tau_avg,
                                       const AccessorSumRD<  Vec3, N> utau_y_avg,
                                       const AccessorSumRD<  Vec3, N> tauGradU_avg,
                                       const AccessorSumRD<  Vec3, N> pGradU_avg,
                                       const Rect<3> Fluid_bounds,
                                       const double deltaTime,
                                       const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::AvgKineticEnergyBudget_Tau(
                        nType_x, nType_y, nType_z,
                        dcsi_d, deta_d, dzet_d,
                        pressure, velocity,
                        rho, mu,
                        tau_avg, utau_y_avg, tauGradU_avg, pGradU_avg,
                        p, pA, Fluid_bounds, mix, weight);
   }
};

template<direction dir, int N>
__global__
void PrEcAvg_kernel(const AccessorRO<double, 3> temperature,
                    const AccessorRO<VecNSp, 3> MassFracs,
                    const AccessorRO<  Vec3, 3> velocity,
                    const AccessorRO<double, 3> mu,
                    const AccessorRO<double, 3> lam,
                    const AccessorSumRD<double, N> Pr_avg,
                    const AccessorSumRD<double, N> Pr_rms,
                    const AccessorSumRD<double, N> Ec_avg,
                    const AccessorSumRD<double, N> Ec_rms,
                    const AccessorRO<double, 3> dcsi_d,
                    const AccessorRO<double, 3> deta_d,
                    const AccessorRO<double, 3> dzet_d,
                    const double deltaTime,
                    const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::PrEcAvg(
                  temperature, MassFracs, velocity, mu, lam,
                  Pr_avg, Pr_rms,
                  Ec_avg, Ec_rms,
                  p, pA, mix, weight);
   }
};

template<direction dir, int N>
__global__
void MaAvg_kernel(const AccessorRO<  Vec3, 3> velocity,
                  const AccessorRO<double, 3> SoS,
                  const AccessorSumRD<double, N> Ma_avg,
                  const AccessorRO<double, 3> dcsi_d,
                  const AccessorRO<double, 3> deta_d,
                  const AccessorRO<double, 3> dzet_d,
                  const double deltaTime,
                  const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::MaAvg(velocity, SoS, Ma_avg, p, pA, weight);
   }
};

template<direction dir, int N>
__global__
void ScAvg_kernel(const AccessorRO<double, 3> rho,
                  const AccessorRO<double, 3> mu,
                  const AccessorRO<VecNSp, 3> Di,
                  const AccessorSumRD<VecNSp, N> Sc_avg,
                  const AccessorRO<double, 3> dcsi_d,
                  const AccessorRO<double, 3> deta_d,
                  const AccessorRO<double, 3> dzet_d,
                  const double deltaTime,
                  const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::ScAvg(rho, mu, Di, Sc_avg, p, pA, weight);
   }
};

#ifdef ELECTRIC_FIELD
template<direction dir, int N>
__global__
void ElectricChargeAvg_kernel(const AccessorRO<VecNSp, 3> &MolarFracs,
                              const AccessorRO<double, 3> &rho,
                              const AccessorSumRD<double, N> &Crg_avg,
                              const AccessorRO<double, 3> dcsi_d,
                              const AccessorRO<double, 3> deta_d,
                              const AccessorRO<double, 3> dzet_d,
                              const double deltaTime,
                              const Rect<N> r_Avg,
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
      const Point<N> pA = getPAvg<dir, N>(p, r_Avg);
      const double weight = AverageUtils<N>::getWeight(dcsi_d, deta_d, dzet_d, p, deltaTime);
      AverageUtils<N>::ElectricChargeAvg(MolarFracs, rho, Crg_avg, p, pA, mix, weight);
   }
};
#endif

template<direction dir, int N>
__host__
void AddAveragesTask<dir, N>::gpu_base_impl(
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
   const AccessorRO<  Vec3, 3> acc_velocity            (regions[0], FID_velocity);

   // Accessors for cell geometry
   const AccessorRO<  Vec3, 3> acc_centerCoordinates   (regions[1], FID_centerCoordinates);

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

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho                 (regions[1], FID_rho);
   const AccessorRO<double, 3> acc_mu                  (regions[1], FID_mu);
   const AccessorRO<double, 3> acc_lam                 (regions[1], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di                  (regions[1], FID_Di);
   const AccessorRO<double, 3> acc_SoS                 (regions[1], FID_SoS);

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
   const AccessorSumRD<double, N> acc_avg_weight            (regions[iDouble], AVE_FID_weight,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<  Vec3, N> acc_avg_centerCoordinates (regions[iVec3  ], AVE_FID_centerCoordinates,   REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, N> acc_pressure_avg          (regions[iDouble], AVE_FID_pressure_avg,        LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_pressure_rms          (regions[iDouble], AVE_FID_pressure_rms,        LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_temperature_avg       (regions[iDouble], AVE_FID_temperature_avg,     LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_temperature_rms       (regions[iDouble], AVE_FID_temperature_rms,     LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, N> acc_MolarFracs_avg        (regions[iVecNSp], AVE_FID_MolarFracs_avg,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_MolarFracs_rms        (regions[iVecNSp], AVE_FID_MolarFracs_rms,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_MassFracs_avg         (regions[iVecNSp], AVE_FID_MassFracs_avg,       REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_MassFracs_rms         (regions[iVecNSp], AVE_FID_MassFracs_rms,       REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, N> acc_velocity_avg          (regions[iVec3  ], AVE_FID_velocity_avg,        REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_velocity_rms          (regions[iVec3  ], AVE_FID_velocity_rms,        REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_velocity_rey          (regions[iVec3  ], AVE_FID_velocity_rey,        REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, N> acc_pressure_favg         (regions[iDouble], AVE_FID_pressure_favg,       LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_pressure_frms         (regions[iDouble], AVE_FID_pressure_frms,       LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_temperature_favg      (regions[iDouble], AVE_FID_temperature_favg,    LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_temperature_frms      (regions[iDouble], AVE_FID_temperature_frms,    LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, N> acc_MolarFracs_favg       (regions[iVecNSp], AVE_FID_MolarFracs_favg,     REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_MolarFracs_frms       (regions[iVecNSp], AVE_FID_MolarFracs_frms,     REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_MassFracs_favg        (regions[iVecNSp], AVE_FID_MassFracs_favg,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_MassFracs_frms        (regions[iVecNSp], AVE_FID_MassFracs_frms,      REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<  Vec3, N> acc_velocity_favg         (regions[iVec3  ], AVE_FID_velocity_favg,       REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_velocity_frms         (regions[iVec3  ], AVE_FID_velocity_frms,       REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_velocity_frey         (regions[iVec3  ], AVE_FID_velocity_frey,       REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, N> acc_rho_avg               (regions[iDouble], AVE_FID_rho_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_rho_rms               (regions[iDouble], AVE_FID_rho_rms,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_mu_avg                (regions[iDouble], AVE_FID_mu_avg,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_lam_avg               (regions[iDouble], AVE_FID_lam_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, N> acc_Di_avg                (regions[iVecNSp], AVE_FID_Di_avg,              REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, N> acc_SoS_avg               (regions[iDouble], AVE_FID_SoS_avg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_cp_avg                (regions[iDouble], AVE_FID_cp_avg,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Ent_avg               (regions[iDouble], AVE_FID_Ent_avg,             LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<double, N> acc_mu_favg               (regions[iDouble], AVE_FID_mu_favg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_lam_favg              (regions[iDouble], AVE_FID_lam_favg,            LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, N> acc_Di_favg               (regions[iVecNSp], AVE_FID_Di_favg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, N> acc_SoS_favg              (regions[iDouble], AVE_FID_SoS_favg,            LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_cp_favg               (regions[iDouble], AVE_FID_cp_favg,             LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Ent_favg              (regions[iDouble], AVE_FID_Ent_favg,            LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<  Vec3, N> acc_q_avg                 (regions[iVec3  ], AVE_FID_q,                   REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<VecNSp, N> acc_ProductionRates_avg   (regions[iVecNSp], AVE_FID_ProductionRates_avg, REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_ProductionRates_rms   (regions[iVecNSp], AVE_FID_ProductionRates_rms, REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<double, N> acc_HeatReleaseRate_avg   (regions[iDouble], AVE_FID_HeatReleaseRate_avg, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_HeatReleaseRate_rms   (regions[iDouble], AVE_FID_HeatReleaseRate_rms, LEGION_REDOP_SUM_FLOAT64);

   const AccessorSumRD<  Vec3, N> acc_rhoUUv                (regions[iVec3  ], AVE_FID_rhoUUv,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_Up                    (regions[iVec3  ], AVE_FID_Up,                  REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<TauMat, N> acc_tau                   (regions[iVec6  ], AVE_FID_tau,                 REGENT_REDOP_SUM_VEC6);
   const AccessorSumRD<  Vec3, N> acc_utau_y                (regions[iVec3  ], AVE_FID_utau_y,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_tauGradU              (regions[iVec3  ], AVE_FID_tauGradU,            REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_pGradU                (regions[iVec3  ], AVE_FID_pGradU,              REGENT_REDOP_SUM_VEC3);

   const AccessorSumRD<double, N> acc_Pr_avg                (regions[iDouble], AVE_FID_Pr,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Pr_rms                (regions[iDouble], AVE_FID_Pr_rms,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Ec_avg                (regions[iDouble], AVE_FID_Ec,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Ec_rms                (regions[iDouble], AVE_FID_Ec_rms,              LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Ma_avg                (regions[iDouble], AVE_FID_Ma,                  LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<VecNSp, N> acc_Sc_avg                (regions[iVecNSp], AVE_FID_Sc,                  REGENT_REDOP_SUM_VECNSP);

   const AccessorSumRD<  Vec3, N> acc_uT_avg                (regions[iVec3  ], AVE_FID_uT_avg,              REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<  Vec3, N> acc_uT_favg               (regions[iVec3  ], AVE_FID_uT_favg,             REGENT_REDOP_SUM_VEC3);
   const AccessorSumRD<VecNSp, N> acc_uYi_avg               (regions[iVecNSp], AVE_FID_uYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_vYi_avg               (regions[iVecNSp], AVE_FID_vYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_wYi_avg               (regions[iVecNSp], AVE_FID_wYi_avg,             REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_uYi_favg              (regions[iVecNSp], AVE_FID_uYi_favg,            REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_vYi_favg              (regions[iVecNSp], AVE_FID_vYi_favg,            REGENT_REDOP_SUM_VECNSP);
   const AccessorSumRD<VecNSp, N> acc_wYi_favg              (regions[iVecNSp], AVE_FID_wYi_favg,            REGENT_REDOP_SUM_VECNSP);

#ifdef ELECTRIC_FIELD
   const AccessorSumRD<double, N> acc_ePot_avg              (regions[iDouble], AVE_FID_electricPotential_avg, LEGION_REDOP_SUM_FLOAT64);
   const AccessorSumRD<double, N> acc_Crg_avg               (regions[iDouble], AVE_FID_chargeDensity_avg,     LEGION_REDOP_SUM_FLOAT64);
#endif

   // Extract execution domains
   Rect<3> r_Fluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Extract average domain
   Rect<N> r_Avg = runtime->get_index_space_domain(ctx, regions[iDouble].get_logical_region().get_index_space());

   // Wait for the integrator deltaTime
   const double Integrator_deltaTime = futures[0].get_result<double>();

   // Set thread grid
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<dir>(threads_per_block, r_Fluid);
   const dim3 num_blocks_3d = dim3((getSize<Xdir>(r_Fluid) + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (getSize<Ydir>(r_Fluid) + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (getSize<Zdir>(r_Fluid) + (TPB_3d.z - 1)) / TPB_3d.z);

   // assign each kernel to a separate stream using a round-robin logic
   streamsRR<10> streamsList;

   // Position and integrated weight
   PositionAndWeight_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_centerCoordinates, acc_avg_weight, acc_avg_centerCoordinates,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Reynolds and Favre average of primitive variables
   Avg_gpu(acc_pressure,    acc_pressure_avg,    acc_pressure_rms,    acc_pressure_favg,    acc_pressure_frms);
   Avg_gpu(acc_temperature, acc_temperature_avg, acc_temperature_rms, acc_temperature_favg, acc_temperature_frms);
   Avg_gpu(acc_MolarFracs,  acc_MolarFracs_avg,  acc_MolarFracs_rms,  acc_MolarFracs_favg,  acc_MolarFracs_frms);
   Avg_gpu(acc_MassFracs,   acc_MassFracs_avg,   acc_MassFracs_rms,   acc_MassFracs_favg,   acc_MassFracs_frms);
   Avg_gpu(acc_velocity, acc_velocity_avg,  acc_velocity_rms,  acc_velocity_rey,
                         acc_velocity_favg, acc_velocity_frms, acc_velocity_frey);

   // Reynolds and Favre average of properties
   ReyAvg_gpu(acc_rho, acc_rho_avg, acc_rho_rms);
   Avg_gpu(acc_mu,  acc_mu_avg,  acc_mu_favg);
   Avg_gpu(acc_lam, acc_lam_avg, acc_lam_favg);
   Avg_gpu(acc_Di,  acc_Di_avg,  acc_Di_favg);
   Avg_gpu(acc_SoS, acc_SoS_avg, acc_SoS_favg);

   // Other properties averages
   CpEntAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_temperature, acc_MassFracs, acc_rho,
                     acc_cp_avg, acc_cp_favg,
                     acc_Ent_avg, acc_Ent_favg,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Average production rates
   ProdRatesAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_pressure, acc_temperature, acc_MassFracs, acc_rho,
                     acc_ProductionRates_avg, acc_ProductionRates_rms,
                     acc_HeatReleaseRate_avg, acc_HeatReleaseRate_rms,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Average heat flux
   {
   const HeatFluxAvg_kernelArgs<N> kArgs = {
      .nType_x     = acc_nType_x,
      .nType_y     = acc_nType_y,
      .nType_z     = acc_nType_z,
      .dcsi_d      = acc_dcsi_d,
      .deta_d      = acc_deta_d,
      .dzet_d      = acc_dzet_d,
      .temperature = acc_temperature,
      .MolarFracs  = acc_MolarFracs,
      .MassFracs   = acc_MassFracs,
      .rho         = acc_rho,
      .lam         = acc_lam,
      .Di          = acc_Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
      .Ki          = acc_Ki,
      .eField      = acc_eField,
#endif
      .q_avg       = acc_q_avg
   };

#ifdef LEGION_BOUNDS_CHECKS
   DeferredBuffer<HeatFluxAvg_kernelArgs<N>, 1>
      buffer(Rect<1>(Point<1>(0), Point<1>(1)), Memory::Z_COPY_MEM, &kArgs);
   HeatFluxAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     buffer,
                     args.Fluid_bounds,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));
#else
   HeatFluxAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     kArgs,
                     args.Fluid_bounds,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));
#endif
   }

   // Collect averages of kinetic energy budget terms
   AvgKineticEnergyBudget_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_pressure, acc_velocity, acc_rho,
                     acc_rhoUUv, acc_Up,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   AvgKineticEnergyBudget_Tau_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_nType_x, acc_nType_y, acc_nType_z,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     acc_pressure, acc_velocity,
                     acc_rho, acc_mu,
                     acc_tau, acc_utau_y, acc_tauGradU, acc_pGradU,
                     args.Fluid_bounds,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of dimensionless numbers
   PrEcAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_temperature, acc_MassFracs, acc_velocity, acc_mu, acc_lam,
                     acc_Pr_avg, acc_Pr_rms,
                     acc_Ec_avg, acc_Ec_rms,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   MaAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_velocity, acc_SoS,
                     acc_Ma_avg,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   ScAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_rho, acc_mu, acc_Di,
                     acc_Sc_avg,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));

   // Collect averages of correlations
   Cor_gpu(acc_temperature, acc_velocity, acc_uT_avg, acc_uT_favg);
   Cor_gpu(acc_MassFracs,   acc_velocity, acc_uYi_avg,  acc_vYi_avg,  acc_wYi_avg,
                                          acc_uYi_favg, acc_vYi_favg, acc_wYi_favg);

#ifdef ELECTRIC_FIELD
   // Collect averages of electric quantities
   ReyAvg_gpu(acc_ePot, acc_ePot_avg);
   ElectricChargeAvg_kernel<dir, N><<<num_blocks_3d, TPB_3d, 0, ++streamsList>>>(
                     acc_MolarFracs, acc_rho,
                     acc_Crg_avg,
                     acc_dcsi_d, acc_deta_d, acc_dzet_d,
                     Integrator_deltaTime, r_Avg, r_Fluid,
                     getSize<Xdir>(r_Fluid), getSize<Ydir>(r_Fluid), getSize<Zdir>(r_Fluid));
#endif
}

template void AddAveragesTask<Xdir, 2>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void AddAveragesTask<Ydir, 2>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void AddAveragesTask<Zdir, 2>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void AddAveragesTask<Xdir, 3>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void AddAveragesTask<Ydir, 3>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

template void AddAveragesTask<Zdir, 3>::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime);

