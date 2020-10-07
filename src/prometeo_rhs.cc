// Copyright (c) "2019, by Stanford University
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
#include <omp.h>

// Specielize UpdateUsingHybridEulerFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingHybridEulerFluxTask<Xdir>::TASK_NAME = "UpdateUsingHybridEulerFluxX";
template<>
/*static*/ const int             UpdateUsingHybridEulerFluxTask<Xdir>::TASK_ID = TID_UpdateUsingHybridEulerFluxX;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Xdir>::iN = 0;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Xdir>::iT1 = 1;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Xdir>::iT2 = 2;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Xdir>::FID_m_e = FID_dcsi_e;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Xdir>::FID_shockSensor = FID_shockSensorX;

template<>
void UpdateUsingHybridEulerFluxTask<Xdir>::cpu_base_impl(
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
   const AccessorRO<double, 3> acc_m(regions[4], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[5], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Allocate a buffer for the summations of the KG scheme of size (3 * 3 * (nEq+1)) for each thread
   // TODO: can reduce it to (6*(nEq+1))
#ifdef REALM_USE_OPENMP
   double *KGSum = new double[9*(nEq+1)*omp_get_max_threads()];
#else
   double *KGSum = new double[9*(nEq+1)];
#endif

   // update RHS using Euler fluxes
   const coord_t size = getSize<Xdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++) {

         double FluxM[nEq];
         double FluxP[nEq];

#ifdef REALM_USE_OPENMP
         double *myKGSum = &KGSum[9*(nEq+1)*omp_get_thread_num()];
#else
         double *myKGSum = &KGSum[0];
#endif

         // Reconstruct the Euler flux at i-1/2 of the first point
         {
            const Point<3> p   = warpPeriodic<Xdir, Minus>(Fluid_bounds, Point<3>{r_ModCells.lo.x,j,k}, size, -1);
            const Point<3> pM2 = warpPeriodic<Xdir, Minus>(Fluid_bounds, p, size, offM2(acc_nType[p]));
            const Point<3> pM1 = warpPeriodic<Xdir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<Xdir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));

            // Compute KG summations
            ComputeKGSums(&myKGSum[0],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          p, acc_nType[p], size, Fluid_bounds);
            ComputeKGSums(&myKGSum[3*(nEq+1)],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          pM1, acc_nType[p], size, Fluid_bounds);
            ComputeKGSums(&myKGSum[6*(nEq+1)],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          pM2, acc_nType[p], size, Fluid_bounds);

            if (acc_shockSensor[pM1] and
                acc_shockSensor[p  ] and
                acc_shockSensor[pP1])
               // KG reconstruction
               KGFluxReconstruction(FluxM, myKGSum,
                                    acc_Conserved, acc_velocity,  acc_pressure,
                                    p, acc_nType[p], size, Fluid_bounds);
            else
               // TENO reconstruction
               TENOFluxReconstruction(FluxM,
                                      acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                      acc_pressure, acc_MassFracs, acc_temperature,
                                      p, acc_nType[p], args.mix, size, Fluid_bounds);
         }

         for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            const Point<3> pM1 = warpPeriodic<Xdir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<Xdir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));

            // Shift and update KG summations
            for (int l=0; l<3*(nEq+1); l++) myKGSum[6*(nEq+1) + l] = myKGSum[3*(nEq+1) + l];
            for (int l=0; l<3*(nEq+1); l++) myKGSum[3*(nEq+1) + l] = myKGSum[            l];
            ComputeKGSums(&myKGSum[0],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          p, acc_nType[p], size, Fluid_bounds);

            if (acc_shockSensor[pM1] and
                acc_shockSensor[p  ] and
                acc_shockSensor[pP1])
               // KG reconstruction
               KGFluxReconstruction(FluxP, myKGSum,
                                    acc_Conserved, acc_velocity,  acc_pressure,
                                    p, acc_nType[p], size, Fluid_bounds);
            else
               // TENO reconstruction
               TENOFluxReconstruction(FluxP,
                                      acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                      acc_pressure, acc_MassFracs, acc_temperature,
                                      p, acc_nType[p], args.mix, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m[p]*(FluxP[l] - FluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               FluxM[l] = FluxP[l];
         }
      }
   // Cleanup
   delete[] KGSum;
}

// Specielize UpdateUsingHybridEulerFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingHybridEulerFluxTask<Ydir>::TASK_NAME = "UpdateUsingHybridEulerFluxY";
template<>
/*static*/ const int             UpdateUsingHybridEulerFluxTask<Ydir>::TASK_ID = TID_UpdateUsingHybridEulerFluxY;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Ydir>::iN = 1;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Ydir>::iT1 = 0;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Ydir>::iT2 = 2;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Ydir>::FID_m_e = FID_deta_e;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Ydir>::FID_shockSensor = FID_shockSensorY;

template<>
void UpdateUsingHybridEulerFluxTask<Ydir>::cpu_base_impl(
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
   const AccessorRO<double, 3> acc_m(regions[4], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[5], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Allocate a buffer for the summations of the KG scheme of size (3 * 3 * (nEq+1)) for each thread
   // TODO: can reduce it to (6*(nEq+1))
#ifdef REALM_USE_OPENMP
   double *KGSum = new double[9*(nEq+1)*omp_get_max_threads()];
#else
   double *KGSum = new double[9*(nEq+1)];
#endif

   // update RHS using Euler fluxes
   const coord_t size = getSize<Ydir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {

         double FluxM[nEq];
         double FluxP[nEq];

#ifdef REALM_USE_OPENMP
         double *myKGSum = &KGSum[9*(nEq+1)*omp_get_thread_num()];
#else
         double *myKGSum = &KGSum[0];
#endif

         // Reconstruct the Euler flux at j-1/2 of the first point
         {
            const Point<3> p   = warpPeriodic<Ydir, Minus>(Fluid_bounds, Point<3>{i,r_ModCells.lo.y,k}, size, -1);
            const Point<3> pM2 = warpPeriodic<Ydir, Minus>(Fluid_bounds, p, size, offM2(acc_nType[p]));
            const Point<3> pM1 = warpPeriodic<Ydir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<Ydir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));

            // Compute KG summations
            ComputeKGSums(&myKGSum[0],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          p, acc_nType[p], size, Fluid_bounds);
            ComputeKGSums(&myKGSum[3*(nEq+1)],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          pM1, acc_nType[p], size, Fluid_bounds);
            ComputeKGSums(&myKGSum[6*(nEq+1)],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          pM2, acc_nType[p], size, Fluid_bounds);

            if (acc_shockSensor[pM1] and
                acc_shockSensor[p  ] and
                acc_shockSensor[pP1])
               // KG reconstruction
               KGFluxReconstruction(FluxM, myKGSum,
                                    acc_Conserved, acc_velocity,  acc_pressure,
                                    p, acc_nType[p], size, Fluid_bounds);
            else
               // TENO reconstruction
               TENOFluxReconstruction(FluxM,
                                      acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                      acc_pressure, acc_MassFracs, acc_temperature,
                                      p, acc_nType[p], args.mix, size, Fluid_bounds);
         }

         for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++) {
            const Point<3> p = Point<3>{i,j,k};
            const Point<3> pM1 = warpPeriodic<Ydir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<Ydir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));

            // Shift and update KG summations
            for (int l=0; l<3*(nEq+1); l++) myKGSum[6*(nEq+1) + l] = myKGSum[3*(nEq+1) + l];
            for (int l=0; l<3*(nEq+1); l++) myKGSum[3*(nEq+1) + l] = myKGSum[            l];
            ComputeKGSums(&myKGSum[0],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          p, acc_nType[p], size, Fluid_bounds);

            if (acc_shockSensor[pM1] and
                acc_shockSensor[p  ] and
                acc_shockSensor[pP1])
               // KG reconstruction
               KGFluxReconstruction(FluxP, myKGSum,
                                    acc_Conserved, acc_velocity,  acc_pressure,
                                    p, acc_nType[p], size, Fluid_bounds);
            else
               // TENO reconstruction
               TENOFluxReconstruction(FluxP,
                                      acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                      acc_pressure, acc_MassFracs, acc_temperature,
                                      p, acc_nType[p], args.mix, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m[p]*(FluxP[l] - FluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               FluxM[l] = FluxP[l];
         }
      }
   // Cleanup
   delete[] KGSum;
}

// Specielize UpdateUsingHybridEulerFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingHybridEulerFluxTask<Zdir>::TASK_NAME = "UpdateUsingHybridEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingHybridEulerFluxTask<Zdir>::TASK_ID = TID_UpdateUsingHybridEulerFluxZ;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Zdir>::iN = 2;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Zdir>::iT1 = 0;
//template<>
///*static*/ const int             UpdateUsingHybridEulerFluxTask<Zdir>::iT2 = 1;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Zdir>::FID_m_e = FID_dzet_e;
template<>
/*static*/ const FieldID         UpdateUsingHybridEulerFluxTask<Zdir>::FID_shockSensor = FID_shockSensorZ;

template<>
void UpdateUsingHybridEulerFluxTask<Zdir>::cpu_base_impl(
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
   const AccessorRO<double, 3> acc_m(regions[4], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[5], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Allocate a buffer for the summations of the KG scheme of size (3 * 3 * (nEq+1)) for each thread
   // TODO: can reduce it to (6*(nEq+1))
#ifdef REALM_USE_OPENMP
   double *KGSum = new double[9*(nEq+1)*omp_get_max_threads()];
#else
   double *KGSum = new double[9*(nEq+1)];
#endif

   // update RHS using Euler fluxes
   const coord_t size = getSize<Zdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++)
      for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {

         double FluxM[nEq];
         double FluxP[nEq];

#ifdef REALM_USE_OPENMP
         double *myKGSum = &KGSum[9*(nEq+1)*omp_get_thread_num()];
#else
         double *myKGSum = &KGSum[0];
#endif

         // Reconstruct the Euler flux at k-1/2 of the first point
         {
            const Point<3> p   = warpPeriodic<Zdir, Minus>(Fluid_bounds, Point<3>{i,j,r_ModCells.lo.z}, size, -1);
            const Point<3> pM2 = warpPeriodic<Zdir, Minus>(Fluid_bounds, p, size, offM2(acc_nType[p]));
            const Point<3> pM1 = warpPeriodic<Zdir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<Zdir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));

            // Compute KG summations
            ComputeKGSums(&myKGSum[0],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          p, acc_nType[p], size, Fluid_bounds);
            ComputeKGSums(&myKGSum[3*(nEq+1)],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          pM1, acc_nType[p], size, Fluid_bounds);
            ComputeKGSums(&myKGSum[6*(nEq+1)],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          pM2, acc_nType[p], size, Fluid_bounds);

            if (acc_shockSensor[pM1] and
                acc_shockSensor[p  ] and
                acc_shockSensor[pP1])
               // KG reconstruction
               KGFluxReconstruction(FluxM, myKGSum,
                                    acc_Conserved, acc_velocity,  acc_pressure,
                                    p, acc_nType[p], size, Fluid_bounds);
            else
               // TENO reconstruction
               TENOFluxReconstruction(FluxM,
                                      acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                      acc_pressure, acc_MassFracs, acc_temperature,
                                      p, acc_nType[p], args.mix, size, Fluid_bounds);
         }

         for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++) {
            const Point<3> p = Point<3>{i,j,k};
            const Point<3> pM1 = warpPeriodic<Zdir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<Zdir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));

            // Shift and update KG summations
            for (int l=0; l<3*(nEq+1); l++) myKGSum[6*(nEq+1) + l] = myKGSum[3*(nEq+1) + l];
            for (int l=0; l<3*(nEq+1); l++) myKGSum[3*(nEq+1) + l] = myKGSum[            l];
            ComputeKGSums(&myKGSum[0],
                          acc_Conserved, acc_rho, acc_MassFracs,
                          acc_velocity,  acc_pressure,
                          p, acc_nType[p], size, Fluid_bounds);

            if (acc_shockSensor[pM1] and
                acc_shockSensor[p  ] and
                acc_shockSensor[pP1])
               // KG reconstruction
               KGFluxReconstruction(FluxP, myKGSum,
                                    acc_Conserved, acc_velocity,  acc_pressure,
                                    p, acc_nType[p], size, Fluid_bounds);
            else
               // TENO reconstruction
               TENOFluxReconstruction(FluxP,
                                      acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                      acc_pressure, acc_MassFracs, acc_temperature,
                                      p, acc_nType[p], args.mix, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m[p]*(FluxP[l] - FluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               FluxM[l] = FluxP[l];
         }
      }
   // Cleanup
   delete[] KGSum;
}

// Specielize UpdateUsingTENOAEulerFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingTENOAEulerFluxTask<Xdir>::TASK_NAME = "UpdateUsingTENOAEulerFluxX";
template<>
/*static*/ const int             UpdateUsingTENOAEulerFluxTask<Xdir>::TASK_ID = TID_UpdateUsingTENOAEulerFluxX;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Xdir>::iN = 0;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Xdir>::iT1 = 1;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Xdir>::iT2 = 2;
template<>
/*static*/ const FieldID         UpdateUsingTENOAEulerFluxTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingTENOAEulerFluxTask<Xdir>::FID_m_e = FID_dcsi_e;

template<>
void UpdateUsingTENOAEulerFluxTask<Xdir>::cpu_base_impl(
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
   const AccessorRO<VecNSp, 3> acc_MassFracs(  regions[1], FID_MassFracs);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m(regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Euler fluxes
   const coord_t size = getSize<Xdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++) {

         double FluxM[nEq];
         double FluxP[nEq];

         // Reconstruct the Euler flux at i-1/2 of the first point
         {
            const Point<3> p   = warpPeriodic<Xdir, Minus>(Fluid_bounds, Point<3>{r_ModCells.lo.x,j,k}, size, -1);
            // TENOA reconstruction
            TENOAFluxReconstruction(FluxM,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature,
                                    p, acc_nType[p], args.mix, size, Fluid_bounds);
         }

         for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            // TENOA reconstruction
            TENOAFluxReconstruction(FluxP,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature,
                                    p, acc_nType[p], args.mix, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m[p]*(FluxP[l] - FluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               FluxM[l] = FluxP[l];
         }
      }
}

// Specielize UpdateUsingTENOAEulerFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingTENOAEulerFluxTask<Ydir>::TASK_NAME = "UpdateUsingTENOAEulerFluxY";
template<>
/*static*/ const int             UpdateUsingTENOAEulerFluxTask<Ydir>::TASK_ID = TID_UpdateUsingTENOAEulerFluxY;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Ydir>::iN = 1;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Ydir>::iT1 = 0;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Ydir>::iT2 = 2;
template<>
/*static*/ const FieldID         UpdateUsingTENOAEulerFluxTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingTENOAEulerFluxTask<Ydir>::FID_m_e = FID_deta_e;

template<>
void UpdateUsingTENOAEulerFluxTask<Ydir>::cpu_base_impl(
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
   const AccessorRO<VecNSp, 3> acc_MassFracs(  regions[1], FID_MassFracs);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m(regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Euler fluxes
   const coord_t size = getSize<Ydir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {

         double FluxM[nEq];
         double FluxP[nEq];

         // Reconstruct the Euler flux at j-1/2 of the first point
         {
            const Point<3> p   = warpPeriodic<Ydir, Minus>(Fluid_bounds, Point<3>{i,r_ModCells.lo.y,k}, size, -1);
            // TENOA reconstruction
            TENOAFluxReconstruction(FluxM,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature,
                                    p, acc_nType[p], args.mix, size, Fluid_bounds);
         }

         for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++) {
            const Point<3> p = Point<3>{i,j,k};
            // TENOA reconstruction
            TENOAFluxReconstruction(FluxP,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature,
                                    p, acc_nType[p], args.mix, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m[p]*(FluxP[l] - FluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               FluxM[l] = FluxP[l];
         }
      }
}

// Specielize UpdateUsingTENOAEulerFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingTENOAEulerFluxTask<Zdir>::TASK_NAME = "UpdateUsingTENOAEulerFluxZ";
template<>
/*static*/ const int             UpdateUsingTENOAEulerFluxTask<Zdir>::TASK_ID = TID_UpdateUsingTENOAEulerFluxZ;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Zdir>::iN = 2;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Zdir>::iT1 = 0;
//template<>
///*static*/ const int             UpdateUsingTENOAEulerFluxTask<Zdir>::iT2 = 1;
template<>
/*static*/ const FieldID         UpdateUsingTENOAEulerFluxTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingTENOAEulerFluxTask<Zdir>::FID_m_e = FID_dzet_e;

template<>
void UpdateUsingTENOAEulerFluxTask<Zdir>::cpu_base_impl(
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
   const AccessorRO<VecNSp, 3> acc_MassFracs(  regions[1], FID_MassFracs);

   // Accessors for node types
   const AccessorRO<   int, 3> acc_nType(regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m(regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Euler fluxes
   const coord_t size = getSize<Zdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++)
      for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {

         double FluxM[nEq];
         double FluxP[nEq];

         // Reconstruct the Euler flux at k-1/2 of the first point
         {
            const Point<3> p   = warpPeriodic<Zdir, Minus>(Fluid_bounds, Point<3>{i,j,r_ModCells.lo.z}, size, -1);
            // TENOA reconstruction
            TENOAFluxReconstruction(FluxM,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature,
                                    p, acc_nType[p], args.mix, size, Fluid_bounds);
         }

         for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++) {
            const Point<3> p = Point<3>{i,j,k};
            // TENOA reconstruction
            TENOAFluxReconstruction(FluxP,
                                    acc_Conserved, acc_SoS, acc_rho, acc_velocity,
                                    acc_pressure, acc_MassFracs, acc_temperature,
                                    p, acc_nType[p], args.mix, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m[p]*(FluxP[l] - FluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               FluxM[l] = FluxP[l];
         }
      }
}

// Specielize UpdateUsingDiffusionFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingDiffusionFluxTask<Xdir>::TASK_NAME = "UpdateUsingDiffusionFluxX";
template<>
/*static*/ const int             UpdateUsingDiffusionFluxTask<Xdir>::TASK_ID = TID_UpdateUsingDiffusionFluxX;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_vGrad1 = FID_velocityGradientY;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_vGrad2 = FID_velocityGradientZ;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_m_s = FID_dcsi_s;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Xdir>::FID_m_d = FID_dcsi_d;

template<>
void UpdateUsingDiffusionFluxTask<Xdir>::cpu_base_impl(
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
   const AccessorRO<  Vec3, 3> acc_vGradY     (regions[0], FID_velocityGradientY);
   const AccessorRO<  Vec3, 3> acc_vGradZ     (regions[0], FID_velocityGradientZ);

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

   // update RHS using Euler and Diffision fluxes
   const coord_t size = getSize<Xdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++) {
         double DiffFluxM[nEq];
         double DiffFluxP[nEq];

         // Compute flux of first minus inter-cell location
         {
            const Point<3> p = Point<3>{(r_ModCells.lo.x-1 - Fluid_bounds.lo.x + size) % size + Fluid_bounds.lo.x, j, k};
            GetDiffusionFlux(DiffFluxM, acc_nType[p], acc_m_s[p], args.mix,
                             acc_rho, acc_mu, acc_lam, acc_Di,
                             acc_temperature, acc_velocity, acc_MolarFracs,
                             acc_Conserved, acc_vGradY, acc_vGradZ,
                             p, size, Fluid_bounds);
         }

         // Now loop along the x line
         for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            GetDiffusionFlux(DiffFluxP, acc_nType[p], acc_m_s[p], args.mix,
                             acc_rho, acc_mu, acc_lam, acc_Di,
                             acc_temperature, acc_velocity, acc_MolarFracs,
                             acc_Conserved, acc_vGradY, acc_vGradZ,
                             p, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m_d[p]*(DiffFluxP[l] - DiffFluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               DiffFluxM[l] = DiffFluxP[l];

         }
      }
}

// Specielize UpdateUsingDiffusionFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingDiffusionFluxTask<Ydir>::TASK_NAME = "UpdateUsingDiffusionFluxY";
template<>
/*static*/ const int             UpdateUsingDiffusionFluxTask<Ydir>::TASK_ID = TID_UpdateUsingDiffusionFluxY;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_vGrad1 = FID_velocityGradientX;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_vGrad2 = FID_velocityGradientZ;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_m_s = FID_deta_s;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Ydir>::FID_m_d = FID_deta_d;

template<>
void UpdateUsingDiffusionFluxTask<Ydir>::cpu_base_impl(
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
   const AccessorRO<  Vec3, 3> acc_vGradX     (regions[0], FID_velocityGradientX);
   const AccessorRO<  Vec3, 3> acc_vGradZ     (regions[0], FID_velocityGradientZ);

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

   // update RHS using Euler and Diffision fluxes
   const coord_t size = getSize<Ydir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
         double DiffFluxM[nEq];
         double DiffFluxP[nEq];

         // Compute flux of first minus inter-cell location
         {
            const Point<3> p = Point<3>{i,(r_ModCells.lo.y-1 - Fluid_bounds.lo.y + size) % size + Fluid_bounds.lo.y, k};
            GetDiffusionFlux(DiffFluxM, acc_nType[p], acc_m_s[p], args.mix,
                             acc_rho, acc_mu, acc_lam, acc_Di,
                             acc_temperature, acc_velocity, acc_MolarFracs,
                             acc_Conserved, acc_vGradX, acc_vGradZ,
                             p, size, Fluid_bounds);
         }

         // Now loop along the y line
         for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++) {
            const Point<3> p = Point<3>{i,j,k};
            GetDiffusionFlux(DiffFluxP, acc_nType[p], acc_m_s[p], args.mix,
                             acc_rho, acc_mu, acc_lam, acc_Di,
                             acc_temperature, acc_velocity, acc_MolarFracs,
                             acc_Conserved, acc_vGradX, acc_vGradZ,
                             p, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m_d[p]*(DiffFluxP[l] - DiffFluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               DiffFluxM[l] = DiffFluxP[l];

         }
      }
}

// Specielize UpdateUsingDiffusionFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingDiffusionFluxTask<Zdir>::TASK_NAME = "UpdateUsingDiffusionFluxZ";
template<>
/*static*/ const int             UpdateUsingDiffusionFluxTask<Zdir>::TASK_ID = TID_UpdateUsingDiffusionFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_vGrad1 = FID_velocityGradientX;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_vGrad2 = FID_velocityGradientY;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_m_s = FID_dzet_s;
template<>
/*static*/ const FieldID         UpdateUsingDiffusionFluxTask<Zdir>::FID_m_d = FID_dzet_d;

template<>
void UpdateUsingDiffusionFluxTask<Zdir>::cpu_base_impl(
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
   const AccessorRO<  Vec3, 3> acc_vGradX     (regions[0], FID_velocityGradientX);
   const AccessorRO<  Vec3, 3> acc_vGradY     (regions[0], FID_velocityGradientY);

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

   // update RHS using Euler and Diffision fluxes
   const coord_t size = getSize<Zdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++)
      for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
         double DiffFluxM[nEq];
         double DiffFluxP[nEq];

         // Compute flux of first minus inter-cell location
         {
            const Point<3> p = Point<3>{i,j,(r_ModCells.lo.z-1 - Fluid_bounds.lo.z + size) % size + Fluid_bounds.lo.z};
            GetDiffusionFlux(DiffFluxM, acc_nType[p], acc_m_s[p], args.mix,
                             acc_rho, acc_mu, acc_lam, acc_Di,
                             acc_temperature, acc_velocity, acc_MolarFracs,
                             acc_Conserved, acc_vGradX, acc_vGradY,
                             p, size, Fluid_bounds);
         }

         // Now loop along the x line
         for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++) {
            const Point<3> p = Point<3>{i,j,k};
            GetDiffusionFlux(DiffFluxP, acc_nType[p], acc_m_s[p], args.mix,
                             acc_rho, acc_mu, acc_lam, acc_Di,
                             acc_temperature, acc_velocity, acc_MolarFracs,
                             acc_Conserved, acc_vGradX, acc_vGradY,
                             p, size, Fluid_bounds);

            // Update time derivative
            for (int l=0; l<nEq; l++)
               acc_Conserved_t[p][l] += acc_m_d[p]*(DiffFluxP[l] - DiffFluxM[l]);

            // Store plus flux for next point
            for (int l=0; l<nEq; l++)
               DiffFluxM[l] = DiffFluxP[l];

         }
      }
}

// Specielize UpdateUsingFluxNSCBCInflowMinusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowXNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowXNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::FID_m_d = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::FID_vGrad = FID_velocityGradientX;


template<>
void UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>::cpu_base_impl(
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

   const int i = r_BC.lo.x;
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++) {
         const Point<3> p = Point<3>(i,j,k);
         addLODIfluxes(acc_Conserved_t[p].v,
                       acc_MassFracs, acc_pressure,
                       acc_SoS[p], acc_rho[p], acc_temperature[p],
                       acc_velocity[p].v, acc_vGrad[p].v, acc_dudt[p].v, acc_dTdt[p],
                       p, acc_nType[p], acc_m_d[p], args.mix);
      }
};

// Specielize UpdateUsingFluxNSCBCInflowPlusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCInflowYPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCInflowYPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::FID_m_d = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::FID_vGrad = FID_velocityGradientY;


template<>
void UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>::cpu_base_impl(
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

   const int j = r_BC.lo.y;
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
         const Point<3> p = Point<3>(i,j,k);
         addLODIfluxes(acc_Conserved_t[p].v,
                       acc_MassFracs, acc_pressure,
                       acc_SoS[p], acc_rho[p], acc_temperature[p],
                       acc_velocity[p].v, acc_vGrad[p].v, acc_dudt[p].v, acc_dTdt[p],
                       p, acc_nType[p], acc_m_d[p], args.mix);
      }
};

// Specielize UpdateUsingFluxNSCBCOutflowMinusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowYNeg";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowYNeg;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_m_d = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_vGradN  = FID_velocityGradientY;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_vGradT1 = FID_velocityGradientX;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::FID_vGradT2 = FID_velocityGradientZ;

template<>
void UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>::cpu_base_impl(
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

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   const int j = r_BC.lo.y;
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
         const Point<3> p = Point<3>(i,j,k);
         addLODIfluxes(acc_Conserved_t[p].v,
                       acc_MassFracs, acc_rho, acc_mu, acc_pressure,
                       acc_velocity, acc_vGradN, acc_vGradT1, acc_vGradT2,
                       acc_SoS[p], acc_temperature[p], acc_Conserved[p].v,
                       p, acc_nType[p], acc_m_d[p],
                       MaxMach, args.LengthScale, args.PInf, args.mix);
      }
};

// Specielize UpdateUsingFluxNSCBCOutflowPlusSide for the X direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowXPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowXPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_m_d = FID_dcsi_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_vGradN  = FID_velocityGradientX;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_vGradT1 = FID_velocityGradientY;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::FID_vGradT2 = FID_velocityGradientZ;

template<>
void UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>::cpu_base_impl(
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

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   const int i = r_BC.lo.x;
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++) {
         const Point<3> p = Point<3>(i,j,k);
         addLODIfluxes(acc_Conserved_t[p].v,
                       acc_MassFracs, acc_rho, acc_mu, acc_pressure,
                       acc_velocity, acc_vGradN, acc_vGradT1, acc_vGradT2,
                       acc_SoS[p], acc_temperature[p], acc_Conserved[p].v,
                       p, acc_nType[p], acc_m_d[p],
                       MaxMach, args.LengthScale, args.PInf, args.mix);
      }
};

// Specielize UpdateUsingFluxNSCBCOutflowPlusSide for the Y direction
template<>
/*static*/ const char * const    UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::TASK_NAME = "UpdateUsingFluxNSCBCOutflowYPos";
template<>
/*static*/ const int             UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::TASK_ID = TID_UpdateUsingFluxNSCBCOutflowYPos;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_m_d = FID_deta_d;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_vGradN  = FID_velocityGradientY;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_vGradT1 = FID_velocityGradientX;
template<>
/*static*/ const FieldID         UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::FID_vGradT2 = FID_velocityGradientZ;

template<>
void UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>::cpu_base_impl(
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

   // Wait for the maximum Mach number
   const double MaxMach = futures[0].get_result<double>();

   // Extract BC domain
   Rect<3> r_BC = runtime->get_index_space_domain(ctx,
      runtime->get_logical_subregion_by_color(args.Fluid_BC, 0).get_index_space());

   const int j = r_BC.lo.y;
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
         const Point<3> p = Point<3>(i,j,k);
         addLODIfluxes(acc_Conserved_t[p].v,
                       acc_MassFracs, acc_rho, acc_mu, acc_pressure,
                       acc_velocity, acc_vGradN, acc_vGradT1, acc_vGradT2,
                       acc_SoS[p], acc_temperature[p], acc_Conserved[p].v,
                       p, acc_nType[p], acc_m_d[p],
                       MaxMach, args.LengthScale, args.PInf, args.mix);
      }
};

void register_rhs_tasks() {

   TaskHelper::register_hybrid_variants<UpdateUsingHybridEulerFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingHybridEulerFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingHybridEulerFluxTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingTENOAEulerFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOAEulerFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingTENOAEulerFluxTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingDiffusionFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingDiffusionFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingDiffusionFluxTask<Zdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowMinusSideTask<Xdir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCInflowPlusSideTask<Ydir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowMinusSideTask<Ydir>>();

   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowPlusSideTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingFluxNSCBCOutflowPlusSideTask<Ydir>>();
};
