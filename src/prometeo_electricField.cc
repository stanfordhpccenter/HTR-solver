// Copyright (c) "2020, by Centre Européen de Recherche et de Formation Avancée en Calcul Scientifiq
//               Developer: Mario Di Renzo
//               Affiliation: Centre Européen de Recherche et de Formation Avancée en Calcul Scientifique
//               URL: https://cerfacs.fr
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

#include "prometeo_electricField.hpp"
#include "prometeo_electricField.inl"

// GetElectricFieldTask
/*static*/ const char * const    GetElectricFieldTask::TASK_NAME = "GetElectricField";
/*static*/ const int             GetElectricFieldTask::TASK_ID = TID_GetElectricField;

void GetElectricFieldTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<double, 3> acc_ePot    (regions[0], FID_electricPotential);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d  (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d  (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d  (regions[1], FID_dzet_d);

   // Accessors for gradients
   const AccessorWO<  Vec3, 3> acc_eField  (regions[2], FID_electricField);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            acc_eField[p] = -getGrad(acc_ePot, p,
                             acc_nType_x[p], acc_nType_y[p], acc_nType_z[p],
                             acc_dcsi_d[p], acc_deta_d[p], acc_dzet_d[p],
                             Fluid_bounds);
         }
}

#if (nIons > 0)
// Specielize UpdateUsingIonDriftFlux for the X direction
template<>
/*static*/ const char * const    UpdateUsingIonDriftFluxTask<Xdir>::TASK_NAME = "UpdateUsingIonDriftFluxX";
template<>
/*static*/ const int             UpdateUsingIonDriftFluxTask<Xdir>::TASK_ID = TID_UpdateUsingIonDriftFluxX;
template<>
/*static*/ const FieldID         UpdateUsingIonDriftFluxTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateUsingIonDriftFluxTask<Xdir>::FID_m_e = FID_dcsi_e;

template<>
void UpdateUsingIonDriftFluxTask<Xdir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 0);

   // Accessors for EulerGhost region
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_eField     (regions[0], FID_electricField);
   const AccessorRO<VecNIo, 3> acc_Ki         (regions[0], FID_Ki);

   // Accessors for DiffGhost region
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e        (regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[4].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Euler and Diffision fluxes
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++) {
         // Launch the loop for the span
         updateRHSSpan(acc_Conserved_t, acc_m_e, acc_nType,
                  acc_Conserved, acc_MassFracs, acc_Ki, acc_eField,
                  0, getSize<Xdir>(r_MyFluid),
                  0, j-r_MyFluid.lo.y, k-r_MyFluid.lo.z,
                  r_MyFluid, Fluid_bounds, args.mix);
      }
}

// Specielize UpdateUsingIonDriftFlux for the Y direction
template<>
/*static*/ const char * const    UpdateUsingIonDriftFluxTask<Ydir>::TASK_NAME = "UpdateUsingIonDriftFluxY";
template<>
/*static*/ const int             UpdateUsingIonDriftFluxTask<Ydir>::TASK_ID = TID_UpdateUsingIonDriftFluxY;
template<>
/*static*/ const FieldID         UpdateUsingIonDriftFluxTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateUsingIonDriftFluxTask<Ydir>::FID_m_e = FID_deta_e;

template<>
void UpdateUsingIonDriftFluxTask<Ydir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 0);

   // Accessors for EulerGhost region
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_eField     (regions[0], FID_electricField);
   const AccessorRO<VecNIo, 3> acc_Ki         (regions[0], FID_Ki);

   // Accessors for DiffGhost region
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e        (regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[4].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Euler and Diffision fluxes
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
         // Launch the loop for the span
         updateRHSSpan(acc_Conserved_t, acc_m_e, acc_nType,
                  acc_Conserved, acc_MassFracs, acc_Ki, acc_eField,
                  0, getSize<Ydir>(r_MyFluid),
                  i-r_MyFluid.lo.x, 0, k-r_MyFluid.lo.z,
                  r_MyFluid, Fluid_bounds, args.mix);
      }
}

// Specielize UpdateUsingIonDriftFlux for the Z direction
template<>
/*static*/ const char * const    UpdateUsingIonDriftFluxTask<Zdir>::TASK_NAME = "UpdateUsingIonDriftFluxZ";
template<>
/*static*/ const int             UpdateUsingIonDriftFluxTask<Zdir>::TASK_ID = TID_UpdateUsingIonDriftFluxZ;
template<>
/*static*/ const FieldID         UpdateUsingIonDriftFluxTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateUsingIonDriftFluxTask<Zdir>::FID_m_e = FID_dzet_e;

template<>
void UpdateUsingIonDriftFluxTask<Zdir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 5);
   assert(futures.size() == 0);

   // Accessors for EulerGhost region
   const AccessorRO<VecNEq, 3> acc_Conserved  (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_eField     (regions[0], FID_electricField);
   const AccessorRO<VecNIo, 3> acc_Ki         (regions[0], FID_Ki);

   // Accessors for DiffGhost region
   const AccessorRO<VecNSp, 3> acc_MassFracs  (regions[1], FID_MassFracs);

   // Accessors for DivgGhost region
   const AccessorRO<   int, 3> acc_nType      (regions[2], FID_nType);

   // Accessors for metrics
   const AccessorRO<double, 3> acc_m_e        (regions[3], FID_m_e);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t(regions[4], FID_Conserved_t);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[4].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // update RHS using Euler and Diffision fluxes
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
      for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
         // Launch the loop for the span
         updateRHSSpan(acc_Conserved_t, acc_m_e, acc_nType,
                  acc_Conserved, acc_MassFracs, acc_Ki, acc_eField,
                  0, getSize<Zdir>(r_MyFluid),
                  i-r_MyFluid.lo.x, j-r_MyFluid.lo.y, 0,
                  r_MyFluid, Fluid_bounds, args.mix);
      }
}
#endif

// AddIonWindSourcesTask
/*static*/ const char * const    AddIonWindSourcesTask::TASK_NAME = "AddIonWindSources";
/*static*/ const int             AddIonWindSourcesTask::TASK_ID = TID_AddIonWindSources;

void AddIonWindSourcesTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for stencil variables
   const AccessorRO<VecNSp, 3> acc_MolarFracs       (regions[0], FID_MolarFracs);

   // Accessors for node types and metrics
   const AccessorRO<   int, 3> acc_nType_x          (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi             (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta             (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet             (regions[1], FID_dzet_d);

   // Accessors for primitive variables
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[1], FID_velocity);
   const AccessorRO<  Vec3, 3> acc_eField           (regions[1], FID_electricField);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho              (regions[1], FID_rho);
   const AccessorRO<VecNSp, 3> acc_Di               (regions[1], FID_Di);
#if (nIons > 0)
   const AccessorRO<VecNIo, 3> acc_Ki               (regions[1], FID_Ki);
#endif

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t      (regions[2], FID_Conserved_t);

   // Extract execution domain
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>(i,j,k);
            addIonWindSources(acc_Conserved_t[p],
               acc_rho, acc_Di,
#if (nIons > 0)
               acc_Ki,
#endif
               acc_velocity, acc_eField, acc_MolarFracs,
               acc_nType_x, acc_nType_y, acc_nType_z,
               acc_dcsi, acc_deta, acc_dzet,
               p, Fluid_bounds, args.mix);
         }
}

void register_electricField_tasks() {

   // Register task for the Poisson solver
   register_poisson_tasks();

   // Register electric field calculation
   TaskHelper::register_hybrid_variants<GetElectricFieldTask>();

#if (nIons > 0)
   // Refister ion drift flux tasks
   TaskHelper::register_hybrid_variants<UpdateUsingIonDriftFluxTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingIonDriftFluxTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateUsingIonDriftFluxTask<Zdir>>();
#endif

   // Register ion wind source terms task
   TaskHelper::register_hybrid_variants<AddIonWindSourcesTask>();

};
