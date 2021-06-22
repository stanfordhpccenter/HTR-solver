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

#include "prometeo_variables.hpp"

// UpdatePropertiesFromPrimitiveTask
/*static*/ const char * const    UpdatePropertiesFromPrimitiveTask::TASK_NAME = "UpdatePropertiesFromPrimitive";
/*static*/ const int             UpdatePropertiesFromPrimitiveTask::TASK_ID = TID_UpdatePropertiesFromPrimitive;

void UpdatePropertiesFromPrimitiveTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for primitive variables
   const AccessorRO<double, 3> acc_pressure         (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MolarFracs       (regions[0], FID_MolarFracs);
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   const AccessorWO<VecNSp, 3> acc_MassFracs        (regions[1], FID_MassFracs);

   // Accessors for properties
   const AccessorWO<double, 3> acc_rho              (regions[1], FID_rho);
   const AccessorWO<double, 3> acc_mu               (regions[1], FID_mu);
   const AccessorWO<double, 3> acc_lam              (regions[1], FID_lam);
   const AccessorWO<VecNSp, 3> acc_Di               (regions[1], FID_Di);
   const AccessorWO<double, 3> acc_SoS              (regions[1], FID_SoS);
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   const AccessorWO<VecNIo, 3> acc_Ki               (regions[1], FID_Ki);
#endif

   // Extract execution domain
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++)
         for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            // Mixture check
            assert(args.mix.CheckMixture(acc_MolarFracs[p]));
            UpdateProperties(acc_pressure, acc_temperature,
                             acc_MolarFracs, acc_velocity,
                             acc_MassFracs,
                             acc_rho, acc_mu, acc_lam,
                             acc_Di, acc_SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                             acc_Ki,
#endif
                             p, args.mix);
         }
}

// UpdateConservedFromPrimitiveTask
/*static*/ const char * const    UpdateConservedFromPrimitiveTask::TASK_NAME = "UpdateConservedFromPrimitive";
/*static*/ const int             UpdateConservedFromPrimitiveTask::TASK_ID = TID_UpdateConservedFromPrimitive;

void UpdateConservedFromPrimitiveTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for primitive variables
   const AccessorRO<VecNSp, 3> acc_MassFracs        (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho              (regions[0], FID_rho);

   // Accessors for conserved variables
   const AccessorWO<VecNEq, 3> acc_Conserved        (regions[1], FID_Conserved);

   // Extract execution domain
   Domain r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());

   // Launch domain might be composed by multiple rectangles
   for (RectInDomainIterator<3> Rit(r_ModCells); Rit(); Rit++) {
      // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
      #pragma omp parallel for collapse(3)
#endif
      for (int k = (*Rit).lo.z; k <= (*Rit).hi.z; k++)
         for (int j = (*Rit).lo.y; j <= (*Rit).hi.y; j++)
            for (int i = (*Rit).lo.x; i <= (*Rit).hi.x; i++) {
               const Point<3> p = Point<3>{i,j,k};
               // Mixture check
               assert(args.mix.CheckMixture(acc_MassFracs[p]));
               UpdateConserved(acc_MassFracs, acc_temperature, acc_velocity,
                               acc_rho, acc_Conserved,
                               p, args.mix);
            }
   }
}

// UpdatePrimitiveFromConservedTask
/*static*/ const char * const    UpdatePrimitiveFromConservedTask::TASK_NAME = "UpdatePrimitiveFromConserved";
/*static*/ const int             UpdatePrimitiveFromConservedTask::TASK_ID = TID_UpdatePrimitiveFromConserved;

void UpdatePrimitiveFromConservedTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for conserved variables
   const AccessorRO<VecNEq, 3> acc_Conserved        (regions[0], FID_Conserved);

   // Accessors for temperature variables
   const AccessorRW<double, 3> acc_temperature      (regions[1], FID_temperature);

   // Accessors for primitive variables
   const AccessorWO<double, 3> acc_pressure         (regions[1], FID_pressure);
   const AccessorWO<VecNSp, 3> acc_MolarFracs       (regions[1], FID_MolarFracs);
   const AccessorWO<  Vec3, 3> acc_velocity         (regions[1], FID_velocity);

   // Extract execution domain
   Rect<3> r_ModCells = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_ModCells.lo.z; k <= r_ModCells.hi.z; k++)
      for (int j = r_ModCells.lo.y; j <= r_ModCells.hi.y; j++)
         for (int i = r_ModCells.lo.x; i <= r_ModCells.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            UpdatePrimitive(acc_Conserved, acc_temperature, acc_pressure,
                            acc_MolarFracs, acc_velocity,
                            p, args.mix);
         }
}


// GetVelocityGradientsTask
/*static*/ const char * const    GetVelocityGradientsTask::TASK_NAME = "GetVelocityGradients";
/*static*/ const int             GetVelocityGradientsTask::TASK_ID = TID_GetVelocityGradients;

void GetVelocityGradientsTask::cpu_base_impl(
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

   // Accessors for gradients
   const AccessorWO<  Vec3, 3> acc_vGradX           (regions[2], FID_velocityGradientX);
   const AccessorWO<  Vec3, 3> acc_vGradY           (regions[2], FID_velocityGradientY);
   const AccessorWO<  Vec3, 3> acc_vGradZ           (regions[2], FID_velocityGradientZ);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Get domain sizes
   const coord_t xsize = getSize<Xdir>(Fluid_bounds);
   const coord_t ysize = getSize<Ydir>(Fluid_bounds);
   const coord_t zsize = getSize<Zdir>(Fluid_bounds);

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            // X gradient
            computeDerivatives<Xdir>(acc_vGradX, acc_velocity, p,
                                     acc_nType_x[p], acc_dcsi_d[p],
                                     xsize, Fluid_bounds);

            // Y gradient
            computeDerivatives<Ydir>(acc_vGradY, acc_velocity, p,
                                     acc_nType_y[p], acc_deta_d[p],
                                     ysize, Fluid_bounds);

            // Z gradient
            computeDerivatives<Zdir>(acc_vGradZ, acc_velocity, p,
                                     acc_nType_z[p], acc_dzet_d[p],
                                     zsize, Fluid_bounds);
         }
}

#if 0
// GetTemperatureGradientTask
/*static*/ const char * const    GetTemperatureGradientTask::TASK_NAME = "GetTemperatureGradient";
/*static*/ const int             GetTemperatureGradientTask::TASK_ID = TID_GetTemperatureGradient;

void GetTemperatureGradientTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);

   // Accessors for metrics
   const AccessorRO<   int, 3> acc_nType_x          (regions[1], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[1], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[1], FID_nType_z);
   const AccessorRO<double, 3> acc_dcsi_d           (regions[1], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d           (regions[1], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d           (regions[1], FID_dzet_d);

   // Accessors for gradients
   const AccessorWO<  Vec3, 3> acc_tGrad            (regions[2], FID_temperatureGradient);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.Fluid.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Get domain sizes
   const coord_t xsize = getSize<Xdir>(Fluid_bounds);
   const coord_t ysize = getSize<Ydir>(Fluid_bounds);
   const coord_t zsize = getSize<Zdir>(Fluid_bounds);

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            // X gradient
            acc_tGrad[p][0] = computeDerivative<Xdir>(acc_temperature, p,
                                                      acc_nType_x[p], acc_dcsi_d[p],
                                                      xsize, Fluid_bounds);

            // Y gradient
            acc_tGrad[p][1] = computeDerivative<Ydir>(acc_temperature, p,
                                                      acc_nType_y[p], acc_deta_d[p],
                                                      ysize, Fluid_bounds);

            // Z gradient
            acc_tGrad[p][2] = computeDerivative<Zdir>(acc_temperature, p,
                                                      acc_nType_z[p], acc_dzet_d[p],
                                                      zsize, Fluid_bounds);
         }
}
#endif


void register_variables_tasks() {

   TaskHelper::register_hybrid_variants<UpdatePropertiesFromPrimitiveTask>();

   TaskHelper::register_hybrid_variants<UpdateConservedFromPrimitiveTask>();

   TaskHelper::register_hybrid_variants<UpdatePrimitiveFromConservedTask>();

   TaskHelper::register_hybrid_variants<GetVelocityGradientsTask>();

//   TaskHelper::register_hybrid_variants<GetTemperatureGradientTask>();

};
