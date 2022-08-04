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

// AddRecycleAverageTask
/*static*/ const char * const    AddRecycleAverageTask::TASK_NAME = "AddRecycleAverage";
/*static*/ const int             AddRecycleAverageTask::TASK_ID = TID_AddRecycleAverageBC;

void AddRecycleAverageTask::cpu_base_impl(
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_plane.lo.z; k <= r_plane.hi.z; k++)
      for (int j = r_plane.lo.y; j <= r_plane.hi.y; j++)
         for (int i = r_plane.lo.x; i <= r_plane.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            collectAverages(acc_dcsi_d, acc_deta_d, acc_dzet_d,
                            acc_MolarFracs_profile, acc_temperature_profile, acc_velocity_profile,
                            acc_avg_MolarFracs, acc_avg_velocity,
                            acc_avg_temperature, acc_avg_rho,
                            args.Pbc, p, args.mix);
         }
}


// SetNSCBC_InflowBCTask
template<direction dir>
void SetNSCBC_InflowBCTask<dir>::cpu_base_impl(
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

   // Index of normal direction
   constexpr int iN = normalIndex(dir);

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            acc_MolarFracs[p] = acc_MolarFracs_profile[p];
            acc_temperature[p] = acc_temperature_profile[p];
            acc_velocity[p] = acc_velocity_profile[p];
            if (fabs(acc_velocity_profile[p][iN]) >= acc_SoS[p])
               // It is supersonic, everything is imposed by the BC
               acc_pressure[p] = args.Pbc;
            else
               // Compute pressure from NSCBC conservation equations
               setInflowPressure(acc_Conserved, acc_MolarFracs_profile, acc_temperature_profile,
                                 acc_pressure, p, args.mix);
         }
}

// Specielize SetNSCBC_InflowBCTask for the X direction
template<>
/*static*/ const char * const    SetNSCBC_InflowBCTask<Xdir>::TASK_NAME = "SetNSCBC_InflowBC";
template<>
/*static*/ const int             SetNSCBC_InflowBCTask<Xdir>::TASK_ID = TID_SetNSCBC_InflowBC_X;

// Specielize SetNSCBC_InflowBCTask for the Y direction
template<>
/*static*/ const char * const    SetNSCBC_InflowBCTask<Ydir>::TASK_NAME = "SetNSCBC_InflowBC";
template<>
/*static*/ const int             SetNSCBC_InflowBCTask<Ydir>::TASK_ID = TID_SetNSCBC_InflowBC_Y;

// Specielize SetNSCBC_InflowBCTask for the Z direction
template<>
/*static*/ const char * const    SetNSCBC_InflowBCTask<Zdir>::TASK_NAME = "SetNSCBC_InflowBC";
template<>
/*static*/ const int             SetNSCBC_InflowBCTask<Zdir>::TASK_ID = TID_SetNSCBC_InflowBC_Z;

// SetNSCBC_OutflowBCTask
/*static*/ const char * const    SetNSCBC_OutflowBCTask::TASK_NAME = "SetNSCBC_OutflowBC";
/*static*/ const int             SetNSCBC_OutflowBCTask::TASK_ID = TID_SetNSCBC_OutflowBC;

void SetNSCBC_OutflowBCTask::cpu_base_impl(
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            UpdatePrimitiveFromConservedTask::UpdatePrimitive(
                            acc_Conserved, acc_temperature, acc_pressure,
                            acc_MolarFracs, acc_velocity,
                            p, args.mix);
         }
}


// SetIncomingShockBCTask
/*static*/ const char * const    SetIncomingShockBCTask::TASK_NAME = "SetIncomingShockBC";
/*static*/ const int             SetIncomingShockBCTask::TASK_ID = TID_SetIncomingShockBC;

void SetIncomingShockBCTask::cpu_base_impl(
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            if ((i < args.params.iShock - 1) or
                (i > args.params.iShock + 1)){
               // Threat as an outflow
               UpdatePrimitiveFromConservedTask::UpdatePrimitive(
                               acc_Conserved, acc_temperature, acc_pressure,
                               acc_MolarFracs, acc_velocity,
                               p, args.mix);
            // Inject the shock over four points
            } else if (i == args.params.iShock - 1) {
               acc_MolarFracs[p]  = VecNSp(args.params.MolarFracs);
               acc_velocity[p]    = 0.75*Vec3(args.params.velocity0) + 0.25*Vec3(args.params.velocity1);
               acc_temperature[p] = 0.75*args.params.temperature0    + 0.25*args.params.temperature1;
               acc_pressure[p]    = 0.75*args.params.pressure0       + 0.25*args.params.pressure1;
            } else if (i == args.params.iShock) {
               acc_MolarFracs[p]  = VecNSp(args.params.MolarFracs);
               acc_velocity[p]    = 0.25*Vec3(args.params.velocity0) + 0.75*Vec3(args.params.velocity1);
               acc_temperature[p] = 0.25*args.params.temperature0    + 0.75*args.params.temperature1;
               acc_pressure[p]    = 0.25*args.params.pressure0       + 0.75*args.params.pressure1;
            } else if (i == args.params.iShock + 1) {
               acc_MolarFracs[p]  = VecNSp(args.params.MolarFracs);
               acc_velocity[p]    = Vec3(args.params.velocity1);
               acc_temperature[p] = args.params.temperature1;
               acc_pressure[p]    = args.params.pressure1;
            }
         }
}

// SetRecycleRescalingBCTask
/*static*/ const char * const    SetRecycleRescalingBCTask::TASK_NAME = "SetRecycleRescalingBC";
/*static*/ const int             SetRecycleRescalingBCTask::TASK_ID = TID_SetRecycleRescalingBC;

void SetRecycleRescalingBCTask::cpu_base_impl(
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};

            // Compute the rescaled primitive quantities
            double temperatureR; Vec3 velocityR; VecNSp MolarFracsR;
            GetRescaled(temperatureR, velocityR, MolarFracsR, acc_centerCoordinates,
                        acc_temperature_recycle, acc_velocity_recycle, acc_MolarFracs_recycle,
                        acc_temperature_profile, acc_velocity_profile, acc_MolarFracs_profile,
                        acc_avg_y, acc_FI_xloc, acc_FI_iloc, args.FIdata, p,
                        yInnFact, yOutFact, uInnFact, uOutFact, idelta99Inl);

            // Set boundary conditions
            acc_MolarFracs[p]  = MolarFracsR;
            acc_velocity[p]    = velocityR;
            acc_temperature[p] = temperatureR;
            if (fabs(velocityR[0]) >= acc_SoS[p])
               // It is supersonic, everything is imposed by the BC
               acc_pressure[p] = args.Pbc;
            else
               // Compute pressure from NSCBC conservation equations
               acc_pressure[p] = setPressure(acc_Conserved, temperatureR, MolarFracsR, p, args.mix);
         }
}

#if (defined(ELECTRIC_FIELD) && (nIons > 0))
// CorrectIonsBCTask
template<direction dir, side s>
void CorrectIonsBCTask<dir, s>::cpu_base_impl(
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_BC.lo.z; k <= r_BC.hi.z; k++)
      for (int j = r_BC.lo.y; j <= r_BC.hi.y; j++)
         for (int i = r_BC.lo.x; i <= r_BC.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            const Point<3> pInt = getPIntBC<dir, s>(p);
            const double dPhi = acc_ePotInt[pInt] - acc_ePot[p];
            for (int i = 0; i < nIons; i++) {
               int ind = args.mix.ions[i];
               if (args.mix.GetSpeciesChargeNumber(ind)*dPhi > 0)
                  // the ion is flowing into the BC
                  acc_MolarFracs[p][ind] = acc_MolarFracsInt[pInt][ind];
               else
                  // the ion is repelled by the BC
                  acc_MolarFracs[p][ind] = 1e-60;
            }
         }
}

// Specielize CorrectIonsBCTask for the X direction, Minus side
template<>
/*static*/ const char * const    CorrectIonsBCTask<Xdir, Minus>::TASK_NAME = "CorrectIonsBCXNeg";
template<>
/*static*/ const int             CorrectIonsBCTask<Xdir, Minus>::TASK_ID = TID_CorrectIonsBCXNeg;

// Specielize CorrectIonsBCTask for the X direction, Plus  side
template<>
/*static*/ const char * const    CorrectIonsBCTask<Xdir, Plus >::TASK_NAME = "CorrectIonsBCXPos";
template<>
/*static*/ const int             CorrectIonsBCTask<Xdir, Plus >::TASK_ID = TID_CorrectIonsBCXPos;

// Specielize CorrectIonsBCTask for the Y direction, Minus side
template<>
/*static*/ const char * const    CorrectIonsBCTask<Ydir, Minus>::TASK_NAME = "CorrectIonsBCYNeg";
template<>
/*static*/ const int             CorrectIonsBCTask<Ydir, Minus>::TASK_ID = TID_CorrectIonsBCYNeg;

// Specielize CorrectIonsBCTask for the Y direction, Plus  side
template<>
/*static*/ const char * const    CorrectIonsBCTask<Ydir, Plus >::TASK_NAME = "CorrectIonsBCYPos";
template<>
/*static*/ const int             CorrectIonsBCTask<Ydir, Plus >::TASK_ID = TID_CorrectIonsBCYPos;

// Specielize CorrectIonsBCTask for the Z direction, Minus side
template<>
/*static*/ const char * const    CorrectIonsBCTask<Zdir, Minus>::TASK_NAME = "CorrectIonsBCZNeg";
template<>
/*static*/ const int             CorrectIonsBCTask<Zdir, Minus>::TASK_ID = TID_CorrectIonsBCZNeg;

// Specielize CorrectIonsBCTask for the Z direction, Plus  side
template<>
/*static*/ const char * const    CorrectIonsBCTask<Zdir, Plus >::TASK_NAME = "CorrectIonsBCZPos";
template<>
/*static*/ const int             CorrectIonsBCTask<Zdir, Plus >::TASK_ID = TID_CorrectIonsBCZPos;
#endif

void register_bc_tasks() {

   TaskHelper::register_hybrid_variants<AddRecycleAverageTask>();

   TaskHelper::register_hybrid_variants<SetNSCBC_InflowBCTask<Xdir>>();
   TaskHelper::register_hybrid_variants<SetNSCBC_InflowBCTask<Ydir>>();
   TaskHelper::register_hybrid_variants<SetNSCBC_InflowBCTask<Zdir>>();

   TaskHelper::register_hybrid_variants<SetNSCBC_OutflowBCTask>();

   TaskHelper::register_hybrid_variants<SetIncomingShockBCTask>();

   TaskHelper::register_hybrid_variants<SetRecycleRescalingBCTask>();

#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   TaskHelper::register_hybrid_variants<CorrectIonsBCTask<Xdir, Minus>>();
   TaskHelper::register_hybrid_variants<CorrectIonsBCTask<Xdir, Plus >>();
   TaskHelper::register_hybrid_variants<CorrectIonsBCTask<Ydir, Minus>>();
   TaskHelper::register_hybrid_variants<CorrectIonsBCTask<Ydir, Plus >>();
   TaskHelper::register_hybrid_variants<CorrectIonsBCTask<Zdir, Minus>>();
   TaskHelper::register_hybrid_variants<CorrectIonsBCTask<Zdir, Plus >>();
#endif
};
