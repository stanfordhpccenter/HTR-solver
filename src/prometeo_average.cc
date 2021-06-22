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

// Add2DAveragesTask
template<direction dir>
void Add2DAveragesTask<dir>::cpu_base_impl(
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
   // the order between reduction operators for region requirements is
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_Fluid.lo.z; k <= r_Fluid.hi.z; k++)
      for (int j = r_Fluid.lo.y; j <= r_Fluid.hi.y; j++)
         for (int i = r_Fluid.lo.x; i <= r_Fluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            // TODO: implement some sort of static_if
            const Point<2> pA = (dir == Xdir) ? Point<2>{i, r_Avg.lo.y} :
                                (dir == Ydir) ? Point<2>{j, r_Avg.lo.y} :
                              /*(dir == Zdir)*/ Point<2>{k, r_Avg.lo.y};

            AvgPrimitive(acc_cellWidth, acc_centerCoordinates,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_avg_weight, acc_avg_centerCoordinates,
                        acc_pressure_avg,    acc_pressure_rms,
                        acc_temperature_avg, acc_temperature_rms,
                        acc_MolarFracs_avg,  acc_MolarFracs_rms,
                        acc_MassFracs_avg,   acc_MassFracs_rms,
                        acc_velocity_avg,    acc_velocity_rms, acc_velocity_rey,
                        p, pA, Integrator_deltaTime);

            FavreAvgPrimitive(acc_cellWidth, acc_rho,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_pressure_favg,    acc_pressure_frms,
                        acc_temperature_favg, acc_temperature_frms,
                        acc_MolarFracs_favg,  acc_MolarFracs_frms,
                        acc_MassFracs_favg,   acc_MassFracs_frms,
                        acc_velocity_favg,    acc_velocity_frms, acc_velocity_frey,
                        p, pA, Integrator_deltaTime);

            AvgProperties(acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_rho_avg, acc_rho_rms,
                        acc_mu_avg, acc_lam_avg, acc_Di_avg,
                        acc_SoS_avg, acc_cp_avg, acc_Ent_avg,
                        p, pA, args.mix, Integrator_deltaTime);

            FavreAvgProperties(acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_mu_favg, acc_lam_favg, acc_Di_favg,
                        acc_SoS_favg, acc_cp_favg, acc_Ent_favg,
                        p, pA, args.mix, Integrator_deltaTime);

            AvgFluxes_ProdRates(acc_cellWidth,
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
                        p, pA, args.Fluid_bounds, args.mix, Integrator_deltaTime);

            AvgKineticEnergyBudget(acc_cellWidth,
                        acc_pressure, acc_velocity,
                        acc_rho, acc_mu,
                        acc_vGradX, acc_vGradY, acc_vGradZ,
                        acc_rhoUUv, acc_Up, acc_tau,
                        acc_utau_y, acc_tauGradU, acc_pGradU,
                        p, pA, args.mix, Integrator_deltaTime);

            AvgDimensionlessNumbers(acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_Pr_avg, acc_Pr_rms,
                        acc_Ec_avg, acc_Ec_rms,
                        acc_Ma_avg, acc_Sc_avg,
                        p, pA, args.mix, Integrator_deltaTime);

            AvgCorrelations(acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity, acc_rho,
                        acc_uT_avg, acc_uT_favg,
                        acc_uYi_avg, acc_vYi_avg, acc_wYi_avg,
                        acc_uYi_favg, acc_vYi_favg, acc_wYi_favg,
                        p, pA, Integrator_deltaTime);

#ifdef ELECTRIC_FIELD
            AvgElectricQuantities(acc_cellWidth,
                        acc_temperature, acc_MolarFracs, acc_rho, acc_ePot,
                        acc_ePot_avg, acc_Crg_avg,
                        p, pA, args.mix, Integrator_deltaTime);
#endif
         }
}

// Specielize Add2DAveragesTask for the X direction
template<>
/*static*/ const char * const    Add2DAveragesTask<Xdir>::TASK_NAME = "Add2DAveragesX";
template<>
/*static*/ const int             Add2DAveragesTask<Xdir>::TASK_ID = TID_Add2DAveragesX;

// Specielize UpdateShockSensorTask for the Y direction
template<>
/*static*/ const char * const    Add2DAveragesTask<Ydir>::TASK_NAME = "Add2DAveragesY";
template<>
/*static*/ const int             Add2DAveragesTask<Ydir>::TASK_ID = TID_Add2DAveragesY;

// Specielize UpdateShockSensorTask for the Z direction
template<>
/*static*/ const char * const    Add2DAveragesTask<Zdir>::TASK_NAME = "Add2DAveragesZ";
template<>
/*static*/ const int             Add2DAveragesTask<Zdir>::TASK_ID = TID_Add2DAveragesZ;

// Add1DAveragesTask
template<direction dir>
void Add1DAveragesTask<dir>::cpu_base_impl(
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_Fluid.lo.z; k <= r_Fluid.hi.z; k++)
      for (int j = r_Fluid.lo.y; j <= r_Fluid.hi.y; j++)
         for (int i = r_Fluid.lo.x; i <= r_Fluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            // TODO: implement some sort of static_if
            const Point<3> pA = (dir == Xdir) ? Point<3>{j, k, r_Avg.lo.z} :
                                (dir == Ydir) ? Point<3>{i, k, r_Avg.lo.z} :
                              /*(dir == Zdir)*/ Point<3>{i, j, r_Avg.lo.z};

            AvgPrimitive(acc_cellWidth, acc_centerCoordinates,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_avg_weight, acc_avg_centerCoordinates,
                        acc_pressure_avg,    acc_pressure_rms,
                        acc_temperature_avg, acc_temperature_rms,
                        acc_MolarFracs_avg,  acc_MolarFracs_rms,
                        acc_MassFracs_avg,   acc_MassFracs_rms,
                        acc_velocity_avg,    acc_velocity_rms, acc_velocity_rey,
                        p, pA, Integrator_deltaTime);

            FavreAvgPrimitive(acc_cellWidth, acc_rho,
                        acc_pressure, acc_temperature, acc_MolarFracs,
                        acc_MassFracs, acc_velocity,
                        acc_pressure_favg,    acc_pressure_frms,
                        acc_temperature_favg, acc_temperature_frms,
                        acc_MolarFracs_favg,  acc_MolarFracs_frms,
                        acc_MassFracs_favg,   acc_MassFracs_frms,
                        acc_velocity_favg,    acc_velocity_frms, acc_velocity_frey,
                        p, pA, Integrator_deltaTime);

            AvgProperties(acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_rho_avg, acc_rho_rms,
                        acc_mu_avg, acc_lam_avg, acc_Di_avg,
                        acc_SoS_avg, acc_cp_avg, acc_Ent_avg,
                        p, pA, args.mix, Integrator_deltaTime);

            FavreAvgProperties(acc_cellWidth, acc_temperature, acc_MassFracs,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_mu_favg, acc_lam_favg, acc_Di_favg,
                        acc_SoS_favg, acc_cp_favg, acc_Ent_favg,
                        p, pA, args.mix, Integrator_deltaTime);

            AvgFluxes_ProdRates(acc_cellWidth,
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
                        p, pA, args.Fluid_bounds, args.mix, Integrator_deltaTime);

            AvgKineticEnergyBudget(acc_cellWidth,
                        acc_pressure, acc_velocity,
                        acc_rho, acc_mu,
                        acc_vGradX, acc_vGradY, acc_vGradZ,
                        acc_rhoUUv, acc_Up, acc_tau,
                        acc_utau_y, acc_tauGradU, acc_pGradU,
                        p, pA, args.mix, Integrator_deltaTime);

            AvgDimensionlessNumbers(acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity,
                        acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
                        acc_Pr_avg, acc_Pr_rms,
                        acc_Ec_avg, acc_Ec_rms,
                        acc_Ma_avg, acc_Sc_avg,
                        p, pA, args.mix, Integrator_deltaTime);

            AvgCorrelations(acc_cellWidth,
                        acc_temperature, acc_MassFracs, acc_velocity, acc_rho,
                        acc_uT_avg, acc_uT_favg,
                        acc_uYi_avg, acc_vYi_avg, acc_wYi_avg,
                        acc_uYi_favg, acc_vYi_favg, acc_wYi_favg,
                        p, pA, Integrator_deltaTime);

#ifdef ELECTRIC_FIELD
            AvgElectricQuantities(acc_cellWidth,
                        acc_temperature, acc_MolarFracs, acc_rho, acc_ePot,
                        acc_ePot_avg, acc_Crg_avg,
                        p, pA, args.mix, Integrator_deltaTime);
#endif
         }
}

// Specielize Add1DAveragesTask for the X direction
template<>
/*static*/ const char * const    Add1DAveragesTask<Xdir>::TASK_NAME = "Add1DAveragesX";
template<>
/*static*/ const int             Add1DAveragesTask<Xdir>::TASK_ID = TID_Add1DAveragesX;

// Specielize Add1DAveragesTask for the Y direction
template<>
/*static*/ const char * const    Add1DAveragesTask<Ydir>::TASK_NAME = "Add1DAveragesY";
template<>
/*static*/ const int             Add1DAveragesTask<Ydir>::TASK_ID = TID_Add1DAveragesY;

// Specielize Add1DAveragesTask for the Z direction
template<>
/*static*/ const char * const    Add1DAveragesTask<Zdir>::TASK_NAME = "Add1DAveragesZ";
template<>
/*static*/ const int             Add1DAveragesTask<Zdir>::TASK_ID = TID_Add1DAveragesZ;

void register_average_tasks() {

   TaskHelper::register_hybrid_variants<Add2DAveragesTask<Xdir>>();
   TaskHelper::register_hybrid_variants<Add2DAveragesTask<Ydir>>();
   TaskHelper::register_hybrid_variants<Add2DAveragesTask<Zdir>>();

   TaskHelper::register_hybrid_variants<Add1DAveragesTask<Xdir>>();
   TaskHelper::register_hybrid_variants<Add1DAveragesTask<Ydir>>();
   TaskHelper::register_hybrid_variants<Add1DAveragesTask<Zdir>>();

};
