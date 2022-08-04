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

// AddAveragesTask
template<direction dir, int N>
void AddAveragesTask<dir, N>::cpu_base_impl(
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
   // the order between reduction operators for region requirements is
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

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_Fluid.lo.z; k <= r_Fluid.hi.z; k++)
      for (int j = r_Fluid.lo.y; j <= r_Fluid.hi.y; j++)
         for (int i = r_Fluid.lo.x; i <= r_Fluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            const Point<N> pA = getPAvg<dir, N>(p, r_Avg);

            const double weight = AverageUtils<N>::getWeight(acc_dcsi_d, acc_deta_d, acc_dzet_d, p, Integrator_deltaTime);
            const double fweight = AverageUtils<N>::getFavreWeight(acc_dcsi_d, acc_deta_d, acc_dzet_d, acc_rho, p, Integrator_deltaTime);

            // Position and integrated weight
            AverageUtils<N>::PositionAndWeight(acc_centerCoordinates, acc_avg_weight, acc_avg_centerCoordinates, p, pA, weight);

            // Reynolds average of primitive variables
            AverageUtils<N>::Avg(acc_pressure,    acc_pressure_avg,    acc_pressure_rms,             p, pA, weight);
            AverageUtils<N>::Avg(acc_temperature, acc_temperature_avg, acc_temperature_rms,          p, pA, weight);
            AverageUtils<N>::Avg(acc_MolarFracs,  acc_MolarFracs_avg,  acc_MolarFracs_rms,           p, pA, weight);
            AverageUtils<N>::Avg(acc_MassFracs,   acc_MassFracs_avg,   acc_MassFracs_rms,            p, pA, weight);
            AverageUtils<N>::Avg(acc_velocity, acc_velocity_avg, acc_velocity_rms, acc_velocity_rey, p, pA, weight);

            // Favre average of primitive variables
            AverageUtils<N>::Avg(acc_pressure,    acc_pressure_favg,    acc_pressure_frms,              p, pA, fweight);
            AverageUtils<N>::Avg(acc_temperature, acc_temperature_favg, acc_temperature_frms,           p, pA, fweight);
            AverageUtils<N>::Avg(acc_MolarFracs , acc_MolarFracs_favg,  acc_MolarFracs_frms,            p, pA, fweight);
            AverageUtils<N>::Avg(acc_MassFracs,   acc_MassFracs_favg,   acc_MassFracs_frms,             p, pA, fweight);
            AverageUtils<N>::Avg(acc_velocity, acc_velocity_favg, acc_velocity_frms, acc_velocity_frey, p, pA, fweight);

            // Reynolds average of properties
            AverageUtils<N>::Avg(acc_rho,    acc_rho_avg,    acc_rho_rms,   p, pA, weight);
            AverageUtils<N>::Avg(acc_mu,     acc_mu_avg,                    p, pA, weight);
            AverageUtils<N>::Avg(acc_lam,    acc_lam_avg,                   p, pA, weight);
            AverageUtils<N>::Avg(acc_Di,     acc_Di_avg,                    p, pA, weight);
            AverageUtils<N>::Avg(acc_SoS,    acc_SoS_avg,                   p, pA, weight);

            // Favre average of properties
            AverageUtils<N>::Avg(acc_mu,     acc_mu_favg,                    p, pA, fweight);
            AverageUtils<N>::Avg(acc_lam,    acc_lam_favg,                   p, pA, fweight);
            AverageUtils<N>::Avg(acc_Di,     acc_Di_favg,                    p, pA, fweight);
            AverageUtils<N>::Avg(acc_SoS,    acc_SoS_favg,                   p, pA, fweight);

            // Other properties averages
            AverageUtils<N>::CpEntAvg(
                     acc_temperature, acc_MassFracs,
                     acc_cp_avg, acc_cp_favg,
                     acc_Ent_avg, acc_Ent_favg,
                     p, pA, args.mix, weight, fweight);

            // Average production rates
            AverageUtils<N>::ProdRatesAvg(
                        acc_pressure, acc_temperature, acc_MassFracs, acc_rho,
                        acc_ProductionRates_avg, acc_ProductionRates_rms,
                        acc_HeatReleaseRate_avg, acc_HeatReleaseRate_rms,
                        p, pA, args.mix, weight);

            // Average heat flux
            AverageUtils<N>::HeatFluxAvg(
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        acc_temperature, acc_MolarFracs, acc_MassFracs,
                        acc_rho, acc_lam, acc_Di,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                        acc_Ki, acc_eField,
#endif
                        acc_q_avg,
                        p, pA, args.Fluid_bounds, args.mix, weight);

            // Average kinetic energy budget terms
            AverageUtils<N>::AvgKineticEnergyBudget(
                        acc_pressure, acc_velocity, acc_rho,
                        acc_rhoUUv, acc_Up,
                        p, pA, weight);

            AverageUtils<N>::AvgKineticEnergyBudget_Tau(
                        acc_nType_x, acc_nType_y, acc_nType_z,
                        acc_dcsi_d, acc_deta_d, acc_dzet_d,
                        acc_pressure, acc_velocity,
                        acc_rho, acc_mu,
                        acc_tau, acc_utau_y, acc_tauGradU, acc_pGradU,
                        p, pA, args.Fluid_bounds, args.mix, weight);

            // Average dimensionless numbers
            AverageUtils<N>::PrEcAvg(
                        acc_temperature, acc_MassFracs, acc_velocity, acc_mu, acc_lam,
                        acc_Pr_avg, acc_Pr_rms,
                        acc_Ec_avg, acc_Ec_rms,
                        p, pA, args.mix, weight);
            AverageUtils<N>::MaAvg(acc_velocity, acc_SoS, acc_Ma_avg, p, pA, weight);
            AverageUtils<N>::ScAvg(acc_rho, acc_mu, acc_Di, acc_Sc_avg, p, pA, weight);

            // Average correlations
            AverageUtils<N>::Cor(acc_temperature, acc_velocity, acc_uT_avg,  p, pA,  weight);
            AverageUtils<N>::Cor(acc_temperature, acc_velocity, acc_uT_favg, p, pA, fweight);
            AverageUtils<N>::Cor(acc_MassFracs,   acc_velocity, acc_uYi_avg, acc_vYi_avg, acc_wYi_avg,     p, pA,  weight);
            AverageUtils<N>::Cor(acc_MassFracs,   acc_velocity, acc_uYi_favg, acc_vYi_favg, acc_wYi_favg,  p, pA, fweight);

#ifdef ELECTRIC_FIELD
            // Average electric quantities
            AverageUtils<N>::Avg(acc_ePot, acc_ePot_avg, p, pA, weight);
            AverageUtils<N>::ElectricChargeAvg(
                        acc_MolarFracs, acc_rho, acc_Crg_avg,
                        p, pA, args.mix, weight);
#endif
         }
}

// Specielize AddAveragesTask for rakes the X direction
template<>
/*static*/ const char * const    AddAveragesTask<Xdir, 2>::TASK_NAME = "Add2DAveragesX";
template<>
/*static*/ const int             AddAveragesTask<Xdir, 2>::TASK_ID = TID_Add2DAveragesX;

// Specielize AddAveragesTask for rakes the Y direction
template<>
/*static*/ const char * const    AddAveragesTask<Ydir, 2>::TASK_NAME = "Add2DAveragesY";
template<>
/*static*/ const int             AddAveragesTask<Ydir, 2>::TASK_ID = TID_Add2DAveragesY;

// Specielize AddAveragesTask for rakes the Z direction
template<>
/*static*/ const char * const    AddAveragesTask<Zdir, 2>::TASK_NAME = "Add2DAveragesZ";
template<>
/*static*/ const int             AddAveragesTask<Zdir, 2>::TASK_ID = TID_Add2DAveragesZ;

// Specielize AddAveragesTask for planes with normal in the X direction
template<>
/*static*/ const char * const    AddAveragesTask<Xdir, 3>::TASK_NAME = "Add1DAveragesX";
template<>
/*static*/ const int             AddAveragesTask<Xdir, 3>::TASK_ID = TID_Add1DAveragesX;

// Specielize AddAveragesTask for planes with normal in the Y direction
template<>
/*static*/ const char * const    AddAveragesTask<Ydir, 3>::TASK_NAME = "Add1DAveragesY";
template<>
/*static*/ const int             AddAveragesTask<Ydir, 3>::TASK_ID = TID_Add1DAveragesY;

// Specielize AddAveragesTask for planes with normal in the Z direction
template<>
/*static*/ const char * const    AddAveragesTask<Zdir, 3>::TASK_NAME = "Add1DAveragesZ";
template<>
/*static*/ const int             AddAveragesTask<Zdir, 3>::TASK_ID = TID_Add1DAveragesZ;

void register_average_tasks() {

   TaskHelper::register_hybrid_variants<AddAveragesTask<Xdir, 2>>();
   TaskHelper::register_hybrid_variants<AddAveragesTask<Ydir, 2>>();
   TaskHelper::register_hybrid_variants<AddAveragesTask<Zdir, 2>>();

   TaskHelper::register_hybrid_variants<AddAveragesTask<Xdir, 3>>();
   TaskHelper::register_hybrid_variants<AddAveragesTask<Ydir, 3>>();
   TaskHelper::register_hybrid_variants<AddAveragesTask<Zdir, 3>>();

};
