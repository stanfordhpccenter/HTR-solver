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

#include "prometeo_chem.hpp"

// UpdateChemistryTask
/*static*/ const char * const    UpdateChemistryTask::TASK_NAME = "UpdateChemistry";
/*static*/ const int             UpdateChemistryTask::TASK_ID = TID_UpdateChemistry;

void UpdateChemistryTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 1);

   // Accessors for RHS
   const AccessorRO<VecNEq, 3> acc_Conserved_t      (regions[0], FID_Conserved_t);

   // Accessors for implicit variables
   const AccessorRW<VecNEq, 3> acc_Conserved        (regions[1], FID_Conserved);
   const AccessorRW<VecNEq, 3> acc_Conserved_t_old  (regions[1], FID_Conserved_t_old);
   const AccessorRW<double, 3> acc_temperature      (regions[1], FID_temperature);

   // Extract execution domain
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Wait for the Integrator_deltaTime
   const double Integrator_deltaTime = futures[0].get_result<double>();

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            acc_Conserved_t_old[p] = acc_Conserved_t[p];
            ImplicitSolver s = ImplicitSolver(acc_Conserved_t_old[p], acc_temperature[p], args.mix);
            s.solve(acc_Conserved[p], Integrator_deltaTime, Integrator_deltaTime);
         }
}


// AddChemistrySourcesTask
/*static*/ const char * const    AddChemistrySourcesTask::TASK_NAME = "AddChemistrySources";
/*static*/ const int             AddChemistrySourcesTask::TASK_ID = TID_AddChemistrySources;

void AddChemistrySourcesTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for primitive variables and properites
   const AccessorRO<double, 3> acc_rho              (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_pressure         (regions[0], FID_pressure);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<VecNSp, 3> acc_MassFracs        (regions[0], FID_MassFracs);

   // Accessors for RHS
   const AccessorRW<VecNEq, 3> acc_Conserved_t      (regions[1], FID_Conserved_t);

   // Extract execution domain
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            VecNSp w; args.mix.GetProductionRates(w, acc_rho[p], acc_pressure[p],
                                         acc_temperature[p], acc_MassFracs[p]);
            for (int i = 0; i<nSpec; i++)
               acc_Conserved_t[p][i] += w[i];
         }
}

void register_chem_tasks() {

   TaskHelper::register_hybrid_variants<UpdateChemistryTask>();

   TaskHelper::register_hybrid_variants<AddChemistrySourcesTask>();

};
