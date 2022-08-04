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

#include "prometeo_cfl.hpp"

// CalculateMaxSpectralRadiusTask
/*static*/ const char * const    CalculateMaxSpectralRadiusTask::TASK_NAME = "CalculateMaxSpectralRadius";
/*static*/ const int             CalculateMaxSpectralRadiusTask::TASK_ID = TID_CalculateMaxSpectralRadius;

double CalculateMaxSpectralRadiusTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 1);
   assert(futures.size() == 0);

   // Accessor for metrics
   const AccessorRO<double, 3> acc_dcsi_d           (regions[0], FID_dcsi_d);
   const AccessorRO<double, 3> acc_deta_d           (regions[0], FID_deta_d);
   const AccessorRO<double, 3> acc_dzet_d           (regions[0], FID_dzet_d);

   // Accessors for primitive variables
   const AccessorRO<VecNSp, 3> acc_MassFracs        (regions[0], FID_MassFracs);
   const AccessorRO<double, 3> acc_temperature      (regions[0], FID_temperature);
   const AccessorRO<  Vec3, 3> acc_velocity         (regions[0], FID_velocity);

   // Accessors for properties
   const AccessorRO<double, 3> acc_rho              (regions[0], FID_rho);
   const AccessorRO<double, 3> acc_mu               (regions[0], FID_mu);
   const AccessorRO<double, 3> acc_lam              (regions[0], FID_lam);
   const AccessorRO<VecNSp, 3> acc_Di               (regions[0], FID_Di);
   const AccessorRO<double, 3> acc_SoS              (regions[0], FID_SoS);
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   const AccessorRO<VecNIo, 3> acc_Ki               (regions[0], FID_Ki);

   // Accessors for primitive variables
   const AccessorRO<  Vec3, 3> acc_eField           (regions[0], FID_electricField);
#endif

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

   // Reduce spectral redii into r
   double r = 0.0;

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3) reduction(max:r)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            const double my_r = CalculateMaxSpectralRadius(
                                                       acc_dcsi_d, acc_deta_d, acc_dzet_d,
                                                       acc_temperature, acc_MassFracs, acc_velocity,
                                                       acc_rho, acc_mu, acc_lam, acc_Di, acc_SoS,
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
                                                       acc_Ki, acc_eField,
#endif
                                                       p, args.mix);
            if (my_r > r) r = my_r;
         }
   return r;
}

void register_cfl_tasks() {

   TaskHelper::register_hybrid_variants<CalculateMaxSpectralRadiusTask, double, DeferredValue<double>>();

};
