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

#ifndef __PROMETEO_CHEM_HPP__
#define __PROMETEO_CHEM_HPP__

#include "legion.h"

using namespace Legion;

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "my_array.hpp"
#include "math_utils.hpp"
#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_chem.h"

//-----------------------------------------------------------------------------
// TASK THAT UPDATES THE CONSERVED VARIABLES BY IMPLICITLY SOLVING THE CHEMISTRY
//-----------------------------------------------------------------------------

// Implicit problem for the chemistry
class ImplicitSolver : public Rosenbrock<nEq> {
public:
   __CUDA_HD__
   ImplicitSolver(const VecNEq &Conserved_t_old_,
                  double &temperature_,
                  const Mix &mix_) :
      Conserved_t_old(Conserved_t_old_),
      temperature(temperature_),
      mix(mix_)
   {};

private:
   __CUDA_HD__
   inline void rhs(VecNEq &r, const VecNEq &x) {
      VecNSp rhoYi;
      __UNROLL__
      for (int i=0; i<nSpec; i++) rhoYi[i] = x[i];
      const double rho = mix.GetRhoFromRhoYi(rhoYi);
      VecNSp Yi; mix.GetYi(Yi, rho, rhoYi);
      mix.ClipYi(Yi);
      assert(mix.CheckMixture(Yi));
      const double MixW = mix.GetMolarWeightFromYi(Yi);
      const double rhoInv = 1.0/rho;
      Vec3 velocity;
      __UNROLL__
      for (int i=0; i<3; i++) velocity[i] = x[i+irU]*rhoInv;
      const double kineticEnergy = 0.5*velocity.mod2();
      const double InternalEnergy = x[irE]*rhoInv - kineticEnergy;
      temperature = mix.GetTFromInternalEnergy(InternalEnergy, temperature, Yi);
      const double pressure = mix.GetPFromRhoAndT(rho, MixW, temperature);
      VecNSp w; mix.GetProductionRates(w, rho, pressure, temperature, Yi);
      __UNROLL__
      for (int i=0; i<nSpec; i++) r[i] = w[i] + Conserved_t_old[i];
      __UNROLL__
      for (int i=nSpec; i<nEq; i++) r[i] = Conserved_t_old[i];
   };
private:
   const VecNEq &Conserved_t_old;
   double       &temperature;
   const Mix    &mix;
};

class UpdateChemistryTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      double Integrator_deltaTime;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

//-----------------------------------------------------------------------------
// TASK THAT ADDS THE CHEMICAL PRODUCTION RATES TO THE RHS OF THE EQUATIONS
//-----------------------------------------------------------------------------

class AddChemistrySourcesTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      Mix mix;
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

#endif // __PROMETEO_CHEM_HPP__
