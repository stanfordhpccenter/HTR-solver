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

#include "Poisson.hpp"

#ifdef REALM_USE_OPENMP
#include <omp.h>
#endif

// InitFFTplansTask
/*static*/ const char * const    initFFTplansTask::TASK_NAME = "initFFTplans";
/*static*/ const int             initFFTplansTask::TASK_ID = TID_initFFTplans;

void initFFTplansTask::cpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for FFT plans
   const AccessorRW<             fftw_plan, 1> acc_fftw_fwd  (regions[1], FID_fftw_fwd);
   const AccessorRW<             fftw_plan, 1> acc_fftw_bwd  (regions[1], FID_fftw_bwd);
   const AccessorRW<legion_address_space_t, 1> acc_id        (regions[1], FID_id);

   // Get size of the FFT execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);
   fftw_complex *aux = new fftw_complex[(size_x*size_z)];

   // Get index of the plans that we are initializing
   Point<1> p = Rect<1>(runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space())).lo;
   // crate plan for direct transform with FFTW
   fftw_make_planner_thread_safe();
   acc_fftw_fwd[p] = fftw_plan_dft_2d(size_x, size_z, aux, aux,  FFTW_FORWARD, FFTW_MEASURE);
   // crate plan for inverse transform with FFTW
   acc_fftw_bwd[p] = fftw_plan_dft_2d(size_x, size_z, aux, aux, FFTW_BACKWARD, FFTW_MEASURE);

   delete[] aux;

   // Store the index of executing processor for future checking
   acc_id[p] = runtime->get_executing_processor(runtime->get_context()).address_space();
}

// destroyFFTplansTask
/*static*/ const char * const    destroyFFTplansTask::TASK_NAME = "destroyFFTplans";
/*static*/ const int             destroyFFTplansTask::TASK_ID = TID_destroyFFTplans;

void destroyFFTplansTask::cpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 1);
   assert(futures.size() == 0);

   // Accessors for FFT plans
   const AccessorRW<             fftw_plan, 1> acc_fftw_fwd(regions[0], FID_fftw_fwd);
   const AccessorRW<             fftw_plan, 1> acc_fftw_bwd(regions[0], FID_fftw_bwd);
   const AccessorRW<legion_address_space_t, 1> acc_id      (regions[0], FID_id);
   // Get index of the plans that we are destroying
   Point<1> p = Rect<1>(runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[p] == runtime->get_executing_processor(runtime->get_context()).address_space());
   // destroy plan for direct transform with FFTW
   fftw_destroy_plan(acc_fftw_fwd[p]);
   // destroy plan for inverse transform with FFTW
   fftw_destroy_plan(acc_fftw_bwd[p]);
}

// performDirFFTTaskFromField
/*static*/ const char * const    performDirFFTFromFieldTask::TASK_NAME = "performDirFFTFromField";
/*static*/ const int             performDirFFTFromFieldTask::TASK_ID = TID_performDirFFTFromField;

void performDirFFTFromFieldTask::cpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Data accessors
   const AccessorRO<                double, 3> acc_src     (regions[0], FID_src);
   const AccessorWO<       complex<double>, 3> acc_fft     (regions[1], FID_fft);

   // Plans accessors
   const AccessorRO<             fftw_plan, 1> acc_fftw_fwd(regions[2], FID_fftw_fwd);
   const AccessorRO<legion_address_space_t, 1> acc_id      (regions[2], FID_id);

   // Get index of the plans
   Point<1> plan = Rect<1>(runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[plan] == runtime->get_executing_processor(runtime->get_context()).address_space());

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);
   const double FFTfact = 1.0/(size_x*size_z);

   // Allocate a buffer for the FFTW of size_x*size_z for each thread
#ifdef REALM_USE_OPENMP
   fftw_complex *aux = new fftw_complex[(size_x*size_z)*omp_get_max_threads()];
#else
   fftw_complex *aux = new fftw_complex[(size_x*size_z)];
#endif
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int j = bounds.lo.y; j <= bounds.hi.y; j++) {
#ifdef REALM_USE_OPENMP
      fftw_complex *myaux = &aux[(size_x*size_z)*omp_get_thread_num()];
#else
      fftw_complex *myaux = &aux[0];
#endif
      // pack the plane in aux
      for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
         for (int i = bounds.lo.x; i <= bounds.hi.x; i++) {
            myaux[k*size_x+i][0] = acc_src[Point<3>(i, j, k)];
            myaux[k*size_x+i][1] = 0.0;
         }
      // FFT transform
      fftw_execute_dft(acc_fftw_fwd[plan], myaux, myaux);
      // unpack the plane from aux
      for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
         for (int i = bounds.lo.x; i <= bounds.hi.x; i++)
            acc_fft[Point<3>(i, j, k)] = complex<double>(myaux[k*size_x+i][0]*FFTfact, myaux[k*size_x+i][1]*FFTfact);
   }
   delete[] aux;
}

// performDirFFTTaskFromMix
/*static*/ const char * const    performDirFFTFromMixTask::TASK_NAME = "performDirFFTFromMix";
/*static*/ const int             performDirFFTFromMixTask::TASK_ID = TID_performDirFFTFromMix;

void performDirFFTFromMixTask::cpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Data accessors
   const AccessorRO<                double, 3> acc_rho        (regions[0], FID_rho);
   const AccessorRO<                VecNSp, 3> acc_MolarFracs (regions[0], FID_MolarFracs);
   const AccessorWO<       complex<double>, 3> acc_fft        (regions[1], FID_fft);

   // Plans accessors
   const AccessorRO<             fftw_plan, 1> acc_fftw_fwd(regions[2], FID_fftw_fwd);
   const AccessorRO<legion_address_space_t, 1> acc_id      (regions[2], FID_id);

   // Get index of the plans
   Point<1> plan = Rect<1>(runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[plan] == runtime->get_executing_processor(runtime->get_context()).address_space());

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);
   const double FFTfact = 1.0/(size_x*size_z);
   const double SrcFact = -1.0/args.mix.GetDielectricPermittivity();

   // Allocate a buffer for the FFTW of size_x*size_z for each thread
#ifdef REALM_USE_OPENMP
   fftw_complex *aux = new fftw_complex[(size_x*size_z)*omp_get_max_threads()];
#else
   fftw_complex *aux = new fftw_complex[(size_x*size_z)];
#endif
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int j = bounds.lo.y; j <= bounds.hi.y; j++) {
#ifdef REALM_USE_OPENMP
      fftw_complex *myaux = &aux[(size_x*size_z)*omp_get_thread_num()];
#else
      fftw_complex *myaux = &aux[0];
#endif
      // pack the plane in aux
      for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
         for (int i = bounds.lo.x; i <= bounds.hi.x; i++) {
            const Point<3> p = Point<3>(i, j, k);
            const double MixW = args.mix.GetMolarWeightFromXi(acc_MolarFracs[p]);
            myaux[k*size_x+i][0] = SrcFact*
                     args.mix.GetElectricChargeDensity(acc_rho[p], MixW, acc_MolarFracs[p]);
            myaux[k*size_x+i][1] = 0.0;
         }
      // FFT transform
      fftw_execute_dft(acc_fftw_fwd[plan], myaux, myaux);
      // unpack the plane from aux
      for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
         for (int i = bounds.lo.x; i <= bounds.hi.x; i++)
            acc_fft[Point<3>(i, j, k)] = complex<double>(myaux[k*size_x+i][0]*FFTfact, myaux[k*size_x+i][1]*FFTfact);
   }
   delete[] aux;
}

// performInvFFTTask
/*static*/ const char * const    performInvFFTTask::TASK_NAME = "performInvFFT";
/*static*/ const int             performInvFFTTask::TASK_ID = TID_performInvFFT;

void performInvFFTTask::cpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Data accessors
   const AccessorWO<         double, 3> acc_out   (regions[0], FID_out);
   const AccessorRW<complex<double>, 3> acc_fft   (regions[1], FID_fft);

   // Plans accessors
   const AccessorRO<             fftw_plan, 1> acc_fftw_bwd(regions[2], FID_fftw_bwd);
   const AccessorRO<legion_address_space_t, 1> acc_id      (regions[2], FID_id);

   // Get index of the plans
   Point<1> plan = Rect<1>(runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[plan] == runtime->get_executing_processor(runtime->get_context()).address_space());

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);

   // Allocate a buffer for the FFTW of size_x*size_z for each thread
#ifdef REALM_USE_OPENMP
   fftw_complex *aux = new fftw_complex[(size_x*size_z)*omp_get_max_threads()];
#else
   fftw_complex *aux = new fftw_complex[(size_x*size_z)];
#endif
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for
#endif
   for (int j = bounds.lo.y; j <= bounds.hi.y; j++) {
#ifdef REALM_USE_OPENMP
      fftw_complex *myaux = &aux[(size_x*size_z)*omp_get_thread_num()];
#else
      fftw_complex *myaux = &aux[0];
#endif
      // pack the plane in aux
      for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
         for (int i = bounds.lo.x; i <= bounds.hi.x; i++) {
            Point<3> p = Point<3>(i, j, k);
            myaux[k*size_x+i][0] = acc_fft[p].real();
            myaux[k*size_x+i][1] = acc_fft[p].imag();
         }
      // FFT transform
      fftw_execute_dft(acc_fftw_bwd[plan], myaux, myaux);
      // unpack the plane from aux
      for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
         for (int i = bounds.lo.x; i <= bounds.hi.x; i++)
            acc_out[Point<3>(i, j, k)] = myaux[k*size_x+i][0];
   }
   delete[] aux;
}

// performInvFFTTask
/*static*/ const char * const    solveTridiagonalsTask::TASK_NAME = "solveTridiagonals";
/*static*/ const int             solveTridiagonalsTask::TASK_ID = TID_solveTridiagonals;

void solveTridiagonalsTask::cpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Data accessors
   const AccessorRW<complex<double>, 3> acc_fft (regions[0], FID_fft);

   // Tridiagonal coefficients accessors
   const AccessorRO<         double, 1> acc_a   (regions[1], FID_a);
   const AccessorRO<         double, 1> acc_b   (regions[1], FID_b);
   const AccessorRO<         double, 1> acc_c   (regions[1], FID_c);

   // Squared complex wave numbers accessors
   const AccessorRO<complex<double>, 1> acc_k2X (regions[2], FID_k2);
   const AccessorRO<complex<double>, 1> acc_k2Z (regions[3], FID_k2);

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_y = getSize<Ydir>(bounds);

   // Allocate a buffer for the Thomas algorithm of size_y for each thread
#ifdef REALM_USE_OPENMP
   complex<double> *aux = new complex<double>[size_y*omp_get_max_threads()];
#else
   complex<double> *aux = new complex<double>[size_y];
#endif
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = bounds.lo.z; k <= bounds.hi.z; k++)
      for (int i = bounds.lo.x; i <= bounds.hi.x; i++) {
#ifdef REALM_USE_OPENMP
         complex<double> *myaux = &aux[size_y*omp_get_thread_num()];
#else
         complex<double> *myaux = &aux[0];
#endif
         solveTridiagonal(acc_fft, acc_a, acc_b, acc_c, myaux, acc_k2X[i], acc_k2Z[k],
                          i, bounds.lo.y, bounds.hi.y, k, args.Robin_bc);
      }
   delete[] aux;
}

void register_poisson_tasks()
{
#ifdef REALM_USE_CUDA
   // Force the runtime to put the FFT plans in the zero copy memory
   LayoutConstraintID z_copy_memory =
      TaskHelper::register_layout_constraint<MemoryConstraint>(MemoryConstraint(Memory::Z_COPY_MEM));
#endif

   // Yplanes will be allocated with the following directions order YZX
   OrderingConstraint orderYZX(true/*contiguous*/);
   orderYZX.ordering.push_back(DIM_X);
   orderYZX.ordering.push_back(DIM_Z);
   orderYZX.ordering.push_back(DIM_Y);
   orderYZX.ordering.push_back(DIM_F);
   LayoutConstraintID YPlane_Order =
      TaskHelper::register_layout_constraint<OrderingConstraint>(orderYZX);

   // XZslubs will be allocated with the following directions order ZXY
   OrderingConstraint orderZXY(true/*contiguous*/);
   orderZXY.ordering.push_back(DIM_Y);
   orderZXY.ordering.push_back(DIM_X);
   orderZXY.ordering.push_back(DIM_Z);
   orderZXY.ordering.push_back(DIM_F);
   LayoutConstraintID XZslubs_Order =
      TaskHelper::register_layout_constraint<OrderingConstraint>(orderYZX);

   std::vector<std::pair<unsigned, LayoutConstraintID>> cpu_constraints;
   std::vector<std::pair<unsigned, LayoutConstraintID>> gpu_constraints;

   cpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, YPlane_Order));
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, YPlane_Order));
#ifdef REALM_USE_CUDA
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(1, z_copy_memory));
#endif
   TaskHelper::register_hybrid_variants<initFFTplansTask>(cpu_constraints, gpu_constraints);
   cpu_constraints.clear();
   gpu_constraints.clear();

#ifdef REALM_USE_CUDA
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, z_copy_memory));
#endif
   TaskHelper::register_hybrid_variants<destroyFFTplansTask>(cpu_constraints, gpu_constraints);
   gpu_constraints.clear();

   cpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, YPlane_Order));
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, YPlane_Order));
   cpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(1, YPlane_Order));
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(1, YPlane_Order));
#ifdef REALM_USE_CUDA
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(2, z_copy_memory));
#endif
   TaskHelper::register_hybrid_variants<performDirFFTFromFieldTask>(cpu_constraints, gpu_constraints);
   TaskHelper::register_hybrid_variants<performDirFFTFromMixTask  >(cpu_constraints, gpu_constraints);
   TaskHelper::register_hybrid_variants<performInvFFTTask         >(cpu_constraints, gpu_constraints);
   cpu_constraints.clear();
   gpu_constraints.clear();

   cpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, XZslubs_Order));
   gpu_constraints.push_back(std::pair<unsigned, LayoutConstraintID>(0, XZslubs_Order));
   TaskHelper::register_hybrid_variants<solveTridiagonalsTask>();
   cpu_constraints.clear();
   gpu_constraints.clear();
}
