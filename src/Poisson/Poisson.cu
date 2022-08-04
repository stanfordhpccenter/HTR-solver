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
#include "cuda_utils.hpp"
#include "cufft.h"

// Declare a constant memory that will hold the Mixture struct (initialized in prometeo_mixture.cu)
extern __device__ __constant__ Mix mix;

//-----------------------------------------------------------------------------
// KERNEL FOR initFFTplansTask
//-----------------------------------------------------------------------------

void initFFTplansTask::gpu_base_impl(
                     const Args &args,
                     const std::vector<PhysicalRegion> &regions,
                     const std::vector<Future>         &futures,
                     Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for FFT plans
   const AccessorRW<             fftw_plan, 1> acc_fftw_fwd(regions[1], FID_fftw_fwd);
   const AccessorRW<             fftw_plan, 1> acc_fftw_bwd(regions[1], FID_fftw_bwd);
   const AccessorRW<           cufftHandle, 1> acc_cufft   (regions[1], FID_cufft);
   const AccessorRW<legion_address_space_t, 1> acc_id      (regions[1], FID_id);

   // Get size of the FFT execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_y = getSize<Ydir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);

   // Get index of the plans that we are initializing
   Point<1> p = Rect<1>(runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space())).lo;

   // Init FFTW plans
   fftw_make_planner_thread_safe();
   fftw_complex *aux = new fftw_complex[(size_x*size_z)];
   // crate plan for direct transform with FFTW
   acc_fftw_fwd[p] = fftw_plan_dft_2d(size_x, size_z, aux, aux,  FFTW_FORWARD, FFTW_MEASURE);
   // crate plan for inverse transform with FFTW
   acc_fftw_bwd[p] = fftw_plan_dft_2d(size_x, size_z, aux, aux, FFTW_BACKWARD, FFTW_MEASURE);
   delete[] aux;

   // Init cuFFT plans
   int dim[2] = {int(size_z), int(size_x)};
   if (cufftPlanMany(acc_cufft.ptr(p), 2, dim,
                     NULL, 1, 0,
                     NULL, 1, 0,
                     CUFFT_Z2Z, int(size_y)) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to create plan\n");
      assert(0);
   }

   // Store the index of executing processor for future checking
   acc_id[p] = runtime->get_executing_processor(runtime->get_context()).address_space();
}

// destroyFFTplansTask
void destroyFFTplansTask::gpu_base_impl(
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
   const AccessorRW<           cufftHandle, 1> acc_cufft   (regions[0], FID_cufft);
   const AccessorRW<legion_address_space_t, 1> acc_id      (regions[0], FID_id);
   // Get index of the plans that we are destroying
   Point<1> p = Rect<1>(runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[p] == runtime->get_executing_processor(runtime->get_context()).address_space());
   // destroy plan for direct transform with FFTW
   fftw_destroy_plan(acc_fftw_fwd[p]);
   // destroy plan for inverse transform with FFTW
   fftw_destroy_plan(acc_fftw_bwd[p]);
   // destroy plan for transform with cuFFT
   cufftDestroy(acc_cufft[p]);
}

//-----------------------------------------------------------------------------
// KERNEL FOR performDirFFTTask
//-----------------------------------------------------------------------------

__global__
void updateRHS_kernel(const AccessorWO<complex<double>, 3>        fft,
                      const AccessorRO<double, 3>                   q,
                      const  double Srcfact,
                      const Rect<3> my_bounds,
                      const coord_t size_x,
                      const coord_t size_y,
                      const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      fft[p] = complex<double>(Srcfact*q[p], 0.0);
   }
}

__global__
void updateRHS_kernel(const AccessorWO<complex<double>, 3>        fft,
                      const AccessorRO<double, 3>                 rho,
                      const AccessorRO<VecNSp, 3>          MolarFracs,
                      const  double Srcfact,
                      const Rect<3> my_bounds,
                      const coord_t size_x,
                      const coord_t size_y,
                      const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      const double MixW = mix.GetMolarWeightFromXi(MolarFracs[p]);
      fft[p] = complex<double>(Srcfact*mix.GetElectricChargeDensity(rho[p], MixW, MolarFracs[p]), 0.0);
   }
}

__host__
void performDirFFTFromFieldTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Data accessors
   const AccessorRO<                double, 3> acc_src  (regions[0], FID_src);
   const AccessorWO<       complex<double>, 3> acc_fft  (regions[1], FID_fft);

   // Plans accessors
   const AccessorRO<           cufftHandle, 1> acc_cufft(regions[2], FID_cufft);
   const AccessorRO<legion_address_space_t, 1> acc_id   (regions[2], FID_id);

   // Get index of the plans
   Point<1> plan = Rect<1>(runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[plan] == runtime->get_executing_processor(runtime->get_context()).address_space());

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_y = getSize<Ydir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);
   const double FFTfact = 1.0/(size_x*size_z);

   // Force cuFFT to run on the default stream of this task
   cudaStream_t default_stream;
   cudaStreamCreate(&default_stream);

   // Store data to be transformed in a deferred buffer
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, bounds);
   const dim3 num_blocks_3d = dim3((size_x + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (size_y + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (size_z + (TPB_3d.z - 1)) / TPB_3d.z);
   updateRHS_kernel<<<num_blocks_3d, TPB_3d, 0, default_stream>>>(acc_fft, acc_src,
                                                                  FFTfact, bounds,
                                                                  size_x, size_y, size_z);

   // Perform the FFT (on the correct stream)
   if (cufftSetStream(acc_cufft[plan], default_stream) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to associate stream to plan\n");
      assert(0);
   }
   if (cufftExecZ2Z(acc_cufft[plan],
                    (cufftDoubleComplex*)(acc_fft.ptr(bounds.lo)),
                    (cufftDoubleComplex*)(acc_fft.ptr(bounds.lo)), CUFFT_FORWARD) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
      assert(0);
   }

   // Cleanup default stream
   cudaStreamDestroy(default_stream);
}

__host__
void performDirFFTFromMixTask::gpu_base_impl(
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
   const AccessorRO<           cufftHandle, 1> acc_cufft(regions[2], FID_cufft);
   const AccessorRO<legion_address_space_t, 1> acc_id   (regions[2], FID_id);

   // Get index of the plans
   Point<1> plan = Rect<1>(runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[plan] == runtime->get_executing_processor(runtime->get_context()).address_space());

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_y = getSize<Ydir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);
   const double FFTfact = 1.0/(size_x*size_z);
   const double SrcFact = -1.0/args.mix.GetDielectricPermittivity();

   // Force cuFFT to run on the default stream of this task
   cudaStream_t default_stream;
   cudaStreamCreate(&default_stream);

   // Update RHS
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, bounds);
   const dim3 num_blocks_3d = dim3((size_x + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (size_y + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (size_z + (TPB_3d.z - 1)) / TPB_3d.z);
   updateRHS_kernel<<<num_blocks_3d, TPB_3d, 0, default_stream>>>(acc_fft, acc_rho, acc_MolarFracs,
                                                                  FFTfact*SrcFact, bounds,
                                                                  size_x, size_y, size_z);

   // Perform the FFT (on the correct stream)
   if (cufftSetStream(acc_cufft[plan], default_stream) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to associate stream to plan\n");
      assert(0);
   }
   if (cufftExecZ2Z(acc_cufft[plan],
                    (cufftDoubleComplex*)(acc_fft.ptr(bounds.lo)),
                    (cufftDoubleComplex*)(acc_fft.ptr(bounds.lo)), CUFFT_FORWARD) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
      assert(0);
   }

   // Cleanup default stream
   cudaStreamDestroy(default_stream);
}

//-----------------------------------------------------------------------------
// KERNEL FOR performInvFFTTask
//-----------------------------------------------------------------------------
__global__
void unpackData_kernel(const AccessorRW<complex<double>, 3> fft,
                       const AccessorWO<double, 3>          Phi,
                       const Rect<3> my_bounds,
                       const coord_t size_x,
                       const coord_t size_y,
                       const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   if ((x < size_x) && (y < size_y) && (z < size_z)) {
      const Point<3> p = Point<3>(x + my_bounds.lo.x,
                                  y + my_bounds.lo.y,
                                  z + my_bounds.lo.z);
      Phi[p] = fft[p].real();
   }
}

__host__
void performInvFFTTask::gpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 3);
   assert(futures.size() == 0);

   // Data accessors
   const AccessorWO<         double, 3> acc_out         (regions[0], FID_out);
   const AccessorRW<complex<double>, 3> acc_fft         (regions[1], FID_fft);

   // Plans accessors
   const AccessorRO<           cufftHandle, 1> acc_cufft(regions[2], FID_cufft);
   const AccessorRO<legion_address_space_t, 1> acc_id   (regions[2], FID_id);

   // Get index of the plans
   Point<1> plan = Rect<1>(runtime->get_index_space_domain(ctx, regions[2].get_logical_region().get_index_space())).lo;
   // check that we are on the right processor
   assert(acc_id[plan] == runtime->get_executing_processor(runtime->get_context()).address_space());

   // Get execution domain
   Rect<3> bounds = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_y = getSize<Ydir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);

   // Force cuFFT to run on the default stream of this task
   cudaStream_t default_stream;
   cudaStreamCreate(&default_stream);

   DeferredBuffer<cufftDoubleComplex, 1> fft(Rect<1>(0, size_x*size_y*size_z), Memory::GPU_FB_MEM);

   // Perform the inverse FFT (on the correct stream)
   if (cufftSetStream(acc_cufft[plan], default_stream) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT Error: Unable to associate stream to plan\n");
      assert(0);
   }
   if (cufftExecZ2Z(acc_cufft[plan],
                    (cufftDoubleComplex*)(acc_fft.ptr(bounds.lo)),
                    (cufftDoubleComplex*)(acc_fft.ptr(bounds.lo)), CUFFT_INVERSE) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: ExecZ2Z Inverse failed");
      assert(0);
   }

   // Retrieve output data
   const int threads_per_block = 256;
   const dim3 TPB_3d = splitThreadsPerBlock<Xdir>(threads_per_block, bounds);
   const dim3 num_blocks_3d = dim3((size_x + (TPB_3d.x - 1)) / TPB_3d.x,
                                   (size_y + (TPB_3d.y - 1)) / TPB_3d.y,
                                   (size_z + (TPB_3d.z - 1)) / TPB_3d.z);
   unpackData_kernel<<<num_blocks_3d, TPB_3d, 0, default_stream>>>(acc_fft, acc_out, bounds,
                                                                   size_x, size_y, size_z);

   // Cleanup default stream
   cudaStreamDestroy(default_stream);
}

//-----------------------------------------------------------------------------
// KERNEL FOR solveTridiagonalsTask
//-----------------------------------------------------------------------------

__global__
void solveTridiagonal_kernel(const AccessorRW<complex<double>, 3> fft,
                             const AccessorRO<         double, 1> a,
                             const AccessorRO<         double, 1> b,
                             const AccessorRO<         double, 1> c,
                             const AccessorRO<complex<double>, 1> k2X,
                             const AccessorRO<complex<double>, 1> k2Z,
                             DeferredBuffer<complex<double>, 1> aux,
                             const bool Robin_bc,
                             const Rect<3> my_bounds,
                             const coord_t size_x,
                             const coord_t size_y,
                             const coord_t size_z)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   //int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   if ((x < size_x) && (z < size_z)) {
      const coord_t i = x + my_bounds.lo.x;
      const coord_t k = z + my_bounds.lo.z;
      solveTridiagonalsTask::solveTridiagonal(fft, a, b, c, aux.ptr((z*size_x + x)*size_y),
                           k2X[i], k2Z[k], i, my_bounds.lo.y, my_bounds.hi.y, k, Robin_bc);
   }
}

__host__
void solveTridiagonalsTask::gpu_base_impl(
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
   coord_t size_x = getSize<Xdir>(bounds);
   coord_t size_y = getSize<Ydir>(bounds);
   coord_t size_z = getSize<Zdir>(bounds);

   // Get kernel launch domain
   const Rect<2> bounds2d = Rect<2>(Point<2>(bounds.lo.x, bounds.lo.z),
                                    Point<2>(bounds.hi.x, bounds.hi.z));

   // use a deferred buffer to store auxiliary data of the Thomas algorithm
   DeferredBuffer<complex<double>, 1> aux(Rect<1>(0, size_x*size_y*size_z), Memory::GPU_FB_MEM);

   // Solve tridiagonals with a 2d launch
   const int threads_per_block = 256;
   const dim3 TPB_2d = splitThreadsPerBlockPlane<Ydir>(threads_per_block, bounds);
   const dim3 num_blocks_2d = numBlocksSpan<Ydir>(TPB_2d, bounds);
   solveTridiagonal_kernel<<<num_blocks_2d, TPB_2d>>>(acc_fft, acc_a, acc_b, acc_c,
                                                      acc_k2X, acc_k2Z, aux, args.Robin_bc, bounds,
                                                      size_x, size_y, size_z);
}
