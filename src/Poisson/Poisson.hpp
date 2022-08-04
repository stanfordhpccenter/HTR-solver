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

#ifndef __POISSON_HPP__
#define __POISSON_HPP__

#include "legion.h"

using namespace Legion;

#include "fftw3.h"

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "Poisson.h"

//-----------------------------------------------------------------------------
// TASK THAT INITIALIZE THE FFT PLANS
//-----------------------------------------------------------------------------

class initFFTplansTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion r;
      LogicalRegion s;
      FieldID r_fields[FID_last - 101];
      FieldID s_fields[FID_FFTplans_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
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
// TASK THAT DESTROYS THE FFT PLANS
//-----------------------------------------------------------------------------

class destroyFFTplansTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion r;
      FieldID r_fields[FID_FFTplans_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
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
// TASK THAT PERFORMS THE DIRECT FFT TRANSFORM USING A FIELD AS SOURCE TERM
//-----------------------------------------------------------------------------

class performDirFFTFromFieldTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion r;
      LogicalRegion s;
      LogicalRegion p;
      Mix mix;
      FieldID r_fields[FID_last - 101];
      FieldID s_fields[1];
      FieldID p_fields[FID_FFTplans_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
private:
   // This will need to metch whatever is defined in the rg file
   // For now we keep it for testing
   static const FieldID FID_src = FID_rho;
public:
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
// TASK THAT PERFORMS THE DIRECT FFT TRANSFORM COMPUTING THE SOURCE TERM FROM MIX
//-----------------------------------------------------------------------------

class performDirFFTFromMixTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion r;
      LogicalRegion s;
      LogicalRegion p;
      Mix mix;
      FieldID r_fields[FID_last - 101];
      FieldID s_fields[1];
      FieldID p_fields[FID_FFTplans_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
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
// TASK THAT PERFORMS THE INVERSE FFT TRANSFORM
//-----------------------------------------------------------------------------

class performInvFFTTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion r;
      LogicalRegion s;
      LogicalRegion p;
      FieldID r_fields[FID_last - 101];
      FieldID s_fields[1];
      FieldID p_fields[FID_FFTplans_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
private:
   static const FieldID FID_out = FID_electricPotential;
public:
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
// TASK THAT SOLVES THE TRIDIAGONAL PROBLEM OF THE TRANSFORMED POISSON EQ.
//-----------------------------------------------------------------------------

class solveTridiagonalsTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion r;
      LogicalRegion c;
      LogicalRegion k2X;
      LogicalRegion k2Z;
      bool Robin_bc;
      FieldID r_fields[1];
      FieldID c_fields[FID_CoeffType_last - 101];
      FieldID k2X_fields[1];
      FieldID k2Z_fields[1];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // This function solves the tridiagonal system defined as:
   //    - acc_fft: constains the rhs in input and the solution in output
   //    - acc_a: subdiagonal coefficients
   //    - acc_b: diagonal coefficients
   //    - acc_c: superdiagonal coefficients
   //    - aux: auxiliary vector (is going to be overwritten) needs to be thread safe
   //    - k2X: squared complex wave number in the x direction
   //    - k2Z: squared complex wave number in the z direction
   //    - i: x index
   //    - lo_j: lower y index
   //    - hi_j: higher y index
   //    - k: z index
   //    - Robin_bc : flag to apply Robin BC (Dirichklet only on the mean)
   __CUDA_H__
   static inline void solveTridiagonal(const AccessorRW<complex<double>, 3> acc_fft,
                                       const AccessorRO<         double, 1> acc_a,
                                       const AccessorRO<         double, 1> acc_b,
                                       const AccessorRO<         double, 1> acc_c,
                                       complex<double>      *aux,
                                       const complex<double> k2X,
                                       const complex<double> k2Z,
                                       const int i,
                                       const int lo_j,
                                       const int hi_j,
                                       const int k,
                                       const bool Robin_bc) {
         const bool Neumann = (Robin_bc && ((i != 0) || (k != 0)));
         // solve the tridiagonal system with Thomas
         const Point<3> p = Point<3>(i, lo_j, k);
         complex<double> beta = complex<double>((Neumann) ?  1.0 : acc_b[lo_j], 0.0);
         complex<double> cm1  = complex<double>((Neumann) ? -1.0 : acc_c[lo_j], 0.0);
         acc_fft[p] /= beta;
         // Forward pass
         __UNROLL__
         for (int j = lo_j+1; j < hi_j; j++) {
            const Point<3> p = Point<3>(i, j, k);
            const Point<3> pm1 = Point<3>(i, j-1, k);
            aux[j] = cm1/beta;
            cm1 = complex<double>(acc_c[j], 0.0);
            beta = k2X + k2Z
                   + complex<double>(acc_b[j], 0.0)
                   - complex<double>(acc_a[j], 0.0)*aux[j];
            acc_fft[p] = (acc_fft[p] - complex<double>(acc_a[j], 0.0)*acc_fft[pm1])/beta;
         }
         {
            const Point<3> p = Point<3>(i, hi_j, k);
            const Point<3> pm1 = Point<3>(i, hi_j-1, k);
            const int j = hi_j;
            const double a = (Neumann) ? -1.0 : acc_a[j];
            const double b = (Neumann) ?  1.0 : acc_b[j];
            aux[j] = cm1/beta;
            beta = complex<double>(b, 0.0)
                 - complex<double>(a, 0.0)*aux[j];
            acc_fft[p] = (acc_fft[p] - complex<double>(a, 0.0)*acc_fft[pm1])/beta;
         }
         // Back substitution
         __UNROLL__
         for (int j = hi_j; j > lo_j; j--) {
            const Point<3> p = Point<3>(i, j, k);
            const Point<3> pm1 = Point<3>(i, j-1, k);
            acc_fft[pm1] -= aux[j]*acc_fft[p];
         }
   }
public:
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

#endif // __POISSON_HPP__
