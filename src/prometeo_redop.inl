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

#include "prometeo_types.h"
#include "task_helper.hpp"

#ifndef __CUDA_H__
#ifdef __CUDACC__
#define __CUDA_H__ __device__
#else
#define __CUDA_H__
#endif
#endif

#ifndef __UNROLL__
#ifdef __CUDACC__
#define __UNROLL__ #pragma unroll
#else
#define __UNROLL__
#endif
#endif

namespace Legion {
//-----------------------------------------------------------------------------
// SumReduction for Vec3
//-----------------------------------------------------------------------------

template<> template<> __CUDA_HD__ inline
void SumReduction<Vec3>::apply<true>(LHS &lhs, RHS rhs) {
   lhs += rhs;
}

template<> template<> __CUDA_HD__ inline
void SumReduction<Vec3>::apply<false>(LHS &lhs, RHS rhs) {
   #ifdef __CUDA_ARCH__
   #if __CUDA_ARCH__ >= 600
      __UNROLL__
      for (int i=0; i<3; i++)
         atomicAdd(&lhs[i], rhs[i]);
   #else
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++) {
         unsigned long long *target = (unsigned long long *)&lhs[i];
         newval.as_int = *target;
         do {
            oldval.as_int = newval.as_int;
            newval.as_float += rhs[i];
            newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
         } while (oldval.as_int != newval.as_int);
      }
   #endif
   #else
      // No atomic floating point operations so use compare and swap
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++) {
         volatile unsigned long long *target = (unsigned long long *)&lhs[i];
         do {
            oldval.as_int = *target;
            newval.as_float = oldval.as_float + rhs[i];
         } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
      }
   #endif
}

template<> template<> __CUDA_HD__ inline
void SumReduction<Vec3>::fold<true>(RHS &rhs1, RHS rhs2) {
   rhs1 += rhs2;
}

template<> template<> __CUDA_HD__ inline
void SumReduction<Vec3>::fold<false>(RHS &rhs1, RHS rhs2) {
   #ifdef __CUDA_ARCH__
   #if __CUDA_ARCH__ >= 600
      __UNROLL__
      for (int i=0; i<3; i++)
         atomicAdd(&rhs1[i], rhs2[i]);
   #else
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++) {
         unsigned long long *target = (unsigned long long *)&rhs1[i];
         newval.as_int = *target;
         do {
            oldval.as_int = newval.as_int;
            newval.as_float += rhs2[i];
            newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
         } while (oldval.as_int != newval.as_int);
      }
   #endif
   #else
      // No atomic floating point operations so use compare and swap
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++) {
         volatile unsigned long long *target = (unsigned long long *)&rhs1[i];
         do {
            oldval.as_int = *target;
            newval.as_float = oldval.as_float + rhs2[i];
         } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
      }
   #endif
}

//-----------------------------------------------------------------------------
// SumReduction for VecNSp
//-----------------------------------------------------------------------------

template<> template<> __CUDA_HD__ inline
void SumReduction<VecNSp>::apply<true>(LHS &lhs, RHS rhs) {
   lhs += rhs;
}

template<> template<> __CUDA_HD__ inline
void SumReduction<VecNSp>::apply<false>(LHS &lhs, RHS rhs) {
   #ifdef __CUDA_ARCH__
   #if __CUDA_ARCH__ >= 600
      __UNROLL__
      for (int i=0; i<nSpec; i++)
         atomicAdd(&lhs[i], rhs[i]);
   #else
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         unsigned long long *target = (unsigned long long *)&lhs[i];
         newval.as_int = *target;
         do {
            oldval.as_int = newval.as_int;
            newval.as_float += rhs[i];
            newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
         } while (oldval.as_int != newval.as_int);
      }
   #endif
   #else
      // No atomic floating point operations so use compare and swap
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         volatile unsigned long long *target = (unsigned long long *)&lhs[i];
         do {
            oldval.as_int = *target;
            newval.as_float = oldval.as_float + rhs[i];
         } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
      }
   #endif
}

template<> template<> __CUDA_HD__ inline
void SumReduction<VecNSp>::fold<true>(RHS &rhs1, RHS rhs2) {
   rhs1 += rhs2;
}

template<> template<> __CUDA_HD__ inline
void SumReduction<VecNSp>::fold<false>(RHS &rhs1, RHS rhs2) {
   #ifdef __CUDA_ARCH__
   #if __CUDA_ARCH__ >= 600
      __UNROLL__
      for (int i=0; i<nSpec; i++)
         atomicAdd(&rhs1[i], rhs2[i]);
   #else
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         unsigned long long *target = (unsigned long long *)&rhs1[i];
         newval.as_int = *target;
         do {
            oldval.as_int = newval.as_int;
            newval.as_float += rhs2[i];
            newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
         } while (oldval.as_int != newval.as_int);
      }
   #endif
   #else
      // No atomic floating point operations so use compare and swap
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         volatile unsigned long long *target = (unsigned long long *)&rhs1[i];
         do {
            oldval.as_int = *target;
            newval.as_float = oldval.as_float + rhs2[i];
         } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
      }
   #endif
}

//-----------------------------------------------------------------------------
// SumReduction for MySymMatrix<double, 3>
//-----------------------------------------------------------------------------

template<> template<> __CUDA_HD__ inline
void SumReduction<MySymMatrix<double, 3>>::apply<true>(LHS &lhs, RHS rhs) {
   lhs += rhs;
}

template<> template<> __CUDA_HD__ inline
void SumReduction<MySymMatrix<double, 3>>::apply<false>(LHS &lhs, RHS rhs) {
   #ifdef __CUDA_ARCH__
   #if __CUDA_ARCH__ >= 600
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++)
            atomicAdd(&lhs(i,j), rhs(i,j));
   #else
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            unsigned long long *target = (unsigned long long *)&lhs(i,j);
            newval.as_int = *target;
            do {
               oldval.as_int = newval.as_int;
               newval.as_float += rhs(i,j);
               newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
            } while (oldval.as_int != newval.as_int);
         }
   #endif
   #else
      // No atomic floating point operations so use compare and swap
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            volatile unsigned long long *target = (unsigned long long *)&lhs(i,j);
            do {
               oldval.as_int = *target;
               newval.as_float = oldval.as_float + rhs(i,j);
            } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
         }
   #endif
}

template<> template<> __CUDA_HD__ inline
void SumReduction<MySymMatrix<double, 3>>::fold<true>(RHS &rhs1, RHS rhs2) {
   rhs1 += rhs2;
}

template<> template<> __CUDA_HD__ inline
void SumReduction<MySymMatrix<double, 3>>::fold<false>(RHS &rhs1, RHS rhs2) {
   #ifdef __CUDA_ARCH__
   #if __CUDA_ARCH__ >= 600
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++)
            atomicAdd(&rhs1(i,j), rhs2(i,j));
   #else
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            unsigned long long *target = (unsigned long long *)&rhs1(i,j);
            newval.as_int = *target;
            do {
               oldval.as_int = newval.as_int;
               newval.as_float += rhs2(i,j);
               newval.as_int = atomicCAS(target, oldval.as_int, newval.as_int);
            } while (oldval.as_int != newval.as_int);
         }
   #endif
   #else
      // No atomic floating point operations so use compare and swap
      union { unsigned long long as_int; double as_float; } oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            volatile unsigned long long *target = (unsigned long long *)&rhs1(i,j);
            do {
               oldval.as_int = *target;
               newval.as_float = oldval.as_float + rhs2(i,j);
            } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
         }
   #endif
}

};

