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
      // Type punning like this is illegal in C++ but the
      // CUDA manual has an example just like it so fuck it
      __UNROLL__
      for (int i=0; i<3; i++) {
         double newval = lhs[i], oldval;
         unsigned long long int *ptr = (unsigned long long int*)&lhs[i];
         do {
            oldval = newval;
            newval += rhs[i];
            newval = __ulonglong_as_double(atomicCAS(ptr,
                                           __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
         } while (oldval != newval);
      }
   #endif
   #else
   #if __cplusplus >= 202002L
      __UNROLL__
      for (int i=0; i<3; i++) {
         std::atomic_ref<double> atomic(lhs[i]);
         double oldval = atomic.load();
         double newval;
         do {
            newval = oldval + rhs[i];
         } while (!atomic.compare_exchange_weak(oldval, newval));
      }
   #else
      // No atomic floating point operations so use compare and swap
      TypePunning::Alias<int64_t,double> oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++) {
         TypePunning::Pointer<int64_t> pointer((void*)&lhs[i]);
         do {
            oldval.load(pointer);
            newval = oldval.as_two() + rhs[i];
         } while (!__sync_bool_compare_and_swap((int64_t*)pointer,
                                                oldval.as_one(), newval.as_one()));
      }
   #endif
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
      __UNROLL__
      for (int i=0; i<3; i++) {
         double newval = rhs1[i], oldval;
         unsigned long long int *ptr = (unsigned long long int*)&rhs1[i];
         do {
            oldval = newval;
            newval += rhs2[i];
            newval = __ulonglong_as_double(atomicCAS(ptr,
                     __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
         } while (oldval != newval);
      }
   #endif
   #else
   #if __cplusplus >= 202002L
      __UNROLL__
      for (int i=0; i<3; i++) {
         std::atomic_ref<double> atomic(rhs1[i]);
         double oldval = atomic.load();
         double newval;
         do {
            newval = oldval + rhs2[i];
         } while (!atomic.compare_exchange_weak(oldval, newval));
      }
   #else
      // No atomic floating point operations so use compare and swap
      TypePunning::Alias<int64_t,double> oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++) {
         TypePunning::Pointer<int64_t> pointer((void*)&rhs1[i]);
         do {
            oldval.load(pointer);
            newval = oldval.as_two() + rhs2[i];
         } while (!__sync_bool_compare_and_swap((int64_t*)pointer,
                                                oldval.as_one(), newval.as_one()));
      }
   #endif
   #endif
}

//-----------------------------------------------------------------------------
// SumReduction for VecNSp
//-----------------------------------------------------------------------------

#if nSpec != 3

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
      // Type punning like this is illegal in C++ but the
      // CUDA manual has an example just like it so fuck it
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         double newval = lhs[i], oldval;
         unsigned long long int *ptr = (unsigned long long int*)&lhs[i];
         do {
            oldval = newval;
            newval += rhs[i];
            newval = __ulonglong_as_double(atomicCAS(ptr,
                     __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
         } while (oldval != newval);
      }
   #endif
   #else
   #if __cplusplus >= 202002L
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         std::atomic_ref<double> atomic(lhs[i]);
         double oldval = atomic.load();
         double newval;
         do {
            newval = oldval + rhs[i];
         } while (!atomic.compare_exchange_weak(oldval, newval));
      }
   #else
      // No atomic floating point operations so use compare and swap
      TypePunning::Alias<int64_t,double> oldval, newval;
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         TypePunning::Pointer<int64_t> pointer((void*)&lhs[i]);
         do {
            oldval.load(pointer);
            newval = oldval.as_two() + rhs[i];
         } while (!__sync_bool_compare_and_swap((int64_t*)pointer,
                                                oldval.as_one(), newval.as_one()));
      }
   #endif
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
      // Type punning like this is illegal in C++ but the
      // CUDA manual has an example just like it so fuck it
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         double newval = rhs1[i], oldval;
         unsigned long long int *ptr = (unsigned long long int*)&rhs1[i];
         do {
            oldval = newval;
            newval += rhs2[i];
            newval = __ulonglong_as_double(atomicCAS(ptr,
                     __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
         } while (oldval != newval);
      }
   #endif
   #else
   #if __cplusplus >= 202002L
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         std::atomic_ref<double> atomic(rhs1[i]);
         double oldval = atomic.load();
         double newval;
         do {
            newval = oldval + rhs2[i];
         } while (!atomic.compare_exchange_weak(oldval, newval));
      }
   #else
      // No atomic floating point operations so use compare and swap
      TypePunning::Alias<int64_t,double> oldval, newval;
      __UNROLL__
      for (int i=0; i<nSpec; i++) {
         TypePunning::Pointer<int64_t> pointer((void*)&rhs1[i]);
         do {
            oldval.load(pointer);
            newval = oldval.as_two() + rhs2[i];
         } while (!__sync_bool_compare_and_swap((int64_t*)pointer,
                                                oldval.as_one(), newval.as_one()));
      }
   #endif
   #endif
}

#endif

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
      // Type punning like this is illegal in C++ but the
      // CUDA manual has an example just like it so fuck it
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            double newval = lhs(i,j), oldval;
            unsigned long long int *ptr = (unsigned long long int*)&lhs(i,j);
            do {
               oldval = newval;
               newval += rhs(i,j);
               newval = __ulonglong_as_double(atomicCAS(ptr,
                        __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
            } while (oldval != newval);
         }
   #endif
   #else
   #if __cplusplus >= 202002L
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
         std::atomic_ref<double> atomic(lhs(i,j));
         double oldval = atomic.load();
         double newval;
         do {
            newval = oldval + rhs(i,j);
         } while (!atomic.compare_exchange_weak(oldval, newval));
      }
   #else
      // No atomic floating point operations so use compare and swap
      TypePunning::Alias<int64_t,double> oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            TypePunning::Pointer<int64_t> pointer((void*)&lhs(i,j));
            do {
               oldval.load(pointer);
               newval = oldval.as_two() + rhs(i,j);
            } while (!__sync_bool_compare_and_swap((int64_t*)pointer,
                                                   oldval.as_one(), newval.as_one()));
         }
   #endif
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
      // Type punning like this is illegal in C++ but the
      // CUDA manual has an example just like it so fuck it
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            double newval = rhs1(i,j), oldval;
            unsigned long long int *ptr = (unsigned long long int*)&rhs1(i,j);
            do {
               oldval = newval;
               newval += rhs2(i,j);
               newval = __ulonglong_as_double(atomicCAS(ptr,
                        __double_as_ulonglong(oldval), __double_as_ulonglong(newval)));
            } while (oldval != newval);
         }
   #endif
   #else
   #if __cplusplus >= 202002L
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            std::atomic_ref<double> atomic(rhs1(i,j));
            double oldval = atomic.load();
            double newval;
            do {
               newval = oldval + rhs2(i,j);
            } while (!atomic.compare_exchange_weak(oldval, newval));
         }
   #else
      // No atomic floating point operations so use compare and swap
      TypePunning::Alias<int64_t,double> oldval, newval;
      __UNROLL__
      for (int i=0; i<3; i++)
         __UNROLL__
         for (int j=i; j<3; j++) {
            TypePunning::Pointer<int64_t> pointer((void*)&rhs1(i,j));
            do {
               oldval.load(pointer);
               newval = oldval.as_two() + rhs2(i,j);
            } while (!__sync_bool_compare_and_swap((int64_t*)pointer,
                                                   oldval.as_one(), newval.as_one()));
         }
   #endif
   #endif
}

};

