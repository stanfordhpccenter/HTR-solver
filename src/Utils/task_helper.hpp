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

#ifndef __TASK_HELPER_HPP__
#define __TASK_HELPER_HPP__

#include "legion.h"

#include "my_array.hpp"

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

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

//-----------------------------------------------------------------------------
// Reduction operations
//-----------------------------------------------------------------------------
namespace Legion {
   template<typename T, int SIZE>
   class SumReduction< MyArray<T, SIZE> > {
   public:
      typedef MyArray<T, SIZE> LHS;
      typedef MyArray<T, SIZE> RHS;

      static constexpr double identity = 0.0;
//      static const int REDOP_ID;

      template<bool EXCLUSIVE> __CUDA_HD__
      static void apply(LHS &lhs, RHS rhs);
      template<bool EXCLUSIVE> __CUDA_HD__
      static void fold(RHS &rhs1, RHS rhs2);
   };

   template<typename T, int SIZE>
   class SumReduction< MySymMatrix<T, SIZE> > {
   public:
      typedef MySymMatrix<T, SIZE> LHS;
      typedef MySymMatrix<T, SIZE> RHS;

      static constexpr double identity = 0.0;
//      static const int REDOP_ID;

      template<bool EXCLUSIVE> __CUDA_HD__
      static void apply(LHS &lhs, RHS rhs);
      template<bool EXCLUSIVE> __CUDA_HD__
      static void fold(RHS &rhs1, RHS rhs2);
   };
};

//-----------------------------------------------------------------------------
// Accessors
//-----------------------------------------------------------------------------
template<typename FT, int N, typename T = coord_t> using AccessorRO = Legion::FieldAccessor< READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = Legion::FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = Legion::FieldAccessor<WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T> >;
template<typename FT, int N, bool EXCLUSIVE = false, typename T = coord_t> using AccessorSumRD = Legion::ReductionAccessor<SumReduction<FT>, EXCLUSIVE, N, T, Realm::AffineAccessor<FT, N, T> >;

//-----------------------------------------------------------------------------
// Utility that registers tasks
//-----------------------------------------------------------------------------
namespace TaskHelper {
   template<typename T>
   void base_cpu_wrapper(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *runtime)
   {
      typename T::Args *a;
      if (task->arglen == 0) {
         assert(task->local_arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->local_args;
      } else {
         assert(task->local_arglen == 0);
         assert(task->arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->args;
      }
      T::cpu_base_impl(*a, regions, task->futures, ctx, runtime);
   }

   template<typename T, typename R>
   R return_cpu_wrapper(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx, Legion::Runtime *runtime)
   {
      typename T::Args *a;
      if (task->arglen == 0) {
         assert(task->local_arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->local_args;
      } else {
         assert(task->local_arglen == 0);
         assert(task->arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->args;
      }
      return T::cpu_base_impl(*a, regions, task->futures, ctx, runtime);
   }

#ifdef LEGION_USE_CUDA
   template<typename T>
   void base_gpu_wrapper(const Legion::Task *task,
                         const std::vector<Legion::PhysicalRegion> &regions,
                         Legion::Context ctx, Legion::Runtime *runtime)
   {
      typename T::Args *a;
      if (task->arglen == 0) {
         assert(task->local_arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->local_args;
      } else {
         assert(task->local_arglen == 0);
         assert(task->arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->args;
      }
      T::gpu_base_impl(*a, regions, task->futures, ctx, runtime);
   }

   template<typename T, typename R>
   R return_gpu_wrapper(const Legion::Task *task,
                        const std::vector<Legion::PhysicalRegion> &regions,
                        Legion::Context ctx, Legion::Runtime *runtime)
   {
      typename T::Args *a;
      if (task->arglen == 0) {
         assert(task->local_arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->local_args;
      } else {
         assert(task->local_arglen == 0);
         assert(task->arglen == sizeof(typename T::Args));
         a = (typename T::Args*)task->args;
      }
      return T::gpu_base_impl(*a, regions, task->futures, ctx, runtime);
   }
#endif

   template<typename T>
   LayoutConstraintID register_layout_constraint(T c)
   {
      LayoutConstraintRegistrar registrar;
      registrar.add_constraint(c);
      return Runtime::preregister_layout(registrar);
   }

   template<typename T>
   void register_hybrid_variants(void)
   {
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         Legion::Runtime::preregister_task_variant<base_cpu_wrapper<T>>(registrar, T::TASK_NAME);
      }
#ifdef REALM_USE_OPENMP
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Legion::Processor::OMP_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         Legion::Runtime::preregister_task_variant<base_cpu_wrapper<T>>(registrar, T::TASK_NAME);
      }
#endif
#ifdef LEGION_USE_CUDA
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Legion::Processor::TOC_PROC));
         registrar.set_leaf(T::GPU_BASE_LEAF);
         Legion::Runtime::preregister_task_variant<base_gpu_wrapper<T>>(registrar, T::TASK_NAME);
      }
#endif
   }

   template<typename T>
   void register_hybrid_variants(const std::vector<std::pair<unsigned, LayoutConstraintID>> &cpu_layout_const,
                                 const std::vector<std::pair<unsigned, LayoutConstraintID>> &gpu_layout_const)
   {
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         for (unsigned i=0; i<cpu_layout_const.size(); i++)
            registrar.add_layout_constraint_set(cpu_layout_const[i].first, cpu_layout_const[i].second);
         Legion::Runtime::preregister_task_variant<base_cpu_wrapper<T>>(registrar, T::TASK_NAME);
      }
#ifdef REALM_USE_OPENMP
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Legion::Processor::OMP_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         for (unsigned i=0; i<cpu_layout_const.size(); i++)
            registrar.add_layout_constraint_set(cpu_layout_const[i].first, cpu_layout_const[i].second);
         Legion::Runtime::preregister_task_variant<base_cpu_wrapper<T>>(registrar, T::TASK_NAME);
      }
#endif
#ifdef LEGION_USE_CUDA
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Legion::Processor::TOC_PROC));
         registrar.set_leaf(T::GPU_BASE_LEAF);
         for (unsigned i=0; i<gpu_layout_const.size(); i++)
            registrar.add_layout_constraint_set(gpu_layout_const[i].first, gpu_layout_const[i].second);
         Legion::Runtime::preregister_task_variant<base_gpu_wrapper<T>>(registrar, T::TASK_NAME);
      }
#endif
   }

   template<typename T, typename R, typename R_GPU>
   void register_hybrid_variants(void)
   {
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         Legion::Runtime::preregister_task_variant<R, return_cpu_wrapper<T, R>>(registrar, T::TASK_NAME);
      }
#ifdef REALM_USE_OPENMP
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Legion::Processor::OMP_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         Legion::Runtime::preregister_task_variant<R, return_cpu_wrapper<T, R>>(registrar, T::TASK_NAME);
      }
#endif
#ifdef LEGION_USE_CUDA
      {
         Legion::TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Legion::Processor::TOC_PROC));
         registrar.set_leaf(T::GPU_BASE_LEAF);
         Legion::Runtime::preregister_task_variant<R_GPU, return_gpu_wrapper<T, R_GPU>>(registrar, T::TASK_NAME);
      }
#endif
   }

};

#endif // __TASK_HELPER_HPP__

