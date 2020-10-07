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

#ifndef __TASK_HELPER_H__
#define __TASK_HELPER_H__

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor< READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

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

#ifndef nSpec
   #error "nSpec is undefined"
#endif

#ifndef nEq
   #error "nEq is undefined"
#endif

template<typename T, int SIZE>
struct MyArray {
public:
   __CUDA_HD__
   inline T& operator[](int index)       {
#ifdef BOUNDS_CHECKS
      assert(index >=   0);
      assert(index < SIZE);
#endif
      return v[index];
   }
   __CUDA_HD__
   inline T  operator[](int index) const {
#ifdef BOUNDS_CHECKS
      assert(index >=   0);
      assert(index < SIZE);
#endif
      return v[index];
   }
public:
   T v[SIZE];
};

typedef MyArray<double,     3> Vec3;
typedef MyArray<double,   nEq> VecNEq;
typedef MyArray<double, nSpec> VecNSp;

enum direction {
   Xdir,
   Ydir,
   Zdir
};

enum side {
   Plus,
   Minus
};

// Utility that computes the size of a Rect<3>
template<direction dir>
__CUDA_HD__
inline coord_t getSize(const Rect<3> bounds);
template<>
__CUDA_HD__
inline coord_t getSize<Xdir>(const Rect<3> bounds) { return bounds.hi.x - bounds.lo.x + 1; };
template<>
__CUDA_HD__
inline coord_t getSize<Ydir>(const Rect<3> bounds) { return bounds.hi.y - bounds.lo.y + 1; };
template<>
__CUDA_HD__
inline coord_t getSize<Zdir>(const Rect<3> bounds) { return bounds.hi.z - bounds.lo.z + 1; };

// Utility that computes the stencil point warping the point aroud periodic boundaries
template<direction dir, side s>
static inline Point<3> warpPeriodic(const Rect<3> bounds, Point<3> p, const coord_t size, const int off);
template<>
__CUDA_HD__
inline Point<3> warpPeriodic<Xdir, Minus>(const Rect<3> bounds, Point<3> p, const coord_t size, const int off) {
   return Point<3>(((p.x + off - bounds.lo.x) % size + size) % size + bounds.lo.x, p.y, p.z);
};
template<>
__CUDA_HD__
inline Point<3> warpPeriodic<Xdir, Plus>(const Rect<3> bounds, Point<3> p, const coord_t size, const int off) {
   return Point<3>((p.x + off - bounds.lo.x) % size + bounds.lo.x, p.y, p.z);
};
template<>
__CUDA_HD__
inline Point<3> warpPeriodic<Ydir, Minus>(const Rect<3> bounds, Point<3> p, const coord_t size, const int off) {
   return Point<3>(p.x, ((p.y + off - bounds.lo.y) % size + size) % size + bounds.lo.y, p.z);
};
template<>
__CUDA_HD__
inline Point<3> warpPeriodic<Ydir, Plus>(const Rect<3> bounds, Point<3> p, const coord_t size, const int off) {
   return Point<3>(p.x, (p.y + off - bounds.lo.y) % size + bounds.lo.y, p.z);
};
template<>
__CUDA_HD__
inline Point<3> warpPeriodic<Zdir, Minus>(const Rect<3> bounds, Point<3> p, const coord_t size, const int off) {
   return Point<3>(p.x, p.y, ((p.z + off - bounds.lo.z) % size + size) % size + bounds.lo.z);
};
template<>
__CUDA_HD__
inline Point<3> warpPeriodic<Zdir, Plus>(const Rect<3> bounds, Point<3> p, const coord_t size, const int off) {
   return Point<3>(p.x, p.y, (p.z + off - bounds.lo.z) % size + bounds.lo.z);
};

// Utility that registers tasks
namespace TaskHelper {
   template<class T>
   void base_cpu_wrapper(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
   {
      assert(task->arglen == 0);
      assert(task->local_arglen == sizeof(typename T::Args));
      const typename T::Args *a = (typename T::Args*)task->local_args;
      T::cpu_base_impl(*a, regions, task->futures, ctx, runtime);
   }

#ifdef LEGION_USE_CUDA
   template<typename T>
   void base_gpu_wrapper(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
   {
      assert(task->arglen == 0);
      assert(task->local_arglen == sizeof(typename T::Args));
      const typename T::Args *a = (typename T::Args*)task->local_args;
      T::gpu_base_impl(*a, regions, task->futures, ctx, runtime);
   }
#endif

   template<typename T>
   void register_hybrid_variants(void)
   {
      {
         TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         Runtime::preregister_task_variant<base_cpu_wrapper<T> >(registrar, T::TASK_NAME);
      }
#ifdef REALM_USE_OPENMP
      {
         TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
         registrar.set_leaf(T::CPU_BASE_LEAF);
         Runtime::preregister_task_variant<base_cpu_wrapper<T> >(registrar, T::TASK_NAME);
      }
#endif
#ifdef LEGION_USE_CUDA
      {
         TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
         registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
         registrar.set_leaf(T::GPU_BASE_LEAF);
         Runtime::preregister_task_variant<base_gpu_wrapper<T> >(registrar, T::TASK_NAME);
      }
#endif
   }
};

#endif // __TASK_HELPER_H__

