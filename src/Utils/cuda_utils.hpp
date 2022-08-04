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

#ifndef __CUDA_UTILS_HPP__
#define __CUDA_UTILS_HPP__

#include "legion.h"

#include "PointDomain_helper.hpp"

//-----------------------------------------------------------------------------
// Utility that splits the number of threads per block along each direction for the 3d kernel
//-----------------------------------------------------------------------------
__host__
inline int findHighestPower2(const int num, const int m = 8) {
   assert(num > 0);
   // start from 2^0
   int j = 0;
   for (int i= 0; i<=m; i++) { if (1<<i > num) break; j = i;}
   return j;
}

template<direction dir>
__host__
inline dim3 splitThreadsPerBlock(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_x = getSize<Xdir>(bounds);
   const Legion::coord_t size_y = getSize<Ydir>(bounds);
   const Legion::coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB));
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB/r.x));
   r.z = 1 << findHighestPower2(size_z, findHighestPower2(TPB/r.x/r.y));
   return r;
};

//-----------------------------------------------------------------------------
// This utility splits the threads per block on a cross plane
//-----------------------------------------------------------------------------
template<direction dir>
__host__
inline dim3 splitThreadsPerBlockPlane(const int TPB, const Legion::Rect<3> bounds);
template<>
__host__
inline dim3 splitThreadsPerBlockPlane<Xdir>(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_y = getSize<Ydir>(bounds);
   const Legion::coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   // Y has priority...
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB));
   // ... then Z ...
   r.z = 1 << findHighestPower2(size_z, findHighestPower2(TPB/r.y));
   // ... and X gets 1
   r.x = 1;
   return r;
};
template<>
__host__
inline dim3 splitThreadsPerBlockPlane<Ydir>(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_x = getSize<Xdir>(bounds);
   const Legion::coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   // X has priority...
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB));
   // ... then Z ...
   r.z = 1 << findHighestPower2(size_z, findHighestPower2(TPB/r.x));
   // ... and Y gets 1
   r.y = 1;
   return r;
};
template<>
__host__
inline dim3 splitThreadsPerBlockPlane<Zdir>(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_x = getSize<Xdir>(bounds);
   const Legion::coord_t size_y = getSize<Ydir>(bounds);
   dim3 r;
   // X has priority...
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB));
   // ... then Y ...
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB/r.x));
   // ... and Z gets 1
   r.z = 1;
   return r;
};

//-----------------------------------------------------------------------------
// This utility splits the threads per block privileging a cross plane and distributes
// the remaining available threads on the normal direction determined by the template variable
//-----------------------------------------------------------------------------
template<direction dir>
__host__
inline dim3 splitThreadsPerBlockSpan(const int TPB, const Legion::Rect<3> bounds);
template<>
__host__
inline dim3 splitThreadsPerBlockSpan<Xdir>(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_x = getSize<Xdir>(bounds);
   const Legion::coord_t size_y = getSize<Ydir>(bounds);
   const Legion::coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   // Y has priority...
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB));
   // ... then Z ...
   r.z = 1 << findHighestPower2(size_z, findHighestPower2(TPB/r.y));
   // ... and finally X
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB/r.y/r.z));
   return r;
};
template<>
__host__
inline dim3 splitThreadsPerBlockSpan<Ydir>(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_x = getSize<Xdir>(bounds);
   const Legion::coord_t size_y = getSize<Ydir>(bounds);
   const Legion::coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   // X has priority...
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB));
   // ... then Z ...
   r.z = 1 << findHighestPower2(size_z, findHighestPower2(TPB/r.x));
   // ... and finally Y
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB/r.x/r.z));
   return r;
};
template<>
__host__
inline dim3 splitThreadsPerBlockSpan<Zdir>(const int TPB, const Legion::Rect<3> bounds) {
   assert(TPB%2 == 0);
   const Legion::coord_t size_x = getSize<Xdir>(bounds);
   const Legion::coord_t size_y = getSize<Ydir>(bounds);
   const Legion::coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   // X has priority...
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB));
   // ... then Y ...
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB/r.x));
   // ... and finally Z
   r.z = 1 << findHighestPower2(size_z, findHighestPower2(TPB/r.x/r.y));
   return r;
};

//-----------------------------------------------------------------------------
// This utility determines the number of blocks in a kernel launch with span loop
//-----------------------------------------------------------------------------
template<direction dir>
__host__
inline dim3 numBlocksSpan(const dim3 TPB, const Legion::Rect<3> bounds);
template<>
__host__
inline dim3 numBlocksSpan<Xdir>(const dim3 TPB, const Legion::Rect<3> bounds) {
   // Needs only one thread along X
   return dim3(1,
               (getSize<Ydir>(bounds) + (TPB.y - 1)) / TPB.y,
               (getSize<Zdir>(bounds) + (TPB.z - 1)) / TPB.z);
};
template<>
__host__
inline dim3 numBlocksSpan<Ydir>(const dim3 TPB, const Legion::Rect<3> bounds) {
   // Needs only one thread along Y
   return dim3((getSize<Xdir>(bounds) + (TPB.x - 1)) / TPB.x,
               1,
               (getSize<Zdir>(bounds) + (TPB.z - 1)) / TPB.z);
};
template<>
__host__
inline dim3 numBlocksSpan<Zdir>(const dim3 TPB, const Legion::Rect<3> bounds) {
   // Needs only one thread along Z
   return dim3((getSize<Xdir>(bounds) + (TPB.x - 1)) / TPB.x,
               (getSize<Ydir>(bounds) + (TPB.y - 1)) / TPB.y,
               1);
};

//-----------------------------------------------------------------------------
// This utility checks that the thread is inside the crossPlane determined by (sx, sy, sz)
//-----------------------------------------------------------------------------
template<direction dir>
__device__
inline bool isThreadInCrossPlane(const Legion::coord_t sx, const Legion::coord_t sy, const Legion::coord_t sz);
template<>
__device__
inline bool isThreadInCrossPlane<Xdir>(const Legion::coord_t sx, const Legion::coord_t sy, const Legion::coord_t sz) {
   assert(blockIdx.x == 0);
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   return ((y < sy) && (z < sz));
};
template<>
__device__
inline bool isThreadInCrossPlane<Ydir>(const Legion::coord_t sx, const Legion::coord_t sy, const Legion::coord_t sz) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   assert(blockIdx.y == 0);
   int z = blockIdx.z * blockDim.z + threadIdx.z;
   return ((x < sx) && (z < sz));
};
template<>
__device__
inline bool isThreadInCrossPlane<Zdir>(const Legion::coord_t sx, const Legion::coord_t sy, const Legion::coord_t sz) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   assert(blockIdx.z == 0);
   return ((x < sx) && (y < sy));
};

//-----------------------------------------------------------------------------
// This utility computes the first index of a thread in a span determined by size
//-----------------------------------------------------------------------------
template<direction dir>
__device__
inline Legion::coord_t firstIndexInSpan(const Legion::coord_t size);
template<>
__device__
inline Legion::coord_t firstIndexInSpan<Xdir>(const Legion::coord_t size) {
   return threadIdx.x*((size + (blockDim.x-1))/blockDim.x);
};
template<>
__device__
inline Legion::coord_t firstIndexInSpan<Ydir>(const Legion::coord_t size) {
   return threadIdx.y*((size + (blockDim.y-1))/blockDim.y);
};
template<>
__device__
inline Legion::coord_t firstIndexInSpan<Zdir>(const Legion::coord_t size) {
   return threadIdx.z*((size + (blockDim.z-1))/blockDim.z);
};

//-----------------------------------------------------------------------------
// This utility computes the last index of a thread in a span determined by size
//-----------------------------------------------------------------------------
template<direction dir>
__device__
inline Legion::coord_t lastIndexInSpan(const Legion::coord_t size);
template<>
__device__
inline Legion::coord_t lastIndexInSpan<Xdir>(const Legion::coord_t size) {
   return min((threadIdx.x+1)*((size + (blockDim.x-1))/blockDim.x), size);
};
template<>
__device__
inline Legion::coord_t lastIndexInSpan<Ydir>(const Legion::coord_t size) {
   return min((threadIdx.y+1)*((size + (blockDim.y-1))/blockDim.y), size);
};
template<>
__device__
inline Legion::coord_t lastIndexInSpan<Zdir>(const Legion::coord_t size) {
   return min((threadIdx.z+1)*((size + (blockDim.z-1))/blockDim.z), size);
};

//-----------------------------------------------------------------------------
// This utility generates a list of cuda streams and performs a round-robin
//-----------------------------------------------------------------------------
template<int N>
class streamsRR {
public:
   streamsRR() {
      idx = 0;
      for (int i = 0; i < N; i++)
         cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
   };

   ~streamsRR() {
      for (int i = 0; i < N; i++)
         cudaStreamDestroy(s[i]);
   };

   cudaStream_t operator++() {
      idx = (idx + 1)%N; 
      return s[idx];
   }

private:
   int idx;
   cudaStream_t s[N];
};

//-----------------------------------------------------------------------------
// These utilities perform butterfly reduction across threads of a warp
//-----------------------------------------------------------------------------
__device__
inline void reduceSum(double my_data, const Legion::DeferredBuffer<double, 1> &buffer) {
   // We know there is never more than 32 warps in a CTA
   __shared__ double trampoline[32];

   // make sure that everyone is in sync
   __syncthreads();

   // Perform a local reduction inside the CTA
   // Butterfly reduction across all threads in all warps
   for (int i = 16; i >= 1; i/=2)
      my_data += __shfl_xor_sync(0xfffffff, my_data, i, 32);
   unsigned laneid;
   asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
   unsigned warpid = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) >> 5;
   // First thread in each warp writes out all values
   if (laneid == 0)
      trampoline[warpid] = my_data;
   __syncthreads();

   // Butterfly reduction across all thread in the first warp
   if (warpid == 0) {
      unsigned numwarps = (blockDim.x * blockDim.y * blockDim.z) >> 5;
      my_data = (laneid < numwarps) ? trampoline[laneid] : 0;
      for (int i = 16; i >= 1; i/=2)
         my_data += __shfl_xor_sync(0xfffffff, my_data, i, 32);
      // First thread writes to the buffer
      if (laneid == 0) {
         unsigned blockId = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
         buffer[blockId] = my_data;
      }
   }
};

__device__
inline void reduceMax(double my_data, const Legion::DeferredBuffer<double, 1> &buffer) {
   // We know there is never more than 32 warps in a CTA
   __shared__ double trampoline[32];

   // make sure that everyone is in sync
   __syncthreads();

   // Perform a local reduction inside the CTA
   // Butterfly reduction across all threads in all warps
   for (int i = 16; i >= 1; i/=2)
      my_data = max(my_data, __shfl_xor_sync(0xfffffff, my_data, i, 32));
   unsigned laneid;
   asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
   unsigned warpid = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) >> 5;
   // First thread in each warp writes out all values
   if (laneid == 0)
      trampoline[warpid] = my_data;
   __syncthreads();

   // Butterfly reduction across all thread in the first warp
   if (warpid == 0) {
      unsigned numwarps = (blockDim.x * blockDim.y * blockDim.z) >> 5;
      my_data = (laneid < numwarps) ? trampoline[laneid] : 0;
      for (int i = 16; i >= 1; i/=2)
         my_data = max(my_data, __shfl_xor_sync(0xfffffff, my_data, i, 32));
      // First thread writes to the buffer
      if (laneid == 0) {
         unsigned blockId = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
         buffer[blockId] = my_data;
      }
   }
};

//-----------------------------------------------------------------------------
// These utilities perform butterfly reduction across a DeferedBuffer
//-----------------------------------------------------------------------------
__global__
void ReduceBufferSum_kernel(const Legion::DeferredBuffer<double, 1> buffer,
                            const Legion::DeferredValue<double> result,
                            const size_t size);
__global__
void ReduceBufferMax_kernel(const Legion::DeferredBuffer<double, 1> buffer,
                            const Legion::DeferredValue<double> result,
                            const size_t size);

#endif // __CUDA_UTILS_HPP__

