// Copyright (c) "2021, by Centre Européen de Recherche et de Formation Avancée en Calcul Scientifiq
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

#include "cuda_utils.hpp"

//-----------------------------------------------------------------------------
// These utilities perform butterfly reduction across a DeferedBuffer
//-----------------------------------------------------------------------------
__global__
void ReduceBufferSum_kernel(const Legion::DeferredBuffer<double, 1> buffer,
                            const Legion::DeferredValue<double> result,
                            const size_t size) {
   // We know there is never more than 32 warps in a CTA
   __shared__ double trampoline[32];

   // Each thread reduces all the correspoinding values
   int offset = threadIdx.x;
   double my_r = 0.0; // Spectral radius cannot be lower than 0
   while (offset < size) {
      my_r += buffer[Legion::Point<1>(offset)];
      offset += blockDim.x;
   }
   // make sure that everyone is done with its reduction
   __syncthreads();

   // Perform a local reduction inside the CTA
   // Butterfly reduction across all threads in all warps
   for (int i = 16; i >= 1; i/=2)
      my_r += __shfl_xor_sync(0xfffffff, my_r, i, 32);
   unsigned laneid;
   asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
   unsigned warpid = threadIdx.x >> 5;
   // First thread in each warp writes out all values
   if (laneid == 0)
      trampoline[warpid] = my_r;
   __syncthreads();

   // Butterfly reduction across all threads in the first warp
   if (warpid == 0) {
      unsigned numwarps = blockDim.x >> 5;
      my_r = (laneid < numwarps) ? trampoline[laneid] : 0;
      for (int i = 16; i >= 1; i/=2)
         my_r += __shfl_xor_sync(0xfffffff, my_r, i, 32);
      // First thread writes to the buffer
      if (laneid == 0)
         result.write(my_r);
   }
}

__global__
void ReduceBufferMax_kernel(const Legion::DeferredBuffer<double, 1> buffer,
                            const Legion::DeferredValue<double> result,
                            const size_t size) {
   // We know there is never more than 32 warps in a CTA
   __shared__ double trampoline[32];

   // Each thread reduces all the correspoinding values
   int offset = threadIdx.x;
   double my_r = 0.0; // Spectral radius cannot be lower than 0
   while (offset < size) {
      my_r = max(my_r, buffer[Legion::Point<1>(offset)]);
      offset += blockDim.x;
   }
   // make sure that everyone is done with its reduction
   __syncthreads();

   // Perform a local reduction inside the CTA
   // Butterfly reduction across all threads in all warps
   for (int i = 16; i >= 1; i/=2)
      my_r = max(my_r, __shfl_xor_sync(0xfffffff, my_r, i, 32));
   unsigned laneid;
   asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
   unsigned warpid = threadIdx.x >> 5;
   // First thread in each warp writes out all values
   if (laneid == 0)
      trampoline[warpid] = my_r;
   __syncthreads();

   // Butterfly reduction across all threads in the first warp
   if (warpid == 0) {
      unsigned numwarps = blockDim.x >> 5;
      my_r = (laneid < numwarps) ? trampoline[laneid] : 0;
      for (int i = 16; i >= 1; i/=2)
         my_r = max(my_r, __shfl_xor_sync(0xfffffff, my_r, i, 32));
      // First thread writes to the buffer
      if (laneid == 0)
         result.write(my_r);
   }
}

