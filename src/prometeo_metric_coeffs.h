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

#ifndef __PROMETEO_METRIC_COEFFS_H__
#define __PROMETEO_METRIC_COEFFS_H__

#include <stdint.h>

#ifndef __CUDA_CONST__
#ifdef __CUDACC__
#define __CUDA_CONST__ __device__ __constant__
#else
#define __CUDA_CONST__
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Stencil indices
#define Stencil1  0
#define Stencil2  1
#define Stencil3  2
#define Stencil4  3
#define nStencils  4

// Node types
#define Std_node   0  // Node with standard stencil
#define L_S_node   1  // Left node on staggered bc
#define Lp1_S_node 2  // Left plus one node on staggered bc
#define Lp2_S_node 3  // Left plus two node on staggered bc
#define Rm3_S_node 4  // Right minus three node on staggered bc
#define Rm2_S_node 5  // Right minus two node on staggered bc
#define Rm1_S_node 6  // Right minus one node on staggered bc
#define R_S_node   7  // Right node on staggered bc
#define L_C_node   8  // Left node on collocated bc
#define Lp1_C_node 9  // Left plus one node on collocated bc
#define Rm2_C_node 10 // Right minus two node on collocated bc
#define Rm1_C_node 11 // Right minus one node on collocated bc
#define R_C_node   12 // Right node on collocated bc

/////////////////////////////////
// Constants for CPU tasks
/////////////////////////////////

// Indices offsets
extern const int8_t Cp_cpu[13][5];

// Face reconstruction operators [c-2, ..., c+3]
extern const double Recon_Plus_cpu [13][24];
extern const double Recon_Minus_cpu[13][24];

// Blending coefficients to obtain sixth order reconstruction
extern const double Coeffs_Plus_cpu[13][4];
extern const double Coeffs_Minus_cpu[13][4];

// Staggered interpolation operator [c, c+1]
extern const float  Interp_cpu[13][2];

// Cell-center gradient operator [c - c-1, c+1 - c]
extern const float  Grad_cpu[13][2];

// Order of the Kennedy reconstruction scheme
extern const int8_t KennedyOrder_cpu[13];
extern const double KennedyCoeff_cpu[13][3];
extern const int8_t KennedyNSum_cpu[13];

// Store powers of 10 in a vector
extern const float p10_cpu[14];

#ifdef LEGION_USE_CUDA
/////////////////////////////////
// Constants for CUDA tasks
/////////////////////////////////

// Indices offsets
extern __CUDA_CONST__ int8_t Cp_gpu[13][5];

// Face reconstruction operators [c-2, ..., c+3]
extern __CUDA_CONST__ double Recon_Plus_gpu [13][24];
extern __CUDA_CONST__ double Recon_Minus_gpu[13][24];

// Blending coefficients to obtain sixth order reconstruction
extern __CUDA_CONST__ double Coeffs_Plus_gpu[13][4];
extern __CUDA_CONST__ double Coeffs_Minus_gpu[13][4];

// Staggered interpolation operator [c, c+1]
extern __CUDA_CONST__ float  Interp_gpu[13][2];

// Cell-center gradient operator [c - c-1, c+1 - c]
extern __CUDA_CONST__ float  Grad_gpu[13][2];

// Order of the Kennedy reconstruction scheme
extern __CUDA_CONST__ int8_t KennedyOrder_gpu[13];
extern __CUDA_CONST__ double KennedyCoeff_gpu[13][3];
extern __CUDA_CONST__ int8_t KennedyNSum_gpu[13];

// Store powers of 10 in a vector
extern __CUDA_CONST__ float p10_gpu[14];

#endif

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
   #define Cp           Cp_gpu
   #define Recon_Plus   Recon_Plus_gpu
   #define Recon_Minus  Recon_Minus_gpu
   #define Coeffs_Plus  Coeffs_Plus_gpu
   #define Coeffs_Minus Coeffs_Minus_gpu
   #define Interp       Interp_gpu
   #define Grad         Grad_gpu
   #define KennedyOrder KennedyOrder_gpu
   #define KennedyCoeff KennedyCoeff_gpu
   #define KennedyNSum  KennedyNSum_gpu
   #define p10          p10_gpu
#else
   #define Cp           Cp_cpu
   #define Recon_Plus   Recon_Plus_cpu
   #define Recon_Minus  Recon_Minus_cpu
   #define Coeffs_Plus  Coeffs_Plus_cpu
   #define Coeffs_Minus Coeffs_Minus_cpu
   #define Interp       Interp_cpu
   #define Grad         Grad_cpu
   #define KennedyOrder KennedyOrder_cpu
   #define KennedyCoeff KennedyCoeff_cpu
   #define KennedyNSum  KennedyNSum_cpu
   #define p10          p10_cpu
#endif

#endif // __PROMETEO_METRIC_COEFFS_H__
