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

#include "prometeo_metric_coeffs.h"
#include "prometeo_metric_coeffs_macros.h"

//-----------------------------------------
// Assemble vectors with coefficients
//----------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// Indices offsets
__CUDA_CONST__ int8_t Cp_gpu[][5]  = {
/*   [Std_node  ] =*/   Std_Cp,
/*   [L_S_node  ] =*/   L_S_Cp,
/*   [Lp1_S_node] =*/ Lp1_S_Cp,
/*   [Lp2_S_node] =*/ Lp2_S_Cp,
/*   [Rm3_S_node] =*/ Rm3_S_Cp,
/*   [Rm2_S_node] =*/ Rm2_S_Cp,
/*   [Rm1_S_node] =*/ Rm1_S_Cp,
/*   [R_S_node  ] =*/   R_S_Cp,
/*   [L_C_node  ] =*/   L_C_Cp,
/*   [Lp1_C_node] =*/ Lp1_C_Cp,
/*   [Rm2_C_node] =*/ Rm2_C_Cp,
/*   [Rm1_C_node] =*/ Rm1_C_Cp,
/*   [R_C_node  ] =*/   R_C_Cp,
};

// Face reconstruction operators [c-2, ..., c+3]
__CUDA_CONST__ double Recon_Plus_gpu[][24]  = {
/*   [  Std_node] =*/   Std_Recon_Plus,
/*   [  L_S_node] =*/   L_S_Recon_Plus,
/*   [Lp1_S_node] =*/ Lp1_S_Recon_Plus,
/*   [Lp2_S_node] =*/ Lp2_S_Recon_Plus,
/*   [Rm3_S_node] =*/ Rm3_S_Recon_Plus,
/*   [Rm2_S_node] =*/ Rm2_S_Recon_Plus,
/*   [Rm1_S_node] =*/ Rm1_S_Recon_Plus,
/*   [  R_S_node] =*/   R_S_Recon_Plus,
/*   [  L_C_node] =*/   L_C_Recon_Plus,
/*   [Lp1_C_node] =*/ Lp1_C_Recon_Plus,
/*   [Rm2_C_node] =*/ Rm2_C_Recon_Plus,
/*   [Rm1_C_node] =*/ Rm1_C_Recon_Plus,
/*   [  R_C_node] =*/   R_C_Recon_Plus,
};

__CUDA_CONST__ double Recon_Minus_gpu[][24]  = {
/*   [  Std_node] =*/   Std_Recon_Minus,
/*   [  L_S_node] =*/   L_S_Recon_Minus,
/*   [Lp1_S_node] =*/ Lp1_S_Recon_Minus,
/*   [Lp2_S_node] =*/ Lp2_S_Recon_Minus,
/*   [Rm3_S_node] =*/ Rm3_S_Recon_Minus,
/*   [Rm2_S_node] =*/ Rm2_S_Recon_Minus,
/*   [Rm1_S_node] =*/ Rm1_S_Recon_Minus,
/*   [  R_S_node] =*/   R_S_Recon_Minus,
/*   [  L_C_node] =*/   L_C_Recon_Minus,
/*   [Lp1_C_node] =*/ Lp1_C_Recon_Minus,
/*   [Rm2_C_node] =*/ Rm2_C_Recon_Minus,
/*   [Rm1_C_node] =*/ Rm1_C_Recon_Minus,
/*   [  R_C_node] =*/   R_C_Recon_Minus,
};

// Blending coefficients to obtain sixth order reconstruction
__CUDA_CONST__ double Coeffs_Plus_gpu[][4] = {
/*   [  Std_node] =*/   Std_Coeffs_Plus,
/*   [  L_S_node] =*/   L_S_Coeffs_Plus,
/*   [Lp1_S_node] =*/ Lp1_S_Coeffs_Plus,
/*   [Lp2_S_node] =*/ Lp2_S_Coeffs_Plus,
/*   [Rm3_S_node] =*/ Rm3_S_Coeffs_Plus,
/*   [Rm2_S_node] =*/ Rm2_S_Coeffs_Plus,
/*   [Rm1_S_node] =*/ Rm1_S_Coeffs_Plus,
/*   [  R_S_node] =*/   R_S_Coeffs_Plus,
/*   [  L_C_node] =*/   L_C_Coeffs_Plus,
/*   [Lp1_C_node] =*/ Lp1_C_Coeffs_Plus,
/*   [Rm2_C_node] =*/ Rm2_C_Coeffs_Plus,
/*   [Rm1_C_node] =*/ Rm1_C_Coeffs_Plus,
/*   [  R_C_node] =*/   R_C_Coeffs_Plus,
};

__CUDA_CONST__ double Coeffs_Minus_gpu[][4] = {
/*   [  Std_node] =*/   Std_Coeffs_Minus,
/*   [  L_S_node] =*/   L_S_Coeffs_Minus,
/*   [Lp1_S_node] =*/ Lp1_S_Coeffs_Minus,
/*   [Lp2_S_node] =*/ Lp2_S_Coeffs_Minus,
/*   [Rm3_S_node] =*/ Rm3_S_Coeffs_Minus,
/*   [Rm2_S_node] =*/ Rm2_S_Coeffs_Minus,
/*   [Rm1_S_node] =*/ Rm1_S_Coeffs_Minus,
/*   [  R_S_node] =*/   R_S_Coeffs_Minus,
/*   [  L_C_node] =*/   L_C_Coeffs_Minus,
/*   [Lp1_C_node] =*/ Lp1_C_Coeffs_Minus,
/*   [Rm2_C_node] =*/ Rm2_C_Coeffs_Minus,
/*   [Rm1_C_node] =*/ Rm1_C_Coeffs_Minus,
/*   [  R_C_node] =*/   R_C_Coeffs_Minus
};

// Staggered interpolation operator [c, c+1]
__CUDA_CONST__ float  Interp_gpu[][2] = {
/*   [  Std_node] =*/ {0.5, 0.5},
/*   [  L_S_node] =*/ {1.0, 0.0},
/*   [Lp1_S_node] =*/ {0.5, 0.5},
/*   [Lp2_S_node] =*/ {0.5, 0.5},
/*   [Rm3_S_node] =*/ {0.5, 0.5},
/*   [Rm2_S_node] =*/ {0.5, 0.5},
/*   [Rm1_S_node] =*/ {0.0, 1.0},
/*   [  R_S_node] =*/ {0.5, 0.5},
/*   [  L_C_node] =*/ {0.5, 0.5},
/*   [Lp1_C_node] =*/ {0.5, 0.5},
/*   [Rm2_C_node] =*/ {0.5, 0.5},
/*   [Rm1_C_node] =*/ {0.5, 0.5},
/*   [  R_C_node] =*/ {0.5, 0.5}
};

// Cell-center gradient operator [c - c-1, c+1 - c]
__CUDA_CONST__ float  Grad_gpu[][2] = {
/*   [  Std_node] =*/ {0.5, 0.5},
/*   [  L_S_node] =*/ {0.0, 2.0},
/*   [Lp1_S_node] =*/ {1.0, 0.5},
/*   [Lp2_S_node] =*/ {0.5, 0.5},
/*   [Rm3_S_node] =*/ {0.5, 0.5},
/*   [Rm2_S_node] =*/ {0.5, 0.5},
/*   [Rm1_S_node] =*/ {0.5, 1.0},
/*   [  R_S_node] =*/ {2.0, 0.0},
/*   [  L_C_node] =*/ {0.0, 1.0},
/*   [Lp1_C_node] =*/ {0.5, 0.5},
/*   [Rm2_C_node] =*/ {0.5, 0.5},
/*   [Rm1_C_node] =*/ {0.5, 0.5},
/*   [  R_C_node] =*/ {1.0, 0.0}
};

// Order of the Kennedy reconstruction scheme
__CUDA_CONST__ int8_t KennedyOrder_gpu[] = {
/*   [  Std_node] =*/   Std_KennedyOrder,
/*   [  L_S_node] =*/   L_S_KennedyOrder,
/*   [Lp1_S_node] =*/ Lp1_S_KennedyOrder,
/*   [Lp2_S_node] =*/ Lp2_S_KennedyOrder,
/*   [Rm3_S_node] =*/ Rm3_S_KennedyOrder,
/*   [Rm2_S_node] =*/ Rm2_S_KennedyOrder,
/*   [Rm1_S_node] =*/ Rm1_S_KennedyOrder,
/*   [  R_S_node] =*/   R_S_KennedyOrder,
/*   [  L_C_node] =*/   L_C_KennedyOrder,
/*   [Lp1_C_node] =*/ Lp1_C_KennedyOrder,
/*   [Rm2_C_node] =*/ Rm2_C_KennedyOrder,
/*   [Rm1_C_node] =*/ Rm1_C_KennedyOrder,
/*   [  R_C_node] =*/   R_C_KennedyOrder
};

__CUDA_CONST__ double KennedyCoeff_gpu[][3] = {
/*   [  Std_node] =*/   Std_KennedyCoeff,
/*   [  L_S_node] =*/   L_S_KennedyCoeff,
/*   [Lp1_S_node] =*/ Lp1_S_KennedyCoeff,
/*   [Lp2_S_node] =*/ Lp2_S_KennedyCoeff,
/*   [Rm3_S_node] =*/ Rm3_S_KennedyCoeff,
/*   [Rm2_S_node] =*/ Rm2_S_KennedyCoeff,
/*   [Rm1_S_node] =*/ Rm1_S_KennedyCoeff,
/*   [  R_S_node] =*/   R_S_KennedyCoeff,
/*   [  L_C_node] =*/   L_C_KennedyCoeff,
/*   [Lp1_C_node] =*/ Lp1_C_KennedyCoeff,
/*   [Rm2_C_node] =*/ Rm2_C_KennedyCoeff,
/*   [Rm1_C_node] =*/ Rm1_C_KennedyCoeff,
/*   [  R_C_node] =*/   R_C_KennedyCoeff
};

__CUDA_CONST__ int8_t KennedyNSum_gpu[] = {
/*   [  Std_node] =*/ 3,
/*   [  L_S_node] =*/ 0,
/*   [Lp1_S_node] =*/ 3,
/*   [Lp2_S_node] =*/ 3,
/*   [Rm3_S_node] =*/ 2,
/*   [Rm2_S_node] =*/ 1,
/*   [Rm1_S_node] =*/ 0,
/*   [  R_S_node] =*/ 0,
/*   [  L_C_node] =*/ 3,
/*   [Lp1_C_node] =*/ 3,
/*   [Rm2_C_node] =*/ 2,
/*   [Rm1_C_node] =*/ 1,
/*   [  R_C_node] =*/ 0
};

// Powers of 10
__CUDA_CONST__  float p10[] = {     1,  1e-1,  1e-2,  1e-3,  1e-4,  1e-5,  1e-6,  1e-7,  1e-8,  1e-9,
                                1e-10, 1e-11, 1e-12, 1e-13};

#ifdef __cplusplus
}
#endif

//-----------------------------------------
// Clean-up
//----------------------------------------

#undef   Std_Cp
#undef   L_S_Cp
#undef Lp1_S_Cp
#undef Lp2_S_Cp
#undef Rm3_S_Cp
#undef Rm2_S_Cp
#undef Rm1_S_Cp
#undef   R_S_Cp
#undef   L_C_Cp
#undef Lp1_C_Cp
#undef Rm2_C_Cp
#undef Rm1_C_Cp
#undef   R_C_Cp

#undef   Std_Recon_Plus
#undef   L_S_Recon_Plus
#undef Lp1_S_Recon_Plus
#undef Lp2_S_Recon_Plus
#undef Rm3_S_Recon_Plus
#undef Rm2_S_Recon_Plus
#undef Rm1_S_Recon_Plus
#undef   R_S_Recon_Plus
#undef   L_C_Recon_Plus
#undef Lp1_C_Recon_Plus
#undef Rm2_C_Recon_Plus
#undef Rm1_C_Recon_Plus
#undef   R_C_Recon_Plus

#undef   Std_Recon_Minus
#undef   L_S_Recon_Minus
#undef Lp1_S_Recon_Minus
#undef Lp2_S_Recon_Minus
#undef Rm3_S_Recon_Minus
#undef Rm2_S_Recon_Minus
#undef Rm1_S_Recon_Minus
#undef   R_S_Recon_Minus
#undef   L_C_Recon_Minus
#undef Lp1_C_Recon_Minus
#undef Rm2_C_Recon_Minus
#undef Rm1_C_Recon_Minus
#undef   R_C_Recon_Minus

#undef   Std_Coeffs_Plus
#undef   L_S_Coeffs_Plus
#undef Lp1_S_Coeffs_Plus
#undef Lp2_S_Coeffs_Plus
#undef Rm3_S_Coeffs_Plus
#undef Rm2_S_Coeffs_Plus
#undef Rm1_S_Coeffs_Plus
#undef   R_S_Coeffs_Plus
#undef   L_C_Coeffs_Plus
#undef Lp1_C_Coeffs_Plus
#undef Rm2_C_Coeffs_Plus
#undef Rm1_C_Coeffs_Plus
#undef   R_C_Coeffs_Plus

#undef   Std_Coeffs_Minus
#undef   L_S_Coeffs_Minus
#undef Lp1_S_Coeffs_Minus
#undef Lp2_S_Coeffs_Minus
#undef Rm3_S_Coeffs_Minus
#undef Rm2_S_Coeffs_Minus
#undef Rm1_S_Coeffs_Minus
#undef   R_S_Coeffs_Minus
#undef   L_C_Coeffs_Minus
#undef Lp1_C_Coeffs_Minus
#undef Rm2_C_Coeffs_Minus
#undef Rm1_C_Coeffs_Minus
#undef   R_C_Coeffs_Minus

#undef   Std_KennedyOrder
#undef   L_S_KennedyOrder
#undef Lp1_S_KennedyOrder
#undef Lp2_S_KennedyOrder
#undef Rm3_S_KennedyOrder
#undef Rm2_S_KennedyOrder
#undef Rm1_S_KennedyOrder
#undef   R_S_KennedyOrder
#undef   L_C_KennedyOrder
#undef Lp1_C_KennedyOrder
#undef Rm2_C_KennedyOrder
#undef Rm1_C_KennedyOrder
#undef   R_C_KennedyOrder

#undef   Std_KennedyCoeff
#undef   L_S_KennedyCoeff
#undef Lp1_S_KennedyCoeff
#undef Lp2_S_KennedyCoeff
#undef Rm3_S_KennedyCoeff
#undef Rm2_S_KennedyCoeff
#undef Rm1_S_KennedyCoeff
#undef   R_S_KennedyCoeff
#undef   L_C_KennedyCoeff
#undef Lp1_C_KennedyCoeff
#undef Rm2_C_KennedyCoeff
#undef Rm1_C_KennedyCoeff
#undef   R_C_KennedyCoeff

