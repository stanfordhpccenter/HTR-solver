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

#ifndef __CUDA_CONST__
#ifdef __CUDACC__
#define __CUDA_CONST__ __constant__
#else
#define __CUDA_CONST__
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

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

//-----------------------------------------------------------------------------
// STANDARD NODE
//-----------------------------------------------------------------------------
//
// dxm2:     |-----------------------------------------|
// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
// dxp2:                                               |---------------------------|
// dxp3:                                               |-----------------------------------------|
//                 c-2           c-1            c     x=0    c+1           c+2           c+3
//           |------x------|------x------|------x------|------x------|------x------|------x------|
//
// Plus reconstruction:
// 1st:                           o-------------o-----> <-----o
// 2nd:                                         o-----> <-----o-------------o
// 3rd:             o-------------o-------------o----->
// 4th:                                         o-----> <-----o-------------o-------------o
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o-------------o
// 2nd:                           o-------------o-----> <-----o
// 3rd:                                                 <-----o-------------o-------------o
// 4th:             o-------------o-------------o-----> <-----o


#define Std_Cp {   -2,           -1,                          1,            2,            3}

#define Std_Recon_Plus { \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                 2.0/6.0,     -7.0/6.0,     11.0/6.0,        0.0,          0.0,          0.0, \
                   0.0,          0.0,        3.0/12.0,    13.0/12.0,    -5.0/12.0,     1.0/12.0}

#define Std_Recon_Minus { \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0,   \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0,   \
                   0.0,          0.0,          0.0,       11.0/6.0,     -7.0/6.0,      2.0/6.0, \
                 1.0/12.0,    -5.0/12.0,    13.0/12.0,     3.0/12.0,       0.0,          0.0}

#define Std_Coeffs_Plus  {9.0/20.0, 6.0/20.0, 1.0/20.0, 4.0/20.0}
#define Std_Coeffs_Minus {9.0/20.0, 6.0/20.0, 1.0/20.0, 4.0/20.0}

//const double Std_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//for i=1,6 do
//   for St =1,nStencils do
//      Std_Recon[i] = Std_Recon[i] + Std_Recon_Plus[(St-1)*6+i]*Std_Coeffs_Plus[St]
//   end
//end

#define Std_KennedyOrder  3 // Sixth-order
#define Std_KennedyCoeff {3.0/4.0, -3.0/20.0, 1.0/60.0}

//-----------------------------------------------------------------------------
// STAGGERED LEFT BC
//-----------------------------------------------------------------------------

// Boundary node is staggered on the face so we do not need any reconstruction
#define L_S_Recon_Plus  {  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0}
#define L_S_Recon_Minus {  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 1.0, 0.0, 0.0, 0.0} 

#define L_S_Cp {0,0,1,1,1}
#define L_S_Coeffs_Plus  {9.0/20.0, 6.0/20.0, 1.0/20.0, 4.0/20.0}
#define L_S_Coeffs_Minus {9.0/20.0, 6.0/20.0, 1.0/20.0, 4.0/20.0}

//local L_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      L_S_Recon[i] = L_S_Recon[i] + L_S_Recon_Plus[(St-1)*6+i]*L_S_Coeffs_Plus[St]
//   end
//end

#define L_S_KennedyOrder 0 // Zero-order
#define L_S_KennedyCoeff {0.0, 0.0, 0.0}


//   dx:                                 |-------------|
// dxp1:                                               |-------------|
// dxp2:                                               |---------------------------|
// dxp3:                                               |-----------------------------------------|
//                                     c-0.5    c     x=0    c+1           c+2           c+3
//                                       x------x------|------x------|------x------|------x------|
//
// Plus reconstruction:
// 1st:                                  o------o-----> <-----o
// 2nd:                                         o-----> <-----o-------------o
// 3rd:                                         does not exist
// 4th:                                         o-----> <-----o-------------o-------------o
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o-------------o
// 2nd:                                  o------o-----> <-----o
// 3rd:                                                 <-----o-------------o-------------o
// 4th:                                         does not exist

#define Lp1_S_Cp { -1,           -1,                          1,            2,            3}

#define Lp1_S_Recon_Plus { \
                   0.0,       -2.0/4.0,      5.0/4.0,      1.0/4.0,        0.0,          0.0,   \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0,   \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0,   \
                   0.0,          0.0,        3.0/12.0,    13.0/12.0,    -5.0/12.0,     1.0/12.0} 

#define Lp1_S_Recon_Minus { \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0,   \
                   0.0,       -2.0/4.0,      5.0/4.0,      1.0/4.0,        0.0,          0.0,   \
                   0.0,          0.0,          0.0,       11.0/6.0,     -7.0/6.0,      2.0/6.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0}

#define Lp1_S_Coeffs_Plus  {2.0/4.0,  1.0/4.0,    0.0,    1.0/4.0}
#define Lp1_S_Coeffs_Minus {7.0/16.0, 8.0/16.0, 1.0/16.0,   0.0  }

//local Lp1_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Lp1_S_Recon[i] = Lp1_S_Recon[i] + Lp1_S_Recon_Plus[(St-1)*6+i]*Lp1_S_Coeffs_Plus[St]
//   end
//end

#define Lp1_S_KennedyOrder 1 // Second-order
#define Lp1_S_KennedyCoeff {0.5, 0.0, 0.0}


// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
// dxp2:                                               |---------------------------|
// dxp3:                                               |-----------------------------------------|
//                       c-1.5   c-1            c     x=0    c+1           c+2           c+3
//                         x------x------|------x------|------x------|------x------|------x------|
//
// Plus reconstruction:
// 1st:                           o-------------o-----> <-----o
// 2nd:                                         o-----> <-----o-------------o
// 3rd:                    o------o-------------o----->
// 4th:                                         o-----> <-----o-------------o-------------o
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o-------------o
// 2nd:                           o-------------o-----> <-----o
// 3rd:                                                 <-----o-------------o-------------o
// 4th:                    o------o-------------o-----> <-----o

#define Lp2_S_Cp { -2,           -1,                          1,            2,            3}

#define Lp2_S_Recon_Plus { \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0,   \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0,   \
                   1.0,         -2.0,          2.0,          0.0,          0.0,          0.0,   \
                   0.0,          0.0,        3.0/12.0,    13.0/12.0,    -5.0/12.0,     1.0/12.0}

#define Lp2_S_Recon_Minus { \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0,   \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0,   \
                   0.0,          0.0,          0.0,       11.0/6.0,     -7.0/6.0,      2.0/6.0, \
                 3.0/9.0,     -7.0/9.0,     11.0/9.0,      2.0/9.0,        0.0,          0.0}   \

#define Lp2_S_Coeffs_Plus  {47.0/100.0, 27.0/100.0, 10.0/100.0, 16.0/100.0}
#define Lp2_S_Coeffs_Minus {39.0/100.0, 27.0/100.0,  4.0/100.0, 30.0/100.0}

//local Lp2_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Lp2_S_Recon[i] = Lp2_S_Recon[i] + Lp2_S_Recon_Plus[(St-1)*6+i]*Lp2_S_Coeffs_Plus[St]
//   end
//end

#define Lp2_S_KennedyOrder 2 // Fourth-order
#define Lp2_S_KennedyCoeff {2.0/3.0, -1.0/12.0, 0.0}


//-------------------------------------------------------------------------------
//-- COLLOCATED LEFT BC
//-------------------------------------------------------------------------------

//   dx:                                 |-------------|
// dxp1:                                               |-------------|
//                                              c     x=0    c+1
//                                       |------x------|------x------|
//
// Plus reconstruction:
// 1st:                                         o----->        
// 2nd:                                         does not exist
// 3rd:                                         does not exist
// 4th:                                         does not exist
//
// Minus reconstruction:
// 1st:                                                 <-----o
// 2nd:                                         does not exist
// 3rd:                                         does not exist
// 4th:                                         does not exist

#define L_C_Cp {    0,            0,                          1,            2,            2}

#define L_C_Recon_Plus  {  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
#define L_C_Recon_Minus {  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

#define L_C_Coeffs_Plus  {1.0, 0.0, 0.0, 0.0}
#define L_C_Coeffs_Minus {1.0, 0.0, 0.0, 0.0}

#define L_C_Recon {0.0, 0.0, 0.5, 0.5, 0.0, 0.0}

#define L_C_KennedyOrder 1 // Second-order
#define L_C_KennedyCoeff {0.5, 0.0, 0.0}


// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
// dxp2:                                               |---------------------------|
// dxp3:                                               |-----------------------------------------|
//                               c-1            c     x=0    c+1           c+2           c+3
//                         |------x------|------x------|------x------|------x------|------x------|
//
// Plus reconstruction:
// 1st:                           o-------------o-----> <-----o
// 2nd:                                         o-----> <-----o-------------o
// 3rd:                                         does not exist
// 4th:                                         does not exist
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o-------------o
// 2nd:                           o-------------o-----> <-----o
// 3rd:                                         does not exist
// 4th:                                         does not exist

#define Lp1_C_Cp { -1,           -1,                          1,            2,            2}

#define Lp1_C_Recon_Plus { \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0}

#define Lp1_C_Recon_Minus { \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0}

#define Lp1_C_Coeffs_Plus  {1.0/2.0, 1.0/2.0, 0.0, 0.0}
#define Lp1_C_Coeffs_Minus {1.0/2.0, 1.0/2.0, 0.0, 0.0}

//local Lp1_C_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Lp1_C_Recon[i] = Lp1_C_Recon[i] + Lp1_C_Recon_Plus[(St-1)*6+i]*Lp1_C_Coeffs_Plus[St]
//   end
//end

#define Lp1_C_KennedyOrder 2 // Fourth-order
#define Lp1_C_KennedyCoeff {2.0/3.0, -1.0/12.0, 0.0}

//-----------------------------------------------------------------------------
// STAGGERED RIGHT BC
//-----------------------------------------------------------------------------

#define R_S_Cp {-1, -1, 0, 0, 0}

#define R_S_Recon_Plus  { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
#define R_S_Recon_Minus { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

#define R_S_Coeffs_Plus  {0.0, 0.0, 0.0, 0.0}
#define R_S_Coeffs_Minus {0.0, 0.0, 0.0, 0.0}

//local R_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})

#define R_S_KennedyOrder 0 // Zero-order
#define R_S_KennedyCoeff {0.0, 0.0, 0.0}

// Boundary node is staggered on the face so we do not need any reconstruction
#define Rm1_S_Cp {-1, -1, 1, 1, 1}

#define Rm1_S_Recon_Plus  {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0}
#define Rm1_S_Recon_Minus {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0}

#define Rm1_S_Coeffs_Plus  {9.0/20.0, 6.0/20.0, 1.0/20.0, 4.0/20.0}
#define Rm1_S_Coeffs_Minus {9.0/20.0, 6.0/20.0, 1.0/20.0, 4.0/20.0}

//local Rm1_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Rm1_S_Recon[i] = Rm1_S_Recon[i] + Rm1_S_Recon_Plus[(St-1)*6+i]*Rm1_S_Coeffs_Plus[St]
//   end
//end

#define Rm1_S_KennedyOrder 0 // Zero-order
#define Rm1_S_KennedyCoeff {0.0, 0.0, 0.0}


// dxm2:     |-----------------------------------------|
// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
//                 c-2           c-1            c     x=0    c+1   c+1.5
//           |------x------|------x------|------x------|------x------x
//
// Plus reconstruction:
// 1st:                           o-------------o-----> <-----o
// 2nd:                                         o-----> <-----o------o
// 3rd:             o-------------o-------------o----->
// 4th:                                         does not exist
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o------o
// 2nd:                           o-------------o-----> <-----o
// 3rd:                                         does not exist
// 4th:             o-------------o-------------o-----> <-----o

#define Rm2_S_Cp { -2,           -1,                          1,            2,            2}

#define Rm2_S_Recon_Plus { \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,        1.0/4.0,      5.0/4.0,     -2.0/4.0,        0.0, \
                 2.0/6.0,     -7.0/6.0,     11.0/6.0,        0.0,          0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0}

#define Rm2_S_Recon_Minus { \
                   0.0,          0.0,        1.0/4.0,      5.0/4.0,     -2.0/4.0,        0.0, \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0, \
                 1.0/12.0,    -5.0/12.0,    13.0/12.0,     3.0/12.0,       0.0,          0.0}

#define Rm2_S_Coeffs_Plus  {7.0/16.0, 8.0/16.0, 1.0/16.0,   0.0  }
#define Rm2_S_Coeffs_Minus {2.0/4.0,  1.0/4.0,    0.0,    1.0/4.0}

//local Rm2_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Rm2_S_Recon[i] = Rm2_S_Recon[i] + Rm2_S_Recon_Plus[(St-1)*6+i]*Rm2_S_Coeffs_Plus[St]
//   end
//end

#define Rm2_S_KennedyOrder 1 // Second-order
#define Rm2_S_KennedyCoeff {0.5, 0.0, 0.0}


// dxm2:     |-----------------------------------------|
// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
// dxp2:                                               |---------------------------|
//                 c-2           c-1            c     x=0    c+1           c+2   c+2.5
//           |------x------|------x------|------x------|------x------|------x------x
//
// Plus reconstruction:
// 1st:                           o-------------o-----> <-----o
// 2nd:                                         o-----> <-----o-------------o
// 3rd:             o-------------o-------------o----->
// 4th:                                         o-----> <-----o-------------o------o
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o-------------o
// 2nd:                           o-------------o-----> <-----o
// 3rd:                                                 <-----o-------------o------o
// 4th:             o-------------o-------------o-----> <-----o

#define Rm3_S_Cp { -2,           -1,                          1,            2,            3}

#define Rm3_S_Recon_Plus  { \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                 2.0/6.0,     -7.0/6.0,     11.0/6.0,        0.0,          0.0,          0.0, \
                   0.0,          0.0,        2.0/9.0,     11.0/9.0,     -7.0/9.0,      3.0/9.0}

#define Rm3_S_Recon_Minus { \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,          0.0,          2.0,         -2.0,          1.0, \
                 1.0/12.0,    -5.0/12.0,    13.0/12.0,     3.0/12.0,       0.0,          0.0}

#define Rm3_S_Coeffs_Plus  {39.0/100.0, 27.0/100.0,  4.0/100.0, 30.0/100.0}
#define Rm3_S_Coeffs_Minus {47.0/100.0, 27.0/100.0, 10.0/100.0, 16.0/100.0}

//local Rm3_S_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Rm3_S_Recon[i] = Rm3_S_Recon[i] + Rm3_S_Recon_Plus[(St-1)*6+i]*Rm3_S_Coeffs_Plus[St]
//   end
//end

#define Rm3_S_KennedyOrder 2 // Fourth-order
#define Rm3_S_KennedyCoeff {2.0/3.0, -1.0/12.0, 0.0}


//-----------------------------------------------------------------------------
// COLLOCATED RIGHT BC
//-----------------------------------------------------------------------------

#define R_C_Cp {-1, -1, 0, 0, 0}

#define R_C_Recon_Plus  {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
#define R_C_Recon_Minus {  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

#define R_C_Coeffs_Plus  {0.0, 0.0, 0.0, 0.0}
#define R_C_Coeffs_Minus {0.0, 0.0, 0.0, 0.0}

//local R_C_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})

#define R_C_KennedyOrder 0 // Zero-order
#define R_C_KennedyCoeff {0.0, 0.0, 0.0}


// dxm2:     |-----------------------------------------|
// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
//                 c-2           c-1            c     x=0    c+1
//           |------x------|------x------|------x------|------x------|
//
// Plus reconstruction:
// 1st:                                         o----->
// 2nd:                                         does not exist
// 3rd:                                         does not exist
// 4th:                                         does not exist
//
// Minus reconstruction:
// 1st:                                                 <-----o
// 2nd:                                         does not exist
// 3rd:                                         does not exist
// 4th:                                         does not exist

#define Rm1_C_Cp { -1,           -1,                          1,            1,            1}

#define Rm1_C_Recon_Plus  {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
#define Rm1_C_Recon_Minus {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

#define Rm1_C_Coeffs_Plus  {1.0, 0.0, 0.0, 0.0}
#define Rm1_C_Coeffs_Minus {1.0, 0.0, 0.0, 0.0}

//local Rm1_C_Recon = terralib.newlist({0.0, 0.0, 0.5, 0.5, 0.0, 0.0})

#define Rm1_C_KennedyOrder 1 // Second-order
#define Rm1_C_KennedyCoeff {0.5, 0.0, 0.0}


// dxm2:     |-----------------------------------------|
// dxm1:                   |---------------------------|
//   dx:                                 |-------------|
// dxp1:                                               |-------------|
// dxp2:                                               |---------------------------|
//                 c-2           c-1            c     x=0    c+1           c+2
//           |------x------|------x------|------x------|------x------|------x------|
//
// Plus reconstruction:
// 1st:                           o-------------o-----> <-----o
// 2nd:                                         o-----> <-----o-------------o
// 3rd:                                         does not exist
// 4th:                                         does not exist
//
// Minus reconstruction:
// 1st:                                         o-----> <-----o-------------o
// 2nd:                           o-------------o-----> <-----o
// 3rd:                                         does not exist
// 4th:                                         does not exist

#define Rm2_C_Cp { -1,           -1,                          1,            2,            2}

#define Rm2_C_Recon_Plus  { \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0}

#define Rm2_C_Recon_Minus { \
                   0.0,          0.0,        2.0/6.0,      5.0/6.0,     -1.0/6.0,        0.0, \
                   0.0,       -1.0/6.0,      5.0/6.0,      2.0/6.0,        0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0, \
                   0.0,          0.0,          0.0,          0.0,          0.0,          0.0}

#define Rm2_C_Coeffs_Plus  {1.0/2.0, 1.0/2.0, 0.0, 0.0}
#define Rm2_C_Coeffs_Minus {1.0/2.0, 1.0/2.0, 0.0, 0.0}

//local Rm2_C_Recon = terralib.newlist({0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
//for i=1,6 do
//   for St =1,nStencils do
//      Rm2_C_Recon[i] = Rm2_C_Recon[i] + Rm2_C_Recon_Plus[(St-1)*6+i]*Rm2_C_Coeffs_Plus[St]
//   end
//end

#define Rm2_C_KennedyOrder 2 // Fourth-order
#define Rm2_C_KennedyCoeff {2.0/3.0, -1.0/12.0, 0.0}


//-----------------------------------------
// Assemble vectors with coefficients
//----------------------------------------

// Indices offsets
const __CUDA_CONST__ int Cp[][5]  = {
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
const __CUDA_CONST__ double Recon_Plus[][24]  = {
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

const __CUDA_CONST__ double Recon_Minus[][24]  = {
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
const __CUDA_CONST__ double Coeffs_Plus[][4] = {
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

const __CUDA_CONST__ double Coeffs_Minus[][4] = {
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
const __CUDA_CONST__ double Interp[][2] = {
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
const __CUDA_CONST__ double Grad[][2] = {
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
const __CUDA_CONST__ int KennedyOrder[] = {
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

const __CUDA_CONST__ double KennedyCoeff[][3] = {
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

const __CUDA_CONST__ int KennedyNSum[] = {
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

#ifdef __cplusplus
}
#endif

#endif // __PROMETEO_METRIC_COEFFS_H__
