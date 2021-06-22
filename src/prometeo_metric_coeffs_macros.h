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

#ifndef __PROMETEO_METRIC_COEFFS_MACROS_H__
#define __PROMETEO_METRIC_COEFFS_MACROS_H__

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

#endif // __PROMETEO_METRIC_COEFFS_MACROS_H__
