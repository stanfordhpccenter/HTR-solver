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


#ifndef __PROMETEO_CONST_H__
#define __PROMETEO_CONST_H__

//-----------------------------------------------------------------------------
// LOAD THE EQUATION OF STATE
//-----------------------------------------------------------------------------
#define QUOTEME(M)       #M
#define INCLUDE_FILE(M)  QUOTEME(M.hpp)
#include INCLUDE_FILE( EOS )
#undef QUOTEME
#undef INCLUDE_FILE

#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// SOLVER CONSTANTS
//-----------------------------------------------------------------------------

// SSP Runge-Kutta method Gottlieb et al. (2001)
// RK_C[*][1] -- Coefficinets for solution at time n
// RK_C[*][2] -- Coefficinets for intermediate solution
// RK_C[*][3] -- Coefficinets for delta t
//Exports.RK_C = {
//   [1] = {    1.0,     0.0,     1.0},
//   [2] = {3.0/4.0, 1.0/4.0, 1.0/4.0},
//   [3] = {1.0/3.0, 2.0/3.0, 2.0/3.0},
//}

// Common groups of variables
//Exports.Primitives = terralib.newlist({
//   'pressure',
//   'temperature',
//   'MolarFracs',
//   'velocity'
//})
//
//Exports.Properties = terralib.newlist({
//   'rho',
//   'mu',
//   'lam',
//   'Di',
//   'SoS'
//})
//
//Exports.ProfilesVars = terralib.newlist({
//   'MolarFracs_profile',
//   'velocity_profile',
//   'temperature_profile'
//})
//
//Exports.RecycleVars = terralib.newlist({
//   'temperature_recycle',
//   'MolarFracs_recycle',
//   'velocity_recycle'
//})

// Variable indices
#define irU (nSpec  )
#define irE (nSpec+3)
#define nEq (nSpec+4)

#ifdef __cplusplus
}
#endif

#endif // __PROMETEO_CONST_H__
