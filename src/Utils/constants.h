// Copyright (c) "2020, by Centre Européen de Recherche et de Formation Avancée en Calcul Scientifiq
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

#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

//-----------------------------------------------------------------------------
// MATH CONSTANTS
//-----------------------------------------------------------------------------

#define PI    3.1415926535898

//-----------------------------------------------------------------------------
// PHYSICAL CONSTANTS
//-----------------------------------------------------------------------------

// Universal gas constant
#define RGAS  8.3144598        // [J/(mol K)]
// Avogadro number
#define Na    6.02214086e23    // [1/mol]
// Boltzmann constant
#define kb    1.38064852e-23   // [m^2 kg /( s^2 K)]
// Dielectric permittivity of the vacuum
#define eps_0 8.8541878128e-12 // [F/m] or [C/(V m)]
// Elementary electric charge
#define eCrg  1.60217662e-19   // [C]

//-----------------------------------------------------------------------------
// UNIT CONVERSION
//-----------------------------------------------------------------------------

#define ATom   1e-10            // Angstrom to meter
#define DToCm  3.33564e-30      // Debye to Coulomb meter
#define calToJ 4.184            // cal to Joule

#endif // __CONSTANTS_H__
