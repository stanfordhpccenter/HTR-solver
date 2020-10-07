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

#ifndef Species_H
#define Species_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGAS  8.3144598        // [J/(mol K)]

// Species geometries
//enum SpeciesGeom {
//   Atom,
//   Linear,
//   NonLinear
//};
#define SpeciesGeom_Atom      0
#define SpeciesGeom_Linear    1
#define SpeciesGeom_NonLinear 3

// NASA polynomials data structure
struct cpCoefficients {
   double   TSwitch1;     // Switch  temperature between Low and Mid  temperature polynomials
   double   TSwitch2;     // Switch  temperature between Mid and High temperature polynomials
   double   TMin;         // Minimum temperature
   double   TMax;         // Maximum temperature
   double   cpH[9];       // High temperature polynomials
   double   cpM[9];       // Mid  temperature polynomials
   double   cpL[9];       // Low  temperature polynomials
};

// Coefficinets for diffusivity
struct DiffCoefficients {
   double   sigma;      // Lennard-Jones collision diameter [m]
   double   kbOveps;    // Boltzmann constant divided by Lennard-Jones potential well depth [1/K]
   double   mu;         // Dipole moment [C*m]
   double   alpha;      // Polarizabilty [m]
   double   Z298;       // Rotational relaxation collision number
};

// Species structure
struct Spec {
   int8_t         Name[10];  // Name of the species
   double         W;         // Molar weight [kg/mol]
   int            inx;       // Index in the species vector
   int            Geom;      // = 0 (Atom), = 1 (Linear), = 2 (Non Linear)
   struct cpCoefficients   cpCoeff;
   struct DiffCoefficients DiffCoeff;
};

#ifdef __cplusplus
}
#endif

#endif // Species_H
