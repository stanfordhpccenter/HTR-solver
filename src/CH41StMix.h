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

#ifndef CH41StMix_H
#define CH41StMix_H

#define nSpec 4
#define nReac 1

#define MAX_NUM_REACTANTS 2
#define MAX_NUM_TB 0

#define RGAS  8.3144598        // [J/(mol K)]
#define Na    6.02214086e23    // [1/mol]
#define kb    1.38064852e-23   // [m^2 kg /( s^2 K)]

#include "Species.h"
#include "Reaction.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Mix {
   struct Spec species[nSpec];
   struct Reaction reactions[nReac];
   // Max an min acceptable temeperatures
   double TMax;
   double TMin;
};

#ifdef __cplusplus
}
#endif

#endif // CH41StMix_H
