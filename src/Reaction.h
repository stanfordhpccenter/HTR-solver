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

#ifndef Reaction_H
#define Reaction_H

#ifndef MAX_NUM_REACTANTS
   #error "MAX_NUM_REACTANTS is undefined"
#endif

#ifndef MAX_NUM_TB
   #error "MAX_NUM_TB is undefined"
#endif

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Generic reactant
struct Reactant {
   int    ind;  // Index in the species vector
   double nu;   // Stoichiometric coefficient
   double ord;  // Order of the reactant
};

struct ThirdBd {
   int    ind;  // Index in the species vector
   double eff;  // Efficiency as a third body
};

// Reaction structure
struct Reaction {
   double   A;                // Arrhenius pre-exponential factor [m^{3*(o-1)}/(mol^(o-1) s)] where o is the order fo the reaction
   double   n;                // Arrhenius temperature exponent
   double   EovR;             // Arrhenius activation energy [K]
   bool     has_backward;     // Self-explenatory

   int      Neducts;          //  number of reactants
   int      Npducts;          //  number of products
   int      Nthirdb;          //  number of third bodies

   struct Reactant educts[MAX_NUM_REACTANTS];  // List of reactants and stoichiometric coefficients
   struct Reactant pducts[MAX_NUM_REACTANTS];  // List of products and stoichiometric coefficients
   struct ThirdBd  thirdb[MAX_NUM_TB];         // List of third bodies and efficiencies
};

#ifdef __cplusplus
}
#endif

#endif // Reaction_H
