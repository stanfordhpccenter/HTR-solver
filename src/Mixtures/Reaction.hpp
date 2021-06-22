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

#ifndef Reaction_HPP
#define Reaction_HPP

#include <stdbool.h>

#ifndef nSpec
   #error "nSpec is undefined"
#endif

#ifndef MAX_NUM_REACTANTS
   #error "MAX_NUM_REACTANTS is undefined"
#endif

#ifndef MAX_NUM_PRODUCTS
   #error "MAX_NUM_PRODUCTS is undefined"
#endif

#ifndef MAX_NUM_TB
   #error "MAX_NUM_TB is undefined"
#endif

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#ifndef __UNROLL__
#ifdef __CUDACC__
#define __UNROLL__ #pragma unroll
#else
#define __UNROLL__
#endif
#endif

#ifndef __CONST__
#ifdef __CUDACC__
#define __CONST__
#else
#define __CONST__ const
#endif
#endif

#include "constants.h"

#ifdef __cplusplus
   // We cannot expose these structs to Regent
   #include "my_array.hpp"

   #ifndef __CUDACC__
   using std::max;
   using std::min;
   #endif

   // Define type for the array that will contain the species
   typedef MyArray<double, nSpec> VecNSp;
#endif

// F types
#define F_Lindemann 0
#define F_Troe2     1
#define F_Troe3     2
#define F_SRI       3

#ifdef __cplusplus
extern "C" {
#endif

// Generic reactant
struct Reactant {
   __CONST__ uint8_t ind; // Index in the species vector
   __CONST__ float    nu; // Stoichiometric coefficient
#ifdef FWD_ORDERS
   __CONST__ float   ord; // Order of the reactant
#endif
};

// Generic product
struct Product {
   __CONST__ uint8_t ind; // Index in the species vector
   __CONST__ float    nu; // Stoichiometric coefficient
};

// Generic collider
struct ThirdBd {
   __CONST__ uint8_t ind; // Index in the species vector
   __CONST__ float   eff; // Efficiency as a third body
};

// Arrhenius coefficients
struct ArrheniusCoeff {
   __CONST__ double   A;                // Pre-exponential factor [m^{3*(o-1)}/(mol^(o-1) s)] where o is the order fo the reaction
   __CONST__ float    n;                // Temperature exponent
   __CONST__ float    EovR;             // Activation energy [K]

#ifdef __cplusplus
   // We cannot expose these methods to Regent

   __CUDA_HD__
   inline double CompRateCoeff(const double T) const;
#endif
};

// Standard reaction
struct Reaction {
   __CONST__ struct ArrheniusCoeff ArrCoeff; // Arrhenius coefficients

   __CONST__ bool     has_backward;     // Self-explenatory

   __CONST__ uint8_t   Neducts;         //  number of reactants
   __CONST__ uint8_t   Npducts;         //  number of products

   __CONST__ struct Reactant educts[MAX_NUM_REACTANTS];  // List of reactants and stoichiometric coefficients
   __CONST__ struct Product  pducts[MAX_NUM_PRODUCTS];   // List of products and stoichiometric coefficients

#ifdef __cplusplus
   // We cannot expose these methods to Regent
private:
   __CUDA_HD__
   inline double GetReactionRate(const double P, const double T, const VecNSp &C, const VecNSp &G) const;

public:
   __CUDA_HD__
   inline void AddProductionRates(VecNSp &w, const double P, const double T, const VecNSp &C, const VecNSp &G) const;
#endif
};

// Thirdbody reaction
struct ThirdbodyReaction {
   __CONST__ struct ArrheniusCoeff ArrCoeff; // Arrhenius coefficients

   __CONST__ bool     has_backward;     // Self-explenatory

   __CONST__ uint8_t   Neducts;         //  number of reactants
   __CONST__ uint8_t   Npducts;         //  number of products
   __CONST__ uint8_t   Nthirdb;         //  number of third bodies

   __CONST__ struct Reactant educts[MAX_NUM_REACTANTS];  // List of reactants and stoichiometric coefficients
   __CONST__ struct Product  pducts[MAX_NUM_PRODUCTS];   // List of products and stoichiometric coefficients
   __CONST__ struct ThirdBd  thirdb[MAX_NUM_TB];         // List of third bodies and efficiencies

#ifdef __cplusplus
   // We cannot expose these methods to Regent
private:
   __CUDA_HD__
   inline double GetReactionRate(const double P, const double T, const VecNSp &C, const VecNSp &G) const;

public:
   __CUDA_HD__
   inline void AddProductionRates(VecNSp &w, const double P, const double T, const VecNSp &C, const VecNSp &G) const;
#endif
};

// Fall-off data structure
union FalloffData {
   struct {
      int __dummy;
   } Lindemann;
   struct {
      float   alpha;              // Troe alpha coefficient
      double  T1;                 // Troe temperature 1
      double  T3;                 // Troe temperature 3
   } Troe2;
   struct {
      float   alpha;              // Troe alpha coefficient
      double  T1;                 // Troe temperature 1
      double  T2;                 // Troe temperature 2
      double  T3;                 // Troe temperature 3
   } Troe3;
   struct {
      float   A;
      float   B;
      float   C;
      float   D;
      float   E;
   } SRI;
};

// Fall-off reaction structure
struct FalloffReaction {
   __CONST__ struct ArrheniusCoeff ArrCoeffL; // Arrhenius coefficients for low pressure
   __CONST__ struct ArrheniusCoeff ArrCoeffH; // Arrhenius coefficients for high pressure

   __CONST__ bool     has_backward;     // Self-explenatory

   __CONST__ uint8_t   Neducts;         //  number of reactants
   __CONST__ uint8_t   Npducts;         //  number of products
   __CONST__ uint8_t   Nthirdb;         //  number of third bodies

   __CONST__ struct Reactant educts[MAX_NUM_REACTANTS];  // List of reactants and stoichiometric coefficients
   __CONST__ struct Product  pducts[MAX_NUM_PRODUCTS];   // List of products and stoichiometric coefficients
   __CONST__ struct ThirdBd  thirdb[MAX_NUM_TB];         // List of third bodies and efficiencies

   __CONST__ uint8_t            Ftype;         // type of Fall-off coefficient
   __CONST__ union FalloffData  FOdata;        // Fall-off coefficient data

#ifdef __cplusplus
   // We cannot expose these methods to Regent
private:
   __CUDA_HD__
   inline double computeF(const double Pr, const double T) const;

   __CUDA_HD__
   inline double GetReactionRate(const double P, const double T, const VecNSp &C, const VecNSp &G) const;

public:
   __CUDA_HD__
   inline void AddProductionRates(VecNSp &w, const double P, const double T, const VecNSp &C, const VecNSp &G) const;
#endif
};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// We cannot expose these methods to Regent
#include "Reaction.inl"
#endif

#endif // Reaction_HPP
