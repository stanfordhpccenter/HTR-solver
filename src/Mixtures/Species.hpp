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

#ifndef Species_HPP
#define Species_HPP

#include <stdint.h>
#include <math.h>

#include "constants.h"

// We only consider constant electron mobilty for now
#ifndef eMob
#define eMob 0.4 // [m^2 / (V s)]
#endif

#ifndef N_NASA_POLY
   #error "N_NASA_POLY is undefined"
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

#ifdef __cplusplus
extern "C" {
#endif

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
   __CONST__ float    TSwitch1;     // Switch  temperature between Low and Mid  temperature polynomials
   __CONST__ float    TSwitch2;     // Switch  temperature between Mid and High temperature polynomials
   __CONST__ float    TMin;         // Minimum temperature
   __CONST__ float    TMax;         // Maximum temperature
#if (N_NASA_POLY == 3)
   __CONST__ double   cpH[9];       // High temperature polynomials
#endif
   __CONST__ double   cpM[9];       // Mid  temperature polynomials
   __CONST__ double   cpL[9];       // Low  temperature polynomials
};

// Coefficinets for diffusivity
struct DiffCoefficients {
   __CONST__ uint8_t  Geom;       // = 0 (Atom), = 1 (Linear), = 2 (Non Linear)
   __CONST__ double   sigma;      // Lennard-Jones collision diameter [m]
   __CONST__ double   kbOveps;    // Boltzmann constant divided by Lennard-Jones potential well depth [1/K]
   __CONST__ double   mu;         // Dipole moment [C*m]
   __CONST__ double   alpha;      // Polarizabilty [m^3]
   __CONST__ double   Z298;       // Rotational relaxation collision number
};

// Species structure
struct Spec {
   __CONST__ char      Name[10];  // Name of the species
   __CONST__ double           W;  // Molar weight [kg/mol]
   __CONST__ uint8_t        inx;  // Index in the species vector
#if (nIons > 0)
   __CONST__ bool    isElectron;  // Flag that determines if the species is a free electron
   __CONST__ int8_t        nCrg;  // Number of elementary charges
#endif
   __CONST__ struct cpCoefficients   cpCoeff;
   __CONST__ struct DiffCoefficients DiffCoeff;

#ifdef __cplusplus
   // We cannot expose these methods to Regent

private:
   __CUDA_HD__
   inline double omega_mu(const double T) const;

   __CUDA_HD__
   inline double omega_D(const double T) const;

   __CUDA_HD__
   inline double omega_D_N64(const double T, const double gamma) const;

public:
   __CUDA_HD__
   inline double GetCp(const double T) const;

   __CUDA_HD__
   inline double GetFreeEnthalpy(const double T) const;

   __CUDA_HD__
   inline double GetEnthalpy(const double T) const;

   __CUDA_HD__
   inline double GetMu(const double T) const;

private:
   __CUDA_HD__
   inline void GetDifCollParam_Stock(double &sigmaij, double &omega11,
                                     const Spec & i, const Spec & n, const double T) const;

#if (nIons > 0)
   __CUDA_HD__
   inline void GetDifCollParam_N64(double &sigmaij, double &omega11,
                                   const Spec & i, const Spec & n, const double T) const;
#endif

public:
   __CUDA_HD__
   inline double GetDif(const Spec & s2, const double P, const double T) const;

   __CUDA_HD__
   inline double GetMob(const Spec & s2, const double P, const double T) const;

private:
   __CUDA_HD__
   inline double GetSelfDiffusion(const double T) const;

   __CUDA_HD__
   inline double GetFZrot(const double T) const;

   __CUDA_HD__
   inline double GetLamAtom(const double T) const;

   __CUDA_HD__
   inline double GetLamLinear(const double T) const;

   __CUDA_HD__
   inline double GetLamNonLinear(const double T) const;

public:
   __CUDA_HD__
   inline double GetLam(const double T) const;

#endif
};

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
// We cannot expose these methods to Regent
#include "Species.inl"
#endif

#endif // Species_HPP
