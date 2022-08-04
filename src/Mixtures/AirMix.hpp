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

#ifndef AirMix_HPP
#define AirMix_HPP

#include "config_schema.h"

// Number of species
#define nSpec 5
// Number of charged species
#define nIons 0
// Number of standard reactions
#define nReac 2
// Number of third bodies reactions
#define nTBReac 3
// Number of falloff reactions
#define nFOReac 0
// Maximum number of reactants in a reaction
#define MAX_NUM_REACTANTS 2
// Maximum number of products in a reaction
#define MAX_NUM_PRODUCTS 2
// Maximum number of colliders in a reaction
#define MAX_NUM_TB 5
// Number of Nasa polynomials
#define N_NASA_POLY 3
// Use mass action kinetics
#undef FWD_ORDERS

#include "MultiComponent.hpp"

#ifdef __cplusplus
// We cannot expose these methods to Regent

//---------------------------------
// Define Species
//---------------------------------

// CH4
#define iN2 0
#define N2 { \
   /*      Name = */       ("N2"), \
   /*         W = */       2*14.0067e-3, \
   /*       inx = */       iN2, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */         20000.0007, \
   /*          cpH = */          { 8.310139160e+08,-6.420733540e+05, 2.020264635e+02,-3.065092046e-02, 2.486903333e-06,-9.705954110e-11, 1.437538881e-15, 4.938707040e+06,-1.672099740e+03 }, \
   /*          cpM = */          { 5.877124060e+05,-2.239249073e+03, 6.066949220e+00,-6.139685500e-04, 1.491806679e-07,-1.923105485e-11, 1.061954386e-15, 1.283210415e+04,-1.586640027e+01 }, \
   /*          cpL = */          { 2.210371497e+04,-3.818461820e+02, 6.082738360e+00,-8.530914410e-03, 1.384646189e-05,-9.625793620e-09, 2.519705809e-12, 7.108460860e+02,-1.076003744e+01 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_Linear, \
   /*        sigma = */          3.621*ATom,  \
   /*      kbOveps = */          1.0/97.530,  \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          1.760*ATom*ATom*ATom,  \
   /*         Z298 = */          4.000        \
                           } \
             }

// O2
#define iO2 1
#define O2 { \
   /*      Name = */       ("O2"), \
   /*         W = */       2*15.9994e-3, \
   /*       inx = */       iO2, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */         20000.0007, \
   /*          cpH = */          { 4.975294300e+08,-2.866106874e+05, 6.690352250e+01,-6.169959020e-03, 3.016396027e-07,-7.421416600e-12, 7.278175770e-17, 2.293554027e+06,-5.530621610e+02 }, \
   /*          cpM = */          {-1.037939022e+06, 2.344830282e+03, 1.819732036e+00, 1.267847582e-03,-2.188067988e-07, 2.053719572e-11,-8.193467050e-16,-1.689010929e+04, 1.738716506e+01 }, \
   /*          cpL = */          {-3.425563420e+04, 4.847000970e+02, 1.119010961e+00, 4.293889240e-03,-6.836300520e-07,-2.023372700e-09, 1.039040018e-12,-3.391454870e+03, 1.849699470e+01 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_Linear, \
   /*        sigma = */          3.458*ATom,  \
   /*      kbOveps = */          1.0/107.40,  \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          1.600*ATom*ATom*ATom,  \
   /*         Z298 = */          3.8000       \
                           } \
             }

// NO
#define iNO 2
#define NO { \
   /*      Name = */       ("NO"), \
   /*         W = */       14.0067e-3+15.9994e-3, \
   /*       inx = */       iNO, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */         20000.0007, \
   /*          cpH = */          {-9.575303540e+08, 5.912434480e+05,-1.384566826e+02, 1.694339403e-02,-1.007351096e-06, 2.912584076e-11,-3.295109350e-16,-4.677501240e+06, 1.242081216e+03 }, \
   /*          cpM = */          { 2.239018716e+05,-1.289651623e+03, 5.433936030e+00,-3.656034900e-04, 9.880966450e-08,-1.416076856e-11, 9.380184620e-16, 1.750317656e+04,-8.501669090e+00 }, \
   /*          cpL = */          {-1.143916503e+04, 1.536467592e+02, 3.431468730e+00,-2.668592368e-03, 8.481399120e-06,-7.685111050e-09, 2.386797655e-12, 9.098214410e+03, 6.728725490e+00 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_Linear, \
   /*        sigma = */          3.621*ATom,  \
   /*      kbOveps = */          1.0/97.530,  \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          1.760*ATom*ATom*ATom,  \
   /*         Z298 = */          4.000        \
                           } \
             }

// N
#define iN 3
#define N { \
   /*      Name = */       ("N"), \
   /*         W = */       14.0067e-3, \
   /*       inx = */       iN, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */         20000.0007, \
   /*          cpH = */          { 5.475181050e+08,-3.107574980e+05, 6.916782740e+01,-6.847988130e-03, 3.827572400e-07,-1.098367709e-11, 1.277986024e-16, 2.550585618e+06,-5.848769753e+02 }, \
   /*          cpM = */          { 8.876501380e+04,-1.071231500e+02, 2.362188287e+00, 2.916720081e-04,-1.729515100e-07, 4.012657880e-11,-2.677227571e-15, 5.697351330e+04, 4.865231506e+00 }, \
   /*          cpL = */          { 0.000000000e+00, 0.000000000e+00, 2.500000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.610463780e+04, 4.193905036e+00 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_Atom, \
   /*        sigma = */          3.298*ATom,  \
   /*      kbOveps = */          1.0/71.400,  \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          0.000*ATom*ATom*ATom,  \
   /*         Z298 = */          0.000        \
                           } \
             }

// O
#define iO 4
#define O { \
   /*      Name = */       ("O"), \
   /*         W = */       15.9994e-3, \
   /*       inx = */       iO, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */         20000.0007, \
   /*          cpH = */          { 1.779004264e+08,-1.082328257e+05, 2.810778365e+01,-2.975232262e-03, 1.854997534e-07,-5.796231540e-12, 7.191720164e-17, 8.890942630e+05,-2.181728151e+02 }, \
   /*          cpM = */          { 2.619020262e+05,-7.298722030e+02, 3.317177270e+00,-4.281334360e-04, 1.036104594e-07,-9.438304330e-12, 2.725038297e-16, 3.392428060e+04,-6.679585350e-01 }, \
   /*          cpL = */          {-7.953611300e+03, 1.607177787e+02, 1.966226438e+00, 1.013670310e-03,-1.110415423e-06, 6.517507500e-10,-1.584779251e-13, 2.840362437e+04, 8.404241820e+00 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_Atom, \
   /*        sigma = */          2.750*ATom,  \
   /*      kbOveps = */          1.0/80.000,  \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          0.000*ATom*ATom*ATom,  \
   /*         Z298 = */          0.000        \
                           } \
             }

//---------------------------------
// Define Reactions
//---------------------------------

// Oxygen dissociation (O2 + M -> 2O + M)
#define R1 { \
   /*      ArrCoeff = */   {  \
                /*    A = */           2.0e15, \
                /*    n = */             -1.5, \
                /* EovR = */            59500, \
                           }, \
   /* has_backward = */    true, \
   /*   .  Neducts = */    1, \
   /*      Npducts = */    1, \
   /*      Nthirdb = */    5, \
   /*      educts  = */    { \
                    { /* ind = */  iO2, /* nu = */ 1.0}, \
                    { /* ind = */    0, /* nu = */ 0.0}, \
                           }, \
   /*      pducts  = */    { \
                    { /* ind = */   iO, /* nu = */ 2.0}, \
                    { /* ind = */    0, /* nu = */ 0.0}, \
                           }, \
   /*      thirdb  = */    { \
                    { /* ind = */  iO2, /* eff = */ 1.0}, \
                    { /* ind = */  iNO, /* eff = */ 1.0}, \
                    { /* ind = */  iN2, /* eff = */ 1.0}, \
                    { /* ind = */   iO, /* eff = */ 5.0}, \
                    { /* ind = */   iN, /* eff = */ 5.0}, \
                           } \
           }

// NO dissociation (NO + M -> N + O + M)
#define R2 { \
   /*      ArrCoeff = */   {  \
                /*    A = */              5e9, \
                /*    n = */              0.0, \
                /* EovR = */            75500, \
                           }, \
   /* has_backward = */    true, \
   /*   .  Neducts = */    1, \
   /*      Npducts = */    2, \
   /*      Nthirdb = */    5, \
   /*      educts  = */    { \
                    { /* ind = */  iNO, /* nu = */ 1.0}, \
                    { /* ind = */    0, /* nu = */ 0.0}, \
                           }, \
   /*      pducts  = */    { \
                    { /* ind = */   iN, /* nu = */ 1.0}, \
                    { /* ind = */   iO, /* nu = */ 1.0}, \
                           }, \
   /*      thirdb  = */    { \
                    { /* ind = */  iO2, /* eff = */  1.0}, \
                    { /* ind = */  iNO, /* eff = */ 22.0}, \
                    { /* ind = */  iN2, /* eff = */  1.0}, \
                    { /* ind = */   iO, /* eff = */ 22.0}, \
                    { /* ind = */   iN, /* eff = */ 22.0}, \
                           } \
           }

// N2 dissociation (N2 + M -> 2N + M)
#define R3 { \
   /*      ArrCoeff = */   {  \
                /*    A = */             7e15, \
                /*    n = */             -1.6, \
                /* EovR = */           113200, \
                           }, \
   /* has_backward = */    true, \
   /*   .  Neducts = */    1, \
   /*      Npducts = */    1, \
   /*      Nthirdb = */    5, \
   /*      educts  = */    { \
                    { /* ind = */  iN2, /* nu = */ 1.0}, \
                    { /* ind = */    0, /* nu = */ 0.0}, \
                           }, \
   /*      pducts  = */    { \
                    { /* ind = */   iN, /* nu = */ 2.0}, \
                    { /* ind = */    0, /* nu = */ 0.0}, \
                           }, \
   /*      thirdb  = */    { \
                    { /* ind = */  iO2, /* eff = */  1.0}, \
                    { /* ind = */  iNO, /* eff = */  1.0}, \
                    { /* ind = */  iN2, /* eff = */  1.0}, \
                    { /* ind = */   iO, /* eff = */ 30.0/7}, \
                    { /* ind = */   iN, /* eff = */ 30.0/7}, \
                           } \
           }

// Zeldovich 1 (N2 + O -> NO + N)
#define R4 { \
   /*      ArrCoeff = */   {  \
                /*    A = */           6.4e11, \
                /*    n = */             -1.0, \
                /* EovR = */            38400, \
                           }, \
   /* has_backward = */    true, \
   /*   .  Neducts = */    2, \
   /*      Npducts = */    2, \
   /*      educts  = */    { \
                    { /* ind = */  iN2, /* nu = */ 1.0}, \
                    { /* ind = */   iO, /* nu = */ 1.0}, \
                           }, \
   /*      pducts  = */    { \
                    { /* ind = */  iNO, /* nu = */ 1.0}, \
                    { /* ind = */   iN, /* nu = */ 1.0}, \
                           } \
           }

// Zeldovich 2 (NO + O -> O2 + N)
#define R5 { \
   /*      ArrCoeff = */   {  \
                /*    A = */            8.4e6, \
                /*    n = */              0.0, \
                /* EovR = */            19400, \
                           }, \
   /* has_backward = */    true, \
   /*   .  Neducts = */    2, \
   /*      Npducts = */    2, \
   /*      educts  = */    { \
                    { /* ind = */  iNO, /* nu = */ 1.0}, \
                    { /* ind = */   iO, /* nu = */ 1.0}, \
                           }, \
   /*      pducts  = */    { \
                    { /* ind = */  iO2, /* nu = */ 1.0}, \
                    { /* ind = */   iN, /* nu = */ 1.0}, \
                           } \
           }

#ifndef __CUDACC__
inline Mix::Mix(const Config &config) :
   species{N2, O2, NO, N, O},
   reactions{R4, R5},
   ThirdbodyReactions{R1, R2, R3}
{
   // This executable is expecting AirMix in the input file
   assert(config.Flow.mixture.type == MixtureModel_AirMix);

   // Store reference quantities
   StoreReferenceQuantities(config.Flow.mixture.u.AirMix.PRef,
                            config.Flow.mixture.u.AirMix.TRef,
                            config.Flow.mixture.u.AirMix.LRef,
                            config.Flow.mixture.u.AirMix.XiRef);
};
#endif

//---------------------------------
// Cleanup
//---------------------------------
#undef iN2
#undef N2
#undef iO2
#undef O2
#undef iNO
#undef NO
#undef iN
#undef N
#undef iO
#undef O

#undef R1
#undef R2
#undef R3
#undef R4
#undef R5

#endif //__cplusplus

#endif // AirMix_HPP
