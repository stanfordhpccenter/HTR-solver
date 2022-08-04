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

#ifndef CH41StMix_HPP
#define CH41StMix_HPP

// Number of species
#define nSpec 4
// Number of charged species
#define nIons 0
// Number of standard reactions
#define nReac 1
// Number of third body reactions
#define nTBReac 0
// Number of falloff reactions
#define nFOReac 0
// Maximum number of reactants in a reaction
#define MAX_NUM_REACTANTS 2
// Maximum number of products in a reaction
#define MAX_NUM_PRODUCTS 2
// Maximum number of colliders in a reaction
#define MAX_NUM_TB 0
// Number of Nasa polynomials
#define N_NASA_POLY 3
// Use specified orders kinetics
#define FWD_ORDERS

#include "MultiComponent.hpp"

#ifdef __cplusplus
// We cannot expose these methods to Regent

//---------------------------------
// Define Species
//---------------------------------

// CH4
#define iCH4 0
#define CH4 { \
   /*      Name = */       ("CH4"), \
   /*         W = */       12.0107e-3 + 4.0*1.00784e-3, \
   /*       inx = */       iCH4, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */          6000.0007, \
   /*          cpH = */          { 3.730042760e+06,-1.383501485e+04, 2.049107091e+01,-1.961974759e-03, 4.727313040e-07,-3.728814690e-11, 1.623737207e-15, 7.532066910e+04,-1.219124889e+02}, \
   /*          cpM = */          { 3.730042760e+06,-1.383501485e+04, 2.049107091e+01,-1.961974759e-03, 4.727313040e-07,-3.728814690e-11, 1.623737207e-15, 7.532066910e+04,-1.219124889e+02}, \
   /*          cpL = */          {-1.766850998e+05, 2.786181020e+03,-1.202577850e+01, 3.917619290e-02,-3.619054430e-05, 2.026853043e-08,-4.976705490e-12,-2.331314360e+04, 8.904322750e+01}  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_NonLinear, \
   /*        sigma = */          3.746*ATom,  \
   /*      kbOveps = */          1.0/141.4,   \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          2.600*ATom*ATom*ATom,  \
   /*         Z298 = */          13.000       \
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

// CO2
#define iCO2 2
#define CO2 { \
   /*      Name = */       ("CO2"), \
   /*         W = */       12.0107e-3+2*15.9994e-3, \
   /*       inx = */       iCO2, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */         20000.0007, \
   /*          cpH = */          {-1.544423287e+09, 1.016847056e+06,-2.561405230e+02, 3.369401080e-02,-2.181184337e-06, 6.991420840e-11,-8.842351500e-16,-8.043214510e+06, 2.254177493e+03 }, \
   /*          cpM = */          { 1.176962419e+05,-1.788791477e+03, 8.291523190e+00,-9.223156780e-05, 4.863676880e-09,-1.891053312e-12, 6.330036590e-16,-3.908350590e+04,-2.652669281e+01 }, \
   /*          cpL = */          { 4.943650540e+04,-6.264116010e+02, 5.301725240e+00, 2.503813816e-03,-2.127308728e-07,-7.689988780e-10, 2.849677801e-13,-4.528198460e+04,-7.048279440e+00 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_Linear, \
   /*        sigma = */          3.763*ATom,  \
   /*      kbOveps = */          1.0/244.0,   \
   /*           mu = */          0.000*DToCm, \
   /*        alpha = */          2.650*ATom*ATom*ATom,  \
   /*         Z298 = */          2.100        \
                           } \
             }

// H2O
#define iH2O 3
#define H2O { \
   /*      Name = */       ("H2O"), \
   /*         W = */       2*1.00784e-3 + 15.9994e-3, \
   /*       inx = */       iH2O, \
   /*   cpCoeff = */       { \
   /*     TSwitch1 = */          1000.0007, \
   /*     TSwitch2 = */          6000.0007, \
   /*         TMin = */          0200.0000, \
   /*         TMax = */          6000.0007, \
   /*          cpH = */          { 1.034972096e+06,-2.412698562e+03, 4.646110780e+00, 2.291998307e-03,-6.836830480e-07, 9.426468930e-11,-4.822380530e-15,-1.384286509e+04,-7.978148510e+00 }, \
   /*          cpM = */          { 1.034972096e+06,-2.412698562e+03, 4.646110780e+00, 2.291998307e-03,-6.836830480e-07, 9.426468930e-11,-4.822380530e-15,-1.384286509e+04,-7.978148510e+00 }, \
   /*          cpL = */          {-3.947960830e+04, 5.755731020e+02, 9.317826530e-01, 7.222712860e-03,-7.342557370e-06, 4.955043490e-09,-1.336933246e-12,-3.303974310e+04, 1.724205775e+01 }  \
                           }, \
   /* DiffCoeff = */       { \
   /*         Geom = */          SpeciesGeom_NonLinear, \
   /*        sigma = */          2.605*ATom,  \
   /*      kbOveps = */          1.0/572.4,   \
   /*           mu = */          1.844*DToCm, \
   /*        alpha = */          0.000*ATom*ATom*ATom,  \
   /*         Z298 = */          4.000        \
                           } \
             }

//---------------------------------
// Define Reactions
//---------------------------------
// Methane oxidation (CH4 + 2 O2 -> 2 H2O + CO2)
#define R1 { \
   /*      ArrCoeff = */   {  \
                /*    A = */             1.1e7, \
                /*    n = */               0.0, \
                /* EovR = */ 20000*calToJ/RGAS, \
                           }, \
   /* has_backward = */    false, \
   /*   .  Neducts = */    2, \
   /*      Npducts = */    2, \
   /*      educts  = */    { \
                    { /* ind = */ iCH4, /* nu = */ 1.0, /* ord = */ 1.0}, \
                    { /* ind = */  iO2, /* nu = */ 2.0, /* ord = */ 0.5}, \
                           }, \
   /*      pducts  = */    { \
                    { /* ind = */ iH2O, /* nu = */ 2.0}, \
                    { /* ind = */ iCO2, /* nu = */ 1.0}, \
                           }, \
           }

#ifndef __CUDACC__
inline Mix::Mix(const Config &config) :
   species{CH4, O2, CO2, H2O},
   reactions{R1}
{
   // This executable is expecting CH41StMix in the input file
   assert(config.Flow.mixture.type == MixtureModel_CH41StMix);

   // Store reference quantities
   StoreReferenceQuantities(config.Flow.mixture.u.CH41StMix.PRef,
                            config.Flow.mixture.u.CH41StMix.TRef,
                            config.Flow.mixture.u.CH41StMix.LRef,
                            config.Flow.mixture.u.CH41StMix.XiRef);
};
#endif

//---------------------------------
// Cleanup
//---------------------------------
#undef iCH4
#undef CH4
#undef iO2
#undef O2
#undef iCO2
#undef CO2
#undef iH2O
#undef H2O

#undef R1

#endif //__cplusplus

#endif // CH41StMix_HPP
