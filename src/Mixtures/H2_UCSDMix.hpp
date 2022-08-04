// Copyright (c) "2020, by Centre Européen de Recherche et de Formation Avancée en Call Scientifique
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

#ifndef __H2_UCSDMIX_HPP__
#define __H2_UCSDMIX_HPP__

// Number of species
#define nSpec 9
// Number of charged species
#define nIons 0
// Number of standard reactions
#define nReac 14
// Number of thirdbody reactions
#define nTBReac 5
// Number of falloff reactions
#define nFOReac 2
// Maximum number of reactants in a reaction
#define MAX_NUM_REACTANTS 2
// Maximum number of products in a reaction
#define MAX_NUM_PRODUCTS 2
// Maximum number of colliders in a reaction
#define MAX_NUM_TB 2
// Number of Nasa polynomials
#define N_NASA_POLY 2
// Switch to mass action kinetics
#undef FWD_ORDERS

#include "MultiComponent.hpp"

#ifdef __cplusplus
// We cannot expose these methods to Regent

//---------------------------------
// Define Species
//---------------------------------

// H2
#define iH2 0
#define H2 { \
   /*       Name = */      ("H2"), \
   /*          W = */      0.001008*2.0, \
   /*        inx = */      iH2, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 3.337279200E+00,-4.940247310E-05, 4.994567780E-07,-1.795663940E-10, 2.002553760E-14,-9.501589220E+02,-3.205023310E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 2.344331120E+00, 7.980520750E-03,-1.947815100E-05, 2.015720940E-08,-7.376117610E-12,-9.179351730E+02, 6.830102380E-01}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_Linear, \
      /*      sigma = */      2.92e-10, \
      /*    kbOveps = */      0.026315789473684213, \
      /*         mu = */      0.0, \
      /*      alpha = */      7.900000000000001e-31, \
      /*       Z298 = */      280.0, \
                           }, \
}

// H
#define iH 1
#define H { \
   /*       Name = */      ("H"), \
   /*          W = */      0.001008*1.0, \
   /*        inx = */      iH, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 2.500000010E+00,-2.308429730E-11, 1.615619480E-14,-4.735152350E-18, 4.981973570E-22, 2.547365990E+04,-4.466829140E-01}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 2.500000000E+00, 7.053328190E-13,-1.995919640E-15, 2.300816320E-18,-9.277323320E-22, 2.547365990E+04,-4.466828530E-01}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_Atom, \
      /*      sigma = */      2.05e-10, \
      /*    kbOveps = */      0.006896551724137932, \
      /*         mu = */      0.0, \
      /*      alpha = */      0.0, \
      /*       Z298 = */      0.0, \
                           }, \
}

// O2
#define iO2 2
#define O2 { \
   /*       Name = */      ("O2"), \
   /*          W = */      0.015999*2.0, \
   /*        inx = */      iO2, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 3.282537840E+00, 1.483087540E-03,-7.579666690E-07, 2.094705550E-10,-2.167177940E-14,-1.088457720E+03, 5.453231290E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 3.782456360E+00,-2.996734160E-03, 9.847302010E-06,-9.681295090E-09, 3.243728370E-12,-1.063943560E+03, 3.657675730E+00}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_Linear, \
      /*      sigma = */      3.4580000000000004e-10, \
      /*    kbOveps = */      0.009310986964618248, \
      /*         mu = */      0.0, \
      /*      alpha = */      1.6e-30, \
      /*       Z298 = */      3.8, \
                           }, \
}

// OH
#define iOH 3
#define OH { \
   /*       Name = */      ("OH"), \
   /*          W = */      0.001008*1.0+0.015999*1.0, \
   /*        inx = */      iOH, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 2.864728860E+00, 1.056504480E-03,-2.590827580E-07, 3.052186740E-11,-1.331958760E-15, 3.718857740E+03, 5.701640730E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 4.125305610E+00,-3.225449390E-03, 6.527646910E-06,-5.798536430E-09, 2.062373790E-12, 3.381538120E+03,-6.904329600E-01}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_Linear, \
      /*      sigma = */      2.7500000000000003e-10, \
      /*    kbOveps = */      0.0125, \
      /*         mu = */      0.0, \
      /*      alpha = */      0.0, \
      /*       Z298 = */      0.0, \
                           }, \
}

// O
#define iO 4
#define O { \
   /*       Name = */      ("O"), \
   /*          W = */      0.015999*1.0, \
   /*        inx = */      iO, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 2.569420780E+00,-8.597411370E-05, 4.194845890E-08,-1.001777990E-11, 1.228336910E-15, 2.921757910E+04, 4.784338640E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 3.168267100E+00,-3.279318840E-03, 6.643063960E-06,-6.128066240E-09, 2.112659710E-12, 2.912225920E+04, 2.051933460E+00}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_Atom, \
      /*      sigma = */      2.7500000000000003e-10, \
      /*    kbOveps = */      0.0125, \
      /*         mu = */      0.0, \
      /*      alpha = */      0.0, \
      /*       Z298 = */      0.0, \
                           }, \
}

// H2O
#define iH2O 5
#define H2O { \
   /*       Name = */      ("H2O"), \
   /*          W = */      0.001008*2.0+0.015999*1.0, \
   /*        inx = */      iH2O, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 3.033992490E+00, 2.176918040E-03,-1.640725180E-07,-9.704198700E-11, 1.682009920E-14,-3.000429710E+04, 4.966770100E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 4.198640560E+00,-2.036434100E-03, 6.520402110E-06,-5.487970620E-09, 1.771978170E-12,-3.029372670E+04,-8.490322080E-01}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_NonLinear, \
      /*      sigma = */      2.6050000000000003e-10, \
      /*    kbOveps = */      0.0017470300489168416, \
      /*         mu = */      6.150921915453923e-30, \
      /*      alpha = */      0.0, \
      /*       Z298 = */      4.0, \
                           }, \
}

// HO2
#define iHO2 6
#define HO2 { \
   /*       Name = */      ("HO2"), \
   /*          W = */      0.001008*1.0+0.015999*2.0, \
   /*        inx = */      iHO2, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 4.017210900E+00, 2.239820130E-03,-6.336581500E-07, 1.142463700E-10,-1.079085350E-14, 1.118567130E+02, 3.785102150E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 4.301798010E+00,-4.749120510E-03, 2.115828910E-05,-2.427638940E-08, 9.292251240E-12, 2.948080400E+02, 3.716662450E+00}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_NonLinear, \
      /*      sigma = */      3.4580000000000004e-10, \
      /*    kbOveps = */      0.009310986964618248, \
      /*         mu = */      0.0, \
      /*      alpha = */      0.0, \
      /*       Z298 = */      1.0, \
                           }, \
}

// H2O2
#define iH2O2 7
#define H2O2 { \
   /*       Name = */      ("H2O2"), \
   /*          W = */      0.001008*2.0+0.015999*2.0, \
   /*        inx = */      iH2O2, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 4.165002850E+00, 4.908316940E-03,-1.901392250E-06, 3.711859860E-10,-2.879083050E-14,-1.786178770E+04, 2.916156620E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 4.276112690E+00,-5.428224170E-04, 1.673357010E-05,-2.157708130E-08, 8.624543630E-12,-1.770258210E+04, 3.435050740E+00}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_NonLinear, \
      /*      sigma = */      3.4580000000000004e-10, \
      /*    kbOveps = */      0.009310986964618248, \
      /*         mu = */      0.0, \
      /*      alpha = */      0.0, \
      /*       Z298 = */      3.8, \
                           }, \
}

// N2
#define iN2 8
#define N2 { \
   /*       Name = */      ("N2"), \
   /*          W = */      0.014007*2.0, \
   /*        inx = */      iN2, \
   /*    cpCoeff = */      {  \
      /*   TSwitch1 = */      1000.0, \
      /*   TSwitch2 = */      1000.0, \
      /*       TMin = */      300.0, \
      /*       TMax = */      5000.0, \
      /*        cpM = */      { 0.000000000E+00, 0.000000000E+00, 2.926640000E+00, 1.487976800E-03,-5.684760000E-07, 1.009703800E-10,-6.753351000E-15,-9.227977000E+02, 5.980528000E+00}, \
      /*        cpL = */      { 0.000000000E+00, 0.000000000E+00, 3.298677000E+00, 1.408240400E-03,-3.963222000E-06, 5.641515000E-09,-2.444854000E-12,-1.020899900E+03, 3.950372000E+00}, \
                           }, \
   /*  DiffCoeff = */      {  \
      /*       Geom = */      SpeciesGeom_Linear, \
      /*      sigma = */      3.621e-10, \
      /*    kbOveps = */      0.010253255408592227, \
      /*         mu = */      0.0, \
      /*      alpha = */      1.76e-30, \
      /*       Z298 = */      4.0, \
                           }, \
}


//---------------------------------
// Define Reactions
//---------------------------------

// R 0: H + O2 <=> O + OH
#define R0 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   3.520000000e+10, \
      /*           n = */   -7.000000000e-01, \
      /*        EovR = */   8.589851761e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iO2 , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iO  , /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          } \
}

// R 1: H2 + O <=> H + OH
#define R1 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   5.060000000e-02, \
      /*           n = */   2.670000000e+00, \
      /*        EovR = */   3.165567894e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH2 , /* nu = */ 1.0 }, \
      { /* ind = */ iO  , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          } \
}

// R 2: H2 + OH <=> H + H2O
#define R2 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   1.170000000e+03, \
      /*           n = */   1.300000000e+00, \
      /*        EovR = */   1.829343906e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH2 , /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iH2O, /* nu = */ 1.0 } \
                          } \
}

// R 3: H2O + O <=> 2 OH
#define R3 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   7.600000000e-06, \
      /*           n = */   3.840000000e+00, \
      /*        EovR = */   6.430964479e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*       educts = */   { \
      { /* ind = */ iH2O, /* nu = */ 1.0 }, \
      { /* ind = */ iO  , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iOH , /* nu = */ 2.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
}

// R 4: 2 H + M <=> H2 + M
#define R4 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   1.300000000e+06, \
      /*           n = */   -1.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 2.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2 , /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.5 }, \
      { /* ind = */ iH2O, /* eff = */ 12.0 } \
                          } \
}

// R 5: H + OH + M <=> H2O + M
#define R5 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   4.000000000e+10, \
      /*           n = */   -2.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O, /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.5 }, \
      { /* ind = */ iH2O, /* eff = */ 12.0 } \
                          } \
}

// R 6: 2 O + M <=> O2 + M
#define R6 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   6.170000000e+03, \
      /*           n = */   -5.000000000e-01, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iO  , /* nu = */ 2.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iO2 , /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.5 }, \
      { /* ind = */ iH2O, /* eff = */ 12.0 } \
                          } \
}

// R 7: H + O + M <=> OH + M
#define R7 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   4.710000000e+06, \
      /*           n = */   -1.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iO  , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iOH , /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.5 }, \
      { /* ind = */ iH2O, /* eff = */ 12.0 } \
                          } \
}

// R 8: O + OH + M <=> HO2 + M
#define R8 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   8.000000000e+03, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iO  , /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iHO2, /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.5 }, \
      { /* ind = */ iH2O, /* eff = */ 12.0 } \
                          } \
}

// R 9: H + O2 (+M) <=> HO2 (+M)
#define R9 { \
   /*    ArrCoeffL = */   { \
      /*           A = */   5.750000000e+07, \
      /*           n = */   -1.400000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /*    ArrCoeffH = */   { \
      /*           A = */   4.650000000e+06, \
      /*           n = */   4.400000000e-01, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iO2 , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iHO2, /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.5 }, \
      { /* ind = */ iH2O, /* eff = */ 16.0 } \
                          }, \
   /*        Ftype = */   F_Troe2, \
   /*       FOData = */   { .Troe2 = { \
   /*            alpha = */   0.5, \
   /*               T1 = */   1.0000000000000002e+30, \
   /*               T3 = */   1e-30  \
                          }}, \
}

// R 10: H + HO2 <=> 2 OH
#define R10 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   7.080000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.484497624e+02, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iHO2, /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iOH , /* nu = */ 2.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
}

// R 11: H + HO2 <=> H2 + O2
#define R11 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   1.660000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   4.140993541e+02, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iHO2, /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2 , /* nu = */ 1.0 }, \
      { /* ind = */ iO2 , /* nu = */ 1.0 } \
                          } \
}

// R 12: H + HO2 <=> H2O + O
#define R12 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   3.100000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   8.659603020e+02, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iHO2, /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O, /* nu = */ 1.0 }, \
      { /* ind = */ iO  , /* nu = */ 1.0 } \
                          } \
}

// R 13: HO2 + O <=> O2 + OH
#define R13 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   2.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iHO2, /* nu = */ 1.0 }, \
      { /* ind = */ iO  , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iO2 , /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          } \
}

// R 14: HO2 + OH <=> H2O + O2
#define R14 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   2.890000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -2.501655267e+02, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iHO2, /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O, /* nu = */ 1.0 }, \
      { /* ind = */ iO2 , /* nu = */ 1.0 } \
                          } \
}

// R 15: 2 OH (+M) <=> H2O2 (+M)
#define R15 { \
   /*    ArrCoeffL = */   { \
      /*           A = */   2.300000000e+06, \
      /*           n = */   -9.000000000e-01, \
      /*        EovR = */   -8.563387445e+02, \
                          }, \
   /*    ArrCoeffH = */   { \
      /*           A = */   7.400000000e+07, \
      /*           n = */   -3.700000000e-01, \
      /*        EovR = */   0.000000000e+00, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   2, \
   /*       educts = */   { \
      { /* ind = */ iOH , /* nu = */ 2.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O2, /* nu = */ 1.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       thirdb = */   { \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 } \
                          }, \
   /*        Ftype = */   F_Troe3, \
   /*       FOData = */   { .Troe3 = { \
   /*            alpha = */   0.735, \
   /*               T1 = */   1756.0, \
   /*               T2 = */   5182.0, \
   /*               T3 = */   94.0  \
                          }}, \
}

// R 16: 2 HO2 <=> H2O2 + O2
#define R16 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   3.020000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   6.975780139e+02, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iHO2, /* nu = */ 2.0 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O2, /* nu = */ 1.0 }, \
      { /* ind = */ iO2 , /* nu = */ 1.0 } \
                          } \
}

// R 17: H + H2O2 <=> H2 + HO2
#define R17 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   2.300000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   4.000620452e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iH2O2, /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2 , /* nu = */ 1.0 }, \
      { /* ind = */ iHO2, /* nu = */ 1.0 } \
                          } \
}

// R 18: H + H2O2 <=> H2O + OH
#define R18 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   1.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.804087317e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH  , /* nu = */ 1.0 }, \
      { /* ind = */ iH2O2, /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O, /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          } \
}

// R 19: H2O2 + OH <=> H2O + HO2
#define R19 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   7.080000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   7.216319076e+02, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH2O2, /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iH2O, /* nu = */ 1.0 }, \
      { /* ind = */ iHO2, /* nu = */ 1.0 } \
                          } \
}

// R 20: H2O2 + O <=> HO2 + OH
#define R20 { \
   /*    ArrCoeff  = */   { \
      /*           A = */   9.630000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   2.008550446e+03, \
                          }, \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   { \
      { /* ind = */ iH2O2, /* nu = */ 1.0 }, \
      { /* ind = */ iO  , /* nu = */ 1.0 } \
                          }, \
   /*       pducts = */   { \
      { /* ind = */ iHO2, /* nu = */ 1.0 }, \
      { /* ind = */ iOH , /* nu = */ 1.0 } \
                          } \
}


//---------------------------------
// Initialization function
//---------------------------------

#ifndef __CUDACC__
inline Mix::Mix(const Config &config) :
   species{H2, H, O2, OH, O, H2O, HO2, H2O2,
           N2 },
   reactions{R0, R1, R2, R3, R10, R11, R12, R13,
             R14, R16, R17, R18, R19, R20 },
   ThirdbodyReactions{R4, R5, R6, R7, R8 },
   FalloffReactions{R9, R15 }
{
   // This executable is expecting H2_UCSDMix in the input file
   assert(config.Flow.mixture.type == MixtureModel_H2_UCSDMix);

   // Store reference quantities
   StoreReferenceQuantities(config.Flow.mixture.u.H2_UCSDMix.PRef,
                            config.Flow.mixture.u.H2_UCSDMix.TRef,
                            config.Flow.mixture.u.H2_UCSDMix.LRef,
                            config.Flow.mixture.u.H2_UCSDMix.XiRef);
};
#endif

//---------------------------------
// Cleanup
//---------------------------------

#undef iH2
#undef H2
#undef iH
#undef H
#undef iO2
#undef O2
#undef iOH
#undef OH
#undef iO
#undef O
#undef iH2O
#undef H2O
#undef iHO2
#undef HO2
#undef iH2O2
#undef H2O2
#undef iN2
#undef N2

#undef R0
#undef R1
#undef R2
#undef R3
#undef R4
#undef R5
#undef R6
#undef R7
#undef R8
#undef R9
#undef R10
#undef R11
#undef R12
#undef R13
#undef R14
#undef R15
#undef R16
#undef R17
#undef R18
#undef R19
#undef R20

#endif

#endif // __H2_UCSDMIX_HPP__
