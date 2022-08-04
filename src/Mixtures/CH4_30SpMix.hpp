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


#ifndef __CH4_30SPMIX_HPP__
#define __CH4_30SPMIX_HPP__

// Number of species
#define nSpec 30
// Number of charged species
#define nIons 0
// Number of standard reactions
#define nReac 156
// Number of thirdbody reactions
#define nTBReac 6
// Number of falloff reactions
#define nFOReac 22
// Maximum number of reactants in a reaction
#define MAX_NUM_REACTANTS 3
// Maximum number of products in a reaction
#define MAX_NUM_PRODUCTS 3
// Number of Nasa polynomials
#define N_NASA_POLY 2
// Maximum number of colliders in a reaction
#define MAX_NUM_TB 7

#include "MultiComponent.hpp"

#ifdef __cplusplus
// We cannot expose these methods to Regent

//---------------------------------
// Define Species
//---------------------------------

// H2
#define iH2 0
#define     H2 { \
   /*      Name = */       ("H2"), \
   /*         W = */       0.00100784*2, \
   /*       inx = */       iH2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.337279200E+00, -4.940247310E-05,  4.994567780E-07, -1.795663940E-10,  2.002553760E-14, -9.501589220E+02, -3.205023310E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.344331120E+00,  7.980520750E-03, -1.947815100E-05,  2.015720940E-08, -7.376117610E-12, -9.179351730E+02,  6.830102380E-01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 2.92*ATom, \
         /*   kbOveps = */ 1.0/38.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.79*ATom*ATom*ATom, \
         /*      Z298 = */ 280.0 \
                              } \
               }

// H
#define iH 1
#define      H { \
   /*      Name = */       ("H"), \
   /*         W = */       0.00100784*1, \
   /*       inx = */       iH, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.500000010E+00, -2.308429730E-11,  1.615619480E-14, -4.735152350E-18,  4.981973570E-22,  2.547365990E+04, -4.466829140E-01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.500000000E+00,  7.053328190E-13, -1.995919640E-15,  2.300816320E-18, -9.277323320E-22,  2.547365990E+04, -4.466828530E-01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Atom, \
         /*     sigma = */ 2.05*ATom, \
         /*   kbOveps = */ 1.0/145.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// O
#define iO 2
#define      O { \
   /*      Name = */       ("O"), \
   /*         W = */       0.0159994*1, \
   /*       inx = */       iO, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.569420780E+00, -8.597411370E-05,  4.194845890E-08, -1.001777990E-11,  1.228336910E-15,  2.921757910E+04,  4.784338640E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.168267100E+00, -3.279318840E-03,  6.643063960E-06, -6.128066240E-09,  2.112659710E-12,  2.912225920E+04,  2.051933460E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Atom, \
         /*     sigma = */ 2.75*ATom, \
         /*   kbOveps = */ 1.0/80.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// O2
#define iO2 3
#define     O2 { \
   /*      Name = */       ("O2"), \
   /*         W = */       0.0159994*2, \
   /*       inx = */       iO2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.282537840E+00,  1.483087540E-03, -7.579666690E-07,  2.094705550E-10, -2.167177940E-14, -1.088457720E+03,  5.453231290E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.782456360E+00, -2.996734160E-03,  9.847302010E-06, -9.681295090E-09,  3.243728370E-12, -1.063943560E+03,  3.657675730E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.458*ATom, \
         /*   kbOveps = */ 1.0/107.4, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 1.6*ATom*ATom*ATom, \
         /*      Z298 = */ 3.8 \
                              } \
               }

// OH
#define iOH 4
#define     OH { \
   /*      Name = */       ("OH"), \
   /*         W = */       0.00100784*1+0.0159994*1, \
   /*       inx = */       iOH, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.092887670E+00,  5.484297160E-04,  1.265052280E-07, -8.794615560E-11,  1.174123760E-14,  3.858657000E+03,  4.476696100E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.992015430E+00, -2.401317520E-03,  4.617938410E-06, -3.881133330E-09,  1.364114700E-12,  3.615080560E+03, -1.039254580E-01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 2.75*ATom, \
         /*   kbOveps = */ 1.0/80.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// H2O
#define iH2O 5
#define    H2O { \
   /*      Name = */       ("H2O"), \
   /*         W = */       0.00100784*2+0.0159994*1, \
   /*       inx = */       iH2O, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.033992490E+00,  2.176918040E-03, -1.640725180E-07, -9.704198700E-11,  1.682009920E-14, -3.000429710E+04,  4.966770100E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.198640560E+00, -2.036434100E-03,  6.520402110E-06, -5.487970620E-09,  1.771978170E-12, -3.029372670E+04, -8.490322080E-01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 2.605*ATom, \
         /*   kbOveps = */ 1.0/572.4, \
         /*        mu = */ 1.844*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 4.0 \
                              } \
               }

// HO2
#define iHO2 6
#define    HO2 { \
   /*      Name = */       ("HO2"), \
   /*         W = */       0.00100784*1+0.0159994*2, \
   /*       inx = */       iHO2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  4.017210900E+00,  2.239820130E-03, -6.336581500E-07,  1.142463700E-10, -1.079085350E-14,  1.118567130E+02,  3.785102150E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.301798010E+00, -4.749120510E-03,  2.115828910E-05, -2.427638940E-08,  9.292251240E-12,  2.948080400E+02,  3.716662450E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.458*ATom, \
         /*   kbOveps = */ 1.0/107.4, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.0 \
                              } \
               }

// H2O2
#define iH2O2 7
#define   H2O2 { \
   /*      Name = */       ("H2O2"), \
   /*         W = */       0.00100784*2+0.0159994*2, \
   /*       inx = */       iH2O2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  4.165002850E+00,  4.908316940E-03, -1.901392250E-06,  3.711859860E-10, -2.879083050E-14, -1.786178770E+04,  2.916156620E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.276112690E+00, -5.428224170E-04,  1.673357010E-05, -2.157708130E-08,  8.624543630E-12, -1.770258210E+04,  3.435050740E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.458*ATom, \
         /*   kbOveps = */ 1.0/107.4, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 3.8 \
                              } \
               }

// C
#define iC 8
#define      C { \
   /*      Name = */       ("C"), \
   /*         W = */       0.0120107*1, \
   /*       inx = */       iC, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.492668880E+00,  4.798892840E-05, -7.243350200E-08,  3.742910290E-11, -4.872778930E-15,  8.545129530E+04,  4.801503730E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.554239550E+00, -3.215377240E-04,  7.337922450E-07, -7.322348890E-10,  2.665214460E-13,  8.544388320E+04,  4.531308480E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Atom, \
         /*     sigma = */ 3.298*ATom, \
         /*   kbOveps = */ 1.0/71.4, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// CH
#define iCH 9
#define     CH { \
   /*      Name = */       ("CH"), \
   /*         W = */       0.0120107*1+0.00100784*1, \
   /*       inx = */       iCH, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.878464730E+00,  9.709136810E-04,  1.444456550E-07, -1.306878490E-10,  1.760793830E-14,  7.101243640E+04,  5.484979990E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.489816650E+00,  3.238355410E-04, -1.688990650E-06,  3.162173270E-09, -1.406090670E-12,  7.079729340E+04,  2.084011080E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 2.75*ATom, \
         /*   kbOveps = */ 1.0/80.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// CH2
#define iCH2 10
#define    CH2 { \
   /*      Name = */       ("CH2"), \
   /*         W = */       0.0120107*1+0.00100784*2, \
   /*       inx = */       iCH2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.874101130E+00,  3.656392920E-03, -1.408945970E-06,  2.601795490E-10, -1.877275670E-14,  4.626360400E+04,  6.171193240E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.762678670E+00,  9.688721430E-04,  2.794898410E-06, -3.850911530E-09,  1.687417190E-12,  4.600404010E+04,  1.562531850E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.8*ATom, \
         /*   kbOveps = */ 1.0/144.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// CH2(S)
#define iCH2_S 11
#define  CH2_S { \
   /*      Name = */       ("CH2(S)"), \
   /*         W = */       0.0120107*1+0.00100784*2, \
   /*       inx = */       iCH2_S, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.292038420E+00,  4.655886370E-03, -2.011919470E-06,  4.179060000E-10, -3.397163650E-14,  5.092599970E+04,  8.626501690E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.198604110E+00, -2.366614190E-03,  8.232962200E-06, -6.688159810E-09,  1.943147370E-12,  5.049681630E+04, -7.691189670E-01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.8*ATom, \
         /*   kbOveps = */ 1.0/144.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// CH3
#define iCH3 12
#define    CH3 { \
   /*      Name = */       ("CH3"), \
   /*         W = */       0.0120107*1+0.00100784*3, \
   /*       inx = */       iCH3, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.285717720E+00,  7.239900370E-03, -2.987143480E-06,  5.956846440E-10, -4.671543940E-14,  1.677558430E+04,  8.480071790E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.673590400E+00,  2.010951750E-03,  5.730218560E-06, -6.871174250E-09,  2.543857340E-12,  1.644499880E+04,  1.604564330E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.8*ATom, \
         /*   kbOveps = */ 1.0/144.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// CH4
#define iCH4 13
#define    CH4 { \
   /*      Name = */       ("CH4"), \
   /*         W = */       0.0120107*1+0.00100784*4, \
   /*       inx = */       iCH4, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  7.485149500E-02,  1.339094670E-02, -5.732858090E-06,  1.222925350E-09, -1.018152300E-13, -9.468344590E+03,  1.843731800E+01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  5.149876130E+00, -1.367097880E-02,  4.918005990E-05, -4.847430260E-08,  1.666939560E-11, -1.024664760E+04, -4.641303760E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.746*ATom, \
         /*   kbOveps = */ 1.0/141.4, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 2.6*ATom*ATom*ATom, \
         /*      Z298 = */ 13.0 \
                              } \
               }

// CO
#define iCO 14
#define     CO { \
   /*      Name = */       ("CO"), \
   /*         W = */       0.0120107*1+0.0159994*1, \
   /*       inx = */       iCO, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.715185610E+00,  2.062527430E-03, -9.988257710E-07,  2.300530080E-10, -2.036477160E-14, -1.415187240E+04,  7.818687720E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.579533470E+00, -6.103536800E-04,  1.016814330E-06,  9.070058840E-10, -9.044244990E-13, -1.434408600E+04,  3.508409280E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.65*ATom, \
         /*   kbOveps = */ 1.0/98.1, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 1.95*ATom*ATom*ATom, \
         /*      Z298 = */ 1.8 \
                              } \
               }

// CO2
#define iCO2 15
#define    CO2 { \
   /*      Name = */       ("CO2"), \
   /*         W = */       0.0120107*1+0.0159994*2, \
   /*       inx = */       iCO2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.857460290E+00,  4.414370260E-03, -2.214814040E-06,  5.234901880E-10, -4.720841640E-14, -4.875916600E+04,  2.271638060E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.356773520E+00,  8.984596770E-03, -7.123562690E-06,  2.459190220E-09, -1.436995480E-13, -4.837196970E+04,  9.901052220E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.763*ATom, \
         /*   kbOveps = */ 1.0/244.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 2.65*ATom*ATom*ATom, \
         /*      Z298 = */ 2.1 \
                              } \
               }

// HCO
#define iHCO 16
#define    HCO { \
   /*      Name = */       ("HCO"), \
   /*         W = */       0.0120107*1+0.00100784*1+0.0159994*1, \
   /*       inx = */       iHCO, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.772174380E+00,  4.956955260E-03, -2.484456130E-06,  5.891617780E-10, -5.335087110E-14,  4.011918150E+03,  9.798344920E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.221185840E+00, -3.243925320E-03,  1.377994460E-05, -1.331440930E-08,  4.337688650E-12,  3.839564960E+03,  3.394372430E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.59*ATom, \
         /*   kbOveps = */ 1.0/498.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 0.0 \
                              } \
               }

// CH2O
#define iCH2O 17
#define   CH2O { \
   /*      Name = */       ("CH2O"), \
   /*         W = */       0.0120107*1+0.00100784*2+0.0159994*1, \
   /*       inx = */       iCH2O, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  1.760690080E+00,  9.200000820E-03, -4.422588130E-06,  1.006412120E-09, -8.838556400E-14, -1.399583230E+04,  1.365632300E+01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.793723150E+00, -9.908333690E-03,  3.732200080E-05, -3.792852610E-08,  1.317726520E-11, -1.430895670E+04,  6.028129000E-01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.59*ATom, \
         /*   kbOveps = */ 1.0/498.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 2.0 \
                              } \
               }

// CH2OH
#define iCH2OH 18
#define  CH2OH { \
   /*      Name = */       ("CH2OH"), \
   /*         W = */       0.0120107*1+0.00100784*3+0.0159994*1, \
   /*       inx = */       iCH2OH, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.692665690E+00,  8.645767970E-03, -3.751011200E-06,  7.872346360E-10, -6.485542010E-14, -3.242506270E+03,  5.810432150E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.863889180E+00,  5.596723040E-03,  5.932717910E-06, -1.045320120E-08,  4.369672780E-12, -3.193913670E+03,  5.473022430E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.69*ATom, \
         /*   kbOveps = */ 1.0/417.0, \
         /*        mu = */ 1.7*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 2.0 \
                              } \
               }

// CH3O
#define iCH3O 19
#define   CH3O { \
   /*      Name = */       ("CH3O"), \
   /*         W = */       0.0120107*1+0.00100784*3+0.0159994*1, \
   /*       inx = */       iCH3O, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3000.0, \
      /*         TMin = */          300.0, \
      /*         TMax = */          3000.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.770799000E+00,  7.871497000E-03, -2.656384000E-06,  3.944431000E-10, -2.112616000E-14,  1.278325200E+02,  2.929575000E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.106204000E+00,  7.216595000E-03,  5.338472000E-06, -7.377636000E-09,  2.075610000E-12,  9.786011000E+02,  1.315217700E+01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.69*ATom, \
         /*   kbOveps = */ 1.0/417.0, \
         /*        mu = */ 1.7*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 2.0 \
                              } \
               }

// CH3OH
#define iCH3OH 20
#define  CH3OH { \
   /*      Name = */       ("CH3OH"), \
   /*         W = */       0.0120107*1+0.00100784*4+0.0159994*1, \
   /*       inx = */       iCH3OH, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  1.789707910E+00,  1.409382920E-02, -6.365008350E-06,  1.381710850E-09, -1.170602200E-13, -2.537487470E+04,  1.450236230E+01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  5.715395820E+00, -1.523091290E-02,  6.524411550E-05, -7.108068890E-08,  2.613526980E-11, -2.564276560E+04, -1.504098230E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.626*ATom, \
         /*   kbOveps = */ 1.0/481.8, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.0 \
                              } \
               }

// C2H2
#define iC2H2 21
#define   C2H2 { \
   /*      Name = */       ("C2H2"), \
   /*         W = */       0.0120107*2+0.00100784*2, \
   /*       inx = */       iC2H2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  4.147569640E+00,  5.961666640E-03, -2.372948520E-06,  4.674121710E-10, -3.612352130E-14,  2.593599920E+04, -1.230281210E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  8.086810940E-01,  2.336156290E-02, -3.551718150E-05,  2.801524370E-08, -8.500729740E-12,  2.642898070E+04,  1.393970510E+01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 4.1*ATom, \
         /*   kbOveps = */ 1.0/209.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 2.5 \
                              } \
               }

// C2H3
#define iC2H3 22
#define   C2H3 { \
   /*      Name = */       ("C2H3"), \
   /*         W = */       0.0120107*2+0.00100784*3, \
   /*       inx = */       iC2H3, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  3.016724000E+00,  1.033022920E-02, -4.680823490E-06,  1.017632880E-09, -8.626070410E-14,  3.461287390E+04,  7.787323780E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.212466450E+00,  1.514791620E-03,  2.592094120E-05, -3.576578470E-08,  1.471508730E-11,  3.485984680E+04,  8.510540250E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 4.1*ATom, \
         /*   kbOveps = */ 1.0/209.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.0 \
                              } \
               }

// C2H4
#define iC2H4 23
#define   C2H4 { \
   /*      Name = */       ("C2H4"), \
   /*         W = */       0.0120107*2+0.00100784*4, \
   /*       inx = */       iC2H4, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.036111160E+00,  1.464541510E-02, -6.710779150E-06,  1.472229230E-09, -1.257060610E-13,  4.939886140E+03,  1.030536930E+01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.959201480E+00, -7.570522470E-03,  5.709902920E-05, -6.915887530E-08,  2.698843730E-11,  5.089775930E+03,  4.097330960E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.971*ATom, \
         /*   kbOveps = */ 1.0/280.8, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.5 \
                              } \
               }

// C2H5
#define iC2H5 24
#define   C2H5 { \
   /*      Name = */       ("C2H5"), \
   /*         W = */       0.0120107*2+0.00100784*5, \
   /*       inx = */       iC2H5, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  1.954656420E+00,  1.739727220E-02, -7.982066680E-06,  1.752176890E-09, -1.496415760E-13,  1.285752000E+04,  1.346243430E+01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.306465680E+00, -4.186588920E-03,  4.971428070E-05, -5.991266060E-08,  2.305090040E-11,  1.284162650E+04,  4.707209240E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 4.302*ATom, \
         /*   kbOveps = */ 1.0/252.3, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.5 \
                              } \
               }

// C2H6
#define iC2H6 25
#define   C2H6 { \
   /*      Name = */       ("C2H6"), \
   /*         W = */       0.0120107*2+0.00100784*6, \
   /*       inx = */       iC2H6, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  1.071881500E+00,  2.168526770E-02, -1.002560670E-05,  2.214120010E-09, -1.900028900E-13, -1.142639320E+04,  1.511561070E+01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  4.291424920E+00, -5.501542700E-03,  5.994382880E-05, -7.084662850E-08,  2.686857710E-11, -1.152220550E+04,  2.666823160E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 4.302*ATom, \
         /*   kbOveps = */ 1.0/252.3, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.5 \
                              } \
               }

// HCCO
#define iHCCO 26
#define   HCCO { \
   /*      Name = */       ("HCCO"), \
   /*         W = */       0.0120107*2+0.00100784*1+0.0159994*1, \
   /*       inx = */       iHCCO, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          4000.0, \
      /*         TMin = */          300.0, \
      /*         TMax = */          4000.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  5.628205800E+00,  4.085340100E-03, -1.593454700E-06,  2.862605200E-10, -1.940783200E-14,  1.932721500E+04, -3.930259500E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.251721400E+00,  1.765502100E-02, -2.372910100E-05,  1.727575900E-08, -5.066481100E-12,  2.005944900E+04,  1.249041700E+01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 2.5*ATom, \
         /*   kbOveps = */ 1.0/150.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 1.0 \
                              } \
               }

// CH2CO
#define iCH2CO 27
#define  CH2CO { \
   /*      Name = */       ("CH2CO"), \
   /*         W = */       0.0120107*2+0.00100784*2+0.0159994*1, \
   /*       inx = */       iCH2CO, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          3500.0, \
      /*         TMin = */          200.0, \
      /*         TMax = */          3500.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  4.511297320E+00,  9.003597450E-03, -4.169396350E-06,  9.233458820E-10, -7.948382010E-14, -7.551053110E+03,  6.322472050E-01 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  2.135836300E+00,  1.811887210E-02, -1.739474740E-05,  9.343975680E-09, -2.014576150E-12, -7.042918040E+03,  1.221564800E+01 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.97*ATom, \
         /*   kbOveps = */ 1.0/436.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 2.0 \
                              } \
               }

// CH2CHO
#define iCH2CHO 28
#define CH2CHO { \
   /*      Name = */       ("CH2CHO"), \
   /*         W = */       0.0120107*2+0.00100784*3+0.0159994*1, \
   /*       inx = */       iCH2CHO, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          5000.0, \
      /*         TMin = */          300.0, \
      /*         TMax = */          5000.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  5.975670000E+00,  8.130591000E-03, -2.743624000E-06,  4.070304000E-10, -2.176017000E-14,  4.903218000E+02, -5.045251000E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.409062000E+00,  1.073857400E-02,  1.891492000E-06, -7.158583000E-09,  2.867385000E-12,  1.521476600E+03,  9.558290000E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_NonLinear, \
         /*     sigma = */ 3.97*ATom, \
         /*   kbOveps = */ 1.0/436.0, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 0.0*ATom*ATom*ATom, \
         /*      Z298 = */ 2.0 \
                              } \
               }

// N2
#define iN2 29
#define     N2 { \
   /*      Name = */       ("N2"), \
   /*         W = */       0.0140067*2, \
   /*       inx = */       iN2, \
   /*    cpCoeff = */       {  \
      /*     TSwitch1 = */          1000.0, \
      /*     TSwitch2 = */          5000.0, \
      /*         TMin = */          300.0, \
      /*         TMax = */          5000.0, \
      /*          cpM = */          {  0.000000000E+00,  0.000000000E+00,  2.926640000E+00,  1.487976800E-03, -5.684760000E-07,  1.009703800E-10, -6.753351000E-15, -9.227977000E+02,  5.980528000E+00 }, \
      /*          cpL = */          {  0.000000000E+00,  0.000000000E+00,  3.298677000E+00,  1.408240400E-03, -3.963222000E-06,  5.641515000E-09, -2.444854000E-12, -1.020899900E+03,  3.950372000E+00 }, \
                                }, \
      /* DiffCoeff = */       {  \
         /*      Geom = */ SpeciesGeom_Linear, \
         /*     sigma = */ 3.621*ATom, \
         /*   kbOveps = */ 1.0/97.53, \
         /*        mu = */ 0.0*DToCm, \
         /*     alpha = */ 1.76*ATom*ATom*ATom, \
         /*      Z298 = */ 4.0 \
                              } \
               }


//---------------------------------
// Define Reactions
//---------------------------------

// R 1: 2 O <=> O2
#define R1 { \
      /* ArrCoeff  = */   {  \
         /*           A = */   1.200000000e+05, \
         /*           n = */   -1.000000000e+00, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.75 }, \
      { /* ind = */ iCO2, /* eff = */ 3.6 }, \
      { /* ind = */ iH2 , /* eff = */ 2.4 }, \
      { /* ind = */ iH2O, /* eff = */ 15.4 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          } \
           }

// R 2: O + H <=> OH
#define R2 { \
      /* ArrCoeff  = */   {  \
         /*           A = */   5.000000000e+05, \
         /*           n = */   -1.000000000e+00, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          } \
           }

// R 3: O + H2 <=> H + OH
#define R3 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.870000000e-02, \
      /*           n = */   2.700000000e+00, \
      /*        EovR = */   3.150155347e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 4: O + HO2 <=> OH + O2
#define R4 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 5: O + H2O2 <=> OH + HO2
#define R5 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   9.630000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   2.012878816e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 6: O + CH <=> H + CO
#define R6 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.700000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 7: O + CH2 <=> H + HCO
#define R7 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   8.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 8: O + CH2(S) <=> H2 + CO
#define R8 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 9: O + CH2(S) <=> H + HCO
#define R9 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 10: O + CH3 <=> H + CH2O
#define R10 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.060000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 11: O + CH4 <=> OH + CH3
#define R11 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.020000000e+03, \
      /*           n = */   1.500000000e+00, \
      /*        EovR = */   4.327689455e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 12: O + CO (+ M) <=> CO2 (+ M)
#define R12 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   6.020000000e+02, \
         /*           n = */   0.000000000e+00, \
         /*        EovR = */   1.509659112e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   1.800000000e+04, \
         /*           n = */   0.000000000e+00, \
         /*        EovR = */   1.200178994e+03, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   7, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 3.5 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ iO2 , /* eff = */ 6.0 } \
                          }, \
   /*      Ftype = */     F_Lindemann, \
   /*     FOData = */     { .Lindemann = { \
   /*            dummy = */     0, \
                          } } \
           }

// R 13: O + HCO <=> OH + CO
#define R13 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 14: O + HCO <=> H + CO2
#define R14 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 15: O + CH2O <=> OH + HCO
#define R15 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.900000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.781397752e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 16: O + CH2OH <=> OH + CH2O
#define R16 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 17: O + CH3O <=> OH + CH2O
#define R17 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 18: O + CH3OH <=> OH + CH2OH
#define R18 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.880000000e-01, \
      /*           n = */   2.500000000e+00, \
      /*        EovR = */   1.559981083e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 19: O + CH3OH <=> OH + CH3O
#define R19 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.300000000e-01, \
      /*           n = */   2.500000000e+00, \
      /*        EovR = */   2.516098520e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 20: O + C2H2 <=> H + HCCO
#define R20 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.350000000e+01, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   9.561174377e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 21: O + C2H2 <=> CO + CH2
#define R21 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.940000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   9.561174377e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 22: O + C2H3 <=> H + CH2CO
#define R22 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 23: O + C2H4 <=> CH3 + HCO
#define R23 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.250000000e+01, \
      /*           n = */   1.830000000e+00, \
      /*        EovR = */   1.107083349e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 24: O + C2H5 <=> CH3 + CH2O
#define R24 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.240000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 25: O + C2H6 <=> OH + C2H5
#define R25 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   8.980000000e+01, \
      /*           n = */   1.920000000e+00, \
      /*        EovR = */   2.863320116e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 26: O + HCCO <=> H + 2 CO
#define R26 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 27: O + CH2CO <=> OH + HCCO
#define R27 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   4.025757633e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 28: O + CH2CO <=> CH2 + CO2
#define R28 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.750000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   6.793466005e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 29: O2 + CO <=> O + CO2
#define R29 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.500000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   2.405390185e+04, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 30: O2 + CH2O <=> HO2 + HCO
#define R30 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   2.012878816e+04, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 31: H + O2 <=> HO2
#define R31 { \
      /* ArrCoeff  = */   {  \
         /*           A = */   2.800000000e+06, \
         /*           n = */   -8.600000000e-01, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 1.5 }, \
      { /* ind = */ iCO , /* eff = */ 0.75 }, \
      { /* ind = */ iCO2, /* eff = */ 1.5 }, \
      { /* ind = */ iH2O, /* eff = */ 0.0 }, \
      { /* ind = */ iN2 , /* eff = */ 0.0 }, \
      { /* ind = */ iO2 , /* eff = */ 0.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          } \
           }

// R 32: H + 2 O2 <=> HO2 + O2
#define R32 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.080000000e+07, \
      /*           n = */   -1.240000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 33: H + O2 + H2O <=> HO2 + H2O
#define R33 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.126000000e+07, \
      /*           n = */   -7.600000000e-01, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   3, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 34: H + O2 + N2 <=> HO2 + N2
#define R34 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.600000000e+07, \
      /*           n = */   -1.240000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   3, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iN2 , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iN2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 35: H + O2 <=> O + OH
#define R35 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.650000000e+10, \
      /*           n = */   -6.707000000e-01, \
      /*        EovR = */   8.575366977e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 36: 2 H <=> H2
#define R36 { \
      /* ArrCoeff  = */   {  \
         /*           A = */   1.000000000e+06, \
         /*           n = */   -1.000000000e+00, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   5, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO2, /* eff = */ 0.0 }, \
      { /* ind = */ iH2 , /* eff = */ 0.0 }, \
      { /* ind = */ iH2O, /* eff = */ 0.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          } \
           }

// R 37: 2 H + H2 <=> 2 H2
#define R37 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   9.000000000e+04, \
      /*           n = */   -6.000000000e-01, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 2 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 38: 2 H + H2O <=> H2 + H2O
#define R38 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.000000000e+07, \
      /*           n = */   -1.250000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 2 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 39: 2 H + CO2 <=> H2 + CO2
#define R39 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.500000000e+08, \
      /*           n = */   -2.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 2 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 40: H + OH <=> H2O
#define R40 { \
      /* ArrCoeff  = */   {  \
         /*           A = */   2.200000000e+10, \
         /*           n = */   -2.000000000e+00, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   4, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 0.73 }, \
      { /* ind = */ iH2O, /* eff = */ 3.65 }, \
      { /* ind = */ 0   , /* eff = */ 1 }, \
      { /* ind = */ 0   , /* eff = */ 1 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          } \
           }

// R 41: H + HO2 <=> O + H2O
#define R41 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.970000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   3.376604214e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 42: H + HO2 <=> O2 + H2
#define R42 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.480000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   5.374386439e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 43: H + HO2 <=> 2 OH
#define R43 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   8.400000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   3.195445121e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 44: H + H2O2 <=> HO2 + H2
#define R44 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.210000000e+01, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   2.616742461e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 45: H + H2O2 <=> OH + H2O
#define R45 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.811590935e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 46: H + CH <=> C + H2
#define R46 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.650000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC  , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 47: H + CH2 (+ M) <=> CH3 (+ M)
#define R47 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.040000000e+14, \
         /*           n = */   -2.760000000e+00, \
         /*        EovR = */   8.051515265e+02, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   6.000000000e+08, \
         /*           n = */   0.000000000e+00, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     5.620000000e-01, \
/*               T1 = */     5.836000000e+03, \
/*               T2 = */     8.552000000e+03, \
/*               T3 = */     9.100000000e+01  \
                       }} \
           }

// R 48: H + CH2(S) <=> CH + H2
#define R48 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 49: H + CH3 (+ M) <=> CH4 (+ M)
#define R49 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   2.620000000e+21, \
         /*           n = */   -4.760000000e+00, \
         /*        EovR = */   1.227856078e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   1.390000000e+10, \
         /*           n = */   -5.340000000e-01, \
         /*        EovR = */   2.697257614e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 3.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.830000000e-01, \
/*               T1 = */     2.941000000e+03, \
/*               T2 = */     6.964000000e+03, \
/*               T3 = */     7.400000000e+01  \
                       }} \
           }

// R 50: H + CH4 <=> CH3 + H2
#define R50 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.600000000e+02, \
      /*           n = */   1.620000000e+00, \
      /*        EovR = */   5.454901592e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 51: H + HCO (+ M) <=> CH2O (+ M)
#define R51 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   2.470000000e+12, \
         /*           n = */   -2.570000000e+00, \
         /*        EovR = */   2.138683742e+02, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   1.090000000e+06, \
         /*           n = */   4.800000000e-01, \
         /*        EovR = */   -1.308371231e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.824000000e-01, \
/*               T1 = */     2.755000000e+03, \
/*               T2 = */     6.570000000e+03, \
/*               T3 = */     2.710000000e+02  \
                       }} \
           }

// R 52: H + HCO <=> H2 + CO
#define R52 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   7.340000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 53: H + CH2O (+ M) <=> CH2OH (+ M)
#define R53 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.270000000e+20, \
         /*           n = */   -4.820000000e+00, \
         /*        EovR = */   3.286024668e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   5.400000000e+05, \
         /*           n = */   4.540000000e-01, \
         /*        EovR = */   1.811590935e+03, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.187000000e-01, \
/*               T1 = */     1.291000000e+03, \
/*               T2 = */     4.160000000e+03, \
/*               T3 = */     1.030000000e+02  \
                       }} \
           }

// R 54: H + CH2O (+ M) <=> CH3O (+ M)
#define R54 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   2.200000000e+18, \
         /*           n = */   -4.800000000e+00, \
         /*        EovR = */   2.797901555e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   5.400000000e+05, \
         /*           n = */   4.540000000e-01, \
         /*        EovR = */   1.308371231e+03, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.580000000e-01, \
/*               T1 = */     1.555000000e+03, \
/*               T2 = */     4.200000000e+03, \
/*               T3 = */     9.400000000e+01  \
                       }} \
           }

// R 55: H + CH2O <=> HCO + H2
#define R55 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.740000000e+01, \
      /*           n = */   1.900000000e+00, \
      /*        EovR = */   1.379828429e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 56: H + CH2OH (+ M) <=> CH3OH (+ M)
#define R56 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   4.360000000e+19, \
         /*           n = */   -4.650000000e+00, \
         /*        EovR = */   2.556356097e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   1.055000000e+06, \
         /*           n = */   5.000000000e-01, \
         /*        EovR = */   4.327689455e+01, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     6.000000000e-01, \
/*               T1 = */     9.000000000e+04, \
/*               T2 = */     1.000000000e+04, \
/*               T3 = */     1.000000000e+02  \
                       }} \
           }

// R 57: H + CH2OH <=> H2 + CH2O
#define R57 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 58: H + CH2OH <=> OH + CH3
#define R58 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.650000000e+05, \
      /*           n = */   6.500000000e-01, \
      /*        EovR = */   -1.429143960e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 59: H + CH2OH <=> CH2(S) + H2O
#define R59 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.280000000e+07, \
      /*           n = */   -9.000000000e-02, \
      /*        EovR = */   3.069640195e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 60: H + CH3O (+ M) <=> CH3OH (+ M)
#define R60 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   4.660000000e+29, \
         /*           n = */   -7.440000000e+00, \
         /*        EovR = */   7.085333433e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   2.430000000e+06, \
         /*           n = */   5.150000000e-01, \
         /*        EovR = */   2.516098520e+01, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.000000000e-01, \
/*               T1 = */     9.000000000e+04, \
/*               T2 = */     1.000000000e+04, \
/*               T3 = */     1.000000000e+02  \
                       }} \
           }

// R 61: H + CH3O <=> H + CH2OH
#define R61 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.150000000e+01, \
      /*           n = */   1.630000000e+00, \
      /*        EovR = */   9.681947106e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 62: H + CH3O <=> H2 + CH2O
#define R62 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 63: H + CH3O <=> OH + CH3
#define R63 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+06, \
      /*           n = */   5.000000000e-01, \
      /*        EovR = */   -5.535416745e+01, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 64: H + CH3O <=> CH2(S) + H2O
#define R64 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.620000000e+08, \
      /*           n = */   -2.300000000e-01, \
      /*        EovR = */   5.384450833e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 65: H + CH3OH <=> CH2OH + H2
#define R65 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.700000000e+01, \
      /*           n = */   2.100000000e+00, \
      /*        EovR = */   2.450679959e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 66: H + CH3OH <=> CH3O + H2
#define R66 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.200000000e+00, \
      /*           n = */   2.100000000e+00, \
      /*        EovR = */   2.450679959e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 67: H + C2H2 (+ M) <=> C2H3 (+ M)
#define R67 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   3.800000000e+28, \
         /*           n = */   -7.270000000e+00, \
         /*        EovR = */   3.633246263e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   5.600000000e+06, \
         /*           n = */   0.000000000e+00, \
         /*        EovR = */   1.207727290e+03, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.507000000e-01, \
/*               T1 = */     1.302000000e+03, \
/*               T2 = */     4.167000000e+03, \
/*               T3 = */     9.850000000e+01  \
                       }} \
           }

// R 68: H + C2H3 (+ M) <=> C2H4 (+ M)
#define R68 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.400000000e+18, \
         /*           n = */   -3.860000000e+00, \
         /*        EovR = */   1.670689417e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   6.080000000e+06, \
         /*           n = */   2.700000000e-01, \
         /*        EovR = */   1.409015171e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.820000000e-01, \
/*               T1 = */     2.663000000e+03, \
/*               T2 = */     6.095000000e+03, \
/*               T3 = */     2.075000000e+02  \
                       }} \
           }

// R 69: H + C2H3 <=> H2 + C2H2
#define R69 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 70: H + C2H4 (+ M) <=> C2H5 (+ M)
#define R70 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   6.000000000e+29, \
         /*           n = */   -7.620000000e+00, \
         /*        EovR = */   3.507441337e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   5.400000000e+05, \
         /*           n = */   4.540000000e-01, \
         /*        EovR = */   9.158598614e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     9.753000000e-01, \
/*               T1 = */     9.840000000e+02, \
/*               T2 = */     4.374000000e+03, \
/*               T3 = */     2.100000000e+02  \
                       }} \
           }

// R 71: H + C2H4 <=> C2H3 + H2
#define R71 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.325000000e+00, \
      /*           n = */   2.530000000e+00, \
      /*        EovR = */   6.159409178e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 72: H + C2H5 (+ M) <=> C2H6 (+ M)
#define R72 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.990000000e+29, \
         /*           n = */   -7.080000000e+00, \
         /*        EovR = */   3.364023722e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   5.210000000e+11, \
         /*           n = */   -9.900000000e-01, \
         /*        EovR = */   7.950871324e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     8.422000000e-01, \
/*               T1 = */     2.219000000e+03, \
/*               T2 = */     6.882000000e+03, \
/*               T3 = */     1.250000000e+02  \
                       }} \
           }

// R 73: H + C2H5 <=> H2 + C2H4
#define R73 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 74: H + C2H6 <=> C2H5 + H2
#define R74 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.150000000e+02, \
      /*           n = */   1.900000000e+00, \
      /*        EovR = */   3.789244372e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 75: H + HCCO <=> CH2(S) + CO
#define R75 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 76: H + CH2CO <=> HCCO + H2
#define R76 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   4.025757633e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 77: H + CH2CO <=> CH3 + CO
#define R77 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.130000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.725037146e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 78: H2 + CO (+ M) <=> CH2O (+ M)
#define R78 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   5.070000000e+15, \
         /*           n = */   -3.420000000e+00, \
         /*        EovR = */   4.244658204e+04, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   4.300000000e+01, \
         /*           n = */   1.500000000e+00, \
         /*        EovR = */   4.005628844e+04, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     9.320000000e-01, \
/*               T1 = */     1.540000000e+03, \
/*               T2 = */     1.030000000e+04, \
/*               T3 = */     1.970000000e+02  \
                       }} \
           }

// R 79: OH + H2 <=> H + H2O
#define R79 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.160000000e+02, \
      /*           n = */   1.510000000e+00, \
      /*        EovR = */   1.726043585e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 80: 2 OH (+ M) <=> H2O2 (+ M)
#define R80 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   2.300000000e+06, \
         /*           n = */   -9.000000000e-01, \
         /*        EovR = */   -8.554734969e+02, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   7.400000000e+07, \
         /*           n = */   -3.700000000e-01, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.346000000e-01, \
/*               T1 = */     1.756000000e+03, \
/*               T2 = */     5.182000000e+03, \
/*               T3 = */     9.400000000e+01  \
                       }} \
           }

// R 81: 2 OH <=> O + H2O
#define R81 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.570000000e-02, \
      /*           n = */   2.400000000e+00, \
      /*        EovR = */   -1.061793576e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 82: OH + HO2 <=> O2 + H2O
#define R82 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.450000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -2.516098520e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 83: OH + H2O2 <=> HO2 + H2O
#define R83 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   2.148748136e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 84: OH + H2O2 <=> HO2 + H2O
#define R84 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.700000000e+12, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.479969150e+04, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 85: OH + C <=> H + CO
#define R85 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC  , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 86: OH + CH <=> H + HCO
#define R86 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 87: OH + CH2 <=> H + CH2O
#define R87 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 88: OH + CH2 <=> CH + H2O
#define R88 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.130000000e+01, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   1.509659112e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 89: OH + CH2(S) <=> H + CH2O
#define R89 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 90: OH + CH3 (+ M) <=> CH3OH (+ M)
#define R90 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   4.000000000e+24, \
         /*           n = */   -5.920000000e+00, \
         /*        EovR = */   1.580109871e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   2.790000000e+12, \
         /*           n = */   -1.430000000e+00, \
         /*        EovR = */   6.692822064e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     4.120000000e-01, \
/*               T1 = */     5.900000000e+03, \
/*               T2 = */     6.394000000e+03, \
/*               T3 = */     1.950000000e+02  \
                       }} \
           }

// R 91: OH + CH3 <=> CH2 + H2O
#define R91 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.600000000e+01, \
      /*           n = */   1.600000000e+00, \
      /*        EovR = */   2.727450796e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 92: OH + CH3 <=> CH2(S) + H2O
#define R92 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.440000000e+11, \
      /*           n = */   -1.340000000e+00, \
      /*        EovR = */   7.130623207e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 93: OH + CH4 <=> CH3 + H2O
#define R93 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+02, \
      /*           n = */   1.600000000e+00, \
      /*        EovR = */   1.570045477e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 94: OH + CO <=> H + CO2
#define R94 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.760000000e+01, \
      /*           n = */   1.228000000e+00, \
      /*        EovR = */   3.522537928e+01, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 95: OH + HCO <=> H2O + CO
#define R95 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 96: OH + CH2O <=> HCO + H2O
#define R96 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.430000000e+03, \
      /*           n = */   1.180000000e+00, \
      /*        EovR = */   -2.249392077e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 97: OH + CH2OH <=> H2O + CH2O
#define R97 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 98: OH + CH3O <=> H2O + CH2O
#define R98 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 99: OH + CH3OH <=> CH2OH + H2O
#define R99 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.440000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   -4.227045514e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 100: OH + CH3OH <=> CH3O + H2O
#define R100 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.300000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   7.548295561e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 101: OH + C2H2 <=> H + CH2CO
#define R101 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.180000000e-10, \
      /*           n = */   4.500000000e+00, \
      /*        EovR = */   -5.032197041e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 102: OH + C2H2 <=> CH3 + CO
#define R102 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.830000000e-10, \
      /*           n = */   4.000000000e+00, \
      /*        EovR = */   -1.006439408e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 103: OH + C2H3 <=> H2O + C2H2
#define R103 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 104: OH + C2H4 <=> C2H3 + H2O
#define R104 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.600000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   1.258049260e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 105: OH + C2H6 <=> C2H5 + H2O
#define R105 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.540000000e+00, \
      /*           n = */   2.120000000e+00, \
      /*        EovR = */   4.378011425e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 106: OH + CH2CO <=> HCCO + H2O
#define R106 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   7.500000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.006439408e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 107: 2 HO2 <=> O2 + H2O2
#define R107 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.300000000e+05, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -8.202481176e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 108: 2 HO2 <=> O2 + H2O2
#define R108 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.200000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   6.038636449e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 109: HO2 + CH2 <=> OH + CH2O
#define R109 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 110: HO2 + CH3 <=> O2 + CH4
#define R110 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 111: HO2 + CH3 <=> OH + CH3O
#define R111 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.780000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 112: HO2 + CO <=> OH + CO2
#define R112 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.187598502e+04, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 113: HO2 + CH2O <=> HCO + H2O2
#define R113 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.600000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   6.038636449e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 114: C + O2 <=> O + CO
#define R114 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.800000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   2.898545495e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iC  , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 115: C + CH3 <=> H + C2H2
#define R115 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iC  , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 116: CH + O2 <=> O + HCO
#define R116 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.710000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 117: CH + H2 <=> H + CH2
#define R117 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.080000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.565013280e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 118: CH + H2O <=> H + CH2O
#define R118 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.710000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -3.799308766e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 119: CH + CH2 <=> H + C2H2
#define R119 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 120: CH + CH3 <=> H + C2H3
#define R120 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 121: CH + CH4 <=> H + C2H4
#define R121 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 122: CH + CO (+ M) <=> HCCO (+ M)
#define R122 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   2.690000000e+16, \
         /*           n = */   -3.740000000e+00, \
         /*        EovR = */   9.742333471e+02, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   5.000000000e+07, \
         /*           n = */   0.000000000e+00, \
         /*        EovR = */   0.000000000e+00, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     5.757000000e-01, \
/*               T1 = */     1.652000000e+03, \
/*               T2 = */     5.069000000e+03, \
/*               T3 = */     2.370000000e+02  \
                       }} \
           }

// R 123: CH + CO2 <=> HCO + CO
#define R123 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.900000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   7.946845567e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 124: CH + CH2O <=> H + CH2CO
#define R124 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   9.460000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -2.591581476e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 125: CH + HCCO <=> CO + C2H2
#define R125 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 126: CH2 + O2 -> OH + H + CO
#define R126 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   7.548295561e+02, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   3, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 } \
                          } \
           }

// R 127: CH2 + H2 <=> H + CH3
#define R127 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e-01, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   3.638278460e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 128: 2 CH2 <=> H2 + C2H2
#define R128 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.600000000e+09, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   6.010456145e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 129: CH2 + CH3 <=> H + C2H4
#define R129 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 130: CH2 + CH4 <=> 2 CH3
#define R130 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.460000000e+00, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   4.161626953e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 131: CH2 + CO (+ M) <=> CH2CO (+ M)
#define R131 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   2.690000000e+21, \
         /*           n = */   -5.110000000e+00, \
         /*        EovR = */   3.570343800e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   8.100000000e+05, \
         /*           n = */   5.000000000e-01, \
         /*        EovR = */   2.269520865e+03, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     5.907000000e-01, \
/*               T1 = */     1.226000000e+03, \
/*               T2 = */     5.185000000e+03, \
/*               T3 = */     2.750000000e+02  \
                       }} \
           }

// R 132: CH2 + HCCO <=> C2H3 + CO
#define R132 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 133: CH2(S) + N2 <=> CH2 + N2
#define R133 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   3.019318224e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iN2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iN2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 134: CH2(S) + O2 <=> H + OH + CO
#define R134 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.800000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   3, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 } \
                          } \
           }

// R 135: CH2(S) + O2 <=> CO + H2O
#define R135 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.200000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 136: CH2(S) + H2 <=> CH3 + H
#define R136 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   7.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 137: CH2(S) + H2O (+ M) <=> CH3OH (+ M)
#define R137 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.880000000e+26, \
         /*           n = */   -6.360000000e+00, \
         /*        EovR = */   2.536227308e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   4.820000000e+11, \
         /*           n = */   -1.160000000e+00, \
         /*        EovR = */   5.761865612e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     6.027000000e-01, \
/*               T1 = */     3.922000000e+03, \
/*               T2 = */     1.018000000e+04, \
/*               T3 = */     2.080000000e+02  \
                       }} \
           }

// R 138: CH2(S) + H2O <=> CH2 + H2O
#define R138 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 139: CH2(S) + CH3 <=> H + C2H4
#define R139 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.200000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -2.868352313e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 140: CH2(S) + CH4 <=> 2 CH3
#define R140 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.600000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -2.868352313e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 141: CH2(S) + CO <=> CH2 + CO
#define R141 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   9.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 142: CH2(S) + CO2 <=> CH2 + CO2
#define R142 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   7.000000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 143: CH2(S) + CO2 <=> CO + CH2O
#define R143 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.400000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 144: CH2(S) + C2H6 <=> CH3 + C2H5
#define R144 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   -2.767708372e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 145: CH3 + O2 <=> O + CH3O
#define R145 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.560000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.533813658e+04, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 146: CH3 + O2 <=> OH + CH2O
#define R146 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.310000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.022290829e+04, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 147: CH3 + H2O2 <=> HO2 + CH4
#define R147 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.450000000e-02, \
      /*           n = */   2.470000000e+00, \
      /*        EovR = */   2.606678067e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iH2O2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 148: 2 CH3 (+ M) <=> C2H6 (+ M)
#define R148 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   3.400000000e+29, \
         /*           n = */   -7.030000000e+00, \
         /*        EovR = */   1.389892823e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   6.770000000e+10, \
         /*           n = */   -1.180000000e+00, \
         /*        EovR = */   3.291056865e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     6.190000000e-01, \
/*               T1 = */     1.180000000e+03, \
/*               T2 = */     9.999000000e+03, \
/*               T3 = */     7.320000000e+01  \
                       }} \
           }

// R 149: 2 CH3 <=> H + C2H5
#define R149 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.840000000e+06, \
      /*           n = */   1.000000000e-01, \
      /*        EovR = */   5.334128863e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 150: CH3 + HCO <=> CH4 + CO
#define R150 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.648000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 151: CH3 + CH2O <=> HCO + CH4
#define R151 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.320000000e-03, \
      /*           n = */   2.810000000e+00, \
      /*        EovR = */   2.948867466e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 152: CH3 + CH3OH <=> CH2OH + CH4
#define R152 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.000000000e+01, \
      /*           n = */   1.500000000e+00, \
      /*        EovR = */   5.002003858e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 153: CH3 + CH3OH <=> CH3O + CH4
#define R153 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+01, \
      /*           n = */   1.500000000e+00, \
      /*        EovR = */   5.002003858e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iCH3OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 154: CH3 + C2H4 <=> C2H3 + CH4
#define R154 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.270000000e-01, \
      /*           n = */   2.000000000e+00, \
      /*        EovR = */   4.629621277e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 155: CH3 + C2H6 <=> C2H5 + CH4
#define R155 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.140000000e+00, \
      /*           n = */   1.740000000e+00, \
      /*        EovR = */   5.258645907e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iC2H6, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ iCH4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 156: HCO + H2O <=> H + CO + H2O
#define R156 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+12, \
      /*           n = */   -1.000000000e+00, \
      /*        EovR = */   8.554734969e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   3, \
   /*       educts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 } \
                          } \
           }

// R 157: HCO <=> H + CO
#define R157 { \
      /* ArrCoeff  = */   {  \
         /*           A = */   1.870000000e+11, \
         /*           n = */   -1.000000000e+00, \
         /*        EovR = */   8.554734969e+03, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 0.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          } \
           }

// R 158: HCO + O2 <=> HO2 + CO
#define R158 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.345000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   2.012878816e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 159: CH2OH + O2 <=> HO2 + CH2O
#define R159 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.800000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   4.528977337e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 160: CH3O + O2 <=> HO2 + CH2O
#define R160 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.280000000e-19, \
      /*           n = */   7.600000000e+00, \
      /*        EovR = */   -1.776365555e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH3O, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 161: C2H3 + O2 <=> HCO + CH2O
#define R161 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   4.580000000e+10, \
      /*           n = */   -1.390000000e+00, \
      /*        EovR = */   5.107679996e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 162: C2H4 (+ M) <=> H2 + C2H2 (+ M)
#define R162 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.580000000e+45, \
         /*           n = */   -9.300000000e+00, \
         /*        EovR = */   4.921488706e+04, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   8.000000000e+12, \
         /*           n = */   4.400000000e-01, \
         /*        EovR = */   4.366437372e+04, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     7.345000000e-01, \
/*               T1 = */     1.035000000e+03, \
/*               T2 = */     5.417000000e+03, \
/*               T3 = */     1.800000000e+02  \
                       }} \
           }

// R 163: C2H5 + O2 <=> HO2 + C2H4
#define R163 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   8.400000000e+05, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   1.949976353e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iC2H5, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 164: HCCO + O2 <=> OH + 2 CO
#define R164 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.200000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   4.297496273e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHCCO, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 165: 2 HCCO <=> 2 CO + C2H2
#define R165 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.000000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   1, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iHCCO, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCO , /* nu = */ 2 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 166: O + CH3 -> H + H2 + CO
#define R166 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.370000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   3, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 } \
                          } \
           }

// R 167: O + C2H4 <=> H + CH2CHO
#define R167 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.700000000e+00, \
      /*           n = */   1.830000000e+00, \
      /*        EovR = */   1.107083349e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iC2H4, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 168: OH + HO2 <=> O2 + H2O
#define R168 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.000000000e+09, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   8.720797471e+03, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 169: OH + CH3 -> H2 + CH2O
#define R169 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   8.000000000e+03, \
      /*           n = */   5.000000000e-01, \
      /*        EovR = */   -8.831505806e+02, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 170: CH + H2 (+ M) <=> CH3 (+ M)
#define R170 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   4.820000000e+13, \
         /*           n = */   -2.800000000e+00, \
         /*        EovR = */   2.968996254e+02, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   1.970000000e+06, \
         /*           n = */   4.300000000e-01, \
         /*        EovR = */   -1.861912905e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iCH , /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     5.780000000e-01, \
/*               T1 = */     2.535000000e+03, \
/*               T2 = */     9.365000000e+03, \
/*               T3 = */     1.220000000e+02  \
                       }} \
           }

// R 171: CH2 + O2 -> 2 H + CO2
#define R171 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   5.800000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   7.548295561e+02, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 2 }, \
      { /* ind = */ iCO2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 172: CH2 + O2 <=> O + CH2O
#define R172 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.400000000e+06, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   7.548295561e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 173: CH2 + CH2 -> 2 H + C2H2
#define R173 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.000000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   5.529881328e+03, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 2 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 174: CH2(S) + H2O -> H2 + CH2O
#define R174 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   6.820000000e+04, \
      /*           n = */   2.500000000e-01, \
      /*        EovR = */   -4.705104233e+02, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iCH2_S, /* nu = */ 1 }, \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 175: C2H3 + O2 <=> O + CH2CHO
#define R175 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.030000000e+05, \
      /*           n = */   2.900000000e-01, \
      /*        EovR = */   5.535416745e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 176: C2H3 + O2 <=> HO2 + C2H2
#define R176 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.337000000e+00, \
      /*           n = */   1.610000000e+00, \
      /*        EovR = */   -1.932363664e+02, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iC2H3, /* nu = */ 1 }, \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHO2, /* nu = */ 1 }, \
      { /* ind = */ iC2H2, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 177: H + CH2CO (+ M) <=> CH2CHO (+ M)
#define R177 { \
      /* ArrCoeffL = */   {  \
         /*           A = */   1.012000000e+30, \
         /*           n = */   -7.630000000e+00, \
         /*        EovR = */   1.939408739e+03, \
                          },  \
      /* ArrCoeffH = */   {  \
         /*           A = */   4.865000000e+05, \
         /*           n = */   4.220000000e-01, \
         /*        EovR = */   -8.831505806e+02, \
                          },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   1, \
   /*      Nthirdb = */   6, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*      thirdb  = */   { \
      { /* ind = */ iC2H6, /* eff = */ 3.0 }, \
      { /* ind = */ iCH4, /* eff = */ 2.0 }, \
      { /* ind = */ iCO , /* eff = */ 1.5 }, \
      { /* ind = */ iCO2, /* eff = */ 2.0 }, \
      { /* ind = */ iH2 , /* eff = */ 2.0 }, \
      { /* ind = */ iH2O, /* eff = */ 6.0 }, \
      { /* ind = */ 0   , /* eff = */ 1 } \
                          }, \
/*      Ftype = */     F_Troe3, \
/*     FOData = */     { .Troe3 = { \
/*            alpha = */     4.650000000e-01, \
/*               T1 = */     1.773000000e+03, \
/*               T2 = */     5.333000000e+03, \
/*               T3 = */     2.010000000e+02  \
                       }} \
           }

// R 178: O + CH2CHO -> H + CH2 + CO2
#define R178 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.500000000e+08, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   3, \
   /*       educts = */   {  \
      { /* ind = */ iO  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2, /* nu = */ 1 }, \
      { /* ind = */ iCO2, /* nu = */ 1 } \
                          } \
           }

// R 179: O2 + CH2CHO -> OH + CO + CH2O
#define R179 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.810000000e+04, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   3, \
   /*       educts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCO , /* nu = */ 1 }, \
      { /* ind = */ iCH2O, /* nu = */ 1 } \
                          } \
           }

// R 180: O2 + CH2CHO -> OH + 2 HCO
#define R180 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.350000000e+04, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   false, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iO2 , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 2 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 181: H + CH2CHO <=> CH3 + HCO
#define R181 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   2.200000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH3, /* nu = */ 1 }, \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 182: H + CH2CHO <=> CH2CO + H2
#define R182 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.100000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iH  , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ iH2 , /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 183: OH + CH2CHO <=> H2O + CH2CO
#define R183 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   1.200000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iH2O, /* nu = */ 1 }, \
      { /* ind = */ iCH2CO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }

// R 184: OH + CH2CHO <=> HCO + CH2OH
#define R184 { \
   /* ArrCoeff  = */   {  \
      /*           A = */   3.010000000e+07, \
      /*           n = */   0.000000000e+00, \
      /*        EovR = */   0.000000000e+00, \
                       },  \
   /* has_backward = */   true, \
   /*      Neducts = */   2, \
   /*      Npducts = */   2, \
   /*       educts = */   {  \
      { /* ind = */ iOH , /* nu = */ 1 }, \
      { /* ind = */ iCH2CHO, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          }, \
   /*       pducts = */   {  \
      { /* ind = */ iHCO, /* nu = */ 1 }, \
      { /* ind = */ iCH2OH, /* nu = */ 1 }, \
      { /* ind = */ 0   , /* nu = */ 1 } \
                          } \
           }


//---------------------------------
// Initialization function
//---------------------------------

#ifndef __CUDACC__
inline Mix::Mix(const Config &config) :
species{H2, H, O, O2, OH, H2O, HO2, H2O2,
        C, CH, CH2, CH2_S, CH3, CH4, CO, CO2,
        HCO, CH2O, CH2OH, CH3O, CH3OH, C2H2, C2H3, C2H4,
        C2H5, C2H6, HCCO, CH2CO, CH2CHO, N2 },
reactions{R3, R4, R5, R6, R7, R8, R9, R10,
         R11, R13, R14, R15, R16, R17, R18, R19,
         R20, R21, R22, R23, R24, R25, R26, R27,
         R28, R29, R30, R32, R33, R34, R35, R37,
         R38, R39, R41, R42, R43, R44, R45, R46,
         R48, R50, R52, R55, R57, R58, R59, R61,
         R62, R63, R64, R65, R66, R69, R71, R73,
         R74, R75, R76, R77, R79, R81, R82, R83,
         R84, R85, R86, R87, R88, R89, R91, R92,
         R93, R94, R95, R96, R97, R98, R99, R100,
         R101, R102, R103, R104, R105, R106, R107, R108,
         R109, R110, R111, R112, R113, R114, R115, R116,
         R117, R118, R119, R120, R121, R123, R124, R125,
         R126, R127, R128, R129, R130, R132, R133, R134,
         R135, R136, R138, R139, R140, R141, R142, R143,
         R144, R145, R146, R147, R149, R150, R151, R152,
         R153, R154, R155, R156, R158, R159, R160, R161,
         R163, R164, R165, R166, R167, R168, R169, R171,
         R172, R173, R174, R175, R176, R178, R179, R180,
         R181, R182, R183, R184 },
ThirdbodyReactions{R1, R2, R31, R36, R40, R157 },
FalloffReactions{R12, R47, R49, R51, R53, R54, R56, R60,
         R67, R68, R70, R72, R78, R80, R90, R122,
         R131, R137, R148, R162, R170, R177 }
{
// This executable is expecting CH4_30SpMix in the input file
assert(config.Flow.mixture.type == MixtureModel_CH4_30SpMix);

// Store reference quantities
StoreReferenceQuantities(config.Flow.mixture.u.CH4_30SpMix.PRef,
                         config.Flow.mixture.u.CH4_30SpMix.TRef,
                         config.Flow.mixture.u.CH4_30SpMix.LRef,
                         config.Flow.mixture.u.CH4_30SpMix.XiRef);
};
#endif

//---------------------------------
// Cleanup
//---------------------------------

#undef iH2
#undef H2
#undef iH
#undef H
#undef iO
#undef O
#undef iO2
#undef O2
#undef iOH
#undef OH
#undef iH2O
#undef H2O
#undef iHO2
#undef HO2
#undef iH2O2
#undef H2O2
#undef iC
#undef C
#undef iCH
#undef CH
#undef iCH2
#undef CH2
#undef iCH2_S
#undef CH2_S
#undef iCH3
#undef CH3
#undef iCH4
#undef CH4
#undef iCO
#undef CO
#undef iCO2
#undef CO2
#undef iHCO
#undef HCO
#undef iCH2O
#undef CH2O
#undef iCH2OH
#undef CH2OH
#undef iCH3O
#undef CH3O
#undef iCH3OH
#undef CH3OH
#undef iC2H2
#undef C2H2
#undef iC2H3
#undef C2H3
#undef iC2H4
#undef C2H4
#undef iC2H5
#undef C2H5
#undef iC2H6
#undef C2H6
#undef iHCCO
#undef HCCO
#undef iCH2CO
#undef CH2CO
#undef iCH2CHO
#undef CH2CHO
#undef iN2
#undef N2

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
#undef R21
#undef R22
#undef R23
#undef R24
#undef R25
#undef R26
#undef R27
#undef R28
#undef R29
#undef R30
#undef R31
#undef R32
#undef R33
#undef R34
#undef R35
#undef R36
#undef R37
#undef R38
#undef R39
#undef R40
#undef R41
#undef R42
#undef R43
#undef R44
#undef R45
#undef R46
#undef R47
#undef R48
#undef R49
#undef R50
#undef R51
#undef R52
#undef R53
#undef R54
#undef R55
#undef R56
#undef R57
#undef R58
#undef R59
#undef R60
#undef R61
#undef R62
#undef R63
#undef R64
#undef R65
#undef R66
#undef R67
#undef R68
#undef R69
#undef R70
#undef R71
#undef R72
#undef R73
#undef R74
#undef R75
#undef R76
#undef R77
#undef R78
#undef R79
#undef R80
#undef R81
#undef R82
#undef R83
#undef R84
#undef R85
#undef R86
#undef R87
#undef R88
#undef R89
#undef R90
#undef R91
#undef R92
#undef R93
#undef R94
#undef R95
#undef R96
#undef R97
#undef R98
#undef R99
#undef R100
#undef R101
#undef R102
#undef R103
#undef R104
#undef R105
#undef R106
#undef R107
#undef R108
#undef R109
#undef R110
#undef R111
#undef R112
#undef R113
#undef R114
#undef R115
#undef R116
#undef R117
#undef R118
#undef R119
#undef R120
#undef R121
#undef R122
#undef R123
#undef R124
#undef R125
#undef R126
#undef R127
#undef R128
#undef R129
#undef R130
#undef R131
#undef R132
#undef R133
#undef R134
#undef R135
#undef R136
#undef R137
#undef R138
#undef R139
#undef R140
#undef R141
#undef R142
#undef R143
#undef R144
#undef R145
#undef R146
#undef R147
#undef R148
#undef R149
#undef R150
#undef R151
#undef R152
#undef R153
#undef R154
#undef R155
#undef R156
#undef R157
#undef R158
#undef R159
#undef R160
#undef R161
#undef R162
#undef R163
#undef R164
#undef R165
#undef R166
#undef R167
#undef R168
#undef R169
#undef R170
#undef R171
#undef R172
#undef R173
#undef R174
#undef R175
#undef R176
#undef R177
#undef R178
#undef R179
#undef R180
#undef R181
#undef R182
#undef R183
#undef R184

#endif

#endif // __CH4_30SPMIX_HPP__
