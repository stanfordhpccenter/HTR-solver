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

#ifndef __PROMETEO_BC_TYPES_H__
#define __PROMETEO_BC_TYPES_H__

#include "legion.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "prometeo_const.h"

#ifndef nSpec
   #error "nSpec is undefined"
#endif

#ifndef nEq
   #error "nEq is undefined"
#endif

struct Averages_columns {
   double weight;
   // Grid point
   double centerCoordinates[3];
   // Primitive variables
   double pressure_avg;
   double pressure_rms;
   double temperature_avg;
   double temperature_rms;
   double MolarFracs_avg[nSpec];
   double MolarFracs_rms[nSpec];
   double MassFracs_avg[nSpec];
   double MassFracs_rms[nSpec];
   double velocity_avg[3];
   double velocity_rms[3];
   double velocity_rey[3];
   // Properties
   double rho_avg;
   double rho_rms;
   double mu_avg;
   double lam_avg;
   double Di_avg[nSpec];
   double SoS_avg;
   double cp_avg;
   double Ent_avg;
   // Electric variables
#ifdef ELECTRIC_FIELD
   double electricPotential_avg;
   double chargeDensity_avg;
#endif
   // Chemical production rates
   double ProductionRates_avg[nSpec];
   double ProductionRates_rms[nSpec];
   double HeatReleaseRate_avg;
   double HeatReleaseRate_rms;
   // Favre averaged primitives
   double pressure_favg;
   double pressure_frms;
   double temperature_favg;
   double temperature_frms;
   double MolarFracs_favg[nSpec];
   double MolarFracs_frms[nSpec];
   double MassFracs_favg[nSpec];
   double MassFracs_frms[nSpec];
   double velocity_favg[3];
   double velocity_frms[3];
   double velocity_frey[3];
   // Favre averaged properties
   double mu_favg;
   double lam_favg;
   double Di_favg[nSpec];
   double SoS_favg;
   double cp_favg;
   double Ent_favg;
   // Kinetic energy budgets (y is the inhomogeneous direction)
   double rhoUUv[3];
   double Up[3];
   double tau[6];
   double utau_y[3];
   double tauGradU[3];
   double pGradU[3];
   // Fluxes
   double q[3];
   // Dimensionless numbers
   double Pr;
   double Pr_rms;
   double Ec;
   double Ec_rms;
   double Ma;
   double Sc[nSpec];
   // Correlations
   double uT_avg[3];
   double uT_favg[3];
   double uYi_avg[nSpec];
   double vYi_avg[nSpec];
   double wYi_avg[nSpec];
   double uYi_favg[nSpec];
   double vYi_favg[nSpec];
   double wYi_favg[nSpec];
};

// The order of this enum must match the declaration of Fluid_columns
enum Averages_FieldIDs {
   // Average weight
   AVE_FID_weight = 101,
   // Grid point
   AVE_FID_centerCoordinates,
   // Primitive variables
   AVE_FID_pressure_avg,
   AVE_FID_pressure_rms,
   AVE_FID_temperature_avg,
   AVE_FID_temperature_rms,
   AVE_FID_MolarFracs_avg,
   AVE_FID_MolarFracs_rms,
   AVE_FID_MassFracs_avg,
   AVE_FID_MassFracs_rms,
   AVE_FID_velocity_avg,
   AVE_FID_velocity_rms,
   AVE_FID_velocity_rey,
   // Properties
   AVE_FID_rho_avg,
   AVE_FID_rho_rms,
   AVE_FID_mu_avg,
   AVE_FID_lam_avg,
   AVE_FID_Di_avg,
   AVE_FID_SoS_avg,
   AVE_FID_cp_avg,
   AVE_FID_Ent_avg,
   // Electric variables
#ifdef ELECTRIC_FIELD
   AVE_FID_electricPotential_avg,
   AVE_FID_chargeDensity_avg,
#endif
   // Chemical production rates
   AVE_FID_ProductionRates_avg,
   AVE_FID_ProductionRates_rms,
   AVE_FID_HeatReleaseRate_avg,
   AVE_FID_HeatReleaseRate_rms,
   // Favre averaged primitives
   AVE_FID_pressure_favg,
   AVE_FID_pressure_frms,
   AVE_FID_temperature_favg,
   AVE_FID_temperature_frms,
   AVE_FID_MolarFracs_favg,
   AVE_FID_MolarFracs_frms,
   AVE_FID_MassFracs_favg,
   AVE_FID_MassFracs_frms,
   AVE_FID_velocity_favg,
   AVE_FID_velocity_frms,
   AVE_FID_velocity_frey,
   // Favre averaged properties
   AVE_FID_mu_favg,
   AVE_FID_lam_favg,
   AVE_FID_Di_favg,
   AVE_FID_SoS_favg,
   AVE_FID_cp_favg,
   AVE_FID_Ent_favg,
   // Kinetic energy budgets (y is the inhomogeneous direction)
   AVE_FID_rhoUUv,
   AVE_FID_Up,
   AVE_FID_tau,
   AVE_FID_utau_y,
   AVE_FID_tauGradU,
   AVE_FID_pGradU,
   // Fluxes
   AVE_FID_q,
   // Dimensionless numbers
   AVE_FID_Pr,
   AVE_FID_Pr_rms,
   AVE_FID_Ec,
   AVE_FID_Ec_rms,
   AVE_FID_Ma,
   AVE_FID_Sc,
   // Correlations
   AVE_FID_uT_avg,
   AVE_FID_uT_favg,
   AVE_FID_uYi_avg,
   AVE_FID_vYi_avg,
   AVE_FID_wYi_avg,
   AVE_FID_uYi_favg,
   AVE_FID_vYi_favg,
   AVE_FID_wYi_favg,
   // keep last for counting
   AVE_FID_last
};

#ifdef __cplusplus
}
#endif

#endif // __PROMETEO_BC_TYPES_H__
