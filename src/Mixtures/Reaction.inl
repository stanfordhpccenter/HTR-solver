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

//---------------------------------
// Arrhenius coefficients
//---------------------------------
__CUDA_HD__
inline double ArrheniusCoeff::CompRateCoeff(const double T) const {
   double Kf = A;
   if ( n != 0.0 )
      Kf *= pow(T, n);
   if ( fabs(EovR) > 1e-5 )
      Kf *= exp(-EovR/T);
   return Kf;
};

//---------------------------------
// Basic utilities
//---------------------------------
//-- Symbols that are going to be exported
//function Exports.Common(Reaction) local Exports = {}
//   __demand(__inline)
//   task Exports.AddEduct(r : Reaction, index : int, nu : double, ord : double)
//      regentlib.assert(r.Neducts < MAX_NUM_REACTANTS, "Increase MAX_NUM_REACTANTS")
//      r.educts[r.Neducts].ind = index
//      r.educts[r.Neducts].nu  = nu
//      r.educts[r.Neducts].ord = ord
//      r.Neducts += 1
//      return r
//   end
//
//   __demand(__inline)
//   task Exports.AddPduct(r : Reaction, index : int, nu : double, ord : double)
//      regentlib.assert(r.Npducts < MAX_NUM_REACTANTS, "Increase MAX_NUM_REACTANTS")
//      r.pducts[r.Npducts].ind = index
//      r.pducts[r.Npducts].nu  = nu
//      r.pducts[r.Npducts].ord = ord
//      r.Npducts += 1
//      return r
//   end
//
//   __demand(__inline)
//   task Exports.AddThirdb(r : Reaction, index : int, eff : double)
//      regentlib.assert(r.Nthirdb < MAX_NUM_TB, "Increase MAX_NUM_TB")
//      r.thirdb[r.Nthirdb].ind = index
//      r.thirdb[r.Nthirdb].eff = eff
//      r.Nthirdb += 1
//      return r
//   end
//   return Exports
//end

// TODO: In order to avoid duplication of sources we should have a
//       base reaction struct and derive each type of reaction from it
//       Unfortunately we cannot do this kind of C++ stuff until we are based on Regent
template<typename Reac>
__CUDA_HD__
inline double CompBackwardRateCoeff(const Reac &r, const double Kf, const double P, const double T, const VecNSp &G) {
   double sumNu = 0.0;
   double sumNuG = 0.0;
   __UNROLL__
   for (int i = 0; i < r.Neducts; i++) {
      sumNu  -= r.educts[i].nu;
      sumNuG -= r.educts[i].nu*G[r.educts[i].ind];
   }
   __UNROLL__
   for (int i = 0; i < r.Npducts; i++) {
      sumNu  += r.pducts[i].nu;
      sumNuG += r.pducts[i].nu*G[r.pducts[i].ind];
   }
   const double lnKc = - sumNuG - sumNu * ( log(T) + log(RGAS/P) );
   return Kf * exp(-lnKc);
}

//---------------------------------
// Standard reactions
//---------------------------------
__CUDA_HD__
inline double Reaction::GetReactionRate(const double P, const double T, const VecNSp &C, const VecNSp &G) const {
   // Forward reaction rate
   const double Kf = ArrCoeff.CompRateCoeff(T);
   double a = 1.0;
   __UNROLL__
   for (int i = 0; i < Neducts; i++) {
      const int ind = educts[i].ind;
#ifdef FWD_ORDERS
      if (educts[i].ord == 1)
#else
      if (educts[i].nu == 1)
#endif
         a *= C[ind];
      else
#ifdef FWD_ORDERS
         a *= pow(C[ind], educts[i].ord);
#else
         a *= pow(C[ind], educts[i].nu);
#endif
   }
   // Backward reaction rate
   double Kb = 0.0;
   double b = 1.0;
   if (has_backward) {
      Kb = CompBackwardRateCoeff<Reaction>(*this, Kf, P, T, G);
      __UNROLL__
      for (int i = 0; i < Npducts; i++) {
         const int ind = pducts[i].ind;
         if (pducts[i].nu == 1)
            b *= C[ind];
         else
            b *= pow(C[ind], pducts[i].nu);
      }
   }
   // Compute reaction rate
   return (Kf*a - Kb*b);
}

__CUDA_HD__
inline void Reaction::AddProductionRates(VecNSp &w, const double P, const double T, const VecNSp &C, const VecNSp &G) const {
   const double R = GetReactionRate(P, T, C, G);
   __UNROLL__
   for (int i = 0; i < Neducts; i++) {
      const int ind = educts[i].ind;
      w[ind] -= educts[i].nu*R;
   }
   __UNROLL__
   for (int i = 0; i < Npducts; i++) {
      const int ind = pducts[i].ind;
      w[ind] += pducts[i].nu*R;
   }
}

//---------------------------------
// Thirdbody reactions
//---------------------------------
__CUDA_HD__
inline double ThirdbodyReaction::GetReactionRate(const double P, const double T, const VecNSp &C, const VecNSp &G) const {
   // Forward reaction rate
   const double Kf = ArrCoeff.CompRateCoeff(T);
   double a = 1.0;
   __UNROLL__
   for (int i = 0; i < Neducts; i++) {
      const int ind = educts[i].ind;
#ifdef FWD_ORDERS
      if (educts[i].ord == 1)
#else
      if (educts[i].nu == 1)
#endif
         a *= C[ind];
      else
#ifdef FWD_ORDERS
         a *= pow(C[ind], educts[i].ord);
#else
         a *= pow(C[ind], educts[i].nu);
#endif
   }
   // Backward reaction rate
   double Kb = 0.0;
   double b = 1.0;
   if (has_backward) {
      Kb = CompBackwardRateCoeff<ThirdbodyReaction>(*this, Kf, P, T, G);
      __UNROLL__
      for (int i = 0; i < Npducts; i++) {
         const int ind = pducts[i].ind;
         if (pducts[i].nu == 1)
            b *= C[ind];
         else
            b *= pow(C[ind], pducts[i].nu);
      }
   }
   // Third body efficiency
   double c = 1.0;
   if (Nthirdb > 0) {
      c = P/(RGAS*T);
      __UNROLL__
      for (int i = 0; i < Nthirdb; i++) {
         const int ind = thirdb[i].ind;
         c += C[ind]*(thirdb[i].eff - 1.0);
      }
   }
   // Compute reaction rate
   return c*(Kf*a - Kb*b);
}

__CUDA_HD__
inline void ThirdbodyReaction::AddProductionRates(VecNSp &w, const double P, const double T, const VecNSp &C, const VecNSp &G) const {
   const double R = GetReactionRate(P, T, C, G);
   __UNROLL__
   for (int i = 0; i < Neducts; i++) {
      const int ind = educts[i].ind;
      w[ind] -= educts[i].nu*R;
   }
   __UNROLL__
   for (int i = 0; i < Npducts; i++) {
      const int ind = pducts[i].ind;
      w[ind] += pducts[i].nu*R;
   }
}

//---------------------------------
// Fall-off reactions
//---------------------------------

__CUDA_HD__
inline double FalloffReaction::computeF(const double Pr, const double T) const {
   if (Ftype == F_Lindemann)
      return 1.0;

   else if (Ftype == F_Troe2) {
      const double Fc = (1 - FOdata.Troe2.alpha)*exp(-T/FOdata.Troe2.T3)
                           + FOdata.Troe2.alpha *exp(-T/FOdata.Troe2.T1);
      const double d = 0.14;
      const double n = 0.75 - 1.27*log10(Fc);
      const double c = -0.4 - 0.67*log10(Fc);
      const double a = log10(Pr) + c;
      const double f = a/(n - d*a);
      return pow(Fc, 1.0/(1 + f*f));

   }
   else if (Ftype == F_Troe3) {
      const double Fc = (1 - FOdata.Troe3.alpha)*exp(-T/FOdata.Troe3.T3)
                           + FOdata.Troe3.alpha* exp(-T/FOdata.Troe3.T1)
                                               + exp(-  FOdata.Troe3.T2/T);
      const double d = 0.14;
      const double n = 0.75 - 1.27*log10(Fc);
      const double c = -0.4 - 0.67*log10(Fc);
      const double a = log10(Pr) + c;
      const double f = a/(n - d*a);
      return pow(Fc, 1.0/(1 + f*f));

   }
   else if (Ftype == F_SRI) {
      const double logPr = log10(Pr);
      const double X = 1.0/(1 + logPr*logPr);
      const double w = FOdata.SRI.A*exp(-FOdata.SRI.B/T)
                                  + exp(-T/FOdata.SRI.C);
      return FOdata.SRI.D*pow(w, X)*pow(T, FOdata.SRI.E);

   }
   else
      assert(false);
      return 0.0;
}

__CUDA_HD__
inline double FalloffReaction::GetReactionRate(const double P, const double T, const VecNSp &C, const VecNSp &G) const {
   // Forward rate coefficient
   const double KfH = ArrCoeffH.CompRateCoeff(T);
   const double KfL = ArrCoeffL.CompRateCoeff(T);
   // Reduced pressure
   double Pr = P/(RGAS * T);
   if (Nthirdb > 0) {
      __UNROLL__
      for (int i = 0; i < Nthirdb; i++) {
         const int ind = thirdb[i].ind;
         Pr += C[ind]*(thirdb[i].eff - 1.0);
      }
   }
   Pr *= (KfL/max(KfH, 1e-60));
   // Use Lindemann formula
   double F = computeF(Pr, T);
   const double Kf = KfH*(Pr/(1 + Pr))*F;
   // Forward reaction rate
   double a = 1.0;
   __UNROLL__
   for (int i = 0; i < Neducts; i++) {
      const int ind = educts[i].ind;
#ifdef FWD_ORDERS
      if (educts[i].ord == 1)
#else
      if (educts[i].nu == 1)
#endif
         a *= C[ind];
      else
#ifdef FWD_ORDERS
         a *= pow(C[ind], educts[i].ord);
#else
         a *= pow(C[ind], educts[i].nu);
#endif
   }
   // Backward reaction rate
   double Kb = 0.0;
   double b = 1.0;
   if (has_backward) {
      Kb = CompBackwardRateCoeff<FalloffReaction>(*this, Kf, P, T, G);
      __UNROLL__
      for (int i = 0; i < Npducts; i++) {
         const int ind = pducts[i].ind;
         if (pducts[i].nu == 1)
            b *= C[ind];
         else
            b *= pow(C[ind], pducts[i].nu);
      }
   }
   // Compute reaction rate
   return Kf*a - Kb*b;
}

__CUDA_HD__
inline void FalloffReaction::AddProductionRates(VecNSp &w, const double P, const double T, const VecNSp &C, const VecNSp &G) const {
   const double R = GetReactionRate(P, T, C, G);
   __UNROLL__
   for (int i = 0; i < Neducts; i++) {
      const int ind = educts[i].ind;
      w[ind] -= educts[i].nu*R;
   }
   __UNROLL__
   for (int i = 0; i < Npducts; i++) {
      const int ind = pducts[i].ind;
      w[ind] += pducts[i].nu*R;
   }
}

