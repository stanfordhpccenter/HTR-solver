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

//  omega_mu() returns the collision integral for mu given dimensionless temperature t/(eps/k).
//  TODO: These come from FlameMaster.
//        At a certain point, verify these implementations.
__CUDA_HD__
inline double Spec::omega_mu(const double T) const {
   const double m1 = 3.3530622607;
   const double m2 = 2.53272006;
   const double m3 = 2.9024238575;
   const double m4 = 0.11186138893;
   const double m5 = 0.8662326188;       // = -0.1337673812 + 1.0
   const double m6 = 1.3913958626;
   const double m7 = 3.158490576;
   const double m8 = 0.18973411754;
   const double m9 = 0.00018682962894;

   const double num = m1 + T*(m2 + T*(m3 + T*m4));
   const double den = m5 + T*(m6 + T*(m7 + T*(m8 + T*m9)));
   return num / den;
};

// omega_D() returns the Stossintegral for a given dimensionless temperature t/(eps/k)
__CUDA_HD__
inline double Spec::omega_D(const double T) const {
   const double m1 = 6.8728271691;
   const double m2 = 9.4122316321;
   const double m3 = 7.7442359037;
   const double m4 = 0.23424661229;
   const double m5 = 1.45337701568;         // = 1.0 + 0.45337701568
   const double m6 = 5.2269794238;
   const double m7 = 9.7108519575;
   const double m8 = 0.46539437353;
   const double m9 = 0.00041908394781;

   const double num = m1 + T * (m2 + T * (m3 + T * m4));
   const double den = m5 + T * (m6 + T * (m7 + T * (m8 + T * m9)));
   return num / den;
};

// Returns the (n,6,4) integral
__CUDA_HD__
inline double Spec::omega_D_N64(const double T, const double gamma) const {
   const double logtstar = log(T);
   if (T <= 0.04)
       // for interval 0.01 to 0.04, SSE = 0.006; R^2 = 1; RMSE = 0.020
      return 2.97 - 12.0 * gamma
             - 0.887 * logtstar
             + 3.86 * gamma * gamma
             - 6.45 * gamma * logtstar
             - 0.275 * logtstar * logtstar
             + 1.20 * gamma * gamma * logtstar
             - 1.24 * gamma * logtstar * logtstar
             - 0.164 * logtstar*logtstar*logtstar;
   else //if (T <= 1000)
      // for interval 0.04 to 1000, SSE = 0.282; R^2 = 1; RMSE = 0.033
      return 1.22 - 0.0343 * gamma
             + (-0.769 + 0.232 * gamma) * logtstar
             + (0.306 - 0.165 * gamma) * logtstar * logtstar
             + (-0.0465 + 0.0388 * gamma) * pow(logtstar, 3)
             + (0.000614 - 0.00285 * gamma) * pow(logtstar, 4)
             + 0.000238 * pow(logtstar, 5);
};

__CUDA_HD__
inline double Spec::GetCp(const double T) const {
   //assert(T < cpCoeff.TMax, "Exceeded maximum temeperature")
   //assert(T > cpCoeff.TMin, "Exceeded minimum temeperature")

   const double rOvW = RGAS/W;
   const double Tinv = 1.0/T;
#if (N_NASA_POLY == 3)
   const double * cpC = ( T > cpCoeff.TSwitch2 ) ? cpCoeff.cpH :
#else
   const double * cpC =
#endif
                        ( T > cpCoeff.TSwitch1 ) ? cpCoeff.cpM :
                                                   cpCoeff.cpL;
   return rOvW*( cpC[0]*Tinv*Tinv + cpC[1]*Tinv +  cpC[2] + T*
                                                 ( cpC[3] + T*
                                                 ( cpC[4] + T*
                                                 ( cpC[5] + T*cpC[6]))));
};

__CUDA_HD__
inline double Spec::GetFreeEnthalpy(const double T) const {
   // This is (H/(RT) - S/R)
   //assert(T < cpCoeff.TMax, "Exceeded maximum temeperature")
   //assert(T > cpCoeff.TMin, "Exceeded minimum temeperature")

   const double Tinv = 1.0/T;
   const double logT = log(T);
#if (N_NASA_POLY == 3)
   const double * cpC = ( T > cpCoeff.TSwitch2 ) ? cpCoeff.cpH :
#else
   const double * cpC =
#endif
                        ( T > cpCoeff.TSwitch1 ) ? cpCoeff.cpM :
                                                   cpCoeff.cpL;
   double G = -0.5*cpC[0]*Tinv*Tinv + cpC[1]*Tinv*(1.0 + logT) + cpC[2]*(1.0 - logT) + cpC[7]*Tinv - cpC[8];
   return G - 0.5*T*( cpC[3]   + T*
                    ( cpC[4]/3 + T*
                    ( cpC[5]/6 + 0.1*T*cpC[6] )));
};

__CUDA_HD__
inline double Spec::GetEnthalpy(const double T) const {
   //assert(T < cpCoeff.TMax, "Exceeded maximum temeperature")
   //assert(T > cpCoeff.TMin, "Exceeded minimum temeperature")

   const double rOvW = RGAS/W;
   const double Tinv = 1.0/T;
#if (N_NASA_POLY == 3)
   const double * cpC = ( T > cpCoeff.TSwitch2 ) ? cpCoeff.cpH :
#else
   const double * cpC =
#endif
                        ( T > cpCoeff.TSwitch1 ) ? cpCoeff.cpM :
                                                   cpCoeff.cpL;
   const double E = -cpC[0]*Tinv + cpC[1]*log(T) + cpC[7]      + T*
                                                 ( cpC[2]      + T*
                                                 ( cpC[3]*0.50 + T*
                                                 ( cpC[4]/3    + T*
                                                 ( cpC[5]*0.25 + cpC[6]/5*T))));
   return E*rOvW;
};

__CUDA_HD__
inline double Spec::GetMu(const double T) const {
   const double num = 5 * sqrt(PI * W/Na * kb * T);
   const double den = 16 * PI * pow(DiffCoeff.sigma,2) * omega_mu( T * DiffCoeff.kbOveps );
   return num/den;
};

// Return kinetic thory parametes for Stockmayer thory
__CUDA_HD__
inline void Spec::GetDifCollParam_Stock(double &sigmaij, double &omega11,
                                        const Spec & s1, const Spec & s2, const double T) const {
   double xi = 1.0;
   if ((s1.DiffCoeff.mu*s2.DiffCoeff.mu == 0.0) and
       (s1.DiffCoeff.mu+s2.DiffCoeff.mu != 0.0)) {
      // If I have a polar to non-polar molecule interaction
      double mup;
      double alp;
      double epr;
      if (s1.DiffCoeff.mu != 0.0) {
         // s1 is the polar molecule and s2 is non-polar
         mup = s1.DiffCoeff.mu/sqrt(4*PI*eps_0*kb*pow(s1.DiffCoeff.sigma, 3)/s1.DiffCoeff.kbOveps);
         alp = s2.DiffCoeff.alpha/pow(s2.DiffCoeff.sigma,3);
         epr = sqrt(s2.DiffCoeff.kbOveps/s1.DiffCoeff.kbOveps);
      } else {
         // s1 is the non-polar molecule and s2 is polar
         mup = s2.DiffCoeff.mu/sqrt(4*PI*eps_0*kb*pow(s2.DiffCoeff.sigma, 3)/s2.DiffCoeff.kbOveps);
         alp = s1.DiffCoeff.alpha/pow(s1.DiffCoeff.sigma,3);
         epr = sqrt(s1.DiffCoeff.kbOveps/s2.DiffCoeff.kbOveps);
      }
      xi = 1 + 0.25*mup*alp*epr;
   }
   // Binary cross-section
   sigmaij = 0.5*(DiffCoeff.sigma + s2.DiffCoeff.sigma)*pow(xi, -1./6);
   // Collision integral
   const double kboEpsij = sqrt(DiffCoeff.kbOveps * s2.DiffCoeff.kbOveps)/(xi*xi);
   omega11 = omega_D(T * kboEpsij);
};

#if (nIons > 0)
// Return kinetic thory parametes for (n,6,4) thory
__CUDA_HD__
inline void Spec::GetDifCollParam_N64(double &sigmaij, double &omega11,
                                      const Spec & i, const Spec & n, const double T) const {
   // Coefficients
   constexpr double K1 = 1.767;
   constexpr double K2 = 1.44;
   constexpr double kappa = 0.095;
   // Ratio or polarizzabilities
   const double r_alpha = i.DiffCoeff.alpha/n.DiffCoeff.alpha;
   // Ratio of dispersion to induction forces
   const double xi = 1e15*i.DiffCoeff.alpha / (i.nCrg*i.nCrg * (1 + pow(2*r_alpha, 2.0/3)) * sqrt(n.DiffCoeff.alpha));
   // Binary cross-section
   sigmaij = K1 * (pow(i.DiffCoeff.alpha, 1.0/3) + pow(n.DiffCoeff.alpha, 1./3)) /
             pow(1e60*i.DiffCoeff.alpha*n.DiffCoeff.alpha*(1 + 1.0/xi), kappa);
   // Well-depth
   const double epsilon = K2 * eCrg*eCrg * i.nCrg*i.nCrg * n.DiffCoeff.alpha * (1 + xi) / (8 * PI * eps_0 * pow(sigmaij, 4));
   // Dispersion coefficients
   const double C6n = exp(1.8846*log(n.DiffCoeff.alpha*1e30)-0.4737)*1e-50;
   const double C6i = (i.nCrg > 0) ? exp(1.8853*log(i.DiffCoeff.alpha*1e30)+0.2682)*1e-50:
                                     exp(3.2246*log(i.DiffCoeff.alpha*1e30)-3.2397)*1e-50;
   // Binary dispersion coefficient
   const double C6 = 2*C6i*C6n/(1.0/r_alpha * C6i + r_alpha * C6n);
   const double gamma = (2*C6/(i.nCrg*i.nCrg) + 2*C6) / (n.DiffCoeff.alpha * sigmaij*sigmaij);//Dimensionless
   // Collision integral
   omega11 = omega_D_N64(T * kb / epsilon, gamma);
};
#endif

__CUDA_HD__
inline double Spec::GetDif(const Spec & s2, const double P, const double T) const{
   double sigmaij;
   double omega11;
#if (defined(ELECTRIC_FIELD) && (nIons > 0))
   // Electrons get the diffusivity based on the Einstein relations
   if (isElectron or s2.isElectron) return eMob*kb*T/eCrg;

   const bool ion1 = (   nCrg != 0);
   const bool ion2 = (s2.nCrg != 0);
   if ((ion1 != ion2) &&
       (DiffCoeff.alpha != 0.0 && s2.DiffCoeff.alpha != 0.0))
   // Apply N-6,4 for charged-neutral interactions
      if (ion1)
         // this is the ion and s2 is neutral
         GetDifCollParam_N64(sigmaij, omega11, *this, s2, T);
      else
         // this is neutral and and s2 is the ion
         GetDifCollParam_N64(sigmaij, omega11, s2, *this, T);
   else
   // Apply Stockmayer for charged-neutral interactions
#endif
      GetDifCollParam_Stock(sigmaij, omega11, *this, s2, T);
   const double invWij = (W + s2.W)/(W*s2.W);
   const double num = 3*sqrt(2*PI*pow(kb,3)*pow(T,3)*Na*invWij);
   const double den = 16*PI*P*sigmaij*sigmaij*omega11;
   return num/den;
};

__CUDA_HD__
inline double Spec::GetMob(const Spec & s2, const double P, const double T) const{
#if (nIons > 0)
   // Electrons have a constant mobility for now
   if (isElectron or s2.isElectron) return eMob;

   // If we call this function on a species that is neutral the result is trivial
   if (nCrg == 0) return 0;

   double sigmaij;
   double omega11;
   if ((s2.nCrg == 0) &&
       (DiffCoeff.alpha != 0.0 && s2.DiffCoeff.alpha != 0.0))
   // Apply N-6,4 for charged-neutral interactions
      GetDifCollParam_N64(sigmaij, omega11, *this, s2, T);
   else
   // Apply Stockmayer for charged-neutral interactions
      GetDifCollParam_Stock(sigmaij, omega11, *this, s2, T);
   const double invWij = (W + s2.W)/(W*s2.W);
   const double num = 3*sqrt(2*PI*kb*T*Na*invWij)*eCrg*abs(int(nCrg));
   const double den = 16*PI*P*sigmaij*sigmaij*omega11;
   return num/den;
#else
   return 0;
#endif
};

__CUDA_HD__
inline double Spec::GetSelfDiffusion(const double T) const {
   // Already multiplied by partial density
   const double num = 3*sqrt( PI*kb*T*W/Na );
   const double den = 8*PI*pow(DiffCoeff.sigma,2)*omega_D(T * DiffCoeff.kbOveps);
   return num/den;
};

__CUDA_HD__
inline double Spec::GetFZrot(const double T) const {
   const double tmp = 1.0/(DiffCoeff.kbOveps*T);
   return 1 + 0.5*pow(PI,1.5)*sqrt(tmp)
            + (2 + 0.25*PI*PI)*tmp
            + pow(PI,1.5)*pow(tmp,1.5);
};

__CUDA_HD__
inline double Spec::GetLamAtom(const double T) const {
   const double mu = GetMu(T);
   return 15.0/4*mu*RGAS/W;
};

__CUDA_HD__
inline double Spec::GetLamLinear(const double T) const {
   const double CvTOvR = 1.5;
   const double CvROvR = 1.0;

   const double CvT = CvTOvR*RGAS;
   const double CvR = CvROvR*RGAS;
   const double CvV = GetCp(T)*W - 3.5*RGAS;

   const double Dkk = GetSelfDiffusion(T);
   const double mu = GetMu(T);

   const double fV = Dkk/mu;

   const double Zrot = DiffCoeff.Z298*GetFZrot(298)/GetFZrot(T);

   const double A = 2.5 - fV;
   const double B = Zrot + 2/PI*(5./3*CvROvR+fV);

   const double fT = 2.5 * (1. - 2*CvR*A/(PI*CvT*B));
   const double fR = fV*(1. + 2*A/(PI*B));

   return mu/W*(fT*CvT + fR*CvR + fV*CvV);
};

__CUDA_HD__
inline double Spec::GetLamNonLinear(const double T) const {
   const double CvTOvR = 1.5;
   const double CvROvR = 1.5;

   const double CvT = CvTOvR*RGAS;
   const double CvR = CvROvR*RGAS;
   const double CvV = GetCp(T)*W - 4.0*RGAS;

   const double Dkk = GetSelfDiffusion(T);
   const double mu  = GetMu(T);

   const double fV = Dkk/mu;

   const double Zrot = DiffCoeff.Z298*GetFZrot(298.0)/GetFZrot(T);

   const double A = 2.5 - fV;
   const double B = Zrot + 2/PI*(5./3*CvROvR+fV);

   const double fT = 2.5 * (1. - 2*CvR*A/(PI*CvT*B));
   const double fR = fV*(1. + 2*A/(PI*B));

   return mu/W*(fT*CvT + fR*CvR + fV*CvV);
};

__CUDA_HD__
inline double Spec::GetLam(const double T) const {
   return ((DiffCoeff.Geom == SpeciesGeom_Atom)   ?      GetLamAtom(T)   :
          ((DiffCoeff.Geom == SpeciesGeom_Linear) ?      GetLamLinear(T) :
         /*(DiffCoeff.Geom == SpeciesGeom_NonLinear) ?*/ GetLamNonLinear(T)));
};

