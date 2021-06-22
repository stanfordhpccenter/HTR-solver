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

#ifndef __POISSON_H__
#define __POISSON_H__

#ifdef __cplusplus
extern "C" {
#endif

//#include "fftw3.h"
//#ifdef USE_CUDA
//#include "cufft.h"
//#endif

//struct fftPlansType {
//   fftw_plan fftw_fwd;        // FFTW plan for direct transform
//   fftw_plan fftw_bwd;        // FFTW plan for inverse transform
//#ifdef USE_CUDA
//   cufftHandle cufft_fwd;     // cuFFT plan for direct transform
//   cufftHandle cufft_bwd;     // cuFFT plan for inverse transform
//#endif
//   legion_address_space_t id; // processor index
//};

enum FieldIDs_FFTplans {
   FID_fftw_fwd = 101,
   FID_fftw_bwd,
   FID_cufft,
   FID_id,
   FID_FFTplans_last
};

enum FieldIDs_CoeffType {
   FID_a = 101,
   FID_b,
   FID_c,
   FID_CoeffType_last
};

enum FieldIDs_k2Type {
   FID_k2 = 101
};

enum FieldIDs_fftType {
   FID_fft = 101
};

void register_poisson_tasks();

#ifdef __cplusplus
}
#endif

#endif // __POISSON_H__

