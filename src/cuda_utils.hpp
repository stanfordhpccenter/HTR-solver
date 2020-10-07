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

#ifndef __CUDA_UTILS_HPP__
#define __CUDA_UTILS_HPP__

//// Utility that unwarps the tid to the point of the FluxGhosts region
//template<direction dir>
//__device__
//inline Point<3> unwarpTidFluxGhosts(coord_t tid, const coord_t size, const Rect<3> bounds, const Rect<3> Fluid_bounds);
//template<>
//__device__
//inline Point<3> unwarpTidFluxGhosts<Xdir>(coord_t tid, const coord_t size_x, const Rect<3> bounds, const Rect<3> Fluid_bounds) {
//   const coord_t size_y = getSize<Ydir>(bounds);
//   const coord_t size_z = getSize<Zdir>(bounds);
//   const coord_t i = tid/(size_y * size_z) + bounds.lo.x;
//   tid -= i * size_y * size_z;
//   const coord_t k = tid/size_y + bounds.lo.z;
//   const coord_t j = tid%size_y + bounds.lo.y;
//   return warpPeriodic<Xdir, Minus>(Fluid_bounds, Point<3>(i, j ,k), size_x, 0);
//};
//template<>
//__device__
//inline Point<3> unwarpTidFluxGhosts<Ydir>(coord_t tid, const coord_t size_y, const Rect<3> bounds, const Rect<3> Fluid_bounds) {
//   const coord_t size_x = getSize<Xdir>(bounds);
//   const coord_t size_z = getSize<Zdir>(bounds);
//   const coord_t j = tid/(size_x * size_z) + bounds.lo.y;
//   tid -= j * size_x * size_z;
//   const coord_t k = tid/size_x + bounds.lo.z;
//   const coord_t i = tid%size_x+ bounds.lo.x;
//   return warpPeriodic<Ydir, Minus>(Fluid_bounds, Point<3>(i, j ,k), size_y, 0);
//};
//template<>
//__device__
//inline Point<3> unwarpTidFluxGhosts<Zdir>(coord_t tid, const coord_t size_z, const Rect<3> bounds, const Rect<3> Fluid_bounds) {
//   const coord_t size_x = getSize<Xdir>(bounds);
//   const coord_t size_y = getSize<Ydir>(bounds);
//   const coord_t k = tid/(size_x * size_y) + bounds.lo.z;
//   tid -= k * size_x * size_y;
//   const coord_t j = tid/size_x + bounds.lo.y;
//   const coord_t i = tid%size_x + bounds.lo.x;
//   return warpPeriodic<Zdir, Minus>(Fluid_bounds, Point<3>(i, j ,k), size_z, 0);
//};

// Utility that splits the numeber of threads per block along each direction for the 3d kernel
__host__
inline int findHighestPower2(const int num, const int m = 8) {
   assert(num > 0);
   // start from 2^0
   int j = 0;
   for (int i= 0; i<=m; i++) { if (1<<i > num) break; j = i;}
   return j;
}
template<direction dir>
__host__
inline dim3 splitThreadsPerBlock(const int TPB, const Rect<3> bounds) {
   assert(TPB%2 == 0);
   const coord_t size_x = getSize<Xdir>(bounds);
   const coord_t size_y = getSize<Ydir>(bounds);
   const coord_t size_z = getSize<Zdir>(bounds);
   dim3 r;
   r.x = 1 << findHighestPower2(size_x, findHighestPower2(TPB));
   r.y = 1 << findHighestPower2(size_y, findHighestPower2(TPB/r.x));
   r.z = min(TPB/r.x/r.y, 1 << findHighestPower2(size_x*size_y*size_z));
   return r;
};

#endif // __CUDA_UTILS_HPP__

