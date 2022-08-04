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

#ifndef __POINTDOMAIN_HELPER_HPP__
#define __POINTDOMAIN_HELPER_HPP__

#include "legion.h"

#include "my_array.hpp"

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#ifndef __CUDA_H__
#ifdef __CUDACC__
#define __CUDA_H__ __device__
#else
#define __CUDA_H__
#endif
#endif

#ifndef __UNROLL__
#ifdef __CUDACC__
#define __UNROLL__ #pragma unroll
#else
#define __UNROLL__
#endif
#endif

enum direction {
   Xdir,
   Ydir,
   Zdir
};

enum side {
   Plus,
   Minus
};

//-----------------------------------------------------------------------------
// Utility that outputs the tangential directions given the normal
//-----------------------------------------------------------------------------
__CUDA_HD__
constexpr direction getT1(direction dir) {return (dir == Xdir) ? Ydir :
                                                 (dir == Ydir) ? Xdir :
                                               /*(dir == Zdir)*/ Xdir ;};

__CUDA_HD__
constexpr direction getT2(direction dir) {return (dir == Xdir) ? Zdir :
                                                 (dir == Ydir) ? Zdir :
                                               /*(dir == Zdir)*/ Ydir ;};

//-----------------------------------------------------------------------------
// Utility that outputs the index of the normal and tangetial components of
// a Vec3 given a direction
//-----------------------------------------------------------------------------
__CUDA_HD__
constexpr int normalIndex(direction dir) {
   return (dir == Xdir) ? 0 :
          (dir == Ydir) ? 1 :
        /*(dir == Zdir)*/ 2;
};

__CUDA_HD__
constexpr int tangential1Index(direction dir) {
   return (dir == Xdir) ? 1 :
          (dir == Ydir) ? 0 :
        /*(dir == Zdir)*/ 0;
};

__CUDA_HD__
constexpr int tangential2Index(direction dir) {
   return (dir == Xdir) ? 2 :
          (dir == Ydir) ? 2 :
        /*(dir == Zdir)*/ 1;
};

//-----------------------------------------------------------------------------
// Utility that computes the first internal point corresponding to a BC point
//-----------------------------------------------------------------------------
template<direction dir, side s>
__CUDA_HD__
inline Legion::Point<3> getPIntBC(const Legion::Point<3> &p);
template<>
__CUDA_HD__
inline Legion::Point<3> getPIntBC<Xdir, Minus>(const Legion::Point<3> &p) { return Legion::Point<3>(p.x+1, p.y  , p.z  ); };
template<>
__CUDA_HD__
inline Legion::Point<3> getPIntBC<Xdir, Plus >(const Legion::Point<3> &p) { return Legion::Point<3>(p.x-1, p.y  , p.z  ); };
template<>
__CUDA_HD__
inline Legion::Point<3> getPIntBC<Ydir, Minus>(const Legion::Point<3> &p) { return Legion::Point<3>(p.x  , p.y+1, p.z  ); };
template<>
__CUDA_HD__
inline Legion::Point<3> getPIntBC<Ydir, Plus >(const Legion::Point<3> &p) { return Legion::Point<3>(p.x  , p.y-1, p.z  ); };
template<>
__CUDA_HD__
inline Legion::Point<3> getPIntBC<Zdir, Minus>(const Legion::Point<3> &p) { return Legion::Point<3>(p.x  , p.y  , p.z+1); };
template<>
__CUDA_HD__
inline Legion::Point<3> getPIntBC<Zdir, Plus >(const Legion::Point<3> &p) { return Legion::Point<3>(p.x  , p.y  , p.z-1); };


//-----------------------------------------------------------------------------
// Utility that computes the size of a Rect<3>
//-----------------------------------------------------------------------------
template<direction dir>
__CUDA_HD__
inline Legion::coord_t getSize(const Legion::Rect<3> bounds);
template<>
__CUDA_HD__
inline Legion::coord_t getSize<Xdir>(const Legion::Rect<3> bounds) { return bounds.hi.x - bounds.lo.x + 1; };
template<>
__CUDA_HD__
inline Legion::coord_t getSize<Ydir>(const Legion::Rect<3> bounds) { return bounds.hi.y - bounds.lo.y + 1; };
template<>
__CUDA_HD__
inline Legion::coord_t getSize<Zdir>(const Legion::Rect<3> bounds) { return bounds.hi.z - bounds.lo.z + 1; };

//-----------------------------------------------------------------------------
// Utility that computes the stencil point warping the point around periodic boundaries
//-----------------------------------------------------------------------------
template<direction dir, side s>
__CUDA_HD__
static inline Legion::Point<3> warpPeriodic(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off);
template<>
__CUDA_HD__
inline Legion::Point<3> warpPeriodic<Xdir, Minus>(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off) {
   return Legion::Point<3>(((p.x + off - bounds.lo.x) % size + size) % size + bounds.lo.x, p.y, p.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> warpPeriodic<Xdir, Plus>(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off) {
   return Legion::Point<3>((p.x + off - bounds.lo.x) % size + bounds.lo.x, p.y, p.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> warpPeriodic<Ydir, Minus>(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off) {
   return Legion::Point<3>(p.x, ((p.y + off - bounds.lo.y) % size + size) % size + bounds.lo.y, p.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> warpPeriodic<Ydir, Plus>(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off) {
   return Legion::Point<3>(p.x, (p.y + off - bounds.lo.y) % size + bounds.lo.y, p.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> warpPeriodic<Zdir, Minus>(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off) {
   return Legion::Point<3>(p.x, p.y, ((p.z + off - bounds.lo.z) % size + size) % size + bounds.lo.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> warpPeriodic<Zdir, Plus>(const Legion::Rect<3> bounds, Legion::Point<3> p, const Legion::coord_t size, const int8_t off) {
   return Legion::Point<3>(p.x, p.y, (p.z + off - bounds.lo.z) % size + bounds.lo.z);
};

//-----------------------------------------------------------------------------
// Utility that extracts cross-plane of a 3d domain
//-----------------------------------------------------------------------------
template<direction dir>
__CUDA_HD__
static inline Legion::Rect<3> crossPlane(const Legion::Rect<3> bounds);
template<>
__CUDA_HD__
inline Legion::Rect<3> crossPlane<Xdir>(const Legion::Rect<3> bounds) {
   return Legion::Rect<3>(Legion::Point<3>(bounds.lo.x, bounds.lo.y, bounds.lo.z), Legion::Point<3>(bounds.lo.x, bounds.hi.y, bounds.hi.z));
};
template<>
__CUDA_HD__
inline Legion::Rect<3> crossPlane<Ydir>(const Legion::Rect<3> bounds) {
   return Legion::Rect<3>(Legion::Point<3>(bounds.lo.x, bounds.lo.y, bounds.lo.z), Legion::Point<3>(bounds.hi.x, bounds.lo.y, bounds.hi.z));
};
template<>
__CUDA_HD__
inline Legion::Rect<3> crossPlane<Zdir>(const Legion::Rect<3> bounds) {
   return Legion::Rect<3>(Legion::Point<3>(bounds.lo.x, bounds.lo.y, bounds.lo.z), Legion::Point<3>(bounds.hi.x, bounds.hi.y, bounds.lo.z));
};

//-----------------------------------------------------------------------------
// Utility that computes the point along the span of a 3d domain
//-----------------------------------------------------------------------------
template<direction dir>
__CUDA_HD__
static inline Legion::Point<3> GetPointInSpan(const Legion::Rect<3> b, const Legion::coord_t idx , const int x, const int y, const int z);
template<>
__CUDA_HD__
inline Legion::Point<3> GetPointInSpan<Xdir>(const Legion::Rect<3> b, const Legion::coord_t idx, const int x, const int y, const int z) {
   return Legion::Point<3>(b.lo.x + idx, y + b.lo.y, z + b.lo.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> GetPointInSpan<Ydir>(const Legion::Rect<3> b, const Legion::coord_t idx, const int x, const int y, const int z) {
   return Legion::Point<3>(x + b.lo.x, b.lo.y + idx, z + b.lo.z);
};
template<>
__CUDA_HD__
inline Legion::Point<3> GetPointInSpan<Zdir>(const Legion::Rect<3> b, const Legion::coord_t idx, const int x, const int y, const int z) {
   return Legion::Point<3>(x + b.lo.x, y + b.lo.y, b.lo.z + idx);
};

#endif // __POINTDOMAIN_HELPER_HPP__

