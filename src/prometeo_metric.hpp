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

#ifndef __PROMETEO_METRIC_HPP__
#define __PROMETEO_METRIC_HPP__

#include "legion.h"

using namespace Legion;

//-----------------------------------------------------------------------------
// LOAD PROMETEO UTILITIES AND MODULES
//-----------------------------------------------------------------------------

#include "task_helper.hpp"
#include "PointDomain_helper.hpp"
#include "prometeo_types.h"
#include "prometeo_metric.h"
#include "prometeo_metric.inl"

//-----------------------------------------------------------------------------
// UTILITY TO UNWARP COORDIATNES AROUND PERIODIC BCS
//-----------------------------------------------------------------------------

template<direction dir>
__CUDA_H__
inline double unwarpCoordinate(double x, const double w, const int off,
                               Point<3> p, const Rect<3> &bounds);
template<>
__CUDA_H__
inline double unwarpCoordinate<Xdir>(double x, const double w, const int off,
                                     Point<3> p, const Rect<3> &bounds) {
   p.x += off;
   // if we are below the lower bound start by shifting:
   // - x by -1 width
   // - the index by 1
   if (p.x < bounds.lo.x) { x -= w; p.x += 1; }
   return x + w*coord_t((p.x - bounds.lo.x)/(bounds.hi.x - bounds.lo.x + 1));
}
template<>
__CUDA_H__
inline double unwarpCoordinate<Ydir>(double y, const double w, const int off,
                                     Point<3> p, const Rect<3> &bounds) {
   p.y += off;
   // if we are below the lower bound start by shifting:
   // - y by -1 width
   // - the index by 1
   if (p.y < bounds.lo.y) { y -= w; p.y += 1; }
   return y + w*coord_t((p.y - bounds.lo.y)/(bounds.hi.y - bounds.lo.y + 1));
}
template<>
__CUDA_H__
inline double unwarpCoordinate<Zdir>(double z, const double w, const int off,
                                     Point<3> p, const Rect<3> &bounds) {
   p.z += off;
   // if we are below the lower bound start by shifting:
   // - z by -1 width
   // - the index by 1
   if (p.z < bounds.lo.z) { z -= w; p.z += 1; }
   return z + w*coord_t((p.z - bounds.lo.z)/(bounds.hi.z - bounds.lo.z + 1));
}

//-----------------------------------------------------------------------------
// TASK THAT COMPUTES THE METRIC OF THE INTERNAL POINTS OF THE GRID
//-----------------------------------------------------------------------------

class InitializeMetricTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion MetricGhost;
      LogicalRegion Fluid;
      Rect<3> Fluid_bounds;
      bBoxType bBox;
      FieldID MetricGhost_fields [FID_last - 101];
      FieldID Fluid_fields [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
public:
   // Direction dependent quantities
   template<direction dir>
   __CUDA_H__
   static inline double reconstructCoordEuler(const AccessorRO<Vec3, 3> &centerCoordinates,
                                              const Point<3> &p,
                                              const double width,
                                              const int nType,
                                              const coord_t dsize,
                                              const Rect<3> &bounds) {

      constexpr int iN = normalIndex(dir);

      // Compute stencil points
      const Point<3> pM2 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM2(nType));
      const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));
      const Point<3> pP2 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP2(nType));
      const Point<3> pP3 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP3(nType));

      return LinearReconstruct(
            unwarpCoordinate<dir>(centerCoordinates[pM2][iN], width, offM2(nType), p, bounds),
            unwarpCoordinate<dir>(centerCoordinates[pM1][iN], width, offM1(nType), p, bounds),
                                  centerCoordinates[p  ][iN],
            unwarpCoordinate<dir>(centerCoordinates[pP1][iN], width, offP1(nType), p, bounds),
            unwarpCoordinate<dir>(centerCoordinates[pP2][iN], width, offP2(nType), p, bounds),
            unwarpCoordinate<dir>(centerCoordinates[pP3][iN], width, offP3(nType), p, bounds),
            nType);
   }

   template<direction dir>
   __CUDA_H__
   static inline void ComputeDiffusionMetrics(const AccessorWO<double, 3> &m_d,
                                              const AccessorWO<double, 3> &m_s,
                                              const AccessorRO<  Vec3, 3> &centerCoordinates,
                                              const Point<3> &p,
                                              const double width,
                                              const int nType,
                                              const coord_t dsize,
                                              const Rect<3> &bounds) {

      constexpr int iN = normalIndex(dir);

      const Point<3> pM1 = warpPeriodic<dir, Minus>(bounds, p, dsize, offM1(nType));
      const Point<3> pP1 = warpPeriodic<dir, Plus >(bounds, p, dsize, offP1(nType));

      // Compute staggered metric for viscous fluxes
      m_s[p] = 1.0/(unwarpCoordinate<dir>(centerCoordinates[pP1][iN], width, 1, p, bounds) -
                                          centerCoordinates[p  ][iN]);

      // Compute collocated metric for viscous fluxes
      m_d[p] = 1.0/(getDeriv(nType,
            unwarpCoordinate<dir>(centerCoordinates[pM1][iN], width, -1, p, bounds),
                                  centerCoordinates[p  ][iN],
            unwarpCoordinate<dir>(centerCoordinates[pP1][iN], width,  1, p, bounds), 1.0));
   }
public:
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

//-----------------------------------------------------------------------------
// TASK THAT CORRECTS THE METRIC OF GHOST POINTS
//-----------------------------------------------------------------------------

template<direction dir>
class CorrectGhostMetricTask {
public:
   struct Args {
      uint64_t arg_mask[1];
      LogicalRegion Fluid;
      FieldID Fluid_fields      [FID_last - 101];
   };
public:
   static const char * const TASK_NAME;
   static const int TASK_ID;
   static const bool CPU_BASE_LEAF = true;
   static const bool GPU_BASE_LEAF = true;
   static const int MAPPER_ID = 0;
private:
   // Direction dependent quantities
   static const FieldID FID_nType;
   static const FieldID FID_m;
public:
   __CUDA_H__
   static inline void CorrectLeftStaggered(const AccessorRW<double, 3> &m,
                                           const AccessorRO<  Vec3, 3> &centerCoordinates,
                                           const Point<3> &p) {
      constexpr int iN = normalIndex(dir);
      Point<3> pp1; Point<3> pp2;
      if      (dir == Xdir) { pp1 = p + Point<3>(1, 0, 0); pp2 = p + Point<3>(2, 0, 0); }
      else if (dir == Ydir) { pp1 = p + Point<3>(0, 1, 0); pp2 = p + Point<3>(0, 2, 0); }
      else if (dir == Zdir) { pp1 = p + Point<3>(0, 0, 1); pp2 = p + Point<3>(0, 0, 2); }
      m[p] = 1.0/(- 8.0/3.0*centerCoordinates[p  ][iN]
                  + 3.0    *centerCoordinates[pp1][iN]
                  - 1.0/3.0*centerCoordinates[pp2][iN]);
      // The staggered metric should be scaled by 0.5 but we avoid this factor
      // for computational efficiency. Remeber this comment when you compute the fluxes
   };
   __CUDA_H__
   static inline void CorrectLeftCollocated(const AccessorRW<double, 3> &m,
                                            const AccessorRO<  Vec3, 3> &centerCoordinates,
                                            const Point<3> &p) {
      constexpr int iN = normalIndex(dir);
      Point<3> pp1; Point<3> pp2;
      if      (dir == Xdir) { pp1 = p + Point<3>(1, 0, 0); pp2 = p + Point<3>(2, 0, 0); }
      else if (dir == Ydir) { pp1 = p + Point<3>(0, 1, 0); pp2 = p + Point<3>(0, 2, 0); }
      else if (dir == Zdir) { pp1 = p + Point<3>(0, 0, 1); pp2 = p + Point<3>(0, 0, 2); }
      m[p] = 1.0/(- 1.5*centerCoordinates[p  ][iN]
                  + 2.0*centerCoordinates[pp1][iN]
                  - 0.5*centerCoordinates[pp2][iN]);
   };
   __CUDA_H__
   static inline void CorrectRightStaggered(const AccessorRW<double, 3> &m,
                                            const AccessorRO<  Vec3, 3> &centerCoordinates,
                                            const Point<3> &p) {
      constexpr int iN = normalIndex(dir);
      Point<3> pm1; Point<3> pm2;
      if      (dir == Xdir) { pm1 = p - Point<3>(1, 0, 0); pm2 = p - Point<3>(2, 0, 0); }
      else if (dir == Ydir) { pm1 = p - Point<3>(0, 1, 0); pm2 = p - Point<3>(0, 2, 0); }
      else if (dir == Zdir) { pm1 = p - Point<3>(0, 0, 1); pm2 = p - Point<3>(0, 0, 2); }
      m[p] = 1.0/(  8.0/3.0*centerCoordinates[p  ][iN]
                  - 3.0    *centerCoordinates[pm1][iN]
                  + 1.0/3.0*centerCoordinates[pm2][iN]);
      // The staggered metric should be scaled by 0.5 but we avoid this factor
      // for computational efficiency. Remeber this comment when you compute the fluxes
   };
   __CUDA_H__
   static inline void CorrectRightCollocated(const AccessorRW<double, 3> &m,
                                             const AccessorRO<  Vec3, 3> &centerCoordinates,
                                             const Point<3> &p) {
      constexpr int iN = normalIndex(dir);
      Point<3> pm1; Point<3> pm2;
      if      (dir == Xdir) { pm1 = p - Point<3>(1, 0, 0); pm2 = p - Point<3>(2, 0, 0); }
      else if (dir == Ydir) { pm1 = p - Point<3>(0, 1, 0); pm2 = p - Point<3>(0, 2, 0); }
      else if (dir == Zdir) { pm1 = p - Point<3>(0, 0, 1); pm2 = p - Point<3>(0, 0, 2); }
      m[p] = 1.0/(  1.5*centerCoordinates[p  ][iN]
                  - 2.0*centerCoordinates[pm1][iN]
                  + 0.5*centerCoordinates[pm2][iN]);
   };
public:
   static void cpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#ifdef LEGION_USE_CUDA
   static void gpu_base_impl(const Args &args,
                             const std::vector<PhysicalRegion> &regions,
                             const std::vector<Future>         &futures,
                             Context ctx, Runtime *runtime);
#endif
};

#endif // __PROMETEO_METRIC_HPP__
