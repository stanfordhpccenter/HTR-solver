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

#include "prometeo_metric.hpp"

// InitializeMetricTask
/*static*/ const char * const    InitializeMetricTask::TASK_NAME = "InitializeMetric";
/*static*/ const int             InitializeMetricTask::TASK_ID = TID_InitializeMetric;

void InitializeMetricTask::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<  Vec3, 3> acc_centerCoordinates(regions[0], FID_centerCoordinates);
   const AccessorRO<   int, 3> acc_nType_x          (regions[0], FID_nType_x);
   const AccessorRO<   int, 3> acc_nType_y          (regions[0], FID_nType_y);
   const AccessorRO<   int, 3> acc_nType_z          (regions[0], FID_nType_z);

   // Accessors for metrics
   const AccessorWO<double, 3> acc_dcsi_e(regions[1], FID_dcsi_e);
   const AccessorWO<double, 3> acc_deta_e(regions[1], FID_deta_e);
   const AccessorWO<double, 3> acc_dzet_e(regions[1], FID_dzet_e);

   const AccessorWO<double, 3> acc_dcsi_d(regions[1], FID_dcsi_d);
   const AccessorWO<double, 3> acc_deta_d(regions[1], FID_deta_d);
   const AccessorWO<double, 3> acc_dzet_d(regions[1], FID_dzet_d);

   const AccessorWO<double, 3> acc_dcsi_s(regions[1], FID_dcsi_s);
   const AccessorWO<double, 3> acc_deta_s(regions[1], FID_deta_s);
   const AccessorWO<double, 3> acc_dzet_s(regions[1], FID_dzet_s);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;

   // Determine the grid width from bounding box
   const double xWidth = args.bBox.v1[0] - args.bBox.v0[0];
   const double yWidth = args.bBox.v3[1] - args.bBox.v0[1];
   const double zWidth = args.bBox.v4[2] - args.bBox.v0[2];

   // loop on the x direction
   const coord_t xsize = getSize<Xdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++) {

         double xM_e;
         double xP_e;

         // Reconstruct the coordinate at i-1/2 of the first point
         {
            const Point<3> p = Point<3>{r_MyFluid.lo.x,j,k};
            const Point<3> pm1 = warpPeriodic<Xdir, Minus>(Fluid_bounds, p, xsize, offM1(acc_nType_x[p]));
            // Reconstruct x for Euler metrics
            xM_e = reconstructCoordEuler<Xdir>(acc_centerCoordinates, pm1, xWidth, acc_nType_x[pm1], xsize, Fluid_bounds);
            xM_e = unwarpCoordinate<Xdir>(xM_e, xWidth, -1, p, Fluid_bounds);
         }

         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            xP_e = reconstructCoordEuler<Xdir>(acc_centerCoordinates, p, xWidth, acc_nType_x[p], xsize, Fluid_bounds);

            // Compute the metrics
            acc_dcsi_e[p] = 1.0/(xP_e - xM_e);
            ComputeDiffusionMetrics<Xdir>(acc_dcsi_d, acc_dcsi_s, acc_centerCoordinates, p,
                                          xWidth, acc_nType_x[p], xsize, Fluid_bounds);

            // Store plus values for next point
            xM_e = xP_e;
         }
      }

   // loop on the y direction
   const coord_t ysize = getSize<Ydir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {

         double yM_e;
         double yP_e;

         // Reconstruct the coordinate at j-1/2 of the first point
         {
            const Point<3> p = Point<3>{i,r_MyFluid.lo.y,k};
            const Point<3> pm1 = warpPeriodic<Ydir, Minus>(Fluid_bounds, p, ysize, offM1(acc_nType_y[p]));
            // Reconstruct y for Euler metrics
            yM_e = reconstructCoordEuler<Ydir>(acc_centerCoordinates, pm1, yWidth, acc_nType_y[pm1], ysize, Fluid_bounds);
            yM_e = unwarpCoordinate<Ydir>(yM_e, yWidth, -1, p, Fluid_bounds);
         }

         for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++) {
            const Point<3> p = Point<3>{i,j,k};
            yP_e = reconstructCoordEuler<Ydir>(acc_centerCoordinates, p, yWidth, acc_nType_y[p], ysize, Fluid_bounds);

            // Compute the metrics
            acc_deta_e[p] = 1.0/(yP_e - yM_e);
            ComputeDiffusionMetrics<Ydir>(acc_deta_d, acc_deta_s, acc_centerCoordinates, p,
                                          yWidth, acc_nType_y[p], ysize, Fluid_bounds);

            // Store plus values for next point
            yM_e = yP_e;
         }
      }

   // loop on the z direction
   const coord_t zsize = getSize<Zdir>(Fluid_bounds);
   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(2)
#endif
   for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
      for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {

         double zM_e;
         double zP_e;

         // Reconstruct the coordinate at k-1/2 of the first point
         {
            const Point<3> p = Point<3>{i,j,r_MyFluid.lo.z};
            const Point<3> pm1 = warpPeriodic<Zdir, Minus>(Fluid_bounds, p, zsize, offM1(acc_nType_z[p]));
            // Reconstruct z for Euler metrics
            zM_e = reconstructCoordEuler<Zdir>(acc_centerCoordinates, pm1, zWidth, acc_nType_z[pm1], zsize, Fluid_bounds);
            zM_e = unwarpCoordinate<Zdir>(zM_e, zWidth, -1, p, Fluid_bounds);
         }

         for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++) {
            const Point<3> p = Point<3>{i,j,k};
            zP_e = reconstructCoordEuler<Zdir>(acc_centerCoordinates, p, zWidth, acc_nType_z[p], zsize, Fluid_bounds);

            // Compute the metrics
            acc_dzet_e[p] = 1.0/(zP_e - zM_e);
            ComputeDiffusionMetrics<Zdir>(acc_dzet_d, acc_dzet_s, acc_centerCoordinates, p,
                                          zWidth, acc_nType_z[p], zsize, Fluid_bounds);

            // Store plus values for next point
            zM_e = zP_e;
         }
      }

}

template<direction dir>
void CorrectGhostMetricTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 2);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<  Vec3, 3> acc_centerCoordinates(regions[0], FID_centerCoordinates);
   const AccessorRO<   int, 3> acc_nType            (regions[0], FID_nType);

   // Accessors for metrics
   const AccessorRW<double, 3> acc_m                (regions[1], FID_m);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            if      (acc_nType[p] == L_S_node) CorrectLeftStaggered(  acc_m, acc_centerCoordinates, p);
            else if (acc_nType[p] == L_C_node) CorrectLeftCollocated( acc_m, acc_centerCoordinates, p);
            else if (acc_nType[p] == R_S_node) CorrectRightStaggered( acc_m, acc_centerCoordinates, p);
            else if (acc_nType[p] == R_C_node) CorrectRightCollocated(acc_m, acc_centerCoordinates, p);
         }
}

// Specielize CorrectGhostMetricTask for the X direction
template<>
/*static*/ const char * const    CorrectGhostMetricTask<Xdir>::TASK_NAME = "CorrectGhostMetricX";
template<>
/*static*/ const int             CorrectGhostMetricTask<Xdir>::TASK_ID = TID_CorrectGhostMetricX;
template<>
/*static*/ const FieldID         CorrectGhostMetricTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         CorrectGhostMetricTask<Xdir>::FID_m = FID_dcsi_e;

// Specielize CorrectGhostMetricTask for the Y direction
template<>
/*static*/ const char * const    CorrectGhostMetricTask<Ydir>::TASK_NAME = "CorrectGhostMetricY";
template<>
/*static*/ const int             CorrectGhostMetricTask<Ydir>::TASK_ID = TID_CorrectGhostMetricY;
template<>
/*static*/ const FieldID         CorrectGhostMetricTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         CorrectGhostMetricTask<Ydir>::FID_m = FID_deta_e;

// Specielize CorrectGhostMetricTask for the Z direction
template<>
/*static*/ const char * const    CorrectGhostMetricTask<Zdir>::TASK_NAME = "CorrectGhostMetricZ";
template<>
/*static*/ const int             CorrectGhostMetricTask<Zdir>::TASK_ID = TID_CorrectGhostMetricZ;
template<>
/*static*/ const FieldID         CorrectGhostMetricTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         CorrectGhostMetricTask<Zdir>::FID_m = FID_dzet_e;

void register_metric_tasks() {

   TaskHelper::register_hybrid_variants<InitializeMetricTask>();

   TaskHelper::register_hybrid_variants<CorrectGhostMetricTask<Xdir>>();
   TaskHelper::register_hybrid_variants<CorrectGhostMetricTask<Ydir>>();
   TaskHelper::register_hybrid_variants<CorrectGhostMetricTask<Zdir>>();

};
