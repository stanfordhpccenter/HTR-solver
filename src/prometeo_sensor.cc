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

#include "prometeo_sensor.hpp"
#include "prometeo_sensor.inl"

template<direction dir>
void UpdateShockSensorTask<dir>::cpu_base_impl(
                      const Args &args,
                      const std::vector<PhysicalRegion> &regions,
                      const std::vector<Future>         &futures,
                      Context ctx, Runtime *runtime)
{
   assert(regions.size() == 4);
   assert(futures.size() == 0);

   // Accessors for variables in the Ghost regions
   const AccessorRO<VecNEq, 3> acc_Conserved        (regions[0], FID_Conserved);
   const AccessorRO<  Vec3, 3> acc_vGradX           (regions[0], FID_velocityGradientX);
   const AccessorRO<  Vec3, 3> acc_vGradY           (regions[0], FID_velocityGradientY);
   const AccessorRO<  Vec3, 3> acc_vGradZ           (regions[0], FID_velocityGradientZ);

   // Accessors for node type
   const AccessorRO<   int, 3> acc_nType            (regions[1], FID_nType);

   // Accessors for shock sensor
   const AccessorWO<  bool, 3> acc_shockSensor      (regions[2], FID_shockSensor);

   // Extract execution domains
   Rect<3> r_MyFluid = runtime->get_index_space_domain(ctx, args.ModCells.get_index_space());
   Rect<3> Fluid_bounds = args.Fluid_bounds;
   const coord_t size = getSize<dir>(Fluid_bounds);

   // Compute vorticity scale
   const double eps = std::max(args.vorticityScale*args.vorticityScale, 1e-6);

   // Here we are assuming C layout of the instance
#ifdef REALM_USE_OPENMP
   #pragma omp parallel for collapse(3)
#endif
   for (int k = r_MyFluid.lo.z; k <= r_MyFluid.hi.z; k++)
      for (int j = r_MyFluid.lo.y; j <= r_MyFluid.hi.y; j++)
         for (int i = r_MyFluid.lo.x; i <= r_MyFluid.hi.x; i++) {
            const Point<3> p = Point<3>{i,j,k};
            const Point<3> pM2 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM2(acc_nType[p]));
            const Point<3> pM1 = warpPeriodic<dir, Minus>(Fluid_bounds, p, size, offM1(acc_nType[p]));
            const Point<3> pP1 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP1(acc_nType[p]));
            const Point<3> pP2 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP2(acc_nType[p]));
            const Point<3> pP3 = warpPeriodic<dir, Plus >(Fluid_bounds, p, size, offP3(acc_nType[p]));

            const double Phi = std::max(std::max(std::max(std::max(std::max(
                                 DucrosSensor(acc_vGradX[pM2].v, acc_vGradY[pM2].v, acc_vGradZ[pM2].v, eps),
                                 DucrosSensor(acc_vGradX[pM1].v, acc_vGradY[pM1].v, acc_vGradZ[pM1].v, eps)),
                                 DucrosSensor(acc_vGradX[p  ].v, acc_vGradY[p  ].v, acc_vGradZ[p  ].v, eps)),
                                 DucrosSensor(acc_vGradX[pP1].v, acc_vGradY[pP1].v, acc_vGradZ[pP1].v, eps)),
                                 DucrosSensor(acc_vGradX[pP2].v, acc_vGradY[pP2].v, acc_vGradZ[pP2].v, eps)),
                                 DucrosSensor(acc_vGradX[pP3].v, acc_vGradY[pP3].v, acc_vGradZ[pP3].v, eps));

            bool sensor = true;
            for (int h=0; h<nSpec; h++)
               sensor = sensor && TENOsensor(acc_Conserved[pM2][h], acc_Conserved[pM1][h], acc_Conserved[p  ][h],
                                             acc_Conserved[pP1][h], acc_Conserved[pP2][h], acc_Conserved[pP3][h],
                                             acc_nType[p], Phi);
            acc_shockSensor[p] = sensor;
         }

}

// Specielize UpdateShockSensorTask for the X direction
template<>
/*static*/ const char * const    UpdateShockSensorTask<Xdir>::TASK_NAME = "UpdateShockSensorX";
template<>
/*static*/ const int             UpdateShockSensorTask<Xdir>::TASK_ID = TID_UpdateShockSensorX;
template<>
/*static*/ const FieldID         UpdateShockSensorTask<Xdir>::FID_nType = FID_nType_x;
template<>
/*static*/ const FieldID         UpdateShockSensorTask<Xdir>::FID_shockSensor = FID_shockSensorX;

// Specielize UpdateShockSensorTask for the Y direction
template<>
/*static*/ const char * const    UpdateShockSensorTask<Ydir>::TASK_NAME = "UpdateShockSensorY";
template<>
/*static*/ const int             UpdateShockSensorTask<Ydir>::TASK_ID = TID_UpdateShockSensorY;
template<>
/*static*/ const FieldID         UpdateShockSensorTask<Ydir>::FID_nType = FID_nType_y;
template<>
/*static*/ const FieldID         UpdateShockSensorTask<Ydir>::FID_shockSensor = FID_shockSensorY;

// Specielize UpdateShockSensorTask for the Z direction
template<>
/*static*/ const char * const    UpdateShockSensorTask<Zdir>::TASK_NAME = "UpdateShockSensorZ";
template<>
/*static*/ const int             UpdateShockSensorTask<Zdir>::TASK_ID = TID_UpdateShockSensorZ;
template<>
/*static*/ const FieldID         UpdateShockSensorTask<Zdir>::FID_nType = FID_nType_z;
template<>
/*static*/ const FieldID         UpdateShockSensorTask<Zdir>::FID_shockSensor = FID_shockSensorZ;

void register_sensor_tasks() {

   TaskHelper::register_hybrid_variants<UpdateShockSensorTask<Xdir>>();
   TaskHelper::register_hybrid_variants<UpdateShockSensorTask<Ydir>>();
   TaskHelper::register_hybrid_variants<UpdateShockSensorTask<Zdir>>();

};
