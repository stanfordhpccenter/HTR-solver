-- Copyright (c) "2019, by Stanford University
--               Developer: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
--                         HTR solver: An open-source exascale-oriented task-based
--                         multi-GPU high-order code for hypersonic aerothermodynamics.
--                         Computer Physics Communications (In Press), 107262"
-- All rights reserved.
-- 
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--    * Redistributions of source code must retain the above copyright
--      notice, this list of conditions and the following disclaimer.
--    * Redistributions in binary form must reproduce the above copyright
--      notice, this list of conditions and the following disclaimer in the
--      documentation and/or other materials provided with the distribution.
-- 
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
-- ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import "regent"

return function(SCHEMA, MIX, Fluid_columns, grid_partitions) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require 'util-desugared'
local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local sin  = regentlib.sin(double)
local exp  = regentlib.exp(double)
local pow  = regentlib.pow(double)

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives   = CONST.Primitives
local ProfilesVars = CONST.ProfilesVars

-------------------------------------------------------------------------------
-- BC ROUTINES
-------------------------------------------------------------------------------
-- Set up stuff for RHS of NSCBC inflow
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.InitializeGhostNSCBC(Fluid : region(ispace(int3d), Fluid_columns),
                                  Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                  mix : MIX.Mixture)
where
   reads(Fluid.[Primitives]),
   writes(Fluid.{velocity_old_NSCBC, temperature_old_NSCBC, dudtBoundary, dTdtBoundary})
do
   var BC   = Fluid_BC[0]

   __demand(__openmp)
   for c in BC do
      BC[c].velocity_old_NSCBC = BC[c].velocity
      BC[c].temperature_old_NSCBC = BC[c].temperature
      BC[c].dudtBoundary = 0.0
      BC[c].dTdtBoundary= 0.0
   end
end

------------------------------------------------------------------------
-- BC utils
-----------------------------------------------------------------------
local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task SetDirichletBC(Fluid    : region(ispace(int3d), Fluid_columns),
                    Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                    Pbc : double)
where
   reads(Fluid.[ProfilesVars]),
   writes(Fluid.[Primitives])
do
   var BC   = Fluid_BC[0]
   __demand(__openmp)
   for c in BC do
      BC[c].MolarFracs  = BC[c].MolarFracs_profile
      BC[c].velocity    = BC[c].velocity_profile
      BC[c].temperature = BC[c].temperature_profile
      BC[c].pressure    = Pbc
   end
end

local function mkSetNSCBC_InflowBC(dir)
   local SetNSCBC_InflowBC
   local idx
   if dir == "x" then
      idx = 0
   elseif dir == "y" then
      idx = 1
   elseif dir == "z" then
      idx = 2
   end
   -- NOTE: It is safe to not pass the ghost regions to this task, because we
   -- always group ghost cells with their neighboring interior cells.
   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetNSCBC_InflowBC(Fluid    : region(ispace(int3d), Fluid_columns),
                          Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                          mix : MIX.Mixture,
                          Pbc : double)
   where
      reads(Fluid.SoS),
      reads(Fluid.Conserved),
      reads(Fluid.[ProfilesVars]),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      __demand(__openmp)
      for c in BC do
         BC[c].MolarFracs  = BC[c].MolarFracs_profile
         BC[c].velocity    = BC[c].velocity_profile
         BC[c].temperature = BC[c].temperature_profile
         if (BC[c].velocity_profile[idx] >= BC[c].SoS) then
            -- It is supersonic, everything is imposed by the BC
            BC[c].pressure = Pbc
         else
            -- Compute pressure from NSCBC conservation equations
            var rhoYi : double[nSpec]
            for i=0, nSpec do
               rhoYi[i] = BC[c].Conserved[i]
            end
            var rho = MIX.GetRhoFromRhoYi(rhoYi)
            var MixW = MIX.GetMolarWeightFromXi(BC[c].MolarFracs_profile, mix)
            BC[c].pressure = MIX.GetPFromRhoAndT(rho, MixW, BC[c].temperature_profile)
         end

      end
   end
   return SetNSCBC_InflowBC
end

local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task SetNSCBC_OutflowBC(Fluid    : region(ispace(int3d), Fluid_columns),
                        Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                        mix : MIX.Mixture)
where
   reads(Fluid.Conserved),
   reads(Fluid.temperature),
   writes(Fluid.[Primitives])
do
   var BC   = Fluid_BC[0]
   var BCst = Fluid_BC[1]

   __demand(__openmp)
   for c in BC do
      -- Compute values from NSCBC conservation equations
      var rhoYi : double[nSpec]
      for i=0, nSpec do
         rhoYi[i] = BC[c].Conserved[i]
      end
      var rho = MIX.GetRhoFromRhoYi(rhoYi)
      var Yi = MIX.GetYi(rho, rhoYi)
      Yi = MIX.ClipYi(Yi)
      var MixW = MIX.GetMolarWeightFromYi(Yi, mix)
      BC[c].MolarFracs = MIX.GetMolarFractions(MixW, Yi, mix)
      var rhoInv = 1.0/rho;
      var velocity = array(BC[c].Conserved[irU+0]*rhoInv,
                           BC[c].Conserved[irU+1]*rhoInv,
                           BC[c].Conserved[irU+2]*rhoInv)
      BC[c].velocity = velocity
      var kineticEnergy = (0.5*MACRO.dot(velocity, velocity))
      var InternalEnergy = BC[c].Conserved[irE]*rhoInv - kineticEnergy
      BC[c].temperature = MIX.GetTFromInternalEnergy(InternalEnergy, BC[c].temperature, Yi, mix);
      BC[c].pressure    = MIX.GetPFromRhoAndT(rho, MixW, BC[c].temperature)
   end
end

local function mkSetAdiabaticWallBC(dir)
   local SetAdiabaticWallBC
   
   local mk_cint
   if dir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif dir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif dir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif dir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif dir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif dir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetAdiabaticWallBC(Fluid    : region(ispace(int3d), Fluid_columns),
                           Fluid_BC : partition(disjoint, Fluid, ispace(int1d)))
   where
      reads(Fluid.{MolarFracs, pressure, temperature}),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];
         BC[c].MolarFracs  = BCst[c_int].MolarFracs
         BC[c].pressure    = BCst[c_int].pressure
         BC[c].temperature = BCst[c_int].temperature
         BC[c].velocity    = array(0.0, 0.0, 0.0)
      end
   end
   return SetAdiabaticWallBC
end

local function mkSetIsothermalWallBC(dir)
   local SetIsothermalWallBC

   local mk_cint
   if dir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif dir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif dir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif dir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif dir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif dir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetIsothermalWallBC(Fluid    : region(ispace(int3d), Fluid_columns),
                            Fluid_BC : partition(disjoint, Fluid, ispace(int1d)))
   where
      reads(Fluid.{MolarFracs, pressure}),
      reads(Fluid.temperature_profile),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];
         BC[c].MolarFracs  = BCst[c_int].MolarFracs
         BC[c].pressure    = BCst[c_int].pressure
         BC[c].temperature = BC[c].temperature_profile
         BC[c].velocity    = array(0.0, 0.0, 0.0)
      end
   end
   return SetIsothermalWallBC
end

local function mkSetSuctionAndBlowingWallBC(dir)
   local SetSuctionAndBlowingWallBC

   local iStrm
   local iNorm
   local iSpan
   if dir == "yNeg" then
      iStrm = 0
      iNorm = 1
      iSpan = 2
   else assert(false) end

   local mk_cint
   if dir == "xNeg" then
      mk_cint = function(c) return rexpr (c+{ 1, 0, 0}) end end
   elseif dir == "xPos" then
      mk_cint = function(c) return rexpr (c+{-1, 0, 0}) end end
   elseif dir == "yNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 1, 0}) end end
   elseif dir == "yPos" then
      mk_cint = function(c) return rexpr (c+{ 0,-1, 0}) end end
   elseif dir == "zNeg" then
      mk_cint = function(c) return rexpr (c+{ 0, 0, 1}) end end
   elseif dir == "zPos" then
      mk_cint = function(c) return rexpr (c+{ 0, 0,-1}) end end
   else assert(false) end

   local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
   task SetSuctionAndBlowingWallBC(Fluid    : region(ispace(int3d), Fluid_columns),
                                   Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                   Xmin   : double,
                                   Xmax   : double,
                                   X0     : double,
                                   sigma  : double,
                                   Zw     : double,
                                   nModes : int,
                                   A      : double[20],
                                   omega  : double[20],
                                   beta   : double[20],
                                   Width  : double,
                                   Origin : double,
                                   time   : double)
   where
      reads(Fluid.{MolarFracs, pressure}),
      reads(Fluid.centerCoordinates),
      reads(Fluid.temperature_profile),
      writes(Fluid.[Primitives])
   do
      var BC   = Fluid_BC[0]
      var BCst = Fluid_BC[1]

      __demand(__openmp)
      for c in BC do
         var c_int = [mk_cint(rexpr c end)];

         BC[c].MolarFracs  = BCst[c_int].MolarFracs
         BC[c].pressure    = BCst[c_int].pressure
         BC[c].temperature = BC[c].temperature_profile
         var velocity      = array(0.0, 0.0, 0.0)

         if ((BC[c].centerCoordinates[iStrm] > Xmin) and
             (BC[c].centerCoordinates[iStrm] < Xmax)) then

            var f = exp(-0.5*pow((BC[c].centerCoordinates[iStrm] - X0)/sigma, 2))

            var g = 1.0 + 0.1*( exp(-pow((BC[c].centerCoordinates[iSpan] - (0.5*Width+Origin) - Zw)/Zw, 2)) - 
                                exp(-pow((BC[c].centerCoordinates[iSpan] - (0.5*Width+Origin) + Zw)/Zw, 2)))

            var h = 0.0
            for i=0, nModes do
               h += A[i]*sin(omega[i]*time -
                             beta[i]*BC[c].centerCoordinates[iSpan])
            end
            velocity[iNorm] = f*g*h
         end
         BC[c].velocity = velocity
      end
   end
   return SetSuctionAndBlowingWallBC
end

-- Update the ghost cells to impose boundary conditions
__demand(__inline)
task Exports.UpdateGhostPrimitives(Fluid : region(ispace(int3d), Fluid_columns),
                                   tiles : ispace(int3d),
                                   Fluid_Partitions : grid_partitions(Fluid, tiles),
                                   config : SCHEMA.Config,
                                   Mix : MIX.Mixture,
                                   Integrator_simTime : double)
where
   reads writes(Fluid)
do
   var BC_xBCLeft  = config.BC.xBCLeft
   var BC_xBCRight = config.BC.xBCRight
   var BC_yBCLeft  = config.BC.yBCLeft
   var BC_yBCRight = config.BC.yBCRight
   var BC_zBCLeft  = config.BC.zBCLeft
   var BC_zBCRight = config.BC.zBCRight
   var {p_All, p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos} = Fluid_Partitions

   -- Start updating BCs that are local
   -- xNeg BC
   if (BC_xBCLeft == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in tiles do
         SetDirichletBC(p_All[c], p_xNeg[c], config.BC.xBCLeftP)
      end
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetNSCBC_InflowBC("x")](p_All[c], p_xNeg[c], Mix, config.BC.xBCLeftP)
      end
   end

   -- xPos BC
   if (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in tiles do
         SetDirichletBC(p_All[c], p_xPos[c], config.BC.xBCRightP)
      end
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      __demand(__index_launch)
      for c in tiles do
         SetNSCBC_OutflowBC(p_All[c], p_xPos[c], Mix)
      end
   end

   -- yNeg BC
   if (BC_yBCLeft == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in tiles do
         SetDirichletBC(p_All[c], p_yNeg[c], config.BC.yBCLeftP)
      end
   end

   -- yPos BC
   if (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in tiles do
         SetDirichletBC(p_All[c], p_yPos[c], config.BC.yBCRightP)
      end
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      __demand(__index_launch)
      for c in tiles do
         SetNSCBC_OutflowBC(p_All[c], p_yPos[c], Mix)
      end
   end

   -- yNeg BC
   if (BC_zBCLeft == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in tiles do
         SetDirichletBC(p_All[c], p_zNeg[c], config.BC.zBCLeftP)
      end
   end

   -- yPos BC
   if (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      __demand(__index_launch)
      for c in tiles do
         SetDirichletBC(p_All[c], p_zPos[c], config.BC.zBCRightP)
      end
   end

   -- BCs that depend on other cells need to be set at the end in the order Z, Y, X
   -- yNeg BC
   if (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetAdiabaticWallBC("yNeg")](p_All[c], p_yNeg[c])
      end
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetIsothermalWallBC("yNeg")](p_All[c], p_yNeg[c])
      end
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetSuctionAndBlowingWallBC("yNeg")](p_All[c], p_yNeg[c],
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.Xmin,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.Xmax,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.X0,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.sigma,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.Zw,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.A.length,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.A.values,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.omega.values,
                                                config.BC.yBCLeftInflowProfile.u.SuctionAndBlowing.beta.values,
                                                config.Grid.zWidth,
                                                config.Grid.origin[2],
                                                Integrator_simTime)
      end
   end

   -- yPos BC
   if (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetAdiabaticWallBC("yPos")](p_All[c], p_yPos[c])
      end
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetIsothermalWallBC("yPos")](p_All[c], p_yPos[c])
      end
   end

   -- xNeg BC
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetAdiabaticWallBC("xNeg")](p_All[c], p_xNeg[c])
      end
   end

   -- xPos BC
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall) then
      __demand(__index_launch)
      for c in tiles do
         [mkSetAdiabaticWallBC("xPos")](p_All[c], p_xPos[c])
      end
   end
end

-- Update the time derivative values needed for the inflow
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task Exports.UpdateNSCBCGhostCellTimeDerivatives(Fluid : region(ispace(int3d), Fluid_columns),
                                                 Fluid_BC : partition(disjoint, Fluid, ispace(int1d)),
                                                 Integrator_deltaTime : double)
where
  reads(Fluid.{velocity, temperature}),
  writes(Fluid.{dudtBoundary, dTdtBoundary}),
  reads writes(Fluid.{velocity_old_NSCBC, temperature_old_NSCBC})
do
   var BC   = Fluid_BC[0]
   __demand(__openmp)
   for c in BC do
      BC[c].dudtBoundary = (BC[c].velocity[0] - BC[c].velocity_old_NSCBC[0]) / Integrator_deltaTime
      BC[c].dTdtBoundary = (BC[c].temperature - BC[c].temperature_old_NSCBC) / Integrator_deltaTime
      BC[c].velocity_old_NSCBC    = BC[c].velocity
      BC[c].temperature_old_NSCBC = BC[c].temperature
   end
end

return Exports end

