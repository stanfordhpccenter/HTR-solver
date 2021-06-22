-- Copyright (c) "2019, by Stanford University
--               Developer: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
--                         HTR solver: An open-source exascale-oriented task-based
--                         multi-GPU high-order code for hypersonic aerothermodynamics.
--                         Computer Physics Communications 255, 107262"
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

return function(SCHEMA, METRIC, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require "util-desugared"
local Config = SCHEMA.Config

-------------------------------------------------------------------------------
-- PARTITIONS FSPACES
-------------------------------------------------------------------------------

local struct indices_columns {
   -- X-stencil indices
   cm2_x : int3d;
   cm1_x : int3d;
   cp1_x : int3d;
   cp2_x : int3d;
   cp3_x : int3d;
   -- Y-stencil indices
   cm2_y : int3d;
   cm1_y : int3d;
   cp1_y : int3d;
   cp2_y : int3d;
   cp3_y : int3d;
   -- Z-stencil indices
   cm2_z : int3d;
   cm1_z : int3d;
   cp1_z : int3d;
   cp2_z : int3d;
   cp3_z : int3d;
}

local fspace zones_partitions(Fluid  : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   -- Partitions
   p_All      : partition(disjoint, Fluid, tiles),
   p_Interior : partition(disjoint, Fluid, tiles),
   p_AllBCs   : partition(disjoint, Fluid, tiles),
   -- Partitions for reconstruction operator
   p_x_faces  : partition(disjoint, Fluid, tiles),
   p_y_faces  : partition(disjoint, Fluid, tiles),
   p_z_faces  : partition(disjoint, Fluid, tiles),
   -- Partitions for divergence operator
   p_x_divg   : partition(disjoint, Fluid, tiles),
   p_y_divg   : partition(disjoint, Fluid, tiles),
   p_z_divg   : partition(disjoint, Fluid, tiles),
   p_solved   : partition(disjoint, Fluid, tiles),
   -- Partitions containing all ghost for each side
   AllxNeg    : partition(disjoint, Fluid, ispace(int1d)),
   AllxPos    : partition(disjoint, Fluid, ispace(int1d)),
   AllyNeg    : partition(disjoint, Fluid, ispace(int1d)),
   AllyPos    : partition(disjoint, Fluid, ispace(int1d)),
   AllzNeg    : partition(disjoint, Fluid, ispace(int1d)),
   AllzPos    : partition(disjoint, Fluid, ispace(int1d)),
   p_AllxNeg  : cross_product(p_All, AllxNeg),
   p_AllxPos  : cross_product(p_All, AllxPos),
   p_AllyNeg  : cross_product(p_All, AllyNeg),
   p_AllyPos  : cross_product(p_All, AllyPos),
   p_AllzNeg  : cross_product(p_All, AllzNeg),
   p_AllzPos  : cross_product(p_All, AllzPos),
   -- BC partitions
   xNeg       : partition(disjoint, Fluid, ispace(int1d)),
   xPos       : partition(disjoint, Fluid, ispace(int1d)),
   yNeg       : partition(disjoint, Fluid, ispace(int1d)),
   yPos       : partition(disjoint, Fluid, ispace(int1d)),
   zNeg       : partition(disjoint, Fluid, ispace(int1d)),
   zPos       : partition(disjoint, Fluid, ispace(int1d)),
   p_xNeg     : cross_product(p_All, xNeg),
   p_xPos     : cross_product(p_All, xPos),
   p_yNeg     : cross_product(p_All, yNeg),
   p_yPos     : cross_product(p_All, yPos),
   p_zNeg     : cross_product(p_All, zNeg),
   p_zPos     : cross_product(p_All, zPos),
   -- BC partitions ispaces
   xNeg_ispace : ispace(int3d),
   xPos_ispace : ispace(int3d),
   yNeg_ispace : ispace(int3d),
   yPos_ispace : ispace(int3d),
   zNeg_ispace : ispace(int3d),
   zPos_ispace : ispace(int3d)
}

local fspace output_partitions(Fluid  : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   -- Restart
   p_Output : partition(disjoint, Fluid, tiles),
   -- Volume probes
   Vprobes  : partition(aliased, Fluid, ispace(int1d)),
   p_Vprobes : cross_product(Vprobes, p_Output),
}

local fspace ghost_partitions(Fluid  : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   -- All With Ghosts
   p_AllWithGhosts : partition(aliased, Fluid, tiles),
   -- Fluxes stencil access
   p_XFluxGhosts : partition(aliased, Fluid, tiles),
   p_YFluxGhosts : partition(aliased, Fluid, tiles),
   p_ZFluxGhosts : partition(aliased, Fluid, tiles),
   -- Diffusion fluxes stencil access
   p_XDiffGhosts : partition(aliased, Fluid, tiles),
   p_YDiffGhosts : partition(aliased, Fluid, tiles),
   p_ZDiffGhosts : partition(aliased, Fluid, tiles),
   -- Euler fluxes stencil access
   p_XEulerGhosts2 : partition(aliased, Fluid, tiles),
   p_YEulerGhosts2 : partition(aliased, Fluid, tiles),
   p_ZEulerGhosts2 : partition(aliased, Fluid, tiles),
   -- Shock sensors stencil access
   p_XSensorGhosts2 : partition(aliased, Fluid, tiles),
   p_YSensorGhosts2 : partition(aliased, Fluid, tiles),
   p_ZSensorGhosts2 : partition(aliased, Fluid, tiles),
   -- Metric routines
   p_MetricGhosts  : partition(aliased, Fluid, tiles),
   -- Euler fluxes routines
   p_XEulerGhosts : partition(aliased, Fluid, tiles),
   p_YEulerGhosts : partition(aliased, Fluid, tiles),
   p_ZEulerGhosts : partition(aliased, Fluid, tiles),
   -- Gradient routines
   p_GradientGhosts : partition(aliased, Fluid, tiles),
   -- Shock sensor ghosts
   p_XSensorGhosts : partition(aliased, Fluid, tiles),
   p_YSensorGhosts : partition(aliased, Fluid, tiles),
   p_ZSensorGhosts : partition(aliased, Fluid, tiles),
}

local fspace average_ghost_partitions(Fluid  : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   -- Gradient routines
   p_GradientGhosts : partition(aliased, Fluid, tiles),
}

Exports.zones_partitions = zones_partitions
Exports.output_partitions = output_partitions
Exports.ghost_partitions = ghost_partitions
Exports.average_ghost_partitions = average_ghost_partitions

-------------------------------------------------------------------------------
-- ZONES PARTITIONING ROUTINES
-------------------------------------------------------------------------------

local function addToBcColoring(c, rect, stencil)
   return rquote
      regentlib.c.legion_multi_domain_point_coloring_color_domain([c], int1d(0), [rect])
      regentlib.c.legion_multi_domain_point_coloring_color_domain([c], int1d(1), [rect] + [stencil])
   end
end

local function isWall(Type)
   return rexpr
      ((Type == SCHEMA.FlowBC_AdiabaticWall)  or
       (Type == SCHEMA.FlowBC_IsothermalWall) or
       (Type == SCHEMA.FlowBC_SuctionAndBlowingWall))
   end
end

local function isDirichlet(Type)
   return rexpr
      (Type == SCHEMA.FlowBC_Dirichlet)
   end
end

local function isNSCBC_Inflow(Type)
   return rexpr
      ((Type == SCHEMA.FlowBC_NSCBC_Inflow)  or
       (Type == SCHEMA.FlowBC_RecycleRescaling))
   end
end

local function isIncomingShock(Type)
   return rexpr
      (Type == SCHEMA.FlowBC_IncomingShock)
   end
end

local function isNSCBC_Outflow(Type)
   return rexpr
      (Type == SCHEMA.FlowBC_NSCBC_Outflow)
   end
end

local function addRegionsToColor(p, c, Name, indices)
   local __quotes = terralib.newlist()

   local stencil
   if     (Name == "xNeg") then stencil = rexpr int3d{ 1, 0, 0} end
   elseif (Name == "xPos") then stencil = rexpr int3d{-1, 0, 0} end
   elseif (Name == "yNeg") then stencil = rexpr int3d{ 0, 1, 0} end
   elseif (Name == "yPos") then stencil = rexpr int3d{ 0,-1, 0} end
   elseif (Name == "zNeg") then stencil = rexpr int3d{ 0, 0, 1} end
   elseif (Name == "zPos") then stencil = rexpr int3d{ 0, 0,-1} end
   else assert(false) end

   for k, ind in pairs(indices) do
      __quotes:insert(addToBcColoring(c, rexpr p[ind].bounds end, stencil))
   end
   return __quotes
end

-- We give the following priorities to BCS:
-- - Walls
-- - Dirichlet
-- - NSCBC_Inflow
-- - IncomingShock
-- - NSCBC_Outflow
--
-- 1 has priority on 2
local function EdgeTieBreakPolicy(Type1, coloring1, Name1,
                                  Type2, coloring2, Name2,
                                  rect)
   local stencil1
   if     (Name1 == "xNeg") then stencil1 = rexpr int3d{ 1, 0, 0} end
   elseif (Name1 == "xPos") then stencil1 = rexpr int3d{-1, 0, 0} end
   elseif (Name1 == "yNeg") then stencil1 = rexpr int3d{ 0, 1, 0} end
   elseif (Name1 == "yPos") then stencil1 = rexpr int3d{ 0,-1, 0} end
   elseif (Name1 == "zNeg") then stencil1 = rexpr int3d{ 0, 0, 1} end
   elseif (Name1 == "zPos") then stencil1 = rexpr int3d{ 0, 0,-1} end
   else assert(false) end

   local stencil2
   if     (Name2 == "xNeg") then stencil2 = rexpr int3d{ 1, 0, 0} end
   elseif (Name2 == "xPos") then stencil2 = rexpr int3d{-1, 0, 0} end
   elseif (Name2 == "yNeg") then stencil2 = rexpr int3d{ 0, 1, 0} end
   elseif (Name2 == "yPos") then stencil2 = rexpr int3d{ 0,-1, 0} end
   elseif (Name2 == "zNeg") then stencil2 = rexpr int3d{ 0, 0, 1} end
   elseif (Name2 == "zPos") then stencil2 = rexpr int3d{ 0, 0,-1} end
   else assert(false) end

   return rquote

      var is_Wall1 = [isWall(Type1)];
      var is_Wall2 = [isWall(Type2)];

      var is_Dirichlet1 = [isDirichlet(Type1)];
      var is_Dirichlet2 = [isDirichlet(Type2)];

      var is_NSCBC_Inflow1 = [isNSCBC_Inflow(Type1)];
      var is_NSCBC_Inflow2 = [isNSCBC_Inflow(Type2)];

      var is_IncomingShock1 = [isIncomingShock(Type1)];
      var is_IncomingShock2 = [isIncomingShock(Type2)];

      var is_NSCBC_Outflow1 = [isNSCBC_Outflow(Type1)];
      var is_NSCBC_Outflow2 = [isNSCBC_Outflow(Type2)];

      -- Walls
      if     is_Wall1 then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif is_Wall2 then
         [addToBcColoring(coloring2, rect, stencil2)]

      -- Dirichlet
      elseif is_Dirichlet1 then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif is_Dirichlet2 then
         [addToBcColoring(coloring2, rect, stencil2)]

      -- NSCBC_Inflow
      elseif is_NSCBC_Inflow1 then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif is_NSCBC_Inflow2 then
         [addToBcColoring(coloring2, rect, stencil2)]

      -- NSCBC_Outflow and IncomingShock
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1) and
              (is_NSCBC_Outflow2 or is_IncomingShock2)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif (is_NSCBC_Outflow1 or is_IncomingShock1) then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif (is_NSCBC_Outflow2 or is_IncomingShock2) then
         [addToBcColoring(coloring2, rect, stencil2)]

      -- Periodic
      elseif ((Type1 == SCHEMA.FlowBC_Periodic) and (Type2 == SCHEMA.FlowBC_Periodic)) then
         -- Nothing to do

      else
         regentlib.assert(false, ["Unhandled case in tie breaking of" .. Name1 .. "-" .. Name2 .. " edge"])
      end
   end
end

-- 1 has priority on 2, 2 has priority on 3
local function CornerTieBreakPolicy(Type1, coloring1, Name1,
                                    Type2, coloring2, Name2,
                                    Type3, coloring3, Name3,
                                    rect)
   local stencil1
   if     (Name1 == "xNeg") then stencil1 = rexpr int3d{ 1, 0, 0} end
   elseif (Name1 == "xPos") then stencil1 = rexpr int3d{-1, 0, 0} end
   elseif (Name1 == "yNeg") then stencil1 = rexpr int3d{ 0, 1, 0} end
   elseif (Name1 == "yPos") then stencil1 = rexpr int3d{ 0,-1, 0} end
   elseif (Name1 == "zNeg") then stencil1 = rexpr int3d{ 0, 0, 1} end
   elseif (Name1 == "zPos") then stencil1 = rexpr int3d{ 0, 0,-1} end
   else assert(false) end

   local stencil2
   if     (Name2 == "xNeg") then stencil2 = rexpr int3d{ 1, 0, 0} end
   elseif (Name2 == "xPos") then stencil2 = rexpr int3d{-1, 0, 0} end
   elseif (Name2 == "yNeg") then stencil2 = rexpr int3d{ 0, 1, 0} end
   elseif (Name2 == "yPos") then stencil2 = rexpr int3d{ 0,-1, 0} end
   elseif (Name2 == "zNeg") then stencil2 = rexpr int3d{ 0, 0, 1} end
   elseif (Name2 == "zPos") then stencil2 = rexpr int3d{ 0, 0,-1} end
   else assert(false) end

   local stencil3
   if     (Name3 == "xNeg") then stencil3 = rexpr int3d{ 1, 0, 0} end
   elseif (Name3 == "xPos") then stencil3 = rexpr int3d{-1, 0, 0} end
   elseif (Name3 == "yNeg") then stencil3 = rexpr int3d{ 0, 1, 0} end
   elseif (Name3 == "yPos") then stencil3 = rexpr int3d{ 0,-1, 0} end
   elseif (Name3 == "zNeg") then stencil3 = rexpr int3d{ 0, 0, 1} end
   elseif (Name3 == "zPos") then stencil3 = rexpr int3d{ 0, 0,-1} end
   else assert(false) end

   return rquote
      var is_Wall1 = [isWall(Type1)];
      var is_Wall2 = [isWall(Type2)];
      var is_Wall3 = [isWall(Type3)];

      var is_Dirichlet1 = [isDirichlet(Type1)];
      var is_Dirichlet2 = [isDirichlet(Type2)];
      var is_Dirichlet3 = [isDirichlet(Type3)];

      var is_NSCBC_Inflow1 = [isNSCBC_Inflow(Type1)];
      var is_NSCBC_Inflow2 = [isNSCBC_Inflow(Type2)];
      var is_NSCBC_Inflow3 = [isNSCBC_Inflow(Type3)];

      var is_IncomingShock1 = [isIncomingShock(Type1)];
      var is_IncomingShock2 = [isIncomingShock(Type2)];
      var is_IncomingShock3 = [isIncomingShock(Type3)];

      var is_NSCBC_Outflow1 = [isNSCBC_Outflow(Type1)];
      var is_NSCBC_Outflow2 = [isNSCBC_Outflow(Type2)];
      var is_NSCBC_Outflow3 = [isNSCBC_Outflow(Type3)];

      -- Walls
      if     is_Wall1 then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif is_Wall2 then
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif is_Wall3 then
         [addToBcColoring(coloring3, rect, stencil3)]

      -- Dirichlet
      elseif is_Dirichlet1 then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif is_Dirichlet2 then
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif is_Dirichlet3 then
         [addToBcColoring(coloring2, rect, stencil3)]

      -- NSCBC_Inflow
      elseif is_NSCBC_Inflow1 then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif is_NSCBC_Inflow2 then
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif is_NSCBC_Inflow3 then
         [addToBcColoring(coloring3, rect, stencil3)]

      -- NSCBC_Outflow and IncomingShock
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1) and
              (is_NSCBC_Outflow2 or is_IncomingShock2) and
              (is_NSCBC_Outflow3 or is_IncomingShock3)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring2, rect, stencil2)];
         [addToBcColoring(coloring3, rect, stencil3)]
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1) and
              (is_NSCBC_Outflow2 or is_IncomingShock2)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1) and
              (is_NSCBC_Outflow3 or is_IncomingShock3)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring3, rect, stencil3)]
      elseif ((is_NSCBC_Outflow2 or is_IncomingShock2) and
              (is_NSCBC_Outflow3 or is_IncomingShock3)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring2, rect, stencil2)];
         [addToBcColoring(coloring3, rect, stencil3)]
      elseif (is_NSCBC_Outflow1 or is_IncomingShock1) then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif (is_NSCBC_Outflow2 or is_IncomingShock2) then
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif (is_NSCBC_Outflow3 or is_IncomingShock3) then
         [addToBcColoring(coloring3, rect, stencil3)]

      -- Periodic
      elseif ((Type1 == SCHEMA.FlowBC_Periodic) and
              (Type2 == SCHEMA.FlowBC_Periodic) and
              (Type3 == SCHEMA.FlowBC_Periodic)) then
         -- Nothing to do

      else
         regentlib.assert(false, ["Unhandled case in tie breaking of" .. Name1 .. "-" .. Name2 .. "-" .. Name3 .. " corner"])
      end
   end
end

__demand(__inline)
task Exports.PartitionZones(Fluid : region(ispace(int3d), Fluid_columns),
                   tiles : ispace(int3d),
                   config : Config,
                   Grid_xBnum : int32, Grid_yBnum : int32, Grid_zBnum : int32)
where
   reads(Fluid)
do
   var p_Fluid =
      [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_All")]
      (Fluid, tiles, int3d{Grid_xBnum,Grid_yBnum,Grid_zBnum}, int3d{0,0,0})

   -- This partition accommodates 27 regions, in the order:
   -- - [ 0]: Interior
   -- (6 faces)
   -- - [ 1]: Faces xNeg
   -- - [ 2]: Faces xPos
   -- - [ 3]: Faces yNeg
   -- - [ 4]: Faces yPos
   -- - [ 5]: Faces zNeg
   -- - [ 6]: Faces zPos
   -- (12 edges)
   -- - [ 7]: Edge xNeg-yNeg
   -- - [ 8]: Edge xNeg-zNeg
   -- - [ 9]: Edge xNeg-yPos
   -- - [10]: Edge xNeg-zPos
   -- - [11]: Edge xPos-yNeg
   -- - [12]: Edge xPos-zNeg
   -- - [13]: Edge xPos-yPos
   -- - [14]: Edge xPos-zPos
   -- - [15]: Edge yNeg-zNeg
   -- - [16]: Edge yNeg-zPos
   -- - [17]: Edge yPos-zNeg
   -- - [18]: Edge yPos-zPos
   -- (8 corners)
   -- - [19]: Corner xNeg-yNeg-zNeg
   -- - [20]: Corner xNeg-yPos-zNeg
   -- - [21]: Corner xNeg-yNeg-zPos
   -- - [22]: Corner xNeg-yPos-zPos
   -- - [23]: Corner xPos-yNeg-zNeg
   -- - [24]: Corner xPos-yPos-zNeg
   -- - [25]: Corner xPos-yNeg-zPos
   -- - [26]: Corner xPos-yPos-zPos
   --
   var Fluid_regions =
      [UTIL.mkPartitionIsInteriorOrGhost(int3d, Fluid_columns, "Fluid_regions")]
      (Fluid, int3d{Grid_xBnum,Grid_yBnum,Grid_zBnum})

   -- Interior points
   var p_Interior = static_cast(partition(disjoint, Fluid, tiles), Fluid_regions[0] & p_Fluid);
   [UTIL.emitPartitionNameAttach(rexpr p_Interior end, "p_Interior")];

   -- All ghost points
   var p_AllBCs = p_Fluid - p_Interior;
   [UTIL.emitPartitionNameAttach(rexpr p_AllBCs end, "p_AllBCs")];

   -- All ghost points for each side
   var AllxNeg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var AllxPos_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var AllyNeg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var AllyPos_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var AllzNeg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var AllzPos_coloring = regentlib.c.legion_multi_domain_point_coloring_create();

   -- xNeg
   [addRegionsToColor(rexpr Fluid_regions end, rexpr AllxNeg_coloring end, "xNeg",
                     {1,  7,  8,  9, 10, 19, 20, 21, 22})];

   -- xPos
   [addRegionsToColor(rexpr Fluid_regions end, rexpr AllxPos_coloring end, "xPos",
                     {2, 11, 12, 13, 14, 23, 24, 25, 26})];

   -- yNeg
   [addRegionsToColor(rexpr Fluid_regions end, rexpr AllyNeg_coloring end, "yNeg",
                     {3,  7, 11, 15, 16, 19, 21, 23, 25})];

   -- yPos
   [addRegionsToColor(rexpr Fluid_regions end, rexpr AllyPos_coloring end, "yPos",
                     {4,  9, 13, 17, 18, 20, 22, 24, 26})];

   -- zNeg
   [addRegionsToColor(rexpr Fluid_regions end, rexpr AllzNeg_coloring end, "zNeg",
                     {5,  8, 12, 15, 17, 19, 20, 23, 24})];

   -- zPos
   [addRegionsToColor(rexpr Fluid_regions end, rexpr AllzPos_coloring end, "zPos",
                     {6, 10, 14, 16, 18, 21, 22, 25, 26})];

   -- Create partitions
   var AllxNeg = partition(disjoint, Fluid, AllxNeg_coloring, ispace(int1d,2))
   var AllxPos = partition(disjoint, Fluid, AllxPos_coloring, ispace(int1d,2))
   var AllyNeg = partition(disjoint, Fluid, AllyNeg_coloring, ispace(int1d,2))
   var AllyPos = partition(disjoint, Fluid, AllyPos_coloring, ispace(int1d,2))
   var AllzNeg = partition(disjoint, Fluid, AllzNeg_coloring, ispace(int1d,2))
   var AllzPos = partition(disjoint, Fluid, AllzPos_coloring, ispace(int1d,2))

   var p_AllxNeg = cross_product(p_Fluid, AllxNeg)
   var p_AllxPos = cross_product(p_Fluid, AllxPos)
   var p_AllyNeg = cross_product(p_Fluid, AllyNeg)
   var p_AllyPos = cross_product(p_Fluid, AllyPos)
   var p_AllzNeg = cross_product(p_Fluid, AllzNeg)
   var p_AllzPos = cross_product(p_Fluid, AllzPos);

   -- Destroy colors
   regentlib.c.legion_multi_domain_point_coloring_destroy(AllxNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(AllxPos_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(AllyNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(AllyPos_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(AllzNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(AllzPos_coloring)

   -----------------------------------------------------------------------------------------------
   -- Boundary conditions regions
   -----------------------------------------------------------------------------------------------
   -- !!! We need to be very careful here !!!
   -- A corner between two outflow conditions requires the bc conditions to be aliased
   -- therefore define one color for each side
   var xNeg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var xPos_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var yNeg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var yPos_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var zNeg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   var zPos_coloring = regentlib.c.legion_multi_domain_point_coloring_create();

   -- The faces are for sure part of the boundary respective boundary partition
   [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[1].bounds end, rexpr int3d{ 1, 0, 0} end)];
   [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[2].bounds end, rexpr int3d{-1, 0, 0} end)];
   [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[3].bounds end, rexpr int3d{ 0, 1, 0} end)];
   [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[4].bounds end, rexpr int3d{ 0,-1, 0} end)];
   [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[5].bounds end, rexpr int3d{ 0, 0, 1} end)];
   [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[6].bounds end, rexpr int3d{ 0, 0,-1} end)];

   -- Here things become arbitrary
   -- We give the following priorities to BCS:
   -- - Walls
   -- - Dirichlet
   -- - NSCBC_Inflow
   -- - IncomingShock
   -- - NSCBC_Outflow
   -- and to the directions:
   -- - X
   -- - Y
   -- - Z

   var BC_xBCLeft  = config.BC.xBCLeft.type
   var BC_xBCRight = config.BC.xBCRight.type
   var BC_yBCLeft  = config.BC.yBCLeft.type
   var BC_yBCRight = config.BC.yBCRight.type
   var BC_zBCLeft  = config.BC.zBCLeft.type
   var BC_zBCRight = config.BC.zBCRight.type;

   ------------------------------------------------------
   -- Break ties with other boundary conditions for edges
   ------------------------------------------------------
   -- [ 7]: Edge xNeg-yNeg
   [EdgeTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                       rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                       rexpr Fluid_regions[7].bounds end)];

   -- [ 8]: Edge xNeg-zNeg
   [EdgeTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                       rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                       rexpr Fluid_regions[8].bounds end)];

   -- [ 9]: Edge xNeg-yPos
   [EdgeTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                       rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                       rexpr Fluid_regions[9].bounds end)];
 
   -- [10]: Edge xNeg-zPos
   [EdgeTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                       rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                       rexpr Fluid_regions[10].bounds end)];

   -- [11]: Edge xPos-yNeg
   [EdgeTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                       rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                       rexpr Fluid_regions[11].bounds end)];

   -- [12]: Edge xPos-zNeg
   [EdgeTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                       rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                       rexpr Fluid_regions[12].bounds end)];

   -- [13]: Edge xPos-yPos
   [EdgeTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                       rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                       rexpr Fluid_regions[13].bounds end)];

   -- [14]: Edge xPos-zPos
   [EdgeTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                       rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                       rexpr Fluid_regions[14].bounds end)];

   -- [15]: Edge yNeg-zNeg
   [EdgeTieBreakPolicy(rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                       rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                       rexpr Fluid_regions[15].bounds end)];

   -- [16]: Edge yNeg-zPos
   [EdgeTieBreakPolicy(rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                       rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                       rexpr Fluid_regions[16].bounds end)];

   -- [17]: Edge yPos-zNeg
   [EdgeTieBreakPolicy(rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                       rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                       rexpr Fluid_regions[17].bounds end)];

   -- [18]: Edge yPos-zPos
   [EdgeTieBreakPolicy(rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                       rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                       rexpr Fluid_regions[18].bounds end)];

   --------------------------------------------------------
   -- Break ties with other boundary conditions for corners
   --------------------------------------------------------

   -- [19]: Corner xNeg-yNeg-zNeg
   [CornerTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                         rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                         rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                         rexpr Fluid_regions[19].bounds end)];

   -- [20]: Corner xNeg-yPos-zNeg
   [CornerTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                         rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                         rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                         rexpr Fluid_regions[20].bounds end)];

   -- [21]: Corner xNeg-yNeg-zPos
   [CornerTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                         rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                         rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                         rexpr Fluid_regions[21].bounds end)];

   -- [22]: Corner xNeg-yPos-zPos
   [CornerTieBreakPolicy(rexpr BC_xBCLeft  end, rexpr xNeg_coloring end, "xNeg",
                         rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                         rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                         rexpr Fluid_regions[22].bounds end)];

   -- [23]: Corner xPos-yNeg-zNeg
   [CornerTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                         rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                         rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                         rexpr Fluid_regions[23].bounds end)];

   -- [24]: Corner xPos-yPos-zNeg
   [CornerTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                         rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                         rexpr BC_zBCLeft  end, rexpr zNeg_coloring end, "zNeg",
                         rexpr Fluid_regions[24].bounds end)];

   -- [25]: Corner xPos-yNeg-zPos
   [CornerTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                         rexpr BC_yBCLeft  end, rexpr yNeg_coloring end, "yNeg",
                         rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                         rexpr Fluid_regions[25].bounds end)];

   -- [26]: Corner xPos-yPos-zPos
   [CornerTieBreakPolicy(rexpr BC_xBCRight end, rexpr xPos_coloring end, "xPos",
                         rexpr BC_yBCRight end, rexpr yPos_coloring end, "yPos",
                         rexpr BC_zBCRight end, rexpr zPos_coloring end, "zPos",
                         rexpr Fluid_regions[26].bounds end)];

   -- Create partitions
   var xNegBC = partition(disjoint, Fluid, xNeg_coloring, ispace(int1d,2))
   var xPosBC = partition(disjoint, Fluid, xPos_coloring, ispace(int1d,2))
   var yNegBC = partition(disjoint, Fluid, yNeg_coloring, ispace(int1d,2))
   var yPosBC = partition(disjoint, Fluid, yPos_coloring, ispace(int1d,2))
   var zNegBC = partition(disjoint, Fluid, zNeg_coloring, ispace(int1d,2))
   var zPosBC = partition(disjoint, Fluid, zPos_coloring, ispace(int1d,2))

   var p_xNegBC = cross_product(p_Fluid, xNegBC)
   var p_xPosBC = cross_product(p_Fluid, xPosBC)
   var p_yNegBC = cross_product(p_Fluid, yNegBC)
   var p_yPosBC = cross_product(p_Fluid, yPosBC)
   var p_zNegBC = cross_product(p_Fluid, zNegBC)
   var p_zPosBC = cross_product(p_Fluid, zPosBC);

   -- Destroy colors
   regentlib.c.legion_multi_domain_point_coloring_destroy(xNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(xPos_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(yNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(yPos_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(zNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(zPos_coloring)

   -- Create relevant ispaces
   var xNeg_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, p_Fluid, xNegBC, p_xNegBC)
   var xPos_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, p_Fluid, xPosBC, p_xPosBC)
   var yNeg_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, p_Fluid, yNegBC, p_yNegBC)
   var yPos_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, p_Fluid, yPosBC, p_yPosBC)
   var zNeg_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, p_Fluid, zNegBC, p_zNegBC)
   var zPos_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, p_Fluid, zPosBC, p_zPosBC)

   -----------------------------------------------------------------------------------------------
   -- END - Boundary conditions regions
   -----------------------------------------------------------------------------------------------
   -----------------------------------------------------------------------------------------------
   -- Regions for RHS functions
   -----------------------------------------------------------------------------------------------

   -- Cells where the divergence operator in x direction is applied
   var xdivg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   -- Cells where the divergence operator in y direction is applied
   var ydivg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   -- Cells where the divergence operator in z direction is applied
   var zdivg_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   -- Cells where the rhs of the equations has to be computed
   var solve_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   -- Cells where the reconstruction operator in x direction is applied
   var xfaces_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   -- Cells where the reconstruction operator in y direction is applied
   var yfaces_coloring = regentlib.c.legion_multi_domain_point_coloring_create()
   -- Cells where the reconstruction operator in z direction is applied
   var zfaces_coloring = regentlib.c.legion_multi_domain_point_coloring_create()

   -- For sure they contain the internal cells
   regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring, int1d(0),  Fluid_regions[0].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring, int1d(0),  Fluid_regions[0].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring, int1d(0),  Fluid_regions[0].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring, int1d(0),  Fluid_regions[0].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[0].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[0].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[0].bounds)

   regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[1].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[3].bounds)
   regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[5].bounds)


   -- Add boundary cells in case of NSCBC conditions
   var is_xBCLeft_NSCBC = ((BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) or
                           (BC_xBCLeft == SCHEMA.FlowBC_RecycleRescaling))

   var is_xBCRight_NSCBC = (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow)

   var is_yBCLeft_NSCBC = (BC_yBCLeft  == SCHEMA.FlowBC_NSCBC_Outflow)

   var is_yBCRight_NSCBC = ((BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) or
                            (BC_yBCRight == SCHEMA.FlowBC_IncomingShock))

   if is_xBCLeft_NSCBC then
      -- xNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[1].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[8].bounds)
   end
   if is_xBCRight_NSCBC then
      -- xPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 2].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[12].bounds)
   end

   if is_yBCLeft_NSCBC then
      -- yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 3].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[15].bounds)
   end
   if is_yBCRight_NSCBC then
      -- yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 4].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[17].bounds)
   end

   if (is_xBCLeft_NSCBC and is_yBCLeft_NSCBC) then
      -- Edge xNeg-yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 7].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[19].bounds)
   end
   if (is_xBCRight_NSCBC and is_yBCLeft_NSCBC) then
      -- Edge xPos-yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[11].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[23].bounds)
   end
   if (is_xBCLeft_NSCBC and is_yBCRight_NSCBC) then
      -- Edge xNeg-yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 9].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[20].bounds)
   end
   if (is_xBCRight_NSCBC and is_yBCRight_NSCBC) then
      -- Edge xPos-yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[13].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[13].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[13].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[24].bounds)
   end

   var xdivg_cells  = partition(disjoint, Fluid, xdivg_coloring,  ispace(int1d,1))
   var ydivg_cells  = partition(disjoint, Fluid, ydivg_coloring,  ispace(int1d,1))
   var zdivg_cells  = partition(disjoint, Fluid, zdivg_coloring,  ispace(int1d,1))
   var solve_cells  = partition(disjoint, Fluid, solve_coloring,  ispace(int1d,1))
   var xfaces_cells = partition(disjoint, Fluid, xfaces_coloring, ispace(int1d,1))
   var yfaces_cells = partition(disjoint, Fluid, yfaces_coloring, ispace(int1d,1))
   var zfaces_cells = partition(disjoint, Fluid, zfaces_coloring, ispace(int1d,1))

   var p_x_divg = p_Fluid & (xdivg_cells[0] & p_Fluid)
   var p_y_divg = p_Fluid & (ydivg_cells[0] & p_Fluid)
   var p_z_divg = p_Fluid & (zdivg_cells[0] & p_Fluid)
   var p_solved = p_Fluid & (solve_cells[0] & p_Fluid);
   [UTIL.emitPartitionNameAttach(rexpr p_x_divg end, "p_x_divg")];
   [UTIL.emitPartitionNameAttach(rexpr p_y_divg end, "p_y_divg")];
   [UTIL.emitPartitionNameAttach(rexpr p_z_divg end, "p_z_divg")];
   [UTIL.emitPartitionNameAttach(rexpr p_solved end, "p_solved")];

   var p_x_faces = p_Fluid & (xfaces_cells[0] & p_Fluid)
   var p_y_faces = p_Fluid & (yfaces_cells[0] & p_Fluid)
   var p_z_faces = p_Fluid & (zfaces_cells[0] & p_Fluid);
   [UTIL.emitPartitionNameAttach(rexpr p_x_faces end, "p_x_faces")];
   [UTIL.emitPartitionNameAttach(rexpr p_y_faces end, "p_y_faces")];
   [UTIL.emitPartitionNameAttach(rexpr p_z_faces end, "p_z_faces")];

   -- Destroy colors
   regentlib.c.legion_multi_domain_point_coloring_destroy( xdivg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy( ydivg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy( zdivg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy( solve_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(xfaces_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(yfaces_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(zfaces_coloring)

   -----------------------------------------------------------------------------------------------
   -- END - Regions for RHS functions
   -----------------------------------------------------------------------------------------------

   return [zones_partitions(Fluid, tiles)]{
      -- Partitions
      p_All      = p_Fluid,
      p_Interior = p_Interior,
      p_AllBCs   = p_AllBCs,
      -- Partitions for reconstruction operator
      p_x_faces = p_x_faces,
      p_y_faces = p_y_faces,
      p_z_faces = p_z_faces,
      -- Partitions for divergence operator
      p_x_divg = p_x_divg,
      p_y_divg = p_y_divg,
      p_z_divg = p_z_divg,
      p_solved = p_solved,
      -- Partitions containing all ghost for each side
      AllxNeg   = AllxNeg,
      AllxPos   = AllxPos,
      AllyNeg   = AllyNeg,
      AllyPos   = AllyPos,
      AllzNeg   = AllzNeg,
      AllzPos   = AllzPos,
      p_AllxNeg = p_AllxNeg,
      p_AllxPos = p_AllxPos,
      p_AllyNeg = p_AllyNeg,
      p_AllyPos = p_AllyPos,
      p_AllzNeg = p_AllzNeg,
      p_AllzPos = p_AllzPos,
      -- BC partitions
      xNeg   = xNegBC,
      xPos   = xPosBC,
      yNeg   = yNegBC,
      yPos   = yPosBC,
      zNeg   = zNegBC,
      zPos   = zPosBC,
      p_xNeg = p_xNegBC,
      p_xPos = p_xPosBC,
      p_yNeg = p_yNegBC,
      p_yPos = p_yPosBC,
      p_zNeg = p_zNegBC,
      p_zPos = p_zPosBC,
      -- BC partitions ispaces
      xNeg_ispace = xNeg_ispace,
      xPos_ispace = xPos_ispace,
      yNeg_ispace = yNeg_ispace,
      yPos_ispace = yPos_ispace,
      zNeg_ispace = zNeg_ispace,
      zPos_ispace = zPos_ispace
   }
end

-------------------------------------------------------------------------------
-- IO PARTITIONING ROUTINES
-------------------------------------------------------------------------------

__demand(__inline)
task Exports.PartitionOutput(Fluid : region(ispace(int3d), Fluid_columns),
                             tiles : ispace(int3d),
                             config : Config,
                             Grid_xBnum : int32, Grid_yBnum : int32, Grid_zBnum : int32)
where
   reads(Fluid)
do
   var p_Output = [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_Output")]
      (Fluid, tiles, int3d{Grid_xBnum,Grid_yBnum,Grid_zBnum}, int3d{0,0,0})

   -- Partitions for volume probes
   var probe_coloring = regentlib.c.legion_domain_point_coloring_create()
   for p=0, config.IO.volumeProbes.length do
      -- Clip rectangles from the input
      var vol = config.IO.volumeProbes.values[p].volume
      vol.fromCell[0] max= Fluid.bounds.lo.x
      vol.fromCell[1] max= Fluid.bounds.lo.y
      vol.fromCell[2] max= Fluid.bounds.lo.z
      vol.uptoCell[0] min= Fluid.bounds.hi.x
      vol.uptoCell[1] min= Fluid.bounds.hi.y
      vol.uptoCell[2] min= Fluid.bounds.hi.z
      -- add to the coloring
      var rect = rect3d{
         lo = int3d{vol.fromCell[0], vol.fromCell[1], vol.fromCell[2]},
         hi = int3d{vol.uptoCell[0], vol.uptoCell[1], vol.uptoCell[2]}}
      regentlib.c.legion_domain_point_coloring_color_domain(probe_coloring, int1d(p), rect)
   end
   -- Add one point to avoid errors
   if config.IO.volumeProbes.length == 0 then
		regentlib.c.legion_domain_point_coloring_color_domain(probe_coloring, int1d(0), rect3d{lo = int3d{0,0,0}, hi = int3d{0,0,0}})
	end
   -- Make partitions of Fluid
   var Vprobes = partition(aliased, Fluid, probe_coloring, ispace(int1d, max(config.IO.volumeProbes.length, 1)))
   -- Split over tiles
   var p_Vprobes = cross_product(Vprobes, p_Output)
   -- Attach names for mapping
   for p=0, config.IO.volumeProbes.length do
      [UTIL.emitPartitionNameAttach(rexpr p_Vprobes[p] end, "p_Vprobes")];
   end
   -- Destroy color
   regentlib.c.legion_domain_point_coloring_destroy(probe_coloring)

   return [output_partitions(Fluid, tiles)]{
      -- Restart
      p_Output = p_Output,
      -- Volume probes
      Vprobes       = Vprobes,
      p_Vprobes     = p_Vprobes,
   }
end

-------------------------------------------------------------------------------
-- GHOST PARTITIONING ROUTINES
-------------------------------------------------------------------------------

local function emitFill(p, t, f, val) return rquote
   var v = [val]
   __demand(__index_launch)
   for c in t do fill(([p][c]).[f], v) end
end end

local __demand(__inline)
task InitializeIndices(r : region(ispace(int3d), indices_columns),
                       tiles : ispace(int3d),
                       p_All : partition(disjoint, r, tiles))
where
   writes(r)
do
   [emitFill(p_All, tiles, "cm2_x", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cm1_x", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp1_x", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp2_x", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp3_x", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cm2_y", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cm1_y", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp1_y", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp2_y", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp3_y", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cm2_z", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cm1_z", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp1_z", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp2_z", rexpr int3d({0, 0, 0}) end)];
   [emitFill(p_All, tiles, "cp3_z", rexpr int3d({0, 0, 0}) end)];
end

--local __demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA (TODO: workaround for Legion issue #812)
task ComputeIndices(Fluid : region(ispace(int3d), Fluid_columns),
                    Aux   : region(ispace(int3d), indices_columns),
                    Fluid_bounds : rect3d)
where
   reads(Fluid.{nType_x, nType_y, nType_z}),
   writes(Aux.{cm2_x, cm1_x, cp1_x, cp2_x, cp3_x}),
   writes(Aux.{cm2_y, cm1_y, cp1_y, cp2_y, cp3_y}),
   writes(Aux.{cm2_z, cm1_z, cp1_z, cp2_z, cp3_z})
do
   __demand(__openmp)
   for c in Fluid do
      -- X direction
      Aux[c].cm2_x = [METRIC.GetCm2("x", rexpr c end, rexpr Fluid[c].nType_x end, rexpr Fluid_bounds end)];
      Aux[c].cm1_x = [METRIC.GetCm1("x", rexpr c end, rexpr Fluid[c].nType_x end, rexpr Fluid_bounds end)];
      Aux[c].cp1_x = [METRIC.GetCp1("x", rexpr c end, rexpr Fluid[c].nType_x end, rexpr Fluid_bounds end)];
      Aux[c].cp2_x = [METRIC.GetCp2("x", rexpr c end, rexpr Fluid[c].nType_x end, rexpr Fluid_bounds end)];
      Aux[c].cp3_x = [METRIC.GetCp3("x", rexpr c end, rexpr Fluid[c].nType_x end, rexpr Fluid_bounds end)];

      -- Y direction
      Aux[c].cm2_y = [METRIC.GetCm2("y", rexpr c end, rexpr Fluid[c].nType_y end, rexpr Fluid_bounds end)];
      Aux[c].cm1_y = [METRIC.GetCm1("y", rexpr c end, rexpr Fluid[c].nType_y end, rexpr Fluid_bounds end)];
      Aux[c].cp1_y = [METRIC.GetCp1("y", rexpr c end, rexpr Fluid[c].nType_y end, rexpr Fluid_bounds end)];
      Aux[c].cp2_y = [METRIC.GetCp2("y", rexpr c end, rexpr Fluid[c].nType_y end, rexpr Fluid_bounds end)];
      Aux[c].cp3_y = [METRIC.GetCp3("y", rexpr c end, rexpr Fluid[c].nType_y end, rexpr Fluid_bounds end)];

      -- Z direction
      Aux[c].cm2_z = [METRIC.GetCm2("z", rexpr c end, rexpr Fluid[c].nType_z end, rexpr Fluid_bounds end)];
      Aux[c].cm1_z = [METRIC.GetCm1("z", rexpr c end, rexpr Fluid[c].nType_z end, rexpr Fluid_bounds end)];
      Aux[c].cp1_z = [METRIC.GetCp1("z", rexpr c end, rexpr Fluid[c].nType_z end, rexpr Fluid_bounds end)];
      Aux[c].cp2_z = [METRIC.GetCp2("z", rexpr c end, rexpr Fluid[c].nType_z end, rexpr Fluid_bounds end)];
      Aux[c].cp3_z = [METRIC.GetCp3("z", rexpr c end, rexpr Fluid[c].nType_z end, rexpr Fluid_bounds end)];
   end
end

local function emitEulerGhostRegion(sdir, t, r, p_r)
   local cm2_d
   local cm1_d
   local cp1_d
   local cp2_d
   local cp3_d
   if sdir == "x" then
      cm2_d = "cm2_x"
      cm1_d = "cm1_x"
      cp1_d = "cp1_x"
      cp2_d = "cp2_x"
      cp3_d = "cp3_x"
   elseif sdir == "y" then
      cm2_d = "cm2_y"
      cm1_d = "cm1_y"
      cp1_d = "cp1_y"
      cp2_d = "cp2_y"
      cp3_d = "cp3_y"
   elseif sdir == "z" then
      cm2_d = "cm2_z"
      cm1_d = "cm1_z"
      cp1_d = "cp1_z"
      cp2_d = "cp2_z"
      cp3_d = "cp3_z"
   end
   return rexpr
      image([t], [p_r], [r].[cm2_d]) |
      image([t], [p_r], [r].[cm1_d]) |
      image([t], [p_r], [r].[cp1_d]) |
      image([t], [p_r], [r].[cp2_d]) |
      image([t], [p_r], [r].[cp3_d])
   end
end

local function emitGhostRegion(sdir, off, t, r, p_r)
   local coff_d = "c"
   if (off == -2) then
      coff_d = coff_d .. "m2"
   elseif (off == -1) then
      coff_d = coff_d .. "m1"
   elseif (off == 1) then
      coff_d = coff_d .. "p1"
   elseif (off == 2) then
      coff_d = coff_d .. "p2"
   elseif (off == 3) then
      coff_d = coff_d .. "p3"
   else
      assert(0)
   end
   coff_d = coff_d .. "_" .. sdir
   return rexpr
      image([t], [p_r], [r].[coff_d])
   end
end

__demand(__inline)
task Exports.PartitionGhost(Fluid : region(ispace(int3d), Fluid_columns),
                  tiles : ispace(int3d),
                  Fluid_Zones : zones_partitions(Fluid, tiles))
where
   reads(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All,
        p_x_faces, p_y_faces, p_z_faces,
        p_x_divg,  p_y_divg,  p_z_divg} = Fluid_Zones

   -- Define an auxiliary region that will store the stencil indices
   var aux = region(Fluid.ispace, indices_columns)
   var All     = aux & Fluid_Zones.p_All
   var x_faces = aux & Fluid_Zones.p_x_faces
   var y_faces = aux & Fluid_Zones.p_y_faces
   var z_faces = aux & Fluid_Zones.p_z_faces
   var x_divg  = aux & Fluid_Zones.p_x_divg
   var y_divg  = aux & Fluid_Zones.p_y_divg
   var z_divg  = aux & Fluid_Zones.p_z_divg

   -- Compute stencil indices
   InitializeIndices(aux, tiles, All)
   __demand(__index_launch)
   for c in tiles do
      ComputeIndices(p_All[c], All[c], Fluid.bounds)
   end

   -- Compute auxiliary partitions
   -- Fluxes are computed at [0:-1] direction by direction
   var x_flux = x_divg | image(aux, x_divg, aux.cm1_x)
   var y_flux = y_divg | image(aux, y_divg, aux.cm1_y)
   var z_flux = z_divg | image(aux, z_divg, aux.cm1_z)

   var x_fluxM2 = [emitGhostRegion("x", -2, Fluid, aux, x_flux)];
   var x_fluxM1 = [emitGhostRegion("x", -1, Fluid, aux, x_flux)];
   var x_fluxP1 = [emitGhostRegion("x",  1, Fluid, aux, x_flux)];
   var x_fluxP2 = [emitGhostRegion("x",  2, Fluid, aux, x_flux)];
   var x_fluxP3 = [emitGhostRegion("x",  3, Fluid, aux, x_flux)];

   var y_fluxM2 = [emitGhostRegion("y", -2, Fluid, aux, y_flux)];
   var y_fluxM1 = [emitGhostRegion("y", -1, Fluid, aux, y_flux)];
   var y_fluxP1 = [emitGhostRegion("y",  1, Fluid, aux, y_flux)];
   var y_fluxP2 = [emitGhostRegion("y",  2, Fluid, aux, y_flux)];
   var y_fluxP3 = [emitGhostRegion("y",  3, Fluid, aux, y_flux)];

   var z_fluxM2 = [emitGhostRegion("z", -2, Fluid, aux, z_flux)];
   var z_fluxM1 = [emitGhostRegion("z", -1, Fluid, aux, z_flux)];
   var z_fluxP1 = [emitGhostRegion("z",  1, Fluid, aux, z_flux)];
   var z_fluxP2 = [emitGhostRegion("z",  2, Fluid, aux, z_flux)];
   var z_fluxP3 = [emitGhostRegion("z",  3, Fluid, aux, z_flux)];

   var AllM1x = [emitGhostRegion("x", -1, aux, aux, All)];
   var AllP1x = [emitGhostRegion("x",  1, aux, aux, All)];

   var AllM1y = [emitGhostRegion("y", -1, aux, aux, All)];
   var AllP1y = [emitGhostRegion("y",  1, aux, aux, All)];

   var AllM1z = [emitGhostRegion("z", -1, aux, aux, All)];
   var AllP1z = [emitGhostRegion("z",  1, aux, aux, All)];

   var x_facesM2 = [emitGhostRegion("x", -2, Fluid, aux, x_faces)];
   var x_facesM1 = [emitGhostRegion("x", -1, Fluid, aux, x_faces)];
   var x_facesP1 = [emitGhostRegion("x",  1, Fluid, aux, x_faces)];
   var x_facesP2 = [emitGhostRegion("x",  2, Fluid, aux, x_faces)];
   var x_facesP3 = [emitGhostRegion("x",  3, Fluid, aux, x_faces)];

   var y_facesM2 = [emitGhostRegion("y", -2, Fluid, aux, y_faces)];
   var y_facesM1 = [emitGhostRegion("y", -1, Fluid, aux, y_faces)];
   var y_facesP1 = [emitGhostRegion("y",  1, Fluid, aux, y_faces)];
   var y_facesP2 = [emitGhostRegion("y",  2, Fluid, aux, y_faces)];
   var y_facesP3 = [emitGhostRegion("y",  3, Fluid, aux, y_faces)];

   var z_facesM2 = [emitGhostRegion("z", -2, Fluid, aux, z_faces)];
   var z_facesM1 = [emitGhostRegion("z", -1, Fluid, aux, z_faces)];
   var z_facesP1 = [emitGhostRegion("z",  1, Fluid, aux, z_faces)];
   var z_facesP2 = [emitGhostRegion("z",  2, Fluid, aux, z_faces)];
   var z_facesP3 = [emitGhostRegion("z",  3, Fluid, aux, z_faces)];

   -- Fluxes stencil accesses [-1:0] direction by direction
   var p_XFluxGhosts = Fluid & x_flux
   var p_YFluxGhosts = Fluid & y_flux
   var p_ZFluxGhosts = Fluid & z_flux;
   [UTIL.emitPartitionNameAttach(rexpr p_XFluxGhosts end, "p_XFluxGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YFluxGhosts end, "p_YFluxGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZFluxGhosts end, "p_ZFluxGhosts")];

   -- Diffusion fluxes stencil accesses [0:+1] direction by direction
   var p_XDiffGhosts = p_XFluxGhosts | x_fluxP1
   var p_YDiffGhosts = p_YFluxGhosts | y_fluxP1
   var p_ZDiffGhosts = p_ZFluxGhosts | z_fluxP1;
   [UTIL.emitPartitionNameAttach(rexpr p_XDiffGhosts end, "p_XDiffGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YDiffGhosts end, "p_YDiffGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZDiffGhosts end, "p_ZDiffGhosts")];

   -- Euler fluxes stencil accesses [-2:+3] direction by direction
   var p_XEulerGhosts2 = p_XFluxGhosts | x_fluxM2 | x_fluxM1 | x_fluxP1 | x_fluxP2 | x_fluxP3
   var p_YEulerGhosts2 = p_YFluxGhosts | y_fluxM2 | y_fluxM1 | y_fluxP1 | y_fluxP2 | y_fluxP3
   var p_ZEulerGhosts2 = p_ZFluxGhosts | z_fluxM2 | z_fluxM1 | z_fluxP1 | z_fluxP2 | z_fluxP3;
   [UTIL.emitPartitionNameAttach(rexpr p_XEulerGhosts2 end, "p_XEulerGhosts2")];
   [UTIL.emitPartitionNameAttach(rexpr p_YEulerGhosts2 end, "p_YEulerGhosts2")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZEulerGhosts2 end, "p_ZEulerGhosts2")];

   -- Shock sensors stencil accesses [-1:+1] direction by direction
   var p_XSensorGhosts2 = p_XFluxGhosts | x_fluxM1 | x_fluxP1
   var p_YSensorGhosts2 = p_YFluxGhosts | y_fluxM1 | y_fluxP1
   var p_ZSensorGhosts2 = p_ZFluxGhosts | z_fluxM1 | z_fluxP1;
   [UTIL.emitPartitionNameAttach(rexpr p_XSensorGhosts2 end, "p_XSensorGhosts2")];
   [UTIL.emitPartitionNameAttach(rexpr p_YSensorGhosts2 end, "p_YSensorGhosts2")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZSensorGhosts2 end, "p_ZSensorGhosts2")];

   -- Gradient tasks use [-1:+1] in all three directions
   var p_GradientGhosts = p_All | (Fluid & (AllM1x | AllP1x | AllM1y | AllP1y | AllM1z | AllP1z));
   [UTIL.emitPartitionNameAttach(rexpr p_GradientGhosts end, "p_GradientGhosts")];

   -- Metric routines (uses the entire stencil in all three directions)
   var MetricGhostsX = All | AllM1x
   var MetricGhostsY = All | AllM1y
   var MetricGhostsZ = All | AllM1z
   var p_MetricGhosts = p_All |
         [emitEulerGhostRegion("x", Fluid, aux, MetricGhostsX)] |
         [emitEulerGhostRegion("y", Fluid, aux, MetricGhostsY)] |
         [emitEulerGhostRegion("z", Fluid, aux, MetricGhostsZ)];
   [UTIL.emitPartitionNameAttach(rexpr p_MetricGhosts end, "p_MetricGhosts")];

   -- Euler fluxes routines (uses [-2:+3] direction by direction)
   var p_XEulerGhosts = p_x_faces | x_facesM2 | x_facesM1 | x_facesP1 | x_facesP2 | x_facesP3
   var p_YEulerGhosts = p_y_faces | y_facesM2 | y_facesM1 | y_facesP1 | y_facesP2 | y_facesP3
   var p_ZEulerGhosts = p_z_faces | z_facesM2 | z_facesM1 | z_facesP1 | z_facesP2 | z_facesP3;
   [UTIL.emitPartitionNameAttach(rexpr p_XEulerGhosts end, "p_XEulerGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YEulerGhosts end, "p_YEulerGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZEulerGhosts end, "p_ZEulerGhosts")];

   -- Shock sensors routines (uses [-1:+1] direction by direction)
   var p_XSensorGhosts = p_x_faces | x_facesM1 | x_facesP1
   var p_YSensorGhosts = p_y_faces | y_facesM1 | y_facesP1
   var p_ZSensorGhosts = p_z_faces | z_facesM1 | z_facesP1;
   [UTIL.emitPartitionNameAttach(rexpr p_XSensorGhosts end, "p_XSensorGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YSensorGhosts end, "p_YSensorGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZSensorGhosts end, "p_ZSensorGhosts")];

   -- All With Ghosts
   var p_AllWithGhosts = p_All | p_GradientGhosts | p_MetricGhosts |
                         p_XFluxGhosts   | p_YFluxGhosts   | p_ZFluxGhosts   |
                         p_XDiffGhosts   | p_YDiffGhosts   | p_ZDiffGhosts   |
                         p_XEulerGhosts2 | p_YEulerGhosts2 | p_ZEulerGhosts2 |
                         p_XSensorGhosts2| p_YSensorGhosts2| p_ZSensorGhosts2|
                         p_XEulerGhosts  | p_YEulerGhosts  | p_ZEulerGhosts  |
                         p_XSensorGhosts | p_YSensorGhosts | p_ZSensorGhosts;
   [UTIL.emitPartitionNameAttach(rexpr p_AllWithGhosts end, "p_AllWithGhosts")];

   -- Delete aux (avoid deletion until Legion issue #812 is fixed)
   --__delete(aux)

   return [ghost_partitions(Fluid, tiles)]{
      -- All With Ghosts
      p_AllWithGhosts = p_AllWithGhosts,
      -- Metric routines
      p_MetricGhosts  = p_MetricGhosts,
      -- Fluxes stencil access
      p_XFluxGhosts = p_XFluxGhosts,
      p_YFluxGhosts = p_YFluxGhosts,
      p_ZFluxGhosts = p_ZFluxGhosts,
      -- Diffusion fluxes stencil access
      p_XDiffGhosts = p_XDiffGhosts,
      p_YDiffGhosts = p_YDiffGhosts,
      p_ZDiffGhosts = p_ZDiffGhosts,
      -- Euler fluxes stencil access
      p_XEulerGhosts2 = p_XEulerGhosts2,
      p_YEulerGhosts2 = p_YEulerGhosts2,
      p_ZEulerGhosts2 = p_ZEulerGhosts2,
      -- Shock sensors stencil access
      p_XSensorGhosts2 = p_XSensorGhosts2,
      p_YSensorGhosts2 = p_YSensorGhosts2,
      p_ZSensorGhosts2 = p_ZSensorGhosts2,
      -- Euler fluxes routines
      p_XEulerGhosts = p_XEulerGhosts,
      p_YEulerGhosts = p_YEulerGhosts,
      p_ZEulerGhosts = p_ZEulerGhosts,
      -- Gradient routines
      p_GradientGhosts =  p_GradientGhosts,
      -- Shock sensor ghosts
      p_XSensorGhosts = p_XSensorGhosts,
      p_YSensorGhosts = p_YSensorGhosts,
      p_ZSensorGhosts = p_ZSensorGhosts
   }
end

__demand(__inline)
task Exports.PartitionAverageGhost(
                  Fluid : region(ispace(int3d), Fluid_columns),
                  p_All : partition(disjoint, Fluid, ispace(int3d)),
                  p_Avg : partition(aliased, Fluid, ispace(int1d)),
                  cr_Avg : cross_product(p_Avg, p_All),
                  n_Avg : int)
where
   reads(Fluid)
do
   var tiles = p_All.colors
   -- This line matches the maximum number of average specified in config_schema.lua:341-347
   var p : average_ghost_partitions(Fluid, tiles)[10]

   if (n_Avg > 0) then
      -- Define an auxiliary region that will store the stencil indices
      var aux = region(Fluid.ispace, indices_columns)
      var All     = aux & p_All

      -- Compute stencil indices
      InitializeIndices(aux, tiles, All)
      __demand(__index_launch)
      for c in tiles do
         ComputeIndices(p_All[c], All[c], Fluid.bounds)
      end

      -- Define the Ghost partition for each partition of the cross product
      for i=0, n_Avg do
         var Avg = aux & cr_Avg[i]

         -- Compute auxiliary partitions
         var AvgM1x = [emitGhostRegion("x", -1, aux, aux, Avg)];
         var AvgP1x = [emitGhostRegion("x",  1, aux, aux, Avg)];

         var AvgM1y = [emitGhostRegion("y", -1, aux, aux, Avg)];
         var AvgP1y = [emitGhostRegion("y",  1, aux, aux, Avg)];

         var AvgM1z = [emitGhostRegion("z", -1, aux, aux, Avg)];
         var AvgP1z = [emitGhostRegion("z",  1, aux, aux, Avg)];

         -- Gradient tasks use [-1:+1] in all three directions
         var p_GradientGhosts = Fluid & (Avg | AvgM1x | AvgP1x |
                                               AvgM1y | AvgP1y |
                                               AvgM1z | AvgP1z);
         [UTIL.emitPartitionNameAttach(rexpr p_GradientGhosts end, "p_GradientGhosts")];

         -- Store in the array of fspaces
         p[i] = [average_ghost_partitions(Fluid, tiles)] {
            -- Gradient routines
            p_GradientGhosts =  p_GradientGhosts,
         }
      end

      -- Delete aux (avoid deletion until Legion issue #812 is fixed)
      --__delete(aux)
   end
   return p
end

return Exports end

