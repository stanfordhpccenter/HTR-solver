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

return function(SCHEMA, Fluid_columns) local Exports = {}

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require "util"
local Config = SCHEMA.Config

-------------------------------------------------------------------------------
-- PARTITIONS FSPACES
-------------------------------------------------------------------------------

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
   p_xNeg     : cross_product(xNeg, p_All),
   p_xPos     : cross_product(xPos, p_All),
   p_yNeg     : cross_product(yNeg, p_All),
   p_yPos     : cross_product(yPos, p_All),
   p_zNeg     : cross_product(zNeg, p_All),
   p_zPos     : cross_product(zPos, p_All),
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
   -- Diffusion fluxes traverse gradients stencil access
   p_XDiffGradGhosts : partition(aliased, Fluid, tiles),
   p_YDiffGradGhosts : partition(aliased, Fluid, tiles),
   p_ZDiffGradGhosts : partition(aliased, Fluid, tiles),
   -- Euler fluxes routines
   p_XEulerGhosts : partition(aliased, Fluid, tiles),
   p_YEulerGhosts : partition(aliased, Fluid, tiles),
   p_ZEulerGhosts : partition(aliased, Fluid, tiles),
   -- Shock sensors stencil access
   p_XSensorGhosts2 : partition(aliased, Fluid, tiles),
   p_YSensorGhosts2 : partition(aliased, Fluid, tiles),
   p_ZSensorGhosts2 : partition(aliased, Fluid, tiles),
   -- Shock sensor ghosts
   p_XSensorGhosts : partition(aliased, Fluid, tiles),
   p_YSensorGhosts : partition(aliased, Fluid, tiles),
   p_ZSensorGhosts : partition(aliased, Fluid, tiles),
   -- Metric routines
   p_MetricGhosts  : partition(aliased, Fluid, tiles),
   -- Gradient routines
   p_GradientGhosts : partition(aliased, Fluid, tiles),
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

local function isNSCBC_FarField(Type)
   return rexpr
      (Type == SCHEMA.FlowBC_NSCBC_FarField)
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
-- - NSCBC_FarField
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

      var is_NSCBC_FarField1 = [isNSCBC_FarField(Type1)];
      var is_NSCBC_FarField2 = [isNSCBC_FarField(Type2)];

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
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1 or is_NSCBC_FarField1) and
              (is_NSCBC_Outflow2 or is_IncomingShock2 or is_NSCBC_FarField2)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif (is_NSCBC_Outflow1 or is_IncomingShock1 or is_NSCBC_FarField1) then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif (is_NSCBC_Outflow2 or is_IncomingShock2 or is_NSCBC_FarField2) then
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

      var is_NSCBC_FarField1 = [isNSCBC_FarField(Type1)];
      var is_NSCBC_FarField2 = [isNSCBC_FarField(Type2)];
      var is_NSCBC_FarField3 = [isNSCBC_FarField(Type3)];

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
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1 or is_NSCBC_FarField1) and
              (is_NSCBC_Outflow2 or is_IncomingShock2 or is_NSCBC_FarField2) and
              (is_NSCBC_Outflow3 or is_IncomingShock3 or is_NSCBC_FarField3)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring2, rect, stencil2)];
         [addToBcColoring(coloring3, rect, stencil3)]
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1 or is_NSCBC_FarField1) and
              (is_NSCBC_Outflow2 or is_IncomingShock2 or is_NSCBC_FarField2)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif ((is_NSCBC_Outflow1 or is_IncomingShock1 or is_NSCBC_FarField1) and
              (is_NSCBC_Outflow3 or is_IncomingShock3 or is_NSCBC_FarField3)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring1, rect, stencil1)];
         [addToBcColoring(coloring3, rect, stencil3)]
      elseif ((is_NSCBC_Outflow2 or is_IncomingShock2 or is_NSCBC_FarField2) and
              (is_NSCBC_Outflow3 or is_IncomingShock3 or is_NSCBC_FarField3)) then
         -- This edge belongs to both bcs
         [addToBcColoring(coloring2, rect, stencil2)];
         [addToBcColoring(coloring3, rect, stencil3)]
      elseif (is_NSCBC_Outflow1 or is_IncomingShock1 or is_NSCBC_FarField1) then
         [addToBcColoring(coloring1, rect, stencil1)]
      elseif (is_NSCBC_Outflow2 or is_IncomingShock2 or is_NSCBC_FarField2) then
         [addToBcColoring(coloring2, rect, stencil2)]
      elseif (is_NSCBC_Outflow3 or is_IncomingShock3 or is_NSCBC_FarField3) then
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

local function isNSCBC(BC)
   return rexpr
      ((BC == SCHEMA.FlowBC_NSCBC_Inflow    ) or
       (BC == SCHEMA.FlowBC_NSCBC_Outflow   ) or
       (BC == SCHEMA.FlowBC_NSCBC_FarField  ) or
       (BC == SCHEMA.FlowBC_RecycleRescaling) or
       (BC == SCHEMA.FlowBC_IncomingShock   ))
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

   var p_xNegBC = cross_product(xNegBC, p_Fluid)
   var p_xPosBC = cross_product(xPosBC, p_Fluid)
   var p_yNegBC = cross_product(yNegBC, p_Fluid)
   var p_yPosBC = cross_product(yPosBC, p_Fluid)
   var p_zNegBC = cross_product(zNegBC, p_Fluid)
   var p_zPosBC = cross_product(zPosBC, p_Fluid);

   [UTIL.emitPartitionNameAttach(rexpr p_xNegBC[0] end, "p_xNeg0")];
   [UTIL.emitPartitionNameAttach(rexpr p_xPosBC[0] end, "p_xPos0")];
   [UTIL.emitPartitionNameAttach(rexpr p_yNegBC[0] end, "p_yNeg0")];
   [UTIL.emitPartitionNameAttach(rexpr p_yPosBC[0] end, "p_yPos0")];
   [UTIL.emitPartitionNameAttach(rexpr p_zNegBC[0] end, "p_zNeg0")];
   [UTIL.emitPartitionNameAttach(rexpr p_zPosBC[0] end, "p_zPos0")];

   [UTIL.emitPartitionNameAttach(rexpr p_xNegBC[1] end, "p_xNeg1")];
   [UTIL.emitPartitionNameAttach(rexpr p_xPosBC[1] end, "p_xPos1")];
   [UTIL.emitPartitionNameAttach(rexpr p_yNegBC[1] end, "p_yNeg1")];
   [UTIL.emitPartitionNameAttach(rexpr p_yPosBC[1] end, "p_yPos1")];
   [UTIL.emitPartitionNameAttach(rexpr p_zNegBC[1] end, "p_zNeg1")];
   [UTIL.emitPartitionNameAttach(rexpr p_zPosBC[1] end, "p_zPos1")];

   -- Destroy colors
   regentlib.c.legion_multi_domain_point_coloring_destroy(xNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(xPos_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(yNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(yPos_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(zNeg_coloring)
   regentlib.c.legion_multi_domain_point_coloring_destroy(zPos_coloring)

   -- Create relevant ispaces
   var xNeg_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, xNegBC, p_Fluid, p_xNegBC)
   var xPos_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, xPosBC, p_Fluid, p_xPosBC)
   var yNeg_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, yNegBC, p_Fluid, p_yNegBC)
   var yPos_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, yPosBC, p_Fluid, p_yPosBC)
   var zNeg_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, zNegBC, p_Fluid, p_zNegBC)
   var zPos_ispace = [UTIL.mkExtractRelevantIspace("cross_product", int3d, int3d, Fluid_columns, 0)](Fluid, zPosBC, p_Fluid, p_zPosBC)

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
   var is_xBCLeft_NSCBC  = [isNSCBC(BC_xBCLeft )];
   var is_xBCRight_NSCBC = [isNSCBC(BC_xBCRight)];
   var is_yBCLeft_NSCBC  = [isNSCBC(BC_yBCLeft )];
   var is_yBCRight_NSCBC = [isNSCBC(BC_yBCRight)];
   var is_zBCLeft_NSCBC  = [isNSCBC(BC_yBCLeft )];
   var is_zBCRight_NSCBC = [isNSCBC(BC_yBCRight)];

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

   if is_zBCLeft_NSCBC then
      -- zNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[ 5].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[ 5].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 5].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 5].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[ 5].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 8].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[15].bounds)
   end
   if is_zBCRight_NSCBC then
      -- zPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[ 6].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[ 6].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 6].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 6].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[ 6].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[10].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[16].bounds)
   end

   -----------------------------------------------------------------------------------------------
   if (is_xBCLeft_NSCBC and is_yBCLeft_NSCBC) then
      -- Edge xNeg-yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 7].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[19].bounds)
   end
   if (is_xBCLeft_NSCBC and is_zBCLeft_NSCBC) then
      -- Edge xNeg-zNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[ 8].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 8].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[ 8].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[19].bounds)
   end
   if (is_xBCLeft_NSCBC and is_yBCRight_NSCBC) then
      -- Edge xNeg-yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 9].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[20].bounds)
   end
   if (is_xBCLeft_NSCBC and is_zBCRight_NSCBC) then
      -- Edge xNeg-zPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[10].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[10].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[10].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[21].bounds)
   end
   -----------------------------------------------------------------------------------------------
   if (is_xBCRight_NSCBC and is_yBCLeft_NSCBC) then
      -- Edge xPos-yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[11].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[23].bounds)
   end
   if (is_xBCRight_NSCBC and is_zBCLeft_NSCBC) then
      -- Edge xPos-zNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[12].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[12].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[12].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[23].bounds)
   end
   if (is_xBCRight_NSCBC and is_yBCRight_NSCBC) then
      -- Edge xPos-yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[13].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[13].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[13].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[24].bounds)
   end
   if (is_xBCRight_NSCBC and is_zBCRight_NSCBC) then
      -- Edge xPos-zPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[14].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[14].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[14].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[25].bounds)
   end
   -----------------------------------------------------------------------------------------------
   if (is_yBCLeft_NSCBC and is_zBCLeft_NSCBC) then
      -- Edge yNeg-zNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[15].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[15].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[15].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[19].bounds)
   end
   if (is_yBCLeft_NSCBC and is_zBCRight_NSCBC) then
      -- Edge yNeg-zPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[16].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[16].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[16].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[21].bounds)
   end
   if (is_yBCRight_NSCBC and is_zBCLeft_NSCBC) then
      -- Edge yPos-zNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[17].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[17].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[17].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[20].bounds)
   end
   if (is_yBCRight_NSCBC and is_zBCRight_NSCBC) then
      -- Edge yPos-zPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[18].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[18].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[18].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[22].bounds)
   end
   -----------------------------------------------------------------------------------------------

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

local mkGhostPartiion = terralib.memoize(function(sdir, pType)
   local GhostPartition

   local offset
   local wrapCondLo
   local wrapCondHi
   local warpLo
   local warpHi
   local chopLo
   local chopHi
   if     sdir == "x" then
      offset = function(off) return rexpr { off, 0, 0} end end
      wrapCondLo = function(b, r) return rexpr (b.lo.x < r.lo.x) end end
      wrapCondHi = function(b, r) return rexpr (b.hi.x > r.hi.x) end end
      warpLo     = function(b, r) return rexpr {r.lo.x, b.lo.y, b.lo.z} end end
      warpHi     = function(b, r) return rexpr {r.hi.x, b.hi.y, b.hi.z} end end
      chopLo     = function(b, r) return rquote b.lo.x = max(b.lo.x, r.lo.x) end end
      chopHi     = function(b, r) return rquote b.hi.x = min(b.hi.x, r.hi.x) end end
   elseif sdir == "y" then
      offset = function(off) return rexpr { 0, off, 0} end end
      wrapCondLo = function(b, r) return rexpr (b.lo.y < r.lo.y) end end
      wrapCondHi = function(b, r) return rexpr (b.hi.y > r.hi.y) end end
      warpLo     = function(b, r) return rexpr {b.lo.x, r.lo.y, b.lo.z} end end
      warpHi     = function(b, r) return rexpr {b.hi.x, r.hi.y, b.hi.z} end end
      chopLo     = function(b, r) return rquote b.lo.y = max(b.lo.y, r.lo.y) end end
      chopHi     = function(b, r) return rquote b.hi.y = min(b.hi.y, r.hi.y) end end
   elseif sdir == "z" then
      offset = function(off) return rexpr { 0, 0, off} end end
      wrapCondLo = function(b, r) return rexpr (b.lo.z < r.lo.z) end end
      wrapCondHi = function(b, r) return rexpr (b.hi.z > r.hi.z) end end
      warpLo     = function(b, r) return rexpr {b.lo.x, b.lo.y, r.lo.z} end end
      warpHi     = function(b, r) return rexpr {b.hi.x, b.hi.y, r.hi.z} end end
      chopLo     = function(b, r) return rquote b.lo.z = max(b.lo.z, r.lo.z) end end
      chopHi     = function(b, r) return rquote b.hi.z = min(b.hi.z, r.hi.z) end end
   else assert(false) end

   local format = require("std/format")

   local addBoxToColor = function(coloring, c, b, r, off, periodic)
      return rquote
         b = b + [offset(off)]
         if periodic then
            -- Add support for points that are warping around
            if [wrapCondHi(b, rexpr r.bounds end)] then
                var lo_w = [warpLo(b, rexpr r.bounds end)];
                var hi_w = b.hi%r.bounds
                C.legion_multi_domain_point_coloring_color_domain(coloring, c,
                     rect3d{lo=lo_w, hi=hi_w})
            end
            if [wrapCondLo(b, rexpr r.bounds end)] then
                var lo_w = b.lo%r.bounds
                var hi_w = [warpHi(b, rexpr r.bounds end)];
                C.legion_multi_domain_point_coloring_color_domain(coloring, c,
                     rect3d{lo=lo_w, hi=hi_w})
            end
         end
         [chopLo(b, rexpr r.bounds end)];
         [chopHi(b, rexpr r.bounds end)];
         C.legion_multi_domain_point_coloring_color_domain(coloring, c, b)
      end
   end

   if pType == disjoint then
      __demand(__inline)
      task GhostPartition(r : region(ispace(int3d), Fluid_columns),
                          p : partition(pType, r, ispace(int3d)),
                          off : int,
                          periodic : bool)
         var coloring = C.legion_multi_domain_point_coloring_create()
         for c in p.colors do
            var b = p[c].bounds;
            [addBoxToColor(coloring, c, b, r, off, periodic)];
         end
         var ip = partition(aliased, r, coloring, p.colors)
         C.legion_multi_domain_point_coloring_destroy(coloring)
         return ip
      end
   elseif pType == aliased then
      __demand(__inline)
      task GhostPartition(r : region(ispace(int3d), Fluid_columns),
                          p : partition(pType, r, ispace(int3d)),
                          off : int,
                          periodic : bool)
         var coloring = C.legion_multi_domain_point_coloring_create()
         for c in p.colors do
            var is = __raw( p[c].ispace )
            var dom = C.legion_index_space_get_domain(__runtime(), is)
            var iter = C.legion_rect_in_domain_iterator_create_3d(dom)
            while (C.legion_rect_in_domain_iterator_valid_3d(iter)) do
               var b : rect3d = C.legion_rect_in_domain_iterator_get_rect_3d(iter);
               [addBoxToColor(coloring, c, b, r, off, periodic)];
               C.legion_rect_in_domain_iterator_step_3d(iter)
            end
            C.legion_rect_in_domain_iterator_destroy_3d(iter)
         end
         var ip = partition(aliased, r, coloring, p.colors)
         C.legion_multi_domain_point_coloring_destroy(coloring)
         return ip
      end
   else assert(false) end
   return GhostPartition
end)

local function emitEulerGhostRegion(sdir, t, r, p_r, flag)
   return rexpr
      [mkGhostPartiion(sdir, t)](r, p_r, -2, flag) |
      [mkGhostPartiion(sdir, t)](r, p_r, -1, flag) |
      [mkGhostPartiion(sdir, t)](r, p_r,  1, flag) |
      [mkGhostPartiion(sdir, t)](r, p_r,  2, flag) |
      [mkGhostPartiion(sdir, t)](r, p_r,  3, flag)
   end
end

__demand(__inline)
task Exports.PartitionGhost(Fluid : region(ispace(int3d), Fluid_columns),
                  tiles : ispace(int3d),
                  Fluid_Zones : zones_partitions(Fluid, tiles),
                  config : Config)
where
   reads(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All,
        p_x_faces, p_y_faces, p_z_faces,
        p_x_divg,  p_y_divg,  p_z_divg} = Fluid_Zones

   -- Check if the setup is periodic
   var Xperiodic = (config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic)
   var Yperiodic = (config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic)
   var Zperiodic = (config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic)

   -- Fluxes are computed at [0:-1] direction by direction
   var p_XFluxGhosts = p_x_divg | [mkGhostPartiion("x", disjoint)](Fluid, p_x_divg, -1, Xperiodic)
   var p_YFluxGhosts = p_y_divg | [mkGhostPartiion("y", disjoint)](Fluid, p_y_divg, -1, Yperiodic)
   var p_ZFluxGhosts = p_z_divg | [mkGhostPartiion("z", disjoint)](Fluid, p_z_divg, -1, Zperiodic);
   [UTIL.emitPartitionNameAttach(rexpr p_XFluxGhosts end, "p_XFluxGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YFluxGhosts end, "p_YFluxGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZFluxGhosts end, "p_ZFluxGhosts")];

   -- Compute auxiliary partitions
   -- X-Euler flux ghosts
   var x_fluxM2 = [mkGhostPartiion("x", aliased)](Fluid, p_XFluxGhosts, -2, Xperiodic)
   var x_fluxM1 = [mkGhostPartiion("x", aliased)](Fluid, p_XFluxGhosts, -1, Xperiodic)
   var x_fluxP1 = [mkGhostPartiion("x", aliased)](Fluid, p_XFluxGhosts,  1, Xperiodic)
   var x_fluxP2 = [mkGhostPartiion("x", aliased)](Fluid, p_XFluxGhosts,  2, Xperiodic)
   var x_fluxP3 = [mkGhostPartiion("x", aliased)](Fluid, p_XFluxGhosts,  3, Xperiodic)

   -- Y-Euler flux ghosts
   var y_fluxM2 = [mkGhostPartiion("y", aliased)](Fluid, p_YFluxGhosts, -2, Yperiodic)
   var y_fluxM1 = [mkGhostPartiion("y", aliased)](Fluid, p_YFluxGhosts, -1, Yperiodic)
   var y_fluxP1 = [mkGhostPartiion("y", aliased)](Fluid, p_YFluxGhosts,  1, Yperiodic)
   var y_fluxP2 = [mkGhostPartiion("y", aliased)](Fluid, p_YFluxGhosts,  2, Yperiodic)
   var y_fluxP3 = [mkGhostPartiion("y", aliased)](Fluid, p_YFluxGhosts,  3, Yperiodic)

   -- Z-Euler flux ghosts
   var z_fluxM2 = [mkGhostPartiion("z", aliased)](Fluid, p_ZFluxGhosts, -2, Zperiodic)
   var z_fluxM1 = [mkGhostPartiion("z", aliased)](Fluid, p_ZFluxGhosts, -1, Zperiodic)
   var z_fluxP1 = [mkGhostPartiion("z", aliased)](Fluid, p_ZFluxGhosts,  1, Zperiodic)
   var z_fluxP2 = [mkGhostPartiion("z", aliased)](Fluid, p_ZFluxGhosts,  2, Zperiodic)
   var z_fluxP3 = [mkGhostPartiion("z", aliased)](Fluid, p_ZFluxGhosts,  3, Zperiodic)

   -- Gradient ghosts
   var AllM1x = [mkGhostPartiion("x", disjoint)](Fluid, p_All, -1, Xperiodic)
   var AllP1x = [mkGhostPartiion("x", disjoint)](Fluid, p_All,  1, Xperiodic)

   var AllM1y = [mkGhostPartiion("y", disjoint)](Fluid, p_All, -1, Yperiodic)
   var AllP1y = [mkGhostPartiion("y", disjoint)](Fluid, p_All,  1, Yperiodic)

   var AllM1z = [mkGhostPartiion("z", disjoint)](Fluid, p_All, -1, Zperiodic)
   var AllP1z = [mkGhostPartiion("z", disjoint)](Fluid, p_All,  1, Zperiodic)

   -- Diffusion fluxes stencil accesses [0:+1] direction by direction...
   var p_XDiffGhosts = p_XFluxGhosts | x_fluxP1
   var p_YDiffGhosts = p_YFluxGhosts | y_fluxP1
   var p_ZDiffGhosts = p_ZFluxGhosts | z_fluxP1;
   [UTIL.emitPartitionNameAttach(rexpr p_XDiffGhosts end, "p_XDiffGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YDiffGhosts end, "p_YDiffGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZDiffGhosts end, "p_ZDiffGhosts")];

   -- ... and of the for traverse gradients
   var p_XDiffGradGhosts = ( p_XFluxGhosts |
                           [mkGhostPartiion("y", aliased)](Fluid, p_XDiffGhosts, -1, Yperiodic) |
                           [mkGhostPartiion("y", aliased)](Fluid, p_XDiffGhosts,  1, Yperiodic) |
                           [mkGhostPartiion("z", aliased)](Fluid, p_XDiffGhosts, -1, Zperiodic) |
                           [mkGhostPartiion("z", aliased)](Fluid, p_XDiffGhosts,  1, Zperiodic) )
   var p_YDiffGradGhosts = ( p_YFluxGhosts |
                           [mkGhostPartiion("x", aliased)](Fluid, p_YDiffGhosts, -1, Xperiodic) |
                           [mkGhostPartiion("x", aliased)](Fluid, p_YDiffGhosts,  1, Xperiodic) |
                           [mkGhostPartiion("z", aliased)](Fluid, p_YDiffGhosts, -1, Zperiodic) |
                           [mkGhostPartiion("z", aliased)](Fluid, p_YDiffGhosts,  1, Zperiodic) )
   var p_ZDiffGradGhosts = ( p_ZFluxGhosts |
                           [mkGhostPartiion("x", aliased)](Fluid, p_ZDiffGhosts, -1, Xperiodic) |
                           [mkGhostPartiion("x", aliased)](Fluid, p_ZDiffGhosts,  1, Xperiodic) |
                           [mkGhostPartiion("y", aliased)](Fluid, p_ZDiffGhosts, -1, Yperiodic) |
                           [mkGhostPartiion("y", aliased)](Fluid, p_ZDiffGhosts,  1, Yperiodic) );
   [UTIL.emitPartitionNameAttach(rexpr p_XDiffGradGhosts end, "p_XDiffGradGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YDiffGradGhosts end, "p_YDiffGradGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZDiffGradGhosts end, "p_ZDiffGradGhosts")];

   -- Euler fluxes stencil accesses [-2:+3] direction by direction
   var p_XEulerGhosts = p_XFluxGhosts | x_fluxM2 | x_fluxM1 | x_fluxP1 | x_fluxP2 | x_fluxP3
   var p_YEulerGhosts = p_YFluxGhosts | y_fluxM2 | y_fluxM1 | y_fluxP1 | y_fluxP2 | y_fluxP3
   var p_ZEulerGhosts = p_ZFluxGhosts | z_fluxM2 | z_fluxM1 | z_fluxP1 | z_fluxP2 | z_fluxP3;
   [UTIL.emitPartitionNameAttach(rexpr p_XEulerGhosts end, "p_XEulerGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YEulerGhosts end, "p_YEulerGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZEulerGhosts end, "p_ZEulerGhosts")];

   -- Shock sensors stencil accesses [-1:+1] direction by direction
   var p_XSensorGhosts2 = p_XFluxGhosts | x_fluxM1 | x_fluxP1
   var p_YSensorGhosts2 = p_YFluxGhosts | y_fluxM1 | y_fluxP1
   var p_ZSensorGhosts2 = p_ZFluxGhosts | z_fluxM1 | z_fluxP1;
   [UTIL.emitPartitionNameAttach(rexpr p_XSensorGhosts2 end, "p_XSensorGhosts2")];
   [UTIL.emitPartitionNameAttach(rexpr p_YSensorGhosts2 end, "p_YSensorGhosts2")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZSensorGhosts2 end, "p_ZSensorGhosts2")];

   -- Gradient tasks use [-1:+1] in all three directions
   var p_GradientGhosts = p_All | (AllM1x | AllP1x | AllM1y | AllP1y | AllM1z | AllP1z);
   [UTIL.emitPartitionNameAttach(rexpr p_GradientGhosts end, "p_GradientGhosts")];

   -- Metric routines (uses the entire stencil in all three directions)
   var MetricGhostsX = p_All | AllM1x
   var MetricGhostsY = p_All | AllM1y
   var MetricGhostsZ = p_All | AllM1z
   var p_MetricGhosts = p_All |
         [emitEulerGhostRegion("x", aliased, Fluid, MetricGhostsX, Xperiodic)] |
         [emitEulerGhostRegion("y", aliased, Fluid, MetricGhostsY, Yperiodic)] |
         [emitEulerGhostRegion("z", aliased, Fluid, MetricGhostsZ, Zperiodic)];
   [UTIL.emitPartitionNameAttach(rexpr p_MetricGhosts end, "p_MetricGhosts")];

   -- Shock sensor routines (uses [-2:+3] direction by direction)
   var p_XSensorGhosts = p_x_faces | [emitEulerGhostRegion("x", disjoint, Fluid, p_x_faces, Xperiodic)];
   var p_YSensorGhosts = p_y_faces | [emitEulerGhostRegion("y", disjoint, Fluid, p_y_faces, Yperiodic)];
   var p_ZSensorGhosts = p_z_faces | [emitEulerGhostRegion("z", disjoint, Fluid, p_z_faces, Zperiodic)];
   [UTIL.emitPartitionNameAttach(rexpr p_XSensorGhosts end, "p_XSensorGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_YSensorGhosts end, "p_YSensorGhosts")];
   [UTIL.emitPartitionNameAttach(rexpr p_ZSensorGhosts end, "p_ZSensorGhosts")];

   -- All With Ghosts
   var p_AllWithGhosts = p_All | p_GradientGhosts | p_MetricGhosts |
                         p_XFluxGhosts     | p_YFluxGhosts     | p_ZFluxGhosts     |
                         p_XDiffGhosts     | p_YDiffGhosts     | p_ZDiffGhosts     |
                         p_XDiffGradGhosts | p_YDiffGradGhosts | p_ZDiffGradGhosts |
                         p_XEulerGhosts    | p_YEulerGhosts    | p_ZEulerGhosts    |
                         p_XSensorGhosts2  | p_YSensorGhosts2  | p_ZSensorGhosts2  |
                         p_XSensorGhosts   | p_YSensorGhosts   | p_ZSensorGhosts;
   [UTIL.emitPartitionNameAttach(rexpr p_AllWithGhosts end, "p_AllWithGhosts")];

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
      -- Diffusion fluxes traverse gradients stencil access
      p_XDiffGradGhosts = p_XDiffGradGhosts,
      p_YDiffGradGhosts = p_YDiffGradGhosts,
      p_ZDiffGradGhosts = p_ZDiffGradGhosts,
      -- Euler fluxes routines
      p_XEulerGhosts = p_XEulerGhosts,
      p_YEulerGhosts = p_YEulerGhosts,
      p_ZEulerGhosts = p_ZEulerGhosts,
      -- Shock sensors stencil access
      p_XSensorGhosts2 = p_XSensorGhosts2,
      p_YSensorGhosts2 = p_YSensorGhosts2,
      p_ZSensorGhosts2 = p_ZSensorGhosts2,
      -- Shock sensor ghosts
      p_XSensorGhosts = p_XSensorGhosts,
      p_YSensorGhosts = p_YSensorGhosts,
      p_ZSensorGhosts = p_ZSensorGhosts,
      -- Gradient routines
      p_GradientGhosts =  p_GradientGhosts
   }
end

__demand(__inline)
task Exports.PartitionAverageGhost(
                  Fluid : region(ispace(int3d), Fluid_columns),
                  p_All : partition(disjoint, Fluid, ispace(int3d)),
                  p_Avg : partition(aliased, Fluid, ispace(int1d)),
                  cr_Avg : cross_product(p_Avg, p_All),
                  n_Avg : int,
                  config : Config)
where
   reads(Fluid)
do
   var tiles = p_All.colors
   -- This line matches the maximum number of average specified in config_schema.lua:341-347
   var p : average_ghost_partitions(Fluid, tiles)[10];

   -- Check if the setup is periodic
   var Xperiodic = (config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic)
   var Yperiodic = (config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic)
   var Zperiodic = (config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic)

   -- Define the Ghost partition for each partition of the cross product
   for i=0, n_Avg do
      var Avg = Fluid & cr_Avg[i]
      -- Compute auxiliary partitions
      var AvgM1x = [mkGhostPartiion("x", disjoint)](Fluid, Avg, -1, Xperiodic)
      var AvgP1x = [mkGhostPartiion("x", disjoint)](Fluid, Avg,  1, Xperiodic)

      var AvgM1y = [mkGhostPartiion("y", disjoint)](Fluid, Avg, -1, Yperiodic)
      var AvgP1y = [mkGhostPartiion("y", disjoint)](Fluid, Avg,  1, Yperiodic)

      var AvgM1z = [mkGhostPartiion("z", disjoint)](Fluid, Avg, -1, Zperiodic)
      var AvgP1z = [mkGhostPartiion("z", disjoint)](Fluid, Avg,  1, Zperiodic)

      -- Gradient tasks use [-1:+1] in all three directions
      var p_AvgGradientGhosts = Fluid & (Avg | AvgM1x | AvgP1x |
                                               AvgM1y | AvgP1y |
                                               AvgM1z | AvgP1z);
      [UTIL.emitPartitionNameAttach(rexpr p_AvgGradientGhosts end, "p_AvgGradientGhosts")];

      -- Store in the array of fspaces
      p[i] = [average_ghost_partitions(Fluid, tiles)] {
         -- Gradient routines
         p_GradientGhosts =  p_AvgGradientGhosts,
      }
   end
   return p
end

return Exports end

