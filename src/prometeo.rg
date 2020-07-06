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
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local MAPPER = terralib.includec("prometeo_mapper.h")
local SCHEMA = terralib.includec("config_schema.h")
local UTIL = require "util-desugared"
local VERSION = require "version"

-------------------------------------------------------------------------------
-- ACTIVATE TIMING
-------------------------------------------------------------------------------

local T
local TIMING = false
if os.getenv("TIMING") == "1" then
   TIMING = true
   T = regentlib.newsymbol()
   print("######################################################################################")
   print("WARNING: You are compiling with timing tools.")
   print("         This might affect the performance of the solver.")
   print("######################################################################################")
end

-------------------------------------------------------------------------------
-- ACTIVATE DEBUG_OUTPUT
-------------------------------------------------------------------------------

local DEBUG_OUTPUT = false
if os.getenv("DEBUG_OUTPUT") == "1" then
   DEBUG_OUTPUT = true
   print("#############################################################################")
   print("WARNING: You are compiling with debug output.")
   print("         This might affect the performance of the solver.")
   print("#############################################################################")
end

-------------------------------------------------------------------------------
-- ACTIVATE ATOMIC COHERENCE MODE
-------------------------------------------------------------------------------

local ATOMIC = true
if os.getenv("NO_ATOMIC") == "1" then
   ATOMIC = false
   print("#############################################################################")
   print("WARNING: You are compiling without atomic coherence mode.")
   print("         This might affect the performance of the solver.")
   print("#############################################################################")
end

-------------------------------------------------------------------------------
-- ACTIVATE AVERAGES
-------------------------------------------------------------------------------

local AVERAGES = true
if os.getenv("AVERAGES") == "0" then
   AVERAGES = false
   print("#############################################################################")
   print("WARNING: You are compiling without averaging tools.")
   print("#############################################################################")
end

-------------------------------------------------------------------------------
-- IMPORT MIXTURE
-------------------------------------------------------------------------------

local MIX
if (os.getenv("EOS") == "ConstPropMix") then
   MIX = (require "ConstPropMix")(SCHEMA)
elseif (os.getenv("EOS") == "AirMix") then
   MIX = (require 'AirMix')(SCHEMA)
elseif (os.getenv("EOS") == nil) then
   error ("You must define EOS enviromnment variable")
else
   error ("Unrecognized mixture: " .. os.getenv("EOS"))
end

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

local Config = SCHEMA.Config
--local MultiConfig = SCHEMA.MultiConfig

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"

-- Runge-Kutta coefficients
local RK_C = CONST.RK_C

-- Variable indices
local nSpec = MIX.nSpec       -- Number of species composing the mixture
local irU = CONST.GetirU(MIX) -- Index of the momentum in Conserved vector
local irE = CONST.GetirE(MIX) -- Index of the total energy density in Conserved vector
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

-- Stencil indices
local Stencil1  = CONST.Stencil1
local Stencil2  = CONST.Stencil2
local Stencil3  = CONST.Stencil3
local Stencil4  = CONST.Stencil4
local nStencils = CONST.nStencils

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

local struct Fluid_columns {
   -- Grid point
   centerCoordinates : double[3];
   cellWidth : double[3];
   -- Face reconstruction operators [c-2, ..., c+3]
   reconXFacePlus : double[nStencils*6];
   reconYFacePlus : double[nStencils*6];
   reconZFacePlus : double[nStencils*6];
   reconXFaceMinus : double[nStencils*6];
   reconYFaceMinus : double[nStencils*6];
   reconZFaceMinus : double[nStencils*6];
   -- Blending coefficients to obtain sixth order reconstruction
   TENOCoeffsXPlus : double[nStencils];
   TENOCoeffsYPlus : double[nStencils];
   TENOCoeffsZPlus : double[nStencils];
   TENOCoeffsXMinus : double[nStencils];
   TENOCoeffsYMinus : double[nStencils];
   TENOCoeffsZMinus : double[nStencils];
   -- Flags for modified reconstruction on BCs
   BCStencilX : bool;
   BCStencilY : bool;
   BCStencilZ : bool;
   -- Face interpolation operator [c, c+1]
   interpXFace : double[2];
   interpYFace : double[2];
   interpZFace : double[2];
   -- Face derivative operator [c+1 - c]
   derivXFace : double;
   derivYFace : double;
   derivZFace : double;
   -- Cell center gradient operator [c - c-1, c+1 - c]
   gradX : double[2];
   gradY : double[2];
   gradZ : double[2];
   -- Properties
   rho  : double;
   mu   : double;
   lam  : double;
   Di   : double[nSpec];
   SoS  : double;
   -- Primitive variables
   pressure    : double;
   temperature : double;
   MolarFracs  : double[nSpec];
   velocity    : double[3];
   -- Gradients
   velocityGradientX   : double[3];
   velocityGradientY   : double[3];
   velocityGradientZ   : double[3];
   temperatureGradient : double[3];
   -- Conserved variables
   Conserved       : double[nEq];
   Conserved_old   : double[nEq];
   Conserved_hat   : double[nEq];
   Conserved_t     : double[nEq];
   Conserved_t_old : double[nEq];
   -- Fluxes
   EulerFluxX : double[nEq];
   EulerFluxY : double[nEq];
   EulerFluxZ : double[nEq];
   FluxX      : double[nEq];
   FluxY      : double[nEq];
   FluxZ      : double[nEq];
   FluxXCorr  : double[nEq];
   FluxYCorr  : double[nEq];
   FluxZCorr  : double[nEq];
   -- NSCBC variables
   dudtBoundary : double[3];
   dTdtBoundary : double;
   velocity_old_NSCBC : double[3];
   temperature_old_NSCBC : double;
   -- Injected profile variables
   MolarFracs_profile  : double[nSpec];
   velocity_profile : double[3];
   temperature_profile : double;
}

local IOVars = terralib.newlist({
   'centerCoordinates',
   'cellWidth',
   'rho',
   'pressure',
   'temperature',
   'MolarFracs',
   'velocity',
   'dudtBoundary',
   'dTdtBoundary'
})

local DebugVars = terralib.newlist({
   'centerCoordinates',
   'cellWidth',
   'rho',
   'pressure',
   'temperature',
   'MolarFracs',
   'velocity',
   'Conserved',
   'FluxX',
   'FluxY',
   'FluxZ'
})

fspace grid_partitions(Fluid  : region(ispace(int3d), Fluid_columns), tiles : ispace(int3d)) {
   -- Partitions
   p_All      : partition(disjoint, Fluid, tiles),
   p_Interior : partition(disjoint, Fluid, tiles),
   p_AllGhost : partition(disjoint, Fluid, tiles),
   -- Partitions for reconstruction operator
-- TODO: store *_face as subregions of Fluid
   x_faces    : partition(disjoint, Fluid, ispace(int1d)),
   y_faces    : partition(disjoint, Fluid, ispace(int1d)),
   z_faces    : partition(disjoint, Fluid, ispace(int1d)),
   p_x_faces  : cross_product(x_faces, p_All),
   p_y_faces  : cross_product(y_faces, p_All),
   p_z_faces  : cross_product(z_faces, p_All),
   -- Partitions for divergence operator
-- TODO: store *_divg as subregions of Fluid
--   x_divg     : region(ispace(int3d), Fluid_columns),
--   y_divg     : region(ispace(int3d), Fluid_columns),
--   z_divg     : region(ispace(int3d), Fluid_columns),
--   p_x_divg   : partition(disjoint, x_divg, tiles),
--   p_y_divg   : partition(disjoint, y_divg, tiles),
--   p_z_divg   : partition(disjoint, z_divg, tiles),
   x_divg     : partition(disjoint, Fluid, ispace(int1d)),
   y_divg     : partition(disjoint, Fluid, ispace(int1d)),
   z_divg     : partition(disjoint, Fluid, ispace(int1d)),
   p_x_divg   : cross_product(x_divg, p_All),
   p_y_divg   : cross_product(y_divg, p_All),
   p_z_divg   : cross_product(z_divg, p_All),
   p_solved   : partition(disjoint, Fluid, tiles),
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
   p_zPos     : cross_product(p_All, zPos)
}
--where
--   x_divg <= Fluid,
--   y_divg <= Fluid,
--   z_divg <= Fluid
--end

-------------------------------------------------------------------------------
-- EXTERNAL MODULES IMPORTS
-------------------------------------------------------------------------------

local HDF = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                                 IOVars,
                                                 {timeStep=int,simTime=double,channelForcing=double},
                                                 {SpeciesNames={nSpec, 20}, Versions={2, VERSION.Length}})

local HDF_DEBUG
if DEBUG_OUTPUT then
   HDF_DEBUG = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                                 DebugVars,
                                                 {timeStep=int,simTime=double,channelForcing=double},
                                                 {SpeciesNames={nSpec, 20}, Versions={2, VERSION.Length}})
end

-- Macro
local MACRO = require "prometeo_macro"

-- Mesh routines
local GRID = (require 'prometeo_grid')(SCHEMA, Fluid_columns)

-- Metric routines
local METRIC = (require 'prometeo_metric')(SCHEMA, Fluid_columns)

-- I/O routines
local IO = (require 'prometeo_IO')(SCHEMA)

-- Stability conditions routines
local CFL = (require 'prometeo_cfl')(MIX, Fluid_columns)

-- Chemistry routines
local CHEM = (require 'prometeo_chem')(SCHEMA, MIX, Fluid_columns, ATOMIC)

-- Initialization routines
local INIT = (require 'prometeo_init')(SCHEMA, MIX, CHEM, Fluid_columns)

-- Conserved->Primitives/Primitives->Conserved and properties routines
local VARS = (require 'prometeo_variables')(SCHEMA, MIX, Fluid_columns)

-- Fluxes routines
local FLUX = (require 'prometeo_flux')(SCHEMA, MIX, Fluid_columns)

-- BCOND routines
local BCOND = (require 'prometeo_bc')(SCHEMA, MIX, Fluid_columns, grid_partitions)

-- RK routines
local RK = (require 'prometeo_rk')(nEq, Fluid_columns)

-- RHS routines
local RHS = (require 'prometeo_rhs')(SCHEMA, MIX, Fluid_columns, ATOMIC)

-- Volume averages routines
local STAT = (require 'prometeo_stat')(MIX, Fluid_columns)

-- Profiles routines
local PROFILES = (require 'prometeo_profiles')(SCHEMA, MIX, Fluid_columns)

-- Averages routines
local AVG
if AVERAGES then
   AVG = (require 'prometeo_average')(SCHEMA, MIX, Fluid_columns)
end

-------------------------------------------------------------------------------
-- INITIALIZATION ROUTINES
-------------------------------------------------------------------------------

local function emitFill(p, t, f, val)
   return rquote
      var v = [val]
      __demand(__index_launch)
      for c in t do
         fill(([p][c]).[f], v)
      end
   end
end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns),
                    tiles : ispace(int3d),
                    Fluid_Partitions : grid_partitions(Fluid, tiles))
where
   writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All} = Fluid_Partitions;

   [emitFill(p_All, tiles, "centerCoordinates", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "cellWidth",         rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "reconXFacePlus",  UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "reconYFacePlus",  UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "reconZFacePlus",  UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "reconXFaceMinus", UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "reconYFaceMinus", UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "reconZFaceMinus", UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "TENOCoeffsXPlus", UTIL.mkArrayConstant(nStencils,   rexpr 0.0 end))];
   [emitFill(p_All, tiles, "TENOCoeffsYPlus", UTIL.mkArrayConstant(nStencils,   rexpr 0.0 end))];
   [emitFill(p_All, tiles, "TENOCoeffsZPlus", UTIL.mkArrayConstant(nStencils,   rexpr 0.0 end))];
   [emitFill(p_All, tiles, "TENOCoeffsXMinus", UTIL.mkArrayConstant(nStencils,  rexpr 0.0 end))];
   [emitFill(p_All, tiles, "TENOCoeffsYMinus", UTIL.mkArrayConstant(nStencils,  rexpr 0.0 end))];
   [emitFill(p_All, tiles, "TENOCoeffsZMinus", UTIL.mkArrayConstant(nStencils,  rexpr 0.0 end))];
   [emitFill(p_All, tiles, "BCStencilX", rexpr false end)];
   [emitFill(p_All, tiles, "BCStencilY", rexpr false end)];
   [emitFill(p_All, tiles, "BCStencilZ", rexpr false end)];
   [emitFill(p_All, tiles, "interpXFace", rexpr array(0.0, 0.0) end)];
   [emitFill(p_All, tiles, "interpYFace", rexpr array(0.0, 0.0) end)];
   [emitFill(p_All, tiles, "interpZFace", rexpr array(0.0, 0.0) end)];
   [emitFill(p_All, tiles, "derivXFace", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "derivYFace", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "derivZFace", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "gradX", rexpr array(0.0, 0.0) end)];
   [emitFill(p_All, tiles, "gradY", rexpr array(0.0, 0.0) end)];
   [emitFill(p_All, tiles, "gradZ", rexpr array(0.0, 0.0) end)];
   [emitFill(p_All, tiles, "rho", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "mu" , rexpr 0.0 end)];
   [emitFill(p_All, tiles, "lam", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "Di" , UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "SoS", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "pressure", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "temperature", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "MolarFracs", UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "velocity",            rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "velocityGradientX",   rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "velocityGradientY",   rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "velocityGradientZ",   rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "temperatureGradient", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "Conserved",       UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_old",   UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_hat",   UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_t",     UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_t_old", UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "EulerFluxX",      UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "EulerFluxY",      UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "EulerFluxZ",      UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "FluxX",           UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "FluxY",           UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "FluxZ",           UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "FluxXCorr",       UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "FluxYCorr",       UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "FluxZCorr",       UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "dudtBoundary", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "dTdtBoundary", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "velocity_old_NSCBC", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "temperature_old_NSCBC", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "MolarFracs_profile", UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "velocity_profile", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "temperature_profile", rexpr 0.0 end)];
end

-------------------------------------------------------------------------------
-- PARTITIONING ROUTINES
-------------------------------------------------------------------------------

local function addToBcColoring(c, rect, stencil)
   return rquote
      regentlib.c.legion_multi_domain_point_coloring_color_domain([c], int1d(0), [rect])
      regentlib.c.legion_multi_domain_point_coloring_color_domain([c], int1d(1), [rect] + [stencil])
   end
end

__demand(__inline)
task PartitionGrid(Fluid : region(ispace(int3d), Fluid_columns),
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
   var p_Fluid_Interior = static_cast(partition(disjoint, Fluid, tiles), cross_product(Fluid_regions, p_Fluid)[0]);
   [UTIL.emitPartitionNameAttach(rexpr p_Fluid_Interior end, "p_Interior")];

   -- All ghost points
   var p_Fluid_AllGhost = p_Fluid - p_Fluid_Interior

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
   var BC_zBCRight = config.BC.zBCRight.type

   ------------------------------------------------------
   -- Break ties with other boundary conditions for edges
   ------------------------------------------------------
   -- [ 7]: Edge xNeg-yNeg
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow
   elseif (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[7].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_xBCLeft == SCHEMA.FlowBC_Periodic) and
           (BC_yBCLeft == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yNeg edge")
   end

   -- [ 8]: Edge xNeg-zNeg
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall) then 
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[8].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[8].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_zBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[8].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[8].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow

   -- Periodic
   elseif ((BC_xBCLeft == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yNeg edge")
   end

   -- [ 9]: Edge xNeg-yPos
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)       then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[9].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_xBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_yBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yPos edge")
   end
 
   -- [10]: Edge xNeg-zPos
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[10].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[10].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[10].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[10].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow

   -- Periodic
   elseif ((BC_xBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-zPos edge")
   end

   -- [11]: Edge xPos-yNeg
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall)     then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow and BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{-1, 0, 0} end)];
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[11].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_yBCLeft  == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-yNeg edge")
   end

   -- [12]: Edge xPos-zNeg
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[12].bounds end, rexpr int3d{-1, 0, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[12].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_zBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[12].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[12].bounds end, rexpr int3d{-1, 0, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft  == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-zNeg edge")
   end

   -- [13]: Edge xPos-yPos
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow and BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      -- This edge belongs to both partitions
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{-1, 0, 0} end)];
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[13].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_yBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-yPos edge")
   end

   -- [14]: Edge xPos-zPos
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[14].bounds end, rexpr int3d{-1, 0, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[14].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[14].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[14].bounds end, rexpr int3d{-1, 0, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-zPos edge")
   end

   -- [15]: Edge yNeg-zNeg
   -- Walls
   if (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[15].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[15].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[15].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_yBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[15].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_zBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[15].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[15].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_yBCLeft == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of yNeg-zNeg edge")
   end

   -- [16]: Edge yNeg-zPos
   -- Walls
   if (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[16].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[16].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[16].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[16].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[16].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[16].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_yBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of yNeg-zPos edge")
   end

   -- [17]: Edge yPos-zNeg
   -- Walls
   if (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[17].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[17].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[17].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_zBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[17].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[17].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_yBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft  == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of yPos-zNeg edge")
   end

   -- [18]: Edge yPos-zPos
   -- Walls
   if (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[18].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[18].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[18].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[18].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[18].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_yBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of yPos-zPos edge")
   end

   --------------------------------------------------------
   -- Break ties with other boundary conditions for corners
   --------------------------------------------------------

   -- [19]: Corner xNeg-yNeg-zNeg
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_zBCLeft == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow
   elseif (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[19].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_xBCLeft == SCHEMA.FlowBC_Periodic) and
           (BC_yBCLeft == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yNeg-zNeg corner")
   end

   -- [20]: Corner xNeg-yPos-zNeg
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)       then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_zBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[20].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_xBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_yBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft  == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yPos-zNeg corner")

   end

   -- [21]: Corner xNeg-yNeg-zPos
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow
   elseif (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[21].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_xBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_yBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yNeg-zPos corner")
   end

   -- [22]: Corner xNeg-yPos-zPos
   -- Walls
   if (BC_xBCLeft == SCHEMA.FlowBC_AdiabaticWall)       then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow
   elseif (BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) then
      [addToBcColoring(rexpr xNeg_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 1, 0, 0} end)]

   -- NSCBC_Outflow
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[22].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_xBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_yBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xNeg-yPos-zPos corner")
   end

   -- [23]: Corner xPos-yNeg-zNeg
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall)     then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_zBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow and BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{-1, 0, 0} end)];
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[23].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_yBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft  == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-yNeg-zNeg corner")
   end

   -- [24]: Corner xPos-yPos-zNeg
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_zBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zNeg_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{ 0, 0, 1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow and BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      -- This edge belongs to both partitions
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{-1, 0, 0} end)];
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[24].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_yBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCLeft  == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-yPos-zNeg corner")
   end

   -- [25]: Corner xPos-yNeg-zPos
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall)     then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_yBCLeft == SCHEMA.FlowBC_SuctionAndBlowingWall) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow and BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{-1, 0, 0} end)];
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 1, 0} end)]
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCLeft  == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yNeg_coloring end , rexpr Fluid_regions[25].bounds end, rexpr int3d{ 0, 1, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_yBCLeft  == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-yNeg-zPos corner")
   end

   -- [26]: Corner xPos-yPos-zPos
   -- Walls
   if (BC_xBCRight == SCHEMA.FlowBC_AdiabaticWall)      then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_AdiabaticWall)  then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_IsothermalWall) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Dirichlet
   elseif (BC_xBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_zBCRight == SCHEMA.FlowBC_Dirichlet) then
      [addToBcColoring(rexpr zPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{ 0, 0,-1} end)]

   -- NSCBC_Inflow

   -- NSCBC_Outflow
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow and BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      -- This edge belongs to both partitions
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{-1, 0, 0} end)];
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{ 0,-1, 0} end)]
   elseif (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr xPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{-1, 0, 0} end)]
   elseif (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      [addToBcColoring(rexpr yPos_coloring end , rexpr Fluid_regions[26].bounds end, rexpr int3d{ 0,-1, 0} end)]

   -- Periodic
   elseif ((BC_xBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_yBCRight == SCHEMA.FlowBC_Periodic) and
           (BC_zBCRight == SCHEMA.FlowBC_Periodic)) then
      -- Nothing to do

   else regentlib.assert(false, "Unhandled case in tie breaking of xPos-yPos-zPos corner")
   end

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
   if (BC_xBCLeft  == SCHEMA.FlowBC_NSCBC_Inflow) then
      -- xNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[1].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[1].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[8].bounds)
   end
   if (BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      -- xPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(ydivg_coloring,  int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 2].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 2].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(yfaces_coloring, int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[12].bounds)
   end

   if (BC_yBCLeft  == SCHEMA.FlowBC_NSCBC_Outflow) then
      -- yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 3].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 3].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[15].bounds)
   end
   if (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow) then
      -- yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xdivg_coloring,  int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 4].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 4].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(xfaces_coloring, int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[17].bounds)
   end

   if ((BC_xBCLeft == SCHEMA.FlowBC_NSCBC_Inflow) and
       (BC_yBCLeft == SCHEMA.FlowBC_NSCBC_Outflow)) then
      -- Edge xNeg-yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 7].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 7].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[19].bounds)
   end
   if ((BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) and
       (BC_yBCLeft  == SCHEMA.FlowBC_NSCBC_Outflow)) then
      -- Edge xPos-yNeg is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[11].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[11].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[23].bounds)
   end
   if ((BC_xBCLeft  == SCHEMA.FlowBC_NSCBC_Inflow) and
       (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow)) then
      -- Edge xNeg-yPos is an NSCBC
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zdivg_coloring,  int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(solve_coloring,  int1d(0), Fluid_regions[ 9].bounds)

      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[ 9].bounds)
      regentlib.c.legion_multi_domain_point_coloring_color_domain(zfaces_coloring, int1d(0), Fluid_regions[20].bounds)
   end
   if ((BC_xBCRight == SCHEMA.FlowBC_NSCBC_Outflow) and
       (BC_yBCRight == SCHEMA.FlowBC_NSCBC_Outflow)) then
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

   var xdivg_part = cross_product(xdivg_cells, p_Fluid)
   var ydivg_part = cross_product(ydivg_cells, p_Fluid)
   var zdivg_part = cross_product(zdivg_cells, p_Fluid)
   var solve_part = static_cast(partition(disjoint, Fluid, tiles), cross_product(solve_cells, p_Fluid)[0]);
   [UTIL.emitPartitionNameAttach(rexpr xdivg_part[0] end, "p_x_divg")];
   [UTIL.emitPartitionNameAttach(rexpr ydivg_part[0] end, "p_y_divg")];
   [UTIL.emitPartitionNameAttach(rexpr zdivg_part[0] end, "p_z_divg")];
   [UTIL.emitPartitionNameAttach(rexpr solve_part    end, "p_solved")];

   var xfaces_part = cross_product(xfaces_cells, p_Fluid)
   var yfaces_part = cross_product(yfaces_cells, p_Fluid)
   var zfaces_part = cross_product(zfaces_cells, p_Fluid);
   [UTIL.emitPartitionNameAttach(rexpr xfaces_part[0] end, "p_x_faces")];
   [UTIL.emitPartitionNameAttach(rexpr yfaces_part[0] end, "p_y_faces")];
   [UTIL.emitPartitionNameAttach(rexpr zfaces_part[0] end, "p_z_faces")];

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

   return [grid_partitions(Fluid, tiles)]{
      -- Partitions
      p_All      = p_Fluid,
      p_Interior = p_Fluid_Interior,
      p_AllGhost = p_Fluid_AllGhost,
      -- Partitions for reconstruction operator
      x_faces   = xfaces_cells,
      y_faces   = yfaces_cells,
      z_faces   = zfaces_cells,
      p_x_faces = xfaces_part,
      p_y_faces = yfaces_part,
      p_z_faces = zfaces_part,
      -- Partitions for divergence operator
      x_divg   = xdivg_cells,
      y_divg   = ydivg_cells,
      z_divg   = zdivg_cells,
      p_x_divg = xdivg_part,
      p_y_divg = ydivg_part,
      p_z_divg = zdivg_part,
      p_solved = solve_part,
      -- BC partitions
      xNeg       = xNegBC,
      xPos       = xPosBC,
      yNeg       = yNegBC,
      yPos       = yPosBC,
      zNeg       = zNegBC,
      zPos       = zPosBC,
      p_xNeg     = p_xNegBC,
      p_xPos     = p_xPosBC,
      p_yNeg     = p_yNegBC,
      p_yPos     = p_yPosBC,
      p_zNeg     = p_zNegBC,
      p_zPos     = p_zPosBC,
   }
end

-------------------------------------------------------------------------------
-- DEBUG ROUTINES
-------------------------------------------------------------------------------

local DetectNaN
local CheckDebugOutput
if DEBUG_OUTPUT then

local isnan = regentlib.isnan(double)
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task DetectNaN(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{pressure, temperature})
do
   var err = 0
   __demand(__openmp)
   for c in Fluid do
      if ([bool](isnan(Fluid[c].pressure   ))) then err += 1 end
      if ([bool](isnan(Fluid[c].temperature))) then err += 1 end
   end
   return err
end

__demand(__inline)
task CheckDebugOutput(Fluid : region(ispace(int3d), Fluid_columns),
                      Fluid_copy : region(ispace(int3d), Fluid_columns),
                      Fluid_bounds : rect3d,
                      tiles : ispace(int3d),
                      Fluid_Partitions : grid_partitions(Fluid, tiles),
                      Fluid_Partitions_copy : grid_partitions(Fluid_copy, tiles),
                      config : Config,
                      Mix : MIX.Mixture,
                      Integrator_timeStep : int,
                      Integrator_simTime : double)
where
   reads(Fluid),
   reads writes(Fluid_copy),
   Fluid * Fluid_copy
do
   -- Unpack the partitions that we are going to need
   var {p_All} = Fluid_Partitions

   var err = 0
   __demand(__index_launch)
   for c in tiles do
      err += DetectNaN(p_All[c])
   end
   if err ~= 0 then
      var SpeciesNames = MIX.GetSpeciesNames(Mix)
      var dirname = [&int8](C.malloc(256))
      C.snprintf(dirname, 256, '%s/debugOut', config.Mapping.outDir)
      var _1 = IO.createDir(dirname)
      _1 = HDF_DEBUG.dump(                 _1, tiles, dirname, Fluid, Fluid_copy, p_All, Fluid_Partitions_copy.p_All)
      _1 = HDF_DEBUG.write.timeStep(       _1, tiles, dirname, Fluid, p_All, Integrator_timeStep)
      _1 = HDF_DEBUG.write.simTime(        _1, tiles, dirname, Fluid, p_All, Integrator_simTime)
      _1 = HDF_DEBUG.write.SpeciesNames(   _1, tiles, dirname, Fluid, p_All, SpeciesNames)
      _1 = HDF_DEBUG.write.Versions(       _1, tiles, dirname, Fluid, p_All, array(regentlib.string([VERSION.SolverVersion]), regentlib.string([VERSION.LegionVersion])))
      _1 = HDF_DEBUG.write.channelForcing( _1, tiles, dirname, Fluid, p_All, config.Flow.turbForcing.u.CHANNEL.Forcing);
      C.free(dirname)

      __fence(__execution, __block)
      regentlib.assert(false, "NaN detected! Debug fields dumped in debugOut")
   end
end
end

-------------------------------------------------------------------------------
-- RK-LOOP ROUTINES
-------------------------------------------------------------------------------

__demand(__inline)
task UpdateFluxesFromConserved(Fluid : region(ispace(int3d), Fluid_columns),
                               Fluid_bounds : rect3d,
                               tiles : ispace(int3d),
                               Fluid_Partitions : grid_partitions(Fluid, tiles),
                               x_faces : region(ispace(int3d), Fluid_columns),
                               y_faces : region(ispace(int3d), Fluid_columns),
                               z_faces : region(ispace(int3d), Fluid_columns),
                               config : Config,
                               Mix : MIX.Mixture,
                               Integrator_simTime : double)
where
   reads writes(Fluid),
   reads writes(x_faces), x_faces <= Fluid,
   reads writes(y_faces), y_faces <= Fluid,
   reads writes(z_faces), z_faces <= Fluid
do

   -- Unpack the partitions that we are going to need
   var {p_All, p_Interior, p_AllGhost, p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos} = Fluid_Partitions;

@ESCAPE if TIMING then @EMIT
   var [T] = IO.Console_WriteTiming(0, config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Update all primitive variables... 
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePrimitiveFromConserved(p_All[c], p_Interior[c], Mix)
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- ...also in the ghost cells
   BCOND.UpdateGhostPrimitives(Fluid,
                               tiles,
                               Fluid_Partitions,
                               config,
                               Mix,
                               Integrator_simTime);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Update the mixture properties everywhere
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePropertiesFromPrimitive(p_All[c], p_All[c], Mix)
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- update values of conserved variables in ghost cells
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdateConservedFromPrimitive(p_All[c], p_AllGhost[c], Mix)
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Compute velocity gradients
   VARS.GetVelocityGradients(Fluid, Fluid_bounds);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Compute local Euler fluxes
   __demand(__index_launch)
   for c in tiles do
      FLUX.GetEulerFlux(p_All[c])
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Compute fluxes
   [FLUX.mkGetFlux("x")](Fluid, x_faces, Fluid_bounds, Mix);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [FLUX.mkGetFlux("y")](Fluid, y_faces, Fluid_bounds, Mix);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [FLUX.mkGetFlux("z")](Fluid, z_faces, Fluid_bounds, Mix);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateFluxesFromConserved", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

end

__demand(__inline)
task UpdateDerivativesFromFluxes(Fluid : region(ispace(int3d), Fluid_columns),
                                 Fluid_bounds : rect3d,
                                 tiles : ispace(int3d),
                                 Fluid_Partitions : grid_partitions(Fluid, tiles),
                                 x_divg : region(ispace(int3d), Fluid_columns),
                                 y_divg : region(ispace(int3d), Fluid_columns),
                                 z_divg : region(ispace(int3d), Fluid_columns),
                                 config : Config,
                                 Mix : MIX.Mixture,
                                 UseOldDerivatives : bool)
where
   reads writes(Fluid),
   reads writes(x_divg), x_divg <= Fluid,
   reads writes(y_divg), y_divg <= Fluid,
   reads writes(z_divg), z_divg <= Fluid
do

   -- Unpack the partitions that we are going to need
   var {p_All, p_Interior, p_solved, p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos} = Fluid_Partitions;

@ESCAPE if TIMING then @EMIT
   var [T] = IO.Console_WriteTiming(0, config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Initialize time derivatives to 0 or minus the old value
   if UseOldDerivatives then
      __demand(__index_launch)
      for c in tiles do
         [RK.mkInitializeTimeDerivatives(true)](p_All[c])
      end
   else
      __demand(__index_launch)
      for c in tiles do
         [RK.mkInitializeTimeDerivatives(false)](p_All[c])
      end
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   if (not config.Integrator.implicitChemistry) then
      -- Add chemistry source terms
      __demand(__index_launch)
      for c in tiles do
         CHEM.AddChemistrySources(p_All[c], p_solved[c], Mix)
      end
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Add body forces
   __demand(__index_launch)
   for c in tiles do
      RHS.AddBodyForces(p_All[c], p_solved[c], config.Flow.bodyForce)
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Add turbulent forcing
   if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
      -- Add forcing
      __demand(__index_launch)
      for c in tiles do
         RHS.AddBodyForces(p_All[c], p_solved[c],
                           array(config.Flow.turbForcing.u.CHANNEL.Forcing, 0.0, 0.0))
      end
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Use fluxes to update conserved value derivatives
   [RHS.mkUpdateUsingFlux("z")](Fluid, z_divg, Fluid_bounds);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [RHS.mkUpdateUsingFlux("y")](Fluid, y_divg, Fluid_bounds);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [RHS.mkUpdateUsingFlux("x")](Fluid, x_divg, Fluid_bounds);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Update using NSCBC_Outflow bcs
   if (config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_All[c], p_Interior[c], 0)
      end
      __demand(__index_launch)
      for c in tiles do
         [RHS.mkUpdateUsingFluxNSCBCOutflow("xPos")](p_All[c],
                                                     p_xPos[c],
                                                     Mix, MaxMach, config.Grid.xWidth, config.BC.xBCRight.u.NSCBC_Outflow.P)
      end
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow or config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
      var MaxMach = 0.0
      __demand(__index_launch)
      for c in tiles do
         MaxMach max= STAT.CalculateMaxMachNumber(p_All[c], p_Interior[c], 1)
      end
      if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in tiles do
            [RHS.mkUpdateUsingFluxNSCBCOutflow("yNeg")](p_All[c],
                                                        p_yNeg[c],
                                                        Mix, MaxMach, config.Grid.yWidth, config.BC.yBCLeft.u.NSCBC_Outflow.P)
         end
      end
      if (config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
         __demand(__index_launch)
         for c in tiles do
            [RHS.mkUpdateUsingFluxNSCBCOutflow("yPos")](p_All[c],
                                                        p_yPos[c],
                                                        Mix, MaxMach, config.Grid.yWidth, config.BC.yBCRight.u.NSCBC_Outflow.P)
         end
      end
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   -- Update using NSCBC_Inflow bcs
   if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) then
      __demand(__index_launch)
      for c in tiles do
         RHS.UpdateUsingFluxNSCBCInflow(p_All[c], p_xNeg[c], Mix)
      end
   end

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "UpdateDerivativesFromFluxes", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

end

__demand(__inline)
task CorrectDerivatives(Fluid : region(ispace(int3d), Fluid_columns),
                        Fluid_bounds : rect3d,
                        tiles : ispace(int3d),
                        Fluid_Partitions : grid_partitions(Fluid, tiles),
                        x_divg : region(ispace(int3d), Fluid_columns),
                        y_divg : region(ispace(int3d), Fluid_columns),
                        z_divg : region(ispace(int3d), Fluid_columns),
                        config : Config,
                        Mix : MIX.Mixture)
where
   reads writes(Fluid),
   reads writes(x_divg), x_divg <= Fluid,
   reads writes(y_divg), y_divg <= Fluid,
   reads writes(z_divg), z_divg <= Fluid
do

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "CorrectDerivatives", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [RHS.mkCorrectUsingFlux("z")](Fluid, z_divg, Fluid_bounds, Mix);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "CorrectDerivatives", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [RHS.mkCorrectUsingFlux("y")](Fluid, y_divg, Fluid_bounds, Mix);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "CorrectDerivatives", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   [RHS.mkCorrectUsingFlux("x")](Fluid, x_divg, Fluid_bounds, Mix);

@ESCAPE if TIMING then @EMIT
   [T] = IO.Console_WriteTiming([T], config.Mapping, "CorrectDerivatives", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

end

-------------------------------------------------------------------------------
-- MAIN SIMULATION
-------------------------------------------------------------------------------

local function mkInstance() local INSTANCE = {}

   -----------------------------------------------------------------------------
   -- Symbols shared between quotes
   -----------------------------------------------------------------------------

   local startTime = regentlib.newsymbol()
   local Grid = {
      xBnum = regentlib.newsymbol(),
      yBnum = regentlib.newsymbol(),
      zBnum = regentlib.newsymbol(),
      NX = regentlib.newsymbol(),
      NY = regentlib.newsymbol(),
      NZ = regentlib.newsymbol(),
      numTiles = regentlib.newsymbol(),
   }
   local BC = {
      readProfiles = regentlib.newsymbol(bool),
      ProfilesDir = regentlib.newsymbol(&int8),
   }

   local Integrator_deltaTime = regentlib.newsymbol()
   local Integrator_simTime   = regentlib.newsymbol()
   local Integrator_timeStep  = regentlib.newsymbol()
   local Integrator_exitCond  = regentlib.newsymbol()

   local Mix = regentlib.newsymbol()

   local Fluid = regentlib.newsymbol("Fluid")
   local Fluid_copy = regentlib.newsymbol()
   local Fluid_bounds = regentlib.newsymbol("Fluid_bounds")

   local tiles = regentlib.newsymbol()

   local Fluid_Partitions = regentlib.newsymbol("Fluid_Partitions")
   local Fluid_Partitions_copy = regentlib.newsymbol()

   local p_All    = regentlib.newsymbol("p_All")
   local x_divg   = regentlib.newsymbol("x_divg")
   local y_divg   = regentlib.newsymbol("y_divg")
   local z_divg   = regentlib.newsymbol("z_divg")
   local p_x_divg = regentlib.newsymbol("p_x_divg")
   local p_y_divg = regentlib.newsymbol("p_y_divg")
   local p_z_divg = regentlib.newsymbol("p_z_divg")
   local x_faces   = regentlib.newsymbol("x_faces")
   local y_faces   = regentlib.newsymbol("y_faces")
   local z_faces   = regentlib.newsymbol("z_faces")
   local p_x_faces = regentlib.newsymbol("p_x_faces")
   local p_y_faces = regentlib.newsymbol("p_y_faces")
   local p_z_faces = regentlib.newsymbol("p_z_faces")

   local Averages
   if AVERAGES then
      Averages = AVG.AvgList
   end

   -----------------------------------------------------------------------------
   -- Exported symbols
   -----------------------------------------------------------------------------

   INSTANCE.Grid = Grid
   INSTANCE.Integrator_deltaTime = Integrator_deltaTime
   INSTANCE.Integrator_simTime   = Integrator_simTime
   INSTANCE.Integrator_timeStep  = Integrator_timeStep
   INSTANCE.Integrator_exitCond  = Integrator_exitCond
   INSTANCE.Fluid = Fluid
   INSTANCE.Fluid_copy = Fluid_copy
   INSTANCE.tiles = tiles
   INSTANCE.Fluid_Partitions = Fluid_Partitions
   INSTANCE.Fluid_Partitions_copy = Fluid_Partitions_copy

   INSTANCE.p_All = p_All
   INSTANCE.x_divg = x_divg
   INSTANCE.y_divg = y_divg
   INSTANCE.z_divg = z_divg
   INSTANCE.p_x_divg = p_x_divg
   INSTANCE.p_y_divg = p_y_divg
   INSTANCE.p_z_divg = p_z_divg
   INSTANCE.x_faces = x_faces
   INSTANCE.y_faces = y_faces
   INSTANCE.z_faces = z_faces
   INSTANCE.p_x_faces = p_x_faces
   INSTANCE.p_y_faces = p_y_faces
   INSTANCE.p_z_faces = p_z_faces

   -----------------------------------------------------------------------------
   -- Symbol declaration & initialization
   -----------------------------------------------------------------------------

   function INSTANCE.DeclSymbols(config) return rquote

      ---------------------------------------------------------------------------
      -- Preparation
      ---------------------------------------------------------------------------

      -- Start timer
      var [startTime] = C.legion_get_current_time_in_micros() / 1000;

      -- Write console header
      IO.Console_WriteHeader(config.Mapping)

--    -- Write probe file headers
--    var probeId = 0
--    while probeId < config.IO.probes.length do
--      IO.Probe_WriteHeader(config, probeId)
--      probeId += 1
--    end

      ---------------------------------------------------------------------------
      -- Initialize the mixture
      ---------------------------------------------------------------------------

      var [Mix] = MIX.InitMixture(config)
 
      ---------------------------------------------------------------------------
      -- Declare & initialize state variables
      ---------------------------------------------------------------------------

      var [BC.readProfiles] = false
      var [BC.ProfilesDir] = ''

      -- Determine number of ghost cells in each direction
      -- 0 ghost cells if periodic and 1 otherwise
      var [Grid.xBnum] = 1
      var [Grid.yBnum] = 1
      var [Grid.zBnum] = 1
      if config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic then Grid.xBnum = 0 end
      if config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic then Grid.yBnum = 0 end
      if config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic then Grid.zBnum = 0 end

      var [Grid.NX] = config.Mapping.tiles[0]
      var [Grid.NY] = config.Mapping.tiles[1]
      var [Grid.NZ] = config.Mapping.tiles[2]
      var [Grid.numTiles] = Grid.NX * Grid.NY * Grid.NZ

      var [Integrator_exitCond] = true
      var [Integrator_simTime]   = config.Integrator.startTime
      var [Integrator_timeStep]  = config.Integrator.startIter
      var [Integrator_deltaTime] = config.Integrator.fixedDeltaTime

      -- Set up flow BC's in x direction
      if (not((config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.xBCRight.type == SCHEMA.FlowBC_Periodic))) then

         if (config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow) then
            [BCOND.CheckNSCBC_Inflow(BC, rexpr config.BC.xBCLeft.u.NSCBC_Inflow end)];

         elseif (config.BC.xBCLeft.type == SCHEMA.FlowBC_Dirichlet) then
            [BCOND.CheckDirichlet(BC, rexpr config.BC.xBCLeft.u.Dirichlet end)];

         elseif (config.BC.xBCLeft.type == SCHEMA.FlowBC_AdiabaticWall) then
            -- Do nothing

         else
            regentlib.assert(false, "Boundary conditions in xBCLeft not implemented")
         end

         if (config.BC.xBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
            -- Do nothing

         elseif (config.BC.xBCRight.type == SCHEMA.FlowBC_Dirichlet) then
            [BCOND.CheckDirichlet(BC, rexpr config.BC.xBCRight.u.Dirichlet end)];

         elseif (config.BC.xBCRight.type == SCHEMA.FlowBC_AdiabaticWall) then
            -- Do nothing

         else
            regentlib.assert(false, "Boundary conditions in xBCRight not implemented")
         end
      end

      -- Set up flow BC's in y direction
      if (not((config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.yBCRight.type == SCHEMA.FlowBC_Periodic))) then

         if (config.BC.yBCLeft.type == SCHEMA.FlowBC_NSCBC_Outflow) then
            -- Do nothing

         elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_Dirichlet) then
            [BCOND.CheckDirichlet(BC, rexpr config.BC.yBCLeft.u.Dirichlet end)];

         elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_AdiabaticWall) then
            -- Do nothing

         elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_IsothermalWall) then
            [BCOND.CheckIsothermalWall(rexpr config.BC.yBCLeft.u.IsothermalWall end)];

         elseif (config.BC.yBCLeft.type == SCHEMA.FlowBC_SuctionAndBlowingWall) then
            [BCOND.CheckSuctionAndBlowingWall(rexpr config.BC.yBCLeft.u.SuctionAndBlowingWall end)];

         else
            regentlib.assert(false, "Boundary conditions in yBCLeft not implemented")
         end

         if (config.BC.yBCRight.type == SCHEMA.FlowBC_NSCBC_Outflow) then
            -- Do nothing

         elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_Dirichlet) then
            [BCOND.CheckDirichlet(BC, rexpr config.BC.yBCRight.u.Dirichlet end)];

         elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_AdiabaticWall) then
            -- Do nothing

         elseif (config.BC.yBCRight.type == SCHEMA.FlowBC_IsothermalWall) then
            [BCOND.CheckIsothermalWall(rexpr config.BC.yBCRight.u.IsothermalWall end)];

         else
            regentlib.assert(false, "Boundary conditions in yBCRight not implemented")
         end
      end

      -- Set up flow BC's in z direction
      if (not((config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.zBCRight.type == SCHEMA.FlowBC_Periodic))) then

         if (config.BC.zBCLeft.type == SCHEMA.FlowBC_Dirichlet) then
            [BCOND.CheckDirichlet(BC, rexpr config.BC.zBCLeft.u.Dirichlet end)];

         else
            regentlib.assert(false, "Boundary conditions in zBCLeft not implemented")
         end

         if (config.BC.zBCRight.type == SCHEMA.FlowBC_Dirichlet) then
            [BCOND.CheckDirichlet(BC, rexpr config.BC.yBCRight.u.Dirichlet end)];

         else
            regentlib.assert(false, "Boundary conditions in zBCRight not implemented")
         end
      end

      -- Check if boundary conditions in each direction are either both periodic or both non-periodic
      if (not (((config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.xBCRight.type == SCHEMA.FlowBC_Periodic))
          or ((not (config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic)) and (not (config.BC.xBCRight.type == SCHEMA.FlowBC_Periodic))))) then
         regentlib.assert(false, "Boundary conditions in x should match for periodicity")
      end
      if (not (((config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.yBCRight.type == SCHEMA.FlowBC_Periodic))
         or ((not (config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic)) and (not (config.BC.yBCRight.type == SCHEMA.FlowBC_Periodic))))) then
         regentlib.assert(false, "Boundary conditions in y should match for periodicity")
      end
      if (not (((config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic) and (config.BC.zBCRight.type == SCHEMA.FlowBC_Periodic))
         or ((not (config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic)) and (not (config.BC.zBCRight.type == SCHEMA.FlowBC_Periodic))))) then
         regentlib.assert(false, "Boundary conditions in z should match for periodicity")
      end

      ---------------------------------------------------------------------------
      -- Initialize forcing values
      ---------------------------------------------------------------------------

      if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
         config.Flow.turbForcing.u.CHANNEL.Forcing max= 1.0
      end

      ---------------------------------------------------------------------------
      -- Create Regions and Partitions
      ---------------------------------------------------------------------------

      var sampleId = config.Mapping.sampleId

      -- Create Fluid Regions
      var is_Fluid = ispace(int3d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                    y = config.Grid.yNum + 2*Grid.yBnum,
                                    z = config.Grid.zNum + 2*Grid.zBnum})
      var [Fluid] = region(is_Fluid, Fluid_columns);
      [UTIL.emitRegionTagAttach(Fluid, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      var [Fluid_bounds] = [Fluid].bounds
      var [Fluid_copy] = region(is_Fluid, Fluid_columns);
      [UTIL.emitRegionTagAttach(Fluid_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

      -- Partitioning domain
      var [tiles] = ispace(int3d, {Grid.NX, Grid.NY, Grid.NZ})

      -- Fluid Partitioning
      var [Fluid_Partitions]      = PartitionGrid(Fluid,      tiles, config, Grid.xBnum, Grid.yBnum, Grid.zBnum)
      var [Fluid_Partitions_copy] = PartitionGrid(Fluid_copy, tiles, config, Grid.xBnum, Grid.yBnum, Grid.zBnum)

      -- Unpack regions that will be used by autoparallelized tasks
      var { p_All_t =   p_All,
            x_divg_t = x_divg, 
            y_divg_t = y_divg, 
            z_divg_t = z_divg, 
            p_x_divg_t = p_x_divg,
            p_y_divg_t = p_y_divg,
            p_z_divg_t = p_z_divg,
            x_faces_t = x_faces, 
            y_faces_t = y_faces, 
            z_faces_t = z_faces, 
            p_x_faces_t = p_x_faces,
            p_y_faces_t = p_y_faces,
            p_z_faces_t = p_z_faces } = Fluid_Partitions
      var [p_All] = p_All_t
      var [x_divg] = x_divg_t[0]
      var [y_divg] = y_divg_t[0]
      var [z_divg] = z_divg_t[0]
      var [p_x_divg] = p_x_divg_t[0]
      var [p_y_divg] = p_y_divg_t[0]
      var [p_z_divg] = p_z_divg_t[0]
      var [x_faces] = x_faces_t[0]
      var [y_faces] = y_faces_t[0]
      var [z_faces] = z_faces_t[0]
      var [p_x_faces] = p_x_faces_t[0]
      var [p_y_faces] = p_y_faces_t[0]
      var [p_z_faces] = p_z_faces_t[0];

      ---------------------------------------------------------------------------
      -- Create one-dimensional averages
      ---------------------------------------------------------------------------
@ESCAPE if AVERAGES then @EMIT
      [AVG.DeclSymbols(Averages, Grid, config, MAPPER)];
@TIME end @EPACSE

   end end -- DeclSymbols

   -----------------------------------------------------------------------------
   -- Region initialization
   -----------------------------------------------------------------------------

   function INSTANCE.InitRegions(config) return rquote

      InitializeCell(Fluid, tiles, Fluid_Partitions)

      -- Unpack the partitions that we are going to need
      var {p_All, p_Interior, p_AllGhost, 
             xNeg,   xPos,   yNeg,   yPos,   zNeg,   zPos,
           p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos} = Fluid_Partitions

      -- If we read a restart file it will contain the geometry
      if config.Flow.initCase ~= SCHEMA.FlowInitCase_Restart then
         __demand(__index_launch)
         for c in tiles do
            GRID.InitializeGeometry(p_All[c],
                                    config.Grid.xType, config.Grid.yType, config.Grid.zType,
                                    config.Grid.xStretching, config.Grid.yStretching, config.Grid.zStretching,
                                    Grid.xBnum, config.Grid.xNum, config.Grid.origin[0], config.Grid.xWidth,
                                    Grid.yBnum, config.Grid.yNum, config.Grid.origin[1], config.Grid.yWidth,
                                    Grid.zBnum, config.Grid.zNum, config.Grid.origin[2], config.Grid.zWidth)
         end

         __demand(__index_launch)
         for c in tiles do
            GRID.InitializeGhostGeometry(p_All[c],
                                         config.Grid.xType, config.Grid.yType, config.Grid.zType,
                                         config.Grid.xStretching, config.Grid.yStretching, config.Grid.zStretching,
                                         Grid.xBnum, config.Grid.xNum, config.Grid.origin[0], config.Grid.xWidth,
                                         Grid.yBnum, config.Grid.yNum, config.Grid.origin[1], config.Grid.yWidth,
                                         Grid.zBnum, config.Grid.zNum, config.Grid.origin[2], config.Grid.zWidth)
         end
      end

@ESCAPE if AVERAGES then @EMIT
      -- Initialize averages
      [AVG.InitRakes(Averages)];
      -- Initialize reduction buffers
      [AVG.InitBuffers(Averages)];
@TIME end @EPACSE

      -- Initialize BC profiles
      -- Read from file...
      -- TODO: this will eventually become a separate call for each BC
      if BC.readProfiles then
         PROFILES.HDF.load(0, tiles, BC.ProfilesDir, Fluid, Fluid_copy, p_All, Fluid_Partitions_copy.p_All)
      end
      -- ... or use the config
      [PROFILES.mkInitializeProfilesField("xBCRight")](xPos[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("xBCLeft" )](xNeg[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("yBCRight")](yPos[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("yBCLeft" )](yNeg[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("zBCRight")](zPos[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("zBCLeft" )](zNeg[0], config, Mix);

      -- Initialize solution
      var initMolarFracs = CHEM.ParseConfigMixture(config.Flow.initMixture, Mix)

      if config.Flow.initCase == SCHEMA.FlowInitCase_Uniform then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeUniform(p_All[c], config.Flow.initParams, initMolarFracs)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_Random then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeRandom(p_All[c], config.Flow.initParams, initMolarFracs)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_TaylorGreen2DVortex then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeTaylorGreen2D(p_All[c],
                                         config.Flow.initParams,
                                         initMolarFracs,
                                         Mix,
                                         Grid.xBnum, config.Grid.xNum, config.Grid.origin[0], config.Grid.xWidth,
                                         Grid.yBnum, config.Grid.yNum, config.Grid.origin[1], config.Grid.yWidth,
                                         Grid.zBnum, config.Grid.zNum, config.Grid.origin[2], config.Grid.zWidth)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_TaylorGreen3DVortex then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeTaylorGreen3D(p_All[c],
                                         config.Flow.initParams,
                                         initMolarFracs,
                                         Mix,
                                         Grid.xBnum, config.Grid.xNum, config.Grid.origin[0], config.Grid.xWidth,
                                         Grid.yBnum, config.Grid.yNum, config.Grid.origin[1], config.Grid.yWidth,
                                         Grid.zBnum, config.Grid.zNum, config.Grid.origin[2], config.Grid.zWidth)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_Perturbed then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializePerturbed(p_All[c], config.Flow.initParams, initMolarFracs)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_RiemannTestOne then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeRiemannTestOne(p_All[c], initMolarFracs, Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_RiemannTestTwo then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeRiemannTestTwo(p_All[c], initMolarFracs, Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_SodProblem then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeSodProblem(p_All[c], initMolarFracs, Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_LaxProblem then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeLaxProblem(p_All[c], initMolarFracs, Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_ShuOsherProblem then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeShuOsherProblem(p_All[c], initMolarFracs, Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_VortexAdvection2D then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeVortexAdvection2D(p_All[c], config.Flow.initParams, initMolarFracs, Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_GrossmanCinnellaProblem then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeGrossmanCinnellaProblem(p_All[c], Mix)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_ChannelFlow then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeChannelFlow(p_All[c], 
                                       config.Flow.initParams,
                                       initMolarFracs,
                                       Mix,
                                       Grid.xBnum, config.Grid.xNum, config.Grid.origin[0], config.Grid.xWidth,
                                       Grid.yBnum, config.Grid.yNum, config.Grid.origin[1], config.Grid.yWidth,
                                       Grid.zBnum, config.Grid.zNum, config.Grid.origin[2], config.Grid.zWidth)
         end
      elseif config.Flow.initCase == SCHEMA.FlowInitCase_Restart then
         Integrator_timeStep = HDF.read.timeStep(0, tiles, config.Flow.restartDir, Fluid, p_All)
         Integrator_simTime  = HDF.read.simTime( 0, tiles, config.Flow.restartDir, Fluid, p_All)
         if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
            config.Flow.turbForcing.u.CHANNEL.Forcing = HDF.read.channelForcing( 0, tiles, config.Flow.restartDir, Fluid, p_All)
         end
         HDF.load(0, tiles, config.Flow.restartDir, Fluid, Fluid_copy, p_All, Fluid_Partitions_copy.p_All);

@ESCAPE if AVERAGES then @EMIT
         [AVG.ReadAverages(Averages, config)];
@TIME end @EPACSE

      else regentlib.assert(false, 'Unhandled case in switch') end

      if config.Integrator.resetTime then
         Integrator_simTime  = config.Integrator.startTime
         Integrator_timeStep = config.Integrator.startIter
      end

      if config.Flow.resetMixture then
         __demand(__index_launch)
         for c in tiles do
            CHEM.ResetMixture(p_All[c], p_Interior[c], initMolarFracs)
         end
      end

      -- Initialize grid operators
      METRIC.InitializeMetric(Fluid,
                              Fluid_bounds,
                              Grid.xBnum, config.Grid.xNum,
                              Grid.yBnum, config.Grid.yNum,
                              Grid.zBnum, config.Grid.zNum);

      -- Enforce BCs on the metric
      [METRIC.mkCorrectGhostMetric("x")](Fluid, Fluid_bounds, config.BC.xBCLeft.type, config.BC.xBCRight.type, Grid.xBnum, config.Grid.xNum);
      [METRIC.mkCorrectGhostMetric("y")](Fluid, Fluid_bounds, config.BC.yBCLeft.type, config.BC.yBCRight.type, Grid.yBnum, config.Grid.yNum);
      [METRIC.mkCorrectGhostMetric("z")](Fluid, Fluid_bounds, config.BC.zBCLeft.type, config.BC.zBCRight.type, Grid.zBnum, config.Grid.zNum)

      -- initialize ghost cells values for NSCBC
      if config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow then
         __demand(__index_launch)
         for c in tiles do
            BCOND.InitializeGhostNSCBC(p_All[c], p_xNeg[c], Mix)
         end
      end

      __demand(__index_launch)
      for c in tiles do
         VARS.UpdatePropertiesFromPrimitive(p_All[c], p_All[c], Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         VARS.UpdateConservedFromPrimitive(p_All[c], p_Interior[c], Mix)
      end

      -- update values of conserved variables in ghost cells
      __demand(__index_launch)
      for c in tiles do
         VARS.UpdateConservedFromPrimitive(p_All[c], p_AllGhost[c], Mix)
      end

      UpdateFluxesFromConserved(Fluid,
                                Fluid_bounds,
                                tiles,
                                Fluid_Partitions,
                                x_faces, y_faces, z_faces,
                                config,
                                Mix,
                                Integrator_simTime)

   end end -- InitRegions

   -----------------------------------------------------------------------------
   -- Main time-step loop header
   -----------------------------------------------------------------------------

   function INSTANCE.MainLoopHeader(config) return rquote

      -- Unpack the partitions that we are going to need
      var {p_All} = Fluid_Partitions;

@ESCAPE if TIMING then @EMIT
      var [T] = IO.Console_WriteTiming(0, config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      -- Calculate exit condition
      Integrator_exitCond =
         (Integrator_timeStep >= config.Integrator.maxIter) or
         (Integrator_simTime  >= config.Integrator.maxTime)

      -- Determine time step size
      if config.Integrator.cfl > 0.0 then
         var Integrator_maxSpectralRadius = 0.0
         __demand(__index_launch)
         for c in tiles do
            Integrator_maxSpectralRadius max= CFL.CalculateMaxSpectralRadius(p_All[c], Mix)
         end
         Integrator_deltaTime = config.Integrator.cfl/Integrator_maxSpectralRadius
      end

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   end end -- MainLoopHeader

   -----------------------------------------------------------------------------
   -- Per-time-step I/O
   -----------------------------------------------------------------------------

   function INSTANCE.PerformIO(config) return rquote

      -- Unpack the partitions that we are going to need
      var {p_All, p_Interior} = Fluid_Partitions;

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      -- Write to console
      var AveragePressure = 0.0
      var AverageTemperature = 0.0
      var AverageKineticEnergy = 0.0
      var averageRhoU = 0.0
      __demand(__index_launch)
      for c in tiles do
         AveragePressure      += STAT.CalculateAveragePressure(p_All[c], p_Interior[c]) 
      end
      __demand(__index_launch)
      for c in tiles do
         AverageTemperature   += STAT.CalculateAverageTemperature(p_All[c], p_Interior[c])
      end
      __demand(__index_launch)
      for c in tiles do
         AverageKineticEnergy += STAT.CalculateAverageKineticEnergy(p_All[c], p_Interior[c])
      end

      -- Rescale channel flow forcing
      if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
         __demand(__index_launch)
         for c in tiles do
            averageRhoU += STAT.CalculateAverageRhoU(p_All[c], p_Interior[c], 0)
         end
      end

      var interior_volume = 0.0
      __demand(__index_launch)
      for c in tiles do
         interior_volume += STAT.CalculateInteriorVolume(p_All[c], p_Interior[c]) 
      end

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      AveragePressure      = (AveragePressure     /interior_volume)
      AverageTemperature   = (AverageTemperature  /interior_volume)
      AverageKineticEnergy = (AverageKineticEnergy/interior_volume)
      IO.Console_Write(config.Mapping,
                       Integrator_timeStep,
                       Integrator_simTime,
                       startTime,
                       Integrator_deltaTime,
                       AveragePressure,
                       AverageTemperature,
                       AverageKineticEnergy)

      if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
         averageRhoU = (averageRhoU / interior_volume)
         config.Flow.turbForcing.u.CHANNEL.Forcing *= config.Flow.turbForcing.u.CHANNEL.RhoUbulk/averageRhoU
      end

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

@ESCAPE if AVERAGES then @EMIT
      -- Add averages
      if (Integrator_timeStep % config.IO.AveragesSamplingInterval == 0 and 
         ((config.IO.YZAverages.length ~= 0) or
          (config.IO.XZAverages.length ~= 0) or
          (config.IO.XYAverages.length ~= 0) )) then

         -- Update temperature gradient for mean heat flux
         VARS.GetTemperatureGradients(Fluid, Fluid_bounds);

         [AVG.AddAverages(Averages, tiles, p_All, Integrator_deltaTime, config, Mix)]

      end

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

@TIME end @EPACSE

      -- Dump restart files
      if config.IO.wrtRestart then
         if Integrator_exitCond or Integrator_timeStep % config.IO.restartEveryTimeSteps == 0 then
            var SpeciesNames = MIX.GetSpeciesNames(Mix)
            var dirname = [&int8](C.malloc(256))
            C.snprintf(dirname, 256, '%s/fluid_iter%010d', config.Mapping.outDir, Integrator_timeStep)
            var _1 = IO.createDir(dirname)
            _1 = HDF.dump(                 _1, tiles, dirname, Fluid, Fluid_copy, p_All, Fluid_Partitions_copy.p_All)
            _1 = HDF.write.timeStep(       _1, tiles, dirname, Fluid, p_All, Integrator_timeStep)
            _1 = HDF.write.simTime(        _1, tiles, dirname, Fluid, p_All, Integrator_simTime)
            _1 = HDF.write.SpeciesNames(   _1, tiles, dirname, Fluid, p_All, SpeciesNames)
            _1 = HDF.write.Versions(       _1, tiles, dirname, Fluid, p_All, array(regentlib.string([VERSION.SolverVersion]), regentlib.string([VERSION.LegionVersion])))
            _1 = HDF.write.channelForcing( _1, tiles, dirname, Fluid, p_All, config.Flow.turbForcing.u.CHANNEL.Forcing);

@ESCAPE if AVERAGES then @EMIT
            [AVG.WriteAverages(Averages, tiles, dirname,IO, SpeciesNames, config)];
@TIME end @EPACSE

            C.free(dirname)
         end
      end

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

   end end -- PerformIO

   -----------------------------------------------------------------------------
   -- Main time-step loop body
   -----------------------------------------------------------------------------

   function INSTANCE.MainLoopBody(config) return rquote

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      var Integrator_time_old = Integrator_simTime

      -- Unpack the partitions that we are going to need
      var {p_All, p_solved, p_xNeg, p_xPos, p_yNeg, p_yPos, p_zNeg, p_zPos} = Fluid_Partitions

      if config.Integrator.implicitChemistry then
         ---------------------------------------------------------------
         -- Update the conserved varialbes using the implicit solver ---
         ---------------------------------------------------------------
         -- Update the time derivatives
         UpdateDerivativesFromFluxes(Fluid,
                                     Fluid_bounds,
                                     tiles,
                                     Fluid_Partitions,
                                     x_divg, y_divg, z_divg,
                                     config,
                                     Mix,
                                     false);

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

         -- TODO: it is not clear if here we need to correct the derivatives to
         --       preserve boundness

         -- Advance chemistry implicitely
         __demand(__index_launch)
         for c in tiles do
            CHEM.UpdateChemistry(p_All[c], p_solved[c], Integrator_deltaTime, Mix)
         end

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

         -- Update the fluxes in preparation to the RK algorithm
         UpdateFluxesFromConserved(Fluid,
                                   Fluid_bounds,
                                   tiles,
                                   Fluid_Partitions,
                                   x_faces, y_faces, z_faces,
                                   config,
                                   Mix,
                                   Integrator_simTime)

         -- The result of the local implicit solver is at 0.5*dt
         Integrator_simTime = Integrator_time_old + Integrator_deltaTime*0.5
      end

@ESCAPE if TIMING then @EMIT
      [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      -- Set iteration-specific fields that persist across RK sub-steps
      __demand(__index_launch)
      for c in tiles do
         RK.InitializeTemporaries(p_All[c])
      end

      -- RK sub-time-stepping loop
      @ESCAPE for STAGE = 1, 3 do @EMIT

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

         -- Update the time derivatives
         UpdateDerivativesFromFluxes(Fluid,
                                     Fluid_bounds,
                                     tiles,
                                     Fluid_Partitions,
                                     x_divg, y_divg, z_divg,
                                     config,
                                     Mix,
                                     config.Integrator.implicitChemistry);

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

         -- Predictor part of the time step
         __demand(__index_launch)
         for c in tiles do
            [RK.mkUpdateVarsPred(STAGE)](p_All[c], Integrator_deltaTime, config.Integrator.implicitChemistry)
         end

         -- Correct time derivatives to preserve boundness
         CorrectDerivatives(Fluid,
                            Fluid_bounds,
                            tiles,
                            Fluid_Partitions,
                            x_divg, y_divg, z_divg,
                            config,
                            Mix)

         -- Corrector part of the time step
         __demand(__index_launch)
         for c in tiles do
            [RK.mkUpdateVarsCorr(STAGE)](p_All[c], Integrator_deltaTime, config.Integrator.implicitChemistry)
         end

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

         -- Update the fluxes
         UpdateFluxesFromConserved(Fluid,
                                   Fluid_bounds,
                                   tiles,
                                   Fluid_Partitions,
                                   x_faces, y_faces, z_faces,
                                   config,
                                   Mix,
                                   Integrator_simTime);

@ESCAPE if DEBUG_OUTPUT then @EMIT
         CheckDebugOutput(Fluid, Fluid_copy,
                          Fluid_bounds, tiles,
                          Fluid_Partitions, Fluid_Partitions_copy,
                          config,
                          Mix,
                          Integrator_timeStep,
                          Integrator_simTime);
@TIME end @EPACSE

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

         -- Advance the time for the next sub-step
         @ESCAPE if STAGE == 3 then @EMIT
               Integrator_simTime = Integrator_time_old + Integrator_deltaTime
         @TIME else @EMIT
               if config.Integrator.implicitChemistry then
                  Integrator_simTime =   Integrator_time_old 
                                       + Integrator_deltaTime * 0.5 *(1.0 + [RK_C[STAGE][3]])
               else
                  Integrator_simTime = Integrator_time_old + [RK_C[STAGE][3]] * Integrator_deltaTime
               end
         @TIME end @EPACSE

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      @TIME end @EPACSE-- RK sub-time-stepping

      -- Update time derivatives at boundary for NSCBC
      if config.BC.xBCLeft.type == SCHEMA.FlowBC_NSCBC_Inflow then
         __demand(__index_launch)
         for c in tiles do
            BCOND.UpdateNSCBCGhostCellTimeDerivatives(p_All[c], p_xNeg[c], Integrator_deltaTime)
         end
      end

@ESCAPE if TIMING then @EMIT
         [T] = IO.Console_WriteTiming([T], config.Mapping, "workSingle", @LINE, C.legion_get_current_time_in_nanos())
@TIME end @EPACSE

      Integrator_timeStep += 1

   end end -- MainLoopBody

   -----------------------------------------------------------------------------
   -- Cleanup code
   -----------------------------------------------------------------------------

   function INSTANCE.Cleanup(config) return rquote

      -- Wait for everything above to finish
      __fence(__execution, __block)

      -- Report final time
      IO.Console_WriteFooter(config.Mapping, startTime)

   end end -- Cleanup

return INSTANCE end -- mkInstance

-------------------------------------------------------------------------------
-- TOP-LEVEL INTERFACE
-------------------------------------------------------------------------------

local function parallelizeFor(sim, stmts)
   return rquote
      __parallelize_with
      sim.tiles,
      disjoint(sim.p_All),
      complete(sim.p_All, sim.Fluid),
      disjoint(sim.p_x_divg),
      complete(sim.p_x_divg, sim.x_divg),
      disjoint(sim.p_y_divg),
      complete(sim.p_y_divg, sim.y_divg),
      disjoint(sim.p_z_divg),
      complete(sim.p_z_divg, sim.z_divg),
      sim.p_x_divg <= sim.p_All,
      sim.p_y_divg <= sim.p_All,
      sim.p_z_divg <= sim.p_All,
      disjoint(sim.p_x_faces),
      complete(sim.p_x_faces, sim.x_faces),
      disjoint(sim.p_y_faces),
      complete(sim.p_y_faces, sim.y_faces),
      disjoint(sim.p_z_faces),
      complete(sim.p_z_faces, sim.z_faces),
      sim.p_x_faces <= sim.p_All,
      sim.p_y_faces <= sim.p_All,
      sim.p_z_faces <= sim.p_All
      do [stmts] end
   end
end

local SIM = mkInstance()

__demand(__inner, __replicable)
task workSingle(config : Config)
   [SIM.DeclSymbols(config)];
   [parallelizeFor(SIM, rquote
      [SIM.InitRegions(config)];
      while true do
         [SIM.MainLoopHeader(config)];
         [SIM.PerformIO(config)];
         if SIM.Integrator_exitCond then
            break
         end
         [SIM.MainLoopBody(config)];
      end
   end)];
   [SIM.Cleanup(config)];
end

__demand(__inline)
task initSingle(config : &Config, launched : int, outDirBase : &int8)
   config.Mapping.sampleId = launched
   C.snprintf([&int8](config.Mapping.outDir), 256, "%s/sample%d", outDirBase, launched)
   UTIL.createDir(config.Mapping.outDir)
end

__demand(__inner)
task main()
   var args = regentlib.c.legion_runtime_get_input_args()
   var outDirBase = '.'
   for i = 1, args.argc do
      if C.strcmp(args.argv[i], '-o') == 0 and i < args.argc-1 then
         outDirBase = args.argv[i+1]
      end
   end
   var launched = 0
   for i = 1, args.argc do
      if C.strcmp(args.argv[i], '-i') == 0 and i < args.argc-1 then
         var config : Config
         SCHEMA.parse_Config(&config, args.argv[i+1])
         initSingle(&config, launched, outDirBase)
         launched += 1
         workSingle(config)
      end
   end
   if launched < 1 then
      var stderr = C.fdopen(2, 'w')
      C.fprintf(stderr, "No testcases supplied.\n")
      C.fflush(stderr)
      C.exit(1)
   end
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "prometeo_"..os.getenv("EOS")..".o", "object", MAPPER.register_mappers)
