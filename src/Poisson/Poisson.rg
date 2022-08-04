-- Copyright (c) "2020, by Centre Européen de Recherche et de Formation Avancée en Calcul Scientifiq
--               Developer: Mario Di Renzo
--               Affiliation: Centre Européen de Recherche et de Formation Avancée en Calcul Scientifique
--               URL: https://cerfacs.fr
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

-- srcFld: list of fields in Fluid_columns required for the source term
-- outFld: field in Fluid_columns containing the solution
-- mXFld: field in Fluid_columns containing the cell centered metric along X
-- mYFld: field in Fluid_columns containing the cell centered metric along Y
-- mZFld: field in Fluid_columns containing the cell centered metric alond Z
-- mY_sFld: field in Fluid_columns containing the staggered metric along Y

return function(SCHEMA, MIX, TYPES, Fluid_columns,
                srcFlds, TID_DirFFT,
                outFld,
                mXFld, mYFld, mZFld, mY_sFld, nType) local Exports = {}

local USE_CUDA = (os.getenv("USE_CUDA") == "1")

if (os.getenv("USE_FFTW") ~= "1") then
   error ("Poisson module needs FFTW.")
end

-------------------------------------------------------------------------------
-- CHECK THAT TID_DirFFT IS CORRCTLY SET
-------------------------------------------------------------------------------
if (TID_DirFFT == TYPES.TID_performDirFFTFromField) then
   assert(table.getn(srcFlds) == 1, "performDirFFTFromField requires only one srcFlds")
elseif (TID_DirFFT == TYPES.TID_performDirFFTFromMix) then
   -- Nothing to do
else
   assert(false, "Unsupported TID_DirFFT in Poisson solver")
end

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------
local C = regentlib.c
local UTIL = require 'util'
local CONST = require "prometeo_const"
local POISSON_H = terralib.includec("Poisson.h", {"-DUSE_CUDA="..os.getenv("USE_CUDA")})

local sin  = regentlib.sin(double)
local cos  = regentlib.cos(double)
local sqrt = regentlib.sqrt(double)

local PI   = CONST.PI

-- Node types
local L_S_node   = CONST.L_S_node
local R_S_node   = CONST.R_S_node
local L_C_node   = CONST.L_C_node
local R_C_node   = CONST.R_C_node

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

-- Data for tridiagonal system
local struct CoeffType {
   -- Sub-diagonal coefficients
   a : double;
   -- diagonal coefficients
   b : double;
   -- super-diagonal coefficients
   c : double;
}

-- FFT plans data
local fftw_plan = terralib.includec("fftw3.h").fftw_plan
local cufftHandle = int
if USE_CUDA then
   cufftHandle =  terralib.includec("cufft.h").cufftHandle
end
local fspace fftPlansType {
   -- FFTW plan for direct transform
   fftw_fwd : fftw_plan,
   -- FFTW plan for inverse transform
   fftw_bwd : fftw_plan,
   -- cuFFT plan
   cufft : cufftHandle,
   -- processor index
   id : C.legion_address_space_t,
}

function Exports.mkDataList()
   return {
      -- Data region
      fft = regentlib.newsymbol(),
      -- FFT plans
      plans = regentlib.newsymbol(),
      -- Data partitions
      Fluid_slubs  = regentlib.newsymbol("Poisson_Fluid_slubs"),
      Fluid_planes = regentlib.newsymbol("Poisson_Fluid_planes"),
      fft_slubs  = regentlib.newsymbol("Poisson_fft_slubs"),
      fft_planes = regentlib.newsymbol("Poisson_fft_planes"),
      fft_yNeg = regentlib.newsymbol("Poisson_fft_yNeg"),
      fft_yPos = regentlib.newsymbol("Poisson_fft_yPos"),
      plans_p = regentlib.newsymbol("Poisson_plans"),
      -- Auxiliary regions and partitions
      Coeffs = regentlib.newsymbol(),
      k2X = regentlib.newsymbol(),
      k2Z = regentlib.newsymbol(),
   }
end

-------------------------------------------------------------------------------
-- PARTITIONING UTILS
-------------------------------------------------------------------------------

-- Partitions a 3D region into XZ slubs
-- TODO: Here we are kind of cheating. We are associating a 2d tiling to a 3d index space
-- so we can use the same functor in the mapper. We could improve this in the future.
local function mkGenerateXZslubs(name)
   local generateXZslubs
   local p = regentlib.newsymbol(name)
   __demand(__inline)
   task generateXZslubs(r : region(ispace(int3d), Fluid_columns),
                        cs : ispace(int3d),
                        halo : int3d,
                        offset : int3d)
      var my_nty = 1
      -- Number of tiles in the x and z directions
      var Ntiles = cs.volume
      var a = int(sqrt(Ntiles))+1
      var my_ntx : int
      for i=a, 0, -1 do
         if ((Ntiles % i) == 0) then
            my_ntx = i
            break
         end
      end
      var my_ntz = Ntiles/my_ntx

      -- Partition points
      var Nx = r.bounds.hi.x - 2*halo.x + 1; var ntx = cs.bounds.hi.x + 1
      var Ny = r.bounds.hi.y - 2*halo.y + 1; var nty = cs.bounds.hi.y + 1
      var Nz = r.bounds.hi.z - 2*halo.z + 1; var ntz = cs.bounds.hi.z + 1
      regentlib.assert(r.bounds.lo == int3d{0,0,0}, "Can only partition root region")
   --   regentlib.assert(Nx % ntx == 0, "Uneven partitioning on x")
   --   regentlib.assert(Ny % nty == 0, "Uneven partitioning on y")
   --   regentlib.assert(Nz % ntz == 0, "Uneven partitioning on z")
      var modP = int3d({Nx % my_ntx, Ny % my_nty, Nz % my_ntz})
      regentlib.assert(-my_ntx <= offset.x and offset.x <= my_ntx, "offset.x too large")
      regentlib.assert(-my_nty <= offset.y and offset.y <= my_nty, "offset.y too large")
      regentlib.assert(-my_ntz <= offset.z and offset.z <= my_ntz, "offset.z too large")
      var coloring = regentlib.c.legion_domain_point_coloring_create()
      for c_real in cs do
         var c = (c_real - offset + {ntx,nty,ntz}) % {ntx,nty,ntz}
         -- project the 3d index space into a 2d index space
         var ind = ((c.z*nty+c.y)*ntx+c.x)
         var c2d   = int3d({ind%my_ntx, (ind/my_ntx)%my_nty, ind/(my_ntx*my_nty)})
         -- define the partition
         var rect = rect3d{
           lo = int3d{halo.x + (Nx/my_ntx)*(c2d.x  ) + min(c2d.x  , modP.x),
                      halo.y + (Ny/my_nty)*(c2d.y  ) + min(c2d.y  , modP.y),
                      halo.z + (Nz/my_ntz)*(c2d.z  ) + min(c2d.z  , modP.z)},
           hi = int3d{halo.x + (Nx/my_ntx)*(c2d.x+1) + min(c2d.x+1, modP.x) - 1,
                      halo.y + (Ny/my_nty)*(c2d.y+1) + min(c2d.y+1, modP.y) - 1,
                      halo.z + (Nz/my_ntz)*(c2d.z+1) + min(c2d.z+1, modP.z) - 1}}
         if c2d.x == 0 then rect.lo.x -= halo.x end
         if c2d.y == 0 then rect.lo.y -= halo.y end
         if c2d.z == 0 then rect.lo.z -= halo.z end
         if c2d.x == my_ntx-1 then rect.hi.x += halo.x end
         if c2d.y == my_nty-1 then rect.hi.y += halo.y end
         if c2d.z == my_ntz-1 then rect.hi.z += halo.z end
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, c_real, rect)
      end
      var [p] = partition(disjoint, r, coloring, cs)
      regentlib.c.legion_domain_point_coloring_destroy(coloring)
      return [p]
   end
   return generateXZslubs
end

-- Partitions a 3D region into Y planes
-- TODO: Here we are kind of cheating. We are associating a 2d tiling to a 3d index space
-- so we can use the same functor in the mapper. We could improve this in the future.
local function mkGenerateYplanes(name)
   local generateYplanes
   local p = regentlib.newsymbol(name)
   __demand(__inline)
   task generateYplanes(r : region(ispace(int3d), Fluid_columns),
                        cs : ispace(int3d),
                        halo : int3d,
                        offset : int3d)
      -- here we are only partitioning along y
      var my_ntx = 1
      var my_nty = cs.volume
      var my_ntz = 1

      -- Partition points
      var Nx = r.bounds.hi.x - 2*halo.x + 1; var ntx = cs.bounds.hi.x + 1
      var Ny = r.bounds.hi.y - 2*halo.y + 1; var nty = cs.bounds.hi.y + 1
      var Nz = r.bounds.hi.z - 2*halo.z + 1; var ntz = cs.bounds.hi.z + 1
      regentlib.assert(r.bounds.lo == int3d{0,0,0}, "Can only partition root region")
   --   regentlib.assert(Nx % ntx == 0, "Uneven partitioning on x")
   --   regentlib.assert(Ny % nty == 0, "Uneven partitioning on y")
   --   regentlib.assert(Nz % ntz == 0, "Uneven partitioning on z")
      var modP = int3d({Nx % my_ntx, Ny % my_nty, Nz % my_ntz})
      regentlib.assert(-my_ntx <= offset.x and offset.x <= my_ntx, "offset.x too large")
      regentlib.assert(-my_nty <= offset.y and offset.y <= my_nty, "offset.y too large")
      regentlib.assert(-my_ntz <= offset.z and offset.z <= my_ntz, "offset.z too large")
      var coloring = regentlib.c.legion_domain_point_coloring_create()
      for c_real in cs do
         var c = (c_real - offset + {ntx,nty,ntz}) % {ntx,nty,ntz}
         -- project the 3d index space into a 2d index space
         var ind = ((c.z*nty+c.y)*ntx+c.x)
         var c1d   = int3d({ind%my_ntx, (ind/my_ntx)%my_nty, ind/(my_ntx*my_nty)})
         -- define the partition
         var rect = rect3d{
           lo = int3d{halo.x + (Nx/my_ntx)*(c1d.x  ) + min(c1d.x  , modP.x),
                      halo.y + (Ny/my_nty)*(c1d.y  ) + min(c1d.y  , modP.y),
                      halo.z + (Nz/my_ntz)*(c1d.z  ) + min(c1d.z  , modP.z)},
           hi = int3d{halo.x + (Nx/my_ntx)*(c1d.x+1) + min(c1d.x+1, modP.x) - 1,
                      halo.y + (Ny/my_nty)*(c1d.y+1) + min(c1d.y+1, modP.y) - 1,
                      halo.z + (Nz/my_ntz)*(c1d.z+1) + min(c1d.z+1, modP.z) - 1}}
         -- Inlcude ghost data in the FFT transform
         if c1d.x == 0 then rect.lo.x -= halo.x end
         if c1d.y == 0 then rect.lo.y -= halo.y end
         if c1d.z == 0 then rect.lo.z -= halo.z end
         if c1d.x == my_ntx-1 then rect.hi.x += halo.x end
         if c1d.y == my_nty-1 then rect.hi.y += halo.y end
         if c1d.z == my_ntz-1 then rect.hi.z += halo.z end
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, c_real, rect)
      end
      var [p] = partition(disjoint, r, coloring, cs)
      regentlib.c.legion_domain_point_coloring_destroy(coloring)
      return [p]
   end
   return generateYplanes
end

-------------------------------------------------------------------------------
-- DECLARE SYMBOLS
-------------------------------------------------------------------------------

function Exports.DeclSymbols(DATA, Fluid, tiles, Fluid_Zones, Grid, config, MAPPER) return rquote

   -- Unpack the partitions that we are going to need
   var {p_All, yNeg, yPos, yNeg_ispace, yPos_ispace} = Fluid_Zones

   if (config.Efield.type == SCHEMA.EFieldStruct_Ybc) then
      ---------------------------------------------------------------------------
      -- Perform checks on the input
      ---------------------------------------------------------------------------
      -- BC periodic on X and Z
      regentlib.assert(config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic,
                       "Boundary conditions in the x direction must be periodic for this Poisson solver")
      regentlib.assert(config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic,
                       "Boundary conditions in the z direction must be periodic for this Poisson solver")
      -- Y BC cannot be periodic for now
      regentlib.assert(config.BC.yBCLeft.type ~= SCHEMA.FlowBC_Periodic,
                       "Boundary conditions in the y direction cannot be periodic for this Poisson solver")
      -- Uniform mesh on X and Z
      -- (User is responsible of ensuring this)
--      regentlib.assert(config.Grid.xType.type == SCHEMA.GridType_Uniform,
--                       "Computational mesh has to be uniform in the x direction")
--      regentlib.assert(config.Grid.zType.type == SCHEMA.GridType_Uniform,
--                       "Computational mesh has to be uniform in the z direction")
   end

   ---------------------------------------------------------------------------
   -- Create Regions and Partitions
   ---------------------------------------------------------------------------
   var sampleId = config.Mapping.sampleId

   -- Define FFT region
   var [DATA.fft] = region(Fluid.ispace, complex64);
   [UTIL.emitRegionTagAttach(DATA.fft, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

   -- Define data partitions for Poisson solver
   var [DATA.Fluid_slubs]  = [mkGenerateXZslubs("Poisson_Fluid_slubs")]
                             (Fluid, tiles, int3d{Grid.xBnum, Grid.yBnum, Grid.zBnum}, int3d{0,0,0})
   var [DATA.Fluid_planes] = [mkGenerateYplanes("Poisson_Fluid_planes")]
                             (Fluid, tiles, int3d{Grid.xBnum, Grid.yBnum, Grid.zBnum}, int3d{0,0,0});

   var [DATA.fft_slubs]  = DATA.fft & DATA.Fluid_slubs
   var [DATA.fft_planes] = DATA.fft & DATA.Fluid_planes;
   [UTIL.emitPartitionNameAttach(rexpr DATA.fft_slubs  end, "Poisson_fft_slubs" )];
   [UTIL.emitPartitionNameAttach(rexpr DATA.fft_planes end, "Poisson_fft_planes")];

   var [DATA.fft_yNeg] = (DATA.fft & yNeg)[0] & DATA.fft_slubs
   var [DATA.fft_yPos] = (DATA.fft & yPos)[0] & DATA.fft_slubs;
   [UTIL.emitPartitionNameAttach(rexpr DATA.fft_yNeg end, "Poisson_fft_yNeg")];
   [UTIL.emitPartitionNameAttach(rexpr DATA.fft_yPos end, "Poisson_fft_yPos")];

   -- Define plans for FFTs
   var [DATA.plans] = region(ispace(int1d, Grid.numTiles), fftPlansType);
   [UTIL.emitRegionTagAttach(DATA.plans, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   var [DATA.plans_p] = [UTIL.mkPartitionByTile(int1d, int3d, fftPlansType, "Poisson_plans")]
                        (DATA.plans, tiles, 0, int3d{0,0,0});

   -- Define region for tridiagonal coefficients
   var [DATA.Coeffs] = region(ispace(int1d, Fluid.bounds.hi.y+1, Fluid.bounds.lo.y), CoeffType);
   [UTIL.emitRegionTagAttach(DATA.Coeffs, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

   -- Define regions for squared complex wavenumbers
   var [DATA.k2X] = region(ispace(int1d, Fluid.bounds.hi.x+1, Fluid.bounds.lo.x), complex64)
   var [DATA.k2Z] = region(ispace(int1d, Fluid.bounds.hi.z+1, Fluid.bounds.lo.z), complex64);
   [UTIL.emitRegionTagAttach(DATA.k2X, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
   [UTIL.emitRegionTagAttach(DATA.k2Z, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

end end

-------------------------------------------------------------------------------
-- INITIALIZATION FUNCTIONS
-------------------------------------------------------------------------------

-- Fills the FFT plans
local __demand(__inline)
task fillFFTplans(r : region(ispace(int1d), fftPlansType))
where
   writes(r)
do
   fill(r.fftw_fwd, [fftw_plan](0))
   fill(r.fftw_bwd, [fftw_plan](0))
   fill(r.cufft,  [cufftHandle](0))
   fill(r.id, [C.legion_address_space_t](0))
end

-- Initializes the FFT plans
local extern task initFFTplans(r : region(ispace(int3d), Fluid_columns),
                               p : region(ispace(int1d), fftPlansType))
where
   reads writes(p)
end
initFFTplans:set_task_id(TYPES.TID_initFFTplans)

-- Initializes the coefficients for the tridiagonal system
local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA
task initCoefficients(r : region(ispace(int3d), Fluid_columns),
                      s : region(ispace(int1d), CoeffType),
                      Ng : int)
where
   reads(r.[mYFld]),
   reads(r.[mY_sFld]),
   reads(r.[nType]),
   writes(s)
do
   __demand(__openmp)
   for c in s do
      var cr    = int3d{0, int(c), 0}
      var cr_m1 = cr - int3d{0, 1, 0}
      if int(c) >= Ng then
         s[c].a =   r[cr_m1].[mY_sFld]                      *r[cr].[mYFld]
         s[c].b = -(r[cr   ].[mY_sFld] + r[cr_m1].[mY_sFld])*r[cr].[mYFld]
         s[c].c =   r[cr   ].[mY_sFld]                      *r[cr].[mYFld]
      end
   end

   -- Add BCs
   if     (r[int3d{0, s.bounds.lo, 0}].[nType] == L_S_node) then
      s[s.bounds.lo].a = 0.0
      s[s.bounds.lo].b = 1.0
      s[s.bounds.lo].c = 0.0
   elseif (r[int3d{0, s.bounds.lo, 0}].[nType] == L_C_node) then
      s[s.bounds.lo].a = 0.0
      s[s.bounds.lo].b = 0.5
      s[s.bounds.lo].c = 0.5
   end

   if     (r[int3d{0, s.bounds.hi, 0}].[nType] == R_S_node) then
      s[s.bounds.hi].a = 0.0
      s[s.bounds.hi].b = 1.0
      s[s.bounds.hi].c = 0.0
   elseif (r[int3d{0, s.bounds.hi, 0}].[nType] == R_C_node) then
      s[s.bounds.hi].a = 0.5
      s[s.bounds.hi].b = 0.5
      s[s.bounds.hi].c = 0.0
   end
end

local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task PrintCoefficients(c : region(ispace(int1d), CoeffType))
where
   reads(c.{a, b, c})
do
   C.printf("Coefficients:\n")
   for i in c do
      C.printf("%d %10.4e %10.4e %10.4e\n", i, c[i].a, c[i].b, c[i].c)
   end
end

-- Initializes the squared complex wavenumbers
local function mkInitWaveNumbers(sdir)
   local InitWaveNumbers
   local m
   local mkcr
   if sdir == "x" then
      m = mXFld
      mkcr = function(c, r) return rexpr int3d({       int(c), r.bounds.lo.y, r.bounds.hi.z}) end end
   elseif sdir == "z" then
      m = mZFld
      mkcr = function(c, r) return rexpr int3d({r.bounds.lo.x, r.bounds.lo.y,        int(c)}) end end
   else assert(false) end

   __demand(__leaf, __cuda) -- MANUALLY PARALLELIZED
   task InitWaveNumbers(k : region(ispace(int1d), complex64),
                        r : region(ispace(int3d), Fluid_columns))
   where
      reads(r.[m]),
      reads writes(k)
   do
      var N = k.ispace.volume
      for c in k do
         var cr = [mkcr(c, r)];
         var theta = double(c - ((c*2-1)/N)*N)*2.0*PI/N
         var a1 = complex64({cos( theta), sin( theta)})
         var a2 = complex64({cos(-theta), sin(-theta)})
         k[c] = (a1 + a2 - 2.0)*r[cr].[m]*r[cr].[m]
      end
   end
   return InitWaveNumbers
end

function Exports.Init(DATA, tiles, Grid, config) return rquote
   if (config.Efield.type == SCHEMA.EFieldStruct_Ybc) then
      -- Init fft data
      fill([DATA.fft], complex64{0.0, 0.0})

      -- Init plans for FFTs
      fillFFTplans(DATA.plans)
      __demand(__index_launch)
      for c in tiles do
         initFFTplans(DATA.Fluid_planes[c], DATA.plans_p[c])
      end

      -- Init tridiagonal coefficients
      fill([DATA.Coeffs].a, 0.0)
      fill([DATA.Coeffs].b, 0.0)
      fill([DATA.Coeffs].c, 0.0)
      initCoefficients(DATA.Fluid_slubs[tiles.bounds.lo], DATA.Coeffs, Grid.yBnum)
      --PrintCoefficients(DATA.Coeffs)

      -- Init regions for squared complex wavenumbers
      fill([DATA.k2X], complex64{0.0, 0.0})
      fill([DATA.k2Z], complex64{0.0, 0.0});
      [mkInitWaveNumbers("x")](DATA.k2X, DATA.Fluid_planes[tiles.bounds.lo]);
      [mkInitWaveNumbers("z")](DATA.k2Z, DATA.Fluid_planes[tiles.bounds.lo]);
   end
end end

-------------------------------------------------------------------------------
-- CLEANUP FUNCTIONS
-------------------------------------------------------------------------------

local extern task destroyFFTplans(p : region(ispace(int1d), fftPlansType))
where
   reads writes(p)
end
destroyFFTplans:set_task_id(TYPES.TID_destroyFFTplans)

function Exports.Cleanup(DATA, tiles, config) return rquote
   if (config.Efield.type == SCHEMA.EFieldStruct_Ybc) then
      -- Destroy plans for FFTs
      __demand(__index_launch)
      for c in tiles do
         destroyFFTplans(DATA.plans_p[c])
      end
   end
end end

-------------------------------------------------------------------------------
-- POISSON ROUTINES
-------------------------------------------------------------------------------

local __demand(__leaf, __cuda) -- MANUALLY PARALLELIZED
task setFFTBCs(fft : region(ispace(int3d), complex64),
               bc : double)
where
   writes(fft)
do
   __demand(__openmp)
   for c in fft do
      if ((c.x == 0) and (c.z == 0)) then
         -- this is the 0 wave number
         fft[c] = complex64{ bc, 0.0}
      else
         fft[c] = complex64{0.0, 0.0}
      end
   end
end

local extern task performDirFFT(r : region(ispace(int3d), Fluid_columns),
                                s : region(ispace(int3d), complex64),
                                p : region(ispace(int1d), fftPlansType),
                                Mix : MIX.Mixture)
where
   reads(r.[srcFlds]),
   reads(p),
   writes(s)
end
performDirFFT:set_task_id(TID_DirFFT)

local extern task solveTridiagonals(r   : region(ispace(int3d), complex64),
                                    c   : region(ispace(int1d), CoeffType),
                                    k2X : region(ispace(int1d), complex64),
                                    k2Z : region(ispace(int1d), complex64),
                                    Robin_bc : bool)
where
   reads writes(r),
   reads(c.{a, b, c}),
   reads(k2X, k2Z)
end
solveTridiagonals:set_task_id(TYPES.TID_solveTridiagonals)

local extern task performInvFFT(r : region(ispace(int3d), Fluid_columns),
                                s : region(ispace(int3d), complex64),
                                p : region(ispace(int1d), fftPlansType))
where
   reads(p),
   reads writes(s),
   writes(r.[outFld])
end
performInvFFT:set_task_id(TYPES.TID_performInvFFT)

function Exports.Solve(DATA, tiles, Mix, config) return rquote
   if (config.Efield.type == SCHEMA.EFieldStruct_Ybc) then
      -- Perform planar FFTs
      __demand(__index_launch)
      for c in tiles do
         performDirFFT(DATA.Fluid_planes[c], DATA.fft_planes[c], DATA.plans_p[c], Mix)
      end

      -- Set electric potential bcs
      __demand(__index_launch)
      for c in tiles do
         setFFTBCs(DATA.fft_yNeg[c], config.Efield.u.Ybc.Phi_bottom)
      end
      __demand(__index_launch)
      for c in tiles do
         setFFTBCs(DATA.fft_yPos[c], config.Efield.u.Ybc.Phi_top)
      end

      -- Solve the tridiagonal
      __demand(__index_launch)
      for c in tiles do
         solveTridiagonals(DATA.fft_slubs[c], DATA.Coeffs, DATA.k2X, DATA.k2Z,
                           config.Efield.u.Ybc.Robin_bc)
      end

      -- Perform planar inverse FFTs
      __demand(__index_launch)
      for c in tiles do
         performInvFFT(DATA.Fluid_planes[c], DATA.fft_planes[c], DATA.plans_p[c])
      end
   end
end end

return Exports end

