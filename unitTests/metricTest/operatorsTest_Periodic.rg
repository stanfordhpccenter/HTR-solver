import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local UTIL = require 'util-desugared'
local CONST = require "prometeo_const"

-- Node types
local Std_node = CONST.Std_node

-- Stencil indices
local Stencil1  = CONST.Stencil1
local Stencil2  = CONST.Stencil2
local Stencil3  = CONST.Stencil3
local Stencil4  = CONST.Stencil4
local nStencils = CONST.nStencils

local struct Fluid_columns {
   -- Grid point
   centerCoordinates : double[3];
   cellWidth : double[3];
      -- Node types
   nType_x : int;
   nType_y : int;
   nType_z : int;
   -- Cell-center metrics for Euler fluxes
   dcsi_e : double;
   deta_e : double;
   dzet_e : double;
   -- Cell-center metrics for diffusion fluxes
   dcsi_d : double;
   deta_d : double;
   dzet_d : double;
   -- Staggered metrics
   dcsi_s : double;
   deta_s : double;
   dzet_s : double;
}

--External modules
local MACRO = require "prometeo_macro"
local METRIC = (require 'prometeo_metric')(SCHEMA, Fluid_columns)

-- Test parameters
local Npx = 32
local Npy = 32
local Npz = 32
local Nx = 2
local Ny = 2
local Nz = 2
local xO = 0.0
local yO = 0.0
local zO = 0.0
local xW = 1.0
local yW = 1.0
local zW = 1.0

function checkInternal(r, c, b, sdir)
   local dir
   local nType
   local mk_cm2
   local mk_cm1
   local mk_cp1
   local mk_cp2
   local mk_cp3
   if sdir == "x" then
      dir = 0
      nType = "nType_x"
      mk_cm2 =  rexpr (c+{-2, 0, 0}) % b end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) % b end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) % b end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) % b end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) % b end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) % b end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) % b end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) % b end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) % b end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) % b end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) % b end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) % b end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) % b end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) % b end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) % b end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end, b)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end, b)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end, b)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end, b)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end, b)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] == [mk_cm2], ["operatorsTest: error in Internal Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] == [mk_cm1], ["operatorsTest: error in Internal Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] == [mk_cp1], ["operatorsTest: error in Internal Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] == [mk_cp2], ["operatorsTest: error in Internal Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] == [mk_cp3], ["operatorsTest: error in Internal Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == -1.0/6.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] ==  5.0/6.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] ==  2.0/6.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] ==  2.0/6.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] ==  5.0/6.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == -1.0/6.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] ==  2.0/6.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] == -7.0/6.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] == 11.0/6.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] ==       0.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] ==       0.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] ==  3.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] == 13.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] == -5.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] ==  1.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] ==  2.0/6.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] ==  5.0/6.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == -1.0/6.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == -1.0/6.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] ==  5.0/6.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] ==  2.0/6.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] ==      0.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] == 11.0/6.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] == -7.0/6.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] ==  2.0/6.0, ["operatorsTest: error in Internal Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] ==  1.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] == -5.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] == 13.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] ==  3.0/12.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] ==       0.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] ==       0.0, ["operatorsTest: error in Internal Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 9.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 6.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 1.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 4.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 9.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 6.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 1.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 4.0/20.0, ["operatorsTest: error in Internal Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == Std_node, ["operatorsTest: error in node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in Internal Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in Internal Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in Internal Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in Internal Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 3, ["operatorsTest: error in Internal Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] ==   3.0/4.0, ["operatorsTest: error in Internal Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == -3.0/20.0, ["operatorsTest: error in Internal Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] ==  1.0/60.0, ["operatorsTest: error in Internal Operators on KennedyCoeff " .. sdir])
   end
end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
   fill(Fluid.cellWidth, array(0.0, 0.0, 0.0))
   fill(Fluid.nType_x, 0)
   fill(Fluid.nType_y, 0)
   fill(Fluid.nType_z, 0)
   fill(Fluid.dcsi_e, 0.0)
   fill(Fluid.deta_e, 0.0)
   fill(Fluid.dzet_e, 0.0)
   fill(Fluid.dcsi_d, 0.0)
   fill(Fluid.deta_d, 0.0)
   fill(Fluid.dzet_d, 0.0)
   fill(Fluid.dcsi_s, 0.0)
   fill(Fluid.deta_s, 0.0)
   fill(Fluid.dzet_s, 0.0)
   fill(Fluid.dcsi_s, 0.0)
   fill(Fluid.deta_s, 0.0)
   fill(Fluid.dzet_s, 0.0)
end

__demand(__inline)
task checkOperators(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{nType_x, nType_y, nType_z})
do
   for c in Fluid do
      -- Check x-direction
      [checkInternal(rexpr Fluid end, rexpr c end, rexpr Fluid.bounds end, "x")];
      -- Check y-direction
      [checkInternal(rexpr Fluid end, rexpr c end, rexpr Fluid.bounds end, "y")];
      -- Check z-direction
      [checkInternal(rexpr Fluid end, rexpr c end, rexpr Fluid.bounds end, "z")];
   end
end

task main()

   C.printf("operatorsTest_Periodic: run...\n")

   -- No ghost cells
   var xBnum = 0
   var yBnum = 0
   var zBnum = 0

   -- Define the domain
   var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                 y = Npy + 2*yBnum,
                                 z = Npz + 2*zBnum})
   var Fluid = region(is_Fluid, Fluid_columns)
   var Fluid_bounds = Fluid.bounds

   -- Partitioning domain
   var tiles = ispace(int3d, {Nx, Ny, Nz})

   -- Fluid Partitioning
   var p_Fluid =
      [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_All")]
      (Fluid, tiles, int3d{xBnum,yBnum,zBnum}, int3d{0,0,0})

   InitializeCell(Fluid)

   METRIC.InitializeOperators(Fluid, tiles, p_Fluid)

   checkOperators(Fluid)

   __fence(__execution, __block)

   C.printf("operatorsTest_Periodic: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "operatorsTest_Periodic.o", "object")
