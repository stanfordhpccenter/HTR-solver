import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local UTIL = require 'util'
local CONST = require "prometeo_const"

-- Node types
local Std_node   = CONST.Std_node
local L_S_node   = CONST.L_S_node
local Lp1_S_node = CONST.Lp1_S_node
local Lp2_S_node = CONST.Lp2_S_node
local Rm3_S_node = CONST.Rm3_S_node
local Rm2_S_node = CONST.Rm2_S_node
local Rm1_S_node = CONST.Rm1_S_node
local R_S_node   = CONST.R_S_node

-- Stencil indices
local Stencil1  = CONST.Stencil1
local Stencil2  = CONST.Stencil2
local Stencil3  = CONST.Stencil3
local Stencil4  = CONST.Stencil4
local nStencils = CONST.nStencils

-------------------------------------------------------------------------------
-- ACTIVATE ELECTRIC FIELD SOLVER
-------------------------------------------------------------------------------

local ELECTRIC_FIELD = false
if os.getenv("ELECTRIC_FIELD") == "1" then
   ELECTRIC_FIELD = true
   print("#############################################################################")
   print("WARNING: You are compiling with electric field solver.")
   print("#############################################################################")
end

local types_inc_flags = terralib.newlist({"-DEOS="..os.getenv("EOS")})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns

--External modules
local MACRO = require "prometeo_macro"
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)

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

function checkInternal(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
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
end

local function checkLeft(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] == int3d([c]), ["operatorsTest: error in Left Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] == int3d([c]), ["operatorsTest: error in Left Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] ==   [mk_cp1], ["operatorsTest: error in Left Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] ==   [mk_cp1], ["operatorsTest: error in Left Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] ==   [mk_cp1], ["operatorsTest: error in Left Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] == 1.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] == 0.0, ["operatorsTest: error in Left Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 9.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 6.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 1.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 4.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 9.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 6.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 1.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 4.0/20.0, ["operatorsTest: error in Left Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == L_S_node, ["operatorsTest: error in Left Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 1.0, ["operatorsTest: error in Left Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in Left Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 0.0, ["operatorsTest: error in Left Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 2.0, ["operatorsTest: error in Left Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 0, ["operatorsTest: error in Left Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] == 0.0, ["operatorsTest: error in Left Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in Left Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] == 0.0, ["operatorsTest: error in Left Operators on KennedyCoeff " .. sdir])
   end
end

local function checkLeftPlusOne(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] ==   [mk_cm1], ["operatorsTest: error in LeftPlusOne Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] ==   [mk_cm1], ["operatorsTest: error in LeftPlusOne Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] ==   [mk_cp1], ["operatorsTest: error in LeftPlusOne Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] ==   [mk_cp2], ["operatorsTest: error in LeftPlusOne Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] ==   [mk_cp3], ["operatorsTest: error in LeftPlusOne Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == -2.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] ==  5.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] ==  1.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] ==  5.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == -1.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] ==       0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] ==       0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] ==  3.0/12.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] == 13.0/12.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] == -5.0/12.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] ==  1.0/12.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] ==  5.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == -1.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == -2.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] ==  5.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] ==  1.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] == 11.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] == -7.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 2.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 1.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] ==     0.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 1.0/4.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 7.0/16.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 8.0/16.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 1.0/16.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] ==      0.0, ["operatorsTest: error in LeftPlusOne Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == Lp1_S_node, ["operatorsTest: error in LeftPlusOne Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in LeftPlusOne Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in LeftPlusOne Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 1.0, ["operatorsTest: error in LeftPlusOne Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in LeftPlusOne Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 1, ["operatorsTest: error in LeftPlusOne Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in LeftPlusOne Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in LeftPlusOne Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] == 0.0, ["operatorsTest: error in LeftPlusOne Operators on KennedyCoeff " .. sdir])
   end
end

local function checkLeftPlusTwo(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] == [mk_cm2], ["operatorsTest: error in LeftPlusTwo Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] == [mk_cm1], ["operatorsTest: error in LeftPlusTwo Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] == [mk_cp1], ["operatorsTest: error in LeftPlusTwo Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] == [mk_cp2], ["operatorsTest: error in LeftPlusTwo Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] == [mk_cp3], ["operatorsTest: error in LeftPlusTwo Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == -1.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] ==  5.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] ==  5.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == -1.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] ==      1.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] ==     -2.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] ==      2.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] ==       0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] ==       0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] ==  3.0/12.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] == 13.0/12.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] == -5.0/12.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] ==  1.0/12.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] ==  5.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == -1.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == -1.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] ==  5.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] == 11.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] == -7.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] ==  2.0/6.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] ==  3.0/9.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] == -7.0/9.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] == 11.0/9.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] ==  2.0/9.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] ==      0.0, ["operatorsTest: error in LeftPlusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 4.7000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 2.7000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 1.0000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 1.6000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 3.9000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 2.7000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 4.0000e-02, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 3.0000e-01, ["operatorsTest: error in LeftPlusTwo Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == Lp2_S_node, ["operatorsTest: error in LeftPlusTwo Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in LeftPlusTwo Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in LeftPlusTwo Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in LeftPlusTwo Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in LeftPlusTwo Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 2, ["operatorsTest: error in LeftPlusTwo Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] ==   2.0/3.0, ["operatorsTest: error in LeftPlusTwo Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == -1.0/12.0, ["operatorsTest: error in LeftPlusTwo Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] ==       0.0, ["operatorsTest: error in LeftPlusTwo Operators on KennedyCoeff " .. sdir])
   end
end

local function checkRightMinusThree(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] == [mk_cm2], ["operatorsTest: error in RightMinusThree Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] == [mk_cm1], ["operatorsTest: error in RightMinusThree Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] == [mk_cp1], ["operatorsTest: error in RightMinusThree Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] == [mk_cp2], ["operatorsTest: error in RightMinusThree Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] == [mk_cp3], ["operatorsTest: error in RightMinusThree Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == -1.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] ==  5.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] ==  2.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] ==  2.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] ==  5.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == -1.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] ==  2.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] == -7.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] == 11.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] ==  2.0/9.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] == 11.0/9.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] == -7.0/9.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] ==  3.0/9.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] ==  2.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] ==  5.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == -1.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == -1.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] ==  5.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] ==  2.0/6.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] ==      0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] ==      2.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] ==     -2.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] ==      1.0, ["operatorsTest: error in RightMinusThree Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] ==  1.0/12.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] == -5.0/12.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] == 13.0/12.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] ==  3.0/12.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] ==       0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] ==       0.0, ["operatorsTest: error in RightMinusThree Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 3.9000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 2.7000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 4.0000e-02, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 3.0000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 4.7000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 2.7000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 1.0000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 1.6000e-01, ["operatorsTest: error in RightMinusThree Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == Rm3_S_node, ["operatorsTest: error in RightMinusThree Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in RightMinusThree Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in RightMinusThree Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in RightMinusThree Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in RightMinusThree Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 2, ["operatorsTest: error in RightMinusThree Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] ==   2.0/3.0, ["operatorsTest: error in RightMinusThree Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == -1.0/12.0, ["operatorsTest: error in RightMinusThree Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] ==       0.0, ["operatorsTest: error in RightMinusThree Operators on KennedyCoeff " .. sdir])
   end
end

local function checkRightMinusTwo(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] == [mk_cm2], ["operatorsTest: error in RightMinusTwo Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] == [mk_cm1], ["operatorsTest: error in RightMinusTwo Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] == [mk_cp1], ["operatorsTest: error in RightMinusTwo Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] == [mk_cp2], ["operatorsTest: error in RightMinusTwo Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] == [mk_cp2], ["operatorsTest: error in RightMinusTwo Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == -1.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] ==  5.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] ==  2.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] ==  1.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] ==  5.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == -2.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] ==  2.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] == -7.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] == 11.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] ==  1.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] ==  5.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == -2.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == -1.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] ==  5.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] ==  2.0/6.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] ==  1.0/12.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] == -5.0/12.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] == 13.0/12.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] ==  3.0/12.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] ==       0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] ==       0.0, ["operatorsTest: error in RightMinusTwo Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 7.0/16.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 8.0/16.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 1.0/16.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] ==      0.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 2.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 1.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] ==     0.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 1.0/4.0, ["operatorsTest: error in RightMinusTwo Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == Rm2_S_node, ["operatorsTest: error in RightMinusTwo Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in RightMinusTwo Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in RightMinusTwo Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in RightMinusTwo Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in RightMinusTwo Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 1, ["operatorsTest: error in RightMinusTwo Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in RightMinusTwo Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in RightMinusTwo Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] == 0.0, ["operatorsTest: error in RightMinusTwo Operators on KennedyCoeff " .. sdir])
   end
end

local function checkRightMinusOne(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] == [mk_cm1], ["operatorsTest: error in RightMinusOne Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] == [mk_cm1], ["operatorsTest: error in RightMinusOne Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] == [mk_cp1], ["operatorsTest: error in RightMinusOne Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] == [mk_cp1], ["operatorsTest: error in RightMinusOne Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] == [mk_cp1], ["operatorsTest: error in RightMinusOne Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 9.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 6.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 1.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 4.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 9.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 6.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 1.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 4.0/20.0, ["operatorsTest: error in RightMinusOne Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == Rm1_S_node, ["operatorsTest: error in RightMinusOne Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in RightMinusOne Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 1.0, ["operatorsTest: error in RightMinusOne Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 0, ["operatorsTest: error in RightMinusOne Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] == 0.0, ["operatorsTest: error in RightMinusOne Operators on KennedyCoeff " .. sdir])
   end
end

local function checkRight(r, c, sdir)
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
      mk_cm2 =  rexpr (c+{-2, 0, 0}) end
      mk_cm1 =  rexpr (c+{-1, 0, 0}) end
      mk_cp1 =  rexpr (c+{ 1, 0, 0}) end
      mk_cp2 =  rexpr (c+{ 2, 0, 0}) end
      mk_cp3 =  rexpr (c+{ 3, 0, 0}) end
   elseif sdir == "y" then
      dir = 1
      nType = "nType_y"
      mk_cm2 =  rexpr (c+{ 0,-2, 0}) end
      mk_cm1 =  rexpr (c+{ 0,-1, 0}) end
      mk_cp1 =  rexpr (c+{ 0, 1, 0}) end
      mk_cp2 =  rexpr (c+{ 0, 2, 0}) end
      mk_cp3 =  rexpr (c+{ 0, 3, 0}) end
   elseif sdir == "z" then
      dir = 2
      nType = "nType_z"
      mk_cm2 =  rexpr (c+{ 0, 0,-2}) end
      mk_cm1 =  rexpr (c+{ 0, 0,-1}) end
      mk_cp1 =  rexpr (c+{ 0, 0, 1}) end
      mk_cp2 =  rexpr (c+{ 0, 0, 2}) end
      mk_cp3 =  rexpr (c+{ 0, 0, 3}) end
   end
   local cm2_d = METRIC.GetCm2(sdir, c, rexpr [r][c].[nType] end)
   local cm1_d = METRIC.GetCm1(sdir, c, rexpr [r][c].[nType] end)
   local cp1_d = METRIC.GetCp1(sdir, c, rexpr [r][c].[nType] end)
   local cp2_d = METRIC.GetCp2(sdir, c, rexpr [r][c].[nType] end)
   local cp3_d = METRIC.GetCp3(sdir, c, rexpr [r][c].[nType] end)
   return rquote
      -- Stencil indices
      regentlib.assert([cm2_d] ==   [mk_cm1], ["operatorsTest: error in Right Operators on cm2 " .. sdir])
      regentlib.assert([cm1_d] ==   [mk_cm1], ["operatorsTest: error in Right Operators on cm1 " .. sdir])
      regentlib.assert([cp1_d] == int3d([c]), ["operatorsTest: error in Right Operators on cp1 " .. sdir])
      regentlib.assert([cp2_d] == int3d([c]), ["operatorsTest: error in Right Operators on cp2 " .. sdir])
      regentlib.assert([cp3_d] == int3d([c]), ["operatorsTest: error in Right Operators on cp3 " .. sdir])

      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Plus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil2, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Plus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil3, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Plus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Plus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Plus(rexpr r[c].[nType] end, Stencil4, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Plus_" .. sdir])
      -- Minus side
      -- Stencil1
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil1, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil1 of Recon_Minus_" .. sdir])
      -- Stencil2
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil2, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil2 of Recon_Minus_" .. sdir])
      -- Stencil3
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil3, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil3 of Recon_Minus_" .. sdir])
      -- Stencil4
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 0)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 1)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 2)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 3)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 4)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Minus_" .. sdir])
      regentlib.assert([METRIC.GetRecon_Minus(rexpr r[c].[nType] end, Stencil4, 5)] == 0.0, ["operatorsTest: error in Right Operators on Stencil4 of Recon_Minus_" .. sdir])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil1)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil2)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil3)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Plus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Plus(rexpr r[c].[nType] end, Stencil4)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Plus_" .. sdir])
      -- Minus side
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil1)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil2)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil3)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Minus_" .. sdir])
      regentlib.assert([METRIC.GetCoeffs_Minus(rexpr r[c].[nType] end, Stencil4)] == 0.0, ["operatorsTest: error in Right Operators on Coeffs_Minus_" .. sdir])
      -- Node Type
      regentlib.assert(r[c].[nType] == R_S_node, ["operatorsTest: error in Right Operators on node type on " .. sdir])
      -- Staggered interpolation operator
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 0)] == 0.5, ["operatorsTest: error in Right Operators on Interp " .. sdir])
      regentlib.assert([METRIC.GetInterp(rexpr r[c].[nType] end, 1)] == 0.5, ["operatorsTest: error in Right Operators on Interp " .. sdir])
      -- Cell-center gradient operator
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 0)] == 2.0, ["operatorsTest: error in Right Operators on Grad " .. sdir])
      regentlib.assert([METRIC.GetGrad(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in Right Operators on Grad " .. sdir])
      -- Kennedy order
      regentlib.assert([METRIC.GetKennedyOrder(rexpr r[c].[nType] end)] == 0, ["operatorsTest: error in Right Operators on KennedyOrder " .. sdir])
      -- Kennedy coefficients
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 0)] == 0.0, ["operatorsTest: error in Right Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 1)] == 0.0, ["operatorsTest: error in Right Operators on KennedyCoeff " .. sdir])
      regentlib.assert([METRIC.GetKennedyCoeff(rexpr r[c].[nType] end, 2)] == 0.0, ["operatorsTest: error in Right Operators on KennedyCoeff " .. sdir])
   end
end

__demand(__inline)
task checkOperators(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{nType_x, nType_y, nType_z})
do
   for c in Fluid do
      -- Check x-direction
      if c.x == 0 then
         [checkLeft(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == 1 then
         [checkLeftPlusOne(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == 2 then
         [checkLeftPlusTwo(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == Npx-2 then
         [checkRightMinusThree(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == Npx-1 then
         [checkRightMinusTwo(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == Npx   then
         [checkRightMinusOne(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == Npx+1 then
         [checkRight(rexpr Fluid end, rexpr c end, "x")];
      else
         [checkInternal(rexpr Fluid end, rexpr c end, "x")];
      end
      -- Check y-direction
      if c.y == 0 then
         [checkLeft(rexpr Fluid end, rexpr c end, "y")];
      elseif c.y == 1 then
         [checkLeftPlusOne(rexpr Fluid end, rexpr c end, "y")];
      elseif c.y == 2 then
         [checkLeftPlusTwo(rexpr Fluid end, rexpr c end, "y")];
      elseif c.y == Npy-2 then
         [checkRightMinusThree(rexpr Fluid end, rexpr c end, "y")];
      elseif c.y == Npy-1 then
         [checkRightMinusTwo(rexpr Fluid end, rexpr c end, "y")];
      elseif c.y == Npy   then
         [checkRightMinusOne(rexpr Fluid end, rexpr c end, "y")];
      elseif c.y == Npy+1 then
         [checkRight(rexpr Fluid end, rexpr c end, "y")];
      else
         [checkInternal(rexpr Fluid end, rexpr c end, "y")];
      end
      -- Check z-direction
      if c.z == 0 then
         [checkLeft(rexpr Fluid end, rexpr c end, "z")];
      elseif c.z == 1 then
         [checkLeftPlusOne(rexpr Fluid end, rexpr c end, "z")];
      elseif c.z == 2 then
         [checkLeftPlusTwo(rexpr Fluid end, rexpr c end, "z")];
      elseif c.z == Npz-2 then
         [checkRightMinusThree(rexpr Fluid end, rexpr c end, "z")];
      elseif c.z == Npz-1 then
         [checkRightMinusTwo(rexpr Fluid end, rexpr c end, "z")];
      elseif c.z == Npz   then
         [checkRightMinusOne(rexpr Fluid end, rexpr c end, "z")];
      elseif c.z == Npz+1 then
         [checkRight(rexpr Fluid end, rexpr c end, "z")];
      else
         [checkInternal(rexpr Fluid end, rexpr c end, "z")];
      end
   end
end

task main()

   C.printf("operatorsTest_Staggered: run...")

   var config : SCHEMA.Config

   config.BC.xBCLeft.type  = SCHEMA.FlowBC_Dirichlet
   config.BC.xBCRight.type = SCHEMA.FlowBC_Dirichlet
   config.BC.yBCLeft.type  = SCHEMA.FlowBC_Dirichlet
   config.BC.yBCRight.type = SCHEMA.FlowBC_Dirichlet
   config.BC.zBCLeft.type  = SCHEMA.FlowBC_Dirichlet
   config.BC.zBCRight.type = SCHEMA.FlowBC_Dirichlet

   config.Grid.xNum = Npx
   config.Grid.yNum = Npy
   config.Grid.zNum = Npz

   -- No ghost cells
   var xBnum = 1
   var yBnum = 1
   var zBnum = 1

   -- Define the domain
   var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                 y = Npy + 2*yBnum,
                                 z = Npz + 2*zBnum})
   var Fluid = region(is_Fluid, Fluid_columns);
   var Fluid_bounds = Fluid.bounds

   -- Partitioning domain
   var tiles = ispace(int3d, {Nx, Ny, Nz})

   -- Fluid Partitioning
   var Fluid_Zones = PART.PartitionZones(Fluid, tiles, config, xBnum, yBnum, zBnum)

   InitializeCell(Fluid)

   METRIC.InitializeOperators(Fluid, tiles, Fluid_Zones, config,
                              xBnum, yBnum, zBnum)

   checkOperators(Fluid)

   __fence(__execution, __block)

   C.printf("operatorsTest_Staggered: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "operatorsTest_Staggered.o", "object")
