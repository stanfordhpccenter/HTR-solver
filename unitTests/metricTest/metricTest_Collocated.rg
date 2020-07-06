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
}

--External modules
local MACRO = require "prometeo_macro"
local GRID = (require 'prometeo_grid')(SCHEMA, Fluid_columns)
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

function checkInternal(r, c, sdir)
   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end
   return rquote
      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+1]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+0]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+1]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+2]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+3]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+4]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+5]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+0]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+1]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+2]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+3]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+4]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+5]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconPlus])
      -- Minus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+1]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+0]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+1]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+2]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+3]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+4]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+5]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+0]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+1]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+2]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+3]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+4]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+5]           )      ) < 1e-8, ["metricTest: error in Internal Metric on " .. reconMinus])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil1]/(9.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil2]/(6.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil3]/(1.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil4]/(4.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsPlus])
      -- Minus side
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil1]/(9.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil2]/(6.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil3]/(1.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil4]/(4.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. TENOCoeffsMinus])
      -- BC flag
      regentlib.assert(r[c].[BCStencil] == false, ["metricTest: error in Internal Metric on " .. BCStencil])
      -- Face interpolation operator
      regentlib.assert(fabs((r[c].[interp][0]*2.0) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. interp])
      regentlib.assert(fabs((r[c].[interp][1]*2.0) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. interp])
      -- Face derivative operator
      regentlib.assert(fabs((r[c].[deriv]*r[c].cellWidth[dir]) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. deriv])
      -- Gradient operator
      regentlib.assert(fabs((r[c].[grad][0]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. grad])
      regentlib.assert(fabs((r[c].[grad][1]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in Internal Metric on " .. grad])
   end
end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
   fill(Fluid.cellWidth, array(0.0, 0.0, 0.0))
   fill(Fluid.reconXFacePlus,  [UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end)])
   fill(Fluid.reconYFacePlus,  [UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end)])
   fill(Fluid.reconZFacePlus,  [UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end)])
   fill(Fluid.reconXFaceMinus, [UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end)])
   fill(Fluid.reconYFaceMinus, [UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end)])
   fill(Fluid.reconZFaceMinus, [UTIL.mkArrayConstant(nStencils*6, rexpr 0.0 end)])
   fill(Fluid.TENOCoeffsXPlus,  [UTIL.mkArrayConstant(nStencils, rexpr 0.0 end)])
   fill(Fluid.TENOCoeffsYPlus,  [UTIL.mkArrayConstant(nStencils, rexpr 0.0 end)])
   fill(Fluid.TENOCoeffsZPlus,  [UTIL.mkArrayConstant(nStencils, rexpr 0.0 end)])
   fill(Fluid.TENOCoeffsXMinus, [UTIL.mkArrayConstant(nStencils, rexpr 0.0 end)])
   fill(Fluid.TENOCoeffsYMinus, [UTIL.mkArrayConstant(nStencils, rexpr 0.0 end)])
   fill(Fluid.TENOCoeffsZMinus, [UTIL.mkArrayConstant(nStencils, rexpr 0.0 end)])
   fill(Fluid.BCStencilX, false)
   fill(Fluid.BCStencilY, false)
   fill(Fluid.BCStencilZ, false)
   fill(Fluid.interpXFace, array(0.0, 0.0))
   fill(Fluid.interpYFace, array(0.0, 0.0))
   fill(Fluid.interpZFace, array(0.0, 0.0))
   fill(Fluid.derivXFace, 0.0)
   fill(Fluid.derivYFace, 0.0)
   fill(Fluid.derivZFace, 0.0)
   fill(Fluid.gradX, array(0.0, 0.0))
   fill(Fluid.gradY, array(0.0, 0.0))
   fill(Fluid.gradZ, array(0.0, 0.0))
end

local function checkLeft(r, c, sdir)
   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end
   return rquote
      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+1]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+2]           ) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+3]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+0]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+1]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+2]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+3]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+5]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+0]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+1]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+2]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+3]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+4]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+5]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconPlus])
      -- Minus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+1]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+2]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+3]           ) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+0]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+1]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+2]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+3]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+4]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+5]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+0]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+1]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+2]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+3]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+5]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. reconMinus])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil1]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil2]           ) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil3]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. TENOCoeffsPlus])
      -- Minus side
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil1]           ) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil2]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil3]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil4]           )      ) < 1e-8, ["metricTest: error in Left Metric on " .. TENOCoeffsMinus])
      -- BC flag
      regentlib.assert(r[c].[BCStencil] == true, ["metricTest: error in Left Metric on " .. BCStencil])
      -- Face interpolation operator
      regentlib.assert(fabs((r[c].[interp][0]*2.0) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. interp])
      regentlib.assert(fabs((r[c].[interp][1]*2.0) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. interp])
      -- Face derivative operator
      regentlib.assert(fabs((r[c].[deriv]*r[c].cellWidth[dir]) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. deriv])
      -- Gradient operator
      regentlib.assert(fabs((r[c].[grad][0]                        )      ) < 1e-8, ["metricTest: error in Left Metric on " .. grad])
      regentlib.assert(fabs((r[c].[grad][1]*r[c].cellWidth[dir]    ) - 1.0) < 1e-3, ["metricTest: error in Left Metric on " .. grad])
   end
end

local function checkLeftPlusOne(r, c, sdir)
   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end
   return rquote
      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+1]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+0]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+1]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+2]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+3]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+4]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+5]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+0]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+1]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+2]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+3]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+4]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+5]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconPlus])
      -- Minus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+1]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+0]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+1]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+2]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+3]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+4]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+5]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+0]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+1]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+2]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+3]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+4]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+5]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. reconMinus])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil1]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil2]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil3]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil4]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsPlus])
      -- Minus side
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil1]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil2]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil3]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil4]           )      ) < 1e-8, ["metricTest: error in LeftPlusOne Metric on " .. TENOCoeffsMinus])
      -- BC flag
      regentlib.assert(r[c].[BCStencil] == true, ["metricTest: error in LeftPlusOne Metric on " .. BCStencil])
      -- Face interpolation operator
      regentlib.assert(fabs((r[c].[interp][0]*2.0) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. interp])
      regentlib.assert(fabs((r[c].[interp][1]*2.0) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. interp])
      -- Face derivative operator
      regentlib.assert(fabs((r[c].[deriv]*r[c].cellWidth[dir]) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. deriv])
      -- Gradient operator
      regentlib.assert(fabs((r[c].[grad][0]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. grad])
      regentlib.assert(fabs((r[c].[grad][1]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in LeftPlusOne Metric on " .. grad])
   end
end

local function checkRightMinusTwo(r, c, sdir)
   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end
   return rquote
      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+0]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+1]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+2]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+3]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+2]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+3]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+4]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+5]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconPlus])
      -- Minus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+2]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+3]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+4]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+5]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+0]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+1]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+2]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+3]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. reconMinus])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil1]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil2]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil3]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil4]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsPlus])
      -- Minus side
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil1]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil2]/(0.5     )) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil3]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil4]           )      ) < 1e-8, ["metricTest: error in RightMinusTwo Metric on " .. TENOCoeffsMinus])
      -- BC flag
      regentlib.assert(r[c].[BCStencil] == true, ["metricTest: error in RightMinusTwo Metric on " .. BCStencil])
      -- Face interpolation operator
      regentlib.assert(fabs((r[c].[interp][0]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. interp])
      regentlib.assert(fabs((r[c].[interp][1]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. interp])
      -- Face derivative operator
      regentlib.assert(fabs((r[c].[deriv]*r[c].cellWidth[dir]) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. deriv])
      -- Gradient operator
      regentlib.assert(fabs((r[c].[grad][0]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. grad])
      regentlib.assert(fabs((r[c].[grad][1]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusTwo Metric on " .. grad])
   end
end

local function checkRightMinusOne(r, c, sdir)
   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end
   return rquote
      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+2]           ) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+3]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+0]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+1]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+2]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+3]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+2]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+3]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+4]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+5]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconPlus])
      -- Minus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+2]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+3]           ) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+0]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+1]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+2]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+3]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+4]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+5]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+0]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+1]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+2]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+3]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+5]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. reconMinus])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil1]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil2]           ) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil3]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsPlus])
      -- Minus side
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil1]           ) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil2]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil3]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil4]           )      ) < 1e-8, ["metricTest: error in RightMinusOne Metric on " .. TENOCoeffsMinus])
      -- BC flag
      regentlib.assert(r[c].[BCStencil] == true, ["metricTest: error in RightMinusOne Metric on " .. BCStencil])
      -- Face interpolation operator
      regentlib.assert(fabs((r[c].[interp][0]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. interp])
      regentlib.assert(fabs((r[c].[interp][1]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. interp])
      -- Face derivative operator
      regentlib.assert(fabs((r[c].[deriv]*r[c].cellWidth[dir]) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. deriv])
      -- Gradient operator
      regentlib.assert(fabs((r[c].[grad][0]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. grad])
      regentlib.assert(fabs((r[c].[grad][1]*r[c].cellWidth[dir]*2.0) - 1.0) < 1e-3, ["metricTest: error in RightMinusOne Metric on " .. grad])
   end
end

local function checkRight(r, c, sdir)
   local dir
   local reconPlus
   local reconMinus
   local TENOCoeffsPlus
   local TENOCoeffsMinus
   local BCStencil
   local interp
   local deriv
   local grad
   if sdir == "x" then
      dir = 0
      reconPlus = "reconXFacePlus"
      reconMinus = "reconXFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsXPlus"
      TENOCoeffsMinus = "TENOCoeffsXMinus"
      BCStencil = "BCStencilX"
      interp = "interpXFace"
      deriv = "derivXFace"
      grad = "gradX"
   elseif sdir == "y" then
      dir = 1
      reconPlus = "reconYFacePlus"
      reconMinus = "reconYFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsYPlus"
      TENOCoeffsMinus = "TENOCoeffsYMinus"
      BCStencil = "BCStencilY"
      interp = "interpYFace"
      deriv = "derivYFace"
      grad = "gradY"
   elseif sdir == "z" then
      dir = 2
      reconPlus = "reconZFacePlus"
      reconMinus = "reconZFaceMinus"
      TENOCoeffsPlus = "TENOCoeffsZPlus"
      TENOCoeffsMinus = "TENOCoeffsZMinus"
      BCStencil = "BCStencilZ"
      interp = "interpZFace"
      deriv = "derivZFace"
      grad = "gradZ"
   end
   return rquote
      -- Face reconstruction operators
      -- Plus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+4]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+1]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+0]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+1]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+2]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+3]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+4]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil3*6+5]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+0]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+1]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+2]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+3]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+4]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      regentlib.assert(fabs((r[c].[reconPlus][Stencil4*6+5]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconPlus])
      -- Minus side
      -- Stencil1
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+0]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+1]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+2]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+3]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+4]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil1*6+5]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      -- Stencil2
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+0]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+1]/(-1.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+2]/( 5.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+3]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+4]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil2*6+5]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      -- Stencil3
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+0]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+1]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+2]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+3]/(11.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+4]/(-7.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil3*6+5]/( 2.0/6.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      -- Stencil4
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+0]/( 1.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+1]/(-5.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+2]/(13.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+3]/( 3.0/12.0))- 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+4]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      regentlib.assert(fabs((r[c].[reconMinus][Stencil4*6+5]           )      ) < 1e-8, ["metricTest: error in Right Metric on " .. reconMinus])
      -- TENO blending coefficients
      -- Plus side
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil1]/(9.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil2]/(6.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil3]/(1.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsPlus])
      regentlib.assert(fabs((r[c].[TENOCoeffsPlus][Stencil4]/(4.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsPlus])
      -- Minus side
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil1]/(9.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil2]/(6.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil3]/(1.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsMinus])
      regentlib.assert(fabs((r[c].[TENOCoeffsMinus][Stencil4]/(4.0/20.0)) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. TENOCoeffsMinus])
      -- BC flag
      regentlib.assert(r[c].[BCStencil] == false, ["metricTest: error in Right Metric on " .. BCStencil])
      -- Face interpolation operator
      regentlib.assert(fabs((r[c].[interp][0]*2.0) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. interp])
      regentlib.assert(fabs((r[c].[interp][1]*2.0) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. interp])
      -- Face derivative operator
      regentlib.assert(fabs((r[c].[deriv]*r[c].cellWidth[dir]) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. deriv])
      -- Gradient operator
      regentlib.assert(fabs((r[c].[grad][0]*r[c].cellWidth[dir]    ) - 1.0) < 1e-3, ["metricTest: error in Right Metric on " .. grad])
      regentlib.assert(fabs((r[c].[grad][1]                        )      ) < 1e-8, ["metricTest: error in Right Metric on " .. grad])
   end
end

local task checkMetric(Fluid : region(ispace(int3d), Fluid_columns))
where
reads(Fluid.centerCoordinates),
reads(Fluid.cellWidth),
reads(Fluid.{reconXFacePlus, reconXFaceMinus}),
reads(Fluid.{reconYFacePlus, reconYFaceMinus}),
reads(Fluid.{reconZFacePlus, reconZFaceMinus}),
reads(Fluid.{TENOCoeffsXPlus, TENOCoeffsXMinus}),
reads(Fluid.{TENOCoeffsYPlus, TENOCoeffsYMinus}),
reads(Fluid.{TENOCoeffsZPlus, TENOCoeffsZMinus}),
reads(Fluid.{BCStencilX, BCStencilY, BCStencilZ}),
reads(Fluid.{interpXFace, interpYFace, interpZFace}),
reads(Fluid.{ derivXFace,  derivYFace,  derivZFace}),
reads(Fluid.{  gradX,       gradY,       gradZ})

do
   for c in Fluid do
      -- Check x-direction
      if c.x == 0 then
         [checkLeft(rexpr Fluid end, rexpr c end, "x")];
      elseif c.x == 1 then
         [checkLeftPlusOne(rexpr Fluid end, rexpr c end, "x")];
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

   C.printf("metricTest_Collocated: run...")

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
   var p_Fluid =
      [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_All")]
      (Fluid, tiles, int3d{xBnum,yBnum,zBnum}, int3d{0,0,0})

   __parallelize_with
      tiles,
      disjoint(p_Fluid),
      complete(p_Fluid, Fluid)
   do

      InitializeCell(Fluid)

      __demand(__index_launch)
      for c in tiles do
         GRID.InitializeGeometry(p_Fluid[c],
                                 SCHEMA.GridType_Uniform, SCHEMA.GridType_Uniform, SCHEMA.GridType_Uniform,
                                 1.0, 1.0, 1.0,
                                 xBnum, Npx, xO, xW,
                                 yBnum, Npy, yO, yW,
                                 zBnum, Npz, zO, zW)
      end

      __demand(__index_launch)
      for c in tiles do
         GRID.InitializeGhostGeometry(p_Fluid[c],
                                      SCHEMA.GridType_Uniform, SCHEMA.GridType_Uniform, SCHEMA.GridType_Uniform,
                                      1.0, 1.0, 1.0,
                                      xBnum, Npx, xO, xW,
                                      yBnum, Npy, yO, yW,
                                      zBnum, Npz, zO, zW)
      end

      METRIC.InitializeMetric(Fluid,
                              Fluid_bounds,
                              xBnum, Npx,
                              yBnum, Npy,
                              zBnum, Npz);

      -- Enforce BCs on the metric
      [METRIC.mkCorrectGhostMetric("x")](Fluid,
                                         Fluid_bounds,
                                         SCHEMA.FlowBC_NSCBC_Inflow, SCHEMA.FlowBC_NSCBC_Inflow,
                                         xBnum, Npx);
      [METRIC.mkCorrectGhostMetric("y")](Fluid,
                                         Fluid_bounds,
                                         SCHEMA.FlowBC_NSCBC_Inflow, SCHEMA.FlowBC_NSCBC_Inflow,
                                         yBnum, Npy);
      [METRIC.mkCorrectGhostMetric("z")](Fluid,
                                         Fluid_bounds,
                                         SCHEMA.FlowBC_NSCBC_Inflow, SCHEMA.FlowBC_NSCBC_Inflow,
                                         zBnum, Npz);

   end
   checkMetric(Fluid)

   __fence(__execution, __block)

   C.printf("metricTest_Collocated: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "metricTest_Collocated.o", "object")
