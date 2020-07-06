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

__demand(__inline)
task checkMetric(Fluid : region(ispace(int3d), Fluid_columns))
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
      [checkInternal(rexpr Fluid end, rexpr c end, "x")];
      -- Check y-direction
      [checkInternal(rexpr Fluid end, rexpr c end, "y")];
      -- Check z-direction
      [checkInternal(rexpr Fluid end, rexpr c end, "z")];
   end
end

task main()

   C.printf("metricTest_Periodic: run...\n")

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

      METRIC.InitializeMetric(Fluid,
                              Fluid_bounds,
                              xBnum, Npx,
                              yBnum, Npy,
                              zBnum, Npz)

   end
   checkMetric(Fluid)

   __fence(__execution, __block)

   C.printf("metricTest_Periodic: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "metricTest_Periodic.o", "object")
