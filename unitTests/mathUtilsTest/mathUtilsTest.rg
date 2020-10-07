import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local cos = regentlib.cos(double)
local fabs = regentlib.fabs(double)
local MATH = require 'math_utils'
local PI = 3.1415926535898

task checkMatMul()
   var A = array( 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0)
   var b = array(1.0, 2.0, 3.0)
   var c = [MATH.mkMatMul(3)](b,A)
   regentlib.assert(c[0] == b[0], "mathUtilsTest: ERROR in checkMatMul")
   regentlib.assert(c[1] == b[1], "mathUtilsTest: ERROR in checkMatMul")
   regentlib.assert(c[2] == b[2], "mathUtilsTest: ERROR in checkMatMul")
end

LUdec, ludcmp, lubksb = unpack(MATH.mkLUdec(3))
task checkLUdec()
   var LU : LUdec
   LU.A = array(4.0, 3.0, 0.0,
                3.0, 4.0,-1.0,
                0.0,-1.0, 4.0)
   LU = ludcmp(LU)

   var b = array(24.0, 30.0,-24.0)
   var x = lubksb(LU, b)
   regentlib.assert(fabs(1.0 - x[0]/( 3.0)) < 1e-12, "mathUtilsTest: ERROR in checkLUdec")
   regentlib.assert(fabs(1.0 - x[1]/( 4.0)) < 1e-12, "mathUtilsTest: ERROR in checkLUdec")
   regentlib.assert(fabs(1.0 - x[2]/(-5.0)) < 1e-12, "mathUtilsTest: ERROR in checkLUdec")
end

--task checkInverseMatrix()
--   var A = array(4.0, 3.0, 0.0,
--                 3.0, 4.0,-1.0,
--                 0.0,-1.0, 4.0)
--   var Ainv = [MATH.mkInverseMatrix(3)](A)
--   regentlib.assert(fabs(1.0 - Ainv[0]/( 0.6250000000000000)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[1]/(-0.5000000000000000)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[2]/(-0.1250000000000000)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[3]/(-0.5000000000000000)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[4]/( 0.6666666666666667)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[5]/( 0.1666666666666667)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[6]/(-0.1250000000000000)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[7]/( 0.1666666666666667)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--   regentlib.assert(fabs(1.0 - Ainv[8]/( 0.2916666666666667)) < 1e-12, "mathUtilsTest: ERROR in checkInverseMatrix")
--end

task checkReconCoeff()
   var xf = array(-2.0, -1.0, 0.0, 1.0, 2.0)
   var c = [MATH.mkReconCoeff(4)](xf, 0.0)
   regentlib.assert(fabs(1.0 - c[0]/(-1.0/12)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeff")
   regentlib.assert(fabs(1.0 - c[1]/( 7.0/12)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeff")
   regentlib.assert(fabs(1.0 - c[2]/( 7.0/12)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeff")
   regentlib.assert(fabs(1.0 - c[3]/(-1.0/12)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeff")
end

task checkReconCoeffLeftBC()
   var xf = array(-1.0, 0.0, 1.0, 2.0)
   var c = [MATH.mkReconCoeffLeftBC(4)](xf, 0.0)
   regentlib.assert(fabs(1.0 - c[0]/(-12.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffLeftBC")
   regentlib.assert(fabs(1.0 - c[1]/( 34.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffLeftBC")
   regentlib.assert(fabs(1.0 - c[2]/( 16.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffLeftBC")
   regentlib.assert(fabs(1.0 - c[3]/(- 2.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffLeftBC")
end

task checkReconCoeffRightBC()
   var xf = array(-2.0, -1.0, 0.0, 1.0)
   var c = [MATH.mkReconCoeffRightBC(4)](xf, 0.0)
   regentlib.assert(fabs(1.0 - c[0]/(- 2.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffRightBC")
   regentlib.assert(fabs(1.0 - c[1]/( 16.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffRightBC")
   regentlib.assert(fabs(1.0 - c[2]/( 34.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffRightBC")
   regentlib.assert(fabs(1.0 - c[3]/(-12.0/36)) < 1e-12, "mathUtilsTest: ERROR in checkReconCoeffRightBC")
end

local struct columns {
   Conserved : double[3];
}

local struct dummy {
   aa : int;
}

local ImplicitVars = terralib.newlist({
   "Conserved"
})

-- RHS function for the implicit solver
local __demand(__inline)
task func(  r : region(ispace(int3d), columns),
            c : int3d,
          dum : dummy)
where
   reads writes(r.[ImplicitVars])
do
   return array(-0.013* r[c].Conserved[0] - 1000.0*r[c].Conserved[0]*r[c].Conserved[2],
                -2500.0*r[c].Conserved[1]*r[c].Conserved[2],
                -0.013* r[c].Conserved[0] - 1000.0*r[c].Conserved[0]*r[c].Conserved[2] - 2500.0*r[c].Conserved[1]*r[c].Conserved[2])
end

task checkRosenbrock()
   var dum : dummy
   var r = region(ispace(int3d,{1,1,1}), columns)
   fill(r.Conserved, array(1.0, 1.0, 0.0))
   for c in r do
      var err = [MATH.mkRosenbrock(3, columns, ImplicitVars, "Conserved", dummy, func)]
         (r, c, 1.0e-3, 25.0, dum)
      regentlib.assert(err==0, "mathUtilsTest: ERROR in Rosenbrock returned an error")
      regentlib.assert(fabs(1.0 - r[c].Conserved[0]/( 7.818640e-01)) < 1e-6, "mathUtilsTest: ERROR in checkRosenbrock")
      regentlib.assert(fabs(1.0 - r[c].Conserved[1]/( 1.218133e+00)) < 1e-6, "mathUtilsTest: ERROR in checkRosenbrock")
      regentlib.assert(fabs(1.0 - r[c].Conserved[2]/(-2.655799e-06)) < 1e-6, "mathUtilsTest: ERROR in checkRosenbrock")
   end
end

local struct SrcInterpType {
   y : double;
   x : double;
}
FIData, FIType,
FIInitData, FIInitRegion,
FIFindIndex, FIGetWeight = unpack(MATH.mkFastInterp(SrcInterpType, "x"))
local NFastInterp = 50
local LFastInterp = rexpr 1.0 end
task checkFastInterp()
   var src = region(ispace(int1d, NFastInterp), SrcInterpType)
   -- Fill the source region
   for c in src do
      src[c].x = [LFastInterp]*0.5*(cos(PI*(double(c)/(NFastInterp-1)+1.0))+1.0)
      src[c].y = src[c].x/LFastInterp
   end

   -- Init fast interpolation
   var FIdata = FIInitData(src)
   var FIRegion = region(ispace(int1d, FIdata.nloc), FIType)
   FIInitRegion(FIRegion, src, FIdata)

   -- Test
   var testvals = array(0.0, 0.3389, 0.54545, 0.7898, 1.0)
   for i=0, 5 do
      var x = testvals[i]
      var ind = FIFindIndex(x, FIRegion, FIdata)
      var w = FIGetWeight(x, src[ind].x, src[ind+int1d(1)].x)
      var y = src[ind].y*w + src[ind+int1d(1)].y*(1.0-w)
      regentlib.assert(((x >= src[ind].x) and (x <= src[ind+int1d(1)].x)), "mathUtilsTest: ERROR in checkFastInterp FindIndex")
      regentlib.assert(fabs(y - x) < 1e-12, "mathUtilsTest: ERROR in checkFastInterp GetWeight")
   end
end

task main()
   -- Check MatMul
   checkMatMul()
   -- Check LUdec
   checkLUdec()
--   -- Check InverseMatrix
--   checkInverseMatrix()
   -- Check ReconCoeff
   checkReconCoeff()
   -- Check ReconCoeffLeftBC
   checkReconCoeffLeftBC()
   -- Check ReconCoeffRightBC
   checkReconCoeffRightBC()
   -- Check Rosenbrock
   checkRosenbrock()
   -- Check FastInterp
   checkFastInterp()

   __fence(__execution, __block)

   C.printf("mathUtilsTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "mathUtilsTest.o", "object")
