import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local UTIL = require 'util-desugared'

local struct columns {
   a : double;
   b : double[3];
}

local IOVars = terralib.newlist({
   "a",
   "b"
})

local NP = 32
local NT = 2

local HDF = (require 'hdf_helper')(int3d, int3d, columns,
                                                 IOVars,
                                                 {timeStep=int,simTime=double},
                                                 {})

local __demand(__inline)
task drand48_r(rngState : &C.drand48_data)
  var res : double
  C.drand48_r(rngState, &res)
  return res
end

task init(r0 : region(ispace(int3d), columns),
          r1 : region(ispace(int3d), columns))
where
   writes(r0.[IOVars]),
   writes(r1.[IOVars])
do
   var rngState : C.drand48_data
   C.srand48_r(C.legion_get_current_time_in_nanos(), &rngState)
   for c in r0 do
      var val1 = drand48_r(&rngState)
      var val2 = array(drand48_r(&rngState), drand48_r(&rngState), drand48_r(&rngState))
      r0[c].a = val1
      r1[c].a = val1
      r0[c].b = val2
      r1[c].b = val2
   end
end

task check(r0 : region(ispace(int3d), columns),
           r1 : region(ispace(int3d), columns))
where
   reads(r0.[IOVars]),
   reads(r1.[IOVars])
do
   for c in r0 do
      regentlib.assert(r0[c].a    == r1[c].a,    "mathUtilsTest: ERROR on region")
      regentlib.assert(r0[c].b[0] == r1[c].b[0], "mathUtilsTest: ERROR on region")
      regentlib.assert(r0[c].b[1] == r1[c].b[1], "mathUtilsTest: ERROR on region")
      regentlib.assert(r0[c].b[2] == r1[c].b[2], "mathUtilsTest: ERROR on region")
   end
end


task main()

   var r0      = region(ispace(int3d,{NP,NP,NP}), columns)
   var r1      = region(ispace(int3d,{NP,NP,NP}), columns)
   var r1_copy = region(ispace(int3d,{NP,NP,NP}), columns)

   var tiles = ispace(int3d, {NT,NT,NT})
   var p_r1 =
      [UTIL.mkPartitionByTile(int3d, int3d, columns, "p_All")]
      (r1, tiles, int3d{0,0,0}, int3d{0,0,0})
   var p_r1_copy =
      [UTIL.mkPartitionByTile(int3d, int3d, columns, "p_All")]
      (r1_copy, tiles, int3d{0,0,0}, int3d{0,0,0})

   fill(r0.a, 0.0)
   fill(r0.b, array(0.0, 0.0, 0.0))
   fill(r1.a, 0.0)
   fill(r1.b, array(0.0, 0.0, 0.0))

   init(r0, r1)
   var timeStep = 12345
   var simTime = 123.4567

   var dirname = [&int8](C.malloc(256))
   C.snprintf(dirname, 256, '.')
   var _1 : int
   _1 = HDF.dump(                 _1, tiles, dirname, r1, r1_copy, p_r1, p_r1_copy)
   _1 = HDF.write.timeStep(       _1, tiles, dirname, r1, p_r1, timeStep)
   _1 = HDF.write.simTime(        _1, tiles, dirname, r1, p_r1, simTime)

   __fence(__execution, __block)

   fill(r1.a, 0.0)
   fill(r1.b, array(0.0, 0.0, 0.0))

   timeStep = HDF.read.timeStep(0, tiles, dirname, r1, p_r1)
   simTime  = HDF.read.simTime( 0, tiles, dirname, r1, p_r1)
   HDF.load(0, tiles, dirname, r1, r1_copy, p_r1, p_r1_copy)

   regentlib.assert(timeStep == 12345,    "mathUtilsTest: ERROR on timeStep")
   regentlib.assert(simTime == 123.4567,    "mathUtilsTest: ERROR on simTime")
   check(r0, r1)

   __fence(__execution, __block)

   C.printf("hdfTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "hdfTest.o", "object")
