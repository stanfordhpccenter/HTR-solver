import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local UTIL = require 'util'

local rand = UTIL.mkRand()

local struct columns {
   a : double;
   b : double[3];
}

local IOVars = terralib.newlist({
   "a",
   "b"
})

local struct Astruct {
   c : double;
   d : double[3];
}

local NP = 32
local NT = 2

local HDF = (require 'hdf_helper')(int3d, int3d, columns,
                                                 IOVars,
                                                 {timeStep=int,simTime=double,
                                                 Cstruct=Astruct},
                                                 {})

task init(r0 : region(ispace(int3d), columns),
          r1 : region(ispace(int3d), columns))
where
   writes(r0.[IOVars]),
   writes(r1.[IOVars])
do
   var randSeed = C.legion_get_current_time_in_nanos()
   var xsize = r0.bounds.hi.x - r0.bounds.lo.x + 1
   var ysize = r0.bounds.hi.y - r0.bounds.lo.y + 1
   for c in r0 do
      var ctr1 = 4*(c.x + xsize*(c.y + ysize*c.z))
      var ctr2 = 4*(c.x + xsize*(c.y + ysize*c.z)) + 1
      var ctr3 = 4*(c.x + xsize*(c.y + ysize*c.z)) + 2
      var ctr4 = 4*(c.x + xsize*(c.y + ysize*c.z)) + 3
      var val1 = rand(randSeed, ctr1)
      var val2 = array(rand(randSeed, ctr2), rand(randSeed, ctr3), rand(randSeed, ctr4))
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
      regentlib.assert(r0[c].a    == r1[c].a,    "hdfTest: ERROR on region")
      regentlib.assert(r0[c].b[0] == r1[c].b[0], "hdfTest: ERROR on region")
      regentlib.assert(r0[c].b[1] == r1[c].b[1], "hdfTest: ERROR on region")
      regentlib.assert(r0[c].b[2] == r1[c].b[2], "hdfTest: ERROR on region")
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
   var c : Astruct
   c.c = 1234.567
   c.d = array(12.3, 45.6, 78.9)

   var dirname = [&int8](C.malloc(256))
   C.snprintf(dirname, 256, '.')
   var _1 : int
   _1 = HDF.dump(           _1, tiles, dirname, r1, r1_copy, p_r1, p_r1_copy)
   _1 = HDF.write.timeStep( _1, dirname, timeStep)
   _1 = HDF.write.simTime(  _1, dirname, simTime)
   _1 = HDF.write.Cstruct(  _1, dirname, c)

   __fence(__execution, __block)

   fill(r1.a, 0.0)
   fill(r1.b, array(0.0, 0.0, 0.0))

   timeStep = HDF.read.timeStep(tiles, dirname, r1, p_r1)
   simTime  = HDF.read.simTime( tiles, dirname, r1, p_r1)
   c        = HDF.read.Cstruct( tiles, dirname, r1, p_r1)
   HDF.load(tiles, dirname, r1, r1_copy, p_r1, p_r1_copy)

   regentlib.assert(timeStep == 12345,    "hdfTest: ERROR on timeStep")
   regentlib.assert(simTime == 123.4567,  "hdfTest: ERROR on simTime")
   regentlib.assert(c.c == 1234.567,      "hdfTest: ERROR on Cstruct.c")
   regentlib.assert(c.d[0] == 12.3,      "hdfTest: ERROR on Cstruct.d[0]")
   regentlib.assert(c.d[1] == 45.6,      "hdfTest: ERROR on Cstruct.d[1]")
   regentlib.assert(c.d[2] == 78.9,      "hdfTest: ERROR on Cstruct.d[2]")
   check(r0, r1)

   __fence(__execution, __block)

   -- Repart and read again
   fill(r1.a, 0.0)
   fill(r1.b, array(0.0, 0.0, 0.0))
   var tiles2 = ispace(int3d, {2*NT,2*NT,2*NT})
   var p2_r1 =
      [UTIL.mkPartitionByTile(int3d, int3d, columns, "p2_All")]
      (r1, tiles2, int3d{0,0,0}, int3d{0,0,0})
   var p2_r1_copy =
      [UTIL.mkPartitionByTile(int3d, int3d, columns, "p2_All")]
      (r1_copy, tiles2, int3d{0,0,0}, int3d{0,0,0})

   timeStep = HDF.read.timeStep(tiles2, dirname, r1, p2_r1)
   simTime  = HDF.read.simTime( tiles2, dirname, r1, p2_r1)
   c        = HDF.read.Cstruct( tiles2, dirname, r1, p2_r1)
   HDF.load(tiles2, dirname, r1, r1_copy, p2_r1, p2_r1_copy)

   regentlib.assert(timeStep == 12345,    "hdfTest: ERROR on timeStep")
   regentlib.assert(simTime == 123.4567,  "hdfTest: ERROR on simTime")
   regentlib.assert(c.c == 1234.567,      "hdfTest: ERROR on Cstruct.c")
   regentlib.assert(c.d[0] == 12.3,      "hdfTest: ERROR on Cstruct.d[0]")
   regentlib.assert(c.d[1] == 45.6,      "hdfTest: ERROR on Cstruct.d[1]")
   regentlib.assert(c.d[2] == 78.9,      "hdfTest: ERROR on Cstruct.d[2]")
   check(r0, r1)

   __fence(__execution, __block)

   C.printf("hdfTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "hdfTest.o", "object")
