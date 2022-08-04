-- Generate code for dumping/loading a subset of fields to/from an HDF file.
-- NOTE:
-- * Both functions require an intermediate region to perform the data
--   transfer. This region 's' must have the same size as 'r', and must be
--   partitioned in the same way.
-- * The dimensions will be flipped in the output file.
-- * You need to link to the HDF library to use these functions.

import 'regent'

-------------------------------------------------------------------------------
-- MODULE PARAMETERS
-------------------------------------------------------------------------------

return function(indexType, -- regentlib.index_type
                colorType, -- regentlib.index_type
                fSpace, -- terralib.struct
                flds, -- string*
                attrs, -- map(string,terralib.type)
                StringAttrs -- map(string,{int, int})
               )

local MODULE = {}
MODULE.read = {}
MODULE.write = {}

local VERSION = "1.0.0"

-------------------------------------------------------------------------------
-- FALLBACK MODE
-------------------------------------------------------------------------------

local USE_HDF = assert(os.getenv('USE_HDF')) ~= '0'

if not USE_HDF then

   __demand(__inline)
   task MODULE.dump(_ : int,
                    colors : ispace(colorType),
                    dirname : regentlib.string,
                    r : region(ispace(indexType), fSpace),
                    s : region(ispace(indexType), fSpace),
                    p_r : partition(disjoint, r, colors),
                    p_s : partition(disjoint, s, colors))
   where reads(r.[flds]), reads writes(s.[flds]), r * s do
      regentlib.assert(false, 'Recompile with USE_HDF=1')
      return _
   end

   __demand(__inline)
   task MODULE.load(colors : ispace(colorType),
                    dirname : regentlib.string,
                    r : region(ispace(indexType), fSpace),
                    s : region(ispace(indexType), fSpace),
                    p_r : partition(disjoint, r, colors),
                    p_s : partition(disjoint, s, colors))
   where reads writes(r.[flds]), reads writes(s.[flds]), r * s do
      regentlib.assert(false, 'Recompile with USE_HDF=1')
   end

   for aName,aType in pairs(attrs) do

      local __demand(__inline)
      task writeAttr(_ : int,
                    dirname : regentlib.string,
                    aVal : aType)
         regentlib.assert(false, 'Recompile with USE_HDF=1')
         return _
      end
      MODULE.write[aName] = writeAttr

      local __demand(__inline)
      task readAttr(colors : ispace(colorType),
                    dirname : regentlib.string,
                    r : region(ispace(indexType), fSpace),
                    p_r : partition(disjoint, r, colors))
         regentlib.assert(false, 'Recompile with USE_HDF=1')
         return [aType](0)
      end
      MODULE.read[aName] = readAttr
   end

   for aName,aPar in pairs(StringAttrs) do

      local aNum = aPar[1]

      local __demand(__inline)
      task writeAttr(_ : int,
                     colors : ispace(colorType),
                     dirname : regentlib.string,
                     Strings : regentlib.string[aNum])
         regentlib.assert(false, 'Recompile with USE_HDF=1')
         return _
      end
      MODULE.write[aName] = writeAttr

   end

   return MODULE
end

-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local HDF5 = terralib.includec(assert(os.getenv('HDF_HEADER')))
local UTIL = require 'util'

-- Mark HDF5 tasks as replicable
-- Note: HDF5 must be compiled in thread-safe mode
HDF5.H5Fopen.replicable = true
HDF5.H5Fclose.replicable = true
HDF5.H5Acreate2.replicable = true
HDF5.H5Aexists_by_name.replicable = true
HDF5.H5Aopen.replicable = true
HDF5.H5Aread.replicable = true
HDF5.H5Aclose.replicable = true
HDF5.H5Screate_simple.replicable = true
HDF5.H5Screate.replicable = true
HDF5.H5Sselect_elements.replicable = true
HDF5.H5Sclose.replicable = true
HDF5.H5Tcreate.replicable = true
HDF5.H5Tinsert.replicable = true
HDF5.H5Tcopy.replicable = true
HDF5.H5Tclose.replicable = true
HDF5.H5Dopen2.replicable = true
HDF5.H5Dget_space.replicable = true
HDF5.H5Dread.replicable = true
HDF5.H5Dclose.replicable = true

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------

-- HACK: Hardcoding missing #define's
HDF5.H5F_ACC_RDONLY = 0
HDF5.H5F_ACC_RDWR = 1
HDF5.H5F_ACC_TRUNC = 2
HDF5.H5P_DEFAULT = 0
HDF5.H5P_DATASET_CREATE = HDF5.H5P_CLS_DATASET_CREATE_ID_g

-------------------------------------------------------------------------------
-- MODULE-LOCAL TASKS
-------------------------------------------------------------------------------

-- string, string? -> terralib.quote
local function err(action, fld)
   if fld then
      return quote
         var stderr = C.fdopen(2, 'w')
         C.fprintf(stderr, 'HDF5: Cannot %s for field %s\n', action, fld)
         C.fflush(stderr)
         C.exit(1)
      end
   else
      return quote
         var stderr = C.fdopen(2, 'w')
         C.fprintf(stderr, 'HDF5: Cannot %s\n', action)
         C.fflush(stderr)
         C.exit(1)
      end
   end
end

-- bool, string, string? -> regentlib.rquote
local function HDF5Assert(condition, action, fld)
   local msg
   if fld then
      msg = "HDF5: Cannot ".. action .." for field " .. fld
   else
      msg = "HDF5: Cannot ".. action
   end
   return rquote
      regentlib.assert([condition], [msg])
   end
end

-- terralib.type -> terralib.expr
local function toPrimHType(T)
   return
      -- HACK: Hardcoding missing #define's
      (T == int)    and HDF5.H5T_STD_I32LE_g  or
      (T == int8)   and HDF5.H5T_STD_I8LE_g   or
      (T == int16)  and HDF5.H5T_STD_I16LE_g  or
      (T == int32)  and HDF5.H5T_STD_I32LE_g  or
      (T == int64)  and HDF5.H5T_STD_I64LE_g  or
      (T == uint)   and HDF5.H5T_STD_U32LE_g  or
      (T == uint8)  and HDF5.H5T_STD_U8LE_g   or
      (T == uint16) and HDF5.H5T_STD_U16LE_g  or
      (T == uint32) and HDF5.H5T_STD_U32LE_g  or
      (T == uint64) and HDF5.H5T_STD_U64LE_g  or
      (T == bool)   and HDF5.H5T_STD_U8LE_g   or
      (T == float)  and HDF5.H5T_IEEE_F32LE_g or
      (T == double) and HDF5.H5T_IEEE_F64LE_g or
      assert(false)
end

local terra create(fname : &int8, size : indexType)
   var fid = HDF5.H5Fcreate(fname, HDF5.H5F_ACC_TRUNC,
                            HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
   if fid < 0 then [err('create file')] end
   var dataSpace : HDF5.hid_t
   escape
      if indexType == int1d then
         emit quote
            var sizes : HDF5.hsize_t[1]
            sizes[0] = size.__ptr
            dataSpace = HDF5.H5Screate_simple(1, sizes, [&uint64](0))
            if dataSpace < 0 then [err('create 1d dataspace')] end
         end
      elseif indexType == int2d then
         emit quote
            -- Legion defaults to column-major layout, so we have to reverse.
            var sizes : HDF5.hsize_t[2]
            sizes[1] = size.__ptr.x
            sizes[0] = size.__ptr.y
            dataSpace = HDF5.H5Screate_simple(2, sizes, [&uint64](0))
            if dataSpace < 0 then [err('create 2d dataspace')] end
         end
      elseif indexType == int3d then
         emit quote
            -- Legion defaults to column-major layout, so we have to reverse.
            var sizes : HDF5.hsize_t[3]
            sizes[2] = size.__ptr.x
            sizes[1] = size.__ptr.y
            sizes[0] = size.__ptr.z
            dataSpace = HDF5.H5Screate_simple(3, sizes, [&uint64](0))
            if dataSpace < 0 then [err('create 3d dataspace')] end
         end
      else assert(false) end
      local header = terralib.newlist() -- terralib.quote*
      local footer = terralib.newlist() -- terralib.quote*
      -- terralib.type -> terralib.expr
      local function toHType(T)
         -- TODO: Not supporting: pointers, vectors, non-primitive arrays
         if T:isprimitive() then
            return toPrimHType(T)
         elseif T:isarray() then
            local elemType = toHType(T.type)
            local arrayType = symbol(HDF5.hid_t, 'arrayType')
            header:insert(quote
               var dims : HDF5.hsize_t[1]
               dims[0] = T.N
               var elemType = [elemType]
               var [arrayType] = HDF5.H5Tarray_create2(elemType, 1, dims)
               if arrayType < 0 then [err('create array type')] end
            end)
            footer:insert(quote
               HDF5.H5Tclose(arrayType)
            end)
            return arrayType
         else assert(false) end
      end
      -- terralib.struct, set(string), string -> ()
      local function emitFieldDecls(fs, whitelist, prefix)
         -- TODO: Only supporting pure structs, not fspaces
         assert(fs:isstruct())
         for _,e in ipairs(fs.entries) do
            local name, type = UTIL.parseStructEntry(e)
            if whitelist and not whitelist[name] then
               -- do nothing
            elseif type == int2d then
               -- Hardcode special case: int2d structs are stored packed
               local hName = prefix..name
               local int2dType = symbol(HDF5.hid_t, 'int2dType')
               local dataSet = symbol(HDF5.hid_t, 'dataSet')
               header:insert(quote
                  var [int2dType] = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 16)
                  if int2dType < 0 then [err('create 2d array type', name)] end
                  var x = HDF5.H5Tinsert(int2dType, "x", 0, HDF5.H5T_STD_I64LE_g)
                  if x < 0 then [err('add x to 2d array type', name)] end
                  var y = HDF5.H5Tinsert(int2dType, "y", 8, HDF5.H5T_STD_I64LE_g)
                  if y < 0 then [err('add y to 2d array type', name)] end
                  var [dataSet] = HDF5.H5Dcreate2(fid, hName, int2dType, dataSpace,
                              HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
                  if dataSet < 0 then [err('register 2d array type', name)] end
               end)
               footer:insert(quote
                  HDF5.H5Dclose(dataSet)
                  HDF5.H5Tclose(int2dType)
               end)
            elseif type == int3d then
               -- Hardcode special case: int3d structs are stored packed
               local hName = prefix..name
               local int3dType = symbol(HDF5.hid_t, 'int3dType')
               local dataSet = symbol(HDF5.hid_t, 'dataSet')
               header:insert(quote
                  var [int3dType] = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 24)
                  if int3dType < 0 then [err('create 3d array type', name)] end
                  var x = HDF5.H5Tinsert(int3dType, "x", 0, HDF5.H5T_STD_I64LE_g)
                  if x < 0 then [err('add x to 3d array type', name)] end
                  var y = HDF5.H5Tinsert(int3dType, "y", 8, HDF5.H5T_STD_I64LE_g)
                  if y < 0 then [err('add y to 3d array type', name)] end
                  var z = HDF5.H5Tinsert(int3dType, "z", 16, HDF5.H5T_STD_I64LE_g)
                  if z < 0 then [err('add z to 3d array type', name)] end
                  var [dataSet] = HDF5.H5Dcreate2(fid, hName, int3dType, dataSpace,
                              HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
                  if dataSet < 0 then [err('register 3d array type', name)] end
               end)
               footer:insert(quote
                  HDF5.H5Dclose(dataSet)
                  HDF5.H5Tclose(int3dType)
               end)
            elseif type:isstruct() then
               emitFieldDecls(type, nil, prefix..name..'.')
            else
               local hName = prefix..name
               local hType = toHType(type)
               local dataSet = symbol(HDF5.hid_t, 'dataSet')
               header:insert(quote
                  var hType = [hType]
                  var [dataSet] = HDF5.H5Dcreate2(
                     fid, hName, hType, dataSpace,
                  HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
                  if dataSet < 0 then [err('register type', name)] end
               end)
               footer:insert(quote
                  HDF5.H5Dclose(dataSet)
               end)
            end
         end
      end
      emitFieldDecls(fSpace, flds:toSet(), '')
      emit quote [header] end
      emit quote [footer:reverse()] end
   end
   HDF5.H5Sclose(dataSpace)
   HDF5.H5Fclose(fid)
end

local tileFilename
if indexType == int1d then
   __demand(__inline)
   task tileFilename(dirname : regentlib.string, bounds : rect1d)
      var filename = [&int8](C.malloc(256))
      var lo = bounds.lo
      var hi = bounds.hi
      C.snprintf(filename, 256,
                 '%s/%ld-%ld.hdf', dirname,
                 lo, hi)
      return filename
   end
elseif indexType == int2d then
   __demand(__inline)
   task tileFilename(dirname : regentlib.string, bounds : rect2d)
      var filename = [&int8](C.malloc(256))
      var lo = bounds.lo
      var hi = bounds.hi
      C.snprintf(filename, 256,
                 '%s/%ld,%ld-%ld,%ld.hdf', dirname,
                 lo.x, lo.y, hi.x, hi.y)
      return filename
   end
elseif indexType == int3d then
   __demand(__inline)
   task tileFilename(dirname : regentlib.string, bounds : rect3d)
      var filename = [&int8](C.malloc(256))
      var lo = bounds.lo
      var hi = bounds.hi
      C.snprintf(filename, 256,
                 '%s/%ld,%ld,%ld-%ld,%ld,%ld.hdf', dirname,
                 lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
      return filename
   end
else assert(false) end

local firstColor =
   colorType == int1d and rexpr 0 end or
   colorType == int2d and rexpr {0,0} end or
   colorType == int3d and rexpr {0,0,0} end or
   assert(false)

local oneColor =
   colorType == int1d and rexpr 1 end or
   colorType == int2d and rexpr {1,1} end or
   colorType == int3d and rexpr {1,1,1} end or
   assert(false)

local one =
   indexType == int1d and rexpr 1 end or
   indexType == int2d and rexpr {1,1} end or
   indexType == int3d and rexpr {1,1,1} end or
   assert(false)

local __demand(__inline)
task createVirtualDataset(fid : HDF5.hid_t,
                          r : region(ispace(indexType), fSpace),
                          p_r : partition(disjoint, r, ispace(colorType)))
   var dataSpace : HDF5.hid_t
   rescape
      if indexType == int1d then
         remit rquote
            var sizes : HDF5.hsize_t[1]
            sizes[0] = r.bounds.hi - r.bounds.lo + 1
            dataSpace = HDF5.H5Screate_simple(1, sizes, [&uint64](0));
            [HDF5Assert(rexpr dataSpace >= 0 end, "create 1d dataspace")];
         end
      elseif indexType == int2d then
         remit rquote
            -- Legion defaults to column-major layout, so we have to reverse.
            var sizes : HDF5.hsize_t[2]
            sizes[1] = r.bounds.hi.x - r.bounds.lo.x + 1
            sizes[0] = r.bounds.hi.y - r.bounds.lo.y + 1
            dataSpace = HDF5.H5Screate_simple(2, sizes, [&uint64](0));
            [HDF5Assert(rexpr dataSpace >= 0 end, "create 2d dataspace")];
         end
      elseif indexType == int3d then
         remit rquote
            -- Legion defaults to column-major layout, so we have to reverse.
            var sizes : HDF5.hsize_t[3]
            sizes[2] = r.bounds.hi.x - r.bounds.lo.x + 1
            sizes[1] = r.bounds.hi.y - r.bounds.lo.y + 1
            sizes[0] = r.bounds.hi.z - r.bounds.lo.z + 1
            dataSpace = HDF5.H5Screate_simple(3, sizes, [&uint64](0));
            [HDF5Assert(rexpr dataSpace >= 0 end, "create 3d dataspace")];
         end
      else assert(false) end
      local function setVirtualSources(dcpl, name)
         if indexType == int1d then
            return rquote
               var[dcpl] = HDF5.H5Pcreate(HDF5.H5P_DATASET_CREATE);
               [HDF5Assert(rexpr dcpl >= 0 end, "create 1d dataspace property list")];
               for c in p_r.colors do
                  var b = p_r[c].bounds
                  var start : HDF5.hsize_t[1]
                  start[0] = b.lo - r.bounds.lo
                  var stride : HDF5.hsize_t[1]
                  stride[0] = 1
                  var count : HDF5.hsize_t[1]
                  count[0] = 1
                  var block : HDF5.hsize_t[1]
                  block[0] = b.hi - b.lo + 1
                  var err = HDF5.H5Sselect_hyperslab(dataSpace, HDF5.H5S_SELECT_SET,
                                                     start, stride, count, block);
                  [HDF5Assert(rexpr err >= 0 end, "select hyperslab in 1d dataspace")];
                  var src_dataSpace = HDF5.H5Screate_simple(1, block, [&uint64](0));
                  [HDF5Assert(rexpr dataSpace >= 0 end, "create 1d source dataspace")];
                  var filename = tileFilename("./", b)
                  err = HDF5.H5Pset_virtual(dcpl, dataSpace, filename, name, src_dataSpace);
                  [HDF5Assert(rexpr err >= 0 end, "set virtual dataset", name)];
                  HDF5.H5Sclose(src_dataSpace)
               end
            end
         elseif indexType == int2d then
            return rquote
               var [dcpl] = HDF5.H5Pcreate(HDF5.H5P_DATASET_CREATE);
               [HDF5Assert(rexpr dcpl >= 0 end, "create 2d dataspace property list")];
               for c in p_r.colors do
                  var b = p_r[c].bounds
                  var start : HDF5.hsize_t[2]
                  start[1] = b.lo.x - r.bounds.lo.x
                  start[0] = b.lo.y - r.bounds.lo.y
                  var stride : HDF5.hsize_t[2]
                  stride[1] = 1
                  stride[0] = 1
                  var count : HDF5.hsize_t[2]
                  count[1] = 1
                  count[0] = 1
                  var block : HDF5.hsize_t[2]
                  block[1] = b.hi.x - b.lo.x + 1
                  block[0] = b.hi.y - b.lo.y + 1
                  var err = HDF5.H5Sselect_hyperslab(dataSpace, HDF5.H5S_SELECT_SET,
                                                     start, stride, count, block);
                  [HDF5Assert(rexpr err >= 0 end, "select hyperslab in 2d dataspace")];
                  var src_dataSpace = HDF5.H5Screate_simple(2, block, [&uint64](0));
                  [HDF5Assert(rexpr dataSpace >= 0 end, "create 2d source dataspace")];
                  var filename = tileFilename("./", b)
                  err = HDF5.H5Pset_virtual(dcpl, dataSpace, filename, name, src_dataSpace);
                  [HDF5Assert(rexpr err >= 0 end, "set virtual dataset", name)];
                  HDF5.H5Sclose(src_dataSpace)
               end
            end
         elseif indexType == int3d then
            return rquote
               var [dcpl] = HDF5.H5Pcreate(HDF5.H5P_DATASET_CREATE);
               [HDF5Assert(rexpr dcpl >= 0 end, "create 3d dataspace property list")];
               for c in p_r.colors do
                  var b = p_r[c].bounds
                  var start : HDF5.hsize_t[3]
                  start[2] = b.lo.x - r.bounds.lo.x
                  start[1] = b.lo.y - r.bounds.lo.y
                  start[0] = b.lo.z - r.bounds.lo.z
                  var stride : HDF5.hsize_t[3]
                  stride[2] = 1
                  stride[1] = 1
                  stride[0] = 1
                  var count : HDF5.hsize_t[3]
                  count[2] = 1
                  count[1] = 1
                  count[0] = 1
                  var block : HDF5.hsize_t[3]
                  block[2] = b.hi.x - b.lo.x + 1
                  block[1] = b.hi.y - b.lo.y + 1
                  block[0] = b.hi.z - b.lo.z + 1
                  var err = HDF5.H5Sselect_hyperslab(dataSpace, HDF5.H5S_SELECT_SET,
                                                     start, stride, count, block);
                  [HDF5Assert(rexpr err >= 0 end, "select hyperslab in 3d dataspace")];
                  var src_dataSpace = HDF5.H5Screate_simple(3, block, [&uint64](0));
                  [HDF5Assert(rexpr dataSpace >= 0 end, "create 3d source dataspace")];
                  var filename = tileFilename("./", b)
                  err = HDF5.H5Pset_virtual(dcpl, dataSpace, filename, name, src_dataSpace);
                  [HDF5Assert(rexpr err >= 0 end, "set virtual dataset", name)];
                  HDF5.H5Sclose(src_dataSpace)
               end
            end
         else assert(false) end
      end
      local header = terralib.newlist() -- terralib.quote*
      local footer = terralib.newlist() -- terralib.quote*
      -- terralib.type -> terralib.expr
      local function toHType(T)
         -- TODO: Not supporting: pointers, vectors, non-primitive arrays
         if T:isprimitive() then
            return toPrimHType(T)
         elseif T:isarray() then
            local elemType = toHType(T.type)
            local arrayType = regentlib.newsymbol(HDF5.hid_t, 'arrayType')
            local SIZE = T.N
            header:insert(rquote
               var dims : HDF5.hsize_t[1]
               dims[0] = SIZE
               var elemType = [elemType]
               var [arrayType] = HDF5.H5Tarray_create2(elemType, 1, dims);
               [HDF5Assert(rexpr arrayType >= 0 end, "create array type")];
            end)
            footer:insert(rquote
               HDF5.H5Tclose(arrayType)
            end)
            return arrayType
         else assert(false) end
      end
      -- terralib.struct, set(string), string -> ()
      local function emitFieldDecls(fs, whitelist, prefix)
         -- TODO: Only supporting pure structs, not fspaces
         assert(fs:isstruct())
         for _,e in ipairs(fs.entries) do
            local name, type = UTIL.parseStructEntry(e)
            if whitelist and not whitelist[name] then
               -- do nothing
            elseif type == int2d then
               -- Hardcode special case: int2d structs are stored packed
               local hName = prefix..name
               local int2dType = regentlib.newsymbol(HDF5.hid_t, 'int2dType')
               local dataSet = regentlib.newsymbol(HDF5.hid_t, 'dataSet')
               local dcpl = regentlib.newsymbol(HDF5.hid_t, 'dcpl')
               header:insert(quote
                  var [int2dType] = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 16);
                  [HDF5Assert(rexpr int2dType >= 0 end, "create 2d array type", name)];
                  var x = HDF5.H5Tinsert(int2dType, "x", 0, HDF5.H5T_STD_I64LE_g);
                  [HDF5Assert(rexpr x >= 0 end, "add x to 2d array type", name)];
                  var y = HDF5.H5Tinsert(int2dType, "y", 8, HDF5.H5T_STD_I64LE_g);
                  [HDF5Assert(rexpr y >= 0 end, "add y to 2d array type", name)];
                  [setVirtualSources(dcpl, name)];
                  var [dataSet] = HDF5.H5Dcreate2(fid, hName, int2dType, dataSpace,
                              HDF5.H5P_DEFAULT, dcpl, HDF5.H5P_DEFAULT);
                  [HDF5Assert(rexpr dataSet >= 0 end, "register 2d array type", name)];
                  HDF5.H5Pclose(dcpl)
               end)
               footer:insert(quote
                  HDF5.H5Dclose(dataSet)
                  HDF5.H5Pclose(dcpl)
                  HDF5.H5Tclose(int2dType)
               end)
            elseif type == int3d then
               -- Hardcode special case: int3d structs are stored packed
               local hName = prefix..name
               local int3dType = regentlib.newsymbol(HDF5.hid_t, 'int3dType')
               local dataSet = regentlib.newsymbol(HDF5.hid_t, 'dataSet')
               local dcpl = regentlib.newsymbol(HDF5.hid_t, 'dcpl')
               header:insert(quote
                  var [int3dType] = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 24);
                  [HDF5Assert(rexpr int3dType >= 0  end, "create 3d array type", name)];
                  var x = HDF5.H5Tinsert(int3dType, "x", 0, HDF5.H5T_STD_I64LE_g);
                  [HDF5Assert(rexpr x >= 0  end, "add x to 3d array type", name)];
                  var y = HDF5.H5Tinsert(int3dType, "y", 8, HDF5.H5T_STD_I64LE_g);
                  [HDF5Assert(rexpr y >= 0  end, "add y to 3d array type", name)];
                  var z = HDF5.H5Tinsert(int3dType, "z", 16, HDF5.H5T_STD_I64LE_g);
                  [HDF5Assert(rexpr z >= 0 end, "add z to 3d array type", name)];
                  [setVirtualSources(dcpl, name)];
                  var [dataSet] = HDF5.H5Dcreate2(fid, hName, int3dType, dataSpace,
                              HDF5.H5P_DEFAULT, dcpl, HDF5.H5P_DEFAULT);
                  [HDF5Assert(rexpr dataSet >= 0 end, "register 3d array type", name)];
                  HDF5.H5Pclose(dcpl)
               end)
               footer:insert(quote
                  HDF5.H5Dclose(dataSet)
                  HDF5.H5Pclose(dcpl)
                  HDF5.H5Tclose(int3dType)
               end)
            elseif type:isstruct() then
               emitFieldDecls(type, nil, prefix..name..'.')
            else
               local hName = prefix..name
               local hType = toHType(type)
               local dataSet = regentlib.newsymbol(HDF5.hid_t, 'dataSet')
               local dcpl = regentlib.newsymbol(HDF5.hid_t, 'dcpl')
               header:insert(rquote
                  var hType = [hType];
                  [setVirtualSources(dcpl, name)];
                  var [dataSet] = HDF5.H5Dcreate2(
                        fid, hName, hType, dataSpace,
                        HDF5.H5P_DEFAULT, dcpl, HDF5.H5P_DEFAULT);
                  [HDF5Assert(rexpr dataSet >= 0 end, "register type", name)];
               end)
               footer:insert(rquote
                  HDF5.H5Dclose(dataSet)
                  HDF5.H5Pclose(dcpl)
               end)
            end
         end
      end
      emitFieldDecls(fSpace, flds:toSet(), '')
      remit rquote [header] end
      remit rquote [footer:reverse()] end
   end
   HDF5.H5Sclose(dataSpace)
end

local function mkH5Types(pointType)
   local H5TpointType
   local H5TRectType
   local pStruct
   local point2struct
   local bStruct
   local rect2struct
   if pointType == int1d then
      __demand(__inline)
      task H5TpointType()
         var pointType = HDF5.H5Tcopy(HDF5.H5T_STD_I64LE_g)
         regentlib.assert(pointType >= 0, "HDF5: Cannot create 1d point type")
         return pointType
      end

      __demand(__inline)
      task H5TRectType(pointType : HDF5.hid_t)
         var rectType = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 16)
         regentlib.assert(rectType >= 0, "HDF5: Cannot create rect1d type")
         var lo = HDF5.H5Tinsert(rectType, "lo",  0, pointType)
         regentlib.assert(lo >= 0, "HDF5: Cannot add lo to rect1d type")
         var hi = HDF5.H5Tinsert(rectType, "hi",  8, pointType)
         regentlib.assert(hi >= 0, "HDF5: Cannot add hi to rect1d type")
         return rectType
      end

      pStruct = int64

      __demand(__inline)
      task point2struct(p : int1d)
         var out : int64 = p
         return out
      end

      struct bStruct {
         lo : int64;
         hi : int64;
      }

      __demand(__inline)
      task rect2struct(b : rect1d)
         var out : bStruct
         out.lo = b.lo
         out.hi = b.hi
         return out
      end

   elseif pointType == int2d then
      __demand(__inline)
      task H5TpointType()
         var pointType = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 16)
         regentlib.assert(pointType >= 0 , "HDF5: Cannot create 2d point type")
         var x = HDF5.H5Tinsert(pointType, "x", 0, HDF5.H5T_STD_I64LE_g)
         regentlib.assert(x >= 0, "HDF5: Cannot add x to 2d point type")
         var y = HDF5.H5Tinsert(pointType, "y", 8, HDF5.H5T_STD_I64LE_g)
         regentlib.assert(y >= 0, "HDF5: Cannot add y to 2d point type")
         return pointType
      end

      __demand(__inline)
      task H5TRectType(pointType : HDF5.hid_t)
         var rectType = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 32)
         regentlib.assert(rectType >= 0, "HDF5: Cannot create rect2d type")
         var lo = HDF5.H5Tinsert(rectType, "lo",  0, pointType)
         regentlib.assert(lo >= 0, "HDF5: Cannot add lo to rect2d type")
         var hi = HDF5.H5Tinsert(rectType, "hi", 16, pointType)
         regentlib.assert(hi >= 0, "HDF5: Cannot add hi to rect2d type")
         return rectType
      end

      pStruct = int64[2]

      __demand(__inline)
      task point2struct(p : int2d)
         var out : int64[2]
         out[0] = p.x
         out[1] = p.y
         return out
      end

      struct bStruct {
         lo : int64[2];
         hi : int64[2];
      }

      __demand(__inline)
      task rect2struct(b : rect2d)
         var out : bStruct
         out.lo[0] = b.lo.x
         out.lo[1] = b.lo.y
         out.hi[0] = b.hi.x
         out.hi[1] = b.hi.y
         return out
      end

   elseif pointType == int3d then
      __demand(__inline)
      task H5TpointType()
         var pointType = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 24)
         regentlib.assert(pointType >= 0, "HDF5: Cannot create 3d point type")
         var x = HDF5.H5Tinsert(pointType, "x",  0, HDF5.H5T_STD_I64LE_g)
         regentlib.assert(x >= 0, "HDF5: Cannot add x to 3d point type")
         var y = HDF5.H5Tinsert(pointType, "y",  8, HDF5.H5T_STD_I64LE_g)
         regentlib.assert(y >= 0, "HDF5: Cannot add y to 3d point type")
         var z = HDF5.H5Tinsert(pointType, "z", 16, HDF5.H5T_STD_I64LE_g)
         regentlib.assert(z >= 0, "HDF5: Cannot add z to 3d point type")
         return pointType
      end

      __demand(__inline)
      task H5TRectType(pointType : HDF5.hid_t)
         var rectType = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, 48)
         regentlib.assert(rectType >= 0, "HDF5: Cannot create rect3d type")
         var lo = HDF5.H5Tinsert(rectType, "lo",  0, pointType)
         regentlib.assert(lo >= 0, "HDF5: Cannot add lo to rect3d type")
         var hi = HDF5.H5Tinsert(rectType, "hi", 24, pointType)
         regentlib.assert(hi >= 0, "HDF5: Cannot add hi to rect3d type")
         return rectType
      end

      pStruct = int64[3]

      __demand(__inline)
      task point2struct(p : int3d)
         var out : int64[3]
         out[0] = p.x
         out[1] = p.y
         out[2] = p.z
         return out
      end

      struct bStruct {
         lo : int64[3];
         hi : int64[3];
      }
      __demand(__inline)
      task rect2struct(b : rect3d)
         var out : bStruct
         out.lo[0] = b.lo.x
         out.lo[1] = b.lo.y
         out.lo[2] = b.lo.z
         out.hi[0] = b.hi.x
         out.hi[1] = b.hi.y
         out.hi[2] = b.hi.z
         return out
      end

   else assert(false) end
   return H5TpointType, H5TRectType, pStruct, point2struct, bStruct, rect2struct
end

local H5TindexType, H5TindexRectType, indexStruct, index2struct, IndexRectStruct, indexRect2struct = mkH5Types(indexType)
local H5TcolorType, H5TcolorRectType, colorStruct, color2struct, ColorRectStruct, colorRect2struct = mkH5Types(colorType)

local H5TPartitionSpace
local H5TselectPartitionPoint
if colorType == int1d then
   __demand(__inline)
   task H5TPartitionSpace(colors : ispace(colorType))
      var aSize : HDF5.hsize_t[1]
      aSize[0] = colors.bounds.hi - colors.bounds.lo + 1
      var sid = HDF5.H5Screate_simple(1, aSize, aSize)
      regentlib.assert(sid >= 0, "HDF5: Cannot create partitions attribute dataspace")
      return sid
   end
   __demand(__inline)
   task H5TselectPartitionPoint(c : colorType, sid : HDF5.hid_t)
      var aSize : HDF5.hsize_t[1]
      aSize[0] = c
      var res = HDF5.H5Sselect_elements(sid, HDF5.H5S_SELECT_SET, 1, aSize)
      regentlib.assert(res >= 0, "HDF5: Cannot select partition point in dataspace")
   end
elseif colorType == int2d then
   __demand(__inline)
   task H5TPartitionSpace(colors : ispace(colorType))
      var aSize : HDF5.hsize_t[2]
      aSize[0] = colors.bounds.hi.y - colors.bounds.lo.y + 1
      aSize[1] = colors.bounds.hi.x - colors.bounds.lo.x + 1
      var sid = HDF5.H5Screate_simple(2, aSize, aSize)
      regentlib.assert(sid >= 0, "HDF5: Cannot create partitions attribute dataspace")
      return sid
   end
   __demand(__inline)
   task H5TselectPartitionPoint(c : colorType, sid : HDF5.hid_t)
      var aSize : HDF5.hsize_t[2]
      aSize[0] = c.y
      aSize[1] = c.x
      var res = HDF5.H5Sselect_elements(sid, HDF5.H5S_SELECT_SET, 1, aSize)
      regentlib.assert(res >= 0, "HDF5: Cannot select partition point in dataspace")
   end
elseif colorType == int3d then
   __demand(__inline)
   task H5TPartitionSpace(colors : ispace(colorType))
      var aSize : HDF5.hsize_t[3]
      aSize[0] = colors.bounds.hi.z - colors.bounds.lo.z + 1
      aSize[1] = colors.bounds.hi.y - colors.bounds.lo.y + 1
      aSize[2] = colors.bounds.hi.x - colors.bounds.lo.x + 1
      var sid = HDF5.H5Screate_simple(3, aSize, aSize)
      regentlib.assert(sid >= 0, "HDF5: Cannot create partitions attribute dataspace")
      return sid
   end
   __demand(__inline)
   task H5TselectPartitionPoint(c : colorType, sid : HDF5.hid_t)
      var aSize : HDF5.hsize_t[3]
      aSize[0] = c.z
      aSize[1] = c.y
      aSize[2] = c.x
      var res = HDF5.H5Sselect_elements(sid, HDF5.H5S_SELECT_SET, 1, aSize)
      regentlib.assert(res >= 0, "HDF5: Cannot select partition point in dataspace")
   end
else assert(false) end


-------------------------------------------------------------------------------
-- EXPORTED TASKS
-------------------------------------------------------------------------------

local colorRectType =
   colorType == int1d and rect1d or
   colorType == int2d and rect2d or
   colorType == int3d and rect3d or
   assert(false)


local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task dumpMasterFile(_ : int,
                    dirname : regentlib.string,
                    r : region(ispace(indexType), fSpace),
                    p_r : partition(disjoint, r, ispace(colorType)))

   -- Create file
   var filename = [&int8](C.malloc(256))
   C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
   var fid = HDF5.H5Fcreate(filename, HDF5.H5F_ACC_TRUNC,
                            HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
   regentlib.assert(fid >= 0, "HDF5: Cannot create master file")

   -- Create scalar data space
   var sid = HDF5.H5Screate(HDF5.H5S_SCALAR)
   regentlib.assert(sid >= 0, "HDF5: Cannot create attribute dataspace")

   -- Store IO version
   var stringType = HDF5.H5Tcopy(HDF5.H5T_C_S1_g)
   var res = HDF5.H5Tset_size (stringType, 8)
   regentlib.assert(res >= 0, "HDF5: Cannot set attribute size")
   var aid = HDF5.H5Acreate2(fid, "IO_VERSION", stringType, sid,
                             HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
   regentlib.assert(aid >= 0, "HDF5: Cannot create IO_VERSION attribute")
   var v : int8[8]
   C.snprintf(&v[0], 8, VERSION)
   res = HDF5.H5Awrite(aid, stringType, &v)
   regentlib.assert(res >= 0, "HDF5: Cannot write IO_VERSION attribute")
   HDF5.H5Aclose(aid)

   -- Register data types to describe index spaces
   var indexType = H5TindexType()
   var iRectType = H5TindexRectType(indexType)
   var colorType = H5TcolorType()
   var cRectType = H5TcolorRectType(colorType)

   -- Store main index space
   aid = HDF5.H5Acreate2(fid, "baseIndexSpace", iRectType, sid,
                         HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
   regentlib.assert(aid >= 0, "HDF5: Cannot create baseIndexSpace attribute")
   var b = indexRect2struct(r.bounds)
   res = HDF5.H5Awrite(aid, iRectType, &b)
   regentlib.assert(res >= 0, "HDF5: Cannot write baseIndexSpace attribute")
   HDF5.H5Aclose(aid)

   -- Store colors
   aid = HDF5.H5Acreate2(fid, "tilesNumber", colorType, sid,
                         HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
   regentlib.assert(aid >= 0, "HDF5: Cannot create tilesNumber attribute")
--   regentlib.assert(colors.bounds.lo == firstColor, 'tiles should have 0 as lower bound')
   var colors_hi = color2struct(p_r.colors.bounds.hi - p_r.colors.bounds.lo + oneColor)
   res = HDF5.H5Awrite(aid, colorType, &colors_hi)
   regentlib.assert(res >= 0, "HDF5: Cannot write tilesNumber attribute")
   HDF5.H5Aclose(aid)

   -- Store partitions
   var part_sid = H5TPartitionSpace(p_r.colors)
   var part_aid = HDF5.H5Dcreate2(fid, "partitions", iRectType, part_sid,
                                  HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
   regentlib.assert(part_aid >= 0, "HDF5: Cannot create partitions storage")
   var part_fid = HDF5.H5Dget_space(part_aid)
   regentlib.assert(part_fid >= 0, "HDF5: Cannot create partitions file_id")
   for c in p_r.colors do
      H5TselectPartitionPoint(c, part_fid)
      var b = indexRect2struct(p_r[c].bounds)
      var res = HDF5.H5Dwrite(part_aid, iRectType, sid, part_fid,
                              HDF5.H5P_DEFAULT, &b)
      regentlib.assert(res >= 0, "HDF5: Cannot write partition storage")
   end
   HDF5.H5Sclose(part_fid)
   HDF5.H5Dclose(part_aid)
   HDF5.H5Sclose(part_sid)

   -- Create a virtual dataset that represents the entire index space
   createVirtualDataset(fid, r, p_r)

   -- Clean up
   HDF5.H5Tclose(indexType)
   HDF5.H5Tclose(iRectType)
   HDF5.H5Tclose(colorType)
   HDF5.H5Tclose(cRectType)
   HDF5.H5Tclose(stringType)
   HDF5.H5Sclose(sid)
   HDF5.H5Fclose(fid)
   C.free(filename)
   return _
end

local __demand(__inner) -- NOT LEAF, MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task dumpTile(_ : int,
              dirname : regentlib.string,
              r : region(ispace(indexType), fSpace),
              s : region(ispace(indexType), fSpace))
where reads(r.[flds]), reads writes(s.[flds]), r * s do
   if (r.volume ~= 0) then
      var filename = tileFilename([&int8](dirname), r.bounds)
      create(filename, r.bounds.hi - r.bounds.lo + one)
      attach(hdf5, s.[flds], filename, regentlib.file_read_write)
      acquire(s.[flds])
      copy(r.[flds], s.[flds])
      release(s.[flds])
      detach(hdf5, s.[flds])
      C.free(filename)
   end
end

__demand(__inline)
task MODULE.dump(_ : int,
                 colors : ispace(colorType),
                 dirname : regentlib.string,
                 r : region(ispace(indexType), fSpace),
                 s : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors),
                 p_s : partition(disjoint, s, colors))
where reads(r.[flds]), reads writes(s.[flds]), r * s do
   -- Dump the actual data
   for c in colors do
      dumpTile(_, dirname, p_r[c], p_s[c])
   end
   -- Dump all the domain info into a master file
   return dumpMasterFile(_, dirname, r, p_r)
end

local __demand(__inline)
task checkMasterFile(dirname : regentlib.string,
                     r : region(ispace(indexType), fSpace),
                     p_r : partition(disjoint, r, ispace(colorType)))

   -- This flag is going to tell us if we need to repart our domain
   var repart = false

   -- Open file
   var filename = [&int8](C.malloc(256))
   C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
   var fid = HDF5.H5Fopen(filename, HDF5.H5F_ACC_RDONLY,
                          HDF5.H5P_DEFAULT)
   if (fid < 0) then
      -- We did not find any master file.
      -- Assume everything is ok and we do not need to repart our data
      -- for backward compatibility
      repart = false
   end

   -- TODO: Check IO version?

   -- Register data types to describe index spaces
   var H5indexType = H5TindexType()
   var H5iRectType = H5TindexRectType(H5indexType)
   var H5colorType = H5TcolorType()

   if (fid >= 0) then
      -- Check that we are working on the same index space
      var res = HDF5.H5Aexists_by_name(fid, ".", "baseIndexSpace",
                                       HDF5.H5P_DEFAULT)
      regentlib.assert(res >= 1, "HDF5: Cannot find baseIndexSpace attribute")
      var aid = HDF5.H5Aopen(fid, "baseIndexSpace", HDF5.H5P_DEFAULT)
      regentlib.assert(aid >= 0, "HDF5: Cannot open baseIndexSpace attribute")
      var out : IndexRectStruct
      res = HDF5.H5Aread(aid, H5iRectType, &out)
      regentlib.assert(res >= 0, "HDF5: Cannot read baseIndexSpace attribute")
      HDF5.H5Aclose(aid)
rescape if (indexType == int1d) then remit rquote
      regentlib.assert(int1d(out.lo) == r.bounds.lo  , "HDF5: Cannot match baseIndexSpace.lo and bounds.lo")
      regentlib.assert(int1d(out.hi) == r.bounds.hi  , "HDF5: Cannot match baseIndexSpace.hi and bounds.hi")
end elseif (indexType == int2d) then remit rquote
      regentlib.assert(out.lo[0] == r.bounds.lo.x, "HDF5: Cannot match baseIndexSpace.lo.x and bounds.lo.x")
      regentlib.assert(out.lo[1] == r.bounds.lo.y, "HDF5: Cannot match baseIndexSpace.lo.y and bounds.lo.y")
      regentlib.assert(out.hi[0] == r.bounds.hi.x, "HDF5: Cannot match baseIndexSpace.hi.x and bounds.hi.x")
      regentlib.assert(out.hi[1] == r.bounds.hi.y, "HDF5: Cannot match baseIndexSpace.hi.y and bounds.hi.y")
end elseif (indexType == int3d) then remit rquote
      regentlib.assert(out.lo[0] == r.bounds.lo.x, "HDF5: Cannot match baseIndexSpace.lo.x and bounds.lo.x")
      regentlib.assert(out.lo[1] == r.bounds.lo.y, "HDF5: Cannot match baseIndexSpace.lo.y and bounds.lo.y")
      regentlib.assert(out.lo[2] == r.bounds.lo.z, "HDF5: Cannot match baseIndexSpace.lo.z and bounds.lo.z")
      regentlib.assert(out.hi[0] == r.bounds.hi.x, "HDF5: Cannot match baseIndexSpace.hi.x and bounds.hi.x")
      regentlib.assert(out.hi[1] == r.bounds.hi.y, "HDF5: Cannot match baseIndexSpace.hi.y and bounds.hi.y")
      regentlib.assert(out.hi[2] == r.bounds.hi.z, "HDF5: Cannot match baseIndexSpace.hi.z and bounds.hi.z")
end else assert(false) end end

      -- Check number of tiles
      res = HDF5.H5Aexists_by_name(fid, ".", "tilesNumber",
                                   HDF5.H5P_DEFAULT)
      regentlib.assert(res >= 1, "HDF5: Cannot find tilesNumber attribute")
      aid = HDF5.H5Aopen(fid, "tilesNumber", HDF5.H5P_DEFAULT)
      regentlib.assert(aid >= 0, "HDF5: Cannot open tilesNumber attribute")
      var colors_hi = p_r.colors.bounds.hi - p_r.colors.bounds.lo + oneColor
      var in_tiles : colorStruct
      res = HDF5.H5Aread(aid, H5colorType, &in_tiles)
      regentlib.assert(res >= 0, "HDF5: Cannot read tilesNumber attribute")
rescape if (colorType == int1d) then remit rquote
      repart = not (int1d(in_tiles) == colors_hi)
end elseif (colorType == int2d) then remit rquote
      repart = not ((in_tiles[0] == colors_hi.x) and
                    (in_tiles[1] == colors_hi.y))
end elseif (colorType == int3d) then remit rquote
      repart = not ((in_tiles[0] == colors_hi.x) and
                    (in_tiles[1] == colors_hi.y) and
                    (in_tiles[2] == colors_hi.z))
end else assert(false) end end
      HDF5.H5Aclose(aid)

      if not repart then
         -- The number of partitions is fine, let's check their shape
         var part_sid = H5TPartitionSpace(p_r.colors)
         var part_aid = HDF5.H5Dopen2(fid, "partitions", HDF5.H5P_DEFAULT)
         regentlib.assert(part_aid >= 0, "HDF5: Cannot open partitions storage")
         var part_fid = HDF5.H5Dget_space(part_aid)
         regentlib.assert(part_fid >= 0, "HDF5: Cannot get partitions file_id")
         var sid = HDF5.H5Screate(HDF5.H5S_SCALAR)
         regentlib.assert(sid >= 0, "HDF5: Cannot create attribute dataspace")
         for c in p_r.colors do
            H5TselectPartitionPoint(c, part_fid)
            var in_block : IndexRectStruct
            var res = HDF5.H5Dread(part_aid, H5iRectType, sid, part_fid,
                                   HDF5.H5P_DEFAULT, &in_block)
            regentlib.assert(res >= 0, "HDF5: Cannot read partition")
            var b = p_r[c].bounds
rescape if (colorType == int1d) then remit rquote
            repart = (repart and
                     ((int1d(in_block.lo) == b.lo) and
                      (int1d(in_block.hi) == b.hi)))
end elseif (colorType == int2d) then remit rquote
            repart = (repart and
                     ((in_block.lo[0] == b.lo.x) and (in_block.lo[1] == b.lo.y) and
                      (in_block.hi[0] == b.hi.x) and (in_block.hi[1] == b.hi.y)))
end elseif (colorType == int3d) then remit rquote
            repart = (repart and
                     ((in_block.lo[0] == b.lo.x) and (in_block.lo[1] == b.lo.y) and (in_block.lo[2] == b.lo.z) and
                      (in_block.hi[0] == b.hi.x) and (in_block.hi[1] == b.hi.y) and (in_block.hi[2] == b.hi.z)))
end else assert(false) end end

         end
         -- Clean up
         HDF5.H5Sclose(sid)
         HDF5.H5Sclose(part_fid)
         HDF5.H5Dclose(part_aid)
         HDF5.H5Sclose(part_sid)
      end

      -- Clean up
      HDF5.H5Tclose(H5indexType)
      HDF5.H5Tclose(H5iRectType)
      HDF5.H5Tclose(H5colorType)
      HDF5.H5Fclose(fid)
   end

   -- Clean up
   C.free(filename)

   -- If we made it here, it means that the current partitioning is fine
   return repart
end

local __demand(__inline)
task MasterFileTiles(dirname : regentlib.string)
   -- Open file
   var filename = [&int8](C.malloc(256))
   C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
   var fid = HDF5.H5Fopen(filename, HDF5.H5F_ACC_RDONLY,
                          HDF5.H5P_DEFAULT)

   -- TODO: Check IO version?

   -- Register data types to describe index spaces
   var H5colorType = H5TcolorType()

   -- Read number of tiles
   var res = HDF5.H5Aexists_by_name(fid, ".", "tilesNumber",
                                    HDF5.H5P_DEFAULT)
   regentlib.assert(res >= 1, "HDF5: Cannot find tilesNumber attribute")
   var aid = HDF5.H5Aopen(fid, "tilesNumber", HDF5.H5P_DEFAULT)
   regentlib.assert(aid >= 0, "HDF5: Cannot open tilesNumber attribute")
   var in_tiles : colorStruct
   res = HDF5.H5Aread(aid, H5colorType, &in_tiles)
   regentlib.assert(res >= 0, "HDF5: Cannot read tilesNumber attribute")

   -- Clean up
   HDF5.H5Aclose(aid)
   HDF5.H5Tclose(H5colorType)
   HDF5.H5Fclose(fid)
   C.free(filename)

rescape if (colorType == int1d) then remit rquote
   return [colorType](in_tiles)
end elseif (colorType == int2d) then remit rquote
   return [colorType]({in_tiles[0], in_tiles[1]})
end elseif (colorType == int3d) then remit rquote
   return [colorType]({in_tiles[0], in_tiles[1], in_tiles[2]})
end else assert(false) end end
end

local __demand(__inline)
task MasterFileColors(dirname : regentlib.string,
                      colors : colorType)
   -- Open file
   var filename = [&int8](C.malloc(256))
   C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
   var fid = HDF5.H5Fopen(filename, HDF5.H5F_ACC_RDONLY,
                          HDF5.H5P_DEFAULT)
   -- Define color space
   var coloring = regentlib.c.legion_domain_point_coloring_create()

   -- TODO: Check IO version?

   -- Register data types to describe index spaces
   var H5indexType = H5TindexType()
   var H5iRectType = H5TindexRectType(H5indexType)

   -- Check partitions
   var is = ispace(colorType, colors)
   var part_sid = H5TPartitionSpace(is)
   var part_aid = HDF5.H5Dopen2(fid, "partitions", HDF5.H5P_DEFAULT)
   regentlib.assert(part_aid >= 0, "HDF5: Cannot open partitions storage")
   var part_fid = HDF5.H5Dget_space(part_aid)
   regentlib.assert(part_fid >= 0, "HDF5: Cannot get partitions file_id")
   var sid = HDF5.H5Screate(HDF5.H5S_SCALAR)
   regentlib.assert(sid >= 0, "HDF5: Cannot create attribute dataspace")
   for c in is do
      H5TselectPartitionPoint(c, part_fid);
      var out : IndexRectStruct
      var res = HDF5.H5Dread(part_aid, H5iRectType, sid, part_fid,
                             HDF5.H5P_DEFAULT, &out)
      regentlib.assert(res >= 0, "HDF5: Cannot read partition")
rescape if (indexType == int1d) then remit rquote
      regentlib.c.legion_domain_point_coloring_color_domain(coloring, c, rect1d{
         lo = out.lo,
         hi = out.hi
      })
end elseif (indexType == int2d) then remit rquote
      regentlib.c.legion_domain_point_coloring_color_domain(coloring, c, rect2d{
         lo = {out.lo[0], out.lo[0]},
         hi = {out.hi[0], out.hi[1]}
      })
end elseif (indexType == int3d) then remit rquote
      regentlib.c.legion_domain_point_coloring_color_domain(coloring, c, rect3d{
         lo = {out.lo[0], out.lo[1], out.lo[2]},
         hi = {out.hi[0], out.hi[1], out.hi[2]}
      })
end else assert(false) end end
   end

   -- Clean up
   HDF5.H5Sclose(sid)
   HDF5.H5Sclose(part_fid)
   HDF5.H5Dclose(part_aid)
   HDF5.H5Sclose(part_sid)
   HDF5.H5Tclose(H5indexType)
   HDF5.H5Tclose(H5iRectType)
   HDF5.H5Fclose(fid)
   C.free(filename)

   return coloring
end

local __demand(__inner) -- NOT LEAF, MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task loadTile(dirname : regentlib.string,
              r : region(ispace(indexType), fSpace),
              s : region(ispace(indexType), fSpace))
where reads writes(r.[flds]), reads writes(s.[flds]), r * s do
   var filename = tileFilename([&int8](dirname), r.bounds)
   attach(hdf5, s.[flds], filename, regentlib.file_read_only)
   acquire(s.[flds])
   copy(s.[flds], r.[flds])
   release(s.[flds])
   detach(hdf5, s.[flds])
   C.free(filename)
end

__demand(__inline)
task MODULE.load(colors : ispace(colorType),
                 dirname : regentlib.string,
                 r : region(ispace(indexType), fSpace),
                 s : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors),
                 p_s : partition(disjoint, s, colors))
where reads writes(r.[flds]), reads writes(s.[flds]), r * s do
   -- Check that the index space info are consistent
   -- and that the provided partitions are ok
   var repart = checkMasterFile(dirname, r, p_r)
   if repart then
      -- Create temporary colors and partitions
      var colors_tmp = MasterFileTiles(dirname)
      var coloring = MasterFileColors(dirname, colors_tmp)
      var is_tmp = ispace(colorType, colors_tmp)
      var p_r_tmp = partition(disjoint, r, coloring, is_tmp)
      var p_s_tmp = partition(disjoint, s, coloring, is_tmp)
      regentlib.c.legion_domain_point_coloring_destroy(coloring)
      -- Load the data
      __demand(__index_launch)
      for c in is_tmp do
         loadTile(dirname, p_r_tmp[c], p_s_tmp[c])
      end
   else
      -- Use existing partitions
      -- Load the data
      __demand(__index_launch)
      for c in colors do
         loadTile(dirname, p_r[c], p_s[c])
      end
   end
end

for aName,aType in pairs(attrs) do

   local header = terralib.newlist()
   local fid = symbol(HDF5.hid_t, 'dataSet')
   local sid = symbol(HDF5.hid_t, 'dataSet')
   local footer = terralib.newlist()

   local function toHType(T, prefix)
      -- TODO: Not supporting: pointers, vectors, non-primitive array
      if T:isprimitive() then
         return toPrimHType(T)
      elseif T:isarray() then
         local elemType = toHType(T.type)
         local arrayType = symbol(HDF5.hid_t, 'arrayType')
         header:insert(quote
            var dims : HDF5.hsize_t[1]
            dims[0] = T.N
            var elemType = [elemType]
            var [arrayType] = HDF5.H5Tarray_create2(elemType, 1, dims)
            if arrayType < 0 then [err('create array type')] end
         end)
         footer:insert(quote
            HDF5.H5Tclose(arrayType)
         end)
         return arrayType
      elseif T:isstruct() then
         local datatype = symbol(HDF5.hid_t, 'dataType')
         header:insert(quote
            var [datatype] = HDF5.H5Tcreate(HDF5.H5T_COMPOUND, [sizeof(T)]);
            if datatype < 0 then [err('create struct type', name)] end
         end)
         footer:insert(quote
            HDF5.H5Tclose(datatype)
         end)
         local offset = 0
         for _,e in ipairs(T.entries) do
            local name, type = UTIL.parseStructEntry(e)
            if type:isstruct() then
               toHType(type, prefix..name..'.')
            else
               local hType = toHType(type)
               header:insert(quote
                  var s = HDF5.H5Tinsert(datatype, name, offset, hType);
                  if s < 0 then [err('inserting type', prefix..name)] end
               end)
               offset = offset + sizeof(type)
            end
         end
         return datatype
      else assert(false) end
   end
   local HdataType = toHType(aType, '')


   local terra write(fname : &int8, aVal : aType)
      var [fid] = HDF5.H5Fopen(fname, HDF5.H5F_ACC_RDWR, HDF5.H5P_DEFAULT)
      if fid < 0 then [err('open file for attribute writing')] end
      var [sid] = HDF5.H5Screate(HDF5.H5S_SCALAR)
      if sid < 0 then [err('create attribute dataspace')] end
      [header];
      var aid = HDF5.H5Acreate2(fid, aName, [HdataType], sid,
                                HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
      if aid < 0 then [err('create attribute')] end
      var res = HDF5.H5Awrite(aid, [HdataType], &aVal)
      if res < 0 then [err('write attribute')] end
      [footer:reverse()];
      HDF5.H5Aclose(aid)
      HDF5.H5Sclose(sid)
      HDF5.H5Fclose(fid)
   end

   local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
   task writeTileAttr(_ : int,
                      dirname : regentlib.string,
                      aVal : aType)
      var filename = [&int8](C.malloc(256))
      C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
      write(filename, aVal)
      C.free(filename)
      return _
   end

   local __demand(__inline)
   task writeAttr(_ : int,
                  dirname : regentlib.string,
                  aVal : aType)
      return writeTileAttr(_, dirname, aVal)
   end
   MODULE.write[aName] = writeAttr

   local terra read(fname : &int8) : aType
      var [fid] = HDF5.H5Fopen(fname, HDF5.H5F_ACC_RDONLY, HDF5.H5P_DEFAULT)
      if fid < 0 then [err('open file for attribute writing')] end
      var aid = HDF5.H5Aopen(fid, aName, HDF5.H5P_DEFAULT)
      if aid < 0 then [err('open attribute')] end
      [header];
      var aVal : aType
      var res = HDF5.H5Aread(aid,  [HdataType], &aVal)
      if res < 0 then [err('read attribute')] end
      [footer:reverse()];
      HDF5.H5Aclose(aid)
      HDF5.H5Fclose(fid)
      return aVal
   end

   local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
   task readTileAttr(dirname : regentlib.string,
                     r : region(ispace(indexType), fSpace))
      var aVal : aType
      var filename = [&int8](C.malloc(256))
      C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
      var file = C.fopen(filename, "r")
      if file == nil then
         -- Keep for backward compatibility
         var filenameTile = tileFilename([&int8](dirname), r.bounds)
         aVal = read(filenameTile)
      else
         -- Master file is there
         C.fclose(file)
         aVal = read(filename)
      end
      C.free(filename)
      return aVal
   end

   local __demand(__inline)
   task readAttr(colors : ispace(colorType),
                 dirname : regentlib.string,
                 r : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors))
      -- TODO: Sanity checks: all files should have the same attribute value
      return readTileAttr(dirname, p_r[firstColor])
   end
   MODULE.read[aName] = readAttr

end

for aName,aPar in pairs(StringAttrs) do

   local aNum    = aPar[1]    -- Number of the written strings
   local aLength = aPar[2]+1  -- Length of the written strings (allow for trailing character of C)

   local terra write(fname : &int8, Strings : regentlib.string[aNum])
      var fid = HDF5.H5Fopen(fname, HDF5.H5F_ACC_RDWR, HDF5.H5P_DEFAULT)
      if fid < 0 then [err('open file for string attribute writing')] end
      var aSize : HDF5.hsize_t[1]
      aSize[0] = aNum
      var sid = HDF5.H5Screate_simple(1, aSize, [&uint64](0))
      if sid < 0 then [err('create string attribute dataspace')] end
      var stringType = HDF5.H5Tcopy(HDF5.H5T_C_S1_g)
      var res = HDF5.H5Tset_size (stringType, aLength)
      if res < 0 then [err('set attribute size')] end
      var aid = HDF5.H5Acreate2(fid, aName, stringType, sid,
                                HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
      if aid < 0 then [err('create attribute')] end
      var attr : int8[aLength][aNum]
      for i = 0, aNum do
         C.snprintf(attr[i], aLength, Strings[i])
      end
      res = HDF5.H5Awrite(aid, stringType, &attr)
      if res < 0 then [err('write attribute')] end
      HDF5.H5Aclose(aid)
      HDF5.H5Tclose(stringType)
      HDF5.H5Sclose(sid)
      HDF5.H5Fclose(fid)
   end

   local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
   task writeTileAttr(_ : int,
                      dirname : regentlib.string,
                      Strings : regentlib.string[aNum])
      var filename = [&int8](C.malloc(256))
      C.snprintf(filename, 256, '%s/master.hdf', [&int8](dirname))
      write(filename, Strings)
      C.free(filename)
      return _
   end

   local __demand(__inline)
   task writeAttr(_ : int,
                  dirname : regentlib.string,
                  Strings : regentlib.string[aNum])
      return writeTileAttr(_, dirname, Strings)
   end
   MODULE.write[aName] = writeAttr

end

-------------------------------------------------------------------------------
-- MODULE END
-------------------------------------------------------------------------------

return MODULE end
