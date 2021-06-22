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

-------------------------------------------------------------------------------
-- FALLBACK MODE
-------------------------------------------------------------------------------

local USE_HDF = assert(os.getenv('USE_HDF')) ~= '0'

if not USE_HDF then

  __demand(__inline)
  task MODULE.dump(_ : int,
                   colors : ispace(colorType),
                   dirname : &int8,
                   r : region(ispace(indexType), fSpace),
                   s : region(ispace(indexType), fSpace),
                   p_r : partition(disjoint, r, colors),
                   p_s : partition(disjoint, s, colors))
  where reads(r.[flds]), reads writes(s.[flds]), r * s do
    regentlib.assert(false, 'Recompile with USE_HDF=1')
    return _
  end

  __demand(__inline)
  task MODULE.load(_ : int,
                   colors : ispace(colorType),
                   dirname : &int8,
                   r : region(ispace(indexType), fSpace),
                   s : region(ispace(indexType), fSpace),
                   p_r : partition(disjoint, r, colors),
                   p_s : partition(disjoint, s, colors))
  where reads writes(r.[flds]), reads writes(s.[flds]), r * s do
    regentlib.assert(false, 'Recompile with USE_HDF=1')
    return _
  end

  for aName,aType in pairs(attrs) do

    local __demand(__inline)
    task writeAttr(_ : int,
                   colors : ispace(colorType),
                   dirname : &int8,
                   r : region(ispace(indexType), fSpace),
                   p_r : partition(disjoint, r, colors),
                   aVal : aType)
      regentlib.assert(false, 'Recompile with USE_HDF=1')
      return _
    end
    MODULE.write[aName] = writeAttr

    local __demand(__inline)
    task readAttr(_ : int,
                  colors : ispace(colorType),
                  dirname : &int8,
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
                   dirname : &int8,
                   r : region(ispace(indexType), fSpace),
                   p_r : partition(disjoint, r, colors),
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
local UTIL = require 'util-desugared'

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------

-- HACK: Hardcoding missing #define's
HDF5.H5F_ACC_RDONLY = 0
HDF5.H5F_ACC_RDWR = 1
HDF5.H5F_ACC_TRUNC = 2
HDF5.H5P_DEFAULT = 0

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
            var [dataSet] = HDF5.H5Dcreate2(
              fid, hName, int2dType, dataSpace,
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
            var [dataSet] = HDF5.H5Dcreate2(
              fid, hName, int3dType, dataSpace,
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
  __demand(__inline) task tileFilename(dirname : &int8, bounds : rect1d)
    var filename = [&int8](C.malloc(256))
    var lo = bounds.lo
    var hi = bounds.hi
    C.snprintf(filename, 256,
               '%s/%ld-%ld.hdf', dirname,
               lo, hi)
    return filename
  end
elseif indexType == int2d then
  __demand(__inline) task tileFilename(dirname : &int8, bounds : rect2d)
    var filename = [&int8](C.malloc(256))
    var lo = bounds.lo
    var hi = bounds.hi
    C.snprintf(filename, 256,
               '%s/%ld,%ld-%ld,%ld.hdf', dirname,
               lo.x, lo.y, hi.x, hi.y)
    return filename
  end
elseif indexType == int3d then
  __demand(__inline) task tileFilename(dirname : &int8, bounds : rect3d)
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

local one =
  indexType == int1d and rexpr 1 end or
  indexType == int2d and rexpr {1,1} end or
  indexType == int3d and rexpr {1,1,1} end or
  assert(false)

-------------------------------------------------------------------------------
-- EXPORTED TASKS
-------------------------------------------------------------------------------

local __demand(__inner) -- NOT LEAF, MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task dumpTile(_ : int,
              dirname : regentlib.string,
              r : region(ispace(indexType), fSpace),
              s : region(ispace(indexType), fSpace))
where reads(r.[flds]), reads writes(s.[flds]), r * s do
  if r.volume == 0 then return _ end
  var filename = tileFilename([&int8](dirname), r.bounds)
  create(filename, r.bounds.hi - r.bounds.lo + one)
  attach(hdf5, s.[flds], filename, regentlib.file_read_write)
  acquire(s.[flds])
  copy(r.[flds], s.[flds])
  release(s.[flds])
  detach(hdf5, s.[flds])
  C.free(filename)
  return _
end

__demand(__inline)
task MODULE.dump(_ : int,
                 colors : ispace(colorType),
                 dirname : &int8,
                 r : region(ispace(indexType), fSpace),
                 s : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors),
                 p_s : partition(disjoint, s, colors))
where reads(r.[flds]), reads writes(s.[flds]), r * s do
  -- TODO: Sanity checks: bounds.lo == 0, same size, compatible partitions
  var __ = 0
  for c in colors do
    __ += dumpTile(_, dirname, p_r[c], p_s[c])
  end
  return __
end

local __demand(__inner) -- NOT LEAF, MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
task loadTile(_ : int,
              dirname : regentlib.string,
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
  return _
end

__demand(__inline)
task MODULE.load(_ : int,
                 colors : ispace(colorType),
                 dirname : &int8,
                 r : region(ispace(indexType), fSpace),
                 s : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors),
                 p_s : partition(disjoint, s, colors))
where reads writes(r.[flds]), reads writes(s.[flds]), r * s do
  -- TODO: Sanity checks: bounds.lo == 0, same size, compatible partitions
  -- TODO: Check that the file has the correct size etc.
  var __ = 0
  for c in colors do
    __ += loadTile(_, dirname, p_r[c], p_s[c])
  end
  return __
end

for aName,aType in pairs(attrs) do

  local terra write(fname : &int8, aVal : aType)
    var fid = HDF5.H5Fopen(fname, HDF5.H5F_ACC_RDWR, HDF5.H5P_DEFAULT)
    if fid < 0 then [err('open file for attribute writing')] end
    var sid = HDF5.H5Screate(HDF5.H5S_SCALAR)
    if sid < 0 then [err('create attribute dataspace')] end
    var aid = HDF5.H5Acreate2(fid, aName, [toPrimHType(aType)], sid,
                              HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
    if aid < 0 then [err('create attribute')] end
    var res = HDF5.H5Awrite(aid, [toPrimHType(aType)], &aVal)
    if res < 0 then [err('write attribute')] end
    HDF5.H5Aclose(aid)
    HDF5.H5Sclose(sid)
    HDF5.H5Fclose(fid)
  end

  local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
  task writeTileAttr(_ : int,
                     dirname : regentlib.string,
                     r : region(ispace(indexType), fSpace),
                     aVal : aType)
    if r.volume == 0 then return _ end
    var filename = tileFilename([&int8](dirname), r.bounds)
    write(filename, aVal)
    return _
  end

  local __demand(__inline)
  task writeAttr(_ : int,
                 colors : ispace(colorType),
                 dirname : &int8,
                 r : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors),
                 aVal : aType)
    var __ = 0
    for c in colors do
      __ += writeTileAttr(_, dirname, p_r[c], aVal)
    end
    return __
  end
  MODULE.write[aName] = writeAttr

  local terra read(fname : &int8) : aType
    var fid = HDF5.H5Fopen(fname, HDF5.H5F_ACC_RDONLY, HDF5.H5P_DEFAULT)
    if fid < 0 then [err('open file for attribute writing')] end
    var aid = HDF5.H5Aopen(fid, aName, HDF5.H5P_DEFAULT)
    if aid < 0 then [err('open attribute')] end
    var aVal : aType
    var res = HDF5.H5Aread(aid, [toPrimHType(aType)], &aVal)
    if res < 0 then [err('read attribute')] end
    HDF5.H5Aclose(aid)
    HDF5.H5Fclose(fid)
    return aVal
  end

  local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
  task readTileAttr(_ : int,
                    dirname : regentlib.string,
                    r : region(ispace(indexType), fSpace))
    var filename = tileFilename([&int8](dirname), r.bounds)
    return read(filename)
  end

  local __demand(__inline)
  task readAttr(_ : int,
                colors : ispace(colorType),
                dirname : &int8,
                r : region(ispace(indexType), fSpace),
                p_r : partition(disjoint, r, colors))
    -- TODO: Sanity checks: all files should have the same attribute value
    return readTileAttr(_, dirname, p_r[firstColor])
  end
  MODULE.read[aName] = readAttr

end

for aName,aPar in pairs(StringAttrs) do

  local aNum    = aPar[1]    -- Number of the written strings
  local aLength = aPar[2]+1  -- Length of the written strings (allow for traling characted of C)

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
    HDF5.H5Sclose(sid)
    HDF5.H5Fclose(fid)
  end

  local __demand(__leaf) -- MANUALLY PARALLELIZED, NO CUDA, NO OPENMP
  task writeTileAttr(_ : int,
                     dirname : regentlib.string,
                     r : region(ispace(indexType), fSpace),
                     Strings : regentlib.string[aNum])
    if r.volume == 0 then return _ end
    var filename = tileFilename([&int8](dirname), r.bounds)
    write(filename, Strings)
    return _
  end

  local __demand(__inline)
  task writeAttr(_ : int,
                 colors : ispace(colorType),
                 dirname : &int8,
                 r : region(ispace(indexType), fSpace),
                 p_r : partition(disjoint, r, colors),
                 Strings : regentlib.string[aNum])
    var __ = 0
    for c in colors do
      __ += writeTileAttr(_, dirname, p_r[c], Strings)
    end
    return __
  end
  MODULE.write[aName] = writeAttr

end

-------------------------------------------------------------------------------
-- MODULE END
-------------------------------------------------------------------------------

return MODULE end
