local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]
local JSON = terralib.includec('json.h')
local UTIL = require 'util-desugared'

-------------------------------------------------------------------------------

-- NOTE: Type constructors are placed in the global namespace, so the schema
-- file can access them without imports.

local isStruct
local isSchemaT

-------------------------------------------------------------------------------

local StringMT = {}
StringMT.__index = StringMT

-- int -> String
function String(maxLen)
  assert(UTIL.isPosInt(maxLen))
  return setmetatable({
    maxLen = maxLen,
  }, StringMT)
end

-- SchemaT -> bool
local function isString(typ)
  return type(typ) == 'table' and getmetatable(typ) == StringMT
end

-------------------------------------------------------------------------------

local EnumMT = {}
EnumMT.__index = EnumMT

-- string* -> Enum
function Enum(...)
  local enum = {}
  for i,choice in ipairs({...}) do
    assert(type(choice) == 'string')
    enum[choice] = i-1
  end
  return setmetatable(enum, EnumMT)
end

-- SchemaT -> bool
local function isEnum(typ)
  return type(typ) == 'table' and getmetatable(typ) == EnumMT
end

-------------------------------------------------------------------------------

local ArrayMT = {}
ArrayMT.__index = ArrayMT

-- int, SchemaT -> Array
function Array(num, elemType)
  assert(UTIL.isPosInt(num))
  assert(isSchemaT(elemType))
  return setmetatable({
    num = num,
    elemType = elemType,
  }, ArrayMT)
end

-- SchemaT -> bool
local function isArray(typ)
  return type(typ) == 'table' and getmetatable(typ) == ArrayMT
end

-------------------------------------------------------------------------------

local UpToMT = {}
UpToMT.__index = UpToMT

-- int, SchemaT -> UpTo
function UpTo(max, elemType)
  assert(UTIL.isPosInt(max))
  assert(isSchemaT(elemType))
  return setmetatable({
    max = max,
    elemType = elemType,
  }, UpToMT)
end

-- SchemaT -> bool
local function isUpTo(typ)
  return type(typ) == 'table' and getmetatable(typ) == UpToMT
end

-------------------------------------------------------------------------------

local UnionMT = {}
UnionMT.__index = UnionMT

-- map(string,Struct) -> Union
function Union(choices)
  for name,fields in pairs(choices) do
    assert(type(name) == 'string')
    assert(isStruct(fields))
  end
  return setmetatable(UTIL.copyTable(choices), UnionMT)
end

-- SchemaT -> bool
local function isUnion(typ)
  return type(typ) == 'table' and getmetatable(typ) == UnionMT
end

-------------------------------------------------------------------------------

-- Struct = map(string,SchemaT)

-- SchemaT -> bool
function isStruct(typ)
  if type(typ) ~= 'table' or getmetatable(typ) ~= nil then
    return false
  end
  for k,v in pairs(typ) do
    if type(k) ~= 'string' or not isSchemaT(v) then
      return false
    end
  end
  return true
end

-------------------------------------------------------------------------------

-- SchemaT = 'bool' | 'int' | 'int64' | 'double' | String
--         | Enum | Array | UpTo | Union | Struct

-- A -> bool
function isSchemaT(typ)
  return
    typ == bool or
    typ == int or
    typ == int64 or
    typ == double or
    isString(typ) or
    isEnum(typ) or
    isArray(typ) or
    isUpTo(typ) or
    isUnion(typ) or
    isStruct(typ)
end

-------------------------------------------------------------------------------

-- SchemaT, map(SchemaT,terralib.type) -> terralib.type
local function convertSchemaT(typ, cache)
  if cache[typ] then
    return cache[typ]
  end
  if typ == bool then
    return bool
  elseif typ == int then
    return int
  elseif typ == int64 then
    return int64
  elseif typ == double then
    return double
  elseif isString(typ) then
    return int8[typ.maxLen]
  elseif isEnum(typ) then
    return int
  elseif isArray(typ) then
    return convertSchemaT(typ.elemType, cache)[typ.num]
  elseif isUpTo(typ) then
    return struct {
      length : uint32;
      values : convertSchemaT(typ.elemType, cache)[typ.max];
    }
  elseif isUnion(typ) then
    local u = terralib.types.newstruct()
    u.entries:insert(terralib.newlist())
    for n,t in pairs(typ) do
      u.entries[1]:insert({field=n, type=convertSchemaT(t, cache)})
    end
    local s = terralib.types.newstruct()
    s.entries:insert({field='type', type=int})
    s.entries:insert({field='u', type=u})
    return s
  elseif isStruct(typ) then
    local s = terralib.types.newstruct()
    if UTIL.isEmpty(typ) then
      s.entries:insert({field='__dummy', type=int})
    end
    for n,t in pairs(typ) do
      s.entries:insert({field=n, type=convertSchemaT(t, cache)})
    end
    return s
  else assert(false) end
end

-- string, terralib.expr* -> terralib.quote
local function errorOut(msg, ...)
  local args = {...}
  return quote
    var stderr = C.fdopen(2, 'w')
    C.fprintf(stderr, [msg..'\n'], [args])
    C.fflush(stderr)
    C.exit(1)
  end
end

-- string, terralib.expr -> terralib.quote
local function fldReadErr(msg, name)
  return errorOut(msg..' for field %s', name)
end

-- terralib.symbol, terralib.expr, terralib.expr, SchemaT -> terralib.quote
local function emitValueParser(name, lval, rval, typ)
  if typ == bool then
    return quote
      if [rval].type ~= JSON.json_boolean then
        [fldReadErr('Wrong type', name)]
      end
      [lval] = [bool]([rval].u.boolean)
    end
  elseif typ == int then
    return quote
      if [rval].type ~= JSON.json_integer then
        [fldReadErr('Wrong type', name)]
      end
      if [int]([rval].u.integer) ~= [rval].u.integer then
        [fldReadErr('Integer value overflow', name)]
      end
      [lval] = [rval].u.integer
    end
  elseif typ == int64 then
    return quote
      if [rval].type ~= JSON.json_integer then
        [fldReadErr('Wrong type', name)]
      end
      [lval] = [rval].u.integer
    end
  elseif typ == double then
    return quote
      if [rval].type ~= JSON.json_double then
        [fldReadErr('Wrong type', name)]
      end
      [lval] = [rval].u.dbl
    end
  elseif isString(typ) then
    return quote
      if [rval].type ~= JSON.json_string then
        [fldReadErr('Wrong type', name)]
      end
      if [rval].u.string.length >= [typ.maxLen] then
        [fldReadErr('String too long', name)]
      end
      C.strncpy([lval], [rval].u.string.ptr, [typ.maxLen])
    end
  elseif isEnum(typ) then
    return quote
      if [rval].type ~= JSON.json_string then
        [fldReadErr('Wrong type', name)]
      end
      var found = false
      escape for choice,value in pairs(typ) do emit quote
        if C.strcmp([rval].u.string.ptr, choice) == 0 then
          [lval] = value
          found = true
        end
      end end end
      if not found then
        [fldReadErr('Unexpected value', name)]
      end
    end
  elseif isArray(typ) then
    return quote
      if [rval].type ~= JSON.json_array then
        [fldReadErr('Wrong type', name)]
      end
      if [rval].u.array.length ~= [typ.num] then
        [fldReadErr('Wrong length', name)]
      end
      for i = 0,[typ.num] do
        var rval_i = [rval].u.array.values[i]
        [emitValueParser(name..'[i]', `[lval][i], rval_i, typ.elemType)]
      end
    end
  elseif isUpTo(typ) then
    return quote
      if [rval].type ~= JSON.json_array then
        [fldReadErr('Wrong type', name)]
      end
      if [rval].u.array.length > [typ.max] then
        [fldReadErr('Too many values', name)]
      end
      [lval].length = [rval].u.array.length
      for i = 0,[rval].u.array.length do
        var rval_i = [rval].u.array.values[i]
        [emitValueParser(name..'[i]', `[lval].values[i], rval_i, typ.elemType)]
      end
    end
  elseif isUnion(typ) then
    return quote
      if [rval].type ~= JSON.json_object then
        [fldReadErr('Wrong type', name)]
      end
      var foundType = false
      for i = 0,[rval].u.object.length do
        var nodeName = [rval].u.object.values[i].name
        var nodeValue = [rval].u.object.values[i].value
        if C.strcmp(nodeName, 'type') == 0 then
          foundType = true
          if nodeValue.type ~= JSON.json_string then
            [fldReadErr('Type field on union not a string', name)]
          end
          escape local j = 0; for choice,fields in pairs(typ) do emit quote
            if C.strcmp(nodeValue.u.string.ptr, [choice]) == 0 then
              [lval].type = [j]
              [emitValueParser(name, `[lval].u.[choice], rval, fields)]
              break
            end
          end j = j + 1; end end
          [fldReadErr('Unrecognized type on union', name)]
        end
      end
      if not foundType then
        [fldReadErr('Missing type field on union', name)]
      end
    end
  elseif isStruct(typ) then
    return quote
      var totalParsed = 0
      if [rval].type ~= JSON.json_object then
        [fldReadErr('Wrong type', name)]
      end
      for i = 0,[rval].u.object.length do
        var nodeName = [rval].u.object.values[i].name
        var nodeValue = [rval].u.object.values[i].value
        var parsed = false
        escape for fld,subTyp in pairs(typ) do emit quote
          if C.strcmp(nodeName, fld) == 0 then
            [emitValueParser(name..'.'..fld, `[lval].[fld], nodeValue, subTyp)]
            parsed = true
          end
        end end end
        if parsed then
          totalParsed = totalParsed + 1
        elseif C.strcmp(nodeName, 'type') ~= 0 then
          var stderr = C.fdopen(2, 'w')
          C.fprintf(
            stderr, ['Warning: Ignoring option '..name..'.%s\n'], nodeName)
        end
      end
      -- TODO: Assuming the json file contains no duplicate values
      if totalParsed < [UTIL.tableSize(typ)] then
        [errorOut('Missing fields from input file')]
      end
    end
  else assert(false) end
end

-------------------------------------------------------------------------------

local ext = '.lua'
if #arg < 1 or not arg[1]:endswith(ext) then
  print('Usage: '..arg[0]..' <schema'..ext..'>')
  os.exit(1)
end
local baseName = arg[1]:sub(1, arg[1]:len() - ext:len())

local SCHEMA = dofile(arg[1]) -- map(string,SchemaT)

local type2name = {} -- map(Struct,string)
for name,typ in pairs(SCHEMA) do
  if isStruct(typ) then
    type2name[typ] = name
  end
end

local deps = {} -- map(string,string*)
for srcN,srcT in pairs(SCHEMA) do
  if isStruct(srcT) then
    deps[srcN] = terralib.newlist()
    local function traverse(tgtT)
      local tgtN = type2name[tgtT]
      if tgtN and srcN ~= tgtN then
        deps[srcN]:insert(tgtN)
        return
      end
      if tgtT == bool then
        -- Do nothing
      elseif tgtT == int then
        -- Do nothing
      elseif tgtT == int64 then
        -- Do nothing
      elseif tgtT == double then
        -- Do nothing
      elseif isString(tgtT) then
        -- Do nothing
      elseif isEnum(tgtT) then
        -- Do nothing
      elseif isArray(tgtT) then
        traverse(tgtT.elemType)
      elseif isUpTo(tgtT) then
        traverse(tgtT.elemType)
      elseif isUnion(tgtT) then
        for n,t in pairs(tgtT) do
          traverse(t)
        end
      elseif isStruct(tgtT) then
        for n,t in pairs(tgtT) do
          traverse(t)
        end
      else assert(false) end
    end
    traverse(srcT)
  end
end
local sortedNames = UTIL.revTopoSort(deps) -- string*

local type2terra = {} -- map(Struct,terralib.struct)
local parsers = {} -- map(string,terralib.function)
for _,name in ipairs(sortedNames) do
  local typ = SCHEMA[name]
  local st = convertSchemaT(typ, type2terra)
  st.name = name
  type2terra[typ] = st
  parsers['parse_'..name] = terra(output : &st, fname : &int8)
    var f = C.fopen(fname, 'r')
    if f == nil then [errorOut('Cannot open %s', fname)] end
    var res1 = C.fseek(f, 0, C.SEEK_END)
    if res1 ~= 0 then [errorOut('Cannot seek to end of %s', fname)] end
    var len = C.ftell(f)
    if len < 0 then [errorOut('Cannot ftell %s', fname)] end
    var res2 = C.fseek(f, 0, C.SEEK_SET)
    if res2 ~= 0 then [errorOut('Cannot seek to start of %s', fname)] end
    var buf = [&int8](C.malloc(len))
    if buf == nil then [errorOut('Malloc error while parsing %s', fname)] end
    var res3 = C.fread(buf, 1, len, f)
    if res3 < len then [errorOut('Cannot read from %s', fname)] end
    C.fclose(f)
    var errMsg : int8[256]
    var settings = JSON.json_settings{ 0, 0, nil, nil, nil, 0 }
    settings.settings = JSON.json_enable_comments
    var root = JSON.json_parse_ex(&settings, buf, len, errMsg)
    if root == nil then [errorOut('JSON parsing error: %s', errMsg)] end
    [emitValueParser(name, output, root, typ)]
    JSON.json_value_free(root)
    C.free(buf)
  end
end

local hdrFile = io.open(baseName..'.h', 'w')
hdrFile:write('// DO NOT EDIT THIS FILE, IT IS AUTOMATICALLY GENERATED\n')
hdrFile:write('\n')
hdrFile:write('#ifndef __'..string.upper(baseName)..'_H__\n')
hdrFile:write('#define __'..string.upper(baseName)..'_H__\n')
hdrFile:write('\n')
hdrFile:write('#include <stdbool.h>\n')
hdrFile:write('#include <stdint.h>\n')
for name,typ in pairs(SCHEMA) do
  if isEnum(typ) then
    hdrFile:write('\n')
    hdrFile:write('typedef int '..name..';\n')
    for choice,value in pairs(typ) do
      hdrFile:write('#define '..name..'_'..choice..' '..value..'\n')
    end
  elseif isUnion(typ) then
    hdrFile:write('\n')
    hdrFile:write('typedef int '..name..';\n')
    local i = 0
    for choice,_ in pairs(typ) do
      hdrFile:write('#define '..name..'_'..choice..' '..i..'\n')
      i = i + 1
    end
  end
end
for _,name in ipairs(sortedNames) do
  local st = type2terra[SCHEMA[name]]
  hdrFile:write('\n')
  hdrFile:write(UTIL.prettyPrintStruct(st, true)..';\n')
  hdrFile:write('\n')
  hdrFile:write('#ifdef __cplusplus\n')
  hdrFile:write('extern "C" {\n')
  hdrFile:write('#endif\n')
  hdrFile:write('void parse_'..name..'(struct '..name..'*, char*);\n')
  hdrFile:write('#ifdef __cplusplus\n')
  hdrFile:write('}\n')
  hdrFile:write('#endif\n')
end
hdrFile:write('\n')
hdrFile:write('#endif // __'..string.upper(baseName)..'_H__\n')
hdrFile:close()

terralib.saveobj(baseName..'.o', 'object', parsers)
