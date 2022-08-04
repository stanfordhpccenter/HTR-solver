import 'regent'

local Exports = {}

local C = terralib.includecstring([[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
]])

-------------------------------------------------------------------------------
-- Arrays
-------------------------------------------------------------------------------

-- N, val -> array(val, val, ..., val)
function Exports.mkArrayConstant(N, val)
   local vals = terralib.newlist()
   for i = 1,N do
      vals:insert(val)
   end
   return rexpr array([vals]) end
end

-------------------------------------------------------------------------------
-- Numeric
-------------------------------------------------------------------------------

-- A -> bool
function Exports.isInt(x)
  return type(x) == 'number' and x == math.floor(x)
end

-- A -> bool
function Exports.isPosInt(x)
  return Exports.isInt(x) and x > 0
end

-------------------------------------------------------------------------------
-- Tables
-------------------------------------------------------------------------------

-- table -> bool
function Exports.isEmpty(tbl)
  if not tbl then
    return true
  end
  for _,_ in pairs(tbl) do
    return false
  end
  return true
end

-- map(K,V) -> K*
function Exports.keys(tbl)
  local res = terralib.newlist()
  for k,v in pairs(tbl) do
    res:insert(k)
  end
  return res
end

-- T*, (T -> bool) -> bool
function Exports.all(list, pred)
  assert(terralib.israwlist(list))
  for _,x in ipairs(list) do
    if not pred(x) then return false end
  end
  return true
end

-- T*, (T -> bool) -> bool
function Exports.any(list, pred)
  assert(terralib.israwlist(list))
  for _,x in ipairs(list) do
    if pred(x) then return true end
  end
  return false
end

-- table -> table
function Exports.copyTable(tbl)
  local cpy = {}
  for k,v in pairs(tbl) do cpy[k] = v end
  return cpy
end

-- table -> int
function Exports.tableSize(tbl)
  local size = 0
  for _,_ in pairs(tbl) do
    size = size + 1
  end
  return size
end

-------------------------------------------------------------------------------
-- Lists
-------------------------------------------------------------------------------

local TerraList = getmetatable(terralib.newlist())

-- () -> set(T)
function TerraList:toSet()
  local set = {}
  for _,elem in ipairs(self) do
    set[elem] = true
  end
  return set
end

-- string? -> string
function TerraList:join(sep)
  sep = sep or ' '
  local res = ''
  for i,elem in ipairs(self) do
    if i > 1 then
      res = res..sep
    end
    res = res..tostring(elem)
  end
  return res
end

-- () -> T*
function TerraList:reverse()
  local res = terralib.newlist()
  for i = #self, 1, -1 do
    res:insert(self[i])
  end
  return res
end

-- int, (() -> T) -> T*
function Exports.generate(n, generator)
  local res = terralib.newlist()
  for i = 1,n do
    res:insert(generator())
  end
  return res
end

-- int, int -> T*
function Exports.range(first, last)
  local res = terralib.newlist()
  for i = first,last do
    res:insert(i)
  end
  return res
end

-------------------------------------------------------------------------------
-- Sets
-------------------------------------------------------------------------------

-- set(T) -> T*
function Exports.setToList(set)
  local list = terralib.newlist()
  for e,_ in pairs(set) do
    list:insert(e)
  end
  return list
end

-- T* -> set(T)
function Exports.listToSet(list)
  local set = {}
  for _,e in ipairs(list) do
    set[e] = true
  end
  return set
end

-- set(T) -> T
function Exports.setPop(set)
  local elem
  for e,_ in pairs(set) do
    elem = e
    break
  end
  set[elem] = nil
  return elem
end

-- set(T), T* -> set(T)
function Exports.setSubList(set, list)
  local res = Exports.copyTable(set)
  for _,e in ipairs(list) do
    res[e] = nil
  end
  return res
end

-------------------------------------------------------------------------------
-- Strings
-------------------------------------------------------------------------------

-- string -> string*
function string:split(sep)
  local fields = terralib.newlist()
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) fields:insert(c) end)
  return fields
end

-- string -> bool
function string:startswith(subStr)
  return self:sub(1, subStr:len()) == subStr
end

-- string -> bool
function string:endswith(subStr)
  return self:sub(self:len() - subStr:len() + 1, self:len()) == subStr
end

-------------------------------------------------------------------------------
-- Terra type helpers
-------------------------------------------------------------------------------

-- {string,terralib.type} | {field:string,type:terralib.type} ->
--   string, terralib.type
function Exports.parseStructEntry(entry)
  if terralib.israwlist(entry)
  and #entry == 2
  and type(entry[1]) == 'string'
  and terralib.types.istype(entry[2]) then
    return entry[1], entry[2]
  elseif type(entry) == 'table'
  and entry.field
  and entry.type then
    return entry.field, entry.type
  else assert(false) end
end

-- map(terralib.type,string)
local cBaseType = {
  [int]    = 'int',
  [int8]   = 'int8_t',
  [int16]  = 'int16_t',
  [int32]  = 'int32_t',
  [int64]  = 'int64_t',
  [uint]   = 'unsigned',
  [uint8]  = 'uint8_t',
  [uint16] = 'uint16_t',
  [uint32] = 'uint32_t',
  [uint64] = 'uint64_t',
  [bool]   = 'bool',
  [float]  = 'float',
  [double] = 'double',
}

-- terralib.type, bool, string -> string, string
local function typeDecl(typ, cStyle, indent)
  if typ:isarray() then
    local decl, mods = typeDecl(typ.type, cStyle, indent)
    decl = cStyle and decl or decl..'['..tostring(typ.N)..']'
    mods = cStyle and '['..tostring(typ.N)..']'..mods or mods
    return decl, mods
  elseif typ:isstruct() then
    if typ.name:startswith('anon') then
      return Exports.prettyPrintStruct(typ, cStyle, indent), ''
    elseif cStyle then
      return ('struct '..typ.name), ''
    else
      return typ.name, ''
    end
  elseif typ:isprimitive() then
    return (cStyle and cBaseType[typ] or tostring(typ)), ''
  else assert(false) end
end

-- terralib.struct, bool?, string? -> string
function Exports.prettyPrintStruct(s, cStyle, indent)
  indent = indent or ''
  local lines = terralib.newlist()
  local isUnion = false
  local entries = s.entries
  if #entries == 1 and terralib.israwlist(entries[1]) then
    isUnion = true
    entries = entries[1]
  end
  local name = s.name:startswith('anon') and '' or s.name
  local open =
    (cStyle and isUnion)         and ('union '..name..' {')          or
    (cStyle and not isUnion)     and ('struct '..name..' {')         or
    (not cStyle and isUnion)     and ('struct '..name..' { union {') or
    (not cStyle and not isUnion) and ('struct '..name..' {')         or
    assert(false)
  lines:insert(open)
  for _,e in ipairs(entries) do
    local name, typ = Exports.parseStructEntry(e)
    local s1 = cStyle and '' or (name..' : ')
    local s3 = cStyle and (' '..name) or ''
    local s2, s4 = typeDecl(typ, cStyle, indent..'  ')
    lines:insert(indent..'  '..s1..s2..s3..s4..';')
  end
  local close =
    (cStyle and isUnion)         and '}'   or
    (cStyle and not isUnion)     and '}'   or
    (not cStyle and isUnion)     and '} }' or
    (not cStyle and not isUnion) and '}'   or
    assert(false)
  lines:insert(indent..close)
  return lines:join('\n')
end

-- terralib.struct -> string*
function Exports.fieldNames(s)
  return s.entries:map(function(e)
    local name,typ = Exports.parseStructEntry(e)
    return name
  end)
end

-- string, terralib.struct, string*, map(string,terralib.type)?
--   -> terralib.struct
function Exports.deriveStruct(newName, origStruct, keptFlds, addedFlds)
  addedFlds = addedFlds or {}
  local origFlds = {}
  for _,e in ipairs(origStruct.entries) do
    local name,typ = Exports.parseStructEntry(e)
    assert(not addedFlds[name])
    origFlds[name] = typ
  end
  local newStruct = terralib.types.newstruct(newName)
  for _,name in ipairs(keptFlds) do
    local typ = assert(origFlds[name])
    newStruct.entries:insert({name,typ})
  end
  for name,typ in pairs(addedFlds) do
    newStruct.entries:insert({name,typ})
  end
  return newStruct
end

-------------------------------------------------------------------------------
-- Filesystem
-------------------------------------------------------------------------------

terra Exports.createDir(name : &int8)
  var mode = 493 -- octal 0755 = rwxr-xr-x
  var res = C.mkdir(name, mode)
  if res < 0 then
    var stderr = C.fdopen(2, 'w')
    C.fprintf(stderr, 'Cannot create directory %s: ', name)
    C.fflush(stderr)
    C.perror('')
    C.fflush(stderr)
    C.exit(1)
  end
end

terra Exports.openFile(name : &int8, mode : &int8) : &C.FILE
  var file = C.fopen(name, mode)
  if file == nil then
    var stderr = C.fdopen(2, 'w')
    C.fprintf(stderr, 'Cannot open file %s in mode "%s": ', name, mode)
    C.fflush(stderr)
    C.perror('')
    C.fflush(stderr)
    C.exit(1)
  end
  return file
end

-------------------------------------------------------------------------------
-- Graphs
-------------------------------------------------------------------------------

-- Graph(T) = map(T,T*)

-- Graph(T) -> set(T)
function Exports.getNodes(graph)
  local nodes = {} -- set(T)
  for src,tgts in pairs(graph) do
    nodes[src] = true
    for _,tgt in ipairs(tgts) do
      nodes[tgt] = true
    end
  end
  return nodes
end

-- Graph(T) -> T*
function Exports.revTopoSort(graph)
  local unmarked = Exports.getNodes(graph) -- set(T)
  local tempMarked = {} -- set(T)
  local permMarked = {} -- set(T)
  local res = terralib.newlist() -- T*
  local function visit(src)
    if permMarked[src] then
      return true
    end
    if tempMarked[src] then
      return false
    end
    tempMarked[src] = true
    for _,tgt in ipairs(graph[src] or {}) do
      visit(tgt)
    end
    permMarked[src] = true
    res:insert(src)
    return true
  end
  while not Exports.isEmpty(unmarked) do
    if not visit(Exports.setPop(unmarked)) then
      return nil
    end
  end
  return res
end

-------------------------------------------------------------------------------
-- Regent metaprogramming
-------------------------------------------------------------------------------

-- regentlib.symbol, int, regentlib.rexpr, terralib.type -> regentlib.rquote
function Exports.emitRegionTagAttach(r, tag, value, typ)
  return rquote
    var info : typ[1]
    info[0] = value
    regentlib.c.legion_logical_region_attach_semantic_information(
      __runtime(), __raw(r), tag, [&typ](info), [sizeof(typ)], false)
  end
end

-- regentlib.symbol, string, terralib.type -> regentlib.rquote
function Exports.emitPartitionNameAttach(p, name)
  return rquote
    regentlib.c.legion_logical_partition_attach_name(
      __runtime(), __raw(p), name, false)
  end
end

-- intXd, intXd, terralib.struct -> regentlib.task
function Exports.mkPartitionByTile(r_istype, cs_istype, fs, name)
  local partitionByTile
  local p = regentlib.newsymbol(name)
--  if r_istype == int4d and cs_istype == int4d then
--    __demand(__inline)
--    task partitionByTile(r : region(ispace(int4d), fs),
--                         cs : ispace(int4d),
--                         halo : int4d,
--                         offset : int4d)
--      var Nx = r.bounds.hi.x - 2*halo.x + 1; var ntx = cs.bounds.hi.x + 1
--      var Ny = r.bounds.hi.y - 2*halo.y + 1; var nty = cs.bounds.hi.y + 1
--      var Nz = r.bounds.hi.z - 2*halo.z + 1; var ntz = cs.bounds.hi.z + 1
--      var Nw = r.bounds.hi.w - 2*halo.w + 1; var ntw = cs.bounds.hi.w + 1
--      regentlib.assert(r.bounds.lo == int4d{0,0,0,0}, "Can only partition root region")
--      regentlib.assert(Nx % ntx == 0, "Uneven partitioning on x")
--      regentlib.assert(Ny % nty == 0, "Uneven partitioning on y")
--      regentlib.assert(Nz % ntz == 0, "Uneven partitioning on z")
--      regentlib.assert(Nw % ntw == 0, "Uneven partitioning on w")
--      regentlib.assert(-ntx <= offset.x and offset.x <= ntx, "offset.x too large")
--      regentlib.assert(-nty <= offset.y and offset.y <= nty, "offset.y too large")
--      regentlib.assert(-ntz <= offset.z and offset.z <= ntz, "offset.z too large")
--      regentlib.assert(-ntw <= offset.w and offset.w <= ntw, "offset.w too large")
--      var coloring = regentlib.c.legion_domain_point_coloring_create()
--      for c_real in cs do
--        var c = (c_real - offset + {ntx,nty,ntz,ntw}) % {ntx,nty,ntz,ntw}
--        var rect = rect4d{
--          lo = int4d{halo.x + (Nx/ntx)*(c.x),
--                     halo.y + (Ny/nty)*(c.y),
--                     halo.z + (Nz/ntz)*(c.z),
--                     halo.w + (Nz/ntz)*(c.w)},
--          hi = int4d{halo.x + (Nx/ntx)*(c.x+1) - 1,
--                     halo.y + (Ny/nty)*(c.y+1) - 1,
--                     halo.z + (Nz/ntz)*(c.z+1) - 1,
--                     halo.w + (Nw/ntw)*(c.w+1) - 1}}
--        if c.x == 0 then rect.lo.x -= halo.x end
--        if c.y == 0 then rect.lo.y -= halo.y end
--        if c.z == 0 then rect.lo.z -= halo.z end
--        if c.w == 0 then rect.lo.w -= halo.w end
--        if c.x == ntx-1 then rect.hi.x += halo.x end
--        if c.y == nty-1 then rect.hi.y += halo.y end
--        if c.z == ntz-1 then rect.hi.z += halo.z end
--        if c.w == ntw-1 then rect.hi.w += halo.w end
--        regentlib.c.legion_domain_point_coloring_color_domain(coloring, c_real, rect)
--      end
--      var [p] = partition(disjoint, r, coloring, cs)
--      regentlib.c.legion_domain_point_coloring_destroy(coloring)
--      return [p]
--    end
--  elseif r_istype == int3d and cs_istype == int3d then
  if r_istype == int3d and cs_istype == int3d then
    __demand(__inline)
    task partitionByTile(r : region(ispace(int3d), fs),
                         cs : ispace(int3d),
                         halo : int3d,
                         offset : int3d)
      var Nx = r.bounds.hi.x - 2*halo.x + 1; var ntx = cs.bounds.hi.x + 1
      var Ny = r.bounds.hi.y - 2*halo.y + 1; var nty = cs.bounds.hi.y + 1
      var Nz = r.bounds.hi.z - 2*halo.z + 1; var ntz = cs.bounds.hi.z + 1
      regentlib.assert(r.bounds.lo == int3d{0,0,0}, "Can only partition root region")
      regentlib.assert(Nx % ntx == 0, "Uneven partitioning on x")
      regentlib.assert(Ny % nty == 0, "Uneven partitioning on y")
      regentlib.assert(Nz % ntz == 0, "Uneven partitioning on z")
      regentlib.assert(-ntx <= offset.x and offset.x <= ntx, "offset.x too large")
      regentlib.assert(-nty <= offset.y and offset.y <= nty, "offset.y too large")
      regentlib.assert(-ntz <= offset.z and offset.z <= ntz, "offset.z too large")
      var coloring = regentlib.c.legion_domain_point_coloring_create()
      for c_real in cs do
        var c = (c_real - offset + {ntx,nty,ntz}) % {ntx,nty,ntz}
        var rect = rect3d{
          lo = int3d{halo.x + (Nx/ntx)*(c.x),
                     halo.y + (Ny/nty)*(c.y),
                     halo.z + (Nz/ntz)*(c.z)},
          hi = int3d{halo.x + (Nx/ntx)*(c.x+1) - 1,
                     halo.y + (Ny/nty)*(c.y+1) - 1,
                     halo.z + (Nz/ntz)*(c.z+1) - 1}}
        if c.x == 0 then rect.lo.x -= halo.x end
        if c.y == 0 then rect.lo.y -= halo.y end
        if c.z == 0 then rect.lo.z -= halo.z end
        if c.x == ntx-1 then rect.hi.x += halo.x end
        if c.y == nty-1 then rect.hi.y += halo.y end
        if c.z == ntz-1 then rect.hi.z += halo.z end
        regentlib.c.legion_domain_point_coloring_color_domain(coloring, c_real, rect)
      end
      var [p] = partition(disjoint, r, coloring, cs)
      regentlib.c.legion_domain_point_coloring_destroy(coloring)
      return [p]
    end
  elseif r_istype == int2d and cs_istype == int2d then
    __demand(__inline)
    task partitionByTile(r : region(ispace(int2d), fs),
                         cs : ispace(int2d),
                         halo : int2d,
                         offset : int2d)
      var Nx = r.bounds.hi.x - 2*halo.x + 1; var ntx = cs.bounds.hi.x + 1
      var Ny = r.bounds.hi.y - 2*halo.y + 1; var nty = cs.bounds.hi.y + 1
      regentlib.assert(r.bounds.lo == int2d{0,0}, "Can only partition root region")
      regentlib.assert(Nx % ntx == 0, "Uneven partitioning on x")
      regentlib.assert(Ny % nty == 0, "Uneven partitioning on y")
      regentlib.assert(-ntx <= offset.x and offset.x <= ntx, "offset.x too large")
      regentlib.assert(-nty <= offset.y and offset.y <= nty, "offset.y too large")
      var coloring = regentlib.c.legion_domain_point_coloring_create()
      for c_real in cs do
        var c = (c_real - offset + {ntx,nty}) % {ntx,nty}
        var rect = rect2d{
          lo = int2d{halo.x + (Nx/ntx)*(c.x),
                     halo.y + (Ny/nty)*(c.y)},
          hi = int2d{halo.x + (Nx/ntx)*(c.x+1) - 1,
                     halo.y + (Ny/nty)*(c.y+1) - 1}}
        if c.x == 0 then rect.lo.x -= halo.x end
        if c.y == 0 then rect.lo.y -= halo.y end
        if c.x == ntx-1 then rect.hi.x += halo.x end
        if c.y == nty-1 then rect.hi.y += halo.y end
        regentlib.c.legion_domain_point_coloring_color_domain(coloring, c_real, rect)
      end
      var [p] = partition(disjoint, r, coloring, cs)
      regentlib.c.legion_domain_point_coloring_destroy(coloring)
      return [p]
    end
  elseif r_istype == int1d and cs_istype == int3d then
    __demand(__inline)
    task partitionByTile(r : region(ispace(int1d), fs),
                         cs : ispace(int3d),
                         halo : int64,
                         offset : int3d)
      var N = int64(r.bounds.hi - 2*halo + 1)
      var ntx = cs.bounds.hi.x + 1
      var nty = cs.bounds.hi.y + 1
      var ntz = cs.bounds.hi.z + 1
      regentlib.assert(int64(r.bounds.lo) == 0, "Can only partition root region")
      regentlib.assert(N % (ntx*nty*ntz) == 0, "Uneven partitioning")
      regentlib.assert(-ntx <= offset.x and offset.x <= ntx, "offset.x too large")
      regentlib.assert(-nty <= offset.y and offset.y <= nty, "offset.y too large")
      regentlib.assert(-ntz <= offset.z and offset.z <= ntz, "offset.z too large")
      var coloring = regentlib.c.legion_domain_point_coloring_create()
      for c_real in cs do
        var c = (c_real - offset + {ntx,nty,ntz}) % {ntx,nty,ntz}
        var rect = rect1d{
          lo = halo + (N/ntx/nty/ntz)*(c.x*nty*ntz+c.y*ntz+c.z),
          hi = halo + (N/ntx/nty/ntz)*(c.x*nty*ntz+c.y*ntz+c.z+1) - 1}
        if c.x == 0 and
           c.y == 0 and
           c.z == 0 then rect.lo -= halo end
        if c.x == ntx-1 and
           c.y == nty-1 and
           c.z == ntz-1 then rect.hi += halo end
        regentlib.c.legion_domain_point_coloring_color_domain(coloring, c_real, rect)
      end
      var [p] = partition(disjoint, r, coloring, cs)
      regentlib.c.legion_domain_point_coloring_destroy(coloring)
      return [p]
    end
  else assert(false) end
  return partitionByTile
end

-- intXd, terralib.struct -> regentlib.task
function Exports.mkPartitionIsInteriorOrGhost(r_istype, fs, name)
   local partitionIsInteriorOrGhost
   local p = regentlib.newsymbol(name)
   if r_istype == int3d then
      __demand(__inline)
      task partitionIsInteriorOrGhost(r : region(ispace(int3d), fs), halo : int3d)
         var Nx = r.bounds.hi.x - 2*halo.x + 1
         var Ny = r.bounds.hi.y - 2*halo.y + 1
         var Nz = r.bounds.hi.z - 2*halo.z + 1
         regentlib.assert(r.bounds.lo == int3d{0,0,0}, "Can only partition root region")
         var coloring = regentlib.c.legion_domain_point_coloring_create()

         -- Interior points
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(0), rect3d{
            lo = int3d{halo.x,         halo.y,        halo.z        },
            hi = int3d{halo.x+Nx-1,    halo.y+Ny-1,   halo.z+Nz-1   }})

         -- xNeg faces
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(1), rect3d{
            lo = int3d{ 0,             halo.y,        halo.z        },
            hi = int3d{ halo.x-1,      halo.y+Ny-1,   halo.z+Nz-1   }})
         -- xPos faces
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(2), rect3d{
            lo = int3d{ halo.x+Nx,     halo.y,        halo.z        },
            hi = int3d{ r.bounds.hi.x, halo.y+Ny-1,   halo.z+Nz-1   }})
         -- yNeg faces
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(3), rect3d{
            lo = int3d{ halo.x,        0,             halo.z        },
            hi = int3d{ halo.x+Nx-1,   halo.y-1,      halo.z+Nz-1   }})
         -- yPos faces
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(4), rect3d{
            lo = int3d{ halo.x,        halo.y+Ny,     halo.z        },
            hi = int3d{ halo.x+Nx-1,   r.bounds.hi.y, halo.z+Nz-1   }})
         -- zNeg faces
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(5), rect3d{
            lo = int3d{ halo.x,        halo.y,        0             },
            hi = int3d{ halo.x+Nx-1,   halo.y+Ny-1,   halo.z-1      }})
         -- zPos faces
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(6), rect3d{
            lo = int3d{ halo.x,        halo.y,        halo.z+Nz     },
            hi = int3d{ halo.x+Nx-1,   halo.y+Ny-1,   r.bounds.hi.z }})

         -- xNeg-yNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(7), rect3d{
            lo = int3d{ 0,             0,             halo.z        },
            hi = int3d{ halo.x-1,      halo.y-1,      halo.z+Nz-1   }})
         -- xNeg-zNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(8), rect3d{
            lo = int3d{ 0,             halo.y,        0             },
            hi = int3d{ halo.x-1,      halo.y+Ny-1,   halo.z-1      }})
         -- xNeg-yPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(9), rect3d{
            lo = int3d{ 0,             halo.y+Ny,     halo.z        },
            hi = int3d{ halo.x-1,      r.bounds.hi.y, halo.z+Nz-1   }})
         -- xNeg-yPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(10), rect3d{
            lo = int3d{ 0,             halo.y,        halo.z+Nz     },
            hi = int3d{ halo.x-1,      halo.y+Ny-1,   r.bounds.hi.z }})
         -- xPos-yNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(11), rect3d{
            lo = int3d{ halo.x+Nx,     0,             halo.z        },
            hi = int3d{ r.bounds.hi.x, halo.y-1,      halo.z+Nz-1   }})
         -- xPos-zNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(12), rect3d{
            lo = int3d{ halo.x+Nx,     halo.y,        0             },
            hi = int3d{ r.bounds.hi.x, halo.y+Ny-1,   halo.z-1      }})
         -- xPos-yPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(13), rect3d{
            lo = int3d{ halo.x+Nx,     halo.y+Ny,     halo.z        },
            hi = int3d{ r.bounds.hi.x, r.bounds.hi.y, halo.z+Nz-1   }})
         -- xPos-yPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(14), rect3d{
            lo = int3d{ halo.x+Nx,     halo.y,        halo.z+Nz     },
            hi = int3d{ r.bounds.hi.x, halo.y+Ny-1,   r.bounds.hi.z }})
         -- yNeg-zNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(15), rect3d{
            lo = int3d{ halo.x,        0,             0             },
            hi = int3d{ halo.x+Nx-1,   halo.y-1,      halo.z-1      }})
         -- yNeg-zPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(16), rect3d{
            lo = int3d{ halo.x,        0,             halo.z+Nz     },
            hi = int3d{ halo.x+Nx-1,   halo.y-1,      r.bounds.hi.z }})
         -- yPos-zNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(17), rect3d{
            lo = int3d{ halo.x,        halo.y+Ny,     0             },
            hi = int3d{ halo.x+Nx-1,   r.bounds.hi.y, halo.z-1      }})
         -- yPos-zPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(18), rect3d{
            lo = int3d{ halo.x,        halo.y+Ny,     halo.z+Nz     },
            hi = int3d{ halo.x+Nx-1,   r.bounds.hi.y, r.bounds.hi.z }})

         -- xNeg-yNeg-zNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(19), rect3d{
            lo = int3d{ 0,             0,             0             },
            hi = int3d{ halo.x-1,      halo.y-1,      halo.z-1      }})
         -- xNeg-yPos-zNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(20), rect3d{
            lo = int3d{ 0,             halo.y+Ny,     0             },
            hi = int3d{ halo.x-1,      r.bounds.hi.y, halo.z-1      }})
         -- xNeg-yNeg-zPos corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(21), rect3d{
            lo = int3d{ 0,             0,             halo.z+Nz     },
            hi = int3d{ halo.x-1,      halo.y-1,      r.bounds.hi.z }})
         -- xNeg-yPos-zNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(22), rect3d{
            lo = int3d{ 0,             halo.y+Ny,     halo.z+Nz     },
            hi = int3d{ halo.x-1,      r.bounds.hi.y, r.bounds.hi.z }})
         -- xPos-yNeg-zNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(23), rect3d{
            lo = int3d{ halo.x+Nx,     0,             0             },
            hi = int3d{ r.bounds.hi.x, halo.y-1,      halo.z-1      }})
         -- xPos-yPos-zNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(24), rect3d{
            lo = int3d{ halo.x+Nx,     halo.y+Ny,     0             },
            hi = int3d{ r.bounds.hi.x, r.bounds.hi.y, halo.z-1      }})
         -- xPos-yNeg-zPos corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(25), rect3d{
            lo = int3d{ halo.x+Nx,     0,             halo.z+Nz     },
            hi = int3d{ r.bounds.hi.x, halo.y-1,      r.bounds.hi.z }})
         -- xPos-yPos-zNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(26), rect3d{
            lo = int3d{ halo.x+Nx,     halo.y+Ny,     halo.z+Nz     },
            hi = int3d{ r.bounds.hi.x, r.bounds.hi.y, r.bounds.hi.z }})

         var [p] = partition(disjoint, r, coloring, ispace(int1d,27))
         regentlib.c.legion_domain_point_coloring_destroy(coloring)
         return [p]
      end
   elseif r_istype == int2d then
      __demand(__inline)
      task partitionIsInteriorOrGhost(r : region(ispace(int2d), fs), halo : int2d)
         var Nx = r.bounds.hi.x - 2*halo.x + 1
         var Ny = r.bounds.hi.y - 2*halo.y + 1
         regentlib.assert(r.bounds.lo == int2d{0,0}, "Can only partition root region")
         var coloring = regentlib.c.legion_domain_point_coloring_create()

         -- Interior points
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(0), rect2d{
            lo = int2d{halo.x,         halo.y       },
            hi = int2d{halo.x+Nx-1,    halo.y+Ny-1  }})

         -- xNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(1), rect2d{
            lo = int2d{0,             halo.y        },
            hi = int2d{halo.x-1,      halo.y+Ny-1   }})
         -- xPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(2), rect2d{
            lo = int2d{halo.x+Nx,     halo.y        },
            hi = int2d{r.bounds.hi.x, halo.y+Ny-1   }})
         -- yNeg edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(3), rect2d{
            lo = int2d{halo.x,        0             },
            hi = int2d{halo.x+Nx-1,   halo.y-1,     }})
         -- yPos edge
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(4), rect2d{
            lo = int2d{halo.x,        halo.y+Ny     },
            hi = int2d{halo.x+Nx-1,   r.bounds.hi.y }})

         -- xNeg-yNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(5), rect2d{
            lo = int2d{0,             0             },
            hi = int2d{halo.x-1,      halo.y-1      }})
         -- xNeg-yPos corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(6), rect2d{
            lo = int2d{0,             halo.y+Ny     },
            hi = int2d{halo.x-1,      r.bounds.hi.y }})
         -- xPos-yNeg corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(7), rect2d{
            lo = int2d{halo.x+Nx,     0             },
            hi = int2d{r.bounds.hi.x, halo.y-1      }})
         -- xPos-yPos corner
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(8), rect2d{
            lo = int2d{halo.x+Nx,     halo.y+Ny     },
            hi = int2d{r.bounds.hi.x, r.bounds.hi.y }})

         var [p] = partition(disjoint, r, coloring, ispace(int1d,9))
         regentlib.c.legion_domain_point_coloring_destroy(coloring)
         return [p]
      end
   elseif r_istype == int1d then
      __demand(__inline)
      task partitionIsInteriorOrGhost(r : region(ispace(int1d), fs), halo : int64)
         var N = int64(r.bounds.hi - 2*halo + 1)
         regentlib.assert(r.bounds.lo == 0, "Can only partition root region")
         var coloring = regentlib.c.legion_multi_domain_point_coloring_create()

         -- Interior points
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(0), rect1d{ lo = halo,   hi = halo+N-1    })
         -- xNeg point
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(1), rect1d{ lo = 0,      hi = halo-1      })
         -- xPos point
         regentlib.c.legion_domain_point_coloring_color_domain(coloring, int1d(2), rect1d{ lo = halo+N, hi = r.bounds.hi })

         var [p] = partition(disjoint, r, coloring, ispace(int1d,3))
         regentlib.c.legion_domain_point_coloring_destroy(coloring)
         return [p]
      end
  else assert(false) end
  return partitionIsInteriorOrGhost
end

-- string, intXd, intXd, terralib.struct, int -> regentlib.task
function Exports.mkExtractRelevantIspace(p_type, r_istype, cs_istype, fs, ind)
   local extractRelevantIspace

   if p_type == "cross_product" and r_istype == int3d and cs_istype == int3d and ind == nil then
      __demand(__inline)
      task extractRelevantIspace(r  : region(ispace(int3d), fs),
                                 p1 : partition(aliased, r, ispace(int1d)),
                                 p2 : partition(disjoint, r, ispace(int3d)),
                                 pr : cross_product(p1, p2),
                                 aux : region(ispace(int3d), bool))

         var cs1 = p1.colors
         var cs2 = p2.colors
         var isvalidAux = (aux.ispace.bounds.lo.x == cs2.bounds.lo.x) and
                          (aux.ispace.bounds.lo.y == cs2.bounds.lo.y) and
                          (aux.ispace.bounds.lo.z == cs2.bounds.lo.z) and
                          (aux.ispace.bounds.hi.x == cs2.bounds.hi.x) and
                          (aux.ispace.bounds.hi.y == cs2.bounds.hi.y) and
                          (aux.ispace.bounds.hi.z == cs2.bounds.hi.z)
         regentlib.assert(isvalidAux, "extractRelevantIspace: aux region must have the same index space as p2")
         var coloring = regentlib.c.legion_multi_domain_point_coloring_create()
         for c1 in cs1 do
            for c2 in cs2 do
               var p = pr[c1][c2]
               if p.volume ~= 0 then
                  regentlib.c.legion_multi_domain_point_coloring_color_domain(coloring, c1, rect3d{lo=c2, hi=c2})
               end
            end
         end
         var p_aux = partition(aliased, aux, coloring, cs1)
         regentlib.c.legion_multi_domain_point_coloring_destroy(coloring)
         return p_aux
      end

   elseif p_type == "cross_product" and r_istype == int3d and cs_istype == int3d  and ind ~= nil then
      __demand(__inline)
      task extractRelevantIspace(r  : region(ispace(int3d), fs),
                                 p1 : partition(disjoint, r, ispace(int1d)),
                                 p2 : partition(disjoint, r, ispace(int3d)),
                                 pr : cross_product(p1, p2))

         var cs = p2.colors
         var aux = region(cs, bool)
         var coloring = regentlib.c.legion_multi_domain_point_coloring_create()
         for c in cs do
            var p = pr[ind][c]
            if p.volume ~= 0 then
               regentlib.c.legion_multi_domain_point_coloring_color_domain(coloring, int1d(0), rect3d{lo=c, hi=c})
            end
         end
         var p_aux = partition(disjoint, aux, coloring, ispace(int1d, 1))
         regentlib.c.legion_multi_domain_point_coloring_destroy(coloring)
         return p_aux[0].ispace
      end

   elseif p_type == "partition" and r_istype == int3d and cs_istype == int3d then
      assert(ind == nil)
      __demand(__inline)
      task extractRelevantIspace(r  : region(ispace(int3d), fs),
                                 p  : partition(disjoint, r, ispace(int3d)))

         var cs = p.colors
         var aux = region(cs, bool)
         var coloring = regentlib.c.legion_multi_domain_point_coloring_create()
         for c in cs do
            var p = p[c]
            if p.volume ~= 0 then
               regentlib.c.legion_multi_domain_point_coloring_color_domain(coloring, int1d(0), rect3d{lo=c, hi=c})
            end
         end
         var p_aux = partition(disjoint, aux, coloring, ispace(int1d, 1))
         regentlib.c.legion_multi_domain_point_coloring_destroy(coloring)
         return p_aux[0].ispace
      end

   else assert(false) end
   return extractRelevantIspace
end

-- int, string, regentlib.rexpr, regentlib.rexpr -> regentlib.rquote
function Exports.emitArrayReduce(dims, op, lhs, rhs)
  -- We decompose each array-type reduction into a sequence of primitive
  -- reductions over the array's elements, to make sure the code generator can
  -- emit them as atomic operations if needed.
  return rquote
    var tmp = [rhs];
    rescape for i = 0,dims-1 do
      if     op == '+'   then remit rquote lhs[i] +=   tmp[i] end
      elseif op == '-'   then remit rquote lhs[i] -=   tmp[i] end
      elseif op == '*'   then remit rquote lhs[i] *=   tmp[i] end
      elseif op == '/'   then remit rquote lhs[i] /=   tmp[i] end
      elseif op == 'max' then remit rquote lhs[i] max= tmp[i] end
      elseif op == 'min' then remit rquote lhs[i] min= tmp[i] end
      else assert(false) end
    end end
  end
end

-------------------------------------------------------------------------------
-- Error handling
-------------------------------------------------------------------------------

-- regentlib.rexpr, string, regentlib.rexpr* -> regentlib.rquote
function Exports.emitAssert(cond, format, ...)
  local args = terralib.newlist{...}
  return rquote
    if not cond then
      var stderr = C.fdopen(2, 'w')
      C.fprintf(stderr, format, [args])
      C.fflush(stderr)
      C.exit(1)
    end
  end
end

-------------------------------------------------------------------------------
-- Random number generator
-------------------------------------------------------------------------------

-- Metaprogrammed implementation of philox2x32
-- (http://dx.doi.org/10.1145/2063384.2063405)
-- the recommended number of rounds is 10, but ~5 can be ok if you need something fast
--
-- The produced function will return a random number in the range [0.0, 1.0)
-- key: some sort of seed of the random number succession
-- ctr: is a counter that should be different at each call of the function
function Exports.mkRand(rounds)
   local rand
   -- default number of rounds
   if rounds == nil then rounds = 10 end
   __demand(__inline)
   task rand(key : uint32, ctr : uint64) : double
      var ctr_lo : uint32
      var ctr_hi : uint32
      var prod_lo : uint32
      var prod_hi : uint32
      ctr_lo = ctr
      ctr_hi = ctr >> 32
      rescape
         for i=0, rounds do
            remit rquote
               var prod : uint64 = ctr_lo * 0xD256D193ULL;
               prod_hi = prod >> 32
               prod_lo = prod
               ctr_lo = ctr_hi ^ key ^ prod_hi
               ctr_hi = prod_lo
               key = key + 0x9E3779B9U
            end
         end
      end
      var rand = ([uint64](ctr_hi) << 32) + ctr_lo
      return 0x1p-64 * rand
   end
   return rand
end

-------------------------------------------------------------------------------

return Exports
