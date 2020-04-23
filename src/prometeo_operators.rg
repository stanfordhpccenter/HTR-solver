-- Copyright (c) "2019, by Stanford University
--               Developer: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
--                         HTR solver: An open-source exascale-oriented task-based
--                         multi-GPU high-order code for hypersonic aerothermodynamics.
--                         Computer Physics Communications (In Press), 107262"
-- All rights reserved.
-- 
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions are met:
--    * Redistributions of source code must retain the above copyright
--      notice, this list of conditions and the following disclaimer.
--    * Redistributions in binary form must reproduce the above copyright
--      notice, this list of conditions and the following disclaimer in the
--      documentation and/or other materials provided with the distribution.
-- 
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
-- ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
-- WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
-- DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
-- DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
-- (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
-- LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
-- ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
-- SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import "regent"

local Exports = {}

function Exports.emitderivBase(gradOp, xM1, x, xP1)
   return rexpr [gradOp][0]*([x] - [xM1]) + [gradOp][1]*([xP1] - [x]) end
end

function Exports.emitderivLeftBCBase(gradOp, x, xP1)
   return rexpr [gradOp][1]*([xP1] - [x]) end
end

function Exports.emitderivRightBCBase(gradOp, xM1, x)
   return rexpr [gradOp][0]*([x] - [xM1]) end
end


function Exports.emitXderiv(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivBase(rexpr [r][ind].gradX                      end,
                                rexpr [r][(ind+{-1, 0, 0}) % [bound]].[q] end,
                                rexpr [r][ ind                      ].[q] end,
                                rexpr [r][(ind+{ 1, 0, 0}) % [bound]].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivBase(rexpr [r][ind].gradX                         end,
                                rexpr [r][(ind+{-1, 0, 0}) % [bound]].[q][k] end,
                                rexpr [r][ ind                      ].[q][k] end,
                                rexpr [r][(ind+{ 1, 0, 0}) % [bound]].[q][k] end)]
      end
   end
end

function Exports.emitYderiv(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivBase(rexpr [r][ind].gradY                      end,
                                rexpr [r][(ind+{ 0,-1, 0}) % [bound]].[q] end,
                                rexpr [r][ ind                      ].[q] end,
                                rexpr [r][(ind+{ 0, 1, 0}) % [bound]].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivBase(rexpr [r][ind].gradY                         end,
                                rexpr [r][(ind+{ 0,-1, 0}) % [bound]].[q][k] end,
                                rexpr [r][ ind                      ].[q][k] end,
                                rexpr [r][(ind+{ 0, 1, 0}) % [bound]].[q][k] end)]
      end
   end
end

function Exports.emitZderiv(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivBase(rexpr [r][ind].gradZ                      end,
                                rexpr [r][(ind+{ 0, 0,-1}) % [bound]].[q] end,
                                rexpr [r][ ind                      ].[q] end,
                                rexpr [r][(ind+{ 0, 0, 1}) % [bound]].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivBase(rexpr [r][ind].gradZ                         end,
                                rexpr [r][(ind+{ 0, 0,-1}) % [bound]].[q][k] end,
                                rexpr [r][ ind                      ].[q][k] end,
                                rexpr [r][(ind+{ 0, 0, 1}) % [bound]].[q][k] end)]
      end
   end
end

function Exports.emitXderivLeftBC(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivLeftBCBase(rexpr [r][ind].gradX                      end,
                                      rexpr [r][ ind                      ].[q] end,
                                      rexpr [r][(ind+{ 1, 0, 0}) % [bound]].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivLeftBCBase(rexpr [r][ind].gradX                         end,
                                      rexpr [r][ ind                      ].[q][k] end,
                                      rexpr [r][(ind+{ 1, 0, 0}) % [bound]].[q][k] end)]
      end
   end
end

function Exports.emitYderivLeftBC(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivLeftBCBase(rexpr [r][ind].gradY                      end,
                                      rexpr [r][ ind                      ].[q] end,
                                      rexpr [r][(ind+{ 0, 1, 0}) % [bound]].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivLeftBCBase(rexpr [r][ind].gradY                         end,
                                      rexpr [r][ ind                      ].[q][k] end,
                                      rexpr [r][(ind+{ 0, 1, 0}) % [bound]].[q][k] end)]
      end
   end
end

function Exports.emitZderivLeftBC(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivLeftBCBase(rexpr [r][ind].gradZ                      end,
                                      rexpr [r][ ind                      ].[q] end,
                                      rexpr [r][(ind+{ 0, 0, 1}) % [bound]].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivLeftBCBase(rexpr [r][ind].gradZ                         end,
                                      rexpr [r][ ind                      ].[q][k] end,
                                      rexpr [r][(ind+{ 0, 0, 1}) % [bound]].[q][k] end)]
      end
   end
end

function Exports.emitXderivRightBC(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivRightBCBase(rexpr [r][ind].gradX                      end,
                                       rexpr [r][(ind+{-1, 0, 0}) % [bound]].[q] end,
                                       rexpr [r][ ind                      ].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivRightBCBase(rexpr [r][ind].gradX                         end,
                                       rexpr [r][(ind+{-1, 0, 0}) % [bound]].[q][k] end,
                                       rexpr [r][ ind                      ].[q][k] end)]
      end
   end
end

function Exports.emitYderivRightBC(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivRightBCBase(rexpr [r][ind].gradY                      end,
                                       rexpr [r][(ind+{ 0,-1, 0}) % [bound]].[q] end,
                                       rexpr [r][ ind                      ].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivRightBCBase(rexpr [r][ind].gradY                         end,
                                       rexpr [r][(ind+{ 0,-1, 0}) % [bound]].[q][k] end,
                                       rexpr [r][ ind                      ].[q][k] end)]
      end
   end
end

function Exports.emitZderivRightBC(r, q, ind, bound, k)
   if k == nil then
      return rexpr
         [Exports.emitderivRightBCBase(rexpr [r][ind].gradZ                      end,
                                       rexpr [r][(ind+{ 0, 0,-1}) % [bound]].[q] end,
                                       rexpr [r][ ind                      ].[q] end)]
      end
   else
      return rexpr
         [Exports.emitderivRightBCBase(rexpr [r][ind].gradZ                         end,
                                       rexpr [r][(ind+{ 0, 0,-1}) % [bound]].[q][k] end,
                                       rexpr [r][ ind                      ].[q][k] end)]
      end
   end
end

return Exports

