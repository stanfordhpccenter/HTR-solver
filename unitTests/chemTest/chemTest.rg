import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
--local format = require("std/format")
local SCHEMA = terralib.includec("../../src/config_schema.h")
local REGISTRAR = terralib.includec("registrar.h")
local UTIL = require 'util'

local Config = SCHEMA.Config

local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

-------------------------------------------------------------------------------
-- ACTIVATE ELECTRIC FIELD SOLVER
-------------------------------------------------------------------------------

local ELECTRIC_FIELD = false
if os.getenv("ELECTRIC_FIELD") == "1" then
   ELECTRIC_FIELD = true
   print("#############################################################################")
   print("WARNING: You are compiling with electric field solver.")
   print("#############################################################################")
end

local types_inc_flags = terralib.newlist({"-DEOS=AirMix"})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)
local nSpec = MIX.nSpec
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

--External modules
local CHEM = (require 'prometeo_chem')(SCHEMA, MIX, TYPES, true)

local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 5000.0 end
local Yi = rexpr array(0.78, 0.22, 1.0e-60, 1.0e-60, 1.0e-60) end
local Eprod = rexpr array(-1.658328e-01,-1.445940e+03, 1.332770e-54, 1.658328e-01, 1.445940e+03, 0.0, 0.0, 0.0, 0.0) end
local Cres = rexpr array( 5.460212e-02, 1.296359e-02, 2.994386e-04, 1.502734e-05, 2.321006e-03, 0.0, 0.0, 0.0, 3.141423e+05) end

local LRef = rexpr 1.0 end
local TRef = rexpr 300.0 end
local PRef = rexpr 101325.0 end
local YO2Ref = rexpr 0.22 end
local YN2Ref = rexpr 0.78 end
local MixWRef = rexpr 1.0/([YN2Ref]/28.0134e-3 + [YO2Ref]/(2*15.999e-3)) end
local rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
local eRef = rexpr [PRef]/[rhoRef] end
local wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), TYPES.Fluid_columns), Mix : MIX.Mixture)
where
   writes(Fluid)
do
   var MixW = MIX.GetMolarWeightFromYi([Yi], &Mix)
   var Xi : double[nSpec]; MIX.GetMolarFractions(Xi, MixW, [Yi], &Mix)
   var e = MIX.GetInternalEnergy([T]/[TRef], [Yi], &Mix)
   var rho = MIX.GetRho([P]/[PRef], [T]/[TRef], MixW, &Mix)
   fill(Fluid.pressure, [P]/[PRef])
   fill(Fluid.temperature, [T]/[TRef])
   fill(Fluid.MassFracs, [Yi])
   fill(Fluid.MolarFracs, Xi)
   fill(Fluid.velocity, array(0.0, 0.0, 0.0))
   fill(Fluid.rho, rho)
   fill(Fluid.Conserved, array(rho*[Yi][0],
                               rho*[Yi][1],
                               rho*[Yi][2],
                               rho*[Yi][3],
                               rho*[Yi][4],
                                       0.0,
                                       0.0,
                                       0.0,
                                     rho*e))
   fill(Fluid.Conserved_t,     [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
   fill(Fluid.Conserved_t_old, [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
end

task zero() return 0.0 end

task main()

   -- Define the domain
   var is_Fluid = ispace(int3d, {2, 2, 2})
   var Fluid = region(is_Fluid, TYPES.Fluid_columns)
   var tiles = ispace(int3d, {2, 2, 2})
   var p_All = partition(equal, Fluid, tiles)

   -- Init the mixture
   var config : Config
   config.Flow.mixture.type = SCHEMA.MixtureModel_AirMix
   config.Flow.mixture.u.AirMix.LRef = [LRef]
   config.Flow.mixture.u.AirMix.TRef = [TRef]
   config.Flow.mixture.u.AirMix.PRef = [PRef]
   config.Flow.mixture.u.AirMix.XiRef.Species.length = 2
   C.snprintf([&int8](config.Flow.mixture.u.AirMix.XiRef.Species.values[0].Name), 10, "O2")
   C.snprintf([&int8](config.Flow.mixture.u.AirMix.XiRef.Species.values[1].Name), 10, "N2")
   config.Flow.mixture.u.AirMix.XiRef.Species.values[0].MolarFrac = [MixWRef]*[YO2Ref]/(2*15.999e-3)
   config.Flow.mixture.u.AirMix.XiRef.Species.values[1].MolarFrac = [MixWRef]*[YN2Ref]/28.0134e-3
   var Mix = MIX.InitMixture(Fluid, tiles, p_All, config)

   InitializeCell(Fluid, Mix)

   -- Test explicit chem advancing
   __demand(__index_launch)
   for c in tiles do
      CHEM.AddChemistrySources(p_All[c], Mix)
   end
   for c in Fluid do
      for i = 0, nEq do
         var err = Fluid[c].Conserved_t[i]*[wiRef] - [Eprod][i]
         if ( [Eprod][i] ~= 0) then err /= [Eprod][i] end
         regentlib.assert(fabs(err) < 1e-5, "chemTest: ERROR in explicit chem advancing")
      end
   end

   var dt = 1e-6*sqrt([PRef]/[rhoRef])/[LRef]
   dt += zero() -- so it becomes a future

   -- Test implicit chem advancing
   __demand(__index_launch)
   for c in tiles do
      CHEM.UpdateChemistry(p_All[c], dt, Mix)
   end

   for c in Fluid do
      for i = 0, nEq do
         var err : double
         if i == (nEq-1) then
            err = Fluid[c].Conserved[i]*[rhoRef]*[eRef] - [Cres][i]
         else
            err = Fluid[c].Conserved[i]*[rhoRef] - [Cres][i]
         end
         if ([Cres][i] ~= 0) then err /= [Cres][i] end
         regentlib.assert(fabs(err) < 1e-5, "chemTest: ERROR in implicit chem advancing")
      end
   end

   __fence(__execution, __block)

   C.printf("chemTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "chemTest.o", "object", REGISTRAR.register_tasks)
