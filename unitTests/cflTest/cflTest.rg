import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local SCHEMA = terralib.includec("config_schema.h")
local REGISTRAR = terralib.includec("registrar.h")
local UTIL = require 'util'

local Config = SCHEMA.Config

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

local types_inc_flags = terralib.newlist({"-DEOS=ConstPropMix"})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)
local nSpec = MIX.nSpec

--External modules
local CFL = (require 'prometeo_cfl')(MIX, TYPES)

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.dcsi_d, 1.0)
   fill(Fluid.deta_d, 1.0)
   fill(Fluid.dzet_d, 1.0)
   fill(Fluid.temperature, 0.0)
   fill(Fluid.MassFracs,  [UTIL.mkArrayConstant(nSpec, rexpr 1.0 end)])
   fill(Fluid.MolarFracs, [UTIL.mkArrayConstant(nSpec, rexpr 1.0 end)])
   fill(Fluid.velocity, array(0.0, 0.0, 0.0))
   fill(Fluid.rho, 1.0)
   fill(Fluid.mu,  0.0)
   fill(Fluid.lam, 0.0)
   fill(Fluid.Di, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.SoS, 0.0)
end

task main()

   -- Define the domain
   var is_Fluid = ispace(int3d, {2, 2, 2})
   var Fluid = region(is_Fluid, Fluid_columns)
   var tiles = ispace(int3d, {2, 2, 2})
   var p_All = partition(equal, Fluid, tiles)

   -- Init the mixture
   var config : Config
   config.Flow.mixture.type = SCHEMA.MixtureModel_ConstPropMix
   config.Flow.mixture.u.ConstPropMix.gasConstant = 1.0
   config.Flow.mixture.u.ConstPropMix.gamma = 1.4

   var Mix = MIX.InitMixture(Fluid, tiles, p_All, config)

   InitializeCell(Fluid)

   -- Test acustic cfl
   fill(Fluid.velocity,  array(1.0, 1.0, 1.0))
   fill(Fluid.SoS, 10.0)
   var s = 0.0
   __demand(__index_launch)
   for c in tiles do
      s max= CFL.CalculateMaxSpectralRadius(p_All[c], Mix)
   end
   regentlib.assert(fabs(s/11.0 - 1.0) < 1e-3, "cflTest: ERROR in acustic cfl calculation")
   fill(Fluid.velocity,  array(0.0, 0.0, 0.0))

   -- Test momentum diffusion cfl
   fill(Fluid.mu, 10.0)
   s = 0.0
   __demand(__index_launch)
   for c in tiles do
      s max= CFL.CalculateMaxSpectralRadius(p_All[c], Mix)
   end
   regentlib.assert(fabs(s/40.0 - 1.0) < 1e-3, "cflTest: ERROR in momentum diffusion cfl calculation")
   fill(Fluid.mu, 0.0)

   -- Test energy diffusion cfl
   fill(Fluid.lam, 10.0)
   s = 0.0
   __demand(__index_launch)
   for c in tiles do
      s max= CFL.CalculateMaxSpectralRadius(p_All[c], Mix)
   end
   regentlib.assert(fabs(s/11.42857 - 1.0) < 1e-3, "cflTest: ERROR in energy diffusion cfl calculation")
   fill(Fluid.lam, 0.0)

   -- Test species diffusion cfl
   fill(Fluid.Di, [UTIL.mkArrayConstant(nSpec, rexpr 10.0 end)])
   s = 0.0
   __demand(__index_launch)
   for c in tiles do
      s max= CFL.CalculateMaxSpectralRadius(p_All[c], Mix)
   end
   regentlib.assert(fabs(s/40.0 - 1.0) < 1e-3, "cflTest: ERROR in species diffusion cfl calculation")
   fill(Fluid.Di, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])

   -- TODO: Test ion drift cfl condition

   __fence(__execution, __block)

   C.printf("cflTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "cflTest.o", "object", REGISTRAR.register_tasks)
