import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local UTIL = require 'util-desugared'

local Config = SCHEMA.Config

MIX = (require "ConstPropMix")(SCHEMA)
local nSpec = MIX.nSpec

local struct Fluid_columns {
   -- Grid point
   cellWidth : double[3];
   -- Primitive variables
   temperature : double;
   MassFracs   : double[nSpec];
   MolarFracs  : double[nSpec];
   velocity    : double[3];
   -- Properties
   rho  : double;
   mu   : double;
   lam  : double;
   Di   : double[nSpec];
   SoS  : double;
}

--External modules
local CFL = (require 'prometeo_cfl')(MIX, Fluid_columns)

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.cellWidth, array(1.0, 1.0, 1.0))
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

   -- Init the mixture
   var config : Config
   config.Flow.mixture.type = SCHEMA.MixtureModel_ConstPropMix
   config.Flow.mixture.u.ConstPropMix.gasConstant = 1.0
   config.Flow.mixture.u.ConstPropMix.gamma = 1.4

   var Mix = MIX.InitMixture(config)

   -- Define the domain
   var is_Fluid = ispace(int3d, {1, 1, 1})
   var Fluid = region(is_Fluid, Fluid_columns)

   InitializeCell(Fluid)

   -- Test acustic cfl
   fill(Fluid.velocity,  array(1.0, 1.0, 1.0))
   fill(Fluid.SoS, 10.0)
   var s = CFL.CalculateMaxSpectralRadius(Fluid, Fluid, Mix)
   regentlib.assert(fabs(s/11.0 - 1.0) < 1e-3, "cflTest: ERROR in acustic cfl calculation")
   fill(Fluid.velocity,  array(0.0, 0.0, 0.0))

   -- Test momentum diffusion cfl
   fill(Fluid.mu, 10.0)
   s = CFL.CalculateMaxSpectralRadius(Fluid, Fluid, Mix)
   regentlib.assert(fabs(s/40.0 - 1.0) < 1e-3, "cflTest: ERROR in momentum diffusion cfl calculation")
   fill(Fluid.mu, 0.0)

   -- Test energy diffusion cfl
   fill(Fluid.lam, 10.0)
   s = CFL.CalculateMaxSpectralRadius(Fluid, Fluid, Mix)
   regentlib.assert(fabs(s/11.42857 - 1.0) < 1e-3, "cflTest: ERROR in energy diffusion cfl calculation")
   fill(Fluid.lam, 0.0)

   -- Test species diffusion cfl
   fill(Fluid.Di, [UTIL.mkArrayConstant(nSpec, rexpr 10.0 end)])
   s = CFL.CalculateMaxSpectralRadius(Fluid, Fluid, Mix)
   regentlib.assert(fabs(s/40.0 - 1.0) < 1e-3, "cflTest: ERROR in species diffusion cfl calculation")
   fill(Fluid.Di, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])

   __fence(__execution, __block)

   C.printf("cflTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "cflTest.o", "object")
