import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local UTIL = require 'util-desugared'

local Config = SCHEMA.Config

MIX = (require "AirMix")(SCHEMA)
local nSpec = MIX.nSpec
local nEq = nSpec+4

local struct Fluid_columns {
   -- Primitive variables
   pressure    : double;
   temperature : double;
   MassFracs   : double[nSpec];
   MolarFracs  : double[nSpec];
   velocity    : double[3];
   -- Properties
   rho  : double;
   -- Conserved varaibles
   Conserved       : double[nEq];
   Conserved_t     : double[nEq];
   Conserved_t_old : double[nEq];
}

--External modules
local CHEM = (require 'prometeo_chem')(SCHEMA,MIX, Fluid_columns, true)

local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 5000.0 end
local Yi = rexpr array(0.78, 0.22, 1.0e-60, 1.0e-60, 1.0e-60) end
local Eprod = rexpr array(-1.658328e-01,-1.445940e+03, 1.332770e-54, 1.658328e-01, 1.445940e+03, 0.0, 0.0, 0.0, 0.0) end
local Cres = rexpr array( 5.460212e-02, 1.296359e-02, 2.994386e-04, 1.502734e-05, 2.321006e-03, 0.0, 0.0, 0.0, 3.141423e+05) end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns), Mix : MIX.Mixture)
where
   writes(Fluid)
do
   var MixW = MIX.GetMolarWeightFromYi([Yi], Mix)
   var Xi = MIX.GetMolarFractions(MixW, [Yi], Mix)
   var e = MIX.GetInternalEnergy([T], [Yi], Mix)
   var rho = MIX.GetRho([P], [T], MixW, Mix)
   fill(Fluid.pressure, [P])
   fill(Fluid.temperature, [T])
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

task main()
   -- Init the mixture
   var config : Config
   config.Flow.mixture.type = SCHEMA.MixtureModel_AirMix
   var Mix = MIX.InitMixture(config)

   -- Define the domain
   var is_Fluid = ispace(int3d, {1, 1, 1})
   var Fluid = region(is_Fluid, Fluid_columns)

   InitializeCell(Fluid, Mix)

   -- Test explicit chem advancing
   CHEM.AddChemistrySources(Fluid, Fluid, Mix)
   for c in Fluid do
      for i = 0, nEq do
         var err = Fluid[c].Conserved_t[i] - [Eprod][i]
         if ( [Eprod][i] ~= 0) then err /= [Eprod][i] end
         regentlib.assert(fabs(err) < 1e-5, "chemTest: ERROR in explicit chem advancing")
      end
   end

   -- Test implicit chem advancing
   CHEM.UpdateChemistry(Fluid, Fluid, 1e-6, Mix)

   for c in Fluid do
      for i = 0, nEq do
         var err = Fluid[c].Conserved[i] - [Cres][i]
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

regentlib.saveobj(main, "chemTest.o", "object")
