import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local REGISTRAR = terralib.includec("prometeo_variables.h")
local UTIL = require 'util-desugared'

local Config = SCHEMA.Config

local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"
local MIX = (require "AirMix")(SCHEMA)
local nSpec = MIX.nSpec
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

local Primitives = CONST.Primitives
local Properties = CONST.Properties

local TYPES = terralib.includec("prometeo_types.h", {"-DEOS="..os.getenv("EOS")})
local Fluid_columns = TYPES.Fluid_columns

--External modules
local METRIC = (require 'prometeo_metric')(SCHEMA, Fluid_columns)
local VARS = (require 'prometeo_variables')(SCHEMA, MIX, METRIC, Fluid_columns)

-- Test parameters
local Npx = 16
local Npy = 16
local Npz = 16
local Nx = 2
local Ny = 2
local Nz = 2

--local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 5000.0 end
local Xi = rexpr array(0.4, 0.2, 0.15, 0.15, 0.1) end
local v  = rexpr array(1.0, 2.0, 3.0) end
local Tres = rexpr 500.0 end

-- Expected properties
local eRho = rexpr 6.2899871101668e-02 end
local eMu  = rexpr 1.2424467023580e-04 end
local eLam = rexpr 2.4040340390246e-01 end
local eDi  = rexpr array(2.5146873480781e-03, 2.4389311883618e-03, 2.4550965167542e-03, 3.6168277168130e-03, 3.9165634179846e-03) end
local eSoS = rexpr 1.4518705966651e+03 end

-- Expected conserved variables
local eConserved  = rexpr array(2.7311049167620e-02, 1.5598263689963e-02, 1.0970170602665e-02, 5.1208217189288e-03, 3.8995659224908e-03, 6.2899871101668e-02, 1.2579974220334e-01, 1.8869961330500e-01, 5.4092397631210e+05) end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
   fill(Fluid.cellWidth, array(0.0, 0.0, 0.0))
   fill(Fluid.nType_x, 0)
   fill(Fluid.nType_y, 0)
   fill(Fluid.nType_z, 0)
   fill(Fluid.dcsi_e, 0.0)
   fill(Fluid.deta_e, 0.0)
   fill(Fluid.dzet_e, 0.0)
   fill(Fluid.dcsi_d, 0.0)
   fill(Fluid.deta_d, 0.0)
   fill(Fluid.dzet_d, 0.0)
   fill(Fluid.dcsi_s, 0.0)
   fill(Fluid.deta_s, 0.0)
   fill(Fluid.dzet_s, 0.0)
   fill(Fluid.pressure, [P])
   fill(Fluid.temperature, [T])
   fill(Fluid.MassFracs, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.MolarFracs, [Xi])
   fill(Fluid.velocity, [v])
   fill(Fluid.rho, 0.0)
   fill(Fluid.mu , 0.0)
   fill(Fluid.lam, 0.0)
   fill(Fluid.Di , [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.SoS, 0.0)
   fill(Fluid.velocityGradientX,   array(0.0, 0.0, 0.0))
   fill(Fluid.velocityGradientY,   array(0.0, 0.0, 0.0))
   fill(Fluid.velocityGradientZ,   array(0.0, 0.0, 0.0))
   fill(Fluid.temperatureGradient, array(0.0, 0.0, 0.0))
   fill(Fluid.Conserved, [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
end

__demand(__inline)
task CheckUpdatePropertiesFromPrimitive(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.[Properties])
do
   for c in Fluid do
      regentlib.assert(fabs((Fluid[c].rho/[eRho]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (rho)")
      regentlib.assert(fabs((Fluid[c].mu /[eMu ]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (mu)")
      regentlib.assert(fabs((Fluid[c].lam/[eLam]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (lam)")
      for i=0, nSpec do
         regentlib.assert(fabs((Fluid[c].Di[i]/[eDi][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (Di)")
      end
      regentlib.assert(fabs((Fluid[c].SoS/[eSoS]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (SoS)")
   end
end

__demand(__inline)
task CheckUpdateConservedFromPrimitive(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.Conserved)
do
   for c in Fluid do
      var interior  = MACRO.in_interior(c, 1, Npx, 1, Npy, 1, Npz)
      if interior then
         for i=0, nEq do
            regentlib.assert(fabs((Fluid[c].Conserved[i]/[eConserved][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdateConservedFromPrimitive")
         end
      else
         for i=0, nEq do
            regentlib.assert(Fluid[c].Conserved[i] == 0.0, "variablesTest: ERROR in UpdateConservedFromPrimitive")
         end
      end
   end
end

__demand(__inline)
task CheckUpdateGhostConservedFromPrimitive(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.Conserved)
do
   for c in Fluid do
      var interior  = MACRO.in_interior(c, 1, Npx, 1, Npy, 1, Npz)
      if interior then
         for i=0, nEq do
            regentlib.assert(Fluid[c].Conserved[i] == 0.0, "variablesTest: ERROR in UpdateGhostConservedFromPrimitive")
         end
      else
         for i=0, nEq do
            regentlib.assert(fabs((Fluid[c].Conserved[i]/[eConserved][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdateGhostConservedFromPrimitive")
         end
      end
   end
end

__demand(__inline)
task CheckUpdatePrimitiveFromConserved(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.[Primitives])
do
   for c in Fluid do
      var interior  = MACRO.in_interior(c, 1, Npx, 1, Npy, 1, Npz)
      if interior then
         regentlib.assert(fabs((Fluid[c].pressure/[P]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         regentlib.assert(fabs((Fluid[c].temperature/[T]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         for i=0, nSpec do
            regentlib.assert(fabs((Fluid[c].MolarFracs[i]/[Xi][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         end
         for i=0, 3 do
            regentlib.assert(fabs((Fluid[c].velocity[i]/[v][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         end
      else
         regentlib.assert(Fluid[c].pressure == 0.0, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         regentlib.assert(Fluid[c].temperature == [Tres], "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         for i=0, nSpec do
            regentlib.assert(Fluid[c].MolarFracs[i] == 0.0, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         end
         for i=0, 3 do
            regentlib.assert(Fluid[c].velocity[i] == 0.0, "variablesTest: ERROR in UpdatePrimitiveFromConserved")
         end
      end
   end
end


task main()
   -- Init the mixture
   var config : Config
   config.Flow.mixture.type = SCHEMA.MixtureModel_AirMix
   var Mix = MIX.InitMixture(config)

   -- Define the domain
   var xBnum = 1
   var yBnum = 1
   var zBnum = 1

   -- Define the domain
   var is_Fluid = ispace(int3d, {x = Npx + 2*xBnum,
                                 y = Npy + 2*yBnum,
                                 z = Npz + 2*zBnum})
   var Fluid = region(is_Fluid, Fluid_columns);

   -- Partitioning domain
   var tiles = ispace(int3d, {Nx, Ny, Nz})

   -- Fluid Partitioning
   var p_Fluid =
      [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_All")]
      (Fluid, tiles, int3d{xBnum,yBnum,zBnum}, int3d{0,0,0})

   var Fluid_regions =
      [UTIL.mkPartitionIsInteriorOrGhost(int3d, Fluid_columns, "Fluid_regions")]
      (Fluid, int3d{xBnum,yBnum,zBnum})

   -- Interior points
   var p_Interior = static_cast(partition(disjoint, Fluid, tiles), cross_product(Fluid_regions, p_Fluid)[0])

   -- All ghost points
   var p_AllGhost = p_Fluid - p_Interior

   InitializeCell(Fluid)

   -- Test UpdatePropertiesFromPrimitive
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePropertiesFromPrimitive(p_Fluid[c], p_Fluid[c], Mix)
   end

   CheckUpdatePropertiesFromPrimitive(Fluid)

   -- Test UpdateConservedFromPrimitive for ghosts
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdateConservedFromPrimitive(p_Fluid[c], p_AllGhost[c], Mix)
   end

   CheckUpdateGhostConservedFromPrimitive(Fluid)

   -- Test UpdateConservedFromPrimitive for internal
   fill(Fluid.Conserved, [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdateConservedFromPrimitive(p_Fluid[c], p_Interior[c], Mix)
   end

   CheckUpdateConservedFromPrimitive(Fluid)

   -- Test UpdatePrimitiveFromConserved
   fill(Fluid.pressure, 0.0)
   fill(Fluid.temperature, [Tres])
   fill(Fluid.MolarFracs, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.velocity, array(0.0, 0.0, 0.0))
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePrimitiveFromConserved(p_Fluid[c], p_Interior[c], Mix)
   end

   CheckUpdatePrimitiveFromConserved(Fluid)

   __fence(__execution, __block)

   C.printf("variablesTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "variablesTest.o", "object", REGISTRAR.register_variables_tasks)
