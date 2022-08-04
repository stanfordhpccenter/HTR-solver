import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local REGISTRAR = terralib.includec("registrar.h")
local UTIL = require 'util'

local Config = SCHEMA.Config

local CONST = require "prometeo_const"
local MACRO = require "prometeo_macro"

local Primitives = CONST.Primitives
local Properties = CONST.Properties

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

local types_inc_flags = terralib.newlist({"-DEOS="..os.getenv("EOS")})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)
local nSpec = MIX.nSpec
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver

--External modules
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)
local VARS = (require 'prometeo_variables')(SCHEMA, MIX, METRIC, TYPES, ELECTRIC_FIELD)

-- Test parameters
local Npx = 16
local Npy = 16
local Npz = 16
local Nx = 2
local Ny = 2
local Nz = 2

local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 5000.0 end
local Xi = rexpr array(0.4, 0.2, 0.15, 0.15, 0.1) end
local v  = rexpr array(1.0, 2.0, 3.0) end
local Tres = rexpr 500.0 end

-- Expected properties
local eRho = rexpr 6.2899525132668e-02 end
local eMu  = rexpr 1.2424432854120e-04 end
local eLam = rexpr 2.4040406505225e-01 end
local eDi  = rexpr array(2.5146942638910e-03, 2.4389378958326e-03, 2.4551032686824e-03, 3.6168376636972e-03, 3.9165741891924e-03) end
local eSoS = rexpr 1.4518745895533e+03 end

-- Expected conserved variables
local eConserved  = rexpr array(2.3342371515176e-02, 1.3331617683677e-02, 9.3760512904744e-03, 4.3766946590956e-03, 3.3329044209192e-03, 1.8268107194243e-04, 3.6536214388485e-04, 5.4804321582728e-04, 5.3385045774457e+00) end

-- Normalization quantities
local LRef = rexpr 1.0 end
local PRef = rexpr 101325.0 end
local TRef = rexpr 300.0 end
local YO2Ref = rexpr 0.22 end
local YN2Ref = rexpr 0.78 end
local MixWRef = rexpr 1.0/([YN2Ref]/28.0134e-3 + [YO2Ref]/(2*15.999e-3)) end
local rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
local eRef = rexpr [PRef]/[rhoRef] end
local uRef = rexpr sqrt([PRef]/[rhoRef]) end
local muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
local lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
local DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns))
where
   writes(Fluid)
do
   fill(Fluid.centerCoordinates, array(0.0, 0.0, 0.0))
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
   fill(Fluid.pressure, [P]/[PRef])
   fill(Fluid.temperature, [T]/[TRef])
   fill(Fluid.MassFracs, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.MolarFracs, [Xi])
   fill(Fluid.velocity, array([v][0]/[uRef], [v][1]/[uRef], [v][2]/[uRef]))
   fill(Fluid.rho, 0.0)
   fill(Fluid.mu , 0.0)
   fill(Fluid.lam, 0.0)
   fill(Fluid.Di , [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.SoS, 0.0)
   fill(Fluid.Conserved, [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
end

__demand(__inline)
task CheckUpdatePropertiesFromPrimitive(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.[Properties])
do
   for c in Fluid do
      regentlib.assert(fabs((Fluid[c].rho*[rhoRef]/[eRho]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (rho)")
      regentlib.assert(fabs((Fluid[c].mu *[ muRef]/[eMu ]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (mu)")
      regentlib.assert(fabs((Fluid[c].lam*[lamRef]/[eLam]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (lam)")
      for i=0, nSpec do
         regentlib.assert(fabs((Fluid[c].Di[i]*[DiRef]/[eDi][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (Di)")
      end
      regentlib.assert(fabs((Fluid[c].SoS*[uRef]/[eSoS]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePropertiesFromPrimitive (SoS)")
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
         regentlib.assert(fabs((Fluid[c].pressure*[PRef]/[P]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved (P)")
         regentlib.assert(fabs((Fluid[c].temperature*[TRef]/[T]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved (T)")
         for i=0, nSpec do
            regentlib.assert(fabs((Fluid[c].MolarFracs[i]/[Xi][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved (Xi)")
         end
         for i=0, 3 do
            regentlib.assert(fabs((Fluid[c].velocity[i]*[uRef]/[v][i]) - 1.0) < 1e-8, "variablesTest: ERROR in UpdatePrimitiveFromConserved (v)")
         end
      else
         regentlib.assert(Fluid[c].pressure == 0.0, "variablesTest: ERROR in UpdatePrimitiveFromConserved (P)")
         regentlib.assert(Fluid[c].temperature*[TRef] == [Tres], "variablesTest: ERROR in UpdatePrimitiveFromConserved (T)")
         for i=0, nSpec do
            regentlib.assert(Fluid[c].MolarFracs[i] == 0.0, "variablesTest: ERROR in UpdatePrimitiveFromConserved (Xi)")
         end
         for i=0, 3 do
            regentlib.assert(Fluid[c].velocity[i] == 0.0, "variablesTest: ERROR in UpdatePrimitiveFromConserved (v)")
         end
      end
   end
end


task main()
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

   var Mix = MIX.InitMixture(Fluid, tiles, p_Fluid, config)

   -- Interior points
   var p_Interior = static_cast(partition(disjoint, Fluid, tiles), cross_product(Fluid_regions, p_Fluid)[0])

   -- All ghost points
   var p_AllGhost = p_Fluid - p_Interior

   InitializeCell(Fluid)

   -- Test UpdatePropertiesFromPrimitive
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePropertiesFromPrimitive(p_Fluid[c], Mix)
   end

   CheckUpdatePropertiesFromPrimitive(Fluid)

   -- Test UpdateConservedFromPrimitive for ghosts
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdateConservedFromPrimitive(p_AllGhost[c], Mix)
   end

   CheckUpdateGhostConservedFromPrimitive(Fluid)

   -- Test UpdateConservedFromPrimitive for internal
   fill(Fluid.Conserved, [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdateConservedFromPrimitive(p_Interior[c], Mix)
   end

   CheckUpdateConservedFromPrimitive(Fluid)

   -- Test UpdatePrimitiveFromConserved
   fill(Fluid.pressure, 0.0)
   fill(Fluid.temperature, [Tres]/[TRef])
   fill(Fluid.MolarFracs, [UTIL.mkArrayConstant(nSpec, rexpr 0.0 end)])
   fill(Fluid.velocity, array(0.0, 0.0, 0.0))
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePrimitiveFromConserved(p_Interior[c], Mix)
   end

   CheckUpdatePrimitiveFromConserved(Fluid)

   __fence(__execution, __block)

   C.printf("variablesTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "variablesTest.o", "object", REGISTRAR.register_tasks)
