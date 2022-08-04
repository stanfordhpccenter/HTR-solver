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

local types_inc_flags = terralib.newlist({"-DEOS=FFCM1Mix"})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)

-- Grid
local Grid = {
   xNum = regentlib.newsymbol(),
   yNum = regentlib.newsymbol(),
   zNum = regentlib.newsymbol(),
   xBnum = regentlib.newsymbol(),
   yBnum = regentlib.newsymbol(),
   zBnum = regentlib.newsymbol(),
   NX = regentlib.newsymbol(),
   NY = regentlib.newsymbol(),
   NZ = regentlib.newsymbol(),
   numTiles = regentlib.newsymbol(),
}
local Npx = 16
local Npy = 16
local Npz = 16
local Nx = 2
local Ny = 2
local Nz = 2

-- Laser model and dependencies
local ATOMIC = false
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)
local LASER = (require 'prometeo_laser')(SCHEMA, MIX,
                                         Fluid_columns, PART.zones_partitions, PART.ghost_partitions,
                                         ATOMIC)
local Laser = LASER.mkLaserList()
local LaserData = Laser.LaserData

-- Set gas state
local nSpec = MIX.nSpec
local nEq = CONST.GetnEq(MIX) -- Total number of unknowns for the implicit solver
local R = rexpr 8.3144598 end
local P = rexpr 101325.0 end
local T = rexpr 300.0 end
local Yi = rexpr array( -- Pure O2
   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0
) end

-- True RHS (reference value)
local rhs_true = rexpr array(
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0,  -- Zero RHS for all species
   0.0, 0.0, 0.0, -- Zero RHS for rhou
   5.5635766516 -- RHS for rhoe, due to laser energy source
   ) end

-- Nondimensionalizing factors
local LRef = rexpr 1e-3 end
local PRef = rexpr 101325.0 end
local TRef = rexpr 300.0 end
local YO2Ref = rexpr 1.0 end
local MixWRef = rexpr 2*15.999e-3 end
local rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
local eRef = rexpr [PRef]/[rhoRef] end

-- Laser parameters
local Edot = rexpr 10.0 end
local focalLocation = rexpr array(1.1, 2.2, 3.3) end
local pulseTime = rexpr 1e-3 end
local pulseFWHM = rexpr 1.414213562373095e-04 end
local axialLength = rexpr 2.0 end
local nearRadius = rexpr 0.4 end
local farRadius = rexpr 0.2 end
local beamDirection = rexpr array(0.9396926207859084,0.2620026302293850,0.2198463103929542) end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), TYPES.Fluid_columns), Mix : MIX.Mixture)
where
   writes(Fluid)
do

   -- Initialize coords
   fill(Fluid.centerCoordinates, array(1.0, 2.0, 3.0))

   -- Initialize flow state
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
                               rho*[Yi][5],
                               rho*[Yi][6],
                               rho*[Yi][7],
                               rho*[Yi][8],
                               rho*[Yi][9],
                               rho*[Yi][10],
                               rho*[Yi][11],
                               rho*[Yi][12],
                               rho*[Yi][13],
                               rho*[Yi][14],
                               rho*[Yi][15],
                               rho*[Yi][16],
                               rho*[Yi][17],
                               rho*[Yi][18],
                               rho*[Yi][19],
                               rho*[Yi][20],
                               rho*[Yi][21],
                               rho*[Yi][22],
                               rho*[Yi][23],
                               rho*[Yi][24],
                               rho*[Yi][25],
                               rho*[Yi][26],
                               rho*[Yi][27],
                               rho*[Yi][28],
                               rho*[Yi][29],
                               rho*[Yi][30],
                               rho*[Yi][31],
                               rho*[Yi][32],
                                       0.0,
                                       0.0,
                                       0.0,
                                     rho*e)) -- Assumes zero velocity
   fill(Fluid.Conserved_t,     [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
   fill(Fluid.Conserved_t_old, [UTIL.mkArrayConstant(nEq, rexpr 0.0 end)])
end

task main()

   -- Init config
   var config : Config

   -- Grid size
   config.Grid.xNum = Npx
   config.Grid.yNum = Npy
   config.Grid.zNum = Npz
   var [Grid.NX] = Nx
   var [Grid.NY] = Ny
   var [Grid.NZ] = Nz
   var [Grid.xBnum] = 0
   var [Grid.yBnum] = 0
   var [Grid.zBnum] = 0
   var [Grid.numTiles] = Grid.NX * Grid.NY * Grid.NZ

   -- Regions, tiles, partitions
   var is_Fluid = ispace(int3d, {x = Npx + 2*Grid.xBnum,
                                 y = Npy + 2*Grid.yBnum,
                                 z = Npz + 2*Grid.zBnum})
   var Fluid = region(is_Fluid, TYPES.Fluid_columns)
   var tiles = ispace(int3d, {Grid.NX, Grid.NY, Grid.NZ})
   var p_All =
      [UTIL.mkPartitionByTile(int3d, int3d, Fluid_columns, "p_All")]
      (Fluid, tiles, int3d{Grid.xBnum,Grid.yBnum,Grid.zBnum}, int3d{0,0,0})

   -- Init the mixture; must come after Fluid init
   config.Flow.mixture.type = SCHEMA.MixtureModel_FFCM1Mix
   config.Flow.mixture.u.FFCM1Mix.LRef = [LRef]
   config.Flow.mixture.u.FFCM1Mix.TRef = [TRef]
   config.Flow.mixture.u.FFCM1Mix.PRef = [PRef]
   config.Flow.mixture.u.FFCM1Mix.XiRef.Species.length = 1
   C.snprintf([&int8](config.Flow.mixture.u.FFCM1Mix.XiRef.Species.values[0].Name), 10, "O2")
   config.Flow.mixture.u.FFCM1Mix.XiRef.Species.values[0].MolarFrac = 1.0 --[MixWRef]*[YO2Ref]/(2*15.999e-3)
   var Mix = MIX.InitMixture(Fluid, tiles, p_All, config)

   -- Initialize mesh cells; must come after Mix init
   InitializeCell(Fluid, Mix)

   -- Initialize the laser model
   config.Flow.laser.type = SCHEMA.LaserModel_GeometricKernel
   config.Flow.laser.u.GeometricKernel.volume.fromCell[0] = 0
   config.Flow.laser.u.GeometricKernel.volume.fromCell[1] = 0
   config.Flow.laser.u.GeometricKernel.volume.fromCell[2] = 0
   config.Flow.laser.u.GeometricKernel.volume.uptoCell[0] = Npx-1
   config.Flow.laser.u.GeometricKernel.volume.uptoCell[1] = Npy-1
   config.Flow.laser.u.GeometricKernel.volume.uptoCell[2] = Npz-1
   config.Flow.laser.u.GeometricKernel.dimensions = 3
   config.Flow.laser.u.GeometricKernel.peakEdotPerMass = Edot
   config.Flow.laser.u.GeometricKernel.focalLocation = focalLocation
   config.Flow.laser.u.GeometricKernel.pulseTime = pulseTime
   config.Flow.laser.u.GeometricKernel.pulseFWHM = pulseFWHM
   config.Flow.laser.u.GeometricKernel.axialLength = axialLength
   config.Flow.laser.u.GeometricKernel.nearRadius = nearRadius
   config.Flow.laser.u.GeometricKernel.farRadius = farRadius
   config.Flow.laser.u.GeometricKernel.beamDirection = beamDirection; -- don't forget semicolon here...
   [LASER.DeclSymbols(Laser, Grid, Fluid, tiles, p_All, config)];

   -- Compute kernel profile
   for c in LaserData.Laser_tiles do
      LASER.computeKernelProfile(LaserData.p_Laser[c], config)
   end

   -- Compute laser energy source
   var Integrator_simTime = pulseTime + 0.2*pulseFWHM
   for c in LaserData.Laser_tiles do
      LASER.AddLaserGeometricKernel(LaserData.p_Laser[c], Integrator_simTime, config)
   end

   -- Test
   for c in Fluid do
      for i = 0, nEq do
         var err = Fluid[c].Conserved_t[i] - [rhs_true][i]
         if ( [rhs_true][i] ~= 0) then
            err /= [rhs_true][i]
         end
         --C.printf('i=%i, rho=%17.10e, rhs=%17.10e, rhs_true=%17.10e, err=%17.10e\n',
         --   i,Fluid[c].rho,Fluid[c].Conserved_t[i],[rhs_true][i],err)
         regentlib.assert(fabs(err) < 1e-5, "laserTest: ERROR in RHS.")
      end
   end

   __fence(__execution, __block)

   C.printf("laserTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "laserTest.o", "object", REGISTRAR.register_tasks)
