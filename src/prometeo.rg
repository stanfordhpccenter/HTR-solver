-- Copyright (c) "2019, by Stanford University
--               Developer: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
--                         HTR solver: An open-source exascale-oriented task-based
--                         multi-GPU high-order code for hypersonic aerothermodynamics.
--                         Computer Physics Communications 255, 107262"
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
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local SCHEMA    = terralib.includec("config_schema.h")
local MAPPER    = terralib.includec("prometeo_mapper.h")
local REGISTRAR = terralib.includec("prometeo_registrar.h")
local UTIL = require "util"
local VERSION = require "version"

-------------------------------------------------------------------------------
-- ACTIVATE DEBUG_OUTPUT
-------------------------------------------------------------------------------

local DEBUG_OUTPUT = false
if os.getenv("DEBUG_OUTPUT") == "1" then
   DEBUG_OUTPUT = true
   print("#############################################################################")
   print("WARNING: You are compiling with debug output.")
   print("         This might affect the performance of the solver.")
   print("#############################################################################")
end

-------------------------------------------------------------------------------
-- ACTIVATE ATOMIC COHERENCE MODE
-------------------------------------------------------------------------------

local ATOMIC = true
if os.getenv("NO_ATOMIC") == "1" then
   ATOMIC = false
   print("#############################################################################")
   print("WARNING: You are compiling without atomic coherence mode.")
   print("         This might affect the performance of the solver.")
   print("#############################################################################")
end

-------------------------------------------------------------------------------
-- ACTIVATE AVERAGES
-------------------------------------------------------------------------------

local AVERAGES = true
if os.getenv("AVERAGES") == "0" then
   AVERAGES = false
   print("#############################################################################")
   print("WARNING: You are compiling without averaging tools.")
   print("#############################################################################")
end

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

-------------------------------------------------------------------------------
-- CHECK THAT MIXTURE VARIABLE IS WELL SET
-------------------------------------------------------------------------------

local MIX
if (os.getenv("EOS") == "ConstPropMix") then
elseif (os.getenv("EOS") == "IsentropicMix") then
elseif (os.getenv("EOS") == "AirMix") then
elseif (os.getenv("EOS") == "CH41StMix") then
elseif (os.getenv("EOS") == "CH4_30SpMix") then
elseif (os.getenv("EOS") == "CH4_43SpIonsMix") then
elseif (os.getenv("EOS") == "CH4_26SpIonsMix") then
elseif (os.getenv("EOS") == "FFCM1Mix") then
elseif (os.getenv("EOS") == "BoivinMix") then
elseif (os.getenv("EOS") == "H2_UCSDMix") then
elseif (os.getenv("EOS") == nil) then
   error ("You must define EOS enviromnment variable")
else
   error ("Unrecognized mixture: " .. os.getenv("EOS"))
end

-------------------------------------------------------------------------------
-- DATA STRUCTURES
-------------------------------------------------------------------------------

local Config = SCHEMA.Config
--local MultiConfig = SCHEMA.MultiConfig

local types_inc_flags = terralib.newlist({"-DEOS="..os.getenv("EOS")})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local Fluid_columns = TYPES.Fluid_columns
local bBoxType = TYPES.bBoxType

-------------------------------------------------------------------------------
-- CONSTANTS
-------------------------------------------------------------------------------
local CONST = require "prometeo_const"

-- Runge-Kutta coefficients
local RK_C = CONST.RK_C
local RK_T = CONST.RK_T

-- Variable indices
local nSpec = TYPES.nSpec       -- Number of species composing the mixture
local nEq = CONST.GetnEq(TYPES) -- Total number of unknowns for the implicit solver

-------------------------------------------------------------------------------
-- DEFINE I/O VARIABLES
-------------------------------------------------------------------------------

local IOVars = terralib.newlist({
   'rho',
   'pressure',
   'temperature',
   'MolarFracs',
   'velocity',
   'dudtBoundary',
   'dTdtBoundary'
})

local vProbesVars = terralib.newlist({
   'rho',
   'pressure',
   'temperature',
   'MolarFracs',
   'velocity'
})

local DebugVars = terralib.newlist({
   'rho',
   'pressure',
   'temperature',
   'MolarFracs',
   'velocity',
   'Conserved',
   'shockSensorX',
   'shockSensorY',
   'shockSensorZ'
})

-- Add electric varaibles to the output
if ELECTRIC_FIELD then
   IOVars:insert("electricPotential")
   DebugVars:insert("electricPotential")
end

-------------------------------------------------------------------------------
-- EXTERNAL MODULES IMPORTS
-------------------------------------------------------------------------------

local HDF = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                                 IOVars,
                                                 {timeStep=int, simTime=double, channelForcing=double},
                                                 {SpeciesNames={nSpec, 20}, Versions={2, VERSION.Length}})

local HDF_VPROBES = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                                 vProbesVars,
                                                 {timeStep=int, simTime=double},
                                                 {SpeciesNames={nSpec, 20}, Versions={2, VERSION.Length}})

local HDF_DEBUG
if DEBUG_OUTPUT then
   HDF_DEBUG = (require 'hdf_helper')(int3d, int3d, Fluid_columns,
                                                 DebugVars,
                                                 {timeStep=int, simTime=double, channelForcing=double},
                                                 {SpeciesNames={nSpec, 20}, Versions={2, VERSION.Length}})
end

-- Macro
local MACRO = require "prometeo_macro"

-- I/O routines
local IO = (require 'prometeo_IO')(SCHEMA)

-- Mixture registration routines
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)

-- Partitioning routines
local PART = (require 'prometeo_partitioner')(SCHEMA, Fluid_columns)

-- Mesh routines
local GRID = (require 'prometeo_grid')(SCHEMA, IO, Fluid_columns, bBoxType,
                                       PART.zones_partitions, PART.output_partitions)

-- Metric routines
local METRIC = (require 'prometeo_metric')(SCHEMA, TYPES,
                                           PART.zones_partitions, PART.ghost_partitions)

-- Stability conditions routines
local CFL = (require 'prometeo_cfl')(MIX, TYPES, ELECTRIC_FIELD)

-- Chemistry routines
local CHEM = (require 'prometeo_chem')(SCHEMA, MIX, TYPES, ATOMIC)

-- Initialization routines
local INIT = (require 'prometeo_init')(SCHEMA, MIX, Fluid_columns, bBoxType)

-- Conserved->Primitives/Primitives->Conserved and properties routines
local VARS = (require 'prometeo_variables')(SCHEMA, MIX, METRIC, TYPES, ELECTRIC_FIELD)

-- Fluxes routines
local SENSOR = (require 'prometeo_sensor')(SCHEMA, MIX, TYPES, Fluid_columns,
                                           PART.zones_partitions, PART.ghost_partitions)

-- BCOND routines
local BCOND = (require 'prometeo_bc')(SCHEMA, MIX, TYPES, PART.zones_partitions,
                                      ELECTRIC_FIELD)

-- RK routines
local RK = (require 'prometeo_rk')(nEq, Fluid_columns)


-- Volume averages routines
local STAT = (require 'prometeo_stat')(MIX, Fluid_columns)

-- RHS routines
local RHS = (require 'prometeo_rhs')(SCHEMA, MIX, METRIC, TYPES, STAT,
                                     PART.zones_partitions, PART.ghost_partitions,
                                     BCOND.IncomingShockParams,
                                     ATOMIC)

-- Laser routines
local LASER = (require 'prometeo_laser')(SCHEMA, MIX,
                                         Fluid_columns, PART.zones_partitions, PART.ghost_partitions,
                                         ATOMIC)

-- Profiles routines
local PROFILES = (require 'prometeo_profiles')(SCHEMA, MIX, Fluid_columns)

-- Averages routines
local AVG
if AVERAGES then
   AVG = (require 'prometeo_average')(SCHEMA, MIX, TYPES, PART,
                                      ELECTRIC_FIELD)
end

-- Probes routines
local PROBES = (require 'prometeo_probe')(SCHEMA, MIX, IO, Fluid_columns)

local Efield
if ELECTRIC_FIELD then
   EFIELD = (require "prometeo_electricField")(SCHEMA, MIX, TYPES,
                                               PART.zones_partitions, PART.ghost_partitions,
                                               ATOMIC)
end

-------------------------------------------------------------------------------
-- INITIALIZATION ROUTINES
-------------------------------------------------------------------------------

local function emitFill(p, t, f, val) return rquote
   var v = [val]
   __demand(__index_launch)
   for c in t do fill(([p][c]).[f], v) end
end end

__demand(__inline)
task InitializeCell(Fluid : region(ispace(int3d), Fluid_columns),
                    tiles : ispace(int3d),
                    Fluid_Zones : PART.zones_partitions(Fluid, tiles))
where
   writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All} = Fluid_Zones;

   [emitFill(p_All, tiles, "centerCoordinates", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "nType_x", rexpr 0 end)];
   [emitFill(p_All, tiles, "nType_y", rexpr 0 end)];
   [emitFill(p_All, tiles, "nType_z", rexpr 0 end)];
   [emitFill(p_All, tiles, "dcsi_e", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "deta_e", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "dzet_e", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "dcsi_d", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "deta_d", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "dzet_d", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "dcsi_s", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "deta_s", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "dzet_s", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "rho", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "mu" , rexpr 0.0 end)];
   [emitFill(p_All, tiles, "lam", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "Di" , UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "SoS", rexpr 0.0 end)];
rescape if (ELECTRIC_FIELD and  (MIX.nIons > 0)) then remit rquote
   [emitFill(p_All, tiles, "Ki", UTIL.mkArrayConstant(MIX.nIons, rexpr 0.0 end))];
end end end
   [emitFill(p_All, tiles, "pressure", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "temperature", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "MassFracs",  UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "MolarFracs", UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "velocity",            rexpr array(0.0, 0.0, 0.0) end)];
rescape if ELECTRIC_FIELD then remit rquote
   [emitFill(p_All, tiles, "electricPotential", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "electricField", rexpr array(0.0, 0.0, 0.0) end)];
end end end
   [emitFill(p_All, tiles, "Conserved",       UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_old",   UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_t",     UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "Conserved_t_old", UTIL.mkArrayConstant(nEq, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "shockSensorX", rexpr true end)];
   [emitFill(p_All, tiles, "shockSensorY", rexpr true end)];
   [emitFill(p_All, tiles, "shockSensorZ", rexpr true end)];
   [emitFill(p_All, tiles, "DucrosSensor", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "dudtBoundary", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "dTdtBoundary", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "velocity_old_NSCBC", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "temperature_old_NSCBC", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "MolarFracs_profile", UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "velocity_profile", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "temperature_profile", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "temperature_recycle", rexpr 0.0 end)];
   [emitFill(p_All, tiles, "MolarFracs_recycle", UTIL.mkArrayConstant(nSpec, rexpr 0.0 end))];
   [emitFill(p_All, tiles, "velocity_recycle", rexpr array(0.0, 0.0, 0.0) end)];
   [emitFill(p_All, tiles, "kernelProfile", rexpr 0.0 end)];
end

-------------------------------------------------------------------------------
-- DEBUG ROUTINES
-------------------------------------------------------------------------------

local DetectNaN
local CheckDebugOutput
if DEBUG_OUTPUT then

local isnan = regentlib.isnan(double)
__demand(__cuda, __leaf) -- MANUALLY PARALLELIZED
task DetectNaN(Fluid : region(ispace(int3d), Fluid_columns))
where
   reads(Fluid.{pressure, temperature}),
   reads(Fluid.Conserved)
do
   var err = 0
   __demand(__openmp)
   for c in Fluid do
      if ([bool](isnan(Fluid[c].pressure   ))) then err += 1 end
      if ([bool](isnan(Fluid[c].temperature))) then err += 1 end
      for i=0, nEq do
         if ([bool](isnan(Fluid[c].Conserved[i]))) then err += 1 end
      end
   end
   return err
end

__demand(__inline)
task CheckDebugOutput(Fluid      : region(ispace(int3d), Fluid_columns),
                      Fluid_copy : region(ispace(int3d), Fluid_columns),
                      tiles : ispace(int3d),
                      tiles_output : ispace(int3d),
                      Fluid_Zones : PART.zones_partitions(Fluid, tiles),
                      Fluid_Output      : PART.output_partitions(Fluid,      tiles_output),
                      Fluid_Output_copy : PART.output_partitions(Fluid_copy, tiles_output),
                      config : Config,
                      Mix : MIX.Mixture,
                      Integrator_timeStep : int,
                      Integrator_simTime : double)
where
   reads(Fluid),
   reads writes(Fluid_copy),
   Fluid * Fluid_copy
do
   -- Unpack the partitions that we are going to need
   var {p_All} = Fluid_Zones
   var {p_Output } = Fluid_Output
   var {p_Output_copy=p_Output} = Fluid_Output_copy

   var err = 0
   __demand(__index_launch)
   for c in tiles do
      err += DetectNaN(p_All[c])
   end
   if err ~= 0 then
      var dirname = [&int8](C.malloc(256))
      C.snprintf(dirname, 256, '%s/debugOut', config.Mapping.outDir)
      var _1 = IO.createDir(0, dirname)
      _1 = HDF_DEBUG.dump(                 _1, tiles_output, dirname, Fluid, Fluid_copy, p_Output, p_Output_copy)
      _1 = HDF_DEBUG.write.timeStep(       _1, dirname, Integrator_timeStep)
      _1 = HDF_DEBUG.write.simTime(        _1, dirname, Integrator_simTime)
      _1 = HDF_DEBUG.write.SpeciesNames(   _1, dirname, MIX.GetSpeciesNames(Mix))
      _1 = HDF_DEBUG.write.Versions(       _1, dirname, array(regentlib.string([VERSION.SolverVersion]), regentlib.string([VERSION.LegionVersion])))
      _1 = HDF_DEBUG.write.channelForcing( _1, dirname, config.Flow.turbForcing.u.CHANNEL.Forcing);
      C.free(dirname)

      __fence(__execution, __block)
      regentlib.assert(false, "NaN detected! Debug fields dumped in debugOut")
   end
end
end

-------------------------------------------------------------------------------
-- RK-LOOP ROUTINES
-------------------------------------------------------------------------------

__demand(__inline)
task UpdatePrimitivesAndBCsFromConserved(Fluid : region(ispace(int3d), Fluid_columns),
                                         tiles : ispace(int3d),
                                         Fluid_Zones : PART.zones_partitions(Fluid, tiles),
                                         Fluid_Ghost : PART.ghost_partitions(Fluid, tiles),
                                         BCParams : BCOND.BCParamsStruct(Fluid, tiles),
                                         BCRecycleAverage   : region(ispace(int1d), BCOND.RecycleAverageType),
                                         BCRecycleAverageFI : region(ispace(int1d), BCOND.RecycleAverageFIType),
                                         config             : Config,
                                         Mix                : MIX.Mixture,
                                         bBox               : bBoxType,
                                         Integrator_simTime : double)
where
   reads(BCRecycleAverageFI),
   reads writes(Fluid),
   reads writes(BCRecycleAverage)
do
   -- Unpack the partitions that we are going to need
   var {p_All, p_Interior, p_AllBCs} = Fluid_Zones;
   var {p_GradientGhosts} = Fluid_Ghost;

   -- Update all primitive variables...
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePrimitiveFromConserved(p_Interior[c], Mix)
   end

   -- ...also in the ghost cells
   BCOND.UpdateGhostPrimitives(Fluid,
                               tiles,
                               Fluid_Zones,
                               BCParams,
                               BCRecycleAverage,
                               BCRecycleAverageFI,
                               config,
                               Mix,
                               bBox,
                               Integrator_simTime);

   -- Update the mixture properties everywhere
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdatePropertiesFromPrimitive(p_All[c], Mix)
   end

   -- update values of conserved variables in ghost cells
   __demand(__index_launch)
   for c in tiles do
      VARS.UpdateConservedFromPrimitive(p_AllBCs[c], Mix)
   end
end

__demand(__inline)
task UpdateDerivatives(Fluid : region(ispace(int3d), Fluid_columns),
                       tiles : ispace(int3d),
                       Fluid_Zones : PART.zones_partitions(Fluid, tiles),
                       Fluid_Ghost : PART.ghost_partitions(Fluid, tiles),
                       BCParams : BCOND.BCParamsStruct(Fluid, tiles),
                       LaserData : LASER.LaserStruct(Fluid, tiles),
                       config               : Config,
                       Mix                  : MIX.Mixture,
                       bBox                 : bBoxType,
                       RK_coeffs            : double[2],
                       Integrator_deltaTime : double,
                       Integrator_simTime   : double,
                       interior_volume      : double,
                       UseOldDerivatives    : bool)
where
   reads writes(Fluid)
do
   -- Unpack the partitions that we are going to need
   var {p_All, p_Interior, p_solved} = Fluid_Zones;
   var {p_GradientGhosts} = Fluid_Ghost;

   -- Initialize time derivatives to 0 or minus the old value
   if UseOldDerivatives then
      __demand(__index_launch)
      for c in tiles do
         [RK.mkInitializeTimeDerivatives(true)](p_All[c])
      end
      -- In this case we advance the equations by half deltaTime
      Integrator_deltaTime *= 0.5
   else
      __demand(__index_launch)
      for c in tiles do
         [RK.mkInitializeTimeDerivatives(false)](p_All[c])
      end
   end

   if (not config.Integrator.implicitChemistry) then
      -- Add chemistry source terms
      __demand(__index_launch)
      for c in tiles do
         CHEM.AddChemistrySources(p_solved[c], Mix)
      end
   end

   -- Add body forces
   __demand(__index_launch)
   for c in tiles do
      RHS.AddBodyForces(p_solved[c], config.Flow.bodyForce)
   end

   -- Add ion-wind source terms
rescape if ELECTRIC_FIELD then remit rquote
   __demand(__index_launch)
   for c in tiles do
      EFIELD.AddIonWindSources(p_GradientGhosts[c], p_solved[c], Fluid.bounds, Mix);
   end
end end end

   -- Add laser source
   if (config.Flow.laser.type == SCHEMA.LaserModel_Algebraic) then
      __demand(__index_launch)
      for c in LaserData.Laser_tiles do
         LASER.AddLaserAlgebraic(LaserData.p_Laser[c],
                                 config.Flow.laser.u.Algebraic.Dimension,
                                 config.Flow.laser.u.Algebraic.Amplitude,
                                 config.Flow.laser.u.Algebraic.Center,
                                 config.Flow.laser.u.Algebraic.Radius,
                                 config.Flow.laser.u.Algebraic.Delay,
                                 config.Flow.laser.u.Algebraic.Duration,
                                 Integrator_simTime)
      end
   elseif (config.Flow.laser.type == SCHEMA.LaserModel_GeometricKernel) then
      __demand(__index_launch)
      for c in LaserData.Laser_tiles do
         LASER.AddLaserGeometricKernel(LaserData.p_Laser[c], Integrator_simTime, config)
      end
   end

   -- Add turbulent forcing
   if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
      -- Add forcing
      __demand(__index_launch)
      for c in tiles do
         RHS.AddBodyForces(p_solved[c],
                           array(config.Flow.turbForcing.u.CHANNEL.Forcing, 0.0, 0.0))
      end

   elseif config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_HIT then
      RHS.UpdateUsingHITForcing(Fluid, tiles, Fluid_Zones, Fluid_Ghost,
                                Mix, config, interior_volume)

   end

   -- Use Euler fluxes to update conserved value derivatives
   RHS.UpdateUsingEulerFlux(Fluid, tiles, Fluid_Zones, Fluid_Ghost, RK_coeffs, Integrator_deltaTime, Mix, config);

   -- Use diffusion fluxes to update conserved variables derivatives
   RHS.UpdateUsingDiffusionFlux(Fluid, tiles, Fluid_Zones, Fluid_Ghost, Mix, config);

rescape if (ELECTRIC_FIELD and (MIX.nIons > 0)) then remit rquote
   -- Use ion drift fluxes to update conserved variables derivatives
   EFIELD.UpdateUsingIonDriftFlux(Fluid, tiles, Fluid_Zones, Fluid_Ghost, Mix, config);
end end end

   -- Add buffer zone sources
   if (config.BC.bufferZone.type == SCHEMA.BufferZoneBC_Basic) then
      __demand(__index_launch)
      for c in tiles do
         RHS.AddBufferZoneSource(p_solved[c], Mix, bBox, config)
      end
   end

   -- Update using NSCBC_Outflow bcs
   RHS.UpdateUsingNSCBCOutflow(Fluid, tiles, Fluid_Zones, Fluid_Ghost, Mix, config);

   -- Update using NSCBC_FarField bcs
   RHS.UpdateUsingNSCBCFarField(Fluid, tiles, Fluid_Zones, Fluid_Ghost, Mix, config);

   -- Update using IncomingShock bcs
   RHS.UpdateUsingNSCBCIncomingShock(Fluid, tiles, Fluid_Zones, Fluid_Ghost,
                                     BCParams.IncomingShock, Mix, config);

   -- Update using NSCBC_Inflow bcs
   RHS.UpdateUsingNSCBCInflow(Fluid, tiles, Fluid_Zones, Fluid_Ghost, Mix, config);

end

-------------------------------------------------------------------------------
-- MAIN SIMULATION
-------------------------------------------------------------------------------

local function mkInstance() local INSTANCE = {}

   -----------------------------------------------------------------------------
   -- Symbols shared between quotes
   -----------------------------------------------------------------------------

   local startTime = regentlib.newsymbol()
   local Grid = {
      xBnum = regentlib.newsymbol(),
      yBnum = regentlib.newsymbol(),
      zBnum = regentlib.newsymbol(),
      NX = regentlib.newsymbol(),
      NY = regentlib.newsymbol(),
      NZ = regentlib.newsymbol(),
      boundingBox = regentlib.newsymbol(bBoxType),
      numTiles = regentlib.newsymbol(),
      NXout = regentlib.newsymbol(),
      NYout = regentlib.newsymbol(),
      NZout = regentlib.newsymbol(),
      numTilesOut = regentlib.newsymbol(),
   }

   -- Boundary conditions symbols
   local BC = BCOND.mkBCDataList()

   local Integrator_deltaTime = regentlib.newsymbol()
   local Integrator_simTime   = regentlib.newsymbol()
   local Integrator_timeStep  = regentlib.newsymbol()
   local Integrator_exitCond  = regentlib.newsymbol()

   local Mix = regentlib.newsymbol()

   local Fluid = regentlib.newsymbol("Fluid")
   local Fluid_copy = regentlib.newsymbol("Fluid_copy")
   local Fluid_bounds = regentlib.newsymbol("Fluid_bounds")

   local tiles = regentlib.newsymbol()
   local tiles_output = regentlib.newsymbol()

   local Fluid_Zones = regentlib.newsymbol("Fluid_Zones")
   local Fluid_Ghost = regentlib.newsymbol("Fluid_Ghost")

   local Fluid_Output = regentlib.newsymbol("Fluid_Output")
   local Fluid_Output_copy = regentlib.newsymbol("Fluid_Output_copy")

   local interior_volume = regentlib.newsymbol(double)

   -- Averages symbols
   local Averages
   if AVERAGES then
      Averages = AVG.mkAvgList()
   end

   -- Probes symbols
   local Probes = PROBES.mkProbesList()

   -- Laser data
   local Laser = LASER.mkLaserList()

   -- Electric field symbols
   local EfieldData
   if ELECTRIC_FIELD then
      EfieldData = EFIELD.mkDataList()
   end

   -----------------------------------------------------------------------------
   -- Exported symbols
   -----------------------------------------------------------------------------

   INSTANCE.Grid = Grid
   INSTANCE.Integrator_deltaTime = Integrator_deltaTime
   INSTANCE.Integrator_simTime   = Integrator_simTime
   INSTANCE.Integrator_timeStep  = Integrator_timeStep
   INSTANCE.Integrator_exitCond  = Integrator_exitCond
   INSTANCE.Fluid = Fluid
   INSTANCE.Fluid_copy = Fluid_copy
   INSTANCE.tiles = tiles
   INSTANCE.Fluid_Zones = Fluid_Zones

   -----------------------------------------------------------------------------
   -- Symbol declaration & initialization
   -----------------------------------------------------------------------------

   function INSTANCE.DeclSymbols(config) return rquote

      ---------------------------------------------------------------------------
      -- Preparation
      ---------------------------------------------------------------------------

      -- Start timer
      var [startTime] = __future(int64, C.legion_issue_timing_op_microseconds(__runtime(), __context()));

      -- Write console header
      IO.Console_WriteHeader([&int8](config.Mapping.outDir))

      ---------------------------------------------------------------------------
      -- Declare & initialize state variables
      ---------------------------------------------------------------------------

      -- Determine number of ghost cells in each direction
      -- 0 ghost cells if periodic and 1 otherwise
      var [Grid.xBnum] = 1
      var [Grid.yBnum] = 1
      var [Grid.zBnum] = 1
      if config.BC.xBCLeft.type == SCHEMA.FlowBC_Periodic then Grid.xBnum = 0 end
      if config.BC.yBCLeft.type == SCHEMA.FlowBC_Periodic then Grid.yBnum = 0 end
      if config.BC.zBCLeft.type == SCHEMA.FlowBC_Periodic then Grid.zBnum = 0 end
      if config.BC.xBCLeft.type ~= SCHEMA.FlowBC_Periodic then regentlib.assert(config.Grid.xNum > 4, "HTR needs at least five points along non-periodic boundaries (x dir)") end
      if config.BC.yBCLeft.type ~= SCHEMA.FlowBC_Periodic then regentlib.assert(config.Grid.yNum > 4, "HTR needs at least five points along non-periodic boundaries (y dir)") end
      if config.BC.zBCLeft.type ~= SCHEMA.FlowBC_Periodic then regentlib.assert(config.Grid.zNum > 4, "HTR needs at least five points along non-periodic boundaries (z dir)") end

      var [Grid.NX] = config.Mapping.tiles[0]
      var [Grid.NY] = config.Mapping.tiles[1]
      var [Grid.NZ] = config.Mapping.tiles[2]
      var [Grid.numTiles] = Grid.NX * Grid.NY * Grid.NZ

      var [Grid.NXout] = int(config.Mapping.tiles[0]/config.Mapping.tilesPerRank[0])
      var [Grid.NYout] = int(config.Mapping.tiles[1]/config.Mapping.tilesPerRank[1])
      var [Grid.NZout] = int(config.Mapping.tiles[2]/config.Mapping.tilesPerRank[2])
      var [Grid.numTilesOut] = Grid.NXout * Grid.NYout * Grid.NZout

      var [Integrator_exitCond]  = true
      var [Integrator_simTime]   = config.Integrator.startTime
      var [Integrator_timeStep]  = config.Integrator.startIter
      var [Integrator_deltaTime] = 0.0;

      ---------------------------------------------------------------------------
      -- Initialize forcing values
      ---------------------------------------------------------------------------

      if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
         config.Flow.turbForcing.u.CHANNEL.Forcing max= 1.0
      end

      ---------------------------------------------------------------------------
      -- Create Regions and Partitions
      ---------------------------------------------------------------------------

      var sampleId = config.Mapping.sampleId

      -- Create Fluid Regions
      var is_Fluid = ispace(int3d, {x = config.Grid.xNum + 2*Grid.xBnum,
                                    y = config.Grid.yNum + 2*Grid.yBnum,
                                    z = config.Grid.zNum + 2*Grid.zBnum})
      var [Fluid] = region(is_Fluid, Fluid_columns);
      [UTIL.emitRegionTagAttach(Fluid, MAPPER.SAMPLE_ID_TAG, sampleId, int)];
      var [Fluid_bounds] = [Fluid].bounds
      var [Fluid_copy] = region(is_Fluid, Fluid_columns);
      [UTIL.emitRegionTagAttach(Fluid_copy, MAPPER.SAMPLE_ID_TAG, sampleId, int)];

      -- Partitioning domain
      var [tiles] = ispace(int3d, {Grid.NX, Grid.NY, Grid.NZ})

      -- Fluid Partitioning
      var [Fluid_Zones] = PART.PartitionZones(Fluid, tiles, config, Grid.xBnum, Grid.yBnum, Grid.zBnum)

      -- Create partitions to support stencils
      var [Fluid_Ghost] = PART.PartitionGhost(Fluid, tiles, Fluid_Zones, config)

      -- Unpack the partitions that we are going to need
      var {p_All} = Fluid_Zones;

      -- Output partitionig
      var [tiles_output] = ispace(int3d, {Grid.NXout, Grid.NYout, Grid.NZout})
      var [Fluid_Output]      = PART.PartitionOutput(Fluid,      tiles_output, config,
                                                     Grid.xBnum, Grid.yBnum, Grid.zBnum)
      var [Fluid_Output_copy] = PART.PartitionOutput(Fluid_copy, tiles_output, config,
                                                     Grid.xBnum, Grid.yBnum, Grid.zBnum)

      ---------------------------------------------------------------------------
      -- Initialize the mixture
      ---------------------------------------------------------------------------
      var [Mix] = MIX.InitMixture(Fluid, tiles, Fluid_Zones.p_All, config);

      ---------------------------------------------------------------------------
      -- Declare BC symbols
      ---------------------------------------------------------------------------
      [BCOND.DeclSymbols(BC, Fluid, Grid, config)];

      ---------------------------------------------------------------------------
      -- Declare averages symbols
      ---------------------------------------------------------------------------
rescape if AVERAGES then remit rquote
      [AVG.DeclSymbols(Averages, Grid, Fluid, p_All, config, MAPPER)];
      -- Create averages partitions
      [AVG.InitPartitions(Averages, Grid, Fluid, p_All, config)];
end end end

      ---------------------------------------------------------------------------
      -- Declare probes symbols
      ---------------------------------------------------------------------------
      [PROBES.DeclSymbols(Probes, Grid, Fluid, p_All, config)];

      ---------------------------------------------------------------------------
      -- Create directory for volume probes
      ---------------------------------------------------------------------------
      for p=0, config.IO.volumeProbes.length do
         var vProbe = config.IO.volumeProbes.values[p]
         var dirname = [&int8](C.malloc(256))
         C.snprintf(dirname, 256, '%s/%s', config.Mapping.outDir, vProbe.outDir)
         var _1 = IO.createDir(0, dirname)
         C.free(dirname)
      end

      ---------------------------------------------------------------------------
      -- Declare symbols for laser model
      ---------------------------------------------------------------------------
      [LASER.DeclSymbols(Laser, Grid, Fluid, tiles, p_All, config)];

      ---------------------------------------------------------------------------
      -- Declare electric field solver symbols
      ---------------------------------------------------------------------------
rescape if ELECTRIC_FIELD then remit rquote
      [EFIELD.DeclSymbols(EfieldData, Fluid, tiles, Fluid_Zones, Grid, config, MAPPER)];
end end end

   end end -- DeclSymbols

   -----------------------------------------------------------------------------
   -- Region initialization
   -----------------------------------------------------------------------------

   function INSTANCE.InitRegions(config) return rquote

      ---------------------------------------------------------------------------
      -- Initialize fluid region
      ---------------------------------------------------------------------------
      InitializeCell(Fluid, tiles, Fluid_Zones);

      -- Unpack the partitions that we are going to need
      var {p_All, p_Interior, p_AllBCs,
             xNeg,   xPos,   yNeg,   yPos,   zNeg,   zPos} = Fluid_Zones
      var {p_Output} = Fluid_Output
      var {p_Output_copy=p_Output} = Fluid_Output_copy

      ---------------------------------------------------------------------------
      -- Initialize grid operators
      ---------------------------------------------------------------------------
      METRIC.InitializeOperators(Fluid, tiles, Fluid_Zones, config,
                                 Grid.xBnum, Grid.yBnum, Grid.zBnum)

      ---------------------------------------------------------------------------
      -- Initialize the grid geometry
      ---------------------------------------------------------------------------
      var [Grid.boundingBox] = GRID.InitializeGeometry(Fluid, tiles, Fluid_Zones, config)

      ---------------------------------------------------------------------------
      -- Dump cell center grid once and for all
      ---------------------------------------------------------------------------
      GRID.dumpCellCenterGrid(Fluid, Fluid_copy, tiles_output, Fluid_Output, Fluid_Output_copy, config)

      ---------------------------------------------------------------------------
      -- Initialize averages
      ---------------------------------------------------------------------------
rescape if AVERAGES then remit rquote
      -- Initialize averages
      [AVG.InitRakesAndPlanes(Averages)];
end end end

      ---------------------------------------------------------------------------
      -- Initialize solution
      ---------------------------------------------------------------------------
      if config.Flow.initCase.type == SCHEMA.FlowInitCase_Uniform then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initCase.u.Uniform.molarFracs, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeUniform(p_All[c],
                                   config.Flow.initCase.u.Uniform.pressure,
                                   config.Flow.initCase.u.Uniform.temperature,
                                   config.Flow.initCase.u.Uniform.velocity,
                                   initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_Random then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initCase.u.Random.molarFracs, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeRandom(p_All[c],
                                  config.Flow.initCase.u.Random.pressure,
                                  config.Flow.initCase.u.Random.temperature,
                                  config.Flow.initCase.u.Random.magnitude,
                                  initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_TaylorGreen2DVortex then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initCase.u.TaylorGreen2DVortex.molarFracs, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeTaylorGreen2D(p_All[c],
                                         config.Flow.initCase.u.TaylorGreen2DVortex.pressure,
                                         config.Flow.initCase.u.TaylorGreen2DVortex.temperature,
                                         config.Flow.initCase.u.TaylorGreen2DVortex.velocity,
                                         initMolarFracs,
                                         Mix)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_TaylorGreen3DVortex then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initCase.u.TaylorGreen3DVortex.molarFracs, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeTaylorGreen3D(p_All[c],
                                         config.Flow.initCase.u.TaylorGreen3DVortex.pressure,
                                         config.Flow.initCase.u.TaylorGreen3DVortex.temperature,
                                         config.Flow.initCase.u.TaylorGreen3DVortex.velocity,
                                         initMolarFracs,
                                         Mix)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_Perturbed then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initCase.u.Perturbed.molarFracs, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializePerturbed(p_All[c],
                                     config.Flow.initCase.u.Perturbed.pressure,
                                     config.Flow.initCase.u.Perturbed.temperature,
                                     config.Flow.initCase.u.Perturbed.velocity,
                                     initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_RiemannTestOne then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeRiemannTestOne(p_All[c], initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_RiemannTestTwo then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeRiemannTestTwo(p_All[c], initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_SodProblem then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeSodProblem(p_All[c], initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_LaxProblem then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeLaxProblem(p_All[c], initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_ShuOsherProblem then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeShuOsherProblem(p_All[c], initMolarFracs)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_VortexAdvection2D then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeVortexAdvection2D(p_All[c],
                                             config.Flow.initCase.u.VortexAdvection2D.pressure,
                                             config.Flow.initCase.u.VortexAdvection2D.temperature,
                                             config.Flow.initCase.u.VortexAdvection2D.velocity[0],
                                             config.Flow.initCase.u.VortexAdvection2D.velocity[1],
                                             initMolarFracs,
                                             Mix)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_GrossmanCinnellaProblem then
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeGrossmanCinnellaProblem(p_All[c], Mix)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_ChannelFlow then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initCase.u.ChannelFlow.molarFracs, Mix)
         __demand(__index_launch)
         for c in tiles do
            INIT.InitializeChannelFlow(p_All[c],
                                       config.Flow.initCase.u.ChannelFlow.pressure,
                                       config.Flow.initCase.u.ChannelFlow.temperature,
                                       config.Flow.initCase.u.ChannelFlow.velocity,
                                       config.Flow.initCase.u.ChannelFlow.StreaksIntensity,
                                       config.Flow.initCase.u.ChannelFlow.RandomIntensity,
                                       initMolarFracs,
                                       Mix,
                                       Grid.boundingBox)
         end

      elseif config.Flow.initCase.type == SCHEMA.FlowInitCase_Restart then
         var restartDir = [&int8](config.Flow.initCase.u.Restart.restartDir)
         Integrator_timeStep = HDF.read.timeStep(tiles_output, restartDir, Fluid, p_Output)
         Integrator_simTime  = HDF.read.simTime( tiles_output, restartDir, Fluid, p_Output)
         if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
            config.Flow.turbForcing.u.CHANNEL.Forcing = HDF.read.channelForcing(tiles_output, restartDir, Fluid, p_Output)
         end
         HDF.load(tiles_output, restartDir, Fluid, Fluid_copy, p_Output, p_Output_copy);

rescape if AVERAGES then remit rquote
         [AVG.ReadAverages(Averages, config)];
end end end

      else regentlib.assert(false, 'Unhandled case in switch') end

      if config.Integrator.resetTime then
         Integrator_simTime  = config.Integrator.startTime
         Integrator_timeStep = config.Integrator.startIter
      end

      if config.Flow.resetMixture then
         var initMolarFracs = MIX.ParseConfigMixture(config.Flow.initMixture, Mix)
         __demand(__index_launch)
         for c in tiles do
            CHEM.ResetMixture(p_Interior[c], initMolarFracs)
         end
      end

      ---------------------------------------------------------------------------
      -- Initialize grid metric
      ---------------------------------------------------------------------------
      METRIC.InitializeMetric(Fluid, tiles, Fluid_Zones, Fluid_Ghost,
                              Grid.boundingBox, config);

      ---------------------------------------------------------------------------
      -- Check boundary condition inputs
      ---------------------------------------------------------------------------
      [BCOND.CheckInput(BC, config, Grid.boundingBox)];

      ---------------------------------------------------------------------------
      -- Initialize boundary condition profiles
      ---------------------------------------------------------------------------
      -- Read from file...
      -- TODO: this will eventually become a separate call for each BC
      if BC.readProfiles then
         PROFILES.HDF.load(tiles_output, BC.ProfilesDir, Fluid, Fluid_copy, p_Output, p_Output_copy)
      end
      -- ... or use the config
      [PROFILES.mkInitializeProfilesField("xBCRight")](xPos[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("xBCLeft" )](xNeg[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("yBCRight")](yPos[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("yBCLeft" )](yNeg[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("zBCRight")](zPos[0], config, Mix);
      [PROFILES.mkInitializeProfilesField("zBCLeft" )](zNeg[0], config, Mix);

      ---------------------------------------------------------------------------
      -- Initialize boundary conditions
      ---------------------------------------------------------------------------
      [BCOND.InitBCs(BC, Fluid, tiles, Fluid_Zones, Grid, config, Mix)];

rescape if ELECTRIC_FIELD then remit rquote
      ---------------------------------------------------------------------------
      -- Initialize electric field solver
      ---------------------------------------------------------------------------
      [EFIELD.Init(EfieldData, tiles, Grid, config)];
end end end

      ---------------------------------------------------------------------------
      -- Initialize quantities for time stepping
      ---------------------------------------------------------------------------
      __demand(__index_launch)
      for c in tiles do
         VARS.UpdatePropertiesFromPrimitive(p_All[c], Mix)
      end

      __demand(__index_launch)
      for c in tiles do
         VARS.UpdateConservedFromPrimitive(p_Interior[c], Mix)
      end

      -- update values of conserved variables in ghost cells
      __demand(__index_launch)
      for c in tiles do
         VARS.UpdateConservedFromPrimitive(p_AllBCs[c], Mix)
      end

      UpdatePrimitivesAndBCsFromConserved(Fluid,
                                          tiles,
                                          Fluid_Zones,
                                          Fluid_Ghost,
                                          BC.BCParams,
                                          BC.RecycleAverage,
                                          BC.RecycleAverageFI,
                                          config,
                                          Mix,
                                          Grid.boundingBox,
                                          Integrator_simTime);

rescape if ELECTRIC_FIELD then remit rquote
      ---------------------------------------------------------------------------
      -- Update the electric field
      ---------------------------------------------------------------------------
      [EFIELD.UpdateElectricField(EfieldData, Fluid, Fluid_Zones, Fluid_Ghost,
                                  tiles, Fluid_bounds, Mix, config)];
end end end

      ---------------------------------------------------------------------------
      -- Precompute kernel profile
      ---------------------------------------------------------------------------
      if (config.Flow.laser.type == SCHEMA.LaserModel_GeometricKernel) then
         for c in Laser.LaserData.Laser_tiles do
            LASER.computeKernelProfile(Laser.LaserData.p_Laser[c], config);
         end
      end

      ---------------------------------------------------------------------------
      -- Initialize data for IO
      ---------------------------------------------------------------------------
      var [interior_volume] = 0.0
      __demand(__index_launch)
      for c in tiles do
         interior_volume += STAT.CalculateInteriorVolume(p_Interior[c])
      end

      ---------------------------------------------------------------------------
      -- Initialize probes
      ---------------------------------------------------------------------------
      [PROBES.InitProbes(Probes, config)];

      ---------------------------------------------------------------------------
      -- Initialize volume probes
      ---------------------------------------------------------------------------
      for p=0, config.IO.volumeProbes.length do
         var interval = config.IO.volumeProbes.values[p].interval
         if (interval.type == SCHEMA.IntervalType_DeltaTime) then
            config.IO.volumeProbes.values[p].interval.u.DeltaTime.counter =
               [regentlib.ceil(double)](Integrator_simTime/interval.u.DeltaTime.dt)
         end
      end

      ---------------------------------------------------------------------------
      -- Calculate exit condition
      ---------------------------------------------------------------------------
      Integrator_exitCond =
         (Integrator_timeStep >= config.Integrator.maxIter) or
         (Integrator_simTime  >= config.Integrator.maxTime)

   end end -- InitRegions

   -----------------------------------------------------------------------------
   -- Main time-step loop header
   -----------------------------------------------------------------------------

   function INSTANCE.MainLoopHeader(config) return rquote

      -- Unpack the partitions that we are going to need
      var {p_Interior} = Fluid_Zones;

      -- Determine time step size
      if config.Integrator.TimeStep.type == SCHEMA.TimeStepDefinitions_ConstantCFL then
         var Integrator_maxSpectralRadius = 0.0
         __demand(__index_launch)
         for c in tiles do
            Integrator_maxSpectralRadius max= CFL.CalculateMaxSpectralRadius(p_Interior[c], Mix)
         end
         Integrator_deltaTime = config.Integrator.TimeStep.u.ConstantCFL.cfl/Integrator_maxSpectralRadius
      elseif config.Integrator.TimeStep.type == SCHEMA.TimeStepDefinitions_ConstantDeltaTime then
         Integrator_deltaTime = config.Integrator.TimeStep.u.ConstantDeltaTime.DeltaTime
      else
         regentlib.assert(false, "Unknown time step definition")
      end

   end end -- MainLoopHeader

   -----------------------------------------------------------------------------
   -- Per-time-step I/O
   -----------------------------------------------------------------------------

   function INSTANCE.PerformConsoleIO(config) return rquote

      -- Unpack the partitions that we are going to need
      var {p_All, p_Interior} = Fluid_Zones;

      -- Write to console
      var AveragePressure = 0.0
      var AverageTemperature = 0.0
      var AverageKineticEnergy = 0.0
      var averageRhoU = 0.0
      var MaxSpeed = 0.0
      var MaxDensity = 0.0
      var MaxPressure = 0.0
      var MaxTemperature = 0.0
      var AverageTotalEnergy = 0.0

      -- Calculate averages
      __demand(__index_launch)
      for c in tiles do
         AveragePressure      += STAT.CalculateAveragePressure(p_Interior[c])
      end
      __demand(__index_launch)
      for c in tiles do
         AverageTemperature   += STAT.CalculateAverageTemperature(p_Interior[c])
      end
      __demand(__index_launch)
      for c in tiles do
         AverageKineticEnergy += STAT.CalculateAverageKineticEnergy(p_Interior[c])
      end

      -- Calculate maxes
      --for c in tiles do
         --MaxSpeed max= STAT.CalculateMaxSpeed(p_Interior[c])
      --end
      --for c in tiles do
         --MaxDensity max= STAT.CalculateMaxDensity(p_Interior[c])
      --end
      --for c in tiles do
         --MaxPressure max= STAT.CalculateMaxPressure(p_Interior[c])
      --end
      --for c in tiles do
         --MaxTemperature max= STAT.CalculateMaxTemperature(p_Interior[c])
      --end

      -- Add up energy
      __demand(__index_launch)
      for c in tiles do
         AverageTotalEnergy += STAT.CalculateAverageTotalEnergy(p_Interior[c])
      end

      -- Rescale channel flow forcing
      if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
         __demand(__index_launch)
         for c in tiles do
            averageRhoU += STAT.CalculateAverageRhoU(p_Interior[c], 0)
         end
      end

      AveragePressure      = (AveragePressure     /interior_volume)
      AverageTemperature   = (AverageTemperature  /interior_volume)
      AverageKineticEnergy = (AverageKineticEnergy/interior_volume)
      AverageTotalEnergy   = (AverageTotalEnergy  /interior_volume)
      IO.Console_Write([&int8](config.Mapping.outDir),
                       Integrator_timeStep,
                       Integrator_simTime,
                       startTime,
                       Integrator_deltaTime,
                       AveragePressure,
                       AverageTemperature,
                       AverageKineticEnergy,
                       --MaxSpeed,
                       --MaxDensity,
                       --MaxPressure,
                       --MaxTemperature,
                       AverageTotalEnergy)

      if config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL then
         averageRhoU = (averageRhoU / interior_volume)
         config.Flow.turbForcing.u.CHANNEL.Forcing *= config.Flow.turbForcing.u.CHANNEL.RhoUbulk/averageRhoU
      end

   end end -- PerformConsoleIO

   -----------------------------------------------------------------------------
   -- Data I/O that does not happen at every step
   -----------------------------------------------------------------------------

   function INSTANCE.PerformDataIO(config) return rquote

      -- Unpack the partitions that we are going to need
      var {p_All} = Fluid_Zones
      var {p_GradientGhosts}  = Fluid_Ghost
      var {p_Output, Vprobes, p_Vprobes} = Fluid_Output
      var {p_Output_copy  = p_Output,
           Vprobes_copy   = Vprobes,
           p_Vprobes_copy = p_Vprobes} = Fluid_Output_copy;

      -- Write probe files
      [PROBES.WriteProbes(Probes, Integrator_timeStep, Integrator_simTime, config)];

rescape if AVERAGES then remit rquote
      -- Add averages
      if (Integrator_timeStep % config.IO.AveragesSamplingInterval == 0) then
         [AVG.AddAverages(Averages, Fluid_bounds, Integrator_deltaTime, config, Mix)]
      end
end end end

      -- Dump restart files
      if config.IO.wrtRestart then
         if Integrator_exitCond or Integrator_timeStep % config.IO.restartEveryTimeSteps == 0 then
            var dirname = [&int8](C.malloc(256))
            C.snprintf(dirname, 256, '%s/fluid_iter%010d', config.Mapping.outDir, Integrator_timeStep)
            var _1 = IO.createDir(0, dirname)
            var _2 = HDF.dump(             _1, tiles_output, dirname, Fluid, Fluid_copy, p_Output, p_Output_copy)
            _2 = HDF.write.timeStep(       _2, dirname, Integrator_timeStep)
            _2 = HDF.write.simTime(        _2, dirname, Integrator_simTime)
            _2 = HDF.write.SpeciesNames(   _2, dirname, MIX.GetSpeciesNames(Mix))
            _2 = HDF.write.Versions(       _2, dirname, array(regentlib.string([VERSION.SolverVersion]), regentlib.string([VERSION.LegionVersion])))
            _2 = HDF.write.channelForcing( _2, dirname, config.Flow.turbForcing.u.CHANNEL.Forcing);

rescape if AVERAGES then remit rquote
            [AVG.WriteAverages(_1, Averages, tiles, dirname, IO, rexpr MIX.GetSpeciesNames(Mix) end, config)];
end end end

            C.free(dirname)
         end
      end

      -- Dump volumeProbes files...
      for p=0, config.IO.volumeProbes.length do
         var vProbe = config.IO.volumeProbes.values[p]
         if (vProbe.interval.type == SCHEMA.IntervalType_DeltaStep) then
            -- ... every n steps
            if (Integrator_timeStep % vProbe.interval.u.DeltaStep.delta) == 0 then
               var dirname = [&int8](C.malloc(256))
               C.snprintf(dirname, 256, '%s/%s/iter%010d', config.Mapping.outDir, vProbe.outDir, Integrator_timeStep)
               var _1 = IO.createDir(0, dirname)
               _1 = HDF_VPROBES.dump(_1, tiles_output, dirname, Vprobes[p], Vprobes_copy[p], p_Vprobes[p], p_Vprobes_copy[p])
               _1 = HDF_VPROBES.write.timeStep(    _1, dirname, Integrator_timeStep)
               _1 = HDF_VPROBES.write.simTime(     _1, dirname, Integrator_simTime)
               _1 = HDF_VPROBES.write.SpeciesNames(_1, dirname, MIX.GetSpeciesNames(Mix))
               _1 = HDF_VPROBES.write.Versions(    _1, dirname, array(regentlib.string([VERSION.SolverVersion]), regentlib.string([VERSION.LegionVersion])))
               C.free(dirname)
            end

         elseif (vProbe.interval.type == SCHEMA.IntervalType_DeltaTime) then
            -- ... every dt time
            var dt = vProbe.interval.u.DeltaTime.dt
            var counter = vProbe.interval.u.DeltaTime.counter
            if Integrator_simTime > (dt*counter) then
               config.IO.volumeProbes.values[p].interval.u.DeltaTime.counter += 1
               var dirname = [&int8](C.malloc(256))
               C.snprintf(dirname, 256, '%s/%s/iter%010d', config.Mapping.outDir, vProbe.outDir, Integrator_timeStep)
               var _1 = IO.createDir(0, dirname)
               _1 = HDF_VPROBES.dump(_1, tiles_output, dirname, Vprobes[p], Vprobes_copy[p], p_Vprobes[p], p_Vprobes_copy[p])
               _1 = HDF_VPROBES.write.timeStep(    _1, dirname, Integrator_timeStep)
               _1 = HDF_VPROBES.write.simTime(     _1, dirname, Integrator_simTime)
               _1 = HDF_VPROBES.write.SpeciesNames(_1, dirname, MIX.GetSpeciesNames(Mix))
               _1 = HDF_VPROBES.write.Versions(    _1, dirname, array(regentlib.string([VERSION.SolverVersion]), regentlib.string([VERSION.LegionVersion])))
               C.free(dirname)
            end
         end
      end

   end end -- PerformDataIO

   -----------------------------------------------------------------------------
   -- Main time-step loop body
   -----------------------------------------------------------------------------

   function INSTANCE.MainLoopBody(config) return rquote

      var Integrator_time_old = Integrator_simTime

      -- Unpack the partitions that we are going to need
      var {p_All, p_solved} = Fluid_Zones

      -- Update shock-sensors
      SENSOR.UpdateShockSensors(Fluid, tiles,
                                Fluid_Zones, Fluid_Ghost,
                                Mix, config)

      if config.Integrator.implicitChemistry then
         ---------------------------------------------------------------
         -- Update the conserved varialbes using the implicit solver ---
         ---------------------------------------------------------------
         -- Update the time derivatives
         UpdateDerivatives(Fluid,
                           tiles,
                           Fluid_Zones,
                           Fluid_Ghost,
                           BC.BCParams,
                           Laser.LaserData,
                           config, Mix, Grid.boundingBox,
                           array(0.0, 1.0), -- Start from new state
                           Integrator_deltaTime*0.5,
                           Integrator_simTime,
                           interior_volume,
                           false);

         -- Advance chemistry implicitely
         __demand(__index_launch)
         for c in tiles do
            CHEM.UpdateChemistry(p_solved[c], Integrator_deltaTime, Mix)
         end

         -- Update the fluxes in preparation to the RK algorithm
         UpdatePrimitivesAndBCsFromConserved(Fluid,
                                             tiles,
                                             Fluid_Zones,
                                             Fluid_Ghost,
                                             BC.BCParams,
                                             BC.RecycleAverage,
                                             BC.RecycleAverageFI,
                                             config,
                                             Mix,
                                             Grid.boundingBox,
                                             Integrator_simTime);

         -- The result of the local implicit solver is at 0.5*dt
         Integrator_simTime = Integrator_time_old + Integrator_deltaTime*0.5
      end

      -- Set iteration-specific fields that persist across RK sub-steps
      __demand(__index_launch)
      for c in tiles do
         RK.InitializeTemporaries(p_All[c])
      end

      -- RK sub-time-stepping loop
      rescape for STAGE = 1, 3 do remit rquote

         -- Update the time derivatives
         UpdateDerivatives(Fluid,
                           tiles,
                           Fluid_Zones,
                           Fluid_Ghost,
                           BC.BCParams,
                           Laser.LaserData,
                           config, Mix, Grid.boundingBox,
                           array(double([RK_C[STAGE][1]]), double([RK_C[STAGE][2]])),
                           Integrator_deltaTime*[RK_C[STAGE][3]],
                           Integrator_simTime,
                           interior_volume,
                           config.Integrator.implicitChemistry);

         -- Advance conserved variables
         __demand(__index_launch)
         for c in tiles do
            [RK.mkUpdateVars(STAGE)](p_All[c], Integrator_deltaTime, config.Integrator.implicitChemistry)
         end

         -- Update primitives and impose BCs
         UpdatePrimitivesAndBCsFromConserved(Fluid,
                                             tiles,
                                             Fluid_Zones,
                                             Fluid_Ghost,
                                             BC.BCParams,
                                             BC.RecycleAverage,
                                             BC.RecycleAverageFI,
                                             config,
                                             Mix,
                                             Grid.boundingBox,
                                             Integrator_simTime);

rescape if ELECTRIC_FIELD then remit rquote
         -- Update the electric field
         [EFIELD.UpdateElectricField(EfieldData, Fluid, Fluid_Zones, Fluid_Ghost,
                                     tiles, Fluid_bounds, Mix, config)];
end end end

rescape if DEBUG_OUTPUT then remit rquote
         CheckDebugOutput(Fluid, Fluid_copy,
                          tiles, tiles_output,
                          Fluid_Zones,
                          Fluid_Output,
                          Fluid_Output_copy,
                          config,
                          Mix,
                          Integrator_timeStep,
                          Integrator_simTime);
end end end

         -- Advance the time for the next sub-step
      rescape if STAGE == 3 then remit rquote
         Integrator_simTime = Integrator_time_old + Integrator_deltaTime
      end else remit rquote
         if config.Integrator.implicitChemistry then
            Integrator_simTime =   Integrator_time_old
                                 + Integrator_deltaTime*0.5*(1.0 + [RK_T[STAGE]])
         else
            Integrator_simTime = Integrator_time_old + [RK_T[STAGE]]*Integrator_deltaTime
         end
      end end end

      end end end-- RK sub-time-stepping

      -- Update time derivatives at boundary for BCs
      BCOND.UpdateNSCBCGhostCellTimeDerivatives(Fluid, tiles, Fluid_Zones, config, Integrator_deltaTime);

      Integrator_timeStep += 1

      ---------------------------------------------------------------------------
      -- Calculate exit condition
      ---------------------------------------------------------------------------
      Integrator_exitCond =
         (Integrator_timeStep >= config.Integrator.maxIter) or
         (Integrator_simTime  >= config.Integrator.maxTime)

   end end -- MainLoopBody

   -----------------------------------------------------------------------------
   -- Cleanup code
   -----------------------------------------------------------------------------

   function INSTANCE.Cleanup(config) return rquote

      -- Cleanup electric field solver symbols
rescape if ELECTRIC_FIELD then remit rquote
      [EFIELD.Cleanup(EfieldData, tiles, config)];
end end end

      -- Wait for everything above to finish
      __fence(__execution, __block)

      -- Report final time
      IO.Console_WriteFooter([&int8](config.Mapping.outDir), startTime)

   end end -- Cleanup

return INSTANCE end -- mkInstance

-------------------------------------------------------------------------------
-- TOP-LEVEL INTERFACE
-------------------------------------------------------------------------------

local SIM = mkInstance()

__demand(__inner, __replicable)
task workSingle(config : Config)
   [SIM.DeclSymbols(config)];
   [SIM.InitRegions(config)];
   [SIM.PerformConsoleIO(config)];
   [SIM.PerformDataIO(config)];
   while true do
rescape if (not DEBUG_OUTPUT) then remit rquote
      C.legion_runtime_begin_trace(__runtime(), __context(), config.Mapping.sampleId, false);
end end end
      [SIM.MainLoopHeader(config)];
      [SIM.MainLoopBody(config)];
      [SIM.PerformConsoleIO(config)];
rescape if (not DEBUG_OUTPUT) then remit rquote
      C.legion_runtime_end_trace(__runtime(), __context(), config.Mapping.sampleId)
end end end
      [SIM.PerformDataIO(config)];
      if SIM.Integrator_exitCond then
         break
      end
   end
   [SIM.Cleanup(config)];
end

__demand(__inline)
task initSingle(config : &Config, launched : int, outDirBase : &int8)
   config.Mapping.sampleId = launched
   C.snprintf([&int8](config.Mapping.outDir), 256, "%s/sample%d", outDirBase, launched)
   UTIL.createDir(config.Mapping.outDir)
end

__demand(__inner)
task main()
   var args = regentlib.c.legion_runtime_get_input_args()
   var outDirBase = '.'
   for i = 1, args.argc do
      if C.strcmp(args.argv[i], '-o') == 0 and i < args.argc-1 then
         outDirBase = args.argv[i+1]
      end
   end
   var startTime = C.legion_get_current_time_in_micros() / 1000;
   var launched = 0
   for i = 1, args.argc do
      if C.strcmp(args.argv[i], '-i') == 0 and i < args.argc-1 then
         var config : Config
         SCHEMA.parse_Config(&config, args.argv[i+1])
         initSingle(&config, launched, outDirBase)
         launched += 1
         workSingle(config)
      elseif C.strcmp(args.argv[i], '-lp') == 0 and i < args.argc-1 then
         var config : Config
         SCHEMA.parse_Config(&config, args.argv[i+1])
         initSingle(&config, launched, outDirBase)
         launched += 1
         workSingle(config)
      end
   end

   if launched < 1 then
      var stderr = C.fdopen(2, 'w')
      C.fprintf(stderr, "No testcases supplied.\n")
      C.fflush(stderr)
      C.exit(1)
   end

   -- Wait for everything above to finish
   __fence(__execution, __block)
   var endTime = C.legion_get_current_time_in_micros() / 1000;
   var stdout = C.fdopen(1, 'w')
   C.fprintf(stdout, 'Launched: %d Total time: %llu.%03llu seconds\n',
                     launched,
                     (endTime - startTime) / 1000,
                     (endTime - startTime) % 1000)
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "prometeo_main_"..os.getenv("EOS")..".o", "object", REGISTRAR.register_all)
