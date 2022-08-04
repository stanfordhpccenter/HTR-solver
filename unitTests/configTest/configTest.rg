import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local SCHEMA = terralib.includec("../../src/config_schema.h")
local Config = SCHEMA.Config

function testMixture(fl, name)
   return rquote
      regentlib.assert([fl].Species.length == 1, ["configTest: ERROR on "..name..".Species.length"])
      regentlib.assert(C.strcmp([fl].Species.values[0].Name, "Mix") == 0, ["configTest: ERROR on "..name..".Species.values[0].Name"])
      regentlib.assert(         [fl].Species.values[0].MolarFrac == 1.0,    ["configTest: ERROR on "..name..".Species.values[0].MolarFrac"])
   end
end

-- I do not know why but this inline reduces the compile time a lot!
__demand(__inline)
task check(config : Config)
   -- Mapping section
   var Mapping = config.Mapping
   regentlib.assert(Mapping.tiles[0] == 1,             "configTest: ERROR on config.Mapping.tiles[0]")
   regentlib.assert(Mapping.tiles[1] == 2,             "configTest: ERROR on config.Mapping.tiles[1]")
   regentlib.assert(Mapping.tiles[2] == 3,             "configTest: ERROR on config.Mapping.tiles[2]")
   regentlib.assert(Mapping.tilesPerRank[0] == 1,      "configTest: ERROR on config.Mapping.tilesPerRank[0]")
   regentlib.assert(Mapping.tilesPerRank[1] == 2,      "configTest: ERROR on config.Mapping.tilesPerRank[1]")
   regentlib.assert(Mapping.tilesPerRank[2] == 3,      "configTest: ERROR on config.Mapping.tilesPerRank[2]")
   regentlib.assert(Mapping.sampleId == -1,            "configTest: ERROR on config.Mapping.sampleId")
   regentlib.assert(C.strcmp(Mapping.outDir, "") == 0, "configTest: ERROR on config.Mapping.outDir")
   regentlib.assert(Mapping.wallTime == 10000,         "configTest: ERROR on config.Mapping.wallTime")
   -- Grid section
   var Grid = config.Grid
   regentlib.assert(Grid.xNum == 400,                             "configTest: ERROR on config.Grid.xNum")
   regentlib.assert(Grid.yNum == 100,                             "configTest: ERROR on config.Grid.yNum")
   regentlib.assert(Grid.zNum ==   1,                             "configTest: ERROR on config.Grid.zNum")
   var GridInput = Grid.GridInput
   regentlib.assert(GridInput.type == SCHEMA.GridInputStruct_Cartesian,
                                                                  "configTest: ERROR on config.Grid.GridInput.type")
   var Cart = GridInput.u.Cartesian
   regentlib.assert(Cart.origin[0] == 0.0,                        "configTest: ERROR on config.Grid.GridInput.origin[0]")
   regentlib.assert(Cart.origin[1] == 4.0,                        "configTest: ERROR on config.Grid.GridInput.origin[1]")
   regentlib.assert(Cart.origin[2] == 8.0,                        "configTest: ERROR on config.Grid.GridInput.origin[2]")
   regentlib.assert(Cart.width[0] == 1.0,                         "configTest: ERROR on config.Grid.GridInput.width[0]")
   regentlib.assert(Cart.width[1] == 0.5,                         "configTest: ERROR on config.Grid.GridInput.width[1]")
   regentlib.assert(Cart.width[2] == 0.1,                         "configTest: ERROR on config.Grid.GridInput.width[2]")
   regentlib.assert(Cart.xType.type == SCHEMA.GridTypes_TanhMinus,"configTest: ERROR on config.Grid.GridInput.xType.type")
   regentlib.assert(Cart.xType.u.TanhMinus.Stretching == 0.9,     "configTest: ERROR on config.Grid.GridInput.xType.u.TanhMinus.Stretching")
   regentlib.assert(Cart.yType.type == SCHEMA.GridTypes_TanhMinus,"configTest: ERROR on config.Grid.GridInput.yType.type")
   regentlib.assert(Cart.yType.u.TanhMinus.Stretching == 0.9,     "configTest: ERROR on config.Grid.GridInput.yType.u.TanhMinus.Stretching")
   regentlib.assert(Cart.zType.type == SCHEMA.GridTypes_Uniform,  "configTest: ERROR on config.Grid.GridInput.zType.type")
   -- Integrator section
   var Integrator = config.Integrator
   regentlib.assert(Integrator.startIter == 0,           "configTest: ERROR on config.Integrator.startIter")
   regentlib.assert(Integrator.startTime == 0.0,         "configTest: ERROR on config.Integrator.startTime")
   regentlib.assert(Integrator.resetTime == false,       "configTest: ERROR on config.Integrator.resetTime")
   regentlib.assert(Integrator.maxIter == 200000,        "configTest: ERROR on config.Integrator.maxIter")
   regentlib.assert(Integrator.maxTime == 20.0,          "configTest: ERROR on config.Integrator.maxTime")
   regentlib.assert(Integrator.TimeStep.type == SCHEMA.TimeStepDefinitions_ConstantDeltaTime,
                                                         "configTest: ERROR on config.Integrator.TimeStep.type")
   regentlib.assert(Integrator.TimeStep.u.ConstantDeltaTime.DeltaTime == 4.0e-3,
                                                         "configTest: ERROR on config.Integrator.TimeStep.u.ConstantDeltaTime.DeltaTime")
   regentlib.assert(Integrator.EulerScheme.type == SCHEMA.EulerSchemes_SkewSymmetric,
                                                         "configTest: ERROR on config.Integrator.EulerScheme.type")
   -- Flow section
   var Flow = config.Flow
   regentlib.assert(Flow.mixture.type == SCHEMA.MixtureModel_ConstPropMix,      "configTest: ERROR on config.Flow.mixture.type")
   regentlib.assert(Flow.mixture.u.ConstPropMix.gasConstant == 287.15,          "configTest: ERROR on config.Flow.mixture.gasConstant")
   regentlib.assert(Flow.mixture.u.ConstPropMix.gamma == 1.4,                   "configTest: ERROR on config.Flow.mixture.gamma")
   regentlib.assert(Flow.mixture.u.ConstPropMix.prandtl == 0.71,                "configTest: ERROR on config.Flow.mixture.prandtl")
   regentlib.assert(Flow.mixture.u.ConstPropMix.viscosityModel.type == SCHEMA.ViscosityModel_Constant,
                                                                                "configTest: ERROR on config.Grid.mixture.viscosityModel")
   regentlib.assert(Flow.mixture.u.ConstPropMix.viscosityModel.u.Constant.Visc == 5.0e-3,
                                                                                "configTest: ERROR on config.Flow.constantVisc")
   regentlib.assert(Flow.initCase.type == SCHEMA.FlowInitCase_Uniform,          "configTest: ERROR on config.Flow.initCase.type")
   regentlib.assert(Flow.initCase.u.Uniform.pressure == 1.01325e5,              "configTest: ERROR on config.Flow.initCase.u.Uniform.pressure")
   regentlib.assert(Flow.initCase.u.Uniform.temperature == 300.0,               "configTest: ERROR on config.Flow.initCase.u.Uniform.temperature")
   regentlib.assert(Flow.initCase.u.Uniform.velocity[0] == 10.0,                "configTest: ERROR on config.Flow.initCase.u.Uniform.velocity[0]")
   regentlib.assert(Flow.initCase.u.Uniform.velocity[1] == 20.0,                "configTest: ERROR on config.Flow.initCase.u.Uniform.velocity[1]")
   regentlib.assert(Flow.initCase.u.Uniform.velocity[2] == 30.0,                "configTest: ERROR on config.Flow.initCase.u.Uniform.velocity[2]");
   [testMixture( rexpr Flow.initCase.u.Uniform.molarFracs end, "config.Flow.initCase.u.Uniform.molarFracs")];
   regentlib.assert(Flow.resetMixture == false,                                 "configTest: ERROR on config.Flow.resetMixture");
   [testMixture( rexpr Flow.initMixture end, "config.Flow.initMixture")];
   regentlib.assert(Flow.bodyForce[0] == 1.0,                                   "configTest: ERROR on config.Flow.bodyForce[0]")
   regentlib.assert(Flow.bodyForce[1] == 2.0,                                   "configTest: ERROR on config.Flow.bodyForce[1]")
   regentlib.assert(Flow.bodyForce[2] == 3.0,                                   "configTest: ERROR on config.Flow.bodyForce[2]")
   regentlib.assert(Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL,   "configTest: ERROR on config.Flow.turbForcing.type")
   regentlib.assert(Flow.turbForcing.u.CHANNEL.Forcing == 1000.0,               "configTest: ERROR on config.Flow.turbForcing.u.CHANNEL.Forcing")
   regentlib.assert(Flow.turbForcing.u.CHANNEL.RhoUbulk == 20.0,                "configTest: ERROR on config.Flow.turbForcing.u.CHANNEL.RhoUbulk")
   -- IO section
   var IO = config.IO
   regentlib.assert(IO.wrtRestart == true,               "configTest: ERROR on config.IO.wrtRestart")
   regentlib.assert(IO.restartEveryTimeSteps == 10000,   "configTest: ERROR on config.IO.restartEveryTimeSteps")
   regentlib.assert(IO.probesSamplingInterval == 1,      "configTest: ERROR on config.IO.probesSamplingInterval")
   regentlib.assert(IO.AveragesSamplingInterval == 10,   "configTest: ERROR on config.IO.AveragesSamplingInterval")
   -- Efield section
   var Efield = config.Efield
   regentlib.assert(Efield.type == SCHEMA.EFieldStruct_Off, "configTest: ERROR on config.Efield.type")
   return 1
end

task main()
   var config : Config
   SCHEMA.parse_Config(&config, "test.json")
   var _ = check(config)
   C.printf("configTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "configTest.o", "object")
