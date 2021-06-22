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
   regentlib.assert(config.Mapping.tiles[0] == 1,             "configTest: ERROR on config.Mapping.tiles[0]")
   regentlib.assert(config.Mapping.tiles[1] == 2,             "configTest: ERROR on config.Mapping.tiles[1]")
   regentlib.assert(config.Mapping.tiles[2] == 3,             "configTest: ERROR on config.Mapping.tiles[2]")
   regentlib.assert(config.Mapping.tilesPerRank[0] == 1,      "configTest: ERROR on config.Mapping.tilesPerRank[0]")
   regentlib.assert(config.Mapping.tilesPerRank[1] == 2,      "configTest: ERROR on config.Mapping.tilesPerRank[1]")
   regentlib.assert(config.Mapping.tilesPerRank[2] == 3,      "configTest: ERROR on config.Mapping.tilesPerRank[2]")
   regentlib.assert(config.Mapping.sampleId == -1,            "configTest: ERROR on config.Mapping.sampleId")
   regentlib.assert(C.strcmp(config.Mapping.outDir, "") == 0, "configTest: ERROR on config.Mapping.outDir")
   regentlib.assert(config.Mapping.wallTime == 10000,         "configTest: ERROR on config.Mapping.wallTime")
   -- Grid section
   regentlib.assert(config.Grid.xNum == 400,                        "configTest: ERROR on config.Grid.xNum")
   regentlib.assert(config.Grid.yNum == 100,                        "configTest: ERROR on config.Grid.yNum")
   regentlib.assert(config.Grid.zNum ==   1,                        "configTest: ERROR on config.Grid.zNum")
   regentlib.assert(config.Grid.origin[0] == 0.0,                   "configTest: ERROR on config.Grid.origin[0]")
   regentlib.assert(config.Grid.origin[1] == 4.0,                   "configTest: ERROR on config.Grid.origin[1]")
   regentlib.assert(config.Grid.origin[2] == 8.0,                   "configTest: ERROR on config.Grid.origin[2]")
   regentlib.assert(config.Grid.xWidth == 1.0,                      "configTest: ERROR on config.Grid.xWidth")
   regentlib.assert(config.Grid.yWidth == 0.5,                      "configTest: ERROR on config.Grid.yWidth")
   regentlib.assert(config.Grid.zWidth == 0.1,                      "configTest: ERROR on config.Grid.zWidth")
   regentlib.assert(config.Grid.xType == SCHEMA.GridType_TanhMinus, "configTest: ERROR on config.Mapping.xType")
   regentlib.assert(config.Grid.yType == SCHEMA.GridType_TanhMinus, "configTest: ERROR on config.Mapping.yType")
   regentlib.assert(config.Grid.zType == SCHEMA.GridType_Uniform,   "configTest: ERROR on config.Mapping.zType")
   regentlib.assert(config.Grid.xStretching == 0.9,                 "configTest: ERROR on config.Grid.xStretching")
   regentlib.assert(config.Grid.yStretching == 0.9,                 "configTest: ERROR on config.Grid.yStretching")
   regentlib.assert(config.Grid.zStretching == 1.0,                 "configTest: ERROR on config.Grid.zStretching")
   -- Integrator section
   regentlib.assert(config.Integrator.startIter == 0,           "configTest: ERROR on config.Integrator.startIter")
   regentlib.assert(config.Integrator.startTime == 0.0,         "configTest: ERROR on config.Integrator.startTime")
   regentlib.assert(config.Integrator.resetTime == false,       "configTest: ERROR on config.Integrator.resetTime")
   regentlib.assert(config.Integrator.maxIter == 200000,        "configTest: ERROR on config.Integrator.maxIter")
   regentlib.assert(config.Integrator.maxTime == 20.0,          "configTest: ERROR on config.Integrator.maxTime")
   regentlib.assert(config.Integrator.cfl == 0.9,               "configTest: ERROR on config.Integrator.cfl")
   regentlib.assert(config.Integrator.fixedDeltaTime == 4.0e-3, "configTest: ERROR on config.Integrator.fixedDeltaTime")
   regentlib.assert(config.Integrator.EulerScheme.type == SCHEMA.EulerSchemes_SkewSymmetric,
                                                                "configTest: ERROR on config.Integrator.EulerScheme.type")
   -- Flow section
   regentlib.assert(config.Flow.mixture.type == SCHEMA.MixtureModel_ConstPropMix,      "configTest: ERROR on config.Flow.mixture.type")
   regentlib.assert(config.Flow.mixture.u.ConstPropMix.gasConstant == 287.15,          "configTest: ERROR on config.Flow.mixture.gasConstant")
   regentlib.assert(config.Flow.mixture.u.ConstPropMix.gamma == 1.4,                   "configTest: ERROR on config.Flow.mixture.gamma")
   regentlib.assert(config.Flow.mixture.u.ConstPropMix.prandtl == 0.71,                "configTest: ERROR on config.Flow.mixture.prandtl")
   regentlib.assert(config.Flow.mixture.u.ConstPropMix.viscosityModel.type == SCHEMA.ViscosityModel_Constant,
                                                                                       "configTest: ERROR on config.Grid.mixture.viscosityModel")
   regentlib.assert(config.Flow.mixture.u.ConstPropMix.viscosityModel.u.Constant.Visc == 5.0e-3,
                                                                                       "configTest: ERROR on config.Flow.constantVisc")
   regentlib.assert(config.Flow.initCase.type == SCHEMA.FlowInitCase_Uniform,          "configTest: ERROR on config.Flow.initCase.type")
   regentlib.assert(config.Flow.initCase.u.Uniform.pressure == 1.01325e5,              "configTest: ERROR on config.Flow.initCase.u.Uniform.pressure")
   regentlib.assert(config.Flow.initCase.u.Uniform.temperature == 300.0,               "configTest: ERROR on config.Flow.initCase.u.Uniform.temperature")
   regentlib.assert(config.Flow.initCase.u.Uniform.velocity[0] == 10.0,                "configTest: ERROR on config.Flow.initCase.u.Uniform.velocity[0]")
   regentlib.assert(config.Flow.initCase.u.Uniform.velocity[1] == 20.0,                "configTest: ERROR on config.Flow.initCase.u.Uniform.velocity[1]")
   regentlib.assert(config.Flow.initCase.u.Uniform.velocity[2] == 30.0,                "configTest: ERROR on config.Flow.initCase.u.Uniform.velocity[2]");
   [testMixture( rexpr config.Flow.initCase.u.Uniform.molarFracs end, "config.Flow.initCase.u.Uniform.molarFracs")];
   regentlib.assert(config.Flow.resetMixture == false,                                 "configTest: ERROR on config.Flow.resetMixture");
   [testMixture( rexpr config.Flow.initMixture end, "config.Flow.initMixture")];
   regentlib.assert(config.Flow.bodyForce[0] == 1.0,                                   "configTest: ERROR on config.Flow.bodyForce[0]")
   regentlib.assert(config.Flow.bodyForce[1] == 2.0,                                   "configTest: ERROR on config.Flow.bodyForce[1]")
   regentlib.assert(config.Flow.bodyForce[2] == 3.0,                                   "configTest: ERROR on config.Flow.bodyForce[2]")
   regentlib.assert(config.Flow.turbForcing.type == SCHEMA.TurbForcingModel_CHANNEL,   "configTest: ERROR on config.Flow.turbForcing.type")
   regentlib.assert(config.Flow.turbForcing.u.CHANNEL.Forcing == 1000.0,               "configTest: ERROR on config.Flow.turbForcing.u.CHANNEL.Forcing")
   regentlib.assert(config.Flow.turbForcing.u.CHANNEL.RhoUbulk == 20.0,                "configTest: ERROR on config.Flow.turbForcing.u.CHANNEL.RhoUbulk")
   -- IO section
   regentlib.assert(config.IO.wrtRestart == true,               "configTest: ERROR on config.IO.wrtRestart")
   regentlib.assert(config.IO.restartEveryTimeSteps == 10000,   "configTest: ERROR on config.IO.restartEveryTimeSteps")
   regentlib.assert(config.IO.probesSamplingInterval == 1,      "configTest: ERROR on config.IO.probesSamplingInterval")
   regentlib.assert(config.IO.AveragesSamplingInterval == 10,   "configTest: ERROR on config.IO.AveragesSamplingInterval")
   -- Efield section
   regentlib.assert(config.Efield.type == SCHEMA.EFieldStruct_Off, "configTest: ERROR on config.Efield.type")
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
