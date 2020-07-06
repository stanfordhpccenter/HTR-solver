import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("../../src/config_schema.h")
local Config = SCHEMA.Config

function mktestMixture(name)
   local testMixture

   local config = regentlib.newsymbol()

   local T = 750.0
   local P = 101325.0

   local MIX
   local ENames
   local EYi
   local EXi
   local EMixW
   local Erho
   local Ecp
   local Eh
   local Ehsp
   local EWi
   local Ee
   local Emu
   local Elam
   local Egamma
   local Esos
   local Edif
   local Eprod
   local Edpde
   local Edpdrho

   if (name == "ConstPropMix") then
      local R = 287.15
      local Pr = 0.71
      MIX = (require 'ConstPropMix')(SCHEMA)
      ENames = rexpr array("MIX") end
      EYi = rexpr array( 1.0 ) end
      EXi = rexpr array( 1.0 ) end
      EMixW = rexpr 8.3144598/[R] end
      Erho = rexpr [P]/([R]*[T]) end
      Ecp = rexpr 1.005025e+03 end
      Eh = rexpr [Ecp]*[T] end
      Ehsp = rexpr array([Ecp]*[T]) end
      EWi = rexpr array([EMixW]) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu = rexpr 5.000000e-03 end
      Elam = rexpr [Ecp]*[Emu]/[Pr] end
      Egamma = rexpr 1.4 end
      Esos = rexpr sqrt([Egamma]*[R]*[T]) end
      Edif = rexpr array( 0.0e0 ) end
      Eprod = rexpr array( 0.0e0 ) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( [R]*[T] ) end
   elseif (name == "AirMix") then
      MIX = (require 'AirMix')(SCHEMA)
      local R = 8.3144598
      ENames = rexpr  array(      "N2",         "O2",         "NO",          "N",          "O") end
      EYi = rexpr  array(          0.2,          0.2,          0.2,          0.2,          0.2) end
      EXi = rexpr  array( 1.524403e-01, 1.334541e-01, 1.423168e-01, 3.048806e-01, 2.669082e-01) end
      EMixW = rexpr 2.135186e-02 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 1.206133e+03 end
      Eh = rexpr 1.100436e+07 end
      Ehsp = rexpr array( 4.812927e+05, 4.425045e+05, 3.505703e+06, 3.441706e+07, 1.617526e+07) end
      EWi = rexpr  array( 2*14.0067e-3, 2*15.9994e-3, 30.00610e-03,  14.0067e-03,  15.9994e-03) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu = rexpr 3.723077e-05 end
      Elam = rexpr 5.927382e-02 end
      Egamma = rexpr 1.476782e+00 end
      Esos = rexpr 6.567317e+02 end
      Edif = rexpr array( 1.192983e-04, 1.174019e-04, 1.162986e-04, 1.820668e-04, 1.882347e-04) end
      Eprod = rexpr array( 3.413124e+07, 2.664566e+04,-3.578332e+07,-1.742776e+07, 1.905320e+07) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( 5.206698e+06, 5.184248e+06, 3.742883e+06,-1.064452e+07,-2.029053e+06) end
   else
      error "Unknown mixture"
   end

   __demand(__inline)
   task testMixture(config : Config)

      -- Init the mixture
      var Mix = MIX.InitMixture(config)

      -- check GetSpeciesNames
      for i = 0, Mix.nSpec do
         regentlib.assert(C.strcmp([ENames][i], MIX.GetSpeciesNames(Mix)[i]) == 0, ["mixTest: ERROR in GetSpeciesNames of " .. name])
      end

      -- check GetMolarWeightFromYi
      var MixW = MIX.GetMolarWeightFromYi([EYi], Mix)
      regentlib.assert(fabs((MixW/([EMixW])) - 1.0) < 1e-3, ["mixTest: ERROR in GetMolarWeightFromYi of " .. name])

      -- check GetMolarFractions
      var Xi = MIX.GetMolarFractions(MixW, [EYi], Mix)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((Xi[i]/([EXi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetMolarFractions of " .. name])
      end

      -- check GetMolarWeightFromXi
      var MixW2 = MIX.GetMolarWeightFromXi(Xi, Mix)
      regentlib.assert(fabs((MixW2/MixW) - 1.0) < 1e-3, ["mixTest: ERROR in GetMolarWeightFromXi of " .. name])

      -- check GetMassFractions
      var Yi = MIX.GetMassFractions(MixW, Xi, Mix)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((Yi[i]/([EYi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetMassFractions of " .. name])
      end

      -- check GetRho
      var rho = MIX.GetRho([P], [T], MixW, Mix)
      regentlib.assert(fabs((rho/([Erho])) - 1.0) < 1e-3, ["mixTest: ERROR in GetRho of " .. name])


      -- check GetRhoYiFromYi
      var rhoYi = MIX.GetRhoYiFromYi(rho, Yi)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((rhoYi[i]/([Erho]*[EYi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetRhoYiFromYi of " .. name])
      end

      -- check GetYi
      var Yi2 = MIX.GetYi(rho, rhoYi)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((Yi2[i]/([EYi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetYi of " .. name])
      end

      -- check GetHeatCapacity
      var cp = MIX.GetHeatCapacity([T], Yi, Mix)
      regentlib.assert(fabs((cp/([Ecp])) - 1.0) < 1e-3, ["mixTest: ERROR in GetHeatCapacity of " .. name])

      -- check GetEnthalpy
      var h = MIX.GetEnthalpy([T], Yi, Mix)
      regentlib.assert(fabs((h/([Eh])) - 1.0) < 1e-3, ["mixTest: ERROR in GetEnthalpy of " .. name])

      -- check GetSpeciesEnthalpy
      for i = 0, Mix.nSpec do
         var hsp = MIX.GetSpeciesEnthalpy(i, [T], Mix)
         regentlib.assert(fabs((hsp/([Ehsp][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetEnthalpy of " .. name])
      end

      -- check GetSpeciesMolarWeight
      for i = 0, Mix.nSpec do
         var W = MIX.GetSpeciesMolarWeight(i, Mix)
         regentlib.assert(fabs((W/([EWi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetSpeciesMolarWeight of " .. name])
      end

      -- check GetInternalEnergy
      var e = MIX.GetInternalEnergy([T], Yi, Mix)
      regentlib.assert(fabs((e/([Ee])) - 1.0) < 1e-3, ["mixTest: ERROR in GetInternalEnergy of " .. name])

      -- check GetTFromInternalEnergy
      var T1 = MIX.GetTFromInternalEnergy(e, [T]+100.0, Yi, Mix)
      regentlib.assert(fabs((T1/([T])) - 1.0) < 1e-3, ["mixTest: ERROR in GetTFromInternalEnergy of " .. name])

      -- check isValidInternalEnergy
      regentlib.assert(MIX.isValidInternalEnergy(e  , Yi, Mix) == true,  ["mixTest: ERROR in isValidInternalEnergy of " .. name])
      regentlib.assert(MIX.isValidInternalEnergy(0.0, Yi, Mix) == false, ["mixTest: ERROR in isValidInternalEnergy of " .. name])

      -- check GetTFromRhoAndP
      regentlib.assert(fabs((MIX.GetTFromRhoAndP(rho, MixW, P)/([T])) - 1.0) < 1e-3, ["mixTest: ERROR in GetTFromRhoAndP of " .. name])

      -- check GetPFromRhoAndT
      regentlib.assert(fabs((MIX.GetPFromRhoAndT(rho, MixW, T)/([P])) - 1.0) < 1e-3, ["mixTest: ERROR in GetPFromRhoAndT of " .. name])

      -- check GetViscosity
      regentlib.assert(fabs((MIX.GetViscosity(T, Xi, Mix)/([Emu])) - 1.0) < 1e-3, ["mixTest: ERROR in GetViscosity of " .. name])

      -- check GetHeatConductivity
      regentlib.assert(fabs((MIX.GetHeatConductivity(T, Xi, Mix)/([Elam])) - 1.0) < 1e-3, ["mixTest: ERROR in GetHeatConductivity of " .. name])

      -- check GetGamma
      var gamma = MIX.GetGamma(T, MixW, Yi, Mix)
      regentlib.assert(fabs((gamma/([Egamma])) - 1.0) < 1e-3, ["mixTest: ERROR in GetGamma of " .. name])

      -- check GetSpeedOfSound
      regentlib.assert(fabs((MIX.GetSpeedOfSound(T, gamma, MixW, Mix)/([Esos])) - 1.0) < 1e-3, ["mixTest: ERROR in GetSpeedOfSound of " .. name])

      -- check GetDiffusivity
      var dif = MIX.GetDiffusivity([P], [T], MixW, Xi, Mix)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((dif[i] - [Edif][i])/([Edif][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetDiffusivity of " .. name])
      end

      -- check GetMolarFractions
      var prod = MIX.GetProductionRates(rho, [P], [T], Yi, Mix)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((prod[i] - [Eprod][i])/([Eprod][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetProductionRates of " .. name])
      end

      -- check Getdpdrhoi
      regentlib.assert(fabs((MIX.Getdpde(rho, gamma, Mix)/([Edpde])) - 1.0) < 1e-3, ["mixTest: ERROR in Getdpde of " .. name])

      -- check Getdpdrhoi
      var dpdrhoi = MIX.Getdpdrhoi(gamma, [T], Yi, Mix)
      for i = 0, Mix.nSpec do
         regentlib.assert(fabs((dpdrhoi[i] - [Edpdrho][i])/([Edpdrho][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in Getdpdrhoi of " .. name])
      end

   end
   return testMixture
end

task main()
   var config : Config
   SCHEMA.parse_Config(&config, "test.json");

   -- Test ConstPropMix
   [mktestMixture("ConstPropMix")](config);

   -- Test AirMix
   [mktestMixture("AirMix")](config);

   C.printf("mixTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "mixTest.o", "object")
