import "regent"
-------------------------------------------------------------------------------
-- IMPORTS
-------------------------------------------------------------------------------

local C = regentlib.c
local fabs = regentlib.fabs(double)
local sqrt = regentlib.sqrt(double)
local SCHEMA = terralib.includec("config_schema.h")
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

local REGISTRAR = terralib.includec("prometeo_mixture.h")
local types_inc_flags = terralib.newlist({"-DEOS="..os.getenv("EOS")})
if ELECTRIC_FIELD then
   types_inc_flags:insert("-DELECTRIC_FIELD")
end
local TYPES = terralib.includec("prometeo_types.h", types_inc_flags)
local MIX = (require 'prometeo_mixture')(SCHEMA, TYPES)

function mktestMixture()
   local testMixture

   local config = regentlib.newsymbol()

   local R = 8.3144598
   local Na = 6.02214086e23
   local eCrg = 1.60217662e-19
   local eps_0 = 8.8541878128e-12

   local T = 750.0
   local P = 101325.0

   local name
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
   local Emob
   local Erhoq
   local ESi
   local Eprod
   local Edpde
   local Edpdrho

   -- Normalization quantities
   local LRef
   local PRef
   local TRef
   local MixWRef
   local rhoRef
   local eRef
   local uRef
   local CpRef
   local muRef
   local lamRef
   local DiRef
   local KiRef
   local rhoqRef
   local wiRef
   local dpdeRef
   local dpdrhoiRef
   local Eps0

   if (os.getenv("EOS") == "ConstPropMix") then
      local R = 287.15
      local Pr = 0.71
      name = "ConstPropMix"
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
      Erhoq = rexpr 0.0 end
      ESi = rexpr array( 0 ) end
      Eprod = rexpr array( 0.0e0 ) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( [R]*[T] ) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 1.0 end
      TRef = rexpr 1.0 end
      MixWRef = rexpr 1.0 end
      rhoRef = rexpr 1.0 end
      eRef = rexpr 1.0 end
      uRef = rexpr 1.0 end
      CpRef = rexpr 1.0 end
      muRef = rexpr 1.0 end
      lamRef = rexpr 1.0 end
      DiRef = rexpr 1.0 end
      rhoqRef = rexpr 1.0 end
      wiRef = rexpr 1.0 end
      dpdeRef = rexpr 1.0 end
      dpdrhoiRef = rexpr 1.0 end
      Eps0 = rexpr 1.0 end

   elseif (os.getenv("EOS") == "AirMix") then
      name = "AirMix"
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
      Elam = rexpr 6.736994e-02 end
      Egamma = rexpr 1.476782e+00 end
      Esos = rexpr 6.567317e+02 end
      Edif = rexpr array( 1.192983e-04, 1.174019e-04, 1.162986e-04, 1.820668e-04, 1.882347e-04) end
      Erhoq = rexpr 0.0 end
      ESi = rexpr  array(            0,            0,            0,            0,            0) end
      Eprod = rexpr array( 3.413124e+07, 2.664566e+04,-3.578332e+07,-1.742776e+07, 1.905320e+07) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( 5.206698e+06, 5.184248e+06, 3.742883e+06,-1.064452e+07,-2.029053e+06) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 2.889018800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == "CH41StMix") then
      name = "CH41StMix"
      ENames = rexpr  array(     "CH4",         "O2",        "CO2",        "H2O") end
      EYi = rexpr  array(         0.25,         0.25,         0.25,         0.25) end
      EXi = rexpr  array( 3.628018e-01, 1.818846e-01,   1.322462e-01,  3.230675e-01) end
      EMixW = rexpr 2.328035e-02 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 2.032494e+03 end
      Eh = rexpr -5.965912e+06 end
      Ehsp = rexpr array(-3.294307e+06, 4.425045e+05,-8.481094e+06,-1.253075e+07) end
      EWi = rexpr  array( 16.04206e-03, 2*15.9994e-3, 44.00950e-03, 18.01508e-03) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu = rexpr 3.108639e-05 end
      Elam = rexpr 8.767210e-02 end
      Egamma = rexpr 1.213177e+00 end
      Esos = rexpr 5.700526e+02 end
      Edif = rexpr array( 1.343289e-04, 1.007327e-04, 7.760974e-05, 1.350278e-04) end
      Erhoq = rexpr 0.0 end
      ESi = rexpr  array(            0,            0,            0,            0) end
      Eprod = rexpr array(-2.658098e+00,-1.060412e+01, 7.292178e+00, 5.970037e+00) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array(-1.550405e+05,-1.186804e+06, 6.509754e+05, 1.762304e+06) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == "CH4_30SpMix") then
      name = "CH4_30SpMix"
      local R = 8.3144598
      T = 1500
      ENames = rexpr  array(      "H2",          "H",          "O",         "O2",         "OH",
                                 "H2O",        "HO2",       "H2O2",          "C",         "CH",
                                 "CH2",     "CH2(S)",        "CH3",        "CH4",         "CO",
                                 "CO2",        "HCO",       "CH2O",      "CH2OH",       "CH3O",
                               "CH3OH",       "C2H2",       "C2H3",       "C2H4",       "C2H5",
                                "C2H6",       "HCCO",      "CH2CO",     "CH2CHO",         "N2") end

      EYi = rexpr  array(  5.44971e-04,        1e-60,        1e-60,  2.15391e-01,        1e-60,
                           5.62779e-03,  2.33245e-05,        1e-60,        1e-60,        1e-60,
                                 1e-60,        1e-60,        1e-60,  5.27455e-02,  1.49287e-03,
                           5.98395e-04,        1e-60,  6.19628e-05,        1e-60,        1e-60,
                           1.23090e-05,        1e-60,        1e-60,  1.22500e-05,        1e-60,
                           3.12909e-05,        1e-60,        1e-60,        1e-60,  7.23443e-01) end
      EXi = rexpr  array( 7.407618e-03, 2.718537e-59, 1.712470e-60, 1.844254e-01, 1.610990e-60,
                          8.559107e-03, 1.936145e-05, 8.054952e-61, 2.281174e-60, 2.104575e-60,
                          1.953355e-60, 1.953355e-60, 1.822409e-60, 9.008491e-02, 1.460273e-03,
                          3.725361e-04, 9.441918e-61, 5.654101e-05, 8.828651e-61, 8.828651e-61,
                          1.052537e-05, 1.052288e-60, 1.013074e-60, 1.196430e-05, 9.428057e-61,
                          2.851241e-05, 6.677896e-61, 6.517791e-61, 6.365184e-61, 7.075633e-01) end
      EMixW = rexpr 2.739850e-02 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 1.4689114e+03 end
      Eh = rexpr 1.2239974e+06 end
      Ehsp = rexpr array( 1.800606e+07, 2.410888e+08, 1.715512e+07, 1.268862e+06, 4.479985e+06,
                         -1.074720e+07, 2.033562e+06,-1.907952e+06, 6.175101e+07, 4.882339e+07,
                          3.165640e+07, 3.427167e+07, 1.419724e+07, 3.381412e+05,-2.559395e+06,
                         -7.539517e+06, 3.328419e+06,-1.340443e+06, 2.331136e+06, 3.246672e+06,
                         -3.224028e+06, 1.172948e+07, 1.434182e+07, 5.510683e+06, 8.117429e+06,
                          1.607184e+06, 6.263048e+06, 1.127859e+06, 3.089845e+06, 1.370973e+06) end
      EWi = rexpr  array( 2.015680e-03, 1.007840e-03, 1.599940e-02, 3.199880e-02, 1.700724e-02,
                          1.801508e-02, 3.300664e-02, 3.401448e-02, 1.201070e-02, 1.301854e-02,
                          1.402638e-02, 1.402638e-02, 1.503422e-02, 1.604206e-02, 2.801010e-02,
                          4.400950e-02, 2.901794e-02, 3.002578e-02, 3.103362e-02, 3.103362e-02,
                          3.204146e-02, 2.603708e-02, 2.704492e-02, 2.805276e-02, 2.906060e-02,
                          3.006844e-02, 4.102864e-02, 4.203648e-02, 4.304432e-02, 2.801340e-02) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu    = rexpr 5.4245594e-05 end
      Elam   = rexpr 1.1047398e-01 end
      Egamma = rexpr 1.2603842e+00 end
      Esos   = rexpr 7.5744434e+02 end
      Edif = rexpr array( 1.159092e-03,  1.919946e-03,  4.970050e-04,  3.137183e-04,  4.876624e-04,
                          4.248935e-04,  3.228673e-04,  3.206944e-04,  4.671199e-04,  5.315370e-04,
                          3.640265e-04,  3.640265e-04,  3.559047e-04,  3.673578e-04,  3.204063e-04,
                          2.611356e-04,  2.816701e-04,  2.793719e-04,  2.738767e-04,  2.738767e-04,
                          2.732409e-04,  2.720047e-04,  2.694003e-04,  2.693776e-04,  2.473667e-04,
                          2.453524e-04,  4.018125e-04,  2.370506e-04,  2.359528e-04,  3.184521e-04) end
      Erhoq = rexpr 0.0 end
      ESi = rexpr array( 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0) end
      Eprod = rexpr array(-2.619665e-04, 8.087741e-04, 7.719548e-05,-3.614274e-03, 2.444384e-03,
                          -7.054158e-04,-4.116491e-02, 4.263025e-02,-1.594798e-53,-3.279458e-53,
                           1.364118e-53, 1.622365e-05, 3.088778e-02,-2.795221e-02,-2.962706e-03,
                           4.701254e-03, 3.466765e-03,-3.706480e-03, 9.079769e-05, 9.819966e-08,
                          -8.255164e-05, 2.713259e-05, 4.723412e-08,-3.162481e-05, 1.908141e-05,
                          -4.687640e-03,-8.071670e-55, 1.316211e-54, 2.042675e-56, 0.000000e+00) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( 3.310111e+06,-4.697869e+07,-3.284257e+06, 3.610331e+05,-4.207291e+04,
                             3.871137e+06, 1.469175e+05, 1.159114e+06,-1.457004e+07,-1.130521e+07,
                            -6.921959e+06,-7.602934e+06,-2.450997e+06, 1.092006e+06, 1.427804e+06,
                             2.520530e+06,-1.247804e+05, 1.072735e+06, 9.971174e+04,-1.386793e+05,
                             1.530256e+06,-2.250267e+06,-2.952978e+06,-6.743695e+05,-1.372558e+06,
                             3.044764e+05,-1.047489e+06, 2.804470e+05,-2.391786e+05, 4.043327e+05) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == "CH4_43SpIonsMix") then
      name = "CH4_43SpIonsMix"
      local R = 8.3144598
      T = 1500
      ENames = rexpr  array(      "N2",         "H2",          "H",         "O2",          "O",
                                 "H2O",         "OH",       "H2O2",        "HO2",         "CO",
                                 "CO2",        "CH4",        "CH3",        "CH2",     "CH2(S)",
                                   "C",         "CH",      "CH3O2",      "CH3OH",       "CH3O",
                               "CH2OH",       "CH2O",        "HCO",       "C2H6",       "C2H5",
                                "C2H4",       "C2H3",       "C2H2",        "C2H",     "CH3CHO",
                               "CH3CO",     "CH2CHO",      "CH2CO",       "HCCO",       "C3H6",
                              "C3H5-A",       "CHO+",     "C2H3O+",      "CH5O+",       "H3O+",
                                 "OH-",        "O2-",          "E") end

      EYi = rexpr  array( 7.247482e-01, 2.873951e-04, 1.856406e-05, 7.026078e-03, 1.762206e-04,
                          1.201430e-01, 2.318436e-03, 8.314211e-08, 7.900423e-07, 9.834666e-03,
                          1.354466e-01, 1.795259e-17, 4.100397e-17, 5.867168e-18, 3.613851e-19,
                          1.006555e-17, 2.050666e-18, 8.261078e-23, 4.987982e-18, 3.400659e-18,
                          7.269760e-17, 1.677910e-11, 1.264227e-09, 7.743355e-33, 3.302831e-32,
                          3.662615e-27, 3.704750e-27, 3.582896e-22, 1.861185e-24, 1.773601e-25,
                          2.905161e-24, 8.931894e-25, 1.047018e-19, 3.546749e-20, 6.441795e-42,
                          7.670481e-41, 6.179356e-20, 7.138074e-31, 1.604438e-32, 1.388858e-16,
                          5.764229e-13, 1.908847e-14, 3.996186e-16) end
      EXi = rexpr  array( 7.088565e-01, 3.906026e-03, 5.046134e-04, 6.016396e-03, 3.017937e-04,
                          1.827302e-01, 3.735199e-03, 6.697452e-08, 6.558488e-07, 9.620392e-03,
                          8.432819e-02, 3.066110e-17, 7.472548e-17, 1.146067e-17, 7.059138e-19,
                          2.296174e-17, 4.315823e-18, 4.812606e-23, 4.265319e-18, 3.002421e-18,
                          6.418427e-17, 1.531149e-11, 1.193724e-09, 7.055731e-33, 3.113918e-32,
                          3.577196e-27, 3.753204e-27, 3.770274e-22, 2.037394e-24, 1.103130e-25,
                          1.849242e-24, 5.685480e-25, 6.824468e-20, 2.368568e-20, 4.194374e-42,
                          5.116965e-41, 5.834857e-20, 4.543705e-31, 1.330163e-32, 2.000496e-16,
                          9.286365e-13, 1.634508e-14, 1.995961e-11) end
      EMixW = rexpr 2.739973e-02 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 1.423258e+03 end
      Eh = rexpr -1.313672e+06 end
      Ehsp = rexpr array( 1.369611e+06, 1.799586e+07, 2.410452e+08, 1.269012e+06, 1.715846e+07,
								 -1.075009e+07, 4.372157e+06,-1.881761e+06, 2.045765e+06,-2.560159e+06,
								 -7.540188e+06, 3.376548e+05, 1.417744e+07, 3.155339e+07, 3.418885e+07,
								  6.174649e+07, 4.873545e+07, 2.496426e+06,-3.247268e+06, 3.583998e+06,
								  2.231564e+06,-1.362676e+06, 3.347733e+06, 1.594587e+06, 8.147544e+06,
								  5.502544e+06, 1.417021e+07, 1.172661e+07, 2.513873e+07,-9.915910e+05,
								  2.228818e+06, 2.860186e+06, 1.103988e+06, 6.320237e+06, 4.308916e+06,
								  7.828647e+06, 3.065696e+07, 1.778748e+07, 2.068148e+07, 3.471418e+07,
								 -6.267846e+06,-1.683196e+05, 4.553796e+10) end
      EWi = rexpr  array( 2.801400e-02, 2.016000e-03, 1.008000e-03, 3.199800e-02, 1.599900e-02,
								  1.801500e-02, 1.700700e-02, 3.401400e-02, 3.300600e-02, 2.801000e-02,
								  4.400900e-02, 1.604300e-02, 1.503500e-02, 1.402700e-02, 1.402700e-02,
								  1.201100e-02, 1.301900e-02, 4.703300e-02, 3.204200e-02, 3.103400e-02,
								  3.103400e-02, 3.002600e-02, 2.901800e-02, 3.007000e-02, 2.906200e-02,
								  2.805400e-02, 2.704600e-02, 2.603800e-02, 2.503000e-02, 4.405300e-02,
								  4.304500e-02, 4.304500e-02, 4.203700e-02, 4.102900e-02, 4.208100e-02,
								  4.107300e-02, 2.901745e-02, 4.304445e-02, 3.304945e-02, 1.902245e-02,
								  1.700755e-02, 3.199855e-02, 5.485799e-07) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu    = rexpr 5.519658e-05 end
      Elam   = rexpr 1.137426e-01 end
      Egamma = rexpr 1.270976e+00 end
      Esos   = rexpr 7.605937e+02 end
	if ELECTRIC_FIELD then
      Edif = rexpr array( 3.328026e-04,  1.179778e-03,  1.967375e-03,  3.289734e-04,  5.061709e-04,
								  4.414854e-04,  4.967148e-04,  3.247798e-04,  3.269961e-04,  3.243026e-04,
								  2.557088e-04,  3.581979e-04,  3.598466e-04,  3.680755e-04,  3.680755e-04,
								  4.743545e-04,  5.414407e-04,  2.534395e-04,  2.745194e-04,  2.753915e-04,
								  2.753915e-04,  2.806731e-04,  2.829947e-04,  2.441959e-04,  2.462103e-04,
								  3.140928e-04,  2.946066e-04,  2.974666e-04,  3.005252e-04,  2.357087e-04,
								  2.367718e-04,  2.367718e-04,  2.378805e-04,  4.086740e-04,  2.349290e-04,
								  2.306210e-04,  1.793319e-04,  1.510798e-04,  1.628573e-04,  2.014434e-04,
								  1.856256e-04,  1.648010e-04,  1.034067e-01) end
	else
      Edif = rexpr array( 3.328026e-04,  1.179778e-03,  1.967375e-03,  3.289734e-04,  5.061709e-04,
								  4.414854e-04,  4.967148e-04,  3.247798e-04,  3.269961e-04,  3.243026e-04,
								  2.557088e-04,  3.581979e-04,  3.598466e-04,  3.680755e-04,  3.680755e-04,
								  4.743545e-04,  5.414407e-04,  2.534395e-04,  2.745194e-04,  2.753915e-04,
								  2.753915e-04,  2.806731e-04,  2.829947e-04,  2.441959e-04,  2.462103e-04,
								  3.140928e-04,  2.946066e-04,  2.974666e-04,  3.005252e-04,  2.357087e-04,
								  2.367718e-04,  2.367718e-04,  2.378805e-04,  4.086740e-04,  2.349290e-04,
								  2.306210e-04,  4.282862e-04,  2.366133e-04,  2.502591e-04,  4.139949e-04,
								  4.952764e-04,  3.349915e-04,  1.140070e-05) end
	end
      Emob = rexpr array( 1.387374e-03,  1.168806e-03,  1.259921e-03,  1.558436e-03,  1.436065e-03,
                          1.274958e-03,  7.999900e-01) end
      Erhoq = rexpr -1.638665e-05 end
      ESi = rexpr array(           0,           0,           0,           0,           0,
								           0,           0,           0,           0,           0,
								           0,           0,           0,           0,           0,
								           0,           0,           0,           0,           0,
								           0,           0,           0,           0,           0,
								           0,           0,           0,           0,           0,
								           0,           0,           0,           0,           0,
								           0,           1,           1,           1,           1,
								          -1,          -1,          -1) end
      Eprod = rexpr array( 0.000000e+00,-6.386433e+00, 4.385057e+00, 1.821845e+01, 2.207037e+01,
                           9.281806e+01,-1.423135e+02, 7.461175e-01, 1.515058e-01,-1.806219e+01,
                           2.835990e+01,-1.811037e-13,-8.204296e-14, 3.815141e-12,-7.151828e-12,
                          -8.280009e-12,-3.509156e-12, 1.754485e-14, 1.599034e-12, 1.345945e-10,
                           8.504860e-10,-2.197352e-06, 1.267237e-02, 5.119513e-26, 2.917556e-24,
                          -1.039674e-22, 8.265544e-20,-4.668084e-18,-2.273024e-18, 5.741335e-19,
                           3.487064e-16, 1.107550e-17, 7.089766e-14,-1.558485e-14, 2.547250e-35,
                          -5.293791e-36,-2.894192e-11,-2.447041e-26, 2.378219e-24, 1.943284e-11,
                          -5.760147e-07, 1.563686e-06,-8.228268e-12) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array(-2.846256e+05, 2.506754e+06,-5.007171e+07,-3.278145e+05,-4.138104e+06,
									  3.313571e+06,-7.320436e+05, 4.966088e+05,-5.534242e+05, 7.803281e+05,
									  1.924069e+06, 4.172128e+05,-3.266794e+06,-7.899497e+06,-8.613643e+06,
									 -1.589143e+07,-1.246793e+07,-8.187682e+05, 8.953083e+05,-9.397323e+05,
									 -5.732552e+05, 4.178437e+05,-8.402259e+05,-3.842757e+05,-2.141686e+06,
									 -1.405357e+06,-3.733028e+06,-3.048183e+06,-6.658027e+06, 1.491979e+05,
									 -7.150295e+05,-8.861150e+05,-4.013977e+05,-1.805612e+06,-1.270250e+06,
									 -2.214768e+06,-8.240358e+06,-4.931046e+06,-5.603885e+06,-9.052749e+06,
									  2.151111e+06, 6.165926e+04, 1.655411e+10) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == "CH4_26SpIonsMix") then
      name = "CH4_26SpIonsMix"
      local R = 8.3144598
      T = 1500
      ENames = rexpr  array(      "N2",         "H2",          "H",         "O2",          "O",
                                 "H2O",         "OH",       "H2O2",        "HO2",         "CO",
                                 "CO2",        "CH4",        "CH3",        "CH2",     "CH2(S)",
                                   "C",         "CH",      "CH3OH",       "CH3O",      "CH2OH",
                                "CH2O",        "HCO",       "CHO+",       "H3O+",        "O2-",
                                   "E") end

      EYi = rexpr  array( 7.247482e-01, 2.873951e-04, 1.856406e-05, 7.026078e-03, 1.762206e-04,
                          1.201430e-01, 2.318436e-03, 8.314211e-08, 7.900423e-07, 9.834666e-03,
                          1.354466e-01, 1.795259e-17, 4.100397e-17, 5.867168e-18, 3.613851e-19,
                          1.006555e-17, 2.050666e-18, 8.261078e-23, 4.987982e-18, 3.400659e-18,
                          7.269760e-17, 1.677910e-11, 1.264227e-09, 7.743355e-33, 3.302831e-32,
                          3.662615e-27) end
      EXi = rexpr  array( 7.088564e-01, 3.906026e-03, 5.046133e-04, 6.016396e-03, 3.017936e-04,
                          1.827303e-01, 3.735198e-03, 6.697452e-08, 6.558487e-07, 9.620392e-03,
                          8.432820e-02, 3.066110e-17, 7.472548e-17, 1.146067e-17, 7.059138e-19,
                          2.296173e-17, 4.315822e-18, 7.064206e-23, 4.403859e-18, 3.002421e-18,
                          6.633899e-17, 1.584337e-11, 1.193746e-09, 1.115344e-32, 2.828149e-32,
                          1.829353e-22) end
      EMixW = rexpr 2.739973e-02 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 1.423258e+03 end
      Eh = rexpr -1.313672e+06 end
      Ehsp = rexpr array( 1.369611e+06, 1.799586e+07, 2.410452e+08, 1.269012e+06, 1.715846e+07,
                         -1.075009e+07, 4.372157e+06,-1.881761e+06, 2.045765e+06,-2.560159e+06,
                         -7.540188e+06, 3.376548e+05, 1.417744e+07, 3.155339e+07, 3.418885e+07,
                          6.174649e+07, 4.873545e+07,-3.247268e+06, 3.583998e+06, 2.231564e+06,
                         -1.362676e+06, 3.347733e+06, 3.065696e+07, 3.471418e+07,-1.683196e+05,
                          4.553796e+10) end
      EWi = rexpr  array( 2.801400e-02, 2.016000e-03, 1.008000e-03, 3.199800e-02, 1.599900e-02,
                          1.801500e-02, 1.700700e-02, 3.401400e-02, 3.300600e-02, 2.801000e-02,
                          4.400900e-02, 1.604300e-02, 1.503500e-02, 1.402700e-02, 1.402700e-02,
                          1.201100e-02, 1.301900e-02, 3.204200e-02, 3.103400e-02, 3.103400e-02,
                          3.002600e-02, 2.901800e-02, 2.901745e-02, 1.902245e-02, 3.199855e-02,
                          5.485799e-07) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu    = rexpr 5.519658e-05 end
      Elam   = rexpr 1.137426e-01 end
      Egamma = rexpr 1.270976e+00 end
      Esos   = rexpr 7.605937e+02 end
	if ELECTRIC_FIELD then
      Edif = rexpr array( 3.328026e-04,  1.179778e-03,  1.967375e-03,  3.289734e-04,  5.061709e-04,
                          4.414854e-04,  4.967148e-04,  3.247798e-04,  3.269961e-04,  3.243026e-04,
                          2.557088e-04,  3.581979e-04,  3.598466e-04,  3.680755e-04,  3.680755e-04,
                          4.743545e-04,  5.414407e-04,  2.745194e-04,  2.753915e-04,  2.753915e-04,
                          2.806731e-04,  2.829947e-04,  1.793319e-04,  2.014434e-04,  1.648010e-04,
                          2.585167e-02) end
	else
      Edif = rexpr array( 3.328026e-04,  1.179778e-03,  1.967375e-03,  3.289734e-04,  5.061709e-04,
                          4.414854e-04,  4.967148e-04,  3.247798e-04,  3.269961e-04,  3.243026e-04,
                          2.557088e-04,  3.581979e-04,  3.598466e-04,  3.680755e-04,  3.680755e-04,
                          4.743545e-04,  5.414407e-04,  2.745194e-04,  2.753915e-04,  2.753915e-04,
                          2.806731e-04,  2.829947e-04,  4.282862e-04,  4.139949e-04,  3.349915e-04,
                          1.140070e-05) end
	end
      Emob = rexpr array( 1.387374e-03,  1.558436e-03,  1.274958e-03,  1.999975e-01) end
      Erhoq = rexpr 9.357604e-04 end
      ESi = rexpr array(           0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           1,           1,          -1,
                                  -1) end
      Eprod = rexpr array( 0.000000e+00,-6.386798e+00, 4.385342e+00, 1.821774e+01, 2.207120e+01,
                           9.264324e+01,-1.423134e+02, 7.461175e-01, 1.514038e-01,-1.779316e+01,
                           2.835986e+01,-1.811061e-13,-4.232749e-12, 3.441144e-12,-7.153679e-12,
                          -8.280009e-12,-1.650227e-11, 1.997021e-12,-2.211041e-11,-4.023372e-13,
                           2.077287e-09, 1.533321e-02,-2.813402e-01, 1.844332e-01, 1.777643e-17,
                           2.684284e-18) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array(-2.846256e+05, 2.506754e+06,-5.007171e+07,-3.278144e+05,-4.138104e+06,
                             3.313571e+06,-7.320435e+05, 4.966088e+05,-5.534242e+05, 7.803282e+05,
                             1.924069e+06, 4.172128e+05,-3.266794e+06,-7.899497e+06,-8.613643e+06,
                            -1.589143e+07,-1.246793e+07, 8.953083e+05,-9.397323e+05,-5.732552e+05,
                             4.178437e+05,-8.402259e+05,-8.240358e+06,-9.052749e+06, 6.165927e+04,
                             1.655411e+10) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end
   elseif (os.getenv("EOS") == "FFCM1Mix") then
      name = "FFCM1Mix"
      local R = 8.3144598
      T = 1500
      ENames = rexpr  array(      "H2",          "H",          "O",         "O2",         "OH",
                                 "H2O",        "HO2",       "H2O2",         "CO",        "CO2",
                                   "C",         "CH",        "CH2",     "CH2(S)",        "CH3",
                                 "CH4",        "HCO",       "CH2O",      "CH2OH",       "CH3O",
                               "CH3OH",        "C2H",       "C2H2",       "C2H3",       "C2H4",
                                "C2H5",       "C2H6",       "HCCO",      "CH2CO",     "CH2CHO",
                              "CH3CHO",      "CH3CO",       "H2CC") end

      EYi = rexpr  array( 7.247482e-01, 2.873951e-04, 1.856406e-05, 7.026078e-03, 1.762206e-04,
                          1.201430e-01, 2.318436e-03, 8.314211e-08, 7.900423e-07, 9.834666e-03,
                          1.354466e-01, 1.795259e-17, 4.100397e-17, 5.867168e-18, 3.613851e-19,
                          1.006555e-17, 2.050666e-18, 8.261078e-23, 4.987982e-18, 3.400659e-18,
                          7.269760e-17, 1.677910e-11, 1.264227e-09, 7.743355e-33, 3.302831e-32,
                          3.662615e-27, 3.704750e-27, 3.582896e-22, 1.861185e-24, 1.773601e-25,
                          2.905161e-24, 8.931894e-25, 1.047018e-19) end
      EXi = rexpr  array( 9.504213e-01, 7.537692e-04, 3.067043e-06, 5.804033e-04, 2.738883e-05,
                          1.762840e-02, 1.856711e-04, 6.461120e-09, 7.455663e-08, 5.906957e-04,
                          2.980921e-02, 3.645150e-18, 7.727356e-18, 1.105690e-18, 6.353894e-20,
                          1.658548e-18, 1.868009e-19, 7.272656e-24, 4.248573e-19, 2.896552e-19,
                          5.997336e-18, 1.772032e-12, 1.283463e-10, 7.568226e-34, 3.112156e-33,
                          3.331481e-28, 3.256856e-28, 2.308329e-23, 1.170345e-25, 1.089157e-26,
                          1.743226e-25, 5.485021e-26, 1.062949e-20) end
      EMixW = rexpr 2.643325e-03 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 1.221597e+04 end
      Eh = rexpr  2.014642e+07 end
      Ehsp = rexpr array( 1.802545e+07, 2.410888e+08, 1.715523e+07, 1.267783e+06, 4.360915e+06,
                         -1.074573e+07, 2.025579e+06,-1.907797e+06,-2.560687e+06,-7.541346e+06,
                          6.174925e+07, 4.870763e+07, 3.155535e+07, 3.419105e+07, 1.417898e+07,
                          3.368382e+05, 3.363581e+06,-1.364939e+06, 2.238138e+06, 3.581165e+06,
                         -3.245722e+06, 2.514006e+07, 1.172721e+07, 1.417113e+07, 5.503073e+06,
                          8.148775e+06, 1.596368e+06, 6.320454e+06, 1.104006e+06, 2.864517e+06,
                         -9.908961e+05, 2.229502e+06, 1.860033e+07) end
      EWi = rexpr  array( 2.015680e-03, 1.007840e-03, 1.599940e-02, 3.199880e-02, 1.700724e-02,
                          1.801508e-02, 3.300664e-02, 3.401448e-02, 2.801010e-02, 4.400950e-02,
                          1.201070e-02, 1.301854e-02, 1.402638e-02, 1.402638e-02, 1.503422e-02,
                          1.604206e-02, 2.901794e-02, 3.002578e-02, 3.103362e-02, 3.103362e-02,
                          3.204146e-02, 2.502924e-02, 2.603708e-02, 2.704492e-02, 2.805276e-02,
                          2.906060e-02, 3.006844e-02, 4.102864e-02, 4.203648e-02, 4.304432e-02,
                          4.405216e-02, 4.304432e-02, 2.603708e-02) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu    = rexpr 2.923605e-05 end
      Elam   = rexpr 5.427472e-01 end
      Egamma = rexpr 1.346778e+00 end
      Esos   = rexpr 2.520782e+03 end
      Edif = rexpr array( 7.670583e-03,  3.157698e-03,  1.486029e-03,  1.095894e-03,  1.479138e-03,
                          1.215229e-03,  1.098170e-03,  1.098570e-03,  1.052285e-03,  9.206180e-04,
                          1.177592e-03,  1.511316e-03,  1.020942e-03,  1.020942e-03,  1.015194e-03,
                          1.027881e-03,  9.370184e-04,  9.354684e-04,  9.198897e-04,  9.198897e-04,
                          9.250296e-04,  8.728572e-04,  8.710027e-04,  8.692742e-04,  8.789467e-04,
                          8.059938e-04,  8.046712e-04,  1.469029e-03,  8.329268e-04,  8.322119e-04,
                          8.315274e-04,  8.322119e-04,  8.710027e-04) end
      Erhoq = rexpr 0.000000e+00 end
      ESi = rexpr array(           0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0,           0,           0,
                                   0,           0,           0) end
      Eprod = rexpr array(-5.471383e+02, 2.757591e+02, 9.735357e+02,-1.948217e+03,-1.523801e+02,
                           1.358159e+02,-2.586426e+01, 2.378074e+00, 1.786186e+03,-2.995362e-02,
                          -3.934252e+03, 3.434207e+03, 1.276208e-09,-8.082043e-11, 4.757539e-10,
                           3.108807e-15, 7.943873e-08, 2.971049e-12,-1.441050e-13,-2.311405e-12,
                           1.539214e-13,-1.021105e-04, 1.061436e-04, 4.230531e-08, 8.855515e-15,
                          -8.237810e-22, 4.559531e-23, 1.497008e-08, 3.966690e-09,-1.411255e-20,
                          -1.980992e-21,-8.465224e-18, 3.384368e-10) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( 7.432314e+06,-6.158820e+07, 4.509459e+05, 5.435450e+06, 4.825521e+06,
                             1.000892e+07, 5.156635e+06, 6.505565e+06, 6.837828e+06, 8.347008e+06,
                            -1.466464e+07,-1.025036e+07,-4.395027e+06,-5.309030e+06, 1.550439e+06,
                             6.280403e+06, 4.762595e+06, 6.382913e+06, 5.115277e+06, 4.649545e+06,
                             6.999936e+06,-2.696767e+06, 1.928540e+06, 1.057001e+06, 4.040581e+06,
                             3.102344e+06, 5.355203e+06, 3.567768e+06, 5.366903e+06, 4.747041e+06,
                             6.075086e+06, 4.967250e+06,-4.549095e+05) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == "BoivinMix") then
      name = "BoivinMix"
      local R = 8.3144598
      T = 1500
      ENames = rexpr  array(      "H2",          "H",         "O2",         "OH",          "O",
                                 "H2O",        "HO2",       "H2O2",         "N2") end

      EYi = rexpr  array( 1.000000e-01, 1.000000e-01, 1.000000e-01, 1.000000e-01, 1.000000e-01,
                          1.000000e-01, 1.000000e-01, 1.000000e-01, 2.000000e-01) end
      EXi = rexpr  array( 2.714636e-01, 5.429271e-01, 1.710327e-02, 3.217914e-02, 3.420655e-02,
                          3.037860e-02, 1.658094e-02, 1.608957e-02, 3.907122e-02) end
      EMixW = rexpr 5.472706e-03 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 4.976572e+03 end
      Eh = rexpr  2.739300e+07 end
      Ehsp = rexpr array( 1.799911e+07, 2.410449e+08, 1.269205e+06, 4.357695e+06, 1.715499e+07,
                         -1.074735e+07, 2.017681e+06,-1.907971e+06, 1.370865e+06) end
      EWi = rexpr  array( 2.016000e-03, 1.008000e-03, 3.199800e-02, 1.700700e-02, 1.599900e-02,
                          1.801500e-02, 3.300600e-02, 3.401400e-02, 2.801400e-02) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu    = rexpr 5.007584e-05 end
      Elam   = rexpr 5.577378e-01 end
      Egamma = rexpr 1.439418e+00 end
      Esos   = rexpr 1.811130e+03 end
      Edif = rexpr array( 2.922287e-03,  5.405200e-03,  9.675580e-04,  1.390704e-03,  1.408131e-03,
                          1.239050e-03,  9.631715e-04,  9.588918e-04,  9.004806e-04) end
      Erhoq = rexpr 0.000000e+00 end
      ESi = rexpr array(           0,           0,           0,           0,           0,
                                   0,           0,           0,           0) end
      Eprod = rexpr array(1.137198e+04,-4.437493e+04, 2.871371e+05, 1.217424e+06, 2.572854e+03,
                          7.212189e+04,-1.546185e+06,-6.713723e+01, 0.000000e+00) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( 1.203101e+07,-7.707478e+07, 1.103892e+07, 1.017630e+07, 4.619445e+06,
                             1.675467e+07, 1.069290e+07, 1.240178e+07, 1.107404e+07) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == "H2_UCSDMix") then
      name = "H2_UCSDMix"
      local R = 8.3144598
      T = 1500
      ENames = rexpr  array(      "H2",          "H",         "O2",         "OH",          "O",
                                 "H2O",        "HO2",       "H2O2",         "N2") end

      EYi = rexpr  array( 1.000000e-01, 1.000000e-01, 1.000000e-01, 1.000000e-01, 1.000000e-01,
                          1.000000e-01, 1.000000e-01, 1.000000e-01, 2.000000e-01) end
      EXi = rexpr  array( 2.714636e-01, 5.429271e-01, 1.710327e-02, 3.217914e-02, 3.420655e-02,
                          3.037860e-02, 1.658094e-02, 1.608957e-02, 3.907122e-02) end
      EMixW = rexpr 5.472706e-03 end
      Erho = rexpr [P]*[EMixW]/([R]*[T]) end
      Ecp = rexpr 4.974996e+03 end
      Eh = rexpr  2.739670e+07 end
      Ehsp = rexpr array( 1.800275e+07, 2.410445e+08, 1.268862e+06, 4.375240e+06, 1.715512e+07,
                         -1.074697e+07, 2.033551e+06,-1.907931e+06, 1.370909e+06) end
      EWi = rexpr  array( 2.016000e-03, 1.008000e-03, 3.199800e-02, 1.700700e-02, 1.599900e-02,
                          1.801500e-02, 3.300600e-02, 3.401400e-02, 2.801400e-02) end
      Ee = rexpr Eh - [P]/[Erho] end
      Emu    = rexpr 5.007584e-05 end
      Elam   = rexpr 5.576127e-01 end
      Egamma = rexpr 1.439618e+00 end
      Esos   = rexpr 1.811256e+03 end
      Edif = rexpr array( 2.922287e-03,  5.405200e-03,  9.675580e-04,  1.390704e-03,  1.408131e-03,
                          1.239050e-03,  9.631715e-04,  9.588918e-04,  9.004806e-04) end
      Erhoq = rexpr 0.000000e+00 end
      ESi = rexpr array(           0,           0,           0,           0,           0,
                                   0,           0,           0,           0) end
      Eprod = rexpr array(1.329866e+04,-5.754714e+04, 3.116157e+05, 1.252021e+06, 1.559913e+05,
                          2.967750e+05,-1.869816e+06,-1.023381e+05, 0.000000e+00) end
      Edpde = rexpr [Erho]*([Egamma] - 1.0) end
      Edpdrho = rexpr array( 1.203370e+07,-7.711375e+07, 1.104555e+07, 1.017452e+07, 4.622762e+06,
                             1.676345e+07, 1.069225e+07, 1.240887e+07, 1.108049e+07) end

      -- Normalization quantities
      LRef = rexpr 1.0 end
      PRef = rexpr 101325.0 end
      TRef = rexpr 300.0 end
      MixWRef = rexpr 3.1998800000000e-02 end
      rhoRef = rexpr [PRef]*[MixWRef]/([R]*[TRef]) end
      eRef = rexpr [PRef]/[rhoRef] end
      uRef = rexpr sqrt([PRef]/[rhoRef]) end
      CpRef = rexpr [R]/[MixWRef] end
      muRef = rexpr sqrt([PRef]*[rhoRef])*[LRef] end
      lamRef = rexpr sqrt([PRef]*[rhoRef])*[LRef]*[R]/[MixWRef] end
      DiRef = rexpr sqrt([PRef]/[rhoRef])*[LRef] end
      KiRef = rexpr sqrt([rhoRef]/[PRef])*[Na]*[eCrg]*[LRef]/[MixWRef] end
      rhoqRef = rexpr Na*eCrg*[rhoRef]/[MixWRef] end
      wiRef = rexpr sqrt([PRef]*[rhoRef])/[LRef] end
      dpdeRef = rexpr [rhoRef] end
      dpdrhoiRef = rexpr [eRef] end
      Eps0 = rexpr [PRef]*[MixWRef]*[MixWRef]/([rhoRef]*[rhoRef]*[Na]*[Na]*[eCrg]*[eCrg]*[LRef]*[LRef]) end

   elseif (os.getenv("EOS") == nil) then
      error ("You must define EOS enviromnment variable")
   end

   __demand(__inline)
   task testMixture(config : Config)

      var r = region(ispace(int3d, {1,1,1}), TYPES.Fluid_columns)
      var t = ispace(int3d, {1,1,1})
      var p_r = partition(equal, r, t)

      -- Init the mixture
      SCHEMA.parse_Config(&config, [name..".json"]);
      var Mix = MIX.InitMixture(r, t, p_r, config)

      -- check GetSpeciesNames
      for i = 0, MIX.nSpec do
         regentlib.assert(C.strcmp([ENames][i], MIX.GetSpeciesNames(Mix)[i]) == 0, ["mixTest: ERROR in GetSpeciesNames of " .. name])
      end

      -- check GetMolarWeightFromYi
      var MixW = MIX.GetMolarWeightFromYi([EYi], &Mix)
      regentlib.assert(fabs((MixW/([EMixW])) - 1.0) < 1e-3, ["mixTest: ERROR in GetMolarWeightFromYi of " .. name])

      -- check GetMolarFractions
      var Xi : double[MIX.nSpec]; MIX.GetMolarFractions(Xi, MixW, [EYi], &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((Xi[i]/([EXi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetMolarFractions of " .. name])
      end

      -- check GetMolarWeightFromXi
      var MixW2 = MIX.GetMolarWeightFromXi(Xi, &Mix)
      regentlib.assert(fabs((MixW2/MixW) - 1.0) < 1e-3, ["mixTest: ERROR in GetMolarWeightFromXi of " .. name])

      -- check GetMassFractions
      var Yi : double[MIX.nSpec]; MIX.GetMassFractions(Yi, MixW, Xi, &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((Yi[i]/([EYi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetMassFractions of " .. name])
      end

      -- check GetRho
      var rho = MIX.GetRho([P]/[PRef], [T]/[TRef], MixW, &Mix)*[rhoRef]
      regentlib.assert(fabs((rho/([Erho])) - 1.0) < 1e-3, ["mixTest: ERROR in GetRho of " .. name])

      -- check GetRhoYiFromYi
      var rhoYi : double[MIX.nSpec]; MIX.GetRhoYiFromYi(rhoYi, rho, Yi, &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((rhoYi[i]/([Erho]*[EYi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetRhoYiFromYi of " .. name])
      end

      -- check GetYi
      var Yi2 : double[MIX.nSpec]; MIX.GetYi(Yi2, rho, rhoYi, &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((Yi2[i]/([EYi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetYi of " .. name])
      end

      -- check GetHeatCapacity
      var cp = MIX.GetHeatCapacity([T]/[TRef], Yi, &Mix)*[CpRef]
      regentlib.assert(fabs((cp/([Ecp])) - 1.0) < 1e-3, ["mixTest: ERROR in GetHeatCapacity of " .. name])

      -- check GetEnthalpy
      var h = MIX.GetEnthalpy([T]/[TRef], Yi, &Mix)*[eRef]
      regentlib.assert(fabs((h/([Eh])) - 1.0) < 1e-3, ["mixTest: ERROR in GetEnthalpy of " .. name])

      -- check GetSpeciesEnthalpy
      for i = 0, MIX.nSpec do
         var hsp = MIX.GetSpeciesEnthalpy(i, [T]/[TRef], &Mix)*[eRef]
         regentlib.assert(fabs((hsp/([Ehsp][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetSpeciesEnthalpy of " .. name])
      end

      -- check GetSpeciesMolarWeight
      for i = 0, MIX.nSpec do
         var W = MIX.GetSpeciesMolarWeight(i, &Mix)
         regentlib.assert(fabs((W/([EWi][i])) - 1.0) < 1e-3, ["mixTest: ERROR in GetSpeciesMolarWeight of " .. name])
      end

      -- check GetInternalEnergy
      var e = MIX.GetInternalEnergy([T]/[TRef], Yi, &Mix)*[eRef]
      regentlib.assert(fabs((e/([Ee])) - 1.0) < 1e-3, ["mixTest: ERROR in GetInternalEnergy of " .. name])

      -- check GetTFromInternalEnergy
      var T1 = MIX.GetTFromInternalEnergy(e/[eRef], ([T]+100.0)/[TRef], Yi, &Mix)*[TRef]
      regentlib.assert(fabs((T1/([T])) - 1.0) < 1e-3, ["mixTest: ERROR in GetTFromInternalEnergy of " .. name])

      -- check isValidInternalEnergy
      regentlib.assert(MIX.isValidInternalEnergy(e/[eRef], Yi, &Mix) == true,  ["mixTest: ERROR in isValidInternalEnergy of " .. name])
      regentlib.assert(MIX.isValidInternalEnergy(-1.0e60,  Yi, &Mix) == false, ["mixTest: ERROR in isValidInternalEnergy of " .. name])

      -- check GetTFromRhoAndP
      regentlib.assert(fabs((MIX.GetTFromRhoAndP(rho/[rhoRef], MixW, P/[PRef], &Mix)*[TRef]/([T])) - 1.0) < 1e-3, ["mixTest: ERROR in GetTFromRhoAndP of " .. name])

      -- check GetPFromRhoAndT
      regentlib.assert(fabs((MIX.GetPFromRhoAndT(rho/[rhoRef], MixW, T/[TRef], &Mix)*[PRef]/([P])) - 1.0) < 1e-3, ["mixTest: ERROR in GetPFromRhoAndT of " .. name])

      -- check GetViscosity
      regentlib.assert(fabs((MIX.GetViscosity(T/[TRef], Xi, &Mix)*[muRef]/([Emu])) - 1.0) < 1e-3, ["mixTest: ERROR in GetViscosity of " .. name])

      -- check GetHeatConductivity
      regentlib.assert(fabs((MIX.GetHeatConductivity(T/[TRef], Xi, &Mix)*[lamRef]/([Elam])) - 1.0) < 1e-3, ["mixTest: ERROR in GetHeatConductivity of " .. name])

      -- check GetGamma
      var gamma = MIX.GetGamma(T/[TRef], MixW, Yi, &Mix)
      regentlib.assert(fabs((gamma/([Egamma])) - 1.0) < 1e-3, ["mixTest: ERROR in GetGamma of " .. name])

      -- check GetSpeedOfSound
      regentlib.assert(fabs((MIX.GetSpeedOfSound(T/[TRef], gamma, MixW, &Mix)*[uRef]/([Esos])) - 1.0) < 1e-3, ["mixTest: ERROR in GetSpeedOfSound of " .. name])

      -- check GetDiffusivity
      var dif : double[MIX.nSpec]; MIX.GetDiffusivity(dif, [P]/[PRef], [T]/[TRef], MixW, Xi, &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((dif[i]*[DiRef] - [Edif][i])/([Edif][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetDiffusivity of " .. name])
      end

   [(function() local __quotes = terralib.newlist()
   if (MIX.nIons > 0) then __quotes:insert(rquote
      -- check GetElectricMobility
      var mob : double[MIX.nIons]; MIX.GetElectricMobility(mob, [P]/[PRef], [T]/[TRef], Xi, &Mix)
      for i = 0, MIX.nIons do
         regentlib.assert(fabs((mob[i]*[KiRef] - [Emob][i])/([Emob][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetElectricMobility of " .. name])
      end
   end) end return __quotes end)()];

      -- check GetElectricChargeDensity
      regentlib.assert(fabs((MIX.GetElectricChargeDensity(rho/[rhoRef], MixW, Xi, &Mix)*[rhoqRef] - [Erhoq])/([Erhoq] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetElectricChargeDensity of " .. name])

      -- check GetSpeciesChargeNumnber
      for i = 0, MIX.nSpec do
         regentlib.assert(MIX.GetSpeciesChargeNumber(i, &Mix) == ESi[i], ["mixTest: ERROR in GetSpeciesChargeNumnber of " .. name])
      end

      -- check GetDielectricPermittivity
      regentlib.assert(fabs((MIX.GetDielectricPermittivity(&Mix)/[Eps0] - [eps_0])/([eps_0] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetDielectricPermittivity of " .. name])

      -- check GetProductionRates
      var prod : double[MIX.nSpec]; MIX.GetProductionRates(prod, rho/[rhoRef], [P]/[PRef], [T]/[TRef], Yi, &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((prod[i]*[wiRef] - [Eprod][i])/([Eprod][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in GetProductionRates of " .. name])
      end

      -- check Getdpdrhoi
      regentlib.assert(fabs((MIX.Getdpde(rho/[rhoRef], gamma, &Mix)*[dpdeRef]/([Edpde])) - 1.0) < 1e-3, ["mixTest: ERROR in Getdpde of " .. name])

      -- check Getdpdrhoi
      var dpdrhoi : double[MIX.nSpec]; MIX.Getdpdrhoi(dpdrhoi, gamma, [T]/[TRef], Yi, &Mix)
      for i = 0, MIX.nSpec do
         regentlib.assert(fabs((dpdrhoi[i]*[dpdrhoiRef] - [Edpdrho][i])/([Edpdrho][i] + 1e-60)) < 1e-3, ["mixTest: ERROR in Getdpdrhoi of " .. name])
      end

   end
   return testMixture
end

task main()
   var config : Config
   [mktestMixture()](config);
   C.printf("mixTest: TEST OK!\n")
end

-------------------------------------------------------------------------------
-- COMPILATION CALL
-------------------------------------------------------------------------------

regentlib.saveobj(main, "mixTest_"..os.getenv("EOS")..".o", "object", REGISTRAR.register_mixture_tasks)
