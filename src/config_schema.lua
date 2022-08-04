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

local Exports = {}

-- Helper definitions
Exports.IntervalType = Union{
   {"DeltaStep", {
      {"delta", int},
   }},
   {"DeltaTime", {
      {"dt", double},
      {"counter", Int{default = 0}},
   }},
}

Exports.Volume = {
   {"fromCell", Array(3,int)},
   {"uptoCell", Array(3,int)},
}

Exports.Window = {
   {"fromCell", Array(2,int)},
   {"uptoCell", Array(2,int)},
}

Exports.VolumeProbe = {
   {"outDir", String(256)},
   {"interval", Exports.IntervalType},
   {"volume", Exports.Volume},
}

Exports.Species = {
   {"Name", String(10)},
   {"MolarFrac", double},
}

Exports.Mixture = {
   {"Species", UpTo(20, Exports.Species)},
}

-- Unions & enumeration constants
Exports.MixtureProfile = Union{
   {"Constant", {
      {"Mixture", Exports.Mixture},
   }},
   {"File", {
      {"FileDir", String(256)},
   }},
   {"Incoming", {}},
}
Exports.TempProfile = Union{
   {"Constant", {
      {"temperature", double},
   }},
   {"File", {
      {"FileDir", String(256)},
   }},
   {"Incoming", {}},
}
Exports.InflowProfile = Union{
   {"Constant", {
      {"velocity", Array(3,double)},
   }},
   {"File", {
      {"FileDir", String(256)},
   }},
   {"Incoming", {
      {"addedVelocity", double},
   }},
}

Exports.FlowBC = Union{
   {"Dirichlet", {
      {"VelocityProfile", Exports.InflowProfile},
      {"TemperatureProfile", Exports.TempProfile},
      {"MixtureProfile", Exports.MixtureProfile},
      {"P", double},
   }},
   {"Periodic", {}},
   {"Symmetry", {}},
   {"AdiabaticWall", {}},
   {"IsothermalWall", {
      {"TemperatureProfile", Exports.TempProfile},
   }},
   {"NSCBC_Inflow", {
      {"VelocityProfile", Exports.InflowProfile},
      {"TemperatureProfile", Exports.TempProfile},
      {"MixtureProfile", Exports.MixtureProfile},
      {"P", double},
   }},
   {"NSCBC_Outflow", {
      {"P", double},
      {"LengthScale", Double{default = 0.0}}
   }},
   {"NSCBC_FarField", {
      {"VelocityProfile", Exports.InflowProfile},
      {"TemperatureProfile", Exports.TempProfile},
      {"MixtureProfile", Exports.MixtureProfile},
      {"P", double},
      {"LengthScale", Double{default = 0.0}}
   }},
   {"SuctionAndBlowingWall", {
      {"TemperatureProfile", Exports.TempProfile},
      {"Xmin" , double},
      {"Xmax" , double},
      {"X0"   , double},
      {"sigma", double},
      {"Zw"   , double},
      {"A"    , UpTo(20, double)},
      {"omega", UpTo(20, double)},
      {"beta" , UpTo(20, double)},
   }},
   {"IncomingShock", {
      {"iShock", int},
      {"beta", double},
      {"Mixture", Exports.Mixture},
      {"velocity0", Array(3,double)},
      {"pressure0", double},
      {"temperature0", double},
      {"LengthScale", Double{default = 0.0}},
   }},
   {"RecycleRescaling", {
      {"iRecycle", int},
      {"VelocityProfile", Exports.InflowProfile},
      {"TemperatureProfile", Exports.TempProfile},
      {"MixtureProfile", Exports.MixtureProfile},
      {"P", double},
   }},
}

Exports.BufferZoneBC = Union{
   {"Off", {}},
   {"Basic", {
      {"xBufferLength", double},
      {"yBufferLength", double},
      {"zBufferLength", double},
      {"power", double},
      {"maxAmplitude", double},
      {"pTarget", double},
      {"TTarget", double},
      {"XiTarget", Exports.Mixture}
   }}
}

Exports.ViscosityModel = Union{
   {"Constant", {
      {"Visc", double},
   }},
   {"PowerLaw", {
      {"ViscRef", double},
      {"TempRef", double},
   }},
   {"Sutherland", {
      {"ViscRef", double},
      {"TempRef", double},
      {"SRef", double},
   }},
}

Exports.TurbForcingModel = Union{
   {"OFF", {}},
   {"CHANNEL", {
      {"RhoUbulk", double},
      {"Forcing", double},
   }},
   {"HIT", {
      {"Gain", double},
      {"t_o", double},
      {"K_o", double},
   }},
   options = { default = { type  = "OFF"}, },
}

Exports.LaserModel = Union{
   {"Off", {}},
   {"Algebraic", {
      {"volume", Exports.Volume},
      {"Dimension", int},
      {"Amplitude", double},
      {"Center", Array(3,double)},
      {"Radius", double},
      {"Delay", double},
      {"Duration", double},
   }},
   {"GeometricKernel", {
      {"volume", Exports.Volume},
      {"dimensions", int},
      {"peakEdotPerMass", double},
      {"focalLocation", Array(3,double)},
      {"pulseTime", double},
      {"pulseFWHM", double},
      {"axialLength", double},
      {"nearRadius", double},
      {"farRadius", double},
      {"beamDirection", Array(3,double)},
   }},
   options = { default = { type  = "Off"}, },
}

Exports.MixtureModel = Union{
   {"ConstPropMix", {
      {"gasConstant", double},
      {"gamma", double},
      {"prandtl", double},
      {"viscosityModel", Exports.ViscosityModel},
   }},
   {"IsentropicMix", {
      {"gasConstant", double},
      {"gamma", double},
   }},
   {"AirMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
   }},
   {"CH41StMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
   }},
   {"CH4_30SpMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
   }},
   {"CH4_43SpIonsMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
      -- Relative dielectric permittivity
--      {"esp_r", double},
   }},
   {"CH4_26SpIonsMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
      -- Relative dielectric permittivity
--      {"esp_r", double},
   }},
   {"FFCM1Mix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
   }},
   {"BoivinMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
   }},
   {"H2_UCSDMix", {
      {"LRef", double},
      {"PRef", double},
      {"TRef", double},
      {"XiRef", Exports.Mixture},
   }},
}

Exports.FlowInitCase = Union{
   {"Uniform", {
      {"pressure", double},
      {"temperature", double},
      {"velocity", Array(3,double)},
      {"molarFracs", Exports.Mixture},
   }},
   {"Random", {
      {"pressure", double},
      {"temperature", double},
      {"magnitude", double},
      {"molarFracs", Exports.Mixture},
   }},
   {"TaylorGreen2DVortex", {
      {"pressure", double},
      {"temperature", double},
      {"velocity", double},
      {"molarFracs", Exports.Mixture},
   }},
   {"TaylorGreen3DVortex", {
      {"pressure", double},
      {"temperature", double},
      {"velocity", double},
      {"molarFracs", Exports.Mixture},
   }},
   {"Perturbed", {
      {"pressure", double},
      {"temperature", double},
      {"velocity", Array(3,double)},
      {"molarFracs", Exports.Mixture},
   }},
   {"RiemannTestOne", {}},
   {"RiemannTestTwo", {}},
   {"SodProblem", {}},
   {"LaxProblem", {}},
   {"ShuOsherProblem", {}},
   {"VortexAdvection2D", {
      {"pressure", double},
      {"temperature", double},
      {"velocity", Array(2,double)},
      {"molarFracs", Exports.Mixture},
   }},
   {"GrossmanCinnellaProblem", {}},
   {"ChannelFlow", {
      {"pressure", double},
      {"temperature", double},
      {"velocity", double},
      {"StreaksIntensity", double},
      {"RandomIntensity", double},
      {"molarFracs", Exports.Mixture},
   }},
   {"Restart", {
      {"restartDir", String(256)},
   }},
}

-- Grid definition input
Exports.GridTypes = Union{
   {"Uniform", {}},
   {"Cosine", {}},
   {"TanhMinus", {
      {"Stretching", double},
   }},
   {"TanhPlus", {
      {"Stretching", double},
   }},
   {"Tanh", {
      {"Stretching", double},
   }},
   {"SinhMinus", {
      {"Stretching", double},
   }},
   {"SinhPlus", {
      {"Stretching", double},
   }},
   {"Sinh", {
      {"Stretching", double},
   }},
}

Exports.GridInputStruct = Union{
   {"Cartesian", {
      -- grid's origin
      {"origin", Array(3, double)},
      -- width of the grid
      {"width", Array(3, double)},
      -- grid type in each direction
      {"xType", Exports.GridTypes},
      {"yType", Exports.GridTypes},
      {"zType", Exports.GridTypes},
   }},
   {"FromFile", {
      {"gridDir", String(256)},
   }},
}

-- Input for numerics
Exports.EulerSchemes = Union{
   {"SkewSymmetric", {}},
   {"TENO", {}},
   {"TENOA", {}},
   {"TENOLAD", {}},
   {"Hybrid", {
      {"vorticityScale", double},
   }}
}

Exports.TimeStepDefinitions = Union{
   {"ConstantDeltaTime", {
      {"DeltaTime", double},
   }},
   {"ConstantCFL", {
      {"cfl", double},
   }},
}

-- Electric field model struct
Exports.EFieldStruct = Union{
   {"Off", {}},
   {"Ybc", {
      -- Mean electric potential at the bottom boundary (in computational units)
      {"Phi_bottom", double},
      -- Mean electric potential at the top boundary (in computational units)
      {"Phi_top", double},
      -- Activate Robin boundary conditions
      {"Robin_bc", bool}
   }},
   options = { default = { type  = "Off"}, },
}

-- Sections of config struct
Exports.MappingStruct = {
   -- number of tiles in which to split the domain
   {"tiles", Array(3,int)},
   -- number of tiles to allocate to each rank
   {"tilesPerRank", Array(3,int)},
   -- unique id assigned to each sample, according to its order in the command
   -- line (first sample is 0, second is 1 etc.); the initial value of this
   -- option is irrelevant, it will be overriden by the code
   {"sampleId", Int{default = 0}},
   -- output directory for each sample; the initial value of this option is
   -- irrelevant, it will be overriden by the code
   {"outDir", String(256, {default = ""})},
   -- expected wall-clock execution time [minutes]
   {"wallTime", int},
}

Exports.GridStruct = {
   -- number of cells in the fluid grid
   {"xNum", int},
   {"yNum", int},
   {"zNum", int},
   -- grid input method
   {"GridInput", Exports.GridInputStruct},
}

Exports.BCStruct = {
   {"xBCLeft",  Exports.FlowBC},
   {"xBCRight", Exports.FlowBC},
   {"yBCLeft",  Exports.FlowBC},
   {"yBCRight", Exports.FlowBC},
   {"zBCLeft",  Exports.FlowBC},
   {"zBCRight", Exports.FlowBC},
   {"bufferZone", Exports.BufferZoneBC}
}

Exports.IntegratorStruct = {
   {"startIter", Int{default = 0}},
   {"startTime", double},
   {"resetTime", bool},
   {"maxIter", int},
   {"maxTime", double},
   -- Time step size definition
   {"TimeStep", Exports.TimeStepDefinitions},
   -- implicit or explicit approach for chemistry
   {"implicitChemistry", bool},
   -- Switch for the euler schemes
   {"EulerScheme", Exports.EulerSchemes},
}

Exports.FlowStruct = {
   {"mixture", Exports.MixtureModel},
   {"initCase", Exports.FlowInitCase},
   {"resetMixture", bool},
   {"initMixture", Exports.Mixture},
   {"bodyForce", Array(3,double)},
   {"turbForcing", Exports.TurbForcingModel},
   {"laser", Exports.LaserModel},
}

Exports.IOStruct = {
   -- whether to write restart files (requires compiling with HDF support)
   {"wrtRestart", bool},
   -- how often to write restart files
   {"restartEveryTimeSteps", int},
   -- Probes
   {"probesSamplingInterval", int},
   {"probes", UpTo(5, Exports.Volume)},
   -- Two-diemnsional averages
   {"AveragesSamplingInterval", int},
   {"ResetAverages", bool},
   {"YZAverages", UpTo(10, Exports.Volume)},
   {"XZAverages", UpTo(10, Exports.Volume)},
   {"XYAverages", UpTo(10, Exports.Volume)},
   -- One-diemnsional averages
   {"XAverages", UpTo(10, Exports.Volume)},
   {"YAverages", UpTo(10, Exports.Volume)},
   {"ZAverages", UpTo(10, Exports.Volume)},
   -- Volume probes
   {"volumeProbes", UpTo(10, Exports.VolumeProbe)},
}

-- Main config struct
Exports.Config = {
  {"Mapping", Exports.MappingStruct},
  {"Grid", Exports.GridStruct},
  {"BC", Exports.BCStruct},
  {"Integrator", Exports.IntegratorStruct},
  {"Flow", Exports.FlowStruct},
  {"IO", Exports.IOStruct},
  {"Efield", Exports.EFieldStruct},
}

return Exports
