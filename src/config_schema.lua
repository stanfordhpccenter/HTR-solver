-- Copyright (c) "2019, by Stanford University
--               Contributors: Mario Di Renzo
--               Affiliation: Center for Turbulence Research, Stanford University
--               URL: https://ctr.stanford.edu
--               Citation: Di Renzo M., Lin F. and Urzay J. "HTR solver: An open-source
--                         exascale-oriented task-based multi-GPU high-order code for
--                         hypersonic aerothermodynamics." Computer Physics Communications XXX  (2020)"
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
Exports.Volume = {
  fromCell = Array(3,int),
  uptoCell = Array(3,int),
}
Exports.Window = {
  fromCell = Array(2,int),
  uptoCell = Array(2,int),
}
Exports.Species = {
  Name = String(10),
  MolarFrac = double,
}
Exports.Mixture = {
  Species = UpTo(10, Exports.Species),
}

-- Unions & enumeration constants
Exports.FlowBC = Enum('Dirichlet','Periodic','Symmetry','AdiabaticWall','IsothermalWall','NSCBC_Inflow','NSCBC_Outflow','SuctionAndBlowingWall')
Exports.ViscosityModel = Enum('Constant','PowerLaw','Sutherland')
Exports.FlowInitCase = Enum('Uniform','Random','Restart','Perturbed','TaylorGreen2DVortex','TaylorGreen3DVortex','RiemannTestOne','RiemannTestTwo','SodProblem','LaxProblem','ShuOsherProblem','VortexAdvection2D','GrossmanCinnellaProblem','ChannelFlow')
Exports.GridType = Enum('Uniform','Cosine','TanhMinus','TanhPlus','Tanh')
Exports.MixtureProfile = Union{
  Constant = {
    Mixture = Exports.Mixture,
  },
  File = {
    FileDir = String(256),
  },
  Incoming = {},
}
Exports.TempProfile = Union{
  Constant = {
    temperature = double,
  },
  File = {
    FileDir = String(256),
  },
  Incoming = {},
}
Exports.InflowProfile = Union{
  Constant = {
    velocity = Array(3,double),
  },
  File = {
    FileDir = String(256),
  },
  Incoming = {
    addedVelocity = double,
  },
  SuctionAndBlowing = {
    Xmin  = double,
    Xmax  = double,
    X0    = double,
    sigma = double,
    Zw    = double,
    A     = UpTo(20, double),
    omega = UpTo(20, double),
    beta  = UpTo(20, double),
  },
}
Exports.TurbForcingModel = Union{
  OFF = {},
  CHANNEL = {
     RhoUbulk = double,
     Forcing = double,
  },
}

-- Sections of config struct
Exports.MappingStruct = {
   -- number of tiles in which to split the domain
   tiles = Array(3,int),
   -- number of tiles to allocate to each rank
   tilesPerRank = Array(3,int),
   -- unique id assigned to each sample, according to its order in the command
   -- line (first sample is 0, second is 1 etc.); the initial value of this
   -- option is irrelevant, it will be overriden by the code
   sampleId = int,
   -- output directory for each sample; the initial value of this option is
   -- irrelevant, it will be overriden by the code
   outDir = String(256),
   -- expected wall-clock execution time [minutes]
   wallTime = int,
}

Exports.GridStruct = {
   -- number of cells in the fluid grid
   xNum = int,
   yNum = int,
   zNum = int,
   -- coordinates of the fluid grid's origin [m]
   origin = Array(3,double),
   -- width of the fluid grid [m]
   xWidth = double,
   yWidth = double,
   zWidth = double,
   -- grid type in each direction
   xType = Exports.GridType,
   yType = Exports.GridType,
   zType = Exports.GridType,
   -- grid stretching factor in each direction
   xStretching = double,
   yStretching = double,
   zStretching = double,
}

Exports.BCStruct = {
   xBCLeft = Exports.FlowBC,
   xBCLeftInflowProfile = Exports.InflowProfile,
   xBCLeftP = double,
   xBCLeftHeat = Exports.TempProfile,
   xBCLeftMixture = Exports.MixtureProfile,
   xBCRight = Exports.FlowBC,
   xBCRightInflowProfile = Exports.InflowProfile,
   xBCRightP = double,
   xBCRightHeat = Exports.TempProfile,
   xBCRightMixture = Exports.MixtureProfile,
   yBCLeft = Exports.FlowBC,
   yBCLeftInflowProfile = Exports.InflowProfile,
   yBCLeftP = double,
   yBCLeftHeat = Exports.TempProfile,
   yBCLeftMixture = Exports.MixtureProfile,
   yBCRight = Exports.FlowBC,
   yBCRightInflowProfile = Exports.InflowProfile,
   yBCRightP = double,
   yBCRightHeat = Exports.TempProfile,
   yBCRightMixture = Exports.MixtureProfile,
   zBCLeft = Exports.FlowBC,
   zBCLeftInflowProfile = Exports.InflowProfile,
   zBCLeftP = double,
   zBCLeftHeat = Exports.TempProfile,
   zBCLeftMixture = Exports.MixtureProfile,
   zBCRight = Exports.FlowBC,
   zBCRightInflowProfile = Exports.InflowProfile,
   zBCRightP = double,
   zBCRightHeat = Exports.TempProfile,
   zBCRightMixture = Exports.MixtureProfile,
}

Exports.IntegratorStruct = {
   startIter = int,
   startTime = double,
   resetTime = bool,
   maxIter = int,
   maxTime = double,
   cfl = double,
   fixedDeltaTime = double,
   -- implicit or explicit approach for chemistry
   implicitChemistry = bool,
}

Exports.FlowStruct = {
   mixture = String(20),
   gasConstant = double,
   gamma = double,
   prandtl = double,
   viscosityModel = Exports.ViscosityModel,
   constantVisc = double,
   powerlawViscRef = double,
   powerlawTempRef = double,
   sutherlandViscRef = double,
   sutherlandTempRef = double,
   sutherlandSRef = double,
   initCase = Exports.FlowInitCase,
   restartDir = String(256),
   initParams = Array(5,double),
   resetMixture = bool,
   initMixture = Exports.Mixture,
   bodyForce = Array(3,double),
   turbForcing = Exports.TurbForcingModel,
}

Exports.IOStruct = {
   -- whether to write restart files (requires compiling with HDF support)
   wrtRestart = bool,
   -- how often to write restart files
   restartEveryTimeSteps = int,
   -- temperature probes
   probes = UpTo(5, Exports.Volume),
   -- One-diemnsional averages
   AveragesSamplingInterval = int,
   ResetAverages = bool,
   YZAverages = UpTo(5, Exports.Volume),
   XZAverages = UpTo(5, Exports.Volume),
   XYAverages = UpTo(5, Exports.Volume),
}

-- Main config struct
Exports.Config = {
  Mapping = Exports.MappingStruct,
  Grid = Exports.GridStruct,
  BC = Exports.BCStruct,
  Integrator = Exports.IntegratorStruct,
  Flow = Exports.FlowStruct,
  IO = Exports.IOStruct,
}

return Exports
