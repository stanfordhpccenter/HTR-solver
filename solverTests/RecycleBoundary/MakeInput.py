#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import h5py
import random
from scipy.optimize import fsolve

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix

parser = argparse.ArgumentParser()
parser.add_argument('--np', nargs='?', default=1, type=int,
                    help='number of cores')
parser.add_argument('base_json', type=argparse.FileType('r'), default='base.json')
parser.add_argument('out_json',  type=argparse.FileType('w'), default='Run.json')
args = parser.parse_args()

##############################################################################
#                                 Setup Case                                 #
##############################################################################

# Read base config
config = json.load(args.base_json)

ReIn    = config["Case"]["ReInlet"]
MaInf   = config["Case"]["MaInf"]
TInf    = config["Case"]["TInf"]
PInf    = config["Case"]["PInf"]
TwOvT   = config["Case"]["TwOvTInf"]
SineAmp = config["Case"]["SineAmp"]
yPlus   = config["Case"]["yPlus"]
del config["Case"]

# Read properties
Pr              = config["Flow"]["mixture"]["prandtl"]
gamma           = config["Flow"]["mixture"]["gamma"]
R               = config["Flow"]["mixture"]["gasConstant"]

# Simulation setup
assert config["Flow"]["turbForcing"]["type"] == "OFF"
assert config["Flow"]["initCase"]["type"] == "Restart"
restartDir = "InflowProfile"
config["Flow"]["initCase"]["restartDir"] = restartDir

# Free-stream mixture properties
cInf  = np.sqrt(gamma*R*TInf)
muInf = ConstPropMix.GetViscosity(TInf, config)

# Free-stream conditions
UInf = cInf*MaInf
rhoInf = ConstPropMix.GetDensity(TInf, PInf, config)

# Inlet boundary layer thickness
delta99In = muInf*ReIn/(UInf*rhoInf)

# Wall properties
Tw = TInf*TwOvT
muW = ConstPropMix.GetViscosity(Tw, config)
rhoW = ConstPropMix.GetDensity(Tw, PInf, config)

r = Pr**(1.0/3.0)
Taw = TInf*(1.0 + r*0.5*(gamma - 1.0)*MaInf**2)

config["Integrator"]["vorticityScale"] = UInf/delta99In

##############################################################################
#                        Compute similarity solution                         #
##############################################################################

# See page 545 White's book
def VanDriestUeq(u):
   a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
   b = Taw/Tw - 1.0
   Q = np.sqrt(b**2 + 4.0*a**2)
   return UInf/a*(np.arcsin((2*a**2*u/UInf - b)/Q) + np.arcsin(b/Q))

def FavreU(uEq):
   a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
   b = Taw/Tw - 1.0
   Q = np.sqrt(b**2 + 4.0*a**2)
   return 0.5*UInf/(a**2)*(np.sin(uEq*a/UInf - np.arcsin(b/Q))*Q + b)

def getCfFromTheta(theta):
   Retheta = ReIn*theta/delta99In
   Ft = muInf/muW
   Rethetainc = Ft*Retheta
   Cfinc = 1./(17.08*(np.log10(Rethetainc))**2 + 25.11*np.log10(Rethetainc) + 6.012)

   a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
   b = (Taw/Tw - 1.0)
   Q = np.sqrt(b**2 + 4.0*a**2)
   A = (2*a**2 - b)/Q
   B =           b /Q
   return Cfinc*(np.arcsin(A) + np.arcsin(B))**2/(Taw/TInf - 1.0)

def getVBL(Cf, theta, y, u, rho):
   ddeltadx = Cf*0.5/theta
   vint = 0.
   v = np.zeros(len(y))
   for i in range(1, Np):
      dy = y[i] - y[i-1]
      vint += 0.5*dy*(rho[i-1]*u[i-1] + rho[i-1]*u[i-1])
      v[i] = ddeltadx*(y[i]*u[i] - vint/rho[i])
   return v

Np = 2000
def GetProfile(Cf):
   yp = np.linspace(0, 2000, Np)

   TauW = Cf*(rhoInf*UInf**2)*0.5
   uTau = np.sqrt(TauW/rhoW)
   deltaNu = muW/(uTau*rhoW)

   UeqInf = VanDriestUeq(UInf)
   delta99p = delta99In/deltaNu

   # Von Karman constant
   k = 0.41
   c = 5.2

   def profFunc(y):
      # Parameters for Reichardt's inner layer and Finley's wall law
      c1  = -1./k*np.log(k)+c
      b   = 0.33
      eta = 11.0
      return 1.0/k * np.log(1.0 + k*y) + c1*(1.0 - np.exp(-y/eta) - y/eta*np.exp(-b*y))

   # Coles wake parameter
   cp = 0.5*k*(UeqInf/uTau - profFunc(delta99p))

   Ueq = np.zeros(Np)
   for i in range(Np):
      yovD = yp[i]/delta99p
      if (yovD < 1.0):
         Ueq[i] = uTau*(profFunc(yp[i]) + 2.0*cp/k*(np.sin(0.5*np.pi*yovD))**2)
      else:
         Ueq[i] = UeqInf

   U = FavreU(Ueq)
   T = Tw*(1.0 + (Taw/Tw - 1.0)*(U/UInf) - r*0.5*(gamma-1.0)*MaInf**2*TInf/Tw*(U/UInf)**2)
   rho = ConstPropMix.GetDensity(T, PInf, config)

   assert abs(U[-1]-UInf)/UInf < 1e-3

   return yp*deltaNu, U, T, rho

def getTheta(y, u, rho, UInf):
   t = 0.0
   for i in range(1, Np):
      dy = y[i] - y[i-1]
      t += dy * u[i]/UInf * rho[i]/rhoInf*(1.0 - u[i]/UInf)
   return t

def getCf(Cf):

   def objective(A):
      y, U, T, rho = GetProfile(A)
      theta = getTheta(y, U, rho, UInf)
      return getCfFromTheta(theta) - A

   return fsolve(objective, Cf)

Cf = getCf(1e-4)[0]
TauW = Cf*(rhoInf*UInf**2)*0.5
uTau = np.sqrt(TauW/rhoW)
deltaNu = muW/(uTau*rhoW)

yBL, uBL, TBL, rhoBL = GetProfile(Cf)
theta = getTheta(yBL, uBL, rhoBL, UInf)
vBL = getVBL(Cf, theta, yBL, uBL, rhoBL)

# Rescale quantities
config["Grid"]["xWidth"] *= delta99In
config["Grid"]["yWidth"] *= delta99In
config["Grid"]["zWidth"] *= delta99In

##############################################################################
#                            Boundary conditions                             #
##############################################################################
assert config["BC"]["xBCLeft"]["type"] == "RecycleRescaling"
assert config["BC"]["xBCLeft"]["VelocityProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["VelocityProfile"]["FileDir"] = restartDir
assert config["BC"]["xBCLeft"]["TemperatureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["TemperatureProfile"]["FileDir"] = restartDir
config["BC"]["xBCLeft"]["P"] = PInf

assert config["BC"]["xBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["xBCRight"]["P"] = PInf

assert config["BC"]["yBCLeft"]["type"] == "IsothermalWall"
assert config["BC"]["yBCLeft"]["TemperatureProfile"]["type"] == "Constant"
config['BC']["yBCLeft"]["TemperatureProfile"]["temperature"] = Tw

assert config["BC"]["yBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["yBCRight"]["P"] = PInf

##############################################################################
#                              Generate Grid                                 #
##############################################################################

def objective(yStretching):
   yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                               config["Grid"]["yWidth"],
                               config["Grid"]["yNum"], 
                               config["Grid"]["yType"],
                               yStretching,
                               False)
   return dy[1]/deltaNu - yPlus

config["Grid"]["yStretching"], = fsolve(objective, 1.0)

xGrid, dx = gridGen.GetGrid(config["Grid"]["origin"][0],
                            config["Grid"]["xWidth"],
                            config["Grid"]["xNum"], 
                            config["Grid"]["xType"],
                            config["Grid"]["zStretching"],
                            False)


yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                            config["Grid"]["yWidth"],
                            config["Grid"]["yNum"], 
                            config["Grid"]["yType"],
                            config["Grid"]["yStretching"],
                            False)#,
                            #deltaNu*yPlus)

zGrid, dz = gridGen.GetGrid(config["Grid"]["origin"][2],
                            config["Grid"]["zWidth"],
                            config["Grid"]["zNum"], 
                            config["Grid"]["zType"],
                            config["Grid"]["zStretching"],
                            True)

# Write config file
json.dump(config, args.out_json, indent=3)

print("        L_x        x          L_y        x          L_z")

print(config["Grid"]["xWidth"]/deltaNu, " x ",
      config["Grid"]["yWidth"]/deltaNu, " x ",
      config["Grid"]["zWidth"]/deltaNu)

print(dx[0]/deltaNu, " x ",
      dy[0]/deltaNu, " x ",
      dz[0]/deltaNu)

# Load mapping
assert config["Mapping"]["tiles"][0] % config["Mapping"]["tilesPerRank"][0] == 0
assert config["Mapping"]["tiles"][1] % config["Mapping"]["tilesPerRank"][1] == 0
assert config["Mapping"]["tiles"][2] % config["Mapping"]["tilesPerRank"][2] == 0
Ntiles = config["Mapping"]["tiles"]
Ntiles[0] = int(Ntiles[0]/config["Mapping"]["tilesPerRank"][0])
Ntiles[1] = int(Ntiles[1]/config["Mapping"]["tilesPerRank"][1])
Ntiles[2] = int(Ntiles[2]/config["Mapping"]["tilesPerRank"][2])

assert config["Grid"]["xNum"] % Ntiles[0] == 0 
assert config["Grid"]["yNum"] % Ntiles[1] == 0
assert config["Grid"]["zNum"] % Ntiles[2] == 0

NxTile = int(config["Grid"]["xNum"]/Ntiles[0])
NyTile = int(config["Grid"]["yNum"]/Ntiles[1])
NzTile = int(config["Grid"]["zNum"]/Ntiles[2])

halo = [1, 1, 0]

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
if not os.path.exists(restartDir):
   os.makedirs(restartDir)

def writeTile(xt, yt, zt):
   lo_bound = [(xt  )*NxTile  +halo[0], (yt  )*NyTile  +halo[1], (zt  )*NzTile  +halo[2]]
   hi_bound = [(xt+1)*NxTile-1+halo[0], (yt+1)*NyTile-1+halo[1], (zt+1)*NzTile-1+halo[2]]
   if (xt == 0): lo_bound[0] -= halo[0]
   if (yt == 0): lo_bound[1] -= halo[1]
   if (zt == 0): lo_bound[2] -= halo[2]
   if (xt == Ntiles[0]-1): hi_bound[0] += halo[0]
   if (yt == Ntiles[1]-1): hi_bound[1] += halo[1]
   if (zt == Ntiles[2]-1): hi_bound[2] += halo[2]
   filename = ('%s,%s,%s-%s,%s,%s.hdf'
      % (lo_bound[0],  lo_bound[1],  lo_bound[2],
         hi_bound[0],  hi_bound[1],  hi_bound[2]))
   print("Working on: ", filename)

   shape = [hi_bound[2] - lo_bound[2] +1,
            hi_bound[1] - lo_bound[1] +1,
            hi_bound[0] - lo_bound[0] +1]

   centerCoordinates = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   cellWidth         = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   rho               = np.ndarray(shape)
   pressure          = np.ndarray(shape)
   temperature       = np.ndarray(shape)
   MolarFracs        = np.ndarray(shape, dtype=np.dtype('(1,)f8'))
   velocity_profile  = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   velocity          = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dudtBoundary      = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dTdtBoundary      = np.ndarray(shape)
   pressure[:] = PInf
   dudtBoundary[:] = [0.0, 0.0, 0.0]
   dTdtBoundary[:] = 0.0
   for (k,kc) in enumerate(centerCoordinates):
      for (j,jc) in enumerate(kc):
         for (i,ic) in enumerate(jc):

            u = np.interp(yGrid[j+lo_bound[1]], yBL, uBL)
            v = np.interp(yGrid[j+lo_bound[1]], yBL, vBL)
            w = 0.0
            T = np.interp(yGrid[j+lo_bound[1]], yBL, TBL)

            centerCoordinates[k,j,i] = [xGrid[i+lo_bound[0]], yGrid[j+lo_bound[1]], zGrid[k+lo_bound[2]]]
            cellWidth        [k,j,i] = [   dx[i+lo_bound[0]],    dy[j+lo_bound[1]],    dz[k+lo_bound[2]]]
            temperature      [k,j,i] = T
            rho              [k,j,i] = ConstPropMix.GetDensity(T, PInf, config)
            MolarFracs       [k,j,i] = [1.0,]
            velocity_profile [k,j,i] = [ u, v, w]
            velocity         [k,j,i] = [ u, v, w]

            if yGrid[j+lo_bound[1]] < delta99In:
               velocity[k,j,i][1] += SineAmp*UInf*np.sin(2.0*np.pi*zGrid[k+lo_bound[2]]/config["Grid"]["zWidth"]*8)


   with h5py.File(os.path.join(restartDir, filename), 'w') as fout:
      fout.attrs.create("SpeciesNames", ["MIX".encode()], dtype="S20")
      fout.attrs.create("timeStep", 0)
      fout.attrs.create("simTime", 0.0)
      fout.attrs.create("channelForcing", 0.0)

      fout.create_dataset("centerCoordinates",     shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("cellWidth",             shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("(1,)f8"))
      fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs_profile",    shape=shape, dtype = np.dtype("(1,)f8"))
      fout.create_dataset("velocity_profile",      shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("temperature_profile",   shape=shape, dtype = np.dtype("f8"))

      fout["centerCoordinates"][:] = centerCoordinates
      fout["cellWidth"][:] = cellWidth
      fout["rho"][:] = rho
      fout["pressure"][:] = pressure
      fout["temperature"][:] = temperature
      fout["MolarFracs"][:] = MolarFracs
      fout["velocity"][:] = velocity
      fout["dudtBoundary"][:] = dudtBoundary
      fout["dTdtBoundary"][:] = dTdtBoundary
      fout["MolarFracs_profile"][:] = MolarFracs
      fout["velocity_profile"][:] = velocity_profile
      fout["temperature_profile"][:] = temperature

for x, y, z in np.ndindex((Ntiles[0], Ntiles[1], Ntiles[2])): writeTile(x, y, z)

