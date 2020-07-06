#!/usr/bin/env python2

import argparse
import sys
import os
import json
import numpy as np
import h5py
from random import *
from scipy.integrate import odeint
from scipy.optimize import fsolve
from joblib import Parallel, delayed

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix

parser = argparse.ArgumentParser()
parser.add_argument('--np', nargs='?', default=1, type=int,
                    help='number of cores')
parser.add_argument('base_json', type=argparse.FileType('r'), default='base.json')
args = parser.parse_args()

##############################################################################
#                                 Setup Case                                 #
##############################################################################

# Read base config
config = json.load(args.base_json)

ReIn   = config["Case"]["ReInlet"]
MaInf  = config["Case"]["MaInf"]
TInf   = config["Case"]["TInf"]
PInf   = config["Case"]["PInf"]
TwOvT  = config["Case"]["TwOvTInf"]
xTurb  = config["Case"]["xTurb"]
yPlus  = config["Case"]["yPlus"]
FTT    = config["Case"]["FlowThroughTimesNoStat"]
FTTS   = config["Case"]["FlowThroughTimesStat"]

# Read properties
Pr              = config["Flow"]["prandtl"]
gamma           = config["Flow"]["gamma"]
R               = config["Flow"]["gasConstant"]

# Simulation setup
assert config["Flow"]["turbForcing"]["type"] == "OFF"
assert config["Flow"]["initCase"] == "Restart"
restartDir = "InitialCase"
config["Flow"]["restartDir"] = restartDir

# Free-stream mixture properties
cInf  = np.sqrt(gamma*R*TInf)
muInf = ConstPropMix.GetViscosity(TInf, config)

# Free-stream conditions
UInf = cInf*MaInf
rhoInf = ConstPropMix.GetDensity(TInf, PInf, config)

# Inlet displacement thickness
deltaStarIn = muInf*ReIn/(UInf*rhoInf)

# Wall properties
Tw = TInf*TwOvT
muW = ConstPropMix.GetViscosity(Tw, config)
rhoW = ConstPropMix.GetDensity(Tw, PInf, config)

r = Pr**(1.0/3.0)
Taw = TInf*(1.0 + r*0.5*(gamma - 1.0)*MaInf**2)

##############################################################################
#                        Compute similarity solution                         #
##############################################################################

Np = 500
def GetCBL():
   def CBLFun(U, y):
      u, F, h, G, g = U.T
      T = 1.0 + 0.5*h*(gamma - 1.0)*MaInf**2
      assert T>0
      mu = ConstPropMix.GetViscosity(T*TInf, config)/muInf
      return [ F/mu,
               -0.5*g*F/mu,
               Pr*G/mu,
               -0.5*Pr*G*g/mu - 2*F**2/mu,
               u/T ]

   eta = np.linspace(0, 50, Np)
   u_0 = 0.0
   #F_0 = 0.0
   h_0 = (Tw/TInf-1.0)*2/((gamma - 1.0)*MaInf**2)
   #G_0 = 0.0
   g_0 = 0.0

   def objective(A):
      F_0, G_0 = A
      U = odeint(CBLFun, [u_0, F_0, h_0, G_0, g_0], eta)
      u, F, h, G, g = U.T
      return [u[Np-1] - 1.0, h[Np-1]]

   A = fsolve(objective, [0.01, 0.1])
   F_0, G_0 = A

   U = odeint(CBLFun, [u_0, F_0, h_0, G_0, g_0], eta)
   u, F, h, G, g = U.T
   T = 1.0 + 0.5*h*(gamma - 1.0)*MaInf**2
   v = (eta*u/T - g)*T*0.5
   return eta, u, v, T

etaB, uB, vB, TB = GetCBL()

# Compute distance from leading edge
deltaStarNorm = 0.0
for i in range(Np-1):
   deltaStarNorm += (1.0 - 0.5*(uB[i+1]/TB[i+1] + uB[i]/TB[i]))*(etaB[i+1]- etaB[i])

x0 = (deltaStarIn/deltaStarNorm)**2*rhoInf*UInf/muInf
Rex0 = UInf*rhoInf*x0/muInf

# Reference friction coefficient
def getCfTurb(xGrid):

   def VanDriestII(Cf):
      Rexv = (xGrid-xTurb)*ReIn

      a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
      b = (Taw/Tw - 1.0)

      A = (2*a**2 - b)/np.sqrt(b**2 + 4*a**2)
      B =           b /np.sqrt(b**2 + 4*a**2)

      res = (np.arcsin(A) + np.arcsin(B))/np.sqrt(Cf*(Taw/TInf - 1.0))
      res-= 4.15*np.log10(Rexv*Cf*muInf/muW)
      res-= 1.7
      return res

   cf = fsolve(VanDriestII, 1e-4, xtol=1e-10)

   return cf

Cf = getCfTurb(config["Grid"]["xWidth"]+x0)
TauW = Cf*(rhoInf*UInf**2)*0.5
uTau = np.sqrt(TauW/rhoW)
deltaNu = muW/(uTau*rhoW)
tNu = deltaNu**2*rhoW/muW

# Rescale quantities
uB *= UInf
vB *= UInf
TB *= TInf
config["Grid"]["origin"][0] = x0
config["Grid"]["xWidth"] *= deltaStarIn
config["Grid"]["yWidth"] *= deltaStarIn
config["Grid"]["zWidth"] *= deltaStarIn

##############################################################################
#                            Boundary conditions                             #
##############################################################################
assert config["BC"]["xBCLeft"]["type"] == "NSCBC_Inflow"
assert config["BC"]["xBCLeft"]["VelocityProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["VelocityProfile"]["FileDir"] = restartDir
assert config["BC"]["xBCLeft"]["TemperatureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["TemperatureProfile"]["FileDir"] = restartDir
config["BC"]["xBCLeft"]["P"] = PInf

assert config["BC"]["xBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["xBCRight"]["P"] = PInf

assert config["BC"]["yBCLeft"]["type"] == "SuctionAndBlowingWall"
assert config["BC"]["yBCLeft"]["TemperatureProfile"]["type"] == "Constant"
config['BC']["yBCLeft"]["TemperatureProfile"]["temperature"] = Tw

config["BC"]["yBCLeft"]["Xmin"]  = 15*deltaStarIn + x0
config["BC"]["yBCLeft"]["Xmax"]  = 20*deltaStarIn + x0
config["BC"]["yBCLeft"]["X0"]    = 0.5*(config["BC"]["yBCLeft"]["Xmin"] + config["BC"]["yBCLeft"]["Xmax"])
config["BC"]["yBCLeft"]["sigma"] = 0.3*(config["BC"]["yBCLeft"]["X0"] - config["BC"]["yBCLeft"]["Xmin"])
config["BC"]["yBCLeft"]["Zw"]    = 0.1*config["Grid"]["zWidth"]

config["BC"]["yBCLeft"]["A"]     = [ 0.05*UInf, 0.05*UInf]
config["BC"]["yBCLeft"]["omega"] = [ 0.9*cInf/deltaStarIn, 0.9*cInf/deltaStarIn]
config["BC"]["yBCLeft"]["beta"]  = [ 0.3/deltaStarIn, -0.3/deltaStarIn]

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
                            1.0,
                            False)


yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                            config["Grid"]["yWidth"],
                            config["Grid"]["yNum"], 
                            config["Grid"]["yType"],
                            config["Grid"]["yStretching"],
                            False,
                            deltaNu*yPlus)

zGrid, dz = gridGen.GetGrid(config["Grid"]["origin"][2],
                            config["Grid"]["zWidth"],
                            config["Grid"]["zNum"], 
                            config["Grid"]["zType"],
                            1.0,
                            True)

print("        L_x        x          L_y        x          L_z")

print(config["Grid"]["xWidth"]/deltaNu, " x ",
      config["Grid"]["yWidth"]/deltaNu, " x ",
      config["Grid"]["zWidth"]/deltaNu)

print(dx[0]/deltaNu, " x ",
      dy[0]/deltaNu, " x ",
      dz[0]/deltaNu)

# Load mapping
Ntiles = config["Mapping"]["tiles"]

assert config["Grid"]["xNum"] % Ntiles[0] == 0 
assert config["Grid"]["yNum"] % Ntiles[1] == 0
assert config["Grid"]["zNum"] % Ntiles[2] == 0

NxTile = int(config["Grid"]["xNum"]/Ntiles[0])
NyTile = int(config["Grid"]["yNum"]/Ntiles[1])
NzTile = int(config["Grid"]["zNum"]/Ntiles[2])

halo = [1, 1, 0]

# Set maxTime
config["Integrator"]["maxTime"] = config["Grid"]["xWidth"]/UInf*FTT

with open("NoStats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

config["Integrator"]["maxTime"] = config["Grid"]["xWidth"]/UInf*FTT + FTTS*2*np.pi/config["BC"]["yBCLeftInflowProfile"]["omega"][0]

# Setup averages
config["IO"]["YZAverages"] = [{"fromCell" : [0, 0, 0],          "uptoCell" : [config["Grid"]["xNum"]+1, 0, config["Grid"]["zNum"]]}]

with open("Stats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

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
   velocity          = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dudtBoundary      = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dTdtBoundary      = np.ndarray(shape)
   pressure[:] = PInf
   dudtBoundary[:] = [0.0, 0.0, 0.0]
   dTdtBoundary[:] = 0.0
   for (k,kc) in enumerate(centerCoordinates):
      for (j,jc) in enumerate(kc):
         for (i,ic) in enumerate(jc):
            Re = rhoInf*UInf*xGrid[i+lo_bound[0]]/muInf
            yB = etaB*xGrid[i+lo_bound[0]]/np.sqrt(Re)
            vB1 = vB/np.sqrt(Re)

            u = np.interp(yGrid[j+lo_bound[1]], yB, uB)
            v = np.interp(yGrid[j+lo_bound[1]], yB, vB1)
            T = np.interp(yGrid[j+lo_bound[1]], yB, TB)

            centerCoordinates[k,j,i] = [xGrid[i+lo_bound[0]], yGrid[j+lo_bound[1]], zGrid[k+lo_bound[2]]]
            cellWidth        [k,j,i] = [   dx[i+lo_bound[0]],    dy[j+lo_bound[1]],    dz[k+lo_bound[2]]]
            temperature      [k,j,i] = T
            rho              [k,j,i] = ConstPropMix.GetDensity(T, PInf, config)
            MolarFracs       [k,j,i] = [1.0,]
            velocity         [k,j,i] = [    u,     v,                  0.0]

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
      fout["velocity_profile"][:] = velocity
      fout["temperature_profile"][:] = temperature

Parallel(n_jobs=args.np)(delayed(writeTile)(x, y, z) for x, y, z in np.ndindex((Ntiles[0], Ntiles[1], Ntiles[2])))

