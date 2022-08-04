#!/usr/bin/env python3

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

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix
import HTRrestart

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
Pr    = config["Flow"]["mixture"]["prandtl"]
gamma = config["Flow"]["mixture"]["gamma"]
R     = config["Flow"]["mixture"]["gasConstant"]

# Simulation setup
assert config["Flow"]["turbForcing"]["type"] == "OFF"
assert config["Flow"]["initCase"]["type"] == "Restart"
restartDir = "InitialCase"
config["Flow"]["initCase"]["restartDir"] = restartDir

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

Cf = getCfTurb(config["Grid"]["GridInput"]["width"][0]+x0)
TauW = Cf*(rhoInf*UInf**2)*0.5
uTau = np.sqrt(TauW/rhoW)
deltaNu = muW/(uTau*rhoW)
tNu = deltaNu**2*rhoW/muW

# Get VorticityScale
delta = 0.0
yBin = etaB*x0/np.sqrt(Rex0)
for i in range(Np):
   if (uB[i] > 0.99):
      delta = yBin[i]
      break

config["Integrator"]["EulerScheme"]["vorticityScale"] = UInf/delta

# Rescale quantities
uB *= UInf
vB *= UInf
TB *= TInf
config["Grid"]["GridInput"]["origin"][0] = x0
config["Grid"]["GridInput"]["width"][0] *= deltaStarIn
config["Grid"]["GridInput"]["width"][1] *= deltaStarIn
config["Grid"]["GridInput"]["width"][2] *= deltaStarIn

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
config["BC"]["yBCLeft"]["Zw"]    = 0.1*config["Grid"]["GridInput"]["width"][2]

config["BC"]["yBCLeft"]["A"]     = [ 0.05*UInf, 0.05*UInf]
config["BC"]["yBCLeft"]["omega"] = [ 0.9*cInf/deltaStarIn, 0.9*cInf/deltaStarIn]
config["BC"]["yBCLeft"]["beta"]  = [ 0.3/deltaStarIn, -0.3/deltaStarIn]

assert config["BC"]["yBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["yBCRight"]["P"] = PInf

##############################################################################
#                              Generate Grid                                 #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config, dyMinus = yPlus*deltaNu)

print("        L_x        x          L_y        x          L_z")

print(config["Grid"]["GridInput"]["width"][0]/deltaNu, " x ",
      config["Grid"]["GridInput"]["width"][1]/deltaNu, " x ",
      config["Grid"]["GridInput"]["width"][2]/deltaNu)

print(dx[0]/deltaNu, " x ",
      dy[1]/deltaNu, " x ",
      dz[0]/deltaNu)

# Set maxTime
config["Integrator"]["maxTime"] = config["Grid"]["GridInput"]["width"][0]/UInf*FTT

with open("NoStats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

config["Integrator"]["maxTime"] = config["Grid"]["GridInput"]["width"][0]/UInf*FTT + FTTS*2*np.pi/config["BC"]["yBCLeft"]["omega"][0]

# Setup averages
config["IO"]["YZAverages"] = [{"fromCell" : [0, 0, 0],          "uptoCell" : [config["Grid"]["xNum"]+1, 0, config["Grid"]["zNum"]]}]

with open("Stats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################

shape2D = (xGrid.size, yGrid.size)
u = np.ndarray(shape2D)
v = np.ndarray(shape2D)
T = np.ndarray(shape2D)
for i,x in enumerate(xGrid):
   Re = rhoInf*UInf*xGrid[i]/muInf
   yB1 = etaB*xGrid[i]/np.sqrt(Re)
   vB1 = vB/np.sqrt(Re)
   u[i,:] = np.interp(yGrid, yB1, uB)
   v[i,:] = np.interp(yGrid, yB1, vB1)
   T[i,:] = np.interp(yGrid, yB1, TB)

def pressure(i, j, k):
   return PInf

def temperature(i, j, k):
   return T[i,j]

def MolarFracs(i, j, k):
   return 1.0

def velocity(i, j, k):
   return [u[i,j], v[i,j], 0.0]

restart = HTRrestart.HTRrestart(config)
restart.write(restartDir, 1,
              pressure,
              temperature,
              MolarFracs,
              velocity,
              T_p = temperature,
              Xi_p = MolarFracs,
              U_p = velocity,
              nproc = args.np)

