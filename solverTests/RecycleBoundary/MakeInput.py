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
import HTRrestart

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

config["Integrator"]["EulerScheme"]["vorticityScale"] = UInf/delta99In

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

width = [10.0*delta99In, 2.5*delta99In, 1.0*delta99In]
T = dict()
T["type"] = "Uniform"
xN = gridGen.GetNodes(0.0, width[0], config["Grid"]["xNum"], T)

T["type"] = "SinhMinus"
T["Stretching"] = 1
def objective(s):
   T["Stretching"] = s[0]
   x = gridGen.GetNodes(0.0, width[1], config["Grid"]["yNum"], T)
   return (x[1] - x[0]) - yPlus*deltaNu
T["Stretching"], = fsolve(objective, 1.0)
yN = gridGen.GetNodes(0.0, width[1], config["Grid"]["yNum"], T)

T["type"] = "Uniform"
zN = gridGen.GetNodes(0.0, width[2], config["Grid"]["zNum"], T)

def xNodes(i):
   return xN[i]

def yNodes(j):
   return yN[j]

def zNodes(k):
   return zN[k]

xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config, xNodes=xNodes, yNodes=yNodes, zNodes=zNodes)

print("        L_x        x          L_y        x          L_z")

print(width[0]/deltaNu, " x ",
      width[1]/deltaNu, " x ",
      width[2]/deltaNu)

print(dx[0]/deltaNu, " x ",
      dy[1]/deltaNu, " x ",
      dz[0]/deltaNu)

##############################################################################
# Write config file                                                          #
##############################################################################
json.dump(config, args.out_json, indent=3)

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
def pressure(i, j, k):
   return PInf

T = np.interp(yGrid, yBL, TBL)
def temperature(i, j, k):
   return T[j]

def MolarFracs(i, j, k):
   return 1.0

u = np.interp(yGrid, yBL, uBL)
v = np.interp(yGrid, yBL, vBL)
def velocity(i, j, k):
   U = [u[j], v[j], 0.0]
   if yGrid[j] < delta99In:
      U[1] += SineAmp*UInf*np.sin(2.0*np.pi*zGrid[k]/width[2]*8)
   return U

def velocity_profile(i, j, k):
   return [u[j], v[j], 0.0]

restart = HTRrestart.HTRrestart(config)
restart.write(restartDir, 1,
              pressure,
              temperature,
              MolarFracs,
              velocity,
              T_p = temperature,
              Xi_p = MolarFracs,
              U_p = velocity_profile)

