#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import h5py
import random
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

ReIn    = config["Case"]["ReInlet"]
MaInf   = config["Case"]["MaInf"]
TInf    = config["Case"]["TInf"]
PInf    = config["Case"]["PInf"]
TwOvT   = config["Case"]["TwOvTInf"]
RandAmp = config["Case"]["RandAmp"]
SineAmp = config["Case"]["SineAmp"]
yPlus   = config["Case"]["yPlus"]
FTT     = config["Case"]["TNoStat"]
FTTS    = config["Case"]["TStat"]

# Read properties
Pr      = config["Flow"]["mixture"]["prandtl"]
gamma   = config["Flow"]["mixture"]["gamma"]
R       = config["Flow"]["mixture"]["gasConstant"]

config["Flow"]["mixture"]["viscosityModel"]["SRef"] = 110.4/273.15
config["Flow"]["mixture"]["viscosityModel"]["TempRef"] = 1.0

# Simulation setup
assert config["Flow"]["turbForcing"]["type"] == "OFF"
assert config["Flow"]["initCase"]["type"] == "Restart"
restartDir = "InitialCase"
config["Flow"]["initCase"]["restartDir"] = restartDir

# Free-stream conditions
cInf  = np.sqrt(gamma*R*TInf)
UInf = cInf*MaInf
rhoInf = ConstPropMix.GetDensity(TInf, PInf, config)
config["Flow"]["mixture"]["viscosityModel"]["ViscRef"] = (UInf*rhoInf)/ReIn
muInf = ConstPropMix.GetViscosity(TInf, config)
assert abs((UInf*rhoInf)/(ReIn*muInf) - 1) < 1e-12

# Inlet boundary layer thickness
delta99In = muInf*ReIn/(UInf*rhoInf)
config["Integrator"]["EulerScheme"]["vorticityScale"] = UInf/delta99In

# Wall properties
r = Pr**(1.0/3.0)
Taw = TInf*(1.0 + r*0.5*(gamma - 1.0)*MaInf**2)
TwOvT = Taw/TInf
config["Case"]["TwOvTInf"] = TwOvT
Tw = TInf*TwOvT
muW = ConstPropMix.GetViscosity(Tw, config)
rhoW = ConstPropMix.GetDensity(Tw, PInf, config)

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
config["Grid"]["GridInput"]["width"][0] *= delta99In
config["Grid"]["GridInput"]["width"][1] *= delta99In
config["Grid"]["GridInput"]["width"][2] *= delta99In

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
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config, dyMinus = yPlus*deltaNu)

print("        L_x        x          L_y        x          L_z")

print(config["Grid"]["GridInput"]["width"][0]/deltaNu, " x ",
      config["Grid"]["GridInput"]["width"][1]/deltaNu, " x ",
      config["Grid"]["GridInput"]["width"][2]/deltaNu)

print(dx[0]/deltaNu, " x ",
      dy[1]/deltaNu, " x ",
      dz[0]/deltaNu)

# Set maxTime
config["Integrator"]["maxTime"] = delta99In/UInf*FTT

with open("NoStats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

# Setup averages
config["IO"]["ZAverages"] = [{"fromCell" : [                       0,                        0,                     0],
                              "uptoCell" : [config["Grid"]["xNum"]+1, config["Grid"]["yNum"]+1, config["Grid"]["zNum"]]}]
config["Integrator"]["maxTime"] = delta99In/UInf*(FTT+FTTS)

with open("Stats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
u = np.interp(yGrid, yBL, uBL)
v = np.interp(yGrid, yBL, vBL)
T = np.interp(yGrid, yBL, TBL)

def pressure(i, j, k):
   return PInf

def temperature(i, j, k):
   return T[j]

def MolarFracs(i, j, k):
   return 1.0

def velocity(i, j, k):
   U = [u[j], v[j], 0.0]
   if (yGrid[j] < delta99In):
      U[0] += RandAmp*UInf*random.uniform(-1.0, 1.0)
      U[1] += RandAmp*UInf*random.uniform(-1.0, 1.0)
      U[2] += RandAmp*UInf*random.uniform(-1.0, 1.0)
      U[1] += SineAmp*UInf*np.sin(2.0*np.pi*zGrid[k]/config["Grid"]["GridInput"]["width"][2]*8)
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
              U_p = velocity_profile,
              nproc = args.np)

