#!/usr/bin/env python3

import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
import sys
import os
import h5py
import pandas
from scipy.integrate import odeint
from scipy.optimize import fsolve

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix
import Averages

parser = argparse.ArgumentParser()
parser.add_argument("-json", "--json_file", type=argparse.FileType('r'))
parser.add_argument("-in", "--input_dir")
args = parser.parse_args()

##############################################################################
#                         Read Prometeo Input File                           #
##############################################################################

config = json.load(args.json_file)

xNum    = config["Grid"]["xNum"]
yNum    = config["Grid"]["yNum"]
zNum    = config["Grid"]["zNum"]
xWidth  = config["Grid"]["GridInput"]["width"][0]
yWidth  = config["Grid"]["GridInput"]["width"][1]
zWidth  = config["Grid"]["GridInput"]["width"][2]
xOrigin = config["Grid"]["GridInput"]["origin"][0]
yOrigin = config["Grid"]["GridInput"]["origin"][1]
zOrigin = config["Grid"]["GridInput"]["origin"][2]

ReIn   = config["Case"]["ReInlet"]
MaInf  = config["Case"]["MaInf"]
TInf   = config["Case"]["TInf"]
PInf   = config["Case"]["PInf"]
TwOvT  = config["Case"]["TwOvTInf"]
xTurb  = config["Case"]["xTurb"]
yPlus  = config["Case"]["yPlus"]

R      = config["Flow"]["mixture"]["gasConstant"]
gamma  = config["Flow"]["mixture"]["gamma"]
Pr     = config["Flow"]["mixture"]["prandtl"]

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

##############################################################################
#                               Compute Grid                                 #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

##############################################################################
#                          Load reference solution                           #
##############################################################################

CfProf = pandas.read_csv("Cf.csv")
StProf = pandas.read_csv("St.csv")
UProf  = pandas.read_csv("Uprof.csv")

##############################################################################
#                          Load average files                                #
##############################################################################

Xavg = Averages.avg2D(args.input_dir+"/YZAverages/0,0-" + str(xNum+1) + ",0.hdf")

yrakes = []
for i in range(4):
   yrakes.append(Averages.avg2D(args.input_dir+"/XZAverages/0," + str(i) + "-" + str(yNum+1) + "," + str(i) + ".hdf"))
rakesNames = ["x=400", "x=650", "x=800", "x=950"]

##############################################################################
#                               Plot stuff                                   #
##############################################################################

figureDir = "Figures"
if not os.path.exists(figureDir):
   os.makedirs(figureDir)

# Skin Friction coefficients
def getCfTurb(xGrid):

   cf = np.zeros(xGrid.size)

   for i,x in enumerate(cf):
      def VanDriestII(Cf):
         if (xGrid[i] > 500):
            Rexv = (xGrid[i]-xTurb)*ReIn

            a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
            b = (Taw/Tw - 1.0)

            A = (2*a**2 - b)/np.sqrt(b**2 + 4*a**2)
            B =           b /np.sqrt(b**2 + 4*a**2)

            res = (np.arcsin(A) + np.arcsin(B))/np.sqrt(Cf*(Taw/TInf - 1.0))
            res-= 4.15*np.log10(Rexv*Cf*muInf/muW)
            res-= 1.7
         else:
            res = Cf

         return res

      cf[i] = fsolve(VanDriestII, 1e-4, xtol=1e-10)

      indT = next((i for i, x in enumerate(cf) if x), None)

   return cf, indT

cfLam = 0.664*np.sqrt((rhoW*muW)/(rhoInf*muInf))/np.sqrt((rhoInf*UInf*xGrid/muInf))
cfTurb, indT = getCfTurb((xGrid-xOrigin)/deltaStarIn)

plt.figure(0)
plt.plot((xGrid-xOrigin)/deltaStarIn/100, 2.0*Xavg.tau[:,1]/(rhoInf*UInf**2)*1e4,        label="Present formulation")
plt.plot((xGrid-xOrigin)/deltaStarIn/100,                              cfLam*1e4, '--k', label="Laminar BL")
plt.plot((xGrid[indT:]-xOrigin)/deltaStarIn/100,               cfTurb[indT:]*1e4, '-.k', label="Turbulent BL")
plt.plot(                 CfProf["x"][:].values,          CfProf["Cf"][:].values,  'ok', label="Franko and Lele (2013)")
plt.xlabel(r"$(x-x_0)/\delta_0^* \times 10^{-2}$", fontsize = 20)
plt.ylabel(r"$\overline{C_f} \times 10^{4}$"     , fontsize = 20)
plt.gca().set_xlim(0, 10)
plt.gca().set_ylim(0, 25)
plt.legend()
plt.savefig(os.path.join(figureDir, "Cf.eps"), bbox_inches='tight')

cp = gamma/(gamma-1)*R
StLam  = cfLam*(Pr/(Taw/TInf - TwOvT)*((TB[1]-TB[0])/(uB[1]-uB[0])))
StTurb = cfTurb/(2.0*Pr**(2.0/3.0))
St = -Xavg.q[:,1]/(rhoInf*UInf*cp*(Taw - Tw))

plt.figure(1)
plt.plot(       (xGrid-xOrigin)/deltaStarIn/100,            St*1e4,               label="Present formulation")
plt.plot(       (xGrid-xOrigin)/deltaStarIn/100,         StLam*1e4, '--k',        label="Laminar BL")
plt.plot((xGrid[indT:]-xOrigin)/deltaStarIn/100, StTurb[indT:]*1e4, '-.k',        label="Turbulent BL")
plt.plot(                 StProf["x"][:].values,   StProf["St"][:].values,  'ok', label="Franko and Lele (2013)")
plt.xlabel(r"$(x-x_0)/\delta_0^* \times 10^{-2}$", fontsize = 20)
plt.ylabel(r"$\overline{St} \times 10^{4}$"      , fontsize = 20)
plt.gca().set_xlim(0, 10)
plt.gca().set_ylim(0, 25)
plt.legend()
plt.savefig(os.path.join(figureDir, "St.eps"), bbox_inches='tight')

plt.show()
