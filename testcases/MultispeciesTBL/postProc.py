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
import pandas

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import Averages

parser = argparse.ArgumentParser()
parser.add_argument("-json", "--json_file", type=argparse.FileType('r'))
parser.add_argument("-in", "--input_dir")
args = parser.parse_args()

##############################################################################
#                          Read similarity solution                          #
##############################################################################

CBL = pandas.read_csv("SimilaritySolution.dat")
etaB = CBL["eta"][:].values
fB = CBL["f"][:].values
uB = CBL["u"][:].values
TB = CBL["T"][:].values
N2B = CBL["X_N2"][:].values
O2B = CBL["X_O2"][:].values
NOB = CBL["X_NO"][:].values
NB  = CBL["X_N" ][:].values
OB  = CBL["X_O" ][:].values
muB = CBL["mu"][:].values
rhoB = CBL["rho"][:].values
cB = CBL["SoS"][:].values
Rex0 = CBL["Rex"][0]

x0 = Rex0*muB[-1]/(uB[-1]*rhoB[-1])
deltaStarIn = 0.0
for i in range(uB.size-1):
   deltaStarIn += 0.5*((1.0 - uB[i+1]*rhoB[i+1]/(uB[-1]*rhoB[-1]))*rhoB[-1]/rhoB[i+1] +
                       (1.0 - uB[i  ]*rhoB[i  ]/(uB[-1]*rhoB[-1]))*rhoB[-1]/rhoB[i  ] )*(etaB[i+1] - etaB[i])
deltaStarIn *= x0*np.sqrt(2.0/Rex0)
print("Re_delta0 = ", uB[-1]*rhoB[-1]*deltaStarIn/muB[-1])

yB = np.zeros(etaB.size)
for i in range(etaB.size):
   if (i != 0) :
      rhoMid = 0.5*(rhoB[i  ] + rhoB[i-1])
      dyB = x0*np.sqrt(2/Rex0)*rhoB[-1]/rhoMid*(etaB[i] - etaB[i-1])
      yB[i] = yB[i-1] + dyB

vB = 0.5*yB/x0*uB - rhoB[-1]/rhoB*1.0/np.sqrt(2*Rex0)*fB

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

# Free-stream mixture properties
cInf  =  cB[-1]
muInf = muB[-1]

# Free-stream conditions
assert abs(TB[-1] - TInf) < 1e-3
UInf   =   uB[-1]
rhoInf = rhoB[-1]
TInf   = TB[-1]
assert abs(UInf/cInf - MaInf)  < 1e-3

# Inlet displacement thickness
assert abs(deltaStarIn -muInf*ReIn/(UInf*rhoInf)) < 1e-3

# Wall properties
Tw   =   TB[0]
muW  =  muB[0]
rhoW = rhoB[0]
assert abs(Tw/TB[-1] - TwOvT) < 1e-3

##############################################################################
#                               Compute Grid                                 #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

##############################################################################
#                          Load average files                                #
##############################################################################

Xavg = Averages.avg2D(args.input_dir+"/YZAverages/0,0-" + str(xNum+1) + ",0.hdf")

##############################################################################
#                               Plot stuff                                   #
##############################################################################

figureDir = "Figures"
if not os.path.exists(figureDir):
   os.makedirs(figureDir)

Cf = 2.0*Xavg.tau[:,1]/(rhoInf*UInf**2)
Q  = -Xavg.q[:,1]/(rhoInf*UInf*UInf*UInf)

plt.rcParams.update({'font.size': 12})

plt.figure(0)
plt.plot((xGrid-xOrigin)/deltaStarIn/100, Cf*1e4,  "-k",label=r"$C_f \times 10^{4}$")
plt.plot((xGrid-xOrigin)/deltaStarIn/100,  Q*1e5, "--k",label=r"$-q_w/(\rho_{\infty} U_{\infty}^3)  \times 10^{5}$")
plt.xlabel(r"$(x-x_0)/\delta_0^* \times 10^{-2}$")
#plt.xlabel(r"$(x-x_0)/\delta_0^* \times 10^{-2}$", fontsize = 20)
plt.gca().set_xlim(0, 10)
plt.gca().set_ylim(0, 25)
plt.gca().set_aspect(0.15)
plt.legend()
plt.savefig(os.path.join(figureDir, "Coefficients.eps"), bbox_inches='tight')

plt.show()
