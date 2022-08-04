#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import h5py
from random import *
import pandas

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix
import HTRrestart

parser = argparse.ArgumentParser()
parser.add_argument('base_json', type=argparse.FileType('r'), default='base.json')
args = parser.parse_args()

##############################################################################
#                          Read similarity solution                          #
##############################################################################

CBL = pandas.read_csv("SimilaritySolution.dat")
etaB = CBL["eta"][:].values
fB = CBL["f"][:].values
uB = CBL["u"][:].values
TB = CBL["T"][:].values

##############################################################################
#                                 Setup Case                                 #
##############################################################################

# Read base config
config = json.load(args.base_json)

gamma = config["Flow"]["mixture"]["gamma"]
R     = config["Flow"]["mixture"]["gasConstant"]
Pr    = config["Flow"]["mixture"]["prandtl"]

assert config["Flow"]["initCase"]["type"] == "Restart"
restartDir = config["Flow"]["initCase"]["restartDir"]

config["BC"]["xBCLeft"]["VelocityProfile"]["FileDir"] = restartDir
config["BC"]["xBCLeft"]["TemperatureProfile"]["FileDir"] = restartDir
TInf = 300.0
Tw   = 300.0
P    = config["BC"]["xBCLeft"]["P"]
UInf = 2083.67
Rex0 = 100000

config["Integrator"]["EulerScheme"]["vorticityScale"] = UInf/0.0528088569936

aInf = np.sqrt(gamma*R*TInf)
MaInf = UInf/aInf

RhoInf = ConstPropMix.GetDensity(TInf, P, config)
muInf = ConstPropMix.GetViscosity(TInf, config)
nuInf = muInf/RhoInf

##############################################################################
#                              Generate Grid                                 #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

##############################################################################
#            Rescale Reynodls number of similarity solution                  #
##############################################################################
yB = np.zeros(etaB.size)
for i in range(etaB.size):
   if (i != 0) :
      rhoMid = 0.5*(ConstPropMix.GetDensity(TB[i  ]*TInf, P, config) +
                    ConstPropMix.GetDensity(TB[i-1]*TInf, P, config))
      dyB = np.sqrt(2*Rex0)*muInf/(rhoMid*UInf)*(etaB[i] - etaB[i-1])
      yB[i] = yB[i-1] + dyB

x0 = Rex0*nuInf/UInf
vB = 0.5*yB/x0*uB - TB/np.sqrt(2*Rex0)*fB

uB *= UInf
vB *= UInf
TB *= TInf

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
shape2D = (xGrid.size, yGrid.size)
u = np.ndarray(shape2D)
v = np.ndarray(shape2D)
T = np.ndarray(shape2D)
for i,x in enumerate(xGrid):
   Re = (x+x0)*UInf/nuInf
   yB1 = yB*np.sqrt(Re/Rex0)
   vB1 = vB/np.sqrt(Re/Rex0)
   u[i,:] = np.interp(yGrid, yB1, uB)
   v[i,:] = np.interp(yGrid, yB1, vB1)
   T[i,:] = np.interp(yGrid, yB1, TB)

def pressure(i, j, k):
   return P

def temperature(i, j, k):
   return T[i,j]

def MolarFracs(i, j, k):
   return 1.0

def velocity(i, j, k):
   U = [u[i,j], v[i,j], 0.0]
   return U

restart = HTRrestart.HTRrestart(config)
restart.write(restartDir, 1,
              pressure,
              temperature,
              MolarFracs,
              velocity,
              T_p = temperature,
              Xi_p = MolarFracs,
              U_p = velocity)

