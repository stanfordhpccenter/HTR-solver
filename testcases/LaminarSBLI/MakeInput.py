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
TInf = config["BC"]["xBCLeft"]["TemperatureProfile"]["temperature"]
Tw   = config["BC"]["xBCLeft"]["TemperatureProfile"]["temperature"]
P    = config["BC"]["xBCLeft"]["P"]
UInf = config["BC"]["xBCLeft"]["VelocityProfile"]["velocity"]
Rex0 = config["BC"]["xBCLeft"]["VelocityProfile"]["Reynolds"]

assert config["BC"]["yBCRight"]["type"] == "IncomingShock"

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

# Get VorticityScale
delta = 0.0
for i in range(etaB.size):
   if (uB[i] > 0.99):
      delta = yB[i]
      break
config["Integrator"]["EulerScheme"]["vorticityScale"] = UInf/delta

uB *= UInf
vB *= UInf
TB *= TInf

##############################################################################
#            Setup IncomingShock boundary condition parameters               #
##############################################################################

xImp = config["Grid"]["GridInput"]["origin"][0] + config["Grid"]["GridInput"]["width"][0]*0.5
xTop = xImp - config["Grid"]["GridInput"]["width"][1]/np.tan(config["BC"]["yBCRight"]["beta"]*np.pi/180)

assert (config["BC"]["yBCRight"]["iShock"] == np.abs(xGrid - xTop).argmin())

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

