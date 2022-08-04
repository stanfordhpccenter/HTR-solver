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
Ma_F   = config["Case"]["Ma_F"]
Ma_Ox  = config["Case"]["Ma_Ox"]
TInf   = config["Case"]["TInf"]
PInf   = config["Case"]["PInf"]
FTT    = config["Case"]["FlowThroughTimesNoStat"]
#FTTS   = config["Case"]["FlowThroughTimesStat"]

gamma_Ox = 1.4
gamma_F  = 1.32

Rgas = 8.3144598
W_Ox = 2*15.9994e-3
W_F  = 4*1.00784e-3+12.0107e-3

# Simulation setup
assert config["Flow"]["turbForcing"]["type"] == "OFF"
assert config["Flow"]["initCase"]["type"] == "Restart"
restartDir = "InitialCase"
config["Flow"]["initCase"]["restartDir"] = restartDir

# Oxygen-stream mixture properties
c_Ox  = np.sqrt(gamma_Ox*Rgas/W_Ox*TInf)
c_F   = np.sqrt(gamma_F *Rgas/W_F *TInf)

mu_F   = 1.88e-5
rho_F  = PInf/(Rgas/W_F*TInf)
mu_Ox  = 1.95e-5
rho_Ox = PInf/(Rgas/W_Ox*TInf)

U_F  = c_F *Ma_F
U_Ox = c_Ox*Ma_Ox

# Inlet displacement thickness
h = mu_Ox*ReIn/((U_F-U_Ox)*rho_Ox)

# Rescale quantities
U_F  *= np.sqrt(rho_Ox/PInf)
U_Ox *= np.sqrt(rho_Ox/PInf)
config["Flow"]["mixture"]["LRef"] = h
config["Flow"]["mixture"]["PRef"] = PInf
config["Flow"]["mixture"]["TRef"] = TInf
config["Flow"]["mixture"]["XiRef"] = {"Species" : [{"Name" : "O2", "MolarFrac" : 1.0 }]}
config["Integrator"]["EulerScheme"]["vorticityScale"] = (U_F-U_Ox)/1.0
#config["Grid"]["GridInput"]["width"][0] *= h
#config["Grid"]["GridInput"]["width"][1] *= h
#config["Grid"]["GridInput"]["width"][2] *= h
config["Grid"]["GridInput"]["origin"][1] = -0.5*config["Grid"]["GridInput"]["width"][1]

##############################################################################
#                            Boundary conditions                             #
##############################################################################
assert config["BC"]["xBCLeft"]["type"] == "NSCBC_Inflow"
assert config["BC"]["xBCLeft"]["VelocityProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["VelocityProfile"]["FileDir"] = restartDir
assert config["BC"]["xBCLeft"]["TemperatureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["TemperatureProfile"]["FileDir"] = restartDir
config["BC"]["xBCLeft"]["P"] = 1.0
assert config["BC"]["xBCLeft"]["MixtureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["MixtureProfile"]["FileDir"] = restartDir

assert config["BC"]["xBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["xBCRight"]["P"] = 1.0

assert config["BC"]["yBCLeft"]["type"] == "NSCBC_Outflow"
config['BC']["yBCLeft"]["P"] = 1.0

assert config["BC"]["yBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["yBCRight"]["P"] = 1.0

##############################################################################
#                              Generate Grid                                 #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

# Set maxTime
config["Integrator"]["maxTime"] = config["Grid"]["GridInput"]["width"][0]/(0.5*(U_Ox+U_F))*FTT

with open("NoStats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

#config["Integrator"]["maxTime"] = config["Grid"]["GridInput"]["width"][0]/UInf*FTT + FTTS*2*np.pi/config["BC"]["yBCLeftInflowProfile"]["omega"][0]
#
## Setup averages
#idx1 = (np.abs(xGrid/deltaStarIn -  400.0)).argmin()
#idx2 = (np.abs(xGrid/deltaStarIn -  650.0)).argmin()
#idx3 = (np.abs(xGrid/deltaStarIn -  800.0)).argmin()
#idx4 = (np.abs(xGrid/deltaStarIn -  950.0)).argmin()
#config["IO"]["YZAverages"] = [{"fromCell" : [0, 0, 0],          "uptoCell" : [config["Grid"]["xNum"]+1, 0, config["Grid"]["zNum"]]}]
#config["IO"]["XZAverages"] = [{"fromCell" : [int(idx1)-1, 0, 0], "uptoCell" : [int(idx1)+1, config["Grid"]["yNum"]+1, config["Grid"]["zNum"]]},
#                              {"fromCell" : [int(idx2)-1, 0, 0], "uptoCell" : [int(idx2)+1, config["Grid"]["yNum"]+1, config["Grid"]["zNum"]]},
#                              {"fromCell" : [int(idx3)-1, 0, 0], "uptoCell" : [int(idx3)+1, config["Grid"]["yNum"]+1, config["Grid"]["zNum"]]},
#                              {"fromCell" : [int(idx4)-1, 0, 0], "uptoCell" : [int(idx4)+1, config["Grid"]["yNum"]+1, config["Grid"]["zNum"]]}]
#
#with open("Stats.json", 'w') as fout:
#   json.dump(config, fout, indent=3)

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
def profile(y, U1, U2, ell, theta):
   theta *= ell
   return 0.5*(U1+U2) + 0.5*(U1-U2)*np.tanh(0.5*(y-0.5*ell)/theta)

def pressure(i, j, k):
   return 1.0

def temperature(i, j, k):
   return 1.0

def MolarFracs(i, j, k):
   X_F  = profile(abs(yGrid[j]), 1.0, 1e-60, 1.0, 0.05)
   X_Ox = 1.0-X_F
   return [X_F, X_Ox, 1e-60, 1e-60]

def velocity(i, j, k):
   return [profile(abs(yGrid[j]), U_F,  U_Ox, 1.0, 0.05), 0.0, 0.0]

restart = HTRrestart.HTRrestart(config)
restart.write(restartDir, 4,
              pressure,
              temperature,
              MolarFracs,
              velocity,
              T_p = temperature,
              Xi_p = MolarFracs,
              U_p = velocity,
              nproc = args.np)

