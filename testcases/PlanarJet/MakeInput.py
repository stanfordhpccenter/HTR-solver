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
config["Integrator"]["vorticityScale"] = (U_F-U_Ox)/h

# Rescale quantities
config["Grid"]["xWidth"] *= h
config["Grid"]["yWidth"] *= h
config["Grid"]["zWidth"] *= h
config["Grid"]["origin"][1] = -0.5*config["Grid"]["yWidth"]

##############################################################################
#                            Boundary conditions                             #
##############################################################################
assert config["BC"]["xBCLeft"]["type"] == "NSCBC_Inflow"
assert config["BC"]["xBCLeft"]["VelocityProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["VelocityProfile"]["FileDir"] = restartDir
assert config["BC"]["xBCLeft"]["TemperatureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["TemperatureProfile"]["FileDir"] = restartDir
config["BC"]["xBCLeft"]["P"] = PInf
assert config["BC"]["xBCLeft"]["MixtureProfile"]["type"] == "File"
config["BC"]["xBCLeft"]["MixtureProfile"]["FileDir"] = restartDir

assert config["BC"]["xBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["xBCRight"]["P"] = PInf

assert config["BC"]["yBCLeft"]["type"] == "NSCBC_Outflow"
config['BC']["yBCLeft"]["P"] = PInf

assert config["BC"]["yBCRight"]["type"] == "NSCBC_Outflow"
config["BC"]["yBCRight"]["P"] = PInf

##############################################################################
#                              Generate Grid                                 #
##############################################################################

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
                            False)

zGrid, dz = gridGen.GetGrid(config["Grid"]["origin"][2],
                            config["Grid"]["zWidth"],
                            config["Grid"]["zNum"], 
                            config["Grid"]["zType"],
                            1.0,
                            True)

# Set maxTime
config["Integrator"]["maxTime"] = config["Grid"]["xWidth"]/(0.5*(U_Ox+U_F))*FTT

with open("NoStats.json", 'w') as fout:
   json.dump(config, fout, indent=3)

#config["Integrator"]["maxTime"] = config["Grid"]["xWidth"]/UInf*FTT + FTTS*2*np.pi/config["BC"]["yBCLeftInflowProfile"]["omega"][0]
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

def profile(y, U1, U2, theta):
   return 0.5*(U1+U2) + 0.5*(U1-U2)*np.tanh(0.5*(y-0.5*h)/theta)

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
   MolarFracs        = np.ndarray(shape, dtype=np.dtype('(4,)f8'))
   velocity          = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dudtBoundary      = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dTdtBoundary      = np.ndarray(shape)
   pressure[:] = PInf
   for (k,kc) in enumerate(centerCoordinates):
      for (j,jc) in enumerate(kc):
         for (i,ic) in enumerate(jc):
            y = abs(yGrid[j+lo_bound[1]])
            u    = profile(y, U_F,  U_Ox, 0.05*h)
            X_F  = profile(y, 1.0, 1e-60, 0.05*h)
            X_Ox = 1.0-X_F
            centerCoordinates[k,j,i] = [xGrid[i+lo_bound[0]], yGrid[j+lo_bound[1]], zGrid[k+lo_bound[2]]]
            cellWidth        [k,j,i] = [   dx[i+lo_bound[0]],    dy[j+lo_bound[1]],    dz[k+lo_bound[2]]]
            temperature      [k,j,i] = TInf
            rho              [k,j,i] = 1.0
            MolarFracs       [k,j,i] = [X_F, X_Ox, 1e-60, 1e-60]
            velocity         [k,j,i] = [  u, 0.0, 0.0]
            dudtBoundary     [k,j,i] = [0.0, 0.0, 0.0]
            dTdtBoundary     [k,j,i] = 0.0

   with h5py.File(os.path.join(restartDir, filename), 'w') as fout:
      fout.attrs.create("SpeciesNames", ["CH4".encode(),
                                         "O2".encode(),
                                         "CO2".encode(),
                                         "H2O".encode()], dtype="S20")
      fout.attrs.create("timeStep", 0)
      fout.attrs.create("simTime", 0.0)
      fout.attrs.create("channelForcing", 0.0)

      fout.create_dataset("centerCoordinates",     shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("cellWidth",             shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("(4,)f8"))
      fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs_profile",    shape=shape, dtype = np.dtype("(4,)f8"))
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
      fout["velocity_old_NSCBC"][:] = velocity
      fout["temperature_old_NSCBC"][:] = temperature
      fout["MolarFracs_profile"][:] = MolarFracs
      fout["velocity_profile"][:] = velocity
      fout["temperature_profile"][:] = temperature

Parallel(n_jobs=args.np)(delayed(writeTile)(x, y, z) for x, y, z in np.ndindex((Ntiles[0], Ntiles[1], Ntiles[2])))

