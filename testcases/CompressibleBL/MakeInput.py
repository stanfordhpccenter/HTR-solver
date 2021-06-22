#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import h5py
from random import *
import pandas

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix

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
 
aInf = np.sqrt(gamma*R*TInf)
MaInf = UInf/aInf

RhoInf = ConstPropMix.GetDensity(TInf, P, config)
muInf = ConstPropMix.GetViscosity(TInf, config)
nuInf = muInf/RhoInf

##############################################################################
#                              Generate Grid                                 #
##############################################################################

xGrid, dx = gridGen.GetGrid(config["Grid"]["origin"][0],
                            config["Grid"]["xWidth"],
                            config["Grid"]["xNum"], 
                            config["Grid"]["xType"],
                            config["Grid"]["xStretching"],
                            False)

yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                            config["Grid"]["yWidth"],
                            config["Grid"]["yNum"], 
                            config["Grid"]["yType"],
                            config["Grid"]["yStretching"],
                            False,
                            StagMinus=True)

zGrid, dz = gridGen.GetGrid(config["Grid"]["origin"][2],
                            config["Grid"]["zWidth"],
                            config["Grid"]["zNum"], 
                            config["Grid"]["zType"],
                            config["Grid"]["zStretching"],
                            True)

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
for i in range(Np):
   if (uB[i] > 0.99):
      delta = yB[i]
      break
config["Integrator"]["EulerScheme"]["vorticityScale"] = UInf/delta

uB *= UInf
vB *= UInf
TB *= TInf

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
if not os.path.exists(restartDir):
   os.makedirs(restartDir)

for xt in range(0, Ntiles[0]):
   for yt in range(0, Ntiles[1]):
      for zt in range(0, Ntiles[2]):
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
         pressure[:] = P
         dudtBoundary[:] = [0.0, 0.0, 0.0]
         dTdtBoundary[:] = 0.0
         for (k,kc) in enumerate(centerCoordinates):
            for (j,jc) in enumerate(kc):
               for (i,ic) in enumerate(jc):
                  Re = (xGrid[i+lo_bound[0]]+x0)*UInf/nuInf
                  yB1 = yB*np.sqrt(Re/Rex0)
                  vB1 = vB/np.sqrt(Re/Rex0)

                  u = np.interp(yGrid[j+lo_bound[1]], yB1, uB)
                  v = np.interp(yGrid[j+lo_bound[1]], yB1, vB1)
                  T = np.interp(yGrid[j+lo_bound[1]], yB1, TB)

                  centerCoordinates[k,j,i] = [xGrid[i+lo_bound[0]], yGrid[j+lo_bound[1]], zGrid[k+lo_bound[2]]]
                  cellWidth        [k,j,i] = [   dx[i+lo_bound[0]],    dy[j+lo_bound[1]],    dz[k+lo_bound[2]]]
                  temperature      [k,j,i] = T
                  rho              [k,j,i] = P/(R*temperature[k,j,i])
                  MolarFracs       [k,j,i] = [1.0,]
                  velocity         [k,j,i] = [ u, v, 0.0]

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

