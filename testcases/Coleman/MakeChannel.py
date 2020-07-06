#!/usr/bin/env python2

import argparse
import json
import sys
import os
import numpy as np
from scipy.optimize import fsolve

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix

parser = argparse.ArgumentParser()
parser.add_argument("base_json", type=argparse.FileType("r"), default="base.json")
args = parser.parse_args()

# Read base config
config = json.load(args.base_json)
ReB = config["Case"]["ReB"]
MaB = config["Case"]["MaB"]
ReTau = config["Case"]["Retau"]
dTplus     = config["Case"]["DeltaT"]
dTplusStat = config["Case"]["DeltaTStat"]
del config["Case"]

# Read boundary conditions
assert config["BC"]["yBCLeft"]["type"] == "IsothermalWall"
assert config["BC"]["yBCLeft"]["TemperatureProfile"]["type"] == "Constant"
assert config["BC"]["yBCRight"]["type"] == "IsothermalWall"
assert config["BC"]["yBCRight"]["TemperatureProfile"]["type"] == "Constant"
assert config["BC"]["yBCLeft"]["TemperatureProfile"]["temperature"] == config["BC"]["yBCRight"]["TemperatureProfile"]["temperature"]
Tw = config["BC"]["yBCLeft"]["TemperatureProfile"]["temperature"]

# Read properties
Pb              = config["Flow"]["initParams"][0]
Tb              = config["Flow"]["initParams"][1]
gamma           = config["Flow"]["gamma"]
R               = config["Flow"]["gasConstant"]
assert config["Flow"]["turbForcing"]["type"] == "CHANNEL"
assert config["Flow"]["initCase"] == "ChannelFlow"
assert Tw == Tb

cW = np.sqrt(gamma*R*Tw)
muW = ConstPropMix.GetViscosity(Tw, config)

uB = cW*MaB
rhoB = Pb/(R*Tb)

h = ReB*muW/(rhoB*uB)
print("h = ", h)

rhoW = rhoB
uTau = ReTau*muW/(rhoW*h)
deltaNu = muW/(uTau*rhoW)
TauW = uTau**2*rhoB

yPlusTrg = 0.8

def objective(yStretching):
   yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                               2.0*h,
                               config["Grid"]["yNum"], 
                               config["Grid"]["yType"],
                               yStretching,
                               False)
   return dy[1]/deltaNu - yPlusTrg
   #return (yGrid[1] - config["Grid"]["origin"][1])/deltaNu - yPlusTrg

yStretching, = fsolve(objective, 1.0)

tNu = deltaNu**2*rhoW/muW

# Grid section
config["Grid"]["xWidth"] = 4.0*h*np.pi
config["Grid"]["yWidth"] = 2.0*h
config["Grid"]["zWidth"] = 2.0*h*np.pi
config["Grid"]["yStretching"] = yStretching

# Flow section
config["Flow"]["initParams"][2] = uB
config["Flow"]["turbForcing"]["RhoUbulk"] = rhoB*uB
config["Flow"]["turbForcing"]["Forcing"] = TauW/h

config["Integrator"]["maxTime"] = tNu*dTplus

with open("ChannelFlow.json", "w") as fout:
   json.dump(config, fout, indent=3)

config["Integrator"]["maxTime"] = tNu*(dTplus+dTplusStat)

config["Flow"]["initCase"] == "Restart"

config["IO"]["XZAverages"] = [{"fromCell" : [0, 0, 0], "uptoCell" : [1024, 1024, 1024]}]

with open("ChannelFlowStats.json", "w") as fout:
   json.dump(config, fout, indent=3)
