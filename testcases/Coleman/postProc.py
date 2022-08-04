#!/usr/bin/env python3

import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import sys
import os
import h5py
import pandas

# load grid generator
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen

parser = argparse.ArgumentParser()
parser.add_argument("-json", "--json_file", type=argparse.FileType('r'))
parser.add_argument("-in", "--input_file")
#parser.add_argument("-in", "--input_file", type=argparse.FileType('r'))
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
#gamma = data["Flow"]["gamma"]

##############################################################################
#                               Compute Grid                                 #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

##############################################################################
#                          Load reference solution                           #
##############################################################################

data = pandas.read_csv("Coleman.dat")

##############################################################################
#                          Load average files                                #
##############################################################################

f = h5py.File(args.input_file, "r")

# Get the data
weight          = f["weight"][:][0,:]
pressure_avg    = f["pressure_avg"][:][0,:]
pressure_rms    = f["pressure_rms"][:][0,:]
temperature_avg = f["temperature_avg"][:][0,:]
temperature_rms = f["temperature_rms"][:][0,:]
MolarFracs_avg  = f["MolarFracs_avg"][:][0,:,:]
MolarFracs_rms  = f["MolarFracs_rms"][:][0,:,:]
velocity_avg    = f["velocity_avg"][:][0,:,:]
velocity_rms    = f["velocity_rms"][:][0,:,:]
velocity_rey    = f["velocity_rey"][:][0,:,:]

rho_avg  = f["rho_avg"][:][0,:]
rho_rms  = f["rho_rms"][:][0,:]
mu_avg   = f["mu_avg"][:][0,:]
mu_rms   = f["mu_rms"][:][0,:]
lam_avg  = f["lam_avg"][:][0,:]
lam_rms  = f["lam_rms"][:][0,:]
Di_avg   = f["Di_avg"][:][0,:,:]
Di_rms   = f["Di_rms"][:][0,:,:]
SoS_avg  = f["SoS_avg"][:][0,:]
SoS_rms  = f["SoS_rms"][:][0,:]

temperature_favg = f["temperature_favg"][:][0,:]
temperature_frms = f["temperature_frms"][:][0,:]
MolarFracs_favg  = f["MolarFracs_favg"][:][0,:,:]
MolarFracs_frms  = f["MolarFracs_frms"][:][0,:,:]
velocity_favg    = f["velocity_favg"][:][0,:,:]
velocity_frms    = f["velocity_frms"][:][0,:,:]
velocity_frey    = f["velocity_frey"][:][0,:,:]

rhoUUv   = f["rhoUUv"][:][0,:,:]
Up       = f["Up"][:][0,:,:]
tau      = f["tau"][:][0,:,:]
utau_y   = f["utau_y"][:][0,:,:]
tauGradU = f["tauGradU"][:][0,:,:]
pGradU   = f["pGradU"][:][0,:,:]

q = f["q"][:][0,:,:]


SpeciesNames = f.attrs.get("SpeciesNames")

# average both sides of the channel
weight       += weight[::-1]
pressure_avg += pressure_avg[::-1]
pressure_rms += pressure_rms[::-1]
temperature_avg += temperature_avg[::-1]
temperature_rms += temperature_rms[::-1]
for isp, sp in enumerate(SpeciesNames):
   MolarFracs_avg[:, isp] += MolarFracs_avg[::-1, isp]
   MolarFracs_rms[:, isp] += MolarFracs_rms[::-1, isp]
velocity_avg[:,0] += velocity_avg  [::-1,0]
velocity_avg[:,1] -= velocity_avg  [::-1,1]
velocity_avg[:,2] += velocity_avg  [::-1,2]
velocity_rms      += velocity_rms  [::-1]
velocity_rey[:,0] -= velocity_rey  [::-1,0]
velocity_rey[:,1] += velocity_rey  [::-1,1]
velocity_rey[:,2] -= velocity_rey  [::-1,2]

rho_avg += rho_avg [::-1]
rho_rms += rho_rms [::-1]
mu_avg  += mu_avg [::-1]
mu_rms  += mu_rms [::-1]
lam_avg += lam_avg [::-1]
lam_rms += lam_rms [::-1]
for isp, sp in enumerate(SpeciesNames):
   Di_avg[:, isp] += Di_avg [::-1, isp]
   Di_rms[:, isp] += Di_rms [::-1, isp]
SoS_avg += SoS_avg [::-1]
SoS_rms += SoS_rms [::-1]

temperature_favg += temperature_favg[::-1]
temperature_frms += temperature_frms[::-1]
for isp, sp in enumerate(SpeciesNames):
   MolarFracs_favg[:, isp] += MolarFracs_favg[::-1, isp]
   MolarFracs_frms[:, isp] += MolarFracs_frms[::-1, isp]
velocity_favg[:,0] += velocity_favg  [::-1,0]
velocity_favg[:,1] -= velocity_favg  [::-1,1]
velocity_favg[:,2] += velocity_favg  [::-1,2]
velocity_frms      += velocity_frms  [::-1]
velocity_frey[:,0] -= velocity_frey  [::-1,0]
velocity_frey[:,1] += velocity_frey  [::-1,1]
velocity_frey[:,2] -= velocity_frey  [::-1,2]

# Complete average process
pressure_avg         /= weight
pressure_rms         /= weight
temperature_avg      /= weight
temperature_rms      /= weight
for i in range(3):
   velocity_avg[:,i] /= weight
   velocity_rms[:,i] /= weight
   velocity_rey[:,i] /= weight

for isp, sp in enumerate(SpeciesNames):
   MolarFracs_avg[:,isp] /= weight
   MolarFracs_rms[:,isp] /= weight

pressure_rms    = np.sqrt(np.maximum(   pressure_rms -    pressure_avg**2, 0.0))
temperature_rms = np.sqrt(np.maximum(temperature_rms - temperature_avg**2, 0.0))
MolarFracs_rms  = np.sqrt(np.maximum( MolarFracs_rms -  MolarFracs_avg**2, 0.0))
#velocity_rms    = np.sqrt(np.maximum(   velocity_rms -    velocity_avg**2, 0.0))
velocity_rms    = velocity_rms -    velocity_avg**2

rho_avg  /= weight
rho_rms  /= weight
mu_avg  /= weight
mu_rms  /= weight
lam_avg /= weight
lam_rms /= weight
SoS_avg /= weight
SoS_rms /= weight
for isp, sp in enumerate(SpeciesNames):
   Di_avg[:,isp] /= weight
   Di_rms[:,isp] /= weight

mu_rms  = np.sqrt(np.maximum( mu_rms -  mu_avg**2, 0.0))
lam_rms = np.sqrt(np.maximum(lam_rms - lam_avg**2, 0.0))
Di_rms  = np.sqrt(np.maximum( Di_rms -  Di_avg**2, 0.0))
SoS_rms = np.sqrt(np.maximum(SoS_rms - SoS_avg**2, 0.0))

temperature_favg      /= weight
temperature_frms      /= weight
for i in range(3):
   velocity_favg[:,i] /= weight
   velocity_frms[:,i] /= weight
   velocity_frey[:,i] /= weight

for isp, sp in enumerate(SpeciesNames):
   MolarFracs_favg[:,isp] /= weight
   MolarFracs_frms[:,isp] /= weight

temperature_frms = np.sqrt(np.maximum(temperature_frms - temperature_favg**2, 0.0))
MolarFracs_frms  = np.sqrt(np.maximum( MolarFracs_frms -  MolarFracs_favg**2, 0.0))
#velocity_frms    = np.sqrt(np.maximum(   velocity_rms -    velocity_avg**2, 0.0))
velocity_frms    = velocity_frms -    velocity_favg**2

rhoU_avg = velocity_favg[:,0]

for i in range(3):
   rhoUUv[:,i]   /= 0.5*weight
   Up[:,i]       /= 0.5*weight
   utau_y[:,i]   /= 0.5*weight
   tauGradU[:,i] /= 0.5*weight
   pGradU[:,i]   /= 0.5*weight

for i in range(6):
   tau[:,i]      /= 0.5*weight

for i in range(3):
   q[:,i] /= 0.5*weight

# sanity check
assert weight.shape[0] == yGrid.size

##############################################################################
#                           Print quantities                                 #
##############################################################################

rhoW = rho_avg[0]
muW = mu_avg[0]
tauW = tau[0,1]
uTau = np.sqrt(tauW/rhoW)
ReTau = rhoW*uTau*yWidth*0.5/muW

uStar = np.sqrt(tauW/rho_avg)

yPlusGrid = yGrid*uTau *rhoW   /muW
yStarGrid = yGrid*uStar*rho_avg/mu_avg

rhoBulk = 0.0
rhoUBulk = 0.0
for i in range(yNum+1):
   rhoBulk  += 0.5*( rho_avg[i+1] +  rho_avg[i])*(yGrid[i+1] - yGrid[i])
   rhoUBulk += 0.5*(rhoU_avg[i+1] + rhoU_avg[i])*(yGrid[i+1] - yGrid[i])
rhoBulk  /= yGrid[yNum+1] - yGrid[0]
rhoUBulk /= yGrid[yNum+1] - yGrid[0]
UBulk = rhoUBulk/rhoBulk
print(UBulk)

print("Re_tau = ", ReTau)
print("rho_Bulk  = ",  rhoBulk)
print("rho_Bulk  = ",  rhoBulk/(101325/(287.15*300.0)))
print("Re_Bulk  = ",  rhoUBulk*yWidth*0.5/muW)

print("rho_cl/rho_w = ",        rho_avg[int(yNum*0.5)]/        rho_avg[0])
print("T_cl/T_w = ",    temperature_avg[int(yNum*0.5)]/temperature_avg[0])
print("mu_cl/mu_w = ",           mu_avg[int(yNum*0.5)]/         mu_avg[0])
print("Bq = ", q[0,1]/(rhoW*uTau*3.5*287.15*temperature_avg[0]))

##############################################################################
#                               Plot stuff                                   #
##############################################################################

figureDir = "Figures"
if not os.path.exists(figureDir):
   os.makedirs(figureDir)

yovh = yGrid[:]/config["Grid"]["GridInput"]["width"][1]*2

# Mean profiles
plt.figure(1)
Umy, = plt.plot(                 yovh,               velocity_avg[:,0]/UBulk, "k")
Tmy, = plt.plot(                 yovh, temperature_avg[:]/temperature_avg[0], "r")
Rmy, = plt.plot(                 yovh,         rho_avg[:]/           rhoBulk, "b")
UCo, = plt.plot(data["y"][:].values+1,   data["<u>"][:].values, "ko", markersize=5)
TCo, = plt.plot(data["y"][:].values+1,   data["<T>"][:].values, "ro", markersize=5)
RCo, = plt.plot(data["y"][:].values+1, data["<rho>"][:].values, "bo", markersize=5)
plt.xlabel(r"$y/h$")
#plt.ylabel(r"$\overline{u}/u_B$")
plt.gca().set_xlim(0, 1)
plt.legend([(Umy, UCo), (Tmy, TCo), (Rmy, RCo)],
           [r"$\overline{u}/ u_b$", r"$\overline{T}/T_w$", r"$\overline{\rho}/\rho_b$"],
           handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
plt.savefig(os.path.join(figureDir, "u.eps"), bbox_inches='tight')

# Fluctuations profiles
plt.figure(2)
UUmy, = plt.plot(                 yovh, velocity_rms[:,0]/UBulk**2,  "k")
VVmy, = plt.plot(                 yovh, velocity_rms[:,1]/UBulk**2,  "r")
WWmy, = plt.plot(                 yovh, velocity_rms[:,2]/UBulk**2,  "b")
UUCo, = plt.plot(data["y"][:].values+1,   data["<u'u'>"][:].values, "ko", markersize=5)
VVCo, = plt.plot(data["y"][:].values+1,   data["<v'v'>"][:].values, "ro", markersize=5)
WWCo, = plt.plot(data["y"][:].values+1,   data["<w'w'>"][:].values, "bo", markersize=5)
plt.xlabel(r"$y/h$")
###plt.ylabel(r"$\overline{\rho u'' u''}/u_B^2$")
plt.gca().set_xlim(0, 1)
plt.legend([(UUmy, UUCo), (VVmy, VVCo), (WWmy, WWCo)],
           [r"$\overline{u' u'}/ u_b^2$", r"$\overline{v' v'}/u_b^2$", r"$\overline{w' w'}/u_b^2$"],
           handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
plt.savefig(os.path.join(figureDir, "up.eps"), bbox_inches='tight')

plt.show()
