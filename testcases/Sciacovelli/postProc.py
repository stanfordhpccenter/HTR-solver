#!/usr/bin/env python3

import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.legend_handler import HandlerTuple
import sys
import os
import h5py
import pandas

# load grid generator
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import Averages

parser = argparse.ArgumentParser()
parser.add_argument("-json", "--json_file", type=argparse.FileType('r'))
parser.add_argument("-in", "--input_file")
args = parser.parse_args()

##############################################################################
#                              Read HTR Input File                           #
##############################################################################
data = json.load(args.json_file)

xNum = data["Grid"]["xNum"]
yNum = data["Grid"]["yNum"]
zNum = data["Grid"]["zNum"]
xWidth  = data["Grid"]["GridInput"]["width"][0]
yWidth  = data["Grid"]["GridInput"]["width"][1]
zWidth  = data["Grid"]["GridInput"]["width"][2]
xOrigin = data["Grid"]["GridInput"]["origin"][0]
yOrigin = data["Grid"]["GridInput"]["origin"][1]
zOrigin = data["Grid"]["GridInput"]["origin"][2]

##############################################################################
#                               Compute Grid                                 #
##############################################################################
yGrid, dy = gridGen.GetGrid(data["Grid"]["GridInput"]["origin"][1],
                            data["Grid"]["GridInput"]["width"][1],
                            data["Grid"]["yNum"],
                            data["Grid"]["GridInput"]["yType"],
                            False,
                            StagMinus=True)

# Correct boundaries that are staggered
yGrid[     0] += 0.5*dy[0]
yGrid[yNum+1] -= 0.5*dy[yNum+1]

##############################################################################
#                          Load reference solution                           #
##############################################################################
sciacoEtAl = pandas.read_csv("sciacoEtAl.dat")

##############################################################################
#                          Load average files                                #
##############################################################################
avg = Averages.avg2D(args.input_file, True)

##############################################################################
#                           Print quantities                                 #
##############################################################################
rhoU_avg = avg.velocity_favg[:,0]

rhoW = avg.rho_avg[0]
muW = avg.mu_avg[0]
tauW = avg.tau[0,1]
uTau = np.sqrt(tauW/rhoW)
ReTau = rhoW*uTau*yWidth*0.5/muW

uStar = np.sqrt(tauW/avg.rho_avg)

yPlusGrid = yGrid*uTau *rhoW   /muW
yStarGrid = yGrid*uStar*avg.rho_avg/avg.mu_avg

rhoBulk = 0.0
rhoUBulk = 0.0
for i in range(yNum+1):
   rhoBulk  += 0.5*(avg.rho_avg[i+1] + avg.rho_avg[i])*(yGrid[i+1] - yGrid[i])
   rhoUBulk += 0.5*(   rhoU_avg[i+1] +    rhoU_avg[i])*(yGrid[i+1] - yGrid[i])
rhoBulk  /= yGrid[yNum+1] - yGrid[0]
rhoUBulk /= yGrid[yNum+1] - yGrid[0]
UBulk = rhoUBulk/rhoBulk

print("Re_tau = ", ReTau)
print("rho_Bulk  = ",  rhoBulk)
print("rho_Bulk  = ",  rhoBulk/(101325/(287.15*300.0)))
print("Re_Bulk  = ",  rhoUBulk*yWidth*0.5/muW)
print("Ma_Bulk  = ",  UBulk/avg.SoS_avg[0])
print("rho_cl/rho_w = ",     avg.rho_avg[int(yNum*0.5)]/        avg.rho_avg[0])
print("T_cl/T_w = ", avg.temperature_avg[int(yNum*0.5)]/avg.temperature_avg[0])
print("mu_cl/mu_w = ",        avg.mu_avg[int(yNum*0.5)]/         avg.mu_avg[0])
print("Bq = ", avg.q[0,1]/(rhoW*uTau*3.5*287.15*avg.temperature_avg[0]))

##############################################################################
#                               Plot stuff                                   #
##############################################################################

figureDir = "Figures"
if not os.path.exists(figureDir):
   os.makedirs(figureDir)

plt.figure(0)
Umy, = plt.plot(2.0*yGrid/data["Grid"]["yWidth"],        avg.velocity_avg[:,0]/rhoUBulk*rhoBulk, "k")
Rmy, = plt.plot(2.0*yGrid/data["Grid"]["yWidth"],                 avg.rho_avg[:]/avg.rho_avg[0], "r")
Tmy, = plt.plot(2.0*yGrid/data["Grid"]["yWidth"], avg.temperature_avg[:]/avg.temperature_avg[0], "b")
URe, = plt.plot(     sciacoEtAl["y"][::6].values,                  sciacoEtAl["ub"][::6].values, "ko", markersize=6)
RRe, = plt.plot(     sciacoEtAl["y"][::6].values,                sciacoEtAl["rhow"][::6].values, "ro", markersize=6)
TRe, = plt.plot(     sciacoEtAl["y"][::6].values,                  sciacoEtAl["Tw"][::6].values, "bo", markersize=6)
plt.xlabel(r"$y/h$")
plt.gca().set_xlim(0, 1)
plt.legend([(Umy, URe), (Tmy, TRe), (Rmy, RRe)],
           [r"$\overline{u}/ u_b$", r"$\overline{T}/T_w$", r"$\overline{\rho}/\rho_b$"],
           handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0,
           loc="center right", bbox_to_anchor=(1.0,0.75))
plt.savefig(os.path.join(figureDir, "AvgComp.eps"), bbox_inches='tight')

plt.figure(1)
UUmy, = plt.plot(2.0*yGrid/data["Grid"]["yWidth"],  avg.velocity_rms[:,0]/UBulk**2, "k")
VVmy, = plt.plot(2.0*yGrid/data["Grid"]["yWidth"],  avg.velocity_rms[:,1]/UBulk**2, "r")
WWmy, = plt.plot(2.0*yGrid/data["Grid"]["yWidth"],  avg.velocity_rms[:,2]/UBulk**2, "b")
UURe, = plt.plot(     sciacoEtAl["y"][::6].values, sciacoEtAl["upupb"][::6].values, "ko", markersize=6)
VVRe, = plt.plot(     sciacoEtAl["y"][::6].values, sciacoEtAl["vpvpb"][::6].values, "ro", markersize=6)
WWRe, = plt.plot(     sciacoEtAl["y"][::6].values, sciacoEtAl["wpwpb"][::6].values, "bo", markersize=6)
plt.xlabel(r"$y/h$")
plt.gca().set_xlim(0, 1)
plt.legend([(UUmy, UURe), (VVmy, VVRe), (WWmy, WWRe)],
           [r"$\overline{u' u'}/ u_b^2$", r"$\overline{v' v'}/u_b^2$", r"$\overline{w' w'}/u_b^2$"],
           handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
plt.savefig(os.path.join(figureDir, "rmsComp.eps"), bbox_inches='tight')

plt.show()
