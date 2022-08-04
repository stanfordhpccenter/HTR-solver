#!/usr/bin/env python2

import os
import sys
import h5py
import json
import pandas
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import fsolve

# load grid generator
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import ConstPropMix
import Averages

parser = argparse.ArgumentParser()
parser.add_argument("-json", "--json_file", type=argparse.FileType('r'))
parser.add_argument("-in", "--input_dir")
args = parser.parse_args()

##############################################################################
#                            Read HTR Input File                             #
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
delta99In = muInf*ReIn/(UInf*rhoInf)

# Wall properties
Tw = TInf*TwOvT
muW = ConstPropMix.GetViscosity(Tw, config)
rhoW = ConstPropMix.GetDensity(Tw, PInf, config)

r = Pr**(1.0/3.0)
Taw = TInf*(1.0 + r*0.5*(gamma - 1.0)*MaInf**2)

##############################################################################
#                          Load reference solution                           #
##############################################################################

Ref = pandas.read_csv("RefSolution.dat", delim_whitespace=True)

##############################################################################
#                          Load average files                                #
##############################################################################

avg = Averages.avg1D(args.input_dir+"/ZAverages/", 0)

##############################################################################
#                  Reference Skin friction coefficient                       #
##############################################################################

# Skin Friction coefficient
Cf0 = 2.0*avg.tau[0,0,1]/(rhoInf*UInf**2)
def getX0(Cf0):
   def VanDriestII(x0):
      Rexv = rhoInf*UInf*(avg.centerCoordinates[0,0,0] + x0)/muInf

      a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
      b = (Taw/Tw - 1.0)

      A = (2*a**2 - b)/np.sqrt(b**2 + 4*a**2)
      B =           b /np.sqrt(b**2 + 4*a**2)

      res = (np.arcsin(A) + np.arcsin(B))/np.sqrt(Cf0*(Taw/TInf - 1.0))
      res-= 4.15*np.log10(Rexv*Cf0*muInf/muW)
      res-= 1.7
      return res
   x0 = fsolve(VanDriestII, 1.0, xtol=1e-10)
   return x0[0]
x0 = getX0(Cf0)

def getCfTurb(xGrid):
   cf = np.zeros(xGrid.size)
   for i,x in enumerate(cf):
      def VanDriestII(Cf):
         Rexv = (avg.centerCoordinates[0,i,0] + x0)*ReIn

         a = np.sqrt(r*0.5*(gamma - 1.0)*MaInf**2*TInf/Tw)
         b = (Taw/Tw - 1.0)

         A = (2*a**2 - b)/np.sqrt(b**2 + 4*a**2)
         B =           b /np.sqrt(b**2 + 4*a**2)

         res = (np.arcsin(A) + np.arcsin(B))/np.sqrt(Cf*(Taw/TInf - 1.0))
         res-= 4.15*np.log10(Rexv*Cf*muInf/muW)
         res-= 1.7
         return res
      cf[i] = fsolve(VanDriestII, 1e-4, xtol=1e-10)
   return cf

cfTurb = getCfTurb(avg.centerCoordinates[0,:,0])

##############################################################################
#                     Compute boundary layer thickness                       #
##############################################################################

nx = len(avg.centerCoordinates[0,:,0])
ny = len(avg.centerCoordinates[:,0,0])
delta99 = np.zeros(nx)
deltaS  = np.zeros(nx)
theta   = np.zeros(nx)
ufav = avg.velocity_favg[:,:,0]/avg.rho_avg[:,:]

for i in range(nx):
   rhoUNorm = avg.velocity_favg[:,i,0]/avg.velocity_favg[-1,i,0]
   UNorm = ufav[:,i]/ufav[-1,i]

   # delta99
#   u99 = avg.velocity_avg[-1,i,0]*0.99
   u99 = ufav[-1,i]*0.99
   j99 = 0
   for j in range(ny):
      #if (avg.velocity_avg[j,i,0] > u99):
      if (ufav[j,i] > u99):
         j99 = j
         break
   dy = avg.centerCoordinates[j99,i,1] - avg.centerCoordinates[j99-1,i,1]
   #du = avg.velocity_avg[j99,i,0] - avg.velocity_avg[j99-1,i,0]
   du = ufav[j99,i] - ufav[j99-1,i]
   delta99[i] = (avg.centerCoordinates[j99,i,1] - 
                 (ufav[j99,i] - u99)*dy/du)
                 #(avg.velocity_avg[j99,i,0] - u99)*dy/du)
   # deltaS and theta
   for j in range(1, ny):
      dy = avg.centerCoordinates[j,i,1] - avg.centerCoordinates[j-1,i,1]
      deltaS[i] += 0.5*dy*((1.0 - rhoUNorm[j  ])+
                           (1.0 - rhoUNorm[j-1]))
      theta[i]  += 0.5*dy*(rhoUNorm[j  ]*(1.0 - UNorm[j  ])+
                           rhoUNorm[j-1]*(1.0 - UNorm[j-1]))

uTau     = np.sqrt(avg.tau[0,:,1]/avg.rho_avg[0,:])
cf       = 2.0*avg.tau[0,:,1]/(rhoInf*UInf**2)
ReTau    = avg.rho_avg[0,:]*uTau*delta99/avg.mu_avg[0,:]
ReDelta  = avg.rho_avg[-1,:]*ufav[-1,:]*delta99/avg.mu_avg[-1,:]
ReTheta  = avg.rho_avg[-1,:]*ufav[-1,:]*  theta/avg.mu_avg[-1,:]
ReDelta2 = avg.rho_avg[-1,:]*ufav[-1,:]*  theta/avg.mu_avg[ 0,:]

iReTau200 = (np.abs(ReTau - 204.8)).argmin()

##############################################################################
#                   Print values at the reference location                   #
##############################################################################

def printSection(xR):
   iR = (np.abs(avg.centerCoordinates[0,:,0] - xR)).argmin()
   print("##############################################")
   print("Sampling location")
   print(" > x = {}".format(avg.centerCoordinates[0,iR,0]))
   print(" > \delta/\delta_0 = {}".format(delta99[iR]))
   print(" > Re_tau = {}".format(ReTau[iR]))
   print(" > Re_delta = {}".format(ReDelta[iR]))
   print(" > Re_theta = {}".format(ReTheta[iR]))
   print(" > Re_delta2 = {}".format(ReDelta2[iR]))
   print(" > H = {}".format(deltaS[iR]/theta[iR]))
   print(" > Cf = {}".format(cf[iR]))
   print(" > M_tau = {}".format(uTau[iR]/avg.SoS_avg[0,iR]))
   print("##############################################")

printSection(avg.centerCoordinates[0,iReTau200,0])
printSection(53.00)
printSection(87.45)

##############################################################################
#                               Plot stuff                                   #
##############################################################################

figureDir = "Figures"
if not os.path.exists(figureDir):
   os.makedirs(figureDir)

##############################
# Cf
##############################
iFig = 0
plt.figure(iFig); iFig +=1
plt.plot(avg.centerCoordinates[0,:,0], 2.0*avg.tau[0,:,1]/(rhoInf*UInf**2),   "k", label="Present formulation")
plt.plot(avg.centerCoordinates[0,:,0],                              cfTurb, "-.k", label="Van Driest II")
plt.xlabel(r"$(x-x_0)/\delta_0$", fontsize = 20)
plt.ylabel(r"$\overline{C_f}$"  , fontsize = 20)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
plt.legend()
plt.savefig(os.path.join(figureDir, "Cf.eps"), bbox_inches='tight')

##############################
# ReTau
##############################
plt.figure(iFig); iFig += 1
plt.plot(avg.centerCoordinates[0,:,0], ReTau)
plt.xlabel(r"$(x-x_0)/\delta_0$", fontsize = 20)
plt.ylabel(r"$Re_{\tau}$ "      , fontsize = 20)
plt.savefig(os.path.join(figureDir, "ReTau.eps"), bbox_inches='tight')

##############################
# Delta99
##############################
plt.figure(iFig); iFig += 1
plt.plot(avg.centerCoordinates[0,:,0], delta99)
plt.xlabel(r"$(x-x_0)/\delta_0$", fontsize = 20)
plt.ylabel(r"$\delta/\delta_0$" , fontsize = 20)
plt.savefig(os.path.join(figureDir, "delta99.eps"), bbox_inches='tight')

##############################
# sqrt(\rho/\rho_w)
##############################
plt.figure(iFig); iFig += 1
plt.plot(avg.centerCoordinates[:,iReTau200,1]/delta99[iReTau200], np.sqrt(avg.rho_avg[:,iReTau200]/avg.rho_avg[0,iReTau200]), "-k", label="HTR_solver")
plt.plot(Ref["y/delta99"], Ref["sqrt(rho/rho_w)"], "-.k", label="Pirozzoli and Bernardini (2011)")
plt.xlabel(r"$y/\delta$"           , fontsize = 20)
plt.ylabel(r"$\sqrt{\rho/\rho_w}$" , fontsize = 20)
plt.legend()
plt.savefig(os.path.join(figureDir, "sqrtRho.eps"), bbox_inches='tight')

##############################
# Velocity profile
##############################
plt.figure(iFig); iFig += 1
yPlus = avg.centerCoordinates[:,iReTau200,1]*uTau[iReTau200]*avg.rho_avg[0,iReTau200]/avg.mu_avg[0,iReTau200]
uPlus = avg.velocity_avg[:,iReTau200,0]/uTau[iReTau200]
rhoNorm = avg.rho_avg[:,iReTau200]/avg.rho_avg[0,iReTau200]
uVD = np.zeros(ny)
for j, c in enumerate(uVD):
   for k in range(j):
      uVD[j] += 0.5*(np.sqrt(rhoNorm[k+1]) + np.sqrt(rhoNorm[k]))*(uPlus[k+1] - uPlus[k])
plt.semilogx(yPlus, uVD, "-k", label="HTR solver")

# log law
logLaw = np.zeros(ny)
for i in range(ny): logLaw[i] = 1/0.41*np.log(max(yPlus[i],1e-30)) + 5.2
plt.semilogx(yPlus,  yPlus, "--k", label="")
plt.semilogx(yPlus, logLaw, "--k", label="Log law")

# Reference solution
plt.semilogx(Ref["y+"], Ref["u_vd+"], "-.k", label="Pirozzoli and Bernardini (2011)")

plt.xlabel(r"$y^+$"   , fontsize = 20)
plt.ylabel(r"$u_{vd}$", fontsize = 20)
plt.gca().set_xlim(1.0, 1e3)
plt.gca().set_ylim(0.0,  25)
plt.legend()
plt.savefig(os.path.join(figureDir, "uProfilesReTau200.eps"), bbox_inches='tight')

##############################
# Velocity RMS profiles
##############################
plt.figure(iFig); iFig += 1
yPlus = avg.centerCoordinates[:,iReTau200,1]*uTau[iReTau200]*avg.rho_avg[0,iReTau200]/avg.mu_avg[0,iReTau200]
urms = np.sqrt(avg.velocity_rms[:,iReTau200,0])/uTau[iReTau200]
vrms = np.sqrt(avg.velocity_rms[:,iReTau200,1])/uTau[iReTau200]
uvrms = avg.velocity_rey[:,iReTau200,0]/uTau[iReTau200]**2
uu1, = plt.semilogx(yPlus,  urms, "-k")
vv1, = plt.semilogx(yPlus,  vrms, "-r")
uv1, = plt.semilogx(yPlus, uvrms, "-b")

# Reference solution
uu2, = plt.semilogx(Ref["y+"], Ref["urms+"], "-.k")
vv2, = plt.semilogx(Ref["y+"], Ref["vrms+"], "-.r")
uv2, = plt.semilogx(Ref["y+"], Ref["uv+"],   "-.b")

plt.xlabel(r"$y^+$"   , fontsize = 20)
plt.gca().set_xlim(1.0, 1e3)
plt.legend([(uu1, uu2), (vv1, vv2), (uv1, uv2)],
	   [r"$\sqrt{u'^2}^+$", r"$\sqrt{v'^2}^+$", r"$u'v'^+$"],
	   handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
plt.savefig(os.path.join(figureDir, "uRMSReTau200.eps"), bbox_inches='tight')

plt.show()
