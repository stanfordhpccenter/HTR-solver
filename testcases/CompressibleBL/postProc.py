import numpy as np
import json
#import argparse
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import h5py
import pandas
from scipy.integrate import odeint
from scipy.optimize import fsolve

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix

dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/CompressibleBL')
prometeo_input_file = os.path.join(dir_name, 'CBL.json')

muExp = 0.75

##############################################################################
#                       Read Prometeo Input File                             #
##############################################################################
with open(prometeo_input_file) as f:
   data = json.load(f)

xNum = data["Grid"]["xNum"]
yNum = data["Grid"]["yNum"]
xWidth  = data["Grid"]["xWidth"]
yWidth  = data["Grid"]["yWidth"]
xOrigin = data["Grid"]["origin"][0]
yOrigin = data["Grid"]["origin"][1]

gamma = data["Flow"]["gamma"]
R     = data["Flow"]["gasConstant"]
muRef = data["Flow"]["powerlawViscRef"]
TRef  = data["Flow"]["powerlawTempRef"]
Pr    = data["Flow"]["prandtl"]

TInf = data["BC"]["xBCLeftHeat"]["temperature"]
Tw = data["BC"]["yBCLeftHeat"]["temperature"]
P = data["BC"]["xBCLeftP"]
U = data["BC"]["xBCLeftInflowProfile"]["velocity"]
Re = data["BC"]["xBCLeftInflowProfile"]["Reynolds"]

aInf = np.sqrt(gamma*R*TInf)
MaInf = U/aInf

def visc(T):
   return muRef*(T/TRef)**0.7

##############################################################################
#                            Process result file                             #
##############################################################################
def process(frac):
   RhoInf = P/(R*TInf)
   nuInf = visc(TInf)/RhoInf

   dt    = data["Integrator"]["fixedDeltaTime"]
   nstep = int(data["Integrator"]["maxIter"])
   time = dt*nstep

   filename = os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-'+str(xNum+1)+','+str(yNum+1)+',0.hdf')
   exists = os.path.isfile(filename)

   if (not exists):
      # merge files from different tiles 
      merge_command = 'python {} {}'.format(os.path.expandvars('$HTR_DIR/scripts/merge.py'),
                                            os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/*.hdf'))
      mv_command = 'mv {} {}'.format('./0,0,0-'+str(xNum+1)+','+str(yNum+1)+',0.hdf',
                                     os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/'))

      try:
         subprocess.call(merge_command, shell=True)
      except OSError:
         print("Failed command: {}".format(merge_command))
         sys.exit()

      try:
         subprocess.call(   mv_command, shell=True)
      except OSError:
         print("Failed command: {}".format(mv_command))
         sys.exit()

##############################################################################
#                        Read Prometeo Output Data                           #
##############################################################################

   f = h5py.File(filename, 'r')

   # Get the data
   centerCoordinates = f['centerCoordinates']
   cellWidth   = f['cellWidth']
   pressure    = f['pressure']
   temperature = f['temperature']
   density     = f['rho']
   velocity    = f['velocity']

   # Get simulation data along a line (ignore ghost cells)
   x_slice_idx = int(frac*xNum)

   x0   = centerCoordinates[0,0,0          ][0]   - xOrigin
   x    = centerCoordinates[0,0,x_slice_idx][0]   - xOrigin
   y    = centerCoordinates[0,:,x_slice_idx][:,1] - yOrigin
   u    =          velocity[0,:,x_slice_idx][:,0]
   v    =          velocity[0,:,x_slice_idx][:,1]
   T    =       temperature[0,:,x_slice_idx]
   p    =          pressure[0,:,x_slice_idx]
   rho  =           density[0,:,x_slice_idx]

   x += Re*nuInf/U-x0
   myRe = U*x/nuInf
   print(myRe)

   eta = np.zeros(y.size)
   for i in range(y.size) :
      if (i > 0) :
         rhoMid = 0.5*(rho[i] + rho[i-1])
         eta[i] = eta[i-1] + U/(visc(TInf)*np.sqrt(2*myRe))*rhoMid*(y[i] - y[i-1])

   return eta, u/U, v*np.sqrt(2.0*myRe)/U, T/TInf, p

##############################################################################
#                          Read similarity solution                          #
##############################################################################

CBL = pandas.read_csv("SimilaritySolution.dat")
etaB = CBL["eta"][:].values
fB   = CBL["f"  ][:].values
uB   = CBL["u"  ][:].values
TB   = CBL["T"  ][:].values

integ = 0
vB = np.zeros(etaB.size)
for i in range(etaB.size):
   if (i>0):
      integ += 0.5*(TB[i] + TB[i-1])*(etaB[i] - etaB[i-1])
   vB[i] = integ*uB[i] - TB[i]*fB[i]

eta1, u1, v1, T1, p1 = process(0.4)
eta2, u2, v2, T2, p2 = process(0.8)

##############################################################################
#                                     Plot                                   #
##############################################################################

plt.rcParams.update({'font.size': 16})
stp = 2

plt.figure(1)
plt.plot(uB[::stp], etaB[::stp], 'xk', label='Self-similar theory')
#plt.plot(u1, eta1, '-r', label='u, x/L = 0.4')
#plt.plot(u2, eta2, '-b', label='u, x/L = 0.8')
plt.plot(u2, eta2, '-r', label='HTR solver')
plt.xlabel(r'$u/U_\infty$', fontsize = 20)
plt.ylabel(r'$\eta$'      , fontsize = 20)
plt.gca().set_ylim(0, 10)
plt.legend()
plt.savefig('U.eps', bbox_inches='tight')

plt.figure(2)
plt.plot(vB[::stp], etaB[::stp], 'xk', label='Self-similar theory')
#plt.plot(v1, eta1, '-r', label='v, x/L = 0.4')
#plt.plot(v2, eta2, '-b', label='v, x/L = 0.8')
plt.plot(v2, eta2, '-r', label='HTR solver')
plt.xlabel(r'$v/U_\infty \sqrt{2 Re_x}$', fontsize = 20)
plt.ylabel(r'$\eta$'                     , fontsize = 20)
plt.gca().set_ylim(0, 10)
plt.legend()
plt.savefig('V.eps', bbox_inches='tight')

plt.figure(3)
plt.plot(TB[::stp], etaB[::stp], 'xk', label='Self-similar theory')
#plt.plot(T1, eta1, '-r', label='T, x/L = 0.4')
#plt.plot(T2, eta2, '-b', label='T, x/L = 0.8')
plt.plot(T2, eta2, '-r', label='HTR solver')
plt.xlabel(r'$T/T_\infty$', fontsize = 20)
plt.ylabel(r'$\eta$'      , fontsize = 20)
plt.gca().set_ylim(0, 10)
plt.legend()
plt.savefig('T.eps', bbox_inches='tight')

plt.show()
