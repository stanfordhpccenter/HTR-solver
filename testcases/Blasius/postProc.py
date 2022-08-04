import numpy as np
import json
#import argparse
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import h5py
from scipy.integrate import odeint
from scipy.optimize import fsolve

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import HTRrestart

#parser = argparse.ArgumentParser()
#parser.add_argument('-n', '--num_times', type=int, default=4)
#args = parser.parse_args()

##############################################################################
#                           Compute Blasius Solution                         #
##############################################################################
def GetBlasius():
   def BlasiusFun(U, y):
      f1, f2, f3 = U
      return [ f2, f3, -0.5*f1*f3 ]

   Np = 100
   eta = np.linspace(0, 10, Np)
   f1_0 = eta[0]
   f2_0 = 0.0

   def objective(f3_0):
      U = odeint(BlasiusFun, [f1_0, f2_0, f3_0], eta)
      f1, f2, f3 = U.T
      return (f2[Np-1] - 1.0)

   f3_0, = fsolve(objective, 0.5)

   U = odeint(BlasiusFun, [f1_0, f2_0, f3_0], eta)
   f1, f2, f3 = U.T
   return eta, f2, (f2*eta - f1)*0.5


def process():
   dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/Blasius')
   prometeo_input_file = os.path.join(dir_name, 'Blasius.json')

##############################################################################
#                         Read Prometeo Input File                           #
##############################################################################

   with open(prometeo_input_file) as f:
      data = json.load(f)

   xNum = data["Grid"]["xNum"]
   yNum = data["Grid"]["yNum"]
   xWidth  = data["Grid"]["GridInput"]["width"][0]
   yWidth  = data["Grid"]["GridInput"]["width"][1]
   xOrigin = data["Grid"]["GridInput"]["origin"][0]
   yOrigin = data["Grid"]["GridInput"]["origin"][1]

   R = data["Flow"]["mixture"]["gasConstant"]
   mu = data["Flow"]["mixture"]["viscosityModel"]["Visc"]

   T  = data["BC"]["xBCLeft"]["TemperatureProfile"]["temperature"]
   P  = data["BC"]["xBCLeft"]["P"]
   U  = data["BC"]["xBCLeft"]["VelocityProfile"]["velocity"]
   Re = data["BC"]["xBCLeft"]["VelocityProfile"]["Reynolds"]

   Rho = P/(R*T)
   nu = mu/Rho

   nstep = int(data["Integrator"]["maxIter"])

   filename = os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-'+str(xNum+1)+','+str(yNum+1)+',0.hdf')

##############################################################################
# Read HTR output data                                                       #
##############################################################################
   restart = HTRrestart.HTRrestart(data)
   restart.attach(sampleDir=os.path.join(dir_name, 'sample0'), step=nstep)

   # Get the data
   centerCoordinates = restart.load("centerCoordinates")
   velocity          = restart.load("velocity")

   # Get simulation data along a line (ignore ghost cells)
   x_slice_idx = int(xNum*0.5)

   x0   = centerCoordinates[0,0,0          ][0]   - xOrigin
   x    = centerCoordinates[0,0,x_slice_idx][0]   - xOrigin
   y    = centerCoordinates[0,:,x_slice_idx][:,1] - yOrigin
   u    =          velocity[0,:,x_slice_idx][:,0]
   v    =          velocity[0,:,x_slice_idx][:,1]

   x += Re*nu/U-x0
   Re = U*x/nu
   print("Reynolds number: ", Re)

   return y*np.sqrt(Re)/x, u/U ,v*np.sqrt(Re)/U

etaB, uB, vB = GetBlasius()
eta,   u,  v = process()

##############################################################################
#                                     Plot                                   #
##############################################################################

plt.rcParams.update({'font.size': 16})

plt.figure(1)
plt.plot(uB, etaB, 'xk', label='Blasius')
plt.plot( u, eta , '-r', label='HTR solver')
plt.xlabel(r'$u/U_\infty$', fontsize = 20)
plt.ylabel(r'$\eta$'      , fontsize = 20)
plt.legend()
plt.savefig('U.eps', bbox_inches='tight')

plt.figure(2)
plt.plot(vB, etaB, 'xk', label='Blasius')
plt.plot( v, eta , '-r', label='HTR solver')
plt.xlabel(r'$v\sqrt{Re_x}/U_\infty$', fontsize = 20)
plt.ylabel(r'$\eta$'                 , fontsize = 20)
plt.legend()
plt.savefig('V.eps', bbox_inches='tight')

plt.show()
