import numpy as np
import json
import sys
import os
import subprocess
import h5py
from scipy.integrate import odeint
from scipy.optimize import fsolve

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix

##############################################################################
#                         Read Prometeo Input File                           #
##############################################################################

dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/Blasius')
prometeo_input_file = os.path.join(dir_name, 'Blasius.json')

with open(prometeo_input_file) as f:
   data = json.load(f)

xNum = data["Grid"]["xNum"]
yNum = data["Grid"]["yNum"]
zNum = data["Grid"]["zNum"]
xWidth  = data["Grid"]["xWidth"]
yWidth  = data["Grid"]["yWidth"]
xOrigin = data["Grid"]["origin"][0]
yOrigin = data["Grid"]["origin"][1]
xType  = data["Grid"]["xType"]
yType  = data["Grid"]["yType"]
xStretching  = data["Grid"]["xStretching"]
yStretching  = data["Grid"]["yStretching"]

gamma = data["Flow"]["gamma"]
R = data["Flow"]["gasConstant"]
Pr    = data["Flow"]["prandtl"]

TInf = data["BC"]["xBCLeft"]["TemperatureProfile"]["temperature"]
Tw   = data["BC"]["xBCLeft"]["TemperatureProfile"]["temperature"]
P    = data["BC"]["xBCLeft"]["P"]
U    = data["BC"]["xBCLeft"]["VelocityProfile"]["velocity"]
Re   = data["BC"]["xBCLeft"]["VelocityProfile"]["Reynolds"]
 
aInf = np.sqrt(gamma*R*TInf)
MaInf = U/aInf

RhoInf = ConstPropMix.GetDensity(TInf, P, data)
muInf = ConstPropMix.GetViscosity(TInf, data)
nuInf = muInf/RhoInf

##############################################################################
#                           Compute Blasius Solution                         #
##############################################################################
def GetCBL():
   def CBLFun(U, y):
      u, F, h, G, g = U.T
      T = 1.0 + 0.5*h*(gamma - 1.0)*MaInf**2
      mu = T**0.7
      return [ F/mu,
               -0.5*g*F/mu,
               Pr*G/mu,
               -0.5*Pr*G*g/mu - 2*F**2/mu,
               u/T ]

   Np = 100
   eta = np.linspace(0, 25, Np)
   u_0 = 0.0
   #F_0 = 0.0
   h_0 = (Tw/TInf-1.0)*2/((gamma - 1.0)*MaInf**2)
   #G_0 = 0.0
   g_0 = 0.0

   def objective(A):
      F_0, G_0 = A
      U = odeint(CBLFun, [u_0, F_0, h_0, G_0, g_0], eta)
      u, F, h, G, g = U.T
      return [u[Np-1] - 1.0, h[Np-1]]

   A = fsolve(objective, [0.01, 0.5])
   F_0, G_0 = A

   U = odeint(CBLFun, [u_0, F_0, h_0, G_0, g_0], eta)
   u, F, h, G, g = U.T
   T = 1.0 + 0.5*h*(gamma - 1.0)*MaInf**2
   v = (eta*u/T - g)*T*0.5
   return eta, u, v, T

##############################################################################
#               Generate y grid that will be used in the solver              #
##############################################################################
y, dy = gridGen.GetGrid(data["Grid"]["origin"][1],
                        data["Grid"]["yWidth"],
                        data["Grid"]["yNum"],
                        data["Grid"]["yType"],
                        data["Grid"]["yStretching"],
                        False)

##############################################################################
#                     Compute the profile on this grid                       #
##############################################################################
x = Re*nuInf/U
etaB, uB, vB, TB = GetCBL()
yB = etaB*x/np.sqrt(Re)
uB *= U
vB *= U/np.sqrt(Re)
TB *= TInf

u = np.interp(y, yB, uB)
v = np.interp(y, yB, vB)
T = np.interp(y, yB, TB)

##############################################################################
#                          Print Prometeo Profile                            #
##############################################################################

profileDir = os.path.join(dir_name, 'InflowProfile')
exists = os.path.isdir(profileDir)

if (not exists):
   mkdir_command = 'mkdir {}'.format(os.path.expandvars(profileDir))
   try:
      subprocess.call(mkdir_command, shell=True)
   except OSError:
      print("Failed command: {}".format(mkdir_command))
      sys.exit()

filename = os.path.join(profileDir, '0,0,0-'+str(xNum+1)+','+str(yNum+1)+',0.hdf')

f = h5py.File(filename, 'w')

shape = [zNum, yNum+2, xNum+2]
MolarFracs_profile  = np.ndarray(shape, dtype=np.dtype('(1,)f8'))
temperature_profile = np.ndarray(shape) 
velocity_profile    = np.ndarray(shape, dtype=np.dtype('(3,)f8')) 
for k in range(0, shape[0]):
   for j in range(0, shape[1]):
      for i in range(0, shape[2]):
         MolarFracs_profile[k,j,i] = [1.0,]
         temperature_profile[k,j,i] = T[j]
         velocity_profile[k,j,i] = np.array([u[j], v[j], 0])

f.create_dataset("MolarFracs_profile",   shape=shape, dtype = np.dtype("(1,)f8"))
f.create_dataset("velocity_profile",     shape=shape, dtype = np.dtype("(3,)f8"))
f.create_dataset("temperature_profile",  shape=shape, dtype = np.dtype("f8"))

f["MolarFracs_profile" ][:] = MolarFracs_profile
f["temperature_profile"][:] = temperature_profile
f["velocity_profile"   ][:] = velocity_profile

f.close()
