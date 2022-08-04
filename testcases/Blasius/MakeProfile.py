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
import HTRrestart

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
xWidth  = data["Grid"]["GridInput"]["width"][0]
yWidth  = data["Grid"]["GridInput"]["width"][1]
xOrigin = data["Grid"]["GridInput"]["origin"][0]
yOrigin = data["Grid"]["GridInput"]["origin"][1]

gamma = data["Flow"]["mixture"]["gamma"]
R     = data["Flow"]["mixture"]["gasConstant"]
Pr    = data["Flow"]["mixture"]["prandtl"]

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
x, y, z, dx, dy, dz = gridGen.getCellCenters(data)

##############################################################################
#                     Compute the profile on this grid                       #
##############################################################################
x = Re*nuInf/U
etaB, uB, vB, TB = GetCBL()
yB = etaB*x/np.sqrt(Re)
uB *= U
vB *= U/np.sqrt(Re)
TB *= TInf

# Get VorticityScale
delta = 0.0
for i in range(len(uB)):
   if (uB[i] > 0.99*U):
      delta = yB[i]
      break
data["Integrator"]["EulerScheme"]["vorticityScale"] = U/delta

u = np.interp(y, yB, uB)
v = np.interp(y, yB, vB)
T = np.interp(y, yB, TB)

##############################################################################
#                          Print profile files                               #
##############################################################################

def temperature(i, j, k):
   return T[j]

def MolarFracs(i, j, k):
   return [1.0,]

def velocity(i, j, k):
   return [u[j], v[j], 0.0]

restart = HTRrestart.HTRrestart(data)
restart.write_profiles('InflowProfile', 1,
              temperature,
              MolarFracs,
              velocity)
