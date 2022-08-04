#!/usr/bin/env python3

import argparse
import sys
import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed

# HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix
import MulticomponentMix

##############################################################################
#                              Command-line arguments                        #
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('base_json', default='base.json', type=argparse.FileType('r'))
args = parser.parse_args()

# Specify parameters for each run.  Use values computed by Cantera
phi = [0.5,1.0,1.5,2.0]
#sL = [3.627554193285, 3.900223653680, 2.288162774171, 0.855539714059] # m/s
sL = [3.006947467633, 3.383376472703, 2.194443653335, 0.885703257707] # m/s
nPhi = np.size(phi)
p0 = 101325.0
Ru = 8.31446261815324 # J/mol
Tf = 3000.0 * np.ones([nPhi])
nu_O2 = 2.0
nu_H2O = 2.0
nu_CO2 = 1.0

# Load base config
config = json.load(args.base_json)
mix = MulticomponentMix.Mix(config['Flow']['mixture'])
uRef = mix.uRef
rhoRef = mix.rhoRef
TRef = mix.TRef
PRef = mix.PRef
WCH4 = 0.0120107*1+0.00100784*4
WO2 = 0.0159994*2
WH2O = 0.00100784*2+0.0159994*1
WCO2 = 0.0120107*1+0.0159994*2
stoich = WCH4 / (2.0*WO2) # fuel-oxidizer stoichiometric (mass-based)
mixtureName = config["Flow"]["mixture"]["type"]
if (mixtureName == "CH41StMix"):
   nSpec = 4
   iCH4 = 0
   iO2 = 1
   iCO2 = 2
   iH2O = 3
elif (mixtureName == "CH4_30SpMix"):
   nSpec = 30
   iCH4 = 13
   iO2 = 3
   iCO2 = 15
   iH2O = 5
elif (mixtureName == "FFCM1Mix"):
   nSpec = 33
   iCH4 = 15
   iO2 = 3
   iCO2 = 9
   iH2O = 5
elif (mixtureName == "ConstPropMix"):
   nSpec = 1
else:
   raise Exception('Unrecognized mixture name: {}'.format(mixtureName))

# Where init files are stored
initBase = 'init'
if (not os.path.isdir(initBase)):
   os.mkdir(initBase)

##############################################################################
#                              Generate Grid                                 #
##############################################################################

# Which directions are periodic?
xBCLeft = config["BC"]["xBCLeft"]["type"]
xBCRight = config["BC"]["xBCRight"]["type"]
yBCLeft = config["BC"]["yBCLeft"]["type"]
yBCRight = config["BC"]["yBCRight"]["type"]
zBCLeft = config["BC"]["zBCLeft"]["type"]
zBCRight = config["BC"]["zBCRight"]["type"]
xPeriodic = (xBCLeft.lower() == "periodic") and (xBCRight.lower() == "periodic")
yPeriodic = (yBCLeft.lower() == "periodic") and (yBCRight.lower() == "periodic")
zPeriodic = (zBCLeft.lower() == "periodic") and (zBCRight.lower() == "periodic")
halo = [0,0,0]
if (not xPeriodic):
   halo[0] = 1
if (not yPeriodic):
   halo[1] = 1
if (not zPeriodic):
   halo[2] = 1

# Grid sizes, used for flow initialization
Lx = config["Grid"]["GridInput"]["width"][0]
Ly = config["Grid"]["GridInput"]["width"][1]
Lz = config["Grid"]["GridInput"]["width"][2]
Nx = config["Grid"]["xNum"]
Ny = config["Grid"]["yNum"]
Nz = config["Grid"]["zNum"]
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

Ntiles = config["Mapping"]["tiles"]
Ntiles[0] = int(Ntiles[0]/config["Mapping"]["tilesPerRank"][0])
Ntiles[1] = int(Ntiles[1]/config["Mapping"]["tilesPerRank"][1])
Ntiles[2] = int(Ntiles[2]/config["Mapping"]["tilesPerRank"][2])
NxTile = int(config["Grid"]["xNum"]/Ntiles[0])
NyTile = int(config["Grid"]["yNum"]/Ntiles[1])
NzTile = int(config["Grid"]["zNum"]/Ntiles[2])

##############################################################################
#                             Define output routine                          #
##############################################################################
def writeTile(xt, yt, zt, rho, T, u, Xk, dir):
   lo_bound = [(xt  )*NxTile  +halo[0], (yt  )*NyTile  +halo[1], (zt  )*NzTile  +halo[2]]
   hi_bound = [(xt+1)*NxTile-1+halo[0], (yt+1)*NyTile-1+halo[1], (zt+1)*NzTile-1+halo[2]]
   if (xt == 0): lo_bound[0] -= halo[0]
   if (yt == 0): lo_bound[1] -= halo[1]
   if (zt == 0): lo_bound[2] -= halo[2]
   if (xt == Ntiles[0]-1): hi_bound[0] += halo[0]
   if (yt == Ntiles[1]-1): hi_bound[1] += halo[1]
   if (zt == Ntiles[2]-1): hi_bound[2] += halo[2]

   # Announce
   filename = '%s/%s,%s,%s-%s,%s,%s.hdf'\
      %(dir,lo_bound[0],lo_bound[1],lo_bound[2],hi_bound[0],hi_bound[1],hi_bound[2])

   # Ordered as z,y,x
   shape = [hi_bound[2] - lo_bound[2] +1,
            hi_bound[1] - lo_bound[1] +1,
            hi_bound[0] - lo_bound[0] +1]

   # Populate arrays
   rho_htr           = np.ndarray(shape)
   pressure          = np.ndarray(shape)
   temperature       = np.ndarray(shape)
   MolarFracs        = np.ndarray(shape, dtype=np.dtype('({},)f8'.format(nSpec)))
   velocity          = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dudtBoundary      = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
   dTdtBoundary      = np.ndarray(shape)
   pressure[:] = p0/PRef
   dudtBoundary[:] = [0.0, 0.0, 0.0]
   dTdtBoundary[:] = 0.0
   for (k,kc) in enumerate(centerCoordinates):
      for (j,jc) in enumerate(kc):
         for (i,ic) in enumerate(jc):
            # Indices
            ii = min(Nx-1,i+lo_bound[0]+halo[0])
            jj = min(Ny-1,j+lo_bound[1]+halo[1])
            kk = min(Nz-1,k+lo_bound[2]+halo[2])

            # Density
            rho_htr[k,j,i] = rho[ii,jj,kk] / rhoRef

            # Temperature
            temperature[k,j,i] = T[ii,jj,kk] / TRef

            # Velocity
            velocity[k,j,i] = np.array([0.0,u[ii,jj,kk],0.0]) / uRef

            # Composition
            MolarFracs[k,j,i] = Xk[ii,jj,kk,:]

   with h5py.File(filename, 'w') as fout:
      fout.attrs.create("SpeciesNames",["CH4".encode(),"O2".encode(),"CO2".encode(),"H2O".encode()],dtype="S20")

      fout.attrs.create("timeStep", 0)
      fout.attrs.create("simTime", 0.0)

      fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("({},)f8".format(nSpec)))
      fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs_profile",    shape=shape, dtype = np.dtype("({},)f8".format(nSpec)))
      fout.create_dataset("velocity_profile",      shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("temperature_profile",   shape=shape, dtype = np.dtype("f8"))

      fout["rho"][:] = rho_htr
      fout["pressure"][:] = pressure
      fout["temperature"][:] = temperature
      fout["MolarFracs"][:] = MolarFracs
      fout["velocity"][:] = velocity
      fout["dudtBoundary"][:] = dudtBoundary
      fout["dTdtBoundary"][:] = dTdtBoundary
      fout["MolarFracs_profile"][:] = MolarFracs
      fout["velocity_profile"][:] = velocity
      fout["temperature_profile"][:] = temperature

##############################################################################
#                    Write input and init files for each case                #
##############################################################################
rho = np.zeros([Nx,Ny,Nz])
T = np.zeros([Nx,Ny,Nz])
Xk = np.zeros([Nx,Ny,Nz,nSpec])
u = np.zeros([Nx,Ny,Nz])
for n in range(nPhi):

   # Unburnt composition
   X_unburnt = np.zeros(4)
   XO2_unburnt = 1.0 / (1.0 + stoich*phi[n]*WO2/WCH4)
   XCH4_unburnt = 1.0 - XO2_unburnt
   unburntMixture = [ \
      {"Name" : "CH4", "MolarFrac" : XCH4_unburnt},\
      {"Name" : "O2",  "MolarFrac" : XO2_unburnt} \
   ]
   T_unburnt = TRef
   W_unburnt = XCH4_unburnt*WCH4 + XO2_unburnt*WO2
   rho_unburnt = p0*W_unburnt/(Ru*T_unburnt)

   # Burnt composition
   X_burnt = np.zeros(4)
   if (phi[n] >= 1.0):
      XCH4_burnt = XCH4_unburnt - XO2_unburnt/nu_O2
      XO2_burnt = 0.0
      XH2O_burnt = (XCH4_unburnt-XCH4_burnt)*nu_H2O
      XCO2_burnt = (XCH4_unburnt-XCH4_burnt)*nu_CO2
   else:
      XCH4_burnt = 0.0
      XO2_burnt = XO2_unburnt - XCH4_unburnt*nu_O2
      XH2O_burnt = XCH4_unburnt*nu_H2O
      XCO2_burnt = XCH4_unburnt*nu_CO2
   burntMixture = [ \
      {"Name" : "CH4",  "MolarFrac" : XCH4_burnt}, \
      {"Name" : "O2",   "MolarFrac" : XO2_burnt}, \
      {"Name" : "H2O",  "MolarFrac" : XH2O_burnt}, \
      {"Name" : "CO2",  "MolarFrac" : XCO2_burnt} \
   ]
   T_burnt = Tf[n]
   W_burnt = XCH4_burnt*WCH4 + XO2_burnt*WO2 + XH2O_burnt*WH2O + XCO2_burnt*WCO2
   rho_burnt = p0*W_burnt/(Ru*T_burnt)

   # Inflow conditions (unburnt)
   config["BC"]["yBCLeft"]["P"] = p0/PRef
   config["BC"]["yBCLeft"]["VelocityProfile"]["velocity"] = [0.0,sL[n]/uRef,0.0]
   config["BC"]["yBCLeft"]["TemperatureProfile"]["temperature"] = 1.0
   config["BC"]["yBCLeft"]["MixtureProfile"]["Mixture"]["Species"] = unburntMixture

   # Outflow conditions
   config["BC"]["yBCRight"]["P"] = p0/PRef

   # Location of initialization file
   initDir = '{}/case-{}'.format(initBase,n+1)
   if (not os.path.isdir(initDir)):
      os.mkdir(initDir)
   config["Flow"]["initCase"]["restartDir"] = initDir

   # Initial state (burnt)
   #config["Flow"]["initCase"]["pressure"] = p0/PRef
   #config["Flow"]["initCase"]["temperature"] = T_burnt/TRef
   #config["Flow"]["initCase"]["velocity"] = [0.0,sL[n]/uRef,0.0]
   #config["Flow"]["initCase"]["molarFracs"]["Species"] = burntMixture
   #config["Flow"]["initMixture"]["Species"] = burntMixture

   # Write input file for HTR run
   fname = 'PlanarFlame1D-{}.json'.format(n+1)
   with open(fname, 'w') as fout:
      json.dump(config, fout, indent=3)
   print('File written: {}'.format(fname))

   # Generate base flow
   y0 = Ly/10.0 # Location of "flame"
   w = 10.0*np.mean(dy) # Thickness of "flame"
   sigma = 2.0/w * np.arctanh(2.0*0.9 - 1.0) # for tanh function
   for j in range(Ny):
      y = yGrid[j+halo[1]]

      # Sigmoid
      s = 0.5 + 0.5*np.tanh(sigma*(y-y0))

      # Temperature
      T[:,j,:] = T_unburnt + (T_burnt-T_unburnt)*s

      # Composition
      XCH4 = XCH4_unburnt + (XCH4_burnt-XCH4_unburnt)*s
      XO2 = XO2_unburnt + (XO2_burnt-XO2_unburnt)*s
      XH2O = XH2O_burnt*s
      XCO2 = 1.0 - XCH4 - XO2 - XH2O
      Xk[:,j,:,iCH4] = XCH4
      Xk[:,j,:,iO2] = XO2
      Xk[:,j,:,iH2O] = XH2O
      Xk[:,j,:,iCO2] = 1.0 - XCH4 - XO2 - XH2O

      # Density -- must come after composition
      W = XCH4*WCH4 + XO2*WO2 + XH2O*WH2O + XCO2*WCO2
      rho[:,j,:] = p0*W/(Ru*T[:,j,:])

      # Velocity -- must come after density.
      u[:,j,:] = rho_unburnt*sL[n] / rho[:,j,:]

   # Write initialization file
   Parallel(n_jobs=1)(delayed(writeTile)(x, y, z, rho, T, u, Xk, initDir) \
      for x, y, z in np.ndindex((Ntiles[0], Ntiles[1], Ntiles[2])))
   print('Initialization file written to {}'.format(initDir))
