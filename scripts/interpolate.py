#!/usr/bin/env python2

import argparse
import h5py
import json
import sys
import os
import re
import numpy as np
from joblib import Parallel, delayed

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen

parser = argparse.ArgumentParser()
parser.add_argument('json_file', type=argparse.FileType('r'),
                    help='original simulation configuration file')
parser.add_argument('--np', nargs='?', default=1, type=int,
                    help='number of cores')
parser.add_argument('--outputdir', nargs='?', const='.', default='.',
                    help='directory where output will be saved')
parser.add_argument('--inputfile', nargs='?', const='.', default='.',
                    help='input file saved')
parser.add_argument('--Xper', nargs='?', default=False, const=True,
                    help="Activate periodic grid in X-direction.")
parser.add_argument('--Yper', nargs='?', default=False, const=True,
                    help="Activate periodic grid in Y-direction.")
parser.add_argument('--Zper', nargs='?', default=False, const=True,
                    help="Activate periodic grid in Z-direction.")
parser.add_argument('--Xscale', nargs=1, default=1.0, type=float,
                    help="Activate grid scaling in X-direction.")
parser.add_argument('--Yscale', nargs=1, default=1.0, type=float,
                    help="Activate grid scaling in Y-direction.")
parser.add_argument('--Zscale', nargs=1, default=1.0, type=float,
                    help="Activate grid scaling in Z-direction.")
args = parser.parse_args()

##############################################################################
#                           Linear interpolation                             #
##############################################################################

def interp(values, i, j, k, w1, w2, w3):
   return(values[i-1,j-1,k-1] *      w1  *      w2  *      w3  +
          values[i-1,j-1,k  ] *      w1  *      w2  * (1.0-w3) +
          values[i-1,j  ,k-1] *      w1  * (1.0-w2) *      w3  +
          values[i  ,j-1,k-1] * (1.0-w1) *      w2  *      w3  +
          values[i-1,j  ,k  ] *      w1  * (1.0-w2) * (1.0-w3) +
          values[i  ,j-1,k  ] * (1.0-w1) *      w2  * (1.0-w3) +
          values[i  ,j  ,k-1] * (1.0-w1) * (1.0-w2) *      w3  +
          values[i  ,j  ,k  ] * (1.0-w1) * (1.0-w2) * (1.0-w3))

##############################################################################
#                            Read Input files                                #
##############################################################################

fin = h5py.File(args.inputfile, 'r')
config = json.load(args.json_file)

xIn = fin["centerCoordinates"][:][0,0,:,0]*args.Xscale
yIn = fin["centerCoordinates"][:][0,:,0,1]*args.Yscale
zIn = fin["centerCoordinates"][:][:,0,0,2]*args.Zscale

velocityIn              = fin["velocity"][:]
pressureIn              = fin["pressure"][:]
rhoIn                   = fin["rho"][:]     
temperatureIn           = fin["temperature"][:]
MolarFracsIn            = fin["MolarFracs"][:]
temperatureIn           = fin["temperature"][:]

channelForcing = fin.attrs["channelForcing"]

nSpec = MolarFracsIn.shape[3]

##############################################################################
#                           New Generate Grid                                #
##############################################################################

xGrid, dx = gridGen.GetGrid(config["Grid"]["origin"][0],
                            config["Grid"]["xWidth"],
                            config["Grid"]["xNum"], 
                            config["Grid"]["xType"],
                            config["Grid"]["yStretching"],
                            args.Xper)

yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                            config["Grid"]["yWidth"],
                            config["Grid"]["yNum"], 
                            config["Grid"]["yType"],
                            config["Grid"]["yStretching"],
                            args.Yper)

zGrid, dz = gridGen.GetGrid(config["Grid"]["origin"][2],
                            config["Grid"]["zWidth"],
                            config["Grid"]["zNum"], 
                            config["Grid"]["zType"],
                            config["Grid"]["zStretching"],
                            args.Zper)

# Load mapping
Ntiles = config["Mapping"]["tiles"]

assert config["Grid"]["xNum"] % Ntiles[0] == 0 
assert config["Grid"]["yNum"] % Ntiles[1] == 0
assert config["Grid"]["zNum"] % Ntiles[2] == 0

NxTile = int(config["Grid"]["xNum"]/Ntiles[0])
NyTile = int(config["Grid"]["yNum"]/Ntiles[1])
NzTile = int(config["Grid"]["zNum"]/Ntiles[2])

halo = [int(0.5*(xGrid.size-config["Grid"]["xNum"])),
        int(0.5*(yGrid.size-config["Grid"]["yNum"])),
        int(0.5*(zGrid.size-config["Grid"]["zNum"]))]

##############################################################################
#                           Produce restart file                             #
##############################################################################
if not os.path.exists(args.outputdir):
   os.makedirs(args.outputdir)

def writeTile(xt, yt, zt):
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

   centerCoordinates = np.ndarray(shape, dtype=np.dtype("(3,)f8"))
   cellWidth         = np.ndarray(shape, dtype=np.dtype("(3,)f8"))
   rho               = np.ndarray(shape)
   pressure          = np.ndarray(shape)
   temperature       = np.ndarray(shape)
   MolarFracs        = np.ndarray(shape, dtype=np.dtype("("+str(nSpec)+",)f8"))
   velocity          = np.ndarray(shape, dtype=np.dtype("(3,)f8"))
   dudtBoundary      = np.ndarray(shape)
   dTdtBoundary      = np.ndarray(shape)

   for (k,kc) in enumerate(centerCoordinates):
      kIn = np.searchsorted(zIn, zGrid[k+lo_bound[2]])
      if (kIn == 0):
         kIn = 1
         zweight = 0.0
      elif (kIn > zIn.size-1):
         kIn = zIn.size-1
         zweight = 1.0
      else:
         zweight = (zIn[kIn] - zGrid[k+lo_bound[2]])/(zIn[kIn] - zIn[kIn-1])

      for (j,jc) in enumerate(kc):
         jIn = np.searchsorted(yIn, yGrid[j+lo_bound[1]])
         if (jIn == 0):
            jIn = 1
            yweight = 0.0
         elif (jIn > yIn.size-1):
            jIn = yIn.size-1
            yweight = 1.0
         else:
            yweight = (yIn[jIn] - yGrid[j+lo_bound[1]])/(yIn[jIn] - yIn[jIn-1])

         for (i,ic) in enumerate(jc):
            iIn = np.searchsorted(xIn, xGrid[i+lo_bound[0]])
            if (iIn == 0):
               iIn = 1
               xweight = 0.0
            elif (iIn > xIn.size-1):
               iIn = xIn.size-1
               xweight = 1.0
            else:
               xweight = (xIn[iIn] - xGrid[i+lo_bound[0]])/(xIn[iIn] - xIn[iIn-1])

            centerCoordinates[k,j,i] = [xGrid[i+lo_bound[0]], yGrid[j+lo_bound[1]], zGrid[k+lo_bound[2]]]
            cellWidth        [k,j,i] = [   dx[i+lo_bound[0]],    dy[j+lo_bound[1]],    dz[k+lo_bound[2]]]
            temperature      [k,j,i] = interp(temperatureIn, kIn, jIn, iIn, zweight, yweight, xweight)
            pressure         [k,j,i] = interp(   pressureIn, kIn, jIn, iIn, zweight, yweight, xweight)
            rho              [k,j,i] = interp(        rhoIn, kIn, jIn, iIn, zweight, yweight, xweight)
            for sp in range(nSpec):
               MolarFracs       [k,j,i,sp] = interp(MolarFracsIn[:,:,:,sp], kIn, jIn, iIn, zweight, yweight, xweight)
            velocity         [k,j,i] = [ interp(velocityIn[:,:,:,0], kIn, jIn, iIn, zweight, yweight, xweight),
                                         interp(velocityIn[:,:,:,1], kIn, jIn, iIn, zweight, yweight, xweight),
                                         interp(velocityIn[:,:,:,2], kIn, jIn, iIn, zweight, yweight, xweight)]
            dudtBoundary     [k,j,i] = 0.0
            dTdtBoundary     [k,j,i] = 0.0

   with h5py.File(os.path.join(args.outputdir, filename), 'w') as fout:
      fout.attrs.create("SpeciesNames", ["MIX".encode()], dtype="S20")
      fout.attrs.create("timeStep", 0)
      fout.attrs.create("simTime", 0.0)
      fout.attrs.create("channelForcing", channelForcing)

      fout.create_dataset("centerCoordinates",     shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("cellWidth",             shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
      fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("velocity_old_NSCBC",    shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("temperature_old_NSCBC", shape=shape, dtype = np.dtype("f8"))

      fout["centerCoordinates"][:] = centerCoordinates
      fout["cellWidth"][:] = cellWidth
      fout["rho"][:] = rho
      fout["pressure"][:] = pressure
      fout["temperature"][:] = temperature
      fout["MolarFracs"][:] = MolarFracs
      fout["velocity"][:] = velocity
      fout["dudtBoundary"][:] = dudtBoundary
      fout["dTdtBoundary"][:] = dTdtBoundary
      fout["velocity_old_NSCBC"][:] = velocity
      fout["temperature_old_NSCBC"][:] = temperature

Parallel(n_jobs=args.np)(delayed(writeTile)(x, y, z) for x, y, z in np.ndindex((Ntiles[0], Ntiles[1], Ntiles[2])))
