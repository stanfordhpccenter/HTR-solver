
# Post-processing script that reads in hdf solution files and checks global quantities.
# It is written for the GeometricKernel laser model.

import argparse
import json
import sys
import os
import glob
import time # for wall clock
import numpy as np
import h5py
from joblib import Parallel, delayed

# HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix
import MulticomponentMix
import HTRrestart

# Plotting
import matplotlib.pyplot as mplot
mplot.switch_backend('agg') # only if you get "DISPLAY variable not defined" error
mplot.rcParams.update({"text.usetex": True,"font.family": "Times New Roman","axes.titlesize": 14,\
   "axes.labelsize": 14,"legend.fontsize": 14,"xtick.labelsize": 11,"ytick.labelsize": 11})

# Helper functions
from helpers import *

# Options
plot_sol = False # Not very pretty, but shows the solution at midplane

##############################################################################
#                           Command-line arguments                           #
##############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('dir')
parser.add_argument('--start',type=int,default=-1)
parser.add_argument('--skip',type=int,default=-1)
parser.add_argument('--stop',type=int,default=-1)
parser.add_argument('--np',type=int,default=-1)

# Parse and extract
args = parser.parse_args()
runDir = args.dir
startIter = args.start
skipIter = args.skip
stopIter = args.stop
nProcs = args.np # nProcs = -1 means Parallel will use maximum allowed
name = args.name
inputFile = '{}/{}.json'.format(runDir,name)
nIter = int((stopIter-startIter)/skipIter) + 1

# Announce
print('Post-processing data in {}.  Input file = {}'.format(runDir,inputFile))

##############################################################################
#                             Read input file                                #
##############################################################################
with open(inputFile) as f:
   config = json.load(f)

# Read config
mixtureName = config['Flow']['mixture']['type']
if (mixtureName != "ConstPropMix"):
   raise Exception("This script assumes ConstPropMix.  Aborting.")

# Which directions are periodic?
xBCLeft = config['BC']['xBCLeft']['type']
xBCRight = config['BC']['xBCRight']['type']
yBCLeft = config['BC']['yBCLeft']['type']
yBCRight = config['BC']['yBCRight']['type']
zBCLeft = config['BC']['zBCLeft']['type']
zBCRight = config['BC']['zBCRight']['type']
xPeriodic = (xBCLeft.lower() == 'periodic') and (xBCRight.lower() == 'periodic')
yPeriodic = (yBCLeft.lower() == 'periodic') and (yBCRight.lower() == 'periodic')
zPeriodic = (zBCLeft.lower() == 'periodic') and (zBCRight.lower() == 'periodic')
Ntiles = config['Mapping']['tiles']
halo = [0,0,0]
if (not xPeriodic):
   halo[0] = 1
if (not yPeriodic):
   halo[1] = 1
if (not zPeriodic):
   halo[2] = 1

# Halo, tiles, grid size
Mx = config['Grid']['xNum']
My = config['Grid']['yNum']
Mz = config['Grid']['zNum']
Lx = config['Grid']['GridInput']['width'][0]
Ly = config['Grid']['GridInput']['width'][1]
Lz = config['Grid']['GridInput']['width'][2]
Nx = Mx + 2*halo[0]
Ny = My + 2*halo[1]
Nz = Mz + 2*halo[2]

# Constants
R = config["Flow"]["mixture"]["gasConstant"]
gamma = config["Flow"]["mixture"]["gamma"]

# Reference quantities, for quantifying relative error (not the same as nondimensionalizing factors)
kernel_length = config["Flow"]["laser"]["axialLength"]
kernel_radius = max(config["Flow"]["laser"]["farRadius"],config["Flow"]["laser"]["nearRadius"])
p_ref = config["Flow"]["initCase"]["pressure"]
T_ref = config["Flow"]["initCase"]["temperature"]
rho_ref = p_ref / (R*T_ref)
u_ref = np.sqrt(gamma*p_ref/rho_ref)
mom_ref = rho_ref*u_ref
V_ref = kernel_length * np.pi*kernel_radius**2
mass_ref = rho_ref * V_ref
E_ref = p_ref/(gamma-1.0) * V_ref

# Determine which iterations to process
if (startIter >= 0 and skipIter > 0 and stopIter >= 0):
   iter = np.linspace(startIter,stopIter,nIter,dtype=int)
else:
   dirList = glob.glob('{}/sample0/fluid_iter*'.format(runDir))
   nIter = len(dirList)
   iter = np.zeros([nIter],dtype=int)
   for n in range(nIter):
      iter[n] = int(dirList[n][-10:])
   iter = np.sort(iter)
   startIter = iter[0]
   stopIter = iter[-1]
   skipIter = 0


##############################################################################
#                                  Main loop                                 #
##############################################################################
dir_name = os.path.join(os.environ['HTR_DIR'],'testcases/LaserInBox')

mass = np.zeros([nIter])
xmom= np.zeros([nIter])
ymom= np.zeros([nIter])
zmom= np.zeros([nIter])
E = np.zeros([nIter])
E_laser = np.zeros([nIter])
E_laser_true = 4.00065e-5
for n in range(nIter):

   print('')

   ##############################################################################
   #                                 Read data                                  #
   ##############################################################################
   tic = time.time()

   restart = HTRrestart.HTRrestart(config)
   restart.attach(sampleDir=os.path.join(dir_name,runDir+'/sample0'),step=iter[n])

   # Extract data
   tic = time.time()
   t = restart.simTime
   centerCoordinates = restart.load('centerCoordinates')
   x = centerCoordinates[:,:,:,0]
   y = centerCoordinates[:,:,:,1]
   z = centerCoordinates[:,:,:,2]
   del centerCoordinates
   velocity = restart.load('velocity')
   ux = velocity[:,:,:,0]
   uy = velocity[:,:,:,1]
   uz = velocity[:,:,:,2]
   del velocity
   rho = restart.load('rho')
   p = restart.load('pressure')
   T = restart.load('temperature')
   rhoetot = p/(gamma-1.0) + 0.5*rho*(ux**2+uy**2+uz**2) # Total energy per volume for ConstPropMix

   ##############################################################################
   #                             Post-process data                              #
   ##############################################################################

   # For trapz
   x1d = x[0,0,:]
   y1d = y[0,:,0]
   z1d = z[:,0,0]

   # Total mass
   mass[n] = np.trapz(np.trapz(np.trapz(rho,z1d,axis=0),y1d,axis=0),x1d,axis=0)

   # Total momentum
   xmom[n] = np.trapz(np.trapz(np.trapz(rho*ux,z1d,axis=0),y1d,axis=0),x1d,axis=0)
   ymom[n] = np.trapz(np.trapz(np.trapz(rho*uy,z1d,axis=0),y1d,axis=0),x1d,axis=0)
   zmom[n] = np.trapz(np.trapz(np.trapz(rho*uz,z1d,axis=0),y1d,axis=0),x1d,axis=0)

   # Total energy
   E[n] = np.trapz(np.trapz(np.trapz(rhoetot,z1d,axis=0),y1d,axis=0),x1d,axis=0)
   E_laser[n] = E[n] - E[0]

   toc = time.time()
   print('Iteration {:010d} processed: t={:9.3e}  ({:5.1f} s)'.format(iter[n],t,toc-tic))
   print('mass={:12.6e}, xmom={:12.6e}, ymom={:12.6e}, zmom={:12.6e}, E={:12.6e}, E_laser={:12.6e}'\
      .format(mass[n],xmom[n],ymom[n],zmom[n],E[n],E_laser[n]))

   ##############################################################################
   #                                 Plot data                                  #
   ##############################################################################

   if (plot_sol): 

      # Use this as a visual check of the energy kernel geometry
      figSize = (12,5)
      fig,axes = mplot.subplots(nrows=1,ncols=2,figsize=figSize)
      mplot.tight_layout()
      k0 = round(float(Nz)/2.0)

      fig.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.95,wspace=0.25,hspace=0.25)
      ms = 0.5
      x_bnds = [0.95*np.min(x), 1.05*np.max(x)]
      y_bnds = [0.95*np.min(y), 1.05*np.max(y)]

      ax = axes[0]
      tmp = ax.contourf(x[k0,:,:],y[k0,:,:],rho[k0,:,:],cmap='jet')
      ax.set(xlabel='$x$',ylabel='$y$',xlim=x_bnds,ylim=y_bnds)
      ax.set(title='$\\rho$ at midplane')
      ax.axis('equal')
      fig.colorbar(tmp,ax=ax)

      ax = axes[1]
      tmp = ax.contourf(x[k0,:,:],y[k0,:,:],T[k0,:,:],cmap='hot')
      ax.set(xlabel='$x$',ylabel='$y$',xlim=x_bnds,ylim=y_bnds)
      ax.set(title='$T$ at midplane')
      ax.axis('equal')
      fig.colorbar(tmp,ax=ax)

      fname = '{}/rho-T-{:010d}.png'.format(runDir,iter[n])
      fig.savefig(fname)
      print('File written: {}'.format(fname))

# Check that (1) global mass and momentum are constant, and (2) correct energy is deposited
okay = True
tol = 1e-4
mass_rel_err = abs(mass[-1]-mass[0]) / mass_ref
mom_rel_err = np.max([abs(xmom[-1]),abs(ymom[-1]),abs(zmom[-1])]) / mom_ref
E_rel_err = abs(E_laser[-1]-E_laser_true) / E_ref
print('E_laser=',E_laser[-1],', E_laser_true=',E_laser_true)
if (mass_rel_err > tol):
   print('\n**WARNING**  mass_rel_err={}\n'.format(mass_rel_err))
   okay = False
if (mom_rel_err > tol):
   print('\n**WARNING**  mom_rel_err={}\n'.format(mom_rel_err))
   okay = False
if (E_rel_err > tol):
   print('\n**WARNING**  E_rel_err={}\n'.format(E_rel_err))
   okay = False
if (okay):
   print("\nAll quantities okay.")

