# Imports
import numpy as np
import json
import sys
import os
import subprocess
import h5py
import pandas

# HTR modules
sys.path.insert(0, os.path.expandvars('$HTR_DIR/scripts/modules'))
import MulticomponentMix

# Plotting package
import matplotlib.pyplot as mplot
mplot.switch_backend('agg') # only if you get 'DISPLAY variable not defined' error
mplot.rcParams.update({'text.usetex': True,'font.family': 'Times New Roman','axes.titlesize': 14,\
   'axes.labelsize': 14,'legend.fontsize': 14,'xtick.labelsize': 11,'ytick.labelsize': 11})

# Directories, files
baseDir = '{}/testcases/PlanarFlame1D'.format(os.environ['HTR_DIR'])

# Which cases?
cases = [1,2,3,4] # corresponding to phi=[0.5,1.0,1.5,2.0] in MakeInput.py
firstIter = [200000,200000,200000,200000]
lastIter = [300000,300000,300000,300000]
outDir_prefix = 'out' # change me as needed

# Files and directories
inputFile_prefix = 'PlanarFlame1D'
figDir = 'figures'
refFile_prefix = 'FFCM1Mix-flame'

def flamePosition(x,T):
   Tmid = (min(T)+max(T))/2.0
   j = np.argmin(abs(T-Tmid))
   return x[j]

# Loop through all cases
nCases = len(cases)
driftSpeed = np.zeros([nCases])
sL = np.zeros([nCases])
sL_ref = np.zeros([nCases])
for i in range(nCases):
   print('')
   case = cases[i]

   ##############################################################################
   # Read reference (Cantera) solution
   refFile = 'ref/{}-{}.dat'.format(refFile_prefix,case)
   raw = np.loadtxt(refFile)
   print('File read: {}'.format(refFile))

   # Organize data
   ref = dict()
   ref['x'] = raw[:,0]
   ref['u'] = raw[:,1]
   ref['rho'] = raw[:,2]
   ref['p'] = raw[:,3]
   ref['T'] = raw[:,4]
   ref['XCH4'] = raw[:,5]
   ref['XO2'] = raw[:,6]
   ref['XH2O'] = raw[:,7]
   ref['XCO2'] = raw[:,8]
   ref['XH'] = raw[:,9]
   ref['XO'] = raw[:,10]
   ref['XOH'] = raw[:,11]
   ref['XCO'] = raw[:,12]

   ##############################################################################
   # Read HTR input file
   inputFile = '{}/{}-{}.json'.format(baseDir,inputFile_prefix,case)
   with open(inputFile) as f:
      config = json.load(f)

   xNum = config['Grid']['xNum']
   yNum = config['Grid']['yNum']
   zNum = config['Grid']['zNum']
   mix = MulticomponentMix.Mix(config['Flow']['mixture'])
   sL_ref[i] = config["BC"]["yBCLeft"]["VelocityProfile"]["velocity"][1] * mix.uRef

   ##############################################################################
   # First HTR time step -- get flame position only
   fname = '{}/{}-{}/sample0/fluid_iter{:010d}/0,0,0-0,{},0.hdf'.format(baseDir,outDir_prefix,case,\
      firstIter[i],yNum+1)
   F = h5py.File(fname, 'r') # Assumes a single tile
   x = F['centerCoordinates'][:][0,:,0,1] * mix.LRef
   t1 = 0.0
   T1 = F['temperature'][:][0,:,0] * mix.TRef
   xf1 = flamePosition(x,T1)

   ##############################################################################
   # Last time step -- get flame position and profile
   fname = '{}/{}-{}/sample0/fluid_iter{:010d}/0,0,0-0,{},0.hdf'.format(baseDir,outDir_prefix,case,\
      lastIter[i],yNum+1)
   F = h5py.File(fname, 'r') # Assumes a single tile
   T2 = F['temperature'][:][0,:,0] * mix.TRef
   t2 = 1.0
   xf2 = flamePosition(x,T2)

   # Calculate flame thickness
   dTdx_max = np.max((T2[2:]-T2[0:-2])/(x[2:]-x[0:-2]))
   thickness = (np.max(T2)-np.min(T2))/dTdx_max

   # Calculate speed at which flame drifts from being stationary
   driftSpeed[i] = (xf2-xf1)/(t2-t1)
   sL[i] = sL_ref[i] - driftSpeed[i]
   sL_error = driftSpeed[i]/sL_ref[i]

   # Announce
   print('sL={:10.4e}, drift/sL_ref={:8.2e}, thickness={:8.2e}'.format(sL[i],sL_error,thickness))
   if (np.abs(sL_error) > 0.05):
      print('**WARNING** Error in flame speed = {:8.2e}'.format(error))
   

   # Get flame profile
   htr = dict()
   speciesNames = F.attrs.get('SpeciesNames')
   htr['x'] = F['centerCoordinates'][:][0,:,0,1] * mix.LRef
   htr['p'] = F['pressure'][:][0,:,0] * mix.PRef
   htr['T'] = F['temperature'][:][0,:,0] * mix.TRef
   htr['rho'] = F['rho'][:][0,:,0] * mix.rhoRef
   htr['u'] = F['velocity'][:][0,:,0,1] * mix.uRef
   htr['XCH4'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('CH4',speciesNames)]
   htr['XO2'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('O2',speciesNames)]
   htr['XH2O'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('H2O',speciesNames)]
   htr['XCO2'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('CO2',speciesNames)]
   htr['XH'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('H',speciesNames)]
   htr['XO'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('O',speciesNames)]
   htr['XOH'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('OH',speciesNames)]
   htr['XCO'] = F['MolarFracs'][:][0,:,0,mix.FindSpecies('CO',speciesNames)]

   # Check mass conservation
   rhou = htr['rho']*htr['u']
   print('min={}, max={}'.format(np.min(rhou),np.max(rhou)))

   #print('x={}'.format(htr['x'][0:10]))
   #print('p={}'.format(htr['p'][0:10]))
   #print('T={}'.format(htr['T'][0:10]))
   #print('rho={}'.format(htr['rho'][0:10]))
   #print('u={}'.format(htr['u'][0:10]))
   #print('XCH4={}'.format(htr['XCH4'][0:10]))
   #print('XO2={}'.format(htr['XO2'][0:10]))

   # Shift Cantera data so that it is overlayed with HTR data
   xf_ref = flamePosition(ref['x'],ref['T'])
   ref['x'] += xf2 - xf_ref

   ##############################################################################
   # Plot flame profile
   ##############################################################################

   # Plotting bounds
   M_TO_MM = 1e3
   x_bnds = np.array([xf2-0.5e-3,xf2+0.5e-3])*M_TO_MM
   u_bnds = [0.0,1.1*np.max(ref['u'])]
   rho_bnds = [0.9*np.min(ref['rho']),1.1*np.max(ref['rho'])]
   p_bnds = [0.0,1.1*np.max(ref['p'])]
   T_bnds = [0.0,1.1*np.max(ref['T'])]

   # Initialize figure
   figSize = (12,6)
   fig,axes = mplot.subplots(nrows=2,ncols=3,figsize=figSize,sharex='col')
   mplot.tight_layout()
   fig.subplots_adjust(left=0.06,bottom=0.1,top=0.95,right=0.97,wspace=0.3,hspace=0.1)
   lw = 1.5
   ms = 4
   every = 3

   # u
   ax = axes[0,0]
   ax.plot(ref['x']*M_TO_MM,ref['u'],'ok',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(htr['x']*M_TO_MM,htr['u'],'-r',linewidth=lw)
   ax.legend(['Cantera','HTR'],loc='best')
   ax.set(xlim=x_bnds,ylim=u_bnds,ylabel='$u$\quad [m/s]')

   # rho
   ax = axes[1,0]
   ax.plot(ref['x']*M_TO_MM,ref['rho'],'ok',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(htr['x']*M_TO_MM,htr['rho'],'-r',linewidth=lw)
   ax.legend(['Cantera','HTR'],loc='best')
   ax.set(xlim=x_bnds,ylim=[0.01,2.0],xlabel='$x$\quad[mm]',ylabel='$\\rho$\quad [kg/m$^3$]')
   ax.set(yscale='log')

   # T
   ax = axes[0,1]
   ax.plot(ref['x']*M_TO_MM,ref['T'],'ok',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(htr['x']*M_TO_MM,htr['T'],'-r',linewidth=lw)
   ax.legend(['Cantera','HTR'],loc='best')
   ax.set(xlim=x_bnds,ylim=T_bnds,ylabel='$T$\quad [K]')

   # p
   ax = axes[1,1]
   ax.plot(ref['x']*M_TO_MM,ref['p'],'ok',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(htr['x']*M_TO_MM,htr['p'],'-r',linewidth=lw)
   ax.legend(['Cantera','HTR'],loc='best')
   ax.set(xlim=x_bnds,ylim=p_bnds,xlabel='$x$\quad[mm]',ylabel='$p$\quad [Pa]')

   # Major
   ax = axes[0,2]
   ax.plot(htr['x']*M_TO_MM,htr['XCH4'],'-',color='red',linewidth=lw)
   ax.plot(htr['x']*M_TO_MM,htr['XO2'],'-',color='green',linewidth=lw)
   ax.plot(htr['x']*M_TO_MM,htr['XH2O'],'-',color='blue',linewidth=lw)
   ax.plot(htr['x']*M_TO_MM,htr['XCO2'],'-',color='grey',linewidth=lw)
   ax.plot(ref['x']*M_TO_MM,ref['XCH4'],'o',color='red',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(ref['x']*M_TO_MM,ref['XO2'],'o',color='green',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(ref['x']*M_TO_MM,ref['XH2O'],'o',color='blue',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(ref['x']*M_TO_MM,ref['XCO2'],'o',color='grey',linewidth=lw,markersize=ms,markevery=every)
   ax.legend(['$X_{CH_4}$','$X_{O_2}$','$X_{H_2O}$','$X_{CO_2}$'],loc='best')
   ax.set(xlim=x_bnds,ylim=[0.0,1.0],ylabel='$X_i$')

   # Minor
   ax = axes[1,2]
   ax.plot(htr['x']*M_TO_MM,htr['XH'],'-',color='red',linewidth=lw)
   ax.plot(htr['x']*M_TO_MM,htr['XO'],'-',color='green',linewidth=lw)
   ax.plot(htr['x']*M_TO_MM,htr['XOH'],'-',color='blue',linewidth=lw)
   ax.plot(ref['x']*M_TO_MM,ref['XH'],'o',color='red',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(ref['x']*M_TO_MM,ref['XO'],'o',color='green',linewidth=lw,markersize=ms,markevery=every)
   ax.plot(ref['x']*M_TO_MM,ref['XOH'],'o',color='blue',linewidth=lw,markersize=ms,markevery=every)
   ax.legend(['$X_{H}$','$X_{O}$','$X_{OH}$'],loc='best')
   ax.set(xlim=x_bnds,ylim=[1e-6,1e0],yscale='log',xlabel='$x$\quad[mm]',ylabel='$X_i$')

   fname = '{}/flame-profile-{}.png'.format(figDir,case)
   fig.savefig(fname)
   print('File written: {}'.format(fname))


