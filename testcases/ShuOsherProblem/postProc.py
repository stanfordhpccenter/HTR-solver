import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
import h5py

###############################################################################
##                              Read Exact Solution                          ##
###############################################################################

def readExact():
   # Open file
   f = open('exact.txt', 'r')

   x = []
   u = []
   p = []
   rho = []
   e = []

   # Loop over lines and extract variables of interest
   for line in f:
      line = line.strip()
      columns = line.split()
      x.append(  float(columns[0]))
      u.append(  float(columns[1]))
      p.append(  float(columns[2]))
      rho.append(float(columns[3]))

   f.close()

   e = [xp/xr/0.4 for xp, xr in zip(p,rho)]

   return x, u, p, rho, e

###############################################################################
#                               Read Exact Solution                           #
###############################################################################

xExact, uExact, pExact, rhoExact, eExact = readExact()

def L2(err,dx):
   tot = sum(map(sum, err**2*dx[:,:,0]*dx[:,:,1]*dx[:,:,2]))
   vol = sum(map(sum,        dx[:,:,0]*dx[:,:,1]*dx[:,:,2]))
   return np.sqrt(tot/vol)

def process(case, i):
   dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/ShuOsherProblem')
   soleil_input_file = os.path.join(dir_name, str(case)+'.json')

##############################################################################
#                         Read Prometeo Input File                           #
##############################################################################

   with open(soleil_input_file) as f:
      data = json.load(f)

   xNum = data["Grid"]["xNum"]
   xWidth  = data["Grid"]["GridInput"]["width"][0]

   nstep = data["Integrator"]["maxIter"]

   filename = os.path.join(dir_name, str(case)+'/sample0/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-'+str(case+1)+',0,0.hdf')
   hdf_filename = filename

##############################################################################
#                        Read Prometeo Output Data                           #
##############################################################################

   f = h5py.File(hdf_filename, 'r')

   # Get the data
   x   = f['centerCoordinates'][0,0,:][:,0]
   u   = f['velocity'         ][0,0,:][:,0]
   p   = f['pressure'         ][0,0,:]
   rho = f['rho'              ][0,0,:]
   T   = f['temperature'      ][0,0,:]

   e = T/0.4

   ##############################################################################
   #                                     Plot                                   #
   ##############################################################################

   plt.figure(1+i*4)
   plt.plot(xExact, uExact,  '-k', label='Reference solution')
   plt.plot(     x,      u, '-ob', label='HTR solver', markersize=4.5)
   plt.xlabel(r'$x$', fontsize = 20)
   plt.ylabel(r'$u$', fontsize = 20)
   plt.legend()
   plt.savefig('Velocity_'+str(case)+'.eps', bbox_inches='tight')

   plt.figure(2+i*4)
   plt.plot(xExact, pExact,  '-k', label='Reference solution')
   plt.plot(     x,      p, '-ob', label='HTR solver', markersize=4.5)
   plt.xlabel(r'$x$', fontsize = 20)
   plt.ylabel(r'$P$', fontsize = 20)
   plt.legend()
   plt.savefig('Pressure_'+str(case)+'.eps', bbox_inches='tight')

   plt.figure(3+i*4)
   plt.plot(xExact, rhoExact, '-k', label='Reference solution')
   plt.plot(     x,     rho, '-ob', label='HTR solver', markersize=4.5)
   plt.xlabel(r'$x$'   , fontsize = 20)
   plt.ylabel(r'$\rho$', fontsize = 20)
   plt.legend()
   plt.savefig('Density_'+str(case)+'.eps', bbox_inches='tight')

   plt.figure(4+i*4)
   plt.plot(xExact, rhoExact, '-k', label='Reference solution')
   plt.plot(     x,     rho, '-ob', label='HTR solver', markersize=4.5)
   plt.xlabel(r'$x$'   , fontsize = 20)
   plt.ylabel(r'$\rho$', fontsize = 20)
   plt.xlim(4.5, 8.0)
   plt.ylim(3.0, 5.0)
   plt.legend()
   plt.savefig('DensityZoom_'+str(case)+'.eps', bbox_inches='tight')

   plt.figure(5+i*4)
   plt.plot(xExact,  eExact,  '-k', label='Reference solution')
   plt.plot(     x,       e, '-ob', label='HTR solver', markersize=4.5)
   plt.xlabel(r'$x$', fontsize = 20)
   plt.ylabel(r'$e$', fontsize = 20)
   plt.legend()
   plt.savefig('InternalEnergy_'+str(case)+'.eps', bbox_inches='tight')

   return x, u, p, rho, e

plt.rcParams.update({'font.size': 12})
x, u, p, r, e = process(200, 0)
plt.show()
