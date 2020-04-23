import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import sys
import os
import h5py
import pandas as pd

###########################################################################
###                         Read Exact Solution                          ##
###########################################################################

def readExact(name):
   # Open file
   f = open(name, 'r')
   x = []
   u = []
   # Loop over lines and extract variables of interest
   for line in f:
      line = line.strip()
      columns = line.split()
      x.append(  float(columns[0]))
      u.append(  float(columns[1]))
   f.close()
   return x, u

def Yi2Xi(Yi):
   W = np.array([2*14.0067e-3, 2*15.9994e-3, 14.0067e-3+15.9994e-3, 14.0067e-3, 15.9994e-3])
   Wmix = 1.0/sum(Yi/W)
   Xi = Yi*Wmix/W
   return Xi

def L2(err,dx):
   tot = sum(map(sum, err**2*dx[:,:,0]*dx[:,:,1]*dx[:,:,2]))
   vol = sum(map(sum,        dx[:,:,0]*dx[:,:,1]*dx[:,:,2]))
   return np.sqrt(tot/vol)

def process(case, i):
   dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/GrossmanCinnellaProblem')
   prometeo_input_file = os.path.join(dir_name, str(case)+'.json')

##############################################################################
#                         Read Prometeo Input File                           #
##############################################################################

   with open(prometeo_input_file) as f:
      data = json.load(f)

   xNum = data["Grid"]["xNum"]
   xWidth  = data["Grid"]["xWidth"]

   dt    = data["Integrator"]["fixedDeltaTime"]
   nstep = data["Integrator"]["maxIter"]
   time = dt*nstep

   filename = os.path.join(dir_name, str(case)+'/sample0/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-'+str(xNum+1)+',0,0.hdf')

##############################################################################
#                         Read Prometeo Output Data                          #
##############################################################################

   f = h5py.File(filename, 'r')

   SpNames = f.attrs.get("SpeciesNames")

   # Get the data
   x   = f['centerCoordinates'][0,0,:][:,0]
   dx  = f['cellWidth'        ][0,0,:][:,0]
   u   = f['velocity'         ][0,0,:][:,0]
   p   = f['pressure'         ][0,0,:]
   rho = f['rho'              ][0,0,:]
   T   = f['temperature'      ][0,0,:]
   Xi  = f['MolarFracs'       ][0,0,:][:,:]

   u   /= 3.477201e+02
   p   /= abs(  p[xNum-1])
   rho /= abs(rho[xNum-1])


   ##############################################################################
   #                                     Plot                                   #
   ##############################################################################
   
   plt.figure(1)
   xExact, uExact = readExact("u.csv")
   plt.plot(xExact, uExact,  'xk', label='Reference solution')
   plt.plot(     x,    u, '-b', label='HTR solver')
   plt.xlabel(r'$x$'    , fontsize = 20)
   plt.ylabel(r'$u/a_R$', fontsize = 20)
   plt.ylim(-0.25, 3.2) 
   plt.legend()
   plt.savefig('Velocity_'+str(case)+'.eps', bbox_inches='tight')
   
   plt.figure(2)
   xExact, pExact = readExact("p.csv")
   plt.plot(xExact, pExact,  'xk', label='Reference solution')
   plt.plot(     x,    p, '-b', label='HTR solver')
   plt.xlabel(r'$x$'    , fontsize = 20)
   plt.ylabel(r'$p/p_R$', fontsize = 20)
   plt.legend()
   plt.savefig('Pressure_'+str(case)+'.eps', bbox_inches='tight')
   
   plt.figure(3)
   xExact, rhoExact = readExact("rho.csv")
   plt.plot(xExact, rhoExact,  'xk', label='Reference solution')
   plt.plot(     x,        rho, '-b', label='HTR solver')
   plt.xlabel(r'$x$'          , fontsize = 20)
   plt.ylabel(r'$\rho/\rho_R$', fontsize = 20)
   plt.legend()
   plt.savefig('Density_'+str(case)+'.eps', bbox_inches='tight')
   
   plt.figure(4)
   plt.xlabel(r'$x$'  , fontsize = 20)
   plt.ylabel(r'$X_i$', fontsize = 20)
   df=pd.read_csv('Spec.csv', sep='\t',header=None)
   xExact  = np.array(df)[:,0]
   YiExact = np.array(df)[:,1:6]
   XiExact = YiExact
   for i, y in enumerate(YiExact):
      XiExact [i,:] = Yi2Xi(YiExact[i,:])
   line = []
   lable = []
   for isp, sp in enumerate(SpNames):
      l1, = plt.plot(     x, Xi[:,isp],       '-')
      l2, = plt.plot(xExact, XiExact[:,isp],  'x', color=l1.get_color())
      lable.append(r'$X_{'+sp.decode()+'}$')
      line.append((l1, l2))
   plt.legend(line,
              lable,
              handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
   plt.savefig('Species_'+str(case)+'.eps', bbox_inches='tight')

   return x, u, p, rho

plt.rcParams.update({'font.size': 12})
x, u, p, r = process(600, 0)
plt.show()
