import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
import sys
import os
import h5py

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import HTRrestart

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_times', type=int, default=4)
args = parser.parse_args()

##############################################################################
#                          Compute Analytical Solution                       #
##############################################################################
def u(xy,mu,rho,time):
   return  np.sin(xy[:,:,0])*np.cos(xy[:,:,1])*np.exp(-2*mu/rho[:]*time)

def v(xy,mu,rho,time):
   return -np.cos(xy[:,:,0])*np.sin(xy[:,:,1])*np.exp(-2*mu/rho[:]*time)

def L2(err):
   tot = sum(map(sum, err**2))
   return np.sqrt(tot/err.size)

def process(case):
   dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/TaylorGreen2D')
   soleil_input_file = os.path.join(dir_name, 'TaylorGreen2D_'+str(case)+'.json')

##############################################################################
#                         Read Prometeo Input File                           #
##############################################################################

   with open(soleil_input_file) as f:
      data = json.load(f)

   xNum = data["Grid"]["xNum"]
   yNum = data["Grid"]["yNum"]
   xWidth  = data["Grid"]["GridInput"]["width"][0]
   yWidth  = data["Grid"]["GridInput"]["width"][1]
   constantVisc = data["Flow"]["mixture"]["viscosityModel"]["Visc"]

   Area = xWidth*yWidth

   dt    = data["Integrator"]["TimeStep"]["DeltaTime"]
   nstep = data["Integrator"]["maxIter"]
   time = dt*nstep

   filename = os.path.join(dir_name, str(case)+'/sample0/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-'+str(case-1)+','+str(case-1)+',0.hdf')
   hdf_filename = filename

##############################################################################
#                        Read Prometeo Output Data                           #
##############################################################################

   restart = HTRrestart.HTRrestart(data)
   restart.attach(sampleDir=os.path.join(dir_name, str(case)+'/sample0'), step=nstep)

   # Get the data
   centerCoordinates = restart.load('centerCoordinates')
   pressure          = restart.load('pressure')
   rho               = restart.load('rho')
   velocity          = restart.load('velocity')

   # Get dimension of data
   Nx = rho.shape[2]
   Ny = rho.shape[1]
   Nz = rho.shape[0]

   # Get simulation data along a line (ignore ghost cells)
   z_slice_idx = 0

   xy_slice  = centerCoordinates[z_slice_idx,:,:][:,:]
   u_slice   =          velocity[z_slice_idx,:,:][:,:]
   rho_slice =               rho[z_slice_idx,:,:]

##############################################################################
#                          Compute Analytical Solution                       #
##############################################################################

   u_slice_analytical = u(xy_slice, constantVisc, rho_slice, time)
   v_slice_analytical = v(xy_slice, constantVisc, rho_slice, time)

##############################################################################
#                               Compare Solutions                            #
##############################################################################

   uerr = (u_slice[:,:,0]-u_slice_analytical)
   verr = (u_slice[:,:,1]-v_slice_analytical)
   U_L2_error = L2(uerr)
   V_L2_error = L2(verr)
   print('U_L2 Error = {}'.format(U_L2_error))
   print('V_L2 Error = {}'.format(V_L2_error))

   return U_L2_error, V_L2_error

Nc = args.num_times
cases = []
U_L2 = []
V_L2 = []

cases.append(16)
for i in range(1,Nc):
   cases.append(cases[i-1]*2)

for i in range(Nc):
   U_L2.append(1.0)
   V_L2.append(1.0)
   U_L2[i], V_L2[i] = process(cases[i])

##############################################################################
#                                     Plot                                   #
##############################################################################

firOrd = []
secOrd = []
thiOrd = []
for i in range(Nc):
   firOrd.append(1.0/(cases[i]   )*1e-1)
   secOrd.append(1.0/(cases[i]**2)*1e-1)
   thiOrd.append(1.0/(cases[i]**3)*1e-1)

plt.figure(1)
plt.loglog(cases, U_L2, '-ob', label='u')
plt.loglog(cases, V_L2, '-or', label='v')
plt.loglog(cases,  firOrd, '-k' , label='$1^{st}$ ord')
plt.loglog(cases,  secOrd, '--k', label='$2^{nd}$ ord')
#plt.loglog(cases,  thiOrd, '-.k', label='$3^{rd}$ ord')
plt.xlabel(r'$Np$', fontsize = 20)
plt.ylabel(r'$L_2(err)$', fontsize = 20)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().set_xticks([16, 32, 64, 128])
plt.xlim([14,146])
plt.ylim([5e-6,1e-2])
plt.legend(loc=3)
plt.savefig('Convergence.eps', bbox_inches='tight')

plt.show()
