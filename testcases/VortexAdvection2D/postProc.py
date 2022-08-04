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

beta = 5.0
xc = 0.0
yc = 0.0

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_times', type=int, default=4)
args = parser.parse_args()

def L2(err):
   tot = sum(map(sum, err**2))
   return np.sqrt(tot/err.size)

def L1(err):
   tot = sum(map(sum, err))
   vol = sum(map(sum,    ))
   return abs(tot)/err.size

def Linf(err):
   tot = max(map(max, abs(err)))
   return tot

def process(case):
   dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/VortexAdvection2D')
   prometeo_input_file = os.path.join(dir_name, 'VortexAdvection2D_'+str(case)+'.json')

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
   Tinf = data["Flow"]["initCase"]["temperature"]
   Uinf = data["Flow"]["initCase"]["velocity"][0]
   Vinf = data["Flow"]["initCase"]["velocity"][1]
   gamma = data["Flow"]["mixture"]["gamma"]

   dx = xWidth/xNum
   dy = yWidth/yNum
   Area = xWidth*yWidth

   dt    = data["Integrator"]["TimeStep"]["DeltaTime"]
   nstep = data["Integrator"]["maxIter"]
   time = dt*nstep

   ##############################################################################
   #                             Analytical Solutions                           #
   ##############################################################################
   def unroll(x,dx,width,origin) :
      xx = x - dx%width
      for i in range(len(x)):
         for j in range(len(x[i])):
            if xx[i][j] < origin :
               xx[i][j] += width
            elif xx[i][j] > width+origin :
               xx[i][j] -= width
      return xx

   def u(xy,time):
      x = unroll(xy[:,:,0], Uinf*time, xWidth, xOrigin)
      y = unroll(xy[:,:,1], Vinf*time, yWidth, yOrigin)
      rx = x - xc
      ry = y - yc
      r2 = rx**2 + ry**2
      return Uinf - beta/(2*np.pi)*np.exp(0.5*(1-r2))*(ry)

   def v(xy,time):
      x = unroll(xy[:,:,0], Uinf*time, xWidth, xOrigin)
      y = unroll(xy[:,:,1], Vinf*time, yWidth, yOrigin)
      rx = x - xc
      ry = y - yc
      r2 = rx**2 + ry**2
      return Vinf + beta/(2*np.pi)*np.exp(0.5*(1-r2))*(rx)

   def T(xy,time):
      x = unroll(xy[:,:,0], Uinf*time, xWidth, xOrigin)
      y = unroll(xy[:,:,1], Vinf*time, yWidth, yOrigin)
      rx = x - xc
      ry = y - yc
      r2 = rx**2 + ry**2
      return Tinf*(1.0 - (gamma-1)*beta**2/(8*gamma*np.pi**2)*np.exp(1-r2))

   def rho(xy,time):
      return T(xy,time)**(1.0/(gamma-1.0))

   def p(xy,time):
      return T(xy,time)**(gamma/(gamma-1.0))

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
   temperature       = restart.load('temperature')
   density           = restart.load('rho')
   velocity          = restart.load('velocity')

   # Get dimension of data
   Nx = density.shape[2]
   Ny = density.shape[1]
   Nz = density.shape[0]

   # Get simulation data along a line (ignore ghost cells)
   z_slice_idx = 0
   # Avoid the error due to periodic boundaries
   lo = int(0.10*Nx)
   hi = int(0.90*Nx)

   xy_slice   = centerCoordinates[z_slice_idx,lo:hi,lo:hi][:,:]
   u_slice    =          velocity[z_slice_idx,lo:hi,lo:hi][:,:]
   T_slice    =       temperature[z_slice_idx,lo:hi,lo:hi]
   p_slice    =          pressure[z_slice_idx,lo:hi,lo:hi]
   rho_slice  =           density[z_slice_idx,lo:hi,lo:hi]

##############################################################################
#                          Compute Analytical Solution                       #
##############################################################################

   u_slice_analytical = u(xy_slice, time)
   v_slice_analytical = v(xy_slice, time)
   T_slice_analytical =     T(xy_slice, time)
   p_slice_analytical =     p(xy_slice, time)
   rho_slice_analytical = rho(xy_slice, time)

##############################################################################
#                               Compare Solutions                            #
##############################################################################

   uerr = (  u_slice[:,:,0]-   u_slice_analytical)[:,:]
   verr = (  u_slice[:,:,1]-   v_slice_analytical)[:,:]
   Terr = (  T_slice       -   T_slice_analytical)[:,:]
   perr = (  p_slice       -   p_slice_analytical)[:,:]
   rhoerr = (rho_slice       - rho_slice_analytical)[:,:]

   U_L2_error   = L2(  uerr)
   V_L2_error   = L2(  verr)
   T_L2_error   = L2(  Terr)
   P_L2_error   = L2(  perr)
   Rho_L2_error = L2(rhoerr)
   print('U_L2 Error = {}'.format(U_L2_error))
   print('V_L2 Error = {}'.format(V_L2_error))
   print('T_L2 Error = {}'.format(T_L2_error))
   print('P_L2 Error = {}'.format(P_L2_error))
   print('Rho_L2 Error = {}'.format(Rho_L2_error))

   return U_L2_error, V_L2_error, T_L2_error, P_L2_error, Rho_L2_error

Nc = args.num_times
cases = []
U_L2 = []
V_L2 = []
T_L2 = []
P_L2 = []
Rho_L2 = []

cases.append(16)
for i in range(1,Nc):
   cases.append(cases[i-1]*2)

for i in range(Nc):
   U_L2.append(1.0)
   V_L2.append(1.0)
   T_L2.append(1.0)
   P_L2.append(1.0)
   Rho_L2.append(1.0)
   U_L2[i], V_L2[i], T_L2[i], P_L2[i], Rho_L2[i]  = process(cases[i])

##############################################################################
#                                     Plot                                   #
##############################################################################

firOrd = []
secOrd = []
thiOrd = []
fouOrd = []
fivOrd = []
sixOrd = []
for i in range(Nc):
   firOrd.append(1.0/(cases[i]   )*1e0)
   secOrd.append(1.0/(cases[i]**2)*1e0)
   thiOrd.append(1.0/(cases[i]**3)*1e2)
   fouOrd.append(1.0/(cases[i]**4)*1e3)
   fivOrd.append(1.0/(cases[i]**5)*1e4)
   sixOrd.append(1.0/(cases[i]**6)*1e5)

plt.figure(1)
plt.loglog(cases, U_L2, '-ob', label='U')
plt.loglog(cases, V_L2, '-or', label='V')
plt.loglog(cases, T_L2, '-oy', label='T')
plt.loglog(cases, P_L2, '-og', label='P')
#plt.loglog(cases,  firOrd, '-k' , label='$1^{st}$ ord')
#plt.loglog(cases,  secOrd, '--k', label='$2^{nd}$ ord')
plt.loglog(cases,  thiOrd,  '-k', label='$3^{rd}$ ord')
plt.loglog(cases,  fouOrd, '--k', label='$4^{rd}$ ord')
plt.loglog(cases,  fivOrd, '-.k', label='$5^{rd}$ ord')
plt.loglog(cases,  sixOrd,  ':k', label='$6^{rd}$ ord')
plt.xlabel(r'$Np$', fontsize = 20)
plt.ylabel(r'$L_2(err)$', fontsize = 20)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().set_xticks([16, 32, 64, 128, 256])
plt.legend()
plt.savefig('Convergence.eps', bbox_inches='tight')

plt.show()
