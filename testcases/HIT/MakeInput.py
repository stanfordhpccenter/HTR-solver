#!/usr/bin/env python3

import sys
import os
import json
import h5py
import argparse
import numpy as np
from joblib import Parallel, delayed, dump, load

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import ConstPropMix
import HTRrestart

parser = argparse.ArgumentParser()
parser.add_argument('--np', nargs='?', default=1, type=int, help='number of cores')
parser.add_argument('base_json', type=argparse.FileType('r'), default='base.json')
args = parser.parse_args()

##############################################################################
# Setup case                                                                 #
##############################################################################

# Read base config
config = json.load(args.base_json)

# Input parameters
ReLambda = config["Case"]["ReLambda"]
MaTurb   = config["Case"]["MaTurb"]
Tref     = config["Case"]["Tref"]
Pref     = config["Case"]["Pref"]

# Mixture parameters
R     = config["Flow"]["mixture"]["gasConstant"]
gamma = config["Flow"]["mixture"]["gamma"]
Pr    = config["Flow"]["mixture"]["prandtl"]

# Derived quantities
rho   = ConstPropMix.GetDensity(Tref, Pref, config)
mu    = ConstPropMix.GetViscosity(Tref, config)
SoS   = np.sqrt(gamma*R*Tref)
u_rms = MaTurb*SoS
TKE   = 0.5*rho*u_rms**2

# Domain width
L = 2*np.pi
config["Grid"]["GridInput"]["width"][0] = L
config["Grid"]["GridInput"]["width"][1] = L
config["Grid"]["GridInput"]["width"][2] = L

# Integral scales
ell = 0.2*L
u_l = np.sqrt(2.0*TKE/3)
t_l = ell/u_l
epsilon = u_l**2/ell

# Taylor microscales
lam = ReLambda*mu/rho/u_l

# Kolmogorov scales
l_k = ((mu/rho)**3/epsilon)**0.25
u_k = (epsilon*(mu/rho))**0.25
t_k = l_k/u_k

# Forcing setup
assert config["Flow"]["turbForcing"]["type"] == "HIT"
config["Flow"]["turbForcing"]["Gain"] = 500.0
config["Flow"]["turbForcing"]["t_o"] = t_l
config["Flow"]["turbForcing"]["K_o"] = TKE

# Simulation setup
assert config["Flow"]["initCase"]["type"] == "Restart"
restartDir = "InitialCase"
config["Flow"]["initCase"]["restartDir"] = restartDir

##############################################################################
# Boundary conditions                                                        #
##############################################################################
assert config["BC"]["xBCLeft"]["type"] == config["BC"]["xBCRight"]["type"] == "Periodic"
assert config["BC"]["yBCLeft"]["type"] == config["BC"]["yBCRight"]["type"] == "Periodic"
assert config["BC"]["zBCLeft"]["type"] == config["BC"]["zBCRight"]["type"] == "Periodic"

##############################################################################
# Generate Grid                                                              #
##############################################################################
xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

print("Check resolution")
print(" dx/eta = {}".format(dx[0]/l_k))

halo = [0, 0, 0]

##############################################################################
#                              Output input file                             #
##############################################################################

with open("Run.json", 'w') as fout:
   json.dump(config, fout, indent=3)

##############################################################################
# Compute isotropic turbulence                                               #
##############################################################################

def Spectrum(k, k0):
   u0 = 1
   coef = 16*np.sqrt(2/np.pi)*u0**2/k0/(2*np.pi*k0**2)
   kko = k/k0
   return coef*kko**4*np.exp(-2*kko**2)

def generate_isotropic_turbulence(k0, verbose=False):

   Nx = config["Grid"]["xNum"]
   Ny = config["Grid"]["yNum"]
   Nz = config["Grid"]["zNum"]

   deltax = dx[1]
   deltay = dy[1]
   deltaz = dz[1]

   knx = np.pi/Nx
   kny = np.pi/Ny
   knz = np.pi/Nz

   k1 = np.fft.fftshift(np.fft.fftfreq(Nx)*Nx)
   k2 = np.fft.fftshift(np.fft.fftfreq(Ny)*Ny)
   k3 = np.fft.fftshift(np.fft.fftfreq(Nz)*Nz)

   k_mag = np.reshape([np.sqrt(x**2 + y**2 + z**2) for z in k3 for y in k2 for x in k1], (Nz, Ny, Nx))
   Uhat_mag = np.sqrt(Spectrum(k_mag, k0))
   del k_mag

   theta = np.reshape(np.random.uniform(0.0, 2*np.pi, Nx*Ny*Nz*3), (Nz, Ny, Nx, 3))
   Uhat_mag_re = Uhat_mag*np.cos(theta[:,:,:,0])
   Uhat_mag_im = Uhat_mag*np.sin(theta[:,:,:,0])
   del Uhat_mag

   k_p1 = np.sin(k1*knx)/(0.5*np.pi*deltax)
   k_p2 = np.sin(k2*kny)/(0.5*np.pi*deltay)
   k_p3 = np.sin(k3*knz)/(0.5*np.pi*deltaz)

   Uhat = np.zeros([Nz,Ny,Nx], dtype=complex)
   Vhat = np.zeros([Nz,Ny,Nx], dtype=complex)
   What = np.zeros([Nz,Ny,Nx], dtype=complex)

   def computeUHat(i, Uhat, Vhat, What):
      for j in range(1, Ny):
         for k in range(1, Nz):
            K = [k_p3[k], k_p2[j], k_p1[i]]
            V1 = np.cross(np.random.uniform(0.0, 1.0, 3), K)
            V1 /= np.linalg.norm(V1)
            V2 = np.cross(V1, K)
            V2 /= np.linalg.norm(V2)
            Vr1 = V1*np.cos(theta[k,j,i,1]) + V2*np.sin(theta[k,j,i,1])
            Vr2 = V1*np.cos(theta[k,j,i,2]) + V2*np.sin(theta[k,j,i,2])
            Uhat[k,j,i] = complex(Vr1[2]*Uhat_mag_re[k,j,i], Vr2[2]*Uhat_mag_im[k,j,i])*np.exp(-1j*k1[i]*knx)
            Vhat[k,j,i] = complex(Vr1[1]*Uhat_mag_re[k,j,i], Vr2[1]*Uhat_mag_im[k,j,i])*np.exp(-1j*k2[j]*kny)
            What[k,j,i] = complex(Vr1[0]*Uhat_mag_re[k,j,i], Vr2[0]*Uhat_mag_im[k,j,i])*np.exp(-1j*k3[k]*knz)

   folder = './joblib_memmap'
   try:
      os.mkdir(folder)
   except FileExistsError:
      pass
   Uhat_filename_memmap = os.path.join(folder, 'uhat_memmap')
   Vhat_filename_memmap = os.path.join(folder, 'vhat_memmap')
   What_filename_memmap = os.path.join(folder, 'what_memmap')
   dump(Uhat, Uhat_filename_memmap)
   dump(Vhat, Vhat_filename_memmap)
   dump(What, What_filename_memmap)
   Uhat = np.memmap(Uhat_filename_memmap, dtype=complex, shape=Uhat.shape, mode='w+')
   Vhat = np.memmap(Vhat_filename_memmap, dtype=complex, shape=Uhat.shape, mode='w+')
   What = np.memmap(What_filename_memmap, dtype=complex, shape=Uhat.shape, mode='w+')
   Parallel(n_jobs=args.np, verbose=10)(delayed(computeUHat)(i, Uhat, Vhat, What) for i in range(1, int(Nx/2)))
   del theta
   del Uhat_mag_re
   del Uhat_mag_im

   def transposeModes(i, Uhat, Vhat, What):
      for j in range(1, Ny):
         for k in range(1, Nz):
            Uhat[k,j,i] = np.conj(Uhat[Nz-k,Ny-j,Nx-i])
            Vhat[k,j,i] = np.conj(Vhat[Nz-k,Ny-j,Nx-i])
            What[k,j,i] = np.conj(What[Nz-k,Ny-j,Nx-i])
   Parallel(n_jobs=args.np, verbose=10)(delayed(transposeModes)(i, Uhat, Vhat, What) for i in range(int(Nx/2)+1, Nx))

   i = int(Nx/2)
   for j in range(int(Ny/2)+1, Ny):
      for k in range(1, Nz):
         Uhat[k,j,i] = np.conj(Uhat[Nz-k,Ny-j,Nx-i])
         Vhat[k,j,i] = np.conj(Vhat[Nz-k,Ny-j,Nx-i])
         What[k,j,i] = np.conj(What[Nz-k,Ny-j,Nx-i])

   i = int(Nx/2)
   j = int(Ny/2)
   for k in range(int(Nz/2)+1, Nz):
      Uhat[k,j,i] = np.conj(Uhat[Nz-k,Ny-j,Nx-i])
      Vhat[k,j,i] = np.conj(Vhat[Nz-k,Ny-j,Nx-i])
      What[k,j,i] = np.conj(What[Nz-k,Ny-j,Nx-i])

   U = np.fft.ifftn(np.fft.ifftshift(Uhat))*Uhat.size
   V = np.fft.ifftn(np.fft.ifftshift(Vhat))*Vhat.size
   W = np.fft.ifftn(np.fft.ifftshift(What))*What.size

   # cleanup memmaps
   del Uhat
   del Vhat
   del What
   try:
      shutil.rmtree(folder)
   except:  # noqa
      print('Could not clean-up automatically.')

   if verbose:
      print("max(imag(U)) = {}".format(np.max(np.abs(np.imag(U)))))
      print("max(imag(V)) = {}".format(np.max(np.abs(np.imag(V)))))
      print("max(imag(W)) = {}".format(np.max(np.abs(np.imag(W)))))

   u = np.zeros([Nz+2*halo[2],Ny+2*halo[1],Nx+2*halo[0],3])
   u[halo[2]:Nz+halo[2],halo[1]:Ny+halo[1],halo[0]:Nx+halo[0],0] = np.real(U)
   u[halo[2]:Nz+halo[2],halo[1]:Ny+halo[1],halo[0]:Nx+halo[0],1] = np.real(V)
   u[halo[2]:Nz+halo[2],halo[1]:Ny+halo[1],halo[0]:Nx+halo[0],2] = np.real(W)

   if verbose:
      max_div = 0.0
      u_avg = np.mean(u[:,:,:,0])
      v_avg = np.mean(u[:,:,:,1])
      w_avg = np.mean(u[:,:,:,2])
      u_rms = np.mean(u[:,:,:,0]*u[:,:,:,0])
      v_rms = np.mean(u[:,:,:,1]*u[:,:,:,1])
      w_rms = np.mean(u[:,:,:,2]*u[:,:,:,2])
#      for i in range(1+halo[0], Nx-1+halo[0]):
#         for j in range(1+halo[1], Ny-1+halo[1]):
#            for k in range(1+halo[2], Nz-1+halo[2]):
##               div =((u[k  ,j  ,i+1,0] - u[k  ,j  ,i-1,0])/dx[i] +
##                     (u[k  ,j+1,i  ,1] - u[k  ,j-1,i  ,1])/dy[j] +
##                     (u[k+1,j  ,i  ,2] - u[k-1,j  ,i  ,2])/dz[k])
#               div =((u[k  ,j  ,i+1,0] - u[k  ,j  ,i  ,0])/dx[i] +
#                     (u[k  ,j+1,i  ,1] - u[k  ,j  ,i  ,1])/dy[j] +
#                     (u[k+1,j  ,i  ,2] - u[k  ,j  ,i  ,2])/dz[k])
#               max_div = max(abs(div), max_div)
      print("mean(u) = {}".format(u_avg))
      print("mean(v) = {}".format(v_avg))
      print("mean(w) = {}".format(w_avg))
      print("rms(u) = {}".format(np.sqrt(u_rms - u_avg**2)))
      print("rms(v) = {}".format(np.sqrt(v_rms - v_avg**2)))
      print("rms(w) = {}".format(np.sqrt(w_rms - w_avg**2)))
      print("max_div = {}".format(max_div))

   return u

uTurb = generate_isotropic_turbulence(10, True)

##############################################################################
#                                Produce restart                             #
##############################################################################

def pressure(i, j, k):
   return Pref

def temperature(i, j, k):
   return Tref

def MolarFracs(i, j, k):
   return 1.0

def velocity(i, j, k):
   return uTurb[i,j,k,:]*u_rms

restart = HTRrestart.HTRrestart(config)
restart.write(restartDir, 1,
              pressure,
              temperature,
              MolarFracs,
              velocity,
              nproc = 1)

