#!/usr/bin/env python3

import os
import re
import h5py
import glob
import numpy as np

class avg2D:
   def __init__(self, filename, symmetric = False):

      f = h5py.File(filename, "r")

      # Get the data
      weight = f["weight"][:][0,:]

      self.centerCoordinates = f["centerCoordinates"][:][0,:,:]

      self.pressure_avg    = f["pressure_avg"][:][0,:]
      self.pressure_rms    = f["pressure_rms"][:][0,:]
      self.temperature_avg = f["temperature_avg"][:][0,:]
      self.temperature_rms = f["temperature_rms"][:][0,:]
      self.MolarFracs_avg  = f["MolarFracs_avg"][:][0,:,:]
      self.MolarFracs_rms  = f["MolarFracs_rms"][:][0,:,:]
      self.MassFracs_avg   = f["MassFracs_avg"][:][0,:,:]
      self.MassFracs_rms   = f["MassFracs_rms"][:][0,:,:]
      self.velocity_avg    = f["velocity_avg"][:][0,:,:]
      self.velocity_rms    = f["velocity_rms"][:][0,:,:]
      self.velocity_rey    = f["velocity_rey"][:][0,:,:]

      self.rho_avg = f["rho_avg"][:][0,:]
      self.rho_rms = f["rho_rms"][:][0,:]
      self.mu_avg  = f["mu_avg"][:][0,:]
      self.lam_avg = f["lam_avg"][:][0,:]
      self.Di_avg  = f["Di_avg"][:][0,:,:]
      self.SoS_avg = f["SoS_avg"][:][0,:]
      self.cp_avg  = f["cp_avg"][:][0,:]
      self.Ent_avg = f["Ent_avg"][:][0,:]

      self.ProductionRates_avg = f["ProductionRates_avg"][:][0,:,:]
      self.ProductionRates_rms = f["ProductionRates_rms"][:][0,:,:]
      self.HeatReleaseRate_avg = f["HeatReleaseRate_avg"][:][0,:]
      self.HeatReleaseRate_rms = f["HeatReleaseRate_rms"][:][0,:]

      self.pressure_favg    = f["pressure_favg"][:][0,:]
      self.pressure_frms    = f["pressure_frms"][:][0,:]
      self.temperature_favg = f["temperature_favg"][:][0,:]
      self.temperature_frms = f["temperature_frms"][:][0,:]
      self.MolarFracs_favg  = f["MolarFracs_favg"][:][0,:,:]
      self.MolarFracs_frms  = f["MolarFracs_frms"][:][0,:,:]
      self.MassFracs_favg   = f["MassFracs_favg"][:][0,:,:]
      self.MassFracs_frms   = f["MassFracs_frms"][:][0,:,:]
      self.velocity_favg    = f["velocity_favg"][:][0,:,:]
      self.velocity_frms    = f["velocity_frms"][:][0,:,:]
      self.velocity_frey    = f["velocity_frey"][:][0,:,:]

      self.mu_favg  = f["mu_favg"][:][0,:]
      self.lam_favg = f["lam_favg"][:][0,:]
      self.Di_favg  = f["Di_favg"][:][0,:,:]
      self.SoS_favg = f["SoS_favg"][:][0,:]
      self.cp_favg  = f["cp_favg"][:][0,:]
      self.Ent_favg = f["Ent_favg"][:][0,:]

      self.rhoUUv   = f["rhoUUv"][:][0,:,:]
      self.Up       = f["Up"][:][0,:,:]
      self.tau      = f["tau"][:][0,:,:]
      self.utau_y   = f["utau_y"][:][0,:,:]
      self.tauGradU = f["tauGradU"][:][0,:,:]
      self.pGradU   = f["pGradU"][:][0,:,:]

      self.q = f["q"][:][0,:,:]

      self.Pr     = f["Pr"][:][0,:]
      self.Pr_rms = f["Pr_rms"][:][0,:]
      self.Ec     = f["Ec"][:][0,:]
      self.Ec_rms = f["Ec_rms"][:][0,:]
      self.Ma     = f["Ma"][:][0,:]
      self.Sc     = f["Sc"][:][0,:,:]

      self.uT_avg   = f["uT_avg"][:][0,:]
      self.uYi_avg  = f["uYi_avg"][:][0,:]
      self.vYi_avg  = f["vYi_avg"][:][0,:]
      self.wYi_avg  = f["wYi_avg"][:][0,:]

      self.uT_favg  = f["uT_favg"][:][0,:]
      self.uYi_favg = f["uYi_favg"][:][0,:]
      self.vYi_favg = f["vYi_favg"][:][0,:]
      self.wYi_favg = f["wYi_favg"][:][0,:]

      if "electricPotential_avg" in f:
         self.electricPotential_avg = f["electricPotential_avg"][:][0,:]
         self.chargeDensity_avg     = f["chargeDensity_avg"    ][:][0,:]

      self.SpeciesNames = f.attrs.get("SpeciesNames")

      if symmetric : self.avgYSymmetric()

      # Complete average process
      for i in range(3):
         self.centerCoordinates[:,i] /= weight

      self.pressure_avg         /= weight
      self.pressure_rms         /= weight
      self.temperature_avg      /= weight
      self.temperature_rms      /= weight
      for i in range(3):
         self.velocity_avg[:,i] /= weight
         self.velocity_rms[:,i] /= weight
         self.velocity_rey[:,i] /= weight

      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_avg[:,isp] /= weight
         self.MolarFracs_rms[:,isp] /= weight
         self.MassFracs_avg[ :,isp] /= weight
         self.MassFracs_rms[ :,isp] /= weight

      self.pressure_rms    -=    self.pressure_avg**2
      self.temperature_rms -= self.temperature_avg**2
      self.MolarFracs_rms  -=  self.MolarFracs_avg**2
      self.MassFracs_rms   -=   self.MassFracs_avg**2
      self.velocity_rms    -=    self.velocity_avg**2
      self.velocity_rey[:,0] -=  self.velocity_avg[:,0]*self.velocity_avg[:,1]
      self.velocity_rey[:,1] -=  self.velocity_avg[:,0]*self.velocity_avg[:,2]
      self.velocity_rey[:,2] -=  self.velocity_avg[:,1]*self.velocity_avg[:,2]

      self.rho_avg  /= weight
      self.rho_rms  /= weight
      self.mu_avg   /= weight
      self.lam_avg  /= weight
      self.SoS_avg  /= weight
      self.cp_avg   /= weight
      self.Ent_avg  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_avg[:,isp] /= weight

      self.rho_rms = self.rho_rms - self.rho_avg**2

      self.HeatReleaseRate_avg /= weight
      self.HeatReleaseRate_rms /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.ProductionRates_avg[:,isp] /= weight
         self.ProductionRates_rms[:,isp] /= weight

      self.ProductionRates_rms -= self.ProductionRates_avg**2
      self.HeatReleaseRate_rms -= self.HeatReleaseRate_avg**2

      self.pressure_favg      /= weight
      self.pressure_frms      /= weight
      self.temperature_favg   /= weight
      self.temperature_frms   /= weight
      for i in range(3):
         self.velocity_favg[:,i] /= weight
         self.velocity_frms[:,i] /= weight
         self.velocity_frey[:,i] /= weight

      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_favg[:,isp] /= weight
         self.MolarFracs_frms[:,isp] /= weight
         self.MassFracs_favg[ :,isp] /= weight
         self.MassFracs_frms[ :,isp] /= weight

      self.pressure_frms    -=    self.pressure_favg**2/self.rho_avg[:]
      self.temperature_frms -= self.temperature_favg**2/self.rho_avg[:]
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_frms[:,isp] -= self.MolarFracs_favg[:,isp]**2/self.rho_avg[:]
         self.MassFracs_frms[ :,isp] -= self.MassFracs_favg[ :,isp]**2/self.rho_avg[:]
      for i in range(3):
         self.velocity_frms[:,i] -= self.velocity_favg[:,i]**2/self.rho_avg[:]
      self.velocity_frey[:,0] -= self.velocity_favg[:,0]*self.velocity_favg[:,1]/self.rho_avg[:]
      self.velocity_frey[:,1] -= self.velocity_favg[:,0]*self.velocity_favg[:,2]/self.rho_avg[:]
      self.velocity_frey[:,2] -= self.velocity_favg[:,1]*self.velocity_favg[:,2]/self.rho_avg[:]

      self.mu_favg   /= weight
      self.lam_favg  /= weight
      self.SoS_favg  /= weight
      self.cp_favg   /= weight
      self.Ent_favg  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_favg[:,isp] /= weight

      for i in range(3):
         self.rhoUUv[:,i]   /= weight
         self.Up[:,i]       /= weight
         self.utau_y[:,i]   /= weight
         self.tauGradU[:,i] /= weight
         self.pGradU[:,i]   /= weight

      for i in range(6):
         self.tau[:,i]      /= weight

      for i in range(3):
         self.q[:,i] /= weight

      self.Pr     /= weight
      self.Pr_rms /= weight
      self.Ec     /= weight
      self.Ec_rms /= weight
      self.Ma     /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Sc[:,isp] /= weight

      self.Pr_rms -= self.Pr**2
      self.Ec_rms -= self.Ec**2

      for i in range(3):
         self.uT_avg[:,i]  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.uYi_avg[:,isp] /= weight
         self.vYi_avg[:,isp] /= weight
         self.wYi_avg[:,isp] /= weight

      for i in range(3):
         self.uT_favg[:,i]  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.uYi_favg[:,isp] /= weight
         self.vYi_favg[:,isp] /= weight
         self.wYi_favg[:,isp] /= weight

      if hasattr(self, "electricPotential_avg"):
         self.electricPotential_avg /= weight
         self.chargeDensity_avg     /= weight

   def avgYSymmetric(self):
      self.pressure_avg    = 0.5*(self.pressure_avg + self.pressure_avg[::-1])
      self.pressure_rms    = 0.5*(self.pressure_rms + self.pressure_rms[::-1])
      self.temperature_avg = 0.5*(self.temperature_avg + self.temperature_avg[::-1])
      self.temperature_rms = 0.5*(self.temperature_rms + self.temperature_rms[::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_avg[:, isp] = 0.5*(self.MolarFracs_avg[:, isp] + self.MolarFracs_avg[::-1, isp])
         self.MolarFracs_rms[:, isp] = 0.5*(self.MolarFracs_rms[:, isp] + self.MolarFracs_rms[::-1, isp])
         self.MassFracs_avg[ :, isp] = 0.5*(self.MassFracs_avg[ :, isp] + self.MassFracs_avg[ ::-1, isp])
         self.MassFracs_rms[ :, isp] = 0.5*(self.MassFracs_rms[ :, isp] + self.MassFracs_rms[ ::-1, isp])
      self.velocity_avg[:,0] = 0.5*(self.velocity_avg[:,0] + self.velocity_avg[::-1,0])
      self.velocity_avg[:,1] = 0.5*(self.velocity_avg[:,1] - self.velocity_avg[::-1,1])
      self.velocity_avg[:,2] = 0.5*(self.velocity_avg[:,2] + self.velocity_avg[::-1,2])
      self.velocity_rms      = 0.5*(self.velocity_rms[:]   + self.velocity_rms[::-1])
      self.velocity_rey[:,0] = 0.5*(self.velocity_rey[:,0] - self.velocity_rey[::-1,0])
      self.velocity_rey[:,1] = 0.5*(self.velocity_rey[:,1] + self.velocity_rey[::-1,1])
      self.velocity_rey[:,2] = 0.5*(self.velocity_rey[:,2] - self.velocity_rey[::-1,2])

      self.rho_avg  = 0.5*(self.rho_avg  + self.rho_avg [::-1])
      self.rho_rms  = 0.5*(self.rho_rms  + self.rho_rms [::-1])
      self.mu_avg   = 0.5*(self.mu_avg   + self.mu_avg  [::-1])
      self.lam_avg  = 0.5*(self.lam_avg  + self.lam_avg [::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_avg[:, isp] = 0.5*(self.Di_avg[:, isp] + self.Di_avg[::-1, isp])
      self.SoS_avg  = 0.5*(self.SoS_avg  + self.SoS_avg [::-1])
      self.cp_avg   = 0.5*(self.cp_avg   + self.cp_avg  [::-1])
      self.Ent_avg  = 0.5*(self.Ent_avg  + self.Ent_avg [::-1])

      self.ProductionRates_avg = 0.5*(self.ProductionRates_avg + self.ProductionRates_avg[::-1])
      self.ProductionRates_rms = 0.5*(self.ProductionRates_rms + self.ProductionRates_rms[::-1])
      self.HeatReleaseRate_avg = 0.5*(self.HeatReleaseRate_avg + self.HeatReleaseRate_avg[::-1])
      self.HeatReleaseRate_rms = 0.5*(self.HeatReleaseRate_rms + self.HeatReleaseRate_rms[::-1])

      self.pressure_favg    = 0.5*(self.pressure_favg    + self.pressure_favg[   ::-1])
      self.pressure_frms    = 0.5*(self.pressure_frms    + self.pressure_frms[   ::-1])
      self.temperature_favg = 0.5*(self.temperature_favg + self.temperature_favg[::-1])
      self.temperature_frms = 0.5*(self.temperature_frms + self.temperature_frms[::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_favg[:, isp] = 0.5*(self.MolarFracs_favg[:, isp] + self.MolarFracs_favg[::-1, isp])
         self.MolarFracs_frms[:, isp] = 0.5*(self.MolarFracs_frms[:, isp] + self.MolarFracs_frms[::-1, isp])
         self.MassFracs_favg[ :, isp] = 0.5*(self.MassFracs_favg[ :, isp] + self.MassFracs_favg[ ::-1, isp])
         self.MassFracs_frms[ :, isp] = 0.5*(self.MassFracs_frms[ :, isp] + self.MassFracs_frms[ ::-1, isp])
      self.velocity_favg[:,0] = 0.5*(self.velocity_favg[:,0] + self.velocity_favg[::-1,0])
      self.velocity_favg[:,1] = 0.5*(self.velocity_favg[:,1] - self.velocity_favg[::-1,1])
      self.velocity_favg[:,2] = 0.5*(self.velocity_favg[:,2] + self.velocity_favg[::-1,2])
      self.velocity_frms      = 0.5*(self.velocity_frms[:]   + self.velocity_frms[::-1])
      self.velocity_frey[:,0] = 0.5*(self.velocity_frey[:,0] - self.velocity_frey[::-1,0])
      self.velocity_frey[:,1] = 0.5*(self.velocity_frey[:,1] + self.velocity_frey[::-1,1])
      self.velocity_frey[:,2] = 0.5*(self.velocity_frey[:,2] - self.velocity_frey[::-1,2])

      self.mu_favg   = 0.5*(self.mu_favg   + self.mu_favg  [::-1])
      self.lam_favg  = 0.5*(self.lam_favg  + self.lam_favg [::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_favg[:, isp] = 0.5*(self.Di_favg[:, isp] + self.Di_favg[::-1, isp])
      self.SoS_favg  = 0.5*(self.SoS_favg  + self.SoS_favg [::-1])
      self.cp_favg   = 0.5*(self.cp_favg   + self.cp_favg  [::-1])
      self.Ent_favg  = 0.5*(self.Ent_favg  + self.Ent_favg [::-1])

      self.rhoUUv[:,:]   = 0.5*(  self.rhoUUv[:,:] -   self.rhoUUv[::-1,:])
      self.Up[:,0]       = 0.5*(      self.Up[:,0] +       self.Up[::-1,0])
      self.Up[:,1]       = 0.5*(      self.Up[:,1] -       self.Up[::-1,1])
      self.Up[:,2]       = 0.5*(      self.Up[:,2] +       self.Up[::-1,2])
      self.tau[:,0]      = 0.5*(     self.tau[:,0] +      self.tau[::-1,0])
      self.tau[:,1]      = 0.5*(     self.tau[:,1] +      self.tau[::-1,1])
      self.tau[:,2]      = 0.5*(     self.tau[:,2] +      self.tau[::-1,2])
      self.tau[:,3]      = 0.5*(     self.tau[:,3] -      self.tau[::-1,3])
      self.tau[:,4]      = 0.5*(     self.tau[:,4] -      self.tau[::-1,4])
      self.tau[:,5]      = 0.5*(     self.tau[:,5] +      self.tau[::-1,5])
      self.utau_y[:,0]   = 0.5*(  self.utau_y[:,0] -   self.utau_y[::-1,0])
      self.utau_y[:,1]   = 0.5*(  self.utau_y[:,1] -   self.utau_y[::-1,1])
      self.utau_y[:,2]   = 0.5*(  self.utau_y[:,2] -   self.utau_y[::-1,2])
      self.tauGradU[:,0] = 0.5*(self.tauGradU[:,0] + self.tauGradU[::-1,0])
      self.tauGradU[:,1] = 0.5*(self.tauGradU[:,1] + self.tauGradU[::-1,1])
      self.tauGradU[:,2] = 0.5*(self.tauGradU[:,2] + self.tauGradU[::-1,2])
      self.pGradU[:,0]   = 0.5*(  self.pGradU[:,0] +   self.pGradU[::-1,0])
      self.pGradU[:,1]   = 0.5*(  self.pGradU[:,1] -   self.pGradU[::-1,1])
      self.pGradU[:,2]   = 0.5*(  self.pGradU[:,2] +   self.pGradU[::-1,2])

      self.q[:,0] = 0.5*(self.q[:,0] + self.q[::-1,0])
      self.q[:,1] = 0.5*(self.q[:,1] - self.q[::-1,1])
      self.q[:,2] = 0.5*(self.q[:,2] + self.q[::-1,2])

      self.Pr     = 0.5*(self.Pr     + self.Pr    [::-1])
      self.Pr_rms = 0.5*(self.Pr_rms + self.Pr_rms[::-1])
      self.Ec     = 0.5*(self.Ec     + self.Ec    [::-1])
      self.Ec_rms = 0.5*(self.Ec_rms + self.Ec_rms[::-1])
      self.Ma     = 0.5*(self.Ma     + self.Ma    [::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.Sc[:, isp] = 0.5*(self.Sc[:, isp] + self.Sc[::-1, isp])

      self.uT_avg [:,0] = 0.5*(self.uT_avg [:,0] + self.uT_avg [::-1,0])
      self.uT_avg [:,1] = 0.5*(self.uT_avg [:,1] - self.uT_avg [::-1,1])
      self.uT_avg [:,2] = 0.5*(self.uT_avg [:,2] + self.uT_avg [::-1,2])
      self.uYi_avg[:,:] = 0.5*(self.uYi_avg[:,:] + self.uYi_avg[::-1,:])
      self.vYi_avg[:,:] = 0.5*(self.vYi_avg[:,:] - self.vYi_avg[::-1,:])
      self.wYi_avg[:,:] = 0.5*(self.wYi_avg[:,:] + self.wYi_avg[::-1,:])

      self.uT_favg [:,0] = 0.5*(self.uT_favg [:,0] + self.uT_favg [::-1,0])
      self.uT_favg [:,1] = 0.5*(self.uT_favg [:,1] - self.uT_favg [::-1,1])
      self.uT_favg [:,2] = 0.5*(self.uT_favg [:,2] + self.uT_favg [::-1,2])
      self.uYi_favg[:,:] = 0.5*(self.uYi_favg[:,:] + self.uYi_favg[::-1,:])
      self.vYi_favg[:,:] = 0.5*(self.vYi_favg[:,:] - self.vYi_favg[::-1,:])
      self.wYi_favg[:,:] = 0.5*(self.wYi_favg[:,:] + self.wYi_favg[::-1,:])

      if hasattr(self, "electricPotential_avg"):
         self.electricPotential_avg = 0.5*(self.electricPotential_avg + self.electricPotential_avg[::-1])
         self.chargeDensity_avg     = 0.5*(self.chargeDensity_avg     + self.chargeDensity_avg    [::-1])

class avg1D:
   def parseTiles(self, dirname, plane):
      # read tiles and join them
      self.tiles = glob.glob(os.path.join(dirname,"*,"+str(plane)+".hdf"))
      assert(self.tiles != 0)
      self.lo_bound = [] # array(len(tiles), array(2,int))
      self.hi_bound = [] # array(len(tiles), array(2,int))
      for i, t in enumerate(self.tiles):
         base = os.path.basename(t)
         m = re.match(r'([0-9]+),([0-9]+),([0-9]+)-([0-9]+),([0-9]+),([0-9]+).hdf',
                      base)
         assert(m is not None)
         assert(int(m.group(3)) == plane)
         assert(int(m.group(6)) == plane)
         self.lo_bound.append([int(m.group(1)), int(m.group(2))])
         self.hi_bound.append([int(m.group(4)), int(m.group(5))])

      # Sanity checks
      all_lo = [None, None] # array(2, array(len(tiles),int))
      all_hi = [None, None] # array(2, array(len(tiles),int))
      # bounds must be contiguous
      for k in range(2):
         all_lo[k] = sorted(set([c[k] for c in self.lo_bound]))
         all_hi[k] = sorted(set([c[k] for c in self.hi_bound]))
         assert len(all_lo[k]) == len(all_hi[k])
         for (prev_hi,next_lo) in zip(all_hi[k][:-1],all_lo[k][1:]):
            assert prev_hi == next_lo - 1
      # Check that we have the files for each combination of bounds
      for (x_lo,x_hi) in zip(all_lo[0],all_hi[0]):
         for (y_lo,y_hi) in zip(all_lo[1],all_hi[1]):
            found = False
            for i in range(len(self.tiles)):
               if (self.lo_bound[i][0] == x_lo and self.hi_bound[i][0] == x_hi and
                   self.lo_bound[i][1] == y_lo and self.hi_bound[i][1] == y_hi):
                  found = True
                  break
            assert found

      # combine data in a single data structure
      self.shape = (all_hi[1][-1] - all_lo[1][0] + 1,
                    all_hi[0][-1] - all_lo[0][0] + 1)

   def __init__(self, dirname, plane):

      # Parse input files
      self.parseTiles(dirname, plane)

      # Get the data
      with h5py.File(self.tiles[0], "r") as fin:
         self.SpeciesNames = fin.attrs.get("SpeciesNames")
         self.nSpec = len(self.SpeciesNames)
         hasElectric = ("electricPotential_avg" in fin)

      # define data plane
      weight = np.ndarray(self.shape)

      self.centerCoordinates = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))

      self.pressure_avg    = np.ndarray(self.shape)
      self.pressure_rms    = np.ndarray(self.shape)
      self.temperature_avg = np.ndarray(self.shape)
      self.temperature_rms = np.ndarray(self.shape)
      self.MolarFracs_avg  = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.MolarFracs_rms  = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.MassFracs_avg   = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.MassFracs_rms   = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.velocity_avg    = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.velocity_rms    = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.velocity_rey    = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))

      self.rho_avg = np.ndarray(self.shape)
      self.rho_rms = np.ndarray(self.shape)
      self.mu_avg  = np.ndarray(self.shape)
      self.lam_avg = np.ndarray(self.shape)
      self.Di_avg  = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.SoS_avg = np.ndarray(self.shape)
      self.cp_avg  = np.ndarray(self.shape)
      self.Ent_avg = np.ndarray(self.shape)

      self.ProductionRates_avg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.ProductionRates_rms = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.HeatReleaseRate_avg = np.ndarray(self.shape)
      self.HeatReleaseRate_rms = np.ndarray(self.shape)

      self.pressure_favg    = np.ndarray(self.shape)
      self.pressure_frms    = np.ndarray(self.shape)
      self.temperature_favg = np.ndarray(self.shape)
      self.temperature_frms = np.ndarray(self.shape)
      self.MolarFracs_favg  = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.MolarFracs_frms  = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.MassFracs_favg   = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.MassFracs_frms   = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.velocity_favg    = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.velocity_frms    = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.velocity_frey    = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))

      self.mu_favg  = np.ndarray(self.shape)
      self.lam_favg = np.ndarray(self.shape)
      self.Di_favg  = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.SoS_favg = np.ndarray(self.shape)
      self.cp_favg  = np.ndarray(self.shape)
      self.Ent_favg = np.ndarray(self.shape)

      self.rhoUUv   = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.Up       = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.tau      = np.ndarray(self.shape, dtype=np.dtype("(6,)f8"))
      self.utau_y   = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.tauGradU = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.pGradU   = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))

      self.q = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))

      self.Pr     = np.ndarray(self.shape)
      self.Pr_rms = np.ndarray(self.shape)
      self.Ec     = np.ndarray(self.shape)
      self.Ec_rms = np.ndarray(self.shape)
      self.Ma     = np.ndarray(self.shape)
      self.Sc     = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))

      self.uT_avg  = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.uYi_avg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.vYi_avg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.wYi_avg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))

      self.uT_favg  = np.ndarray(self.shape, dtype=np.dtype("(3,)f8"))
      self.uYi_favg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.vYi_favg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))
      self.wYi_favg = np.ndarray(self.shape, dtype=np.dtype("("+str(self.nSpec)+",)f8"))

      if hasElectric:
         self.electricPotential_avg = np.ndarray(self.shape)
         self.chargeDensity_avg     = np.ndarray(self.shape)

      for i, t in enumerate(self.tiles):
         f = h5py.File(t, "r")
         ind = (slice(self.lo_bound[i][1], self.hi_bound[i][1]+1),
                slice(self.lo_bound[i][0], self.hi_bound[i][0]+1))
         for t, n in enumerate(self.SpeciesNames):
            assert(n == f.attrs.get("SpeciesNames")[t])

         weight[ind] = f["weight"][:][0,:,:]

         self.centerCoordinates[ind] = f["centerCoordinates"][:][0,:,:]

         self.pressure_avg[ind]    = f["pressure_avg"][:][0,:,:]
         self.pressure_rms[ind]    = f["pressure_rms"][:][0,:,:]
         self.temperature_avg[ind] = f["temperature_avg"][:][0,:,:]
         self.temperature_rms[ind] = f["temperature_rms"][:][0,:,:]
         self.MolarFracs_avg[ind]  = f["MolarFracs_avg"][:][0,:,:,:]
         self.MolarFracs_rms[ind]  = f["MolarFracs_rms"][:][0,:,:,:]
         self.MassFracs_avg[ind]   = f["MassFracs_avg"][:][0,:,:,:]
         self.MassFracs_rms[ind]   = f["MassFracs_rms"][:][0,:,:,:]
         self.velocity_avg[ind]    = f["velocity_avg"][:][0,:,:,:]
         self.velocity_rms[ind]    = f["velocity_rms"][:][0,:,:,:]
         self.velocity_rey[ind]    = f["velocity_rey"][:][0,:,:,:]

         self.rho_avg[ind] = f["rho_avg"][:][0,:,:]
         self.rho_rms[ind] = f["rho_rms"][:][0,:,:]
         self.mu_avg[ind]  = f["mu_avg"][:][0,:,:]
         self.lam_avg[ind] = f["lam_avg"][:][0,:,:]
         self.Di_avg[ind]  = f["Di_avg"][:][0,:,:,:]
         self.SoS_avg[ind] = f["SoS_avg"][:][0,:,:]
         self.cp_avg[ind]  = f["cp_avg"][:][0,:,:]
         self.Ent_avg[ind] = f["Ent_avg"][:][0,:,:]

         self.ProductionRates_avg[ind] = f["ProductionRates_avg"][:][0,:,:,:]
         self.ProductionRates_rms[ind] = f["ProductionRates_rms"][:][0,:,:,:]
         self.HeatReleaseRate_avg[ind] = f["HeatReleaseRate_avg"][:][0,:,:]
         self.HeatReleaseRate_rms[ind] = f["HeatReleaseRate_rms"][:][0,:,:]

         self.pressure_favg[ind]    = f["pressure_favg"][:][0,:,:]
         self.pressure_frms[ind]    = f["pressure_frms"][:][0,:,:]
         self.temperature_favg[ind] = f["temperature_favg"][:][0,:,:]
         self.temperature_frms[ind] = f["temperature_frms"][:][0,:,:]
         self.MolarFracs_favg[ind]  = f["MolarFracs_favg"][:][0,:,:,:]
         self.MolarFracs_frms[ind]  = f["MolarFracs_frms"][:][0,:,:,:]
         self.MassFracs_favg[ind]   = f["MassFracs_favg"][:][0,:,:,:]
         self.MassFracs_frms[ind]   = f["MassFracs_frms"][:][0,:,:,:]
         self.velocity_favg[ind]    = f["velocity_favg"][:][0,:,:,:]
         self.velocity_frms[ind]    = f["velocity_frms"][:][0,:,:,:]
         self.velocity_frey[ind]    = f["velocity_frey"][:][0,:,:,:]

         self.mu_favg[ind]  = f["mu_favg"][:][0,:,:]
         self.lam_favg[ind] = f["lam_favg"][:][0,:,:]
         self.Di_favg[ind]  = f["Di_favg"][:][0,:,:,:]
         self.SoS_favg[ind] = f["SoS_favg"][:][0,:,:]
         self.cp_favg[ind]  = f["cp_favg"][:][0,:,:]
         self.Ent_favg[ind] = f["Ent_favg"][:][0,:,:]

         self.rhoUUv[ind]   = f["rhoUUv"][:][0,:,:,:]
         self.Up[ind]       = f["Up"][:][0,:,:,:]
         self.tau[ind]      = f["tau"][:][0,:,:,:]
         self.utau_y[ind]   = f["utau_y"][:][0,:,:,:]
         self.tauGradU[ind] = f["tauGradU"][:][0,:,:,:]
         self.pGradU[ind]   = f["pGradU"][:][0,:,:,:]

         self.q[ind] = f["q"][:][0,:,:,:]

         self.Pr[ind]     = f["Pr"][:][0,:,:]
         self.Pr_rms[ind] = f["Pr_rms"][:][0,:,:]
         self.Ec[ind]     = f["Ec"][:][0,:,:]
         self.Ec_rms[ind] = f["Ec_rms"][:][0,:,:]
         self.Ma[ind]     = f["Ma"][:][0,:,:]
         self.Sc[ind]     = f["Sc"][:][0,:,:,:]

         self.uT_avg[ind]   = f["uT_avg"][:][0,:,:]
         self.uYi_avg[ind]  = f["uYi_avg"][:][0,:,:]
         self.vYi_avg[ind]  = f["vYi_avg"][:][0,:,:]
         self.wYi_avg[ind]  = f["wYi_avg"][:][0,:,:]

         self.uT_favg[ind]  = f["uT_favg"][:][0,:,:]
         self.uYi_favg[ind] = f["uYi_favg"][:][0,:,:]
         self.vYi_favg[ind] = f["vYi_favg"][:][0,:,:]
         self.wYi_favg[ind] = f["wYi_favg"][:][0,:,:]

         if hasElectric:
            self.electricPotential_avg[ind] = f["electricPotential_avg"][:][0,:,:]
            self.chargeDensity_avg[ind]     = f["chargeDensity_avg"    ][:][0,:,:]

      # Complete average process
      for i in range(3):
         self.centerCoordinates[:,:,i] /= weight[:,:]

      self.pressure_avg         /= weight
      self.pressure_rms         /= weight
      self.temperature_avg      /= weight
      self.temperature_rms      /= weight
      for i in range(3):
         self.velocity_avg[:,:,i] /= weight
         self.velocity_rms[:,:,i] /= weight
         self.velocity_rey[:,:,i] /= weight

      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_avg[:,:,isp] /= weight
         self.MolarFracs_rms[:,:,isp] /= weight
         self.MassFracs_avg[ :,:,isp] /= weight
         self.MassFracs_rms[ :,:,isp] /= weight

      self.pressure_rms    -=    self.pressure_avg**2
      self.temperature_rms -= self.temperature_avg**2
      self.MolarFracs_rms  -=  self.MolarFracs_avg**2
      self.MassFracs_rms   -=   self.MassFracs_avg**2
      self.velocity_rms    -=    self.velocity_avg**2
      self.velocity_rey[:,:,0] -=  self.velocity_avg[:,:,0]*self.velocity_avg[:,:,1]
      self.velocity_rey[:,:,1] -=  self.velocity_avg[:,:,0]*self.velocity_avg[:,:,2]
      self.velocity_rey[:,:,2] -=  self.velocity_avg[:,:,1]*self.velocity_avg[:,:,2]

      self.rho_avg  /= weight
      self.rho_rms  /= weight
      self.mu_avg   /= weight
      self.lam_avg  /= weight
      self.SoS_avg  /= weight
      self.cp_avg   /= weight
      self.Ent_avg  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_avg[:,:,isp] /= weight

      self.rho_rms = self.rho_rms - self.rho_avg**2

      self.HeatReleaseRate_avg /= weight
      self.HeatReleaseRate_rms /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.ProductionRates_avg[:,:,isp] /= weight
         self.ProductionRates_rms[:,:,isp] /= weight

      self.ProductionRates_rms -= self.ProductionRates_avg**2
      self.HeatReleaseRate_rms -= self.HeatReleaseRate_avg**2

      self.pressure_favg      /= weight
      self.pressure_frms      /= weight
      self.temperature_favg   /= weight
      self.temperature_frms   /= weight
      for i in range(3):
         self.velocity_favg[:,:,i] /= weight
         self.velocity_frms[:,:,i] /= weight
         self.velocity_frey[:,:,i] /= weight

      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_favg[:,:,isp] /= weight
         self.MolarFracs_frms[:,:,isp] /= weight
         self.MassFracs_favg[ :,:,isp] /= weight
         self.MassFracs_frms[ :,:,isp] /= weight

      self.pressure_frms    -=    self.pressure_favg**2/self.rho_avg[:,:]
      self.temperature_frms -= self.temperature_favg**2/self.rho_avg[:,:]
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_frms[:,:,isp] -= self.MolarFracs_favg[:,:,isp]**2/self.rho_avg[:,:]
         self.MassFracs_frms[ :,:,isp] -= self.MassFracs_favg[ :,:,isp]**2/self.rho_avg[:,:]
      for i in range(3):
         self.velocity_frms[:,:,i] -= self.velocity_favg[:,:,i]**2/self.rho_avg[:,:]
      self.velocity_frey[:,:,0] -= self.velocity_favg[:,:,0]*self.velocity_favg[:,:,1]/self.rho_avg[:,:]
      self.velocity_frey[:,:,1] -= self.velocity_favg[:,:,0]*self.velocity_favg[:,:,2]/self.rho_avg[:,:]
      self.velocity_frey[:,:,2] -= self.velocity_favg[:,:,1]*self.velocity_favg[:,:,2]/self.rho_avg[:,:]

      self.mu_favg   /= weight
      self.lam_favg  /= weight
      self.SoS_favg  /= weight
      self.cp_favg   /= weight
      self.Ent_favg  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_favg[:,:,isp] /= weight

      for i in range(3):
         self.rhoUUv[:,:,i]   /= weight
         self.Up[:,:,i]       /= weight
         self.utau_y[:,:,i]   /= weight
         self.tauGradU[:,:,i] /= weight
         self.pGradU[:,:,i]   /= weight

      for i in range(6):
         self.tau[:,:,i]      /= weight

      for i in range(3):
         self.q[:,:,i] /= weight

      self.Pr     /= weight
      self.Pr_rms /= weight
      self.Ec     /= weight
      self.Ec_rms /= weight
      self.Ma     /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Sc[:,:,isp] /= weight

      self.Pr_rms -= self.Pr**2
      self.Ec_rms -= self.Ec**2

      for i in range(3):
         self.uT_avg[:,:,i]  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.uYi_avg[:,:,isp] /= weight
         self.vYi_avg[:,:,isp] /= weight
         self.wYi_avg[:,:,isp] /= weight

      for i in range(3):
         self.uT_favg[:,:,i]  /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.uYi_favg[:,:,isp] /= weight
         self.vYi_favg[:,:,isp] /= weight
         self.wYi_favg[:,:,isp] /= weight

      if hasElectric:
         self.electricPotential_avg /= weight
         self.chargeDensity_avg     /= weight
