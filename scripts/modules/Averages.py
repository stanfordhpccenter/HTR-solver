#!/usr/bin/env python2

import numpy as np
import h5py

class avg:
   def __init__(self, filename, symmetric = False):

      f = h5py.File(filename, "r")

      # Get the data
      weight               = f["weight"][:][0,:]
      self.pressure_avg    = f["pressure_avg"][:][0,:]
      self.pressure_rms    = f["pressure_rms"][:][0,:]
      self.temperature_avg = f["temperature_avg"][:][0,:]
      self.temperature_rms = f["temperature_rms"][:][0,:]
      self.MolarFracs_avg  = f["MolarFracs_avg"][:][0,:,:]
      self.MolarFracs_rms  = f["MolarFracs_rms"][:][0,:,:]
      self.velocity_avg    = f["velocity_avg"][:][0,:,:]
      self.velocity_rms    = f["velocity_rms"][:][0,:,:]
      self.velocity_rey    = f["velocity_rey"][:][0,:,:]

      self.rho_avg  = f["rho_avg"][:][0,:]
      self.rho_rms  = f["rho_rms"][:][0,:]
      self.mu_avg   = f["mu_avg"][:][0,:]
      self.mu_rms   = f["mu_rms"][:][0,:]
      self.lam_avg  = f["lam_avg"][:][0,:]
      self.lam_rms  = f["lam_rms"][:][0,:]
      self.Di_avg   = f["Di_avg"][:][0,:,:]
      self.Di_rms   = f["Di_rms"][:][0,:,:]
      self.SoS_avg  = f["SoS_avg"][:][0,:]
      self.SoS_rms  = f["SoS_rms"][:][0,:]

      self.temperature_favg = f["temperature_favg"][:][0,:]
      self.temperature_frms = f["temperature_frms"][:][0,:]
      self.MolarFracs_favg  = f["MolarFracs_favg"][:][0,:,:]
      self.MolarFracs_frms  = f["MolarFracs_frms"][:][0,:,:]
      self.velocity_favg    = f["velocity_favg"][:][0,:,:]
      self.velocity_frms    = f["velocity_frms"][:][0,:,:]
      self.velocity_frey    = f["velocity_frey"][:][0,:,:]

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

      self.SpeciesNames = f.attrs.get("SpeciesNames")

      if symmetric : self.avgYSymmetric()

      # Complete average process
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

      #self.pressure_rms    = np.sqrt(np.maximum(   self.pressure_rms -    self.pressure_avg**2, 0.0))
      #self.temperature_rms = np.sqrt(np.maximum(self.temperature_rms - self.temperature_avg**2, 0.0))
      #self.MolarFracs_rms  = np.sqrt(np.maximum( self.MolarFracs_rms -  self.MolarFracs_avg**2, 0.0))
      #self.velocity_rms    = np.sqrt(np.maximum(   self.velocity_rms -    self.velocity_avg**2, 0.0))
      self.pressure_rms    =    self.pressure_rms -    self.pressure_avg**2
      self.temperature_rms = self.temperature_rms - self.temperature_avg**2
      self.MolarFracs_rms  =  self.MolarFracs_rms -  self.MolarFracs_avg**2
      self.velocity_rms    =    self.velocity_rms -    self.velocity_avg**2

      self.rho_avg  /= weight
      self.rho_rms  /= weight 
      self.mu_avg  /= weight
      self.mu_rms  /= weight
      self.lam_avg /= weight
      self.lam_rms /= weight
      self.SoS_avg /= weight
      self.SoS_rms /= weight
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_avg[:,isp] /= weight 
         self.Di_rms[:,isp] /= weight 

      self.mu_rms  = np.sqrt(np.maximum( self.mu_rms -  self.mu_avg**2, 0.0))
      self.lam_rms = np.sqrt(np.maximum(self.lam_rms - self.lam_avg**2, 0.0))
      self.Di_rms  = np.sqrt(np.maximum( self.Di_rms -  self.Di_avg**2, 0.0))
      self.SoS_rms = np.sqrt(np.maximum(self.SoS_rms - self.SoS_avg**2, 0.0))

      self.temperature_favg      /= weight
      self.temperature_frms      /= weight
      for i in range(3):
         self.velocity_favg[:,i] /= weight
         self.velocity_frms[:,i] /= weight
         self.velocity_frey[:,i] /= weight

      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_favg[:,isp] /= weight
         self.MolarFracs_frms[:,isp] /= weight

      #self.temperature_frms = np.sqrt(np.maximum(self.temperature_frms - self.temperature_favg**2, 0.0))
      #self.MolarFracs_frms  = np.sqrt(np.maximum( self.MolarFracs_frms -  self.MolarFracs_favg**2, 0.0))
      #self.velocity_frms    = np.sqrt(np.maximum(   self.velocity_rms -    self.velocity_avg**2, 0.0))
      self.temperature_frms = self.temperature_frms - self.temperature_favg**2//self.rho_avg[:]
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_frms[:,isp] = self.MolarFracs_frms[:,isp] -  self.MolarFracs_favg[:,isp]**2/self.rho_avg[:]
      for i in range(3):
         self.velocity_frms[:,i] = self.velocity_frms[:,i] - self.velocity_favg[:,i]**2/self.rho_avg[:]

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

   def avgYSymmetric(self):
      self.pressure_avg    = 0.5*(self.pressure_avg + self.pressure_avg[::-1])
      self.pressure_rms    = 0.5*(self.pressure_rms + self.pressure_rms[::-1])
      self.temperature_avg = 0.5*(self.temperature_avg + self.temperature_avg[::-1])
      self.temperature_rms = 0.5*(self.temperature_rms + self.temperature_rms[::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_avg[:, isp] = 0.5*(self.MolarFracs_avg[:, isp] + self.MolarFracs_avg[::-1, isp])
         self.MolarFracs_rms[:, isp] = 0.5*(self.MolarFracs_rms[:, isp] + self.MolarFracs_rms[::-1, isp])
      self.velocity_avg[:,0] = 0.5*(self.velocity_avg[:,0] + self.velocity_avg[::-1,0])
      self.velocity_avg[:,1] = 0.5*(self.velocity_avg[:,1] - self.velocity_avg[::-1,1])
      self.velocity_avg[:,2] = 0.5*(self.velocity_avg[:,2] + self.velocity_avg[::-1,2])
      self.velocity_rms      = 0.5*(self.velocity_rms[:]   + self.velocity_rms[::-1])
      self.velocity_rey[:,0] = 0.5*(self.velocity_rey[:,0] - self.velocity_rey[::-1,0])
      self.velocity_rey[:,1] = 0.5*(self.velocity_rey[:,1] + self.velocity_rey[::-1,1])
      self.velocity_rey[:,2] = 0.5*(self.velocity_rey[:,2] - self.velocity_rey[::-1,2])

      self.rho_avg  = 0.5*(self.rho_avg + self.rho_avg[::-1]) 
      self.rho_rms  = 0.5*(self.rho_rms + self.rho_rms[::-1]) 
      self.mu_avg   = 0.5*(self.mu_avg  + self.mu_avg [::-1]) 
      self.mu_rms   = 0.5*(self.mu_rms  + self.mu_rms [::-1]) 
      self.lam_avg  = 0.5*(self.lam_avg + self.lam_avg[::-1]) 
      self.lam_rms  = 0.5*(self.lam_rms + self.lam_rms[::-1]) 
      for isp, sp in enumerate(self.SpeciesNames):
         self.Di_avg[:, isp] = 0.5*(self.Di_avg[:, isp]  + self.Di_avg [::-1, isp]) 
         self.Di_rms[:, isp] = 0.5*(self.Di_rms[:, isp]  + self.Di_rms [::-1, isp]) 
      self.SoS_avg  = 0.5*(self.SoS_avg + self.SoS_avg[::-1]) 
      self.SoS_rms  = 0.5*(self.SoS_rms + self.SoS_rms[::-1]) 

      self.temperature_favg = 0.5*(self.temperature_favg + self.temperature_favg[::-1])
      self.temperature_frms = 0.5*(self.temperature_frms + self.temperature_frms[::-1])
      for isp, sp in enumerate(self.SpeciesNames):
         self.MolarFracs_favg[:, isp] = 0.5*(self.MolarFracs_favg[:, isp] + self.MolarFracs_favg[::-1, isp])
         self.MolarFracs_frms[:, isp] = 0.5*(self.MolarFracs_frms[:, isp] + self.MolarFracs_frms[::-1, isp])
      self.velocity_favg[:,0] = 0.5*(self.velocity_favg[:,0] + self.velocity_favg[::-1,0])
      self.velocity_favg[:,1] = 0.5*(self.velocity_favg[:,1] - self.velocity_favg[::-1,1])
      self.velocity_favg[:,2] = 0.5*(self.velocity_favg[:,2] + self.velocity_favg[::-1,2])
      self.velocity_frms      = 0.5*(self.velocity_frms[:]   + self.velocity_frms[::-1])
      self.velocity_frey[:,0] = 0.5*(self.velocity_frey[:,0] - self.velocity_frey[::-1,0])
      self.velocity_frey[:,1] = 0.5*(self.velocity_frey[:,1] + self.velocity_frey[::-1,1])
      self.velocity_frey[:,2] = 0.5*(self.velocity_frey[:,2] - self.velocity_frey[::-1,2])

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

      self.Pr     = 0.5*(self.Pr     + self.Pr    ) 
      self.Pr_rms = 0.5*(self.Pr_rms + self.Pr_rms) 
      self.Ec     = 0.5*(self.Ec     + self.Ec    ) 
      self.Ec_rms = 0.5*(self.Ec_rms + self.Ec_rms) 

      return


