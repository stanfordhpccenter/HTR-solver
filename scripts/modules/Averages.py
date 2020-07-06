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

