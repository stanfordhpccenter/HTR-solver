#!/usr/bin/env python3

import numpy as np

# Constants
RGAS = 8.3144598         # [J/(mol K)]
eps_0 = 8.8541878128e-12 # [F/m] or [C/(V m)]
eCrg  = 1.60217662e-19   # [C]
Na    = 6.02214086e23    # [1/mol]

MolarMass = { "N2"  : 0.0140067*2,                  # [J/(mol K)]
              "O2"  : 0.0159994*2,
              "N"   : 0.0140067,
              "O"   : 0.0159994,
              "NO"  : 0.0140067+0.0159994,
              "H"   : 0.00100784,
              "H2"  : 0.00100784*2,
              "CO"  : 0.0120107+0.0159994,
              "CO2" : 0.0120107+0.0159994*2,
              "CH4" : 0.0120107+0.00100784*4}

class Mix:
   def __init__(self, configMix):
      # Store reference quantities...
      # ... form the input file
      self.LRef  = configMix["LRef"]
      self.TRef  = configMix["TRef"]
      self.PRef  = configMix["PRef"]
      self.XiRef = configMix["XiRef"]["Species"]
      # ... and the derived once
      self.MixWRef = self.GetMolarWeightFromXiref(self.XiRef)
      self.rhoRef  = self.GetRhoRef(self.PRef, self.TRef, self.MixWRef)
      self.eRef    = self.PRef/self.rhoRef
      self.uRef    = np.sqrt(self.eRef)
      self.tRef    = self.LRef/self.uRef
      self.CpRef   = RGAS/self.MixWRef
      self.muRef   = self.LRef*np.sqrt(self.PRef*self.rhoRef)
      self.lamRef  = (self.LRef*np.sqrt(self.PRef*self.rhoRef)*RGAS)/self.MixWRef
      self.DiRef   = self.LRef/np.sqrt(self.rhoRef/self.PRef)
      self.KiRef   = np.sqrt(self.rhoRef/self.PRef)*Na*eCrg*self.LRef/self.MixWRef
      self.rhoqRef = Na*eCrg*self.rhoRef/self.MixWRef
      self.delPhi  = self.PRef*self.MixWRef/(self.rhoRef*Na*eCrg)
      self.wiRef   = np.sqrt(self.rhoRef*self.PRef)/self.LRef
      self.CrgRef  = Na*eCrg*self.rhoRef/self.MixWRef
      self.Eps0    = eps_0*self.PRef*(self.MixWRef/(self.rhoRef*Na*eCrg*self.LRef))**2

   def GetMolarWeightFromXiref(self, XiRef):
      MixW = 0.0
      for s in XiRef:
         assert (s["Name"] in MolarMass), "Please add the species " + s["Name"] + " in the MolarMass dictionary at $HTR_DIR/scripts/modules/MulticomponentMix.py"
         MixW +=  MolarMass[s["Name"]]*s["MolarFrac"]
      return MixW

   def GetRhoRef(self, P, T, MixW):
      return (P * MixW /(RGAS* T))

   def FindSpecies(self, name, speciesNames):
      return (speciesNames.tolist()).index(name.encode("utf-8"))

