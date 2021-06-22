import numpy as np
import json
#import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.legend_handler import HandlerTuple
import sys
import os
import subprocess
import h5py
import pandas

# load local modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import MulticomponentMix

dir_name = os.path.join(os.environ['HTR_DIR'], 'testcases/Speelman_DV250')
input_file = os.path.join(dir_name, 'Speelman.json')
ref_file   = os.path.join(dir_name, 'Ref.dat')

##############################################################################
# Read HTR input file                                                        #
##############################################################################
with open(input_file) as f:
   data = json.load(f)

xNum = data["Grid"]["xNum"]
yNum = data["Grid"]["yNum"]
zNum = data["Grid"]["zNum"]

mix = MulticomponentMix.Mix(data["Flow"]["mixture"])

##############################################################################
# Read reference solution                                                    #
##############################################################################

Ref = pandas.read_csv(ref_file, delim_whitespace=True, encoding= "unicode_escape")

##############################################################################
# Process result file                                                        #
##############################################################################
def process(nstep):
   filename = os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-0,'+str(yNum+1)+',0.hdf')
#   filename = os.path.join(dir_name, 'data_old/fluid_iter'+str(nstep).zfill(10)+'/0,0,0-0,'+str(yNum+1)+',0.hdf')
   exists = os.path.isfile(filename)
   if (not exists):
      # merge files from different tiles
      merge_command = 'python {} {}'.format(os.path.expandvars('$HTR_DIR/scripts/merge.py'),
                                            os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/*.hdf'))
      mv_command = 'mv {} {}'.format('./0,0,0-'+str(xNum+1)+','+str(yNum+1)+',0.hdf',
                                    os.path.join(dir_name, 'sample0/fluid_iter'+str(nstep).zfill(10)+'/'))
      try:
         subprocess.call(merge_command, shell=True)
      except OSError:
         print("Failed command: {}".format(merge_command))
         sys.exit()
      try:
         subprocess.call(   mv_command, shell=True)
      except OSError:
         print("Failed command: {}".format(mv_command))
         sys.exit()

##############################################################################
# Read HTR output data                                                       #
##############################################################################

   f = h5py.File(filename, 'r')

   # Get the data
   x           = f["centerCoordinates"][:][0,:,0,1]
   pressure    = f["pressure"][:][0,:,0]
   temperature = f["temperature"][:][0,:,0]
   density     = f["rho"][:][0,:,0]
   velocity    = f["velocity"][:][0,:,0,:]
   molarFracs  = f["MolarFracs"][:][0,:,0,:]
   ePotential  = f["electricPotential"][:][0,:,0]

   return x*mix.LRef, pressure*mix.PRef, temperature*mix.TRef, density*mix.rhoRef, velocity*mix.uRef, molarFracs, ePotential*mix.delPhi, f.attrs.get("SpeciesNames")

##############################################################################
# Plot                                                                       #
##############################################################################

x , pressure , temperature , density , velocity , molarFracs , ePotential , specieNames  = process(800000)

plt.rcParams.update({'font.size': 16})

plt.figure(1)
plt.plot(x,        temperature, '-k', label="HTR solver")
plt.plot(Ref["y"], Ref["T"],    'ok', label="Cantera", markevery=5e-2)
plt.xlabel(r'$x [m]$', fontsize = 20)
plt.ylabel(r'$T [K]$', fontsize = 20)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
plt.gca().set_xlim(0, 5e-3)
plt.gca().set_ylim(350, 2400)
plt.legend()
plt.savefig('Temperature.eps', bbox_inches='tight')

plt.figure(2)
CH4my, = plt.semilogy(x, molarFracs[:, mix.FindSpecies("CH4", specieNames)], '-k')
CO2my, = plt.semilogy(x, molarFracs[:, mix.FindSpecies("CO2", specieNames)], '-r')
CHmy,  = plt.semilogy(x, molarFracs[:, mix.FindSpecies("CH",  specieNames)], '-b')
H2my,  = plt.semilogy(x, molarFracs[:, mix.FindSpecies("H2",  specieNames)], '-g')
CH4ca, = plt.semilogy(Ref["y"], Ref["X_CH4"], 'ok', markevery=0.05)
CO2ca, = plt.semilogy(Ref["y"], Ref["X_CO2"], 'or', markevery=0.05)
CHca,  = plt.semilogy(Ref["y"], Ref["X_CH"],  'ob', markevery=0.05)
H2ca,  = plt.semilogy(Ref["y"], Ref["X_H2"],  'og', markevery=0.05)
plt.xlabel(r'$x [m]$', fontsize = 20)
plt.ylabel(r'$X_i$', fontsize = 20)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
plt.gca().set_xlim(0, 5e-3)
plt.gca().set_ylim(1e-8, 0.5)
plt.legend([(CH4my, CH4ca), (CO2my, CO2ca), (CHmy, CHca), (H2my, H2ca),],
           [r"$X_{CH_4}$", r"$X_{CO_2}$", r"$X_{CH}$", r"$X_{H2}$"],
           handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
plt.savefig('MolarFractions.eps', bbox_inches='tight')

plt.figure(3)
plt.plot(x,        ePotential, '-k' , label="HTR solver")
plt.plot(Ref["y"], Ref["Phi"],  'ok', label="Cantera", markevery=5e-2)
plt.xlabel(r'$x [m]$', fontsize = 20)
plt.ylabel(r'$\Delta \Phi [V]$', fontsize = 20)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
plt.gca().set_xlim(0, 1e-2)
plt.legend()
plt.savefig('Phi.eps', bbox_inches='tight')

plt.figure(4)
Emy,    = plt.semilogy(x, molarFracs[:, mix.FindSpecies(    "E", specieNames)], '-k')
H3Omy,  = plt.semilogy(x, molarFracs[:, mix.FindSpecies( "H3O+", specieNames)], '-r')
O2Mmy,  = plt.semilogy(x, molarFracs[:, mix.FindSpecies(  "O2-", specieNames)], '-b')
CH5Omy, = plt.semilogy(x, molarFracs[:, mix.FindSpecies("CH5O+", specieNames)], '-g')
Eca,    = plt.semilogy(Ref["y"], Ref["X_E"    ], 'ok', markevery=0.05)
H3Oca,  = plt.semilogy(Ref["y"], Ref["X_H3O+" ], 'or', markevery=0.05)
O2Mca,  = plt.semilogy(Ref["y"], Ref["X_O2-"  ], 'ob', markevery=0.05)
CH5Oca, = plt.semilogy(Ref["y"], Ref["X_CH5O+"], 'og', markevery=0.05)
plt.xlabel(r'$x [m]$', fontsize = 20)
plt.ylabel(r'$X_i$', fontsize = 20)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
#plt.gca().set_xlim(0, 5e-3)
plt.gca().set_xlim(0, 1e-2)
plt.gca().set_ylim(1e-15, 1e-8)
plt.legend([(Emy, Eca), (H3Omy, H3Oca), (O2Mmy, O2Mca), (CH5Omy, CH5Oca)],
           [r"$X_{e^-}$", r"$X_{H_3O^+}$", r"$X_{O_2^-}$", r"$X_{CH_5O^+}$"],
           handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=8.0)
plt.savefig('Ions.eps', bbox_inches='tight')

plt.show()

