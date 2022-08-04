import cantera as ct
import numpy as np
import os
from scipy import interpolate

solution_name = "flame.xml"

gas = ct.Solution("reducedS43R464_0.cti")
gas.TP = 350, ct.one_atm
gas.X = {'O2':0.22, 'N2':0.78}
rhoRef = gas.density
species_names = gas.species_names

gas.set_equivalence_ratio(1.0, 'CH4', {'O2':1.0, 'N2':3.76})
gas.TP = 350, ct.one_atm
gas()

flame = ct.IonBurnerFlame(gas, width=1.0e-2)
flame.transport_model = 'Ion'
flame.burner.T = 350
flame.burner.mdot = 0.4104536576956711
flame.burner.Y = gas.Y
flame.set_refine_criteria(ratio=2.0, slope=0.05, curve=0.05, prune=0.01)
flame.delta_electric_potential = 250
if os.path.isfile(solution_name):
   flame.restore(solution_name)
else:
   flame.solve(stage=2)
   flame.save(solution_name)

uRef = np.sqrt(ct.one_atm/rhoRef)
print("u_0 = %10.5e"%(flame.velocity[0]/uRef))
Xi = flame.X[:,0]

def extrapolateSpecies(yp, y, X):
   f = interpolate.interp1d(y, X, "cubic", fill_value="extrapolate")
   return max(1e-60, f(yp))

for i, X in enumerate(Xi):
   X = extrapolateSpecies(-1.4181995066908476e-05, flame.grid, flame.X[i,:])
   if X > 1e-6:
      print('{{\"Name\" : \"{0:}\", \"MolarFrac\" : {1:10.5e} }},'.format(species_names[i], X))

print("u_o = %10.5e"%(flame.velocity[-1]/uRef))
print("T_o = %10.5e"%(flame.T[-1]/350))
Xo = flame.X[:,-1]
for i, X in enumerate(Xo):
   if X > 1e-6:
      print('{{\"Name\" : \"{0:}\", \"MolarFrac\" : {1:10.5e} }},'.format(species_names[i], X))

f = open("Ref.dat", "w")
f.write((19*"%15s"+"\n")%("y", "T", "rho", "Phi", "rho_q",
                          "X_CH4",  "X_O2", "X_N2",     "X_CH",    "X_CO",  "X_CO2", "X_H2O", "X_H2",
                          "X_H3O+", "X_E",  "X_C2H3O+", "X_CH5O+", "X_O2-", "X_OH-"))

electric_charge_density = flame.electric_charge_density

for i, y in enumerate(flame.grid):
   f.write((19*"%15.7e"+"\n")%(y,
                         flame.T[i],
                         flame.density[i],
                         flame.electric_potential[i],
                         electric_charge_density[i],
                         flame.X[species_names.index(   "CH4")][i],
                         flame.X[species_names.index(    "O2")][i],
                         flame.X[species_names.index(    "N2")][i],
                         flame.X[species_names.index(    "CH")][i],
                         flame.X[species_names.index(    "CO")][i],
                         flame.X[species_names.index(   "CO2")][i],
                         flame.X[species_names.index(   "H2O")][i],
                         flame.X[species_names.index(    "H2")][i],
                         flame.X[species_names.index(  "H3O+")][i],
                         flame.X[species_names.index(     "E")][i],
                         flame.X[species_names.index("C2H3O+")][i],
                         flame.X[species_names.index( "CH5O+")][i],
                         flame.X[species_names.index(   "O2-")][i],
                         flame.X[species_names.index(   "OH-")][i]))
f.close()

import sys
import json
import h5py
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import gridGen
import MulticomponentMix
import HTRrestart

with open("Speelman.json") as f:
   config = json.load(f)

mix = MulticomponentMix.Mix(config["Flow"]["mixture"])

xGrid, yGrid, zGrid, dx, dy, dz = gridGen.getCellCenters(config)

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################
def pressure(i, j, k):
   ct.one_atm/mix.PRef

T = np.interp(yGrid, gridIn, flame.T/mix.TRef)
def temperature(i, j, k):
   return T[j]

MFracs = np.ndarray((yGrid.size, Xi.size), dtype="f8")
for i, x in enumerate(Xi):
   MFracs[:,i] = np.interp(yGrid, gridIn, flame.X[i,:])
def MolarFracs(i, j, k):
   return  MFracs[j,:]

v = np.interp(yGrid, gridIn, flame.velocity/mix.uRef)
def velocity(i, j, k):
   return [ 0.0, v[j], 0.0]

Phi = np.interp(yGrid, gridIn, flame.electric_potential/mix.delPhi)
def electricPotential(i,j,k):
   return Phi[j]

restartDir="restart"
restart = HTRrestart.HTRrestart(config)
restart.write(restartDir, 43,
              pressure,
              temperature,
              MolarFracs,
              Phi = electricPotential,
              T_p = temperature,
              Xi_p = MolarFracs,
              U_p = velocity,
              nproc = 1)
