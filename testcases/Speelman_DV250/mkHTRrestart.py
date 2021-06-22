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

with open("Speelman.json") as f:
   config = json.load(f)

mix = MulticomponentMix.Mix(config["Flow"]["mixture"])

xGrid, dx = gridGen.GetGrid(config["Grid"]["origin"][0],
                            config["Grid"]["xWidth"],
                            config["Grid"]["xNum"],
                            config["Grid"]["xType"],
                            config["Grid"]["xStretching"],
                            True)

yGrid, dy = gridGen.GetGrid(config["Grid"]["origin"][1],
                            config["Grid"]["yWidth"],
                            config["Grid"]["yNum"],
                            config["Grid"]["yType"],
                            config["Grid"]["yStretching"],
                            False)

zGrid, dz = gridGen.GetGrid(config["Grid"]["origin"][2],
                            config["Grid"]["zWidth"],
                            config["Grid"]["zNum"],
                            config["Grid"]["zType"],
                            config["Grid"]["yStretching"],
                            True)

# Load mapping
assert config["Mapping"]["tiles"][0] % config["Mapping"]["tilesPerRank"][0] == 0
assert config["Mapping"]["tiles"][1] % config["Mapping"]["tilesPerRank"][1] == 0
assert config["Mapping"]["tiles"][2] % config["Mapping"]["tilesPerRank"][2] == 0
Ntiles = config["Mapping"]["tiles"]
Ntiles[0] = int(Ntiles[0]/config["Mapping"]["tilesPerRank"][0])
Ntiles[1] = int(Ntiles[1]/config["Mapping"]["tilesPerRank"][1])
Ntiles[2] = int(Ntiles[2]/config["Mapping"]["tilesPerRank"][2])

assert config["Grid"]["xNum"] % Ntiles[0] == 0
assert config["Grid"]["yNum"] % Ntiles[1] == 0
assert config["Grid"]["zNum"] % Ntiles[2] == 0

NxTile = int(config["Grid"]["xNum"]/Ntiles[0])
NyTile = int(config["Grid"]["yNum"]/Ntiles[1])
NzTile = int(config["Grid"]["zNum"]/Ntiles[2])

halo = [0, 1, 0]

##############################################################################
#                     Produce restart and profile files                      #
##############################################################################

restartDir="restart"

if not os.path.exists(restartDir):
   os.makedirs(restartDir)

def writeTile(xt, yt, zt):
   lo_bound = [(xt  )*NxTile  +halo[0], (yt  )*NyTile  +halo[1], (zt  )*NzTile  +halo[2]]
   hi_bound = [(xt+1)*NxTile-1+halo[0], (yt+1)*NyTile-1+halo[1], (zt+1)*NzTile-1+halo[2]]
   if (xt == 0): lo_bound[0] -= halo[0]
   if (yt == 0): lo_bound[1] -= halo[1]
   if (zt == 0): lo_bound[2] -= halo[2]
   if (xt == Ntiles[0]-1): hi_bound[0] += halo[0]
   if (yt == Ntiles[1]-1): hi_bound[1] += halo[1]
   if (zt == Ntiles[2]-1): hi_bound[2] += halo[2]
   filename = ('%s,%s,%s-%s,%s,%s.hdf'
      % (lo_bound[0],  lo_bound[1],  lo_bound[2],
         hi_bound[0],  hi_bound[1],  hi_bound[2]))
   print("Working on: ", filename)

   shape = [hi_bound[2] - lo_bound[2] +1,
            hi_bound[1] - lo_bound[1] +1,
            hi_bound[0] - lo_bound[0] +1]

   gridIn = flame.grid/mix.LRef

   with h5py.File(os.path.join(restartDir, filename), 'w') as fout:
      fout.attrs.create("timeStep", 0)
      fout.attrs.create("simTime", 0.0)
      fout.attrs.create("channelForcing", 0.0)

      fout.create_dataset("centerCoordinates",     shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("cellWidth",             shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("(43,)f8"))
      fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("MolarFracs_profile",    shape=shape, dtype = np.dtype("(43,)f8"))
      fout.create_dataset("velocity_profile",      shape=shape, dtype = np.dtype("(3,)f8"))
      fout.create_dataset("temperature_profile",   shape=shape, dtype = np.dtype("f8"))
      fout.create_dataset("electricPotential",     shape=shape, dtype = np.dtype("f8"))

      fout["centerCoordinates"][:] = np.reshape([(x,y,z)
                                 for z in zGrid[lo_bound[2]:hi_bound[2]+1]
                                 for y in yGrid[lo_bound[1]:hi_bound[1]+1]
                                 for x in xGrid[lo_bound[0]:hi_bound[0]+1]],
                              (shape[0], shape[1], shape[2], 3))

      fout["cellWidth"][:] = np.reshape([(x,y,z)
                                 for z in dz[lo_bound[2]:hi_bound[2]+1]
                                 for y in dy[lo_bound[1]:hi_bound[1]+1]
                                 for x in dx[lo_bound[0]:hi_bound[0]+1]],
                              (shape[0], shape[1], shape[2], 3))

      fout["pressure"][:] = ct.one_atm/mix.PRef
      fout["dTdtBoundary"][:] = np.zeros(shape=shape, dtype = np.dtype("f8"))
      fout["dudtBoundary"][:] = np.zeros(shape=shape, dtype = np.dtype("(3,)f8"))

      v = np.interp(yGrid[lo_bound[1]:hi_bound[1]+1], gridIn, flame.velocity/mix.uRef)
      velocity = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
      for j in range(hi_bound[1]-lo_bound[1]+1): velocity[:,j,:] = [ 0.0, v[j], 0.0]
      fout["velocity"][:] = velocity[:]
      fout["velocity_profile"][:] = velocity[:]
      del velocity

      temperature = np.ndarray(shape)
      T = np.interp(yGrid[lo_bound[1]:hi_bound[1]+1], gridIn, flame.T/mix.TRef)
      for j in range(hi_bound[1]-lo_bound[1]+1): temperature[:,j,:] = T[j]
      fout["temperature"][:] = temperature
      fout["temperature_profile"][:] = temperature
      del temperature

      rho = np.ndarray(shape)
      r = np.interp(yGrid[lo_bound[1]:hi_bound[1]+1], gridIn, flame.density/mix.rhoRef)
      for j in range(hi_bound[1]-lo_bound[1]+1): rho[:,j,:] = r[j]
      fout["rho"][:] = rho
      del rho

      MolarFracs = np.ndarray(shape, dtype=np.dtype('(43,)f8'))
      for i, x in enumerate(Xi):
         X = np.interp(yGrid[lo_bound[1]:hi_bound[1]+1], gridIn, flame.X[i,:])
         for j in range(hi_bound[1]-lo_bound[1]+1): MolarFracs[:,j,:,i] = X[j]
      fout["MolarFracs"][:] = MolarFracs
      fout["MolarFracs_profile"][:] = MolarFracs
      del MolarFracs

      Phi = np.ndarray(shape)
      p = np.interp(yGrid[lo_bound[1]:hi_bound[1]+1], gridIn, flame.electric_potential/mix.delPhi)
      for j in range(hi_bound[1]-lo_bound[1]+1): Phi[:,j,:] = p[j]
      fout["electricPotential"][:] = Phi
      del Phi

for x, y, z in np.ndindex((Ntiles[0], Ntiles[1], Ntiles[2])): writeTile(x, y, z)

