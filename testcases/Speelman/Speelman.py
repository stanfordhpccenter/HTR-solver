import cantera as ct
import numpy as np

gas = ct.Solution("Lu.cti")
gas.set_equivalence_ratio(1.0, 'CH4', {'O2':1.0, 'N2':3.76})
gas.TP = 300, ct.one_atm

species_names = gas.species_names

gas()

flame = ct.BurnerFlame(gas, width=1.0e-2)
flame.burner.T = 350
flame.burner.mdot = 0.40472
flame.burner.Y = gas.Y
flame.set_refine_criteria(ratio=2.0, slope=0.1, curve=0.15, prune=0.03)
flame.solve()

f = open("Ref.dat", "w")
f.write((11*"%15s"+"\n")%("y", "T", "rho", "X_CH4", "X_O2", "X_N2", "X_CH", "X_CO", "X_CO2", "X_H2O", "X_H2"))

for i, y in enumerate(flame.grid):
   f.write((11*"%15.7e"+"\n")%(y, flame.T[i], flame.density[i],
                         flame.X[species_names.index("CH4")][i],
                         flame.X[species_names.index( "O2")][i],
                         flame.X[species_names.index( "N2")][i],
                         flame.X[species_names.index( "CH")][i],
                         flame.X[species_names.index( "CO")][i],
                         flame.X[species_names.index("CO2")][i],
                         flame.X[species_names.index("H2O")][i],
                         flame.X[species_names.index( "H2")][i]))
f.close()

