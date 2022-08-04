
# This script was run with Cantera-2.5.0, which must be installed separately
import cantera as ct
import numpy as np

# Parameters
mechanism = 'FFCM1Mix'
verb = 0 # Cantera flame solver verbosity

# Derived
ctiFile = 'cti/{}.cti'.format(mechanism)
outDir = 'ref'

# Set up gas object -- assumes CH4/O2 mechanism
gas = ct.Solution(ctiFile)
species_names = gas.species_names
iCH4 = gas.species_index('CH4')
iO2 = gas.species_index('O2')
nSpec = gas.n_species
nRxn = gas.n_reactions

# Constants
W_CH4 = 16e-3 # kg/mol
W_O2 = 32e-3 # kg/mol
stoich = W_CH4 / (2.0*W_O2) # fuel-oxidizer stoichiometric (mass-based)

# Ambient conditions
phi = np.array([0.5,1.0,1.5,2.0])
nPhi = np.size(phi)
p0 = 101325.0
T0 = 298.0
sL = np.zeros([nPhi])
Tf = np.zeros([nPhi])
L = 5e-3 # m

for i in range(nPhi):

   # Initial conditions
   gas.set_equivalence_ratio(phi[i],'CH4','O2')
   gas.TP = [T0,p0]

   # Set up flame object
   fl = ct.FreeFlame(gas,width=L)
   fl.set_refine_criteria(slope=0.05,curve=0.05)
   fl.transport_model = 'Mix'

   #fl = ct.BurnerFlame(gas,width=L)
   #fl.set_refine_criteria(slope=0.1,curve=0.1)
   #fl.transport_model = 'Mix'
   #fl.burner.T = T0
   #fl.burner.Y = gas.Y
   #fl.burner.mdot = gas.density * sL[i] / 2.0
   #print('mdot={}'.format(fl.burner.mdot))

   # Solve
   fl.solve(loglevel=verb,refine_grid=True)
   sL[i] = fl.velocity[0] # Check units in cti file
   Tf[i] = fl.T[-1]
   Nx = np.size(fl.grid)

   # Collect data
   lab = []
   data = []
   lab.append('{:14s}'.format('x'));        data.append(fl.grid)
   lab.append('{:14s}'.format('u'));        data.append(fl.velocity)
   lab.append('{:14s}'.format('rho'));      data.append(fl.density)
   lab.append('{:14s}'.format('p'));        data.append(fl.domains[1].P*np.ones(Nx))
   lab.append('{:14s}'.format('T'));        data.append(fl.T)
   lab.append('{:14s}'.format('X_CH4'));    data.append(fl.X[species_names.index("CH4")])
   lab.append('{:14s}'.format('X_O2'));     data.append(fl.X[species_names.index("O2")])
   lab.append('{:14s}'.format('X_H2O'));    data.append(fl.X[species_names.index("H2O")])
   lab.append('{:14s}'.format('X_CO2'));    data.append(fl.X[species_names.index("CO2")])
   lab.append('{:14s}'.format('X_H'));      data.append(fl.X[species_names.index("H")])
   lab.append('{:14s}'.format('X_O'));      data.append(fl.X[species_names.index("O")])
   lab.append('{:14s}'.format('X_OH'));     data.append(fl.X[species_names.index("OH")])
   lab.append('{:14s}'.format('X_CO'));     data.append(fl.X[species_names.index("CO")])
   lab = ' '.join(lab)
   data = np.transpose(np.stack(data,axis=0))

   # Write data
   fname = '{}/{}-flame-{}.dat'.format(outDir,mechanism,i+1)
   np.savetxt(fname,data,header=lab,delimiter=' ',fmt='%14.6e')
   print('File written: {}'.format(fname))
   print('i={}, Nx={}, phi={:6.3f}, sL={:14.12f} m/s, Tf={:14.10f} K'.format(i,Nx,phi[i],sL[i],Tf[i]))
