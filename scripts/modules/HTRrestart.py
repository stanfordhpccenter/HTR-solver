#!/usr/bin/env python3

import os
import re
import h5py
import glob
import numpy as np
from joblib import Parallel, delayed

##############################################################################
# Data types                                                                 #
##############################################################################

# Bounding box
bBoxType = np.dtype([('v0', '<f8', (3,)),
                     ('v1', '<f8', (3,)),
                     ('v2', '<f8', (3,)),
                     ('v3', '<f8', (3,)),
                     ('v4', '<f8', (3,)),
                     ('v5', '<f8', (3,)),
                     ('v6', '<f8', (3,)),
                     ('v7', '<f8', (3,))])

##############################################################################
# 1D node grid IO utility                                                    #
##############################################################################
class HTROneDgrid:
   def __init__(self, folder, x, bBox):

      # Create the directory if it does not exist
      if not os.path.exists(folder):
         os.makedirs(folder)

      self.folder = folder
      self.x = x
      self.shape = (len(x), )
      self.bBox = bBox


   def filename(self, lo, hi):
      return ('%s-%s.hdf' % (lo, hi))

   def makeFilename(self):
      return self.filename(0, len(self.x)-1)

   def dump(self):

      # Write the master file
      with h5py.File(os.path.join(self.folder, "master.hdf"), 'w') as fout:

         # Bounding box
         fout.attrs.create("boundingBox", self.bBox, dtype=bBoxType)

         # Data types
         int1dType  = np.dtype('<i8')
         rect1dType = np.dtype([
                        ('lo', '<i8'),
                        ('hi', '<i8')])

         # IO version
         fout.attrs.create("IO_VERSION", "1.0.0")

         # Index space
         indexSpace = np.zeros(1, dtype=rect1dType)
         indexSpace[0][0] = 0
         indexSpace[0][1] = len(self.x)-1
         fout.attrs.create("baseIndexSpace", indexSpace, dtype=rect1dType)

         # Store colors
         tiles = np.zeros(1, dtype=int1dType)
         tiles[0] = 1
         fout.attrs.create("tilesNumber", tiles, dtype=int1dType)

         # Store partitions
         tiles = np.ndarray(1, dtype=rect1dType)
         tiles[0][0] = 0
         tiles[0][1] = len(self.x)-1
         fout.create_dataset("partitions", shape=(1,), dtype = rect1dType)
         fout["partitions"][:] = tiles


         # TODO: Create virual layout
         def makeVirtualLayout(name, dtype):
            layout = h5py.VirtualLayout(shape=self.shape, dtype=dtype)
            vsource = h5py.VirtualSource(self.makeFilename(), name, shape=self.shape)
            layout[0:len(self.x)] = vsource
            fout.create_virtual_dataset(name, layout)
         makeVirtualLayout("position", "f8")

      # Writes each tile
      with h5py.File(os.path.join(self.folder, self.makeFilename()), 'w') as fout:
         fout.create_dataset("position", shape=self.shape, dtype = np.dtype("f8"))
         fout["position"][:] = self.x

##############################################################################
# HTR Restart utility                                                        #
##############################################################################
class HTRrestart:

   def filename(self, lo_bound, hi_bound):
      return ('%s,%s,%s-%s,%s,%s.hdf'
               % (lo_bound[0], lo_bound[1], lo_bound[2],
                  hi_bound[0], hi_bound[1], hi_bound[2]))

   def makeFilename(self, x, y, z):
      return self.filename(self.lo_bound[x,y,z], self.hi_bound[x,y,z])

   def __init__(self, config=None):
      if config == None:
         # This object is only able to read a restart
         self.reader = True

      else:
         # This object is able to write a restart folder
         self.reader = False

         # Load mapping
         assert config["Mapping"]["tiles"][0] % config["Mapping"]["tilesPerRank"][0] == 0
         assert config["Mapping"]["tiles"][1] % config["Mapping"]["tilesPerRank"][1] == 0
         assert config["Mapping"]["tiles"][2] % config["Mapping"]["tilesPerRank"][2] == 0
         self.Ntiles = np.array(config["Mapping"]["tiles"][:])
         self.Ntiles[0] = int(self.Ntiles[0]/config["Mapping"]["tilesPerRank"][0])
         self.Ntiles[1] = int(self.Ntiles[1]/config["Mapping"]["tilesPerRank"][1])
         self.Ntiles[2] = int(self.Ntiles[2]/config["Mapping"]["tilesPerRank"][2])

         assert config["Grid"]["xNum"] % self.Ntiles[0] == 0
         assert config["Grid"]["yNum"] % self.Ntiles[1] == 0
         assert config["Grid"]["zNum"] % self.Ntiles[2] == 0

         NxTile = int(config["Grid"]["xNum"]/self.Ntiles[0])
         NyTile = int(config["Grid"]["yNum"]/self.Ntiles[1])
         NzTile = int(config["Grid"]["zNum"]/self.Ntiles[2])

         # Determine number of halo points
         halo = [0, 0, 0]

         if config["BC"]["xBCLeft"]["type"] == "Periodic":
            assert config["BC"]["xBCLeft"]["type"] == config["BC"]["xBCRight"]["type"]
         else:
            halo[0] = 1

         if config["BC"]["yBCLeft"]["type"] == "Periodic":
            assert config["BC"]["yBCLeft"]["type"] == config["BC"]["yBCRight"]["type"]
         else:
            halo[1] = 1

         if config["BC"]["zBCLeft"]["type"] == "Periodic":
            assert config["BC"]["zBCLeft"]["type"] == config["BC"]["zBCRight"]["type"]
         else:
            halo[2] = 1

         # Save tiles parameters
         self.tilesSpace = (self.Ntiles[0], self.Ntiles[1], self.Ntiles[2])
         self.lo_bound = np.ndarray([self.Ntiles[0], self.Ntiles[1], self.Ntiles[2], 3], dtype=np.intc)
         self.hi_bound = np.ndarray([self.Ntiles[0], self.Ntiles[1], self.Ntiles[2], 3], dtype=np.intc)
         for x, y, z in np.ndindex(self.tilesSpace):
            self.lo_bound[x,y,z,:] = [(x  )*NxTile  +halo[0], (y  )*NyTile  +halo[1], (z  )*NzTile  +halo[2]]
            self.hi_bound[x,y,z,:] = [(x+1)*NxTile-1+halo[0], (y+1)*NyTile-1+halo[1], (z+1)*NzTile-1+halo[2]]
            if (x == 0): self.lo_bound[x,y,z,0] -= halo[0]
            if (y == 0): self.lo_bound[x,y,z,1] -= halo[1]
            if (z == 0): self.lo_bound[x,y,z,2] -= halo[2]
            if (x == self.Ntiles[0]-1): self.hi_bound[x,y,z,0] += halo[0]
            if (y == self.Ntiles[1]-1): self.hi_bound[x,y,z,1] += halo[1]
            if (z == self.Ntiles[2]-1): self.hi_bound[x,y,z,2] += halo[2]

      # Flag if the object is attached to a restart
      self.attached = False

   # This method writes a new restart folder for HTR
   # Parameters:
   # - restartDir: directory
   # - nSpec: number of species
   # - P : function that provides the pressure field        (i,j,k) -> double
   # - T : function that provides the temperature field     (i,j,k) -> double
   # - Xi: function that provides the molar fractions field (i,j,k) -> double[nSpec]
   # - U : function that provides the velocity field        (i,j,k) -> double[3]
   # - dudt: function that provides du/dt at the boundaries (i,j,k) -> double[3]     (optional)
   # - dTdt: function that provides dT/dt at the boundaries (i,j,k) -> double        (optional)
   # - Phi : function that provides electric potential field(i,j,k) -> double        (optional)
   # - T_p : function that provides the temperature prof.   (i,j,k) -> double        (optional)
   # - Xi_p: function that provides the molar frac. prof.   (i,j,k) -> double[nSpec] (optional)
   # - U_p : function that provides the velocity prof.      (i,j,k) -> double[3]     (optional)
   # - timeStep         : time step number (optional)
   # - simTime          : simulation time (optional)
   # - channelForcing   : channelForcing value (optional)
   # - nproc            : number of processors (optional)
   def write(self, restartDir, nSpec,
             P, T, Xi, U,
             dudt = None, dTdt = None, Phi = None,
             T_p = None, Xi_p = None, U_p = None,
             timeStep = 0, simTime = 0.0, channelForcing = 0.0,
             nproc = 1):
      # Check that the object has been initialised properly
      assert (not self.reader), "HTRrestart object needs to be initialized with `config` in order to use the write function"

      # Create the directory if it does not exist
      if not os.path.exists(restartDir):
         os.makedirs(restartDir)

      # Write the master file
      with h5py.File(os.path.join(restartDir, "master.hdf"), 'w') as fout:
         fout.attrs.create("timeStep", int(timeStep))
         fout.attrs.create("simTime" , float(simTime))
         fout.attrs.create("channelForcing", float(channelForcing))

         # Data types
         int3dType  = np.dtype([('x', '<i8'), ('y', '<i8'), ('z', '<i8')])
         rect3dType = np.dtype([
                        ('lo', [('x', '<i8'), ('y', '<i8'), ('z', '<i8')]),
                        ('hi', [('x', '<i8'), ('y', '<i8'), ('z', '<i8')])])

         # IO version
         fout.attrs.create("IO_VERSION", "1.0.0")

         # Index space
         indexSpace = np.zeros(1, dtype=rect3dType)
         indexSpace[0][0][0] = self.lo_bound[0,0,0,0]
         indexSpace[0][0][1] = self.lo_bound[0,0,0,1]
         indexSpace[0][0][2] = self.lo_bound[0,0,0,2]
         indexSpace[0][1][0] = self.hi_bound[-1,-1,-1,0]
         indexSpace[0][1][1] = self.hi_bound[-1,-1,-1,1]
         indexSpace[0][1][2] = self.hi_bound[-1,-1,-1,2]
         fout.attrs.create("baseIndexSpace", indexSpace, dtype=rect3dType)

         # Store colors
         tiles = np.zeros(1, dtype=int3dType)
         tiles[0] = self.tilesSpace
         fout.attrs.create("tilesNumber", tiles, dtype=int3dType)

         # Store partitions
         tiles = np.ndarray(self.tilesSpace[::-1], dtype=rect3dType)
         for x, y, z in np.ndindex(self.tilesSpace):
            tiles[z,y,x][0][0] = self.lo_bound[x,y,z,0]
            tiles[z,y,x][0][1] = self.lo_bound[x,y,z,1]
            tiles[z,y,x][0][2] = self.lo_bound[x,y,z,2]
            tiles[z,y,x][1][0] = self.hi_bound[x,y,z,0]
            tiles[z,y,x][1][1] = self.hi_bound[x,y,z,1]
            tiles[z,y,x][1][2] = self.hi_bound[x,y,z,2]
         fout.create_dataset("partitions", shape=self.tilesSpace[::-1], dtype = rect3dType)
         fout["partitions"][:] = tiles

         # Create virual layout
         shape = (self.hi_bound[-1,-1,-1,2] - self.lo_bound[0,0,0,2] + 1,
                  self.hi_bound[-1,-1,-1,1] - self.lo_bound[0,0,0,1] + 1,
                  self.hi_bound[-1,-1,-1,0] - self.lo_bound[0,0,0,0] + 1)
         def makeVirtualLayout(name, dtype):
            layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
            for x, y, z in np.ndindex(self.tilesSpace):
               lo_bound = self.lo_bound[x,y,z]
               hi_bound = self.hi_bound[x,y,z]
               my_shape = (hi_bound[2] - lo_bound[2] + 1,
                           hi_bound[1] - lo_bound[1] + 1,
                           hi_bound[0] - lo_bound[0] + 1)
               vsource = h5py.VirtualSource(self.makeFilename(x, y, z), name, shape=my_shape)
               layout[lo_bound[2]:hi_bound[2]+1,
                      lo_bound[1]:hi_bound[1]+1,
                      lo_bound[0]:hi_bound[0]+1] = vsource
            fout.create_virtual_dataset(name, layout)

         makeVirtualLayout("rho",               "f8"    )
         makeVirtualLayout("pressure",          "f8"    )
         makeVirtualLayout("temperature",       "f8"    )
         makeVirtualLayout("MolarFracs",        "("+str(nSpec)+",)f8")
         makeVirtualLayout("velocity",          "(3,)f8")

      # Function that writes each tile
      def writeTile(xt, yt, zt):
         # determine tile shape
         lo_bound = self.lo_bound[xt, yt, zt]
         hi_bound = self.hi_bound[xt, yt, zt]
         shape = (hi_bound[2] - lo_bound[2] + 1,
                  hi_bound[1] - lo_bound[1] + 1,
                  hi_bound[0] - lo_bound[0] + 1)

         with h5py.File(os.path.join(restartDir, self.makeFilename(xt, yt, zt)), 'w') as fout:
            fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
            fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
            fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
            fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
            fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
            fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("(3,)f8"))
            fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))

            if (os.path.expandvars("$ELECTRIC_FIELD") == "1"):
               fout.create_dataset("electricPotential",  shape=shape, dtype = np.dtype("f8"))

            if (Xi_p or U_p or T_p):
               fout.create_dataset("MolarFracs_profile", shape=shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
               fout.create_dataset("velocity_profile",   shape=shape, dtype = np.dtype("(3,)f8"))
               fout.create_dataset("temperature_profile",shape=shape, dtype = np.dtype("f8"))

            # Write pressure field
            fout["pressure"         ][:] = np.reshape([P(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2]))

            # Write temperature field
            fout["temperature"      ][:] = np.reshape([T(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2]))

            # Write MolarFracs field
            fout["MolarFracs"       ][:] = np.reshape([Xi(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2], nSpec))

            # Write velocity field
            fout["velocity"         ][:] = np.reshape([U(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2], 3))

            # Write dTdtBoundary field
            if dTdt:
               fout["dTdtBoundary"  ][:] = np.reshape([dTdt(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2]))
            else:
               fout["dTdtBoundary"  ][:] = np.zeros(shape=shape, dtype = np.dtype("f8"))

            # Write dTdtBoundary field
            if dudt:
               fout["dudtBoundary"  ][:] = np.reshape([dudt(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2], 3))
            else:
               fout["dudtBoundary"  ][:] = np.zeros(shape=shape, dtype = np.dtype("(3,)f8"))

            if ((os.path.expandvars("$ELECTRIC_FIELD") == "1") and Phi):
               # Write electricPotential field
               fout["electricPotential"  ][:] = np.reshape([Phi(i,j,k)
                                             for k in range(lo_bound[2], hi_bound[2]+1)
                                             for j in range(lo_bound[1], hi_bound[1]+1)
                                             for i in range(lo_bound[0], hi_bound[0]+1)],
                                           (shape[0], shape[1], shape[2]))


            if T_p:
               # Write temperature profile
               if T_p == T:
                  fout["temperature_profile"][:] = fout["temperature"][:]
               else:
                  fout["temperature_profile"][:] = np.reshape([T_p(i,j,k)
                                                for k in range(lo_bound[2], hi_bound[2]+1)
                                                for j in range(lo_bound[1], hi_bound[1]+1)
                                                for i in range(lo_bound[0], hi_bound[0]+1)],
                                              (shape[0], shape[1], shape[2]))

            if Xi_p:
               # Write MolarFracs profile
               if Xi_p == Xi:
                  fout["MolarFracs_profile" ][:] = fout["MolarFracs"][:]
               else:
                  fout["MolarFracs_profile" ][:] = np.reshape([Xi_p(i,j,k)
                                                for k in range(lo_bound[2], hi_bound[2]+1)
                                                for j in range(lo_bound[1], hi_bound[1]+1)
                                                for i in range(lo_bound[0], hi_bound[0]+1)],
                                              (shape[0], shape[1], shape[2], nSpec))

            if U_p:
               # Write velocity field
               if U_p == U:
                  fout["velocity_profile"   ][:] = fout["velocity"][:]
               else:
                  fout["velocity_profile"   ][:] = np.reshape([U_p(i,j,k)
                                                for k in range(lo_bound[2], hi_bound[2]+1)
                                                for j in range(lo_bound[1], hi_bound[1]+1)
                                                for i in range(lo_bound[0], hi_bound[0]+1)],
                                              (shape[0], shape[1], shape[2], 3))

      Parallel(n_jobs=nproc)(delayed(writeTile)(x, y, z) for x, y, z in np.ndindex(self.tilesSpace))

   # This method writes a new restart folder for HTR in a faster and less userfriendly way
   # Parameters:
   # - restartDir: directory
   # - nSpec: number of species
   # - P : function that provides the pressure field        (lo_bound, hi_bound, shape) -> double
   # - T : function that provides the temperature field     (lo_bound, hi_bound, shape) -> double
   # - Xi: function that provides the molar fractions field (lo_bound, hi_bound, shape) -> double[nSpec]
   # - U : function that provides the velocity field        (lo_bound, hi_bound, shape) -> double[3]
   # - dudt: function that provides du/dt at the boundaries (lo_bound, hi_bound, shape) -> double[3]     (optional)
   # - dTdt: function that provides dT/dt at the boundaries (lo_bound, hi_bound, shape) -> double        (optional)
   # - Phi : function that provides electric potential field(lo_bound, hi_bound, shape) -> double        (optional)
   # - T_p : function that provides the temperature prof.   (lo_bound, hi_bound, shape) -> double        (optional)
   # - Xi_p: function that provides the molar frac. prof.   (lo_bound, hi_bound, shape) -> double[nSpec] (optional)
   # - U_p : function that provides the velocity prof.      (lo_bound, hi_bound, shape) -> double[3]     (optional)
   # - timeStep         : time step number (optional)
   # - simTime          : simulation time (optional)
   # - channelForcing   : channelForcing value (optional)
   # - nproc            : number of processors (optional)
   def write_fast(self, restartDir, nSpec,
                  P, T, Xi, U,
                  dudt = None, dTdt = None, Phi = None,
                  T_p = None, Xi_p = None, U_p = None,
                  timeStep = 0, simTime = 0.0, channelForcing = 0.0,
                  nproc = 1):
      # Check that the object has been initialised properly
      assert (not self.reader), "HTRrestart object needs to be initialized with `config` in order to use the write_fast function"

      # Create the directory if it does not exist
      if not os.path.exists(restartDir):
         os.makedirs(restartDir)

      # Write the master file
      with h5py.File(os.path.join(restartDir, "master.hdf"), 'w') as fout:
         fout.attrs.create("timeStep", int(timeStep))
         fout.attrs.create("simTime" , float(simTime))
         fout.attrs.create("channelForcing", float(channelForcing))

         # Data types
         int3dType  = np.dtype([('x', '<i8'), ('y', '<i8'), ('z', '<i8')])
         rect3dType = np.dtype([
                        ('lo', [('x', '<i8'), ('y', '<i8'), ('z', '<i8')]),
                        ('hi', [('x', '<i8'), ('y', '<i8'), ('z', '<i8')])])

         # IO version
         fout.attrs.create("IO_VERSION", "1.0.0")

         # Index space
         indexSpace = np.zeros(1, dtype=rect3dType)
         indexSpace[0][0][0] = self.lo_bound[0,0,0,0]
         indexSpace[0][0][1] = self.lo_bound[0,0,0,1]
         indexSpace[0][0][2] = self.lo_bound[0,0,0,2]
         indexSpace[0][1][0] = self.hi_bound[-1,-1,-1,0]
         indexSpace[0][1][1] = self.hi_bound[-1,-1,-1,1]
         indexSpace[0][1][2] = self.hi_bound[-1,-1,-1,2]
         fout.attrs.create("baseIndexSpace", indexSpace, dtype=rect3dType)

         # Store colors
         tiles = np.zeros(1, dtype=int3dType)
         tiles[0] = self.tilesSpace
         fout.attrs.create("tilesNumber", tiles, dtype=int3dType)

         # Store partitions
         tiles = np.ndarray(self.tilesSpace[::-1], dtype=rect3dType)
         for x, y, z in np.ndindex(self.tilesSpace):
            tiles[z,y,x][0][0] = self.lo_bound[x,y,z,0]
            tiles[z,y,x][0][1] = self.lo_bound[x,y,z,1]
            tiles[z,y,x][0][2] = self.lo_bound[x,y,z,2]
            tiles[z,y,x][1][0] = self.hi_bound[x,y,z,0]
            tiles[z,y,x][1][1] = self.hi_bound[x,y,z,1]
            tiles[z,y,x][1][2] = self.hi_bound[x,y,z,2]
         fout.create_dataset("partitions", shape=self.tilesSpace[::-1], dtype = rect3dType)
         fout["partitions"][:] = tiles

         # Create virual layout
         shape = (self.hi_bound[-1,-1,-1,2] - self.lo_bound[0,0,0,2] + 1,
                  self.hi_bound[-1,-1,-1,1] - self.lo_bound[0,0,0,1] + 1,
                  self.hi_bound[-1,-1,-1,0] - self.lo_bound[0,0,0,0] + 1)
         def makeVirtualLayout(name, dtype):
            layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
            for x, y, z in np.ndindex(self.tilesSpace):
               lo_bound = self.lo_bound[x,y,z]
               hi_bound = self.hi_bound[x,y,z]
               my_shape = (hi_bound[2] - lo_bound[2] + 1,
                           hi_bound[1] - lo_bound[1] + 1,
                           hi_bound[0] - lo_bound[0] + 1)
               vsource = h5py.VirtualSource(self.makeFilename(x, y, z), name, shape=my_shape)
               layout[lo_bound[2]:hi_bound[2]+1,
                      lo_bound[1]:hi_bound[1]+1,
                      lo_bound[0]:hi_bound[0]+1] = vsource
            fout.create_virtual_dataset(name, layout)

         makeVirtualLayout("rho",               "f8"    )
         makeVirtualLayout("pressure",          "f8"    )
         makeVirtualLayout("temperature",       "f8"    )
         makeVirtualLayout("MolarFracs",        "("+str(nSpec)+",)f8")
         makeVirtualLayout("velocity",          "(3,)f8")

      # Function that writes each tile
      def writeTile(xt, yt, zt):
         # determine tile shape
         lo_bound = self.lo_bound[xt, yt, zt]
         hi_bound = self.hi_bound[xt, yt, zt]
         shape = (hi_bound[2] - lo_bound[2] + 1,
                  hi_bound[1] - lo_bound[1] + 1,
                  hi_bound[0] - lo_bound[0] + 1)

         with h5py.File(os.path.join(restartDir, self.makeFilename(xt, yt, zt)), 'w') as fout:
            fout.create_dataset("rho",                   shape=shape, dtype = np.dtype("f8"))
            fout.create_dataset("pressure",              shape=shape, dtype = np.dtype("f8"))
            fout.create_dataset("temperature",           shape=shape, dtype = np.dtype("f8"))
            fout.create_dataset("MolarFracs",            shape=shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
            fout.create_dataset("velocity",              shape=shape, dtype = np.dtype("(3,)f8"))
            fout.create_dataset("dudtBoundary",          shape=shape, dtype = np.dtype("(3,)f8"))
            fout.create_dataset("dTdtBoundary",          shape=shape, dtype = np.dtype("f8"))

            if (os.path.expandvars("$ELECTRIC_FIELD") == "1"):
               fout.create_dataset("electricPotential",  shape=shape, dtype = np.dtype("f8"))

            if (Xi_p or U_p or T_p):
               fout.create_dataset("MolarFracs_profile", shape=shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
               fout.create_dataset("velocity_profile",   shape=shape, dtype = np.dtype("(3,)f8"))
               fout.create_dataset("temperature_profile",shape=shape, dtype = np.dtype("f8"))

            # Write pressure field
            fout["pressure"         ][:] = P(lo_bound, hi_bound, shape)

            # Write temperature field
            fout["temperature"      ][:] = T(lo_bound, hi_bound, shape)

            # Write MolarFracs field
            fout["MolarFracs"       ][:] = Xi(lo_bound, hi_bound, shape)

            # Write velocity field
            fout["velocity"         ][:] = U(lo_bound, hi_bound, shape)

            # Write dTdtBoundary field
            if dTdt:
               fout["dTdtBoundary"  ][:] = dTdt(lo_bound, hi_bound, shape)
            else:
               fout["dTdtBoundary"  ][:] = np.zeros(shape=shape, dtype = np.dtype("f8"))

            # Write dTdtBoundary field
            if dudt:
               fout["dudtBoundary"  ][:] = dudt(lo_bound, hi_bound, shape)
            else:
               fout["dudtBoundary"  ][:] = np.zeros(shape=shape, dtype = np.dtype("(3,)f8"))

            if ((os.path.expandvars("$ELECTRIC_FIELD") == "1") and Phi):
               # Write electricPotential field
               fout["electricPotential"  ][:] = Phi(lo_bound, hi_bound, shape)

            if T_p:
               # Write temperature profile
               if T_p == T:
                  fout["temperature_profile"][:] = fout["temperature"][:]
               else:
                  fout["temperature_profile"][:] = T_p(lo_bound, hi_bound, shape)

            if Xi_p:
               # Write MolarFracs profile
               if Xi_p == Xi:
                  fout["MolarFracs_profile" ][:] = fout["MolarFracs"][:]
               else:
                  fout["MolarFracs_profile" ][:] = Xi_p(lo_bound, hi_bound, shape)

            if U_p:
               # Write velocity field
               if U_p == U:
                  fout["velocity_profile"   ][:] = fout["velocity"][:]
               else:
                  fout["velocity_profile"   ][:] = U_p(lo_bound, hi_bound, shape)

      Parallel(n_jobs=nproc)(delayed(writeTile)(x, y, z) for x, y, z in np.ndindex(self.tilesSpace))


   # This method writes a new profiles folder for HTR
   # Parameters:
   # - restartDir: directory
   # - nSpec: number of species
   # - T_p : function that provides the temperature prof.   (i,j,k) -> double
   # - Xi_p: function that provides the molar frac. prof.   (i,j,k) -> double[nSpec]
   # - U_p : function that provides the velocity prof.      (i,j,k) -> double[3]
   # - nproc : number of processors (optional)
   def write_profiles(self, restartDir, nSpec,
                      T_p, Xi_p, U_p,
                      nproc = 1):
      # Check that the object has been initialised properly
      assert (not self.reader), "HTRrestart object needs to be initialized with `config` in order to use the write_profiles function"

      # Create the directory if it does not exist
      if not os.path.exists(restartDir):
         os.makedirs(restartDir)

      # Write the master file
      with h5py.File(os.path.join(restartDir, "master.hdf"), 'w') as fout:
         # Data types
         int3dType  = np.dtype([('x', '<i8'), ('y', '<i8'), ('z', '<i8')])
         rect3dType = np.dtype([
                        ('lo', [('x', '<i8'), ('y', '<i8'), ('z', '<i8')]),
                        ('hi', [('x', '<i8'), ('y', '<i8'), ('z', '<i8')])])

         # IO version
         fout.attrs.create("IO_VERSION", "1.0.0")

         # Index space
         indexSpace = np.zeros(1, dtype=rect3dType)
         indexSpace[0][0][0] = self.lo_bound[0,0,0,0]
         indexSpace[0][0][1] = self.lo_bound[0,0,0,1]
         indexSpace[0][0][2] = self.lo_bound[0,0,0,2]
         indexSpace[0][1][0] = self.hi_bound[-1,-1,-1,0]
         indexSpace[0][1][1] = self.hi_bound[-1,-1,-1,1]
         indexSpace[0][1][2] = self.hi_bound[-1,-1,-1,2]
         fout.attrs.create("baseIndexSpace", indexSpace, dtype=rect3dType)

         # Store colors
         tiles = np.zeros(1, dtype=int3dType)
         tiles[0] = self.tilesSpace
         fout.attrs.create("tilesNumber", tiles, dtype=int3dType)

         # Store partitions
         tiles = np.ndarray(self.tilesSpace[::-1], dtype=rect3dType)
         for x, y, z in np.ndindex(self.tilesSpace):
            tiles[z,y,x][0][0] = self.lo_bound[x,y,z,0]
            tiles[z,y,x][0][1] = self.lo_bound[x,y,z,1]
            tiles[z,y,x][0][2] = self.lo_bound[x,y,z,2]
            tiles[z,y,x][1][0] = self.hi_bound[x,y,z,0]
            tiles[z,y,x][1][1] = self.hi_bound[x,y,z,1]
            tiles[z,y,x][1][2] = self.hi_bound[x,y,z,2]
         fout.create_dataset("partitions", shape=self.tilesSpace[::-1], dtype = rect3dType)
         fout["partitions"][:] = tiles

      # Function that writes each tile
      def writeTile(xt, yt, zt):
         # determine tile shape
         lo_bound = self.lo_bound[xt, yt, zt]
         hi_bound = self.hi_bound[xt, yt, zt]
         shape = (hi_bound[2] - lo_bound[2] + 1,
                  hi_bound[1] - lo_bound[1] + 1,
                  hi_bound[0] - lo_bound[0] + 1)

         with h5py.File(os.path.join(restartDir, self.makeFilename(xt, yt, zt)), 'w') as fout:
            fout.create_dataset("MolarFracs_profile", shape=shape, dtype = np.dtype("("+str(nSpec)+",)f8"))
            fout.create_dataset("velocity_profile",   shape=shape, dtype = np.dtype("(3,)f8"))
            fout.create_dataset("temperature_profile",shape=shape, dtype = np.dtype("f8"))

            # Write temperature profile
            fout["temperature_profile"][:] = np.reshape([T_p(i,j,k)
                                          for k in range(lo_bound[2], hi_bound[2]+1)
                                          for j in range(lo_bound[1], hi_bound[1]+1)
                                          for i in range(lo_bound[0], hi_bound[0]+1)],
                                        (shape[0], shape[1], shape[2]))

            # Write MolarFracs profile
            fout["MolarFracs_profile" ][:] = np.reshape([Xi_p(i,j,k)
                                          for k in range(lo_bound[2], hi_bound[2]+1)
                                          for j in range(lo_bound[1], hi_bound[1]+1)
                                          for i in range(lo_bound[0], hi_bound[0]+1)],
                                        (shape[0], shape[1], shape[2], nSpec))

            # Write velocity field
            fout["velocity_profile"   ][:] = np.reshape([U_p(i,j,k)
                                          for k in range(lo_bound[2], hi_bound[2]+1)
                                          for j in range(lo_bound[1], hi_bound[1]+1)
                                          for i in range(lo_bound[0], hi_bound[0]+1)],
                                        (shape[0], shape[1], shape[2], 3))

      Parallel(n_jobs=nproc)(delayed(writeTile)(x, y, z) for x, y, z in np.ndindex(self.tilesSpace))

   # This method attaches the class to an existing restart folder
   # Parameters:
   # - sampleDir: directory where the sample output has been written
   # - iteration: time step of the restart that we are going to read
   # - restartDir: directory of the restart file
   # - CCgridDir: directory of the cell center grid files
   def attach(self, sampleDir = None, step = None, restartDir = None, CCgridDir = None):

      # Create paths depending on the input
      if restartDir is not None:
         assert sampleDir == None, "Can't use sampleDir and restartDir at the same time"
         assert step == None, "Can't use step and restartDir at the same time"
         if CCgridDir is not None:
            assert os.path.exists(CCgridDir), "{} directory does not exist".format(CCgridDir)
      else:
         assert CCgridDir == None, "Can't use sampleDir and CCgridDir at the same time"
         assert step is not None, "sampleDir must be used in conjunction with step argument"
         CCgridDir  = os.path.join(sampleDir, "cellCenter_grid")
         restartDir = os.path.join(sampleDir, "fluid_iter"+str(step).zfill(10))

      # Sort out the structure of the restart
      # Check that the directory exists
      assert os.path.exists(restartDir), "{} directory does not exist".format(restartDir)

      # Check if we have a master file
      has_masterFile = os.path.exists(os.path.join(restartDir, "master.hdf"))

      # Load attributes
      if has_masterFile:
         with h5py.File(os.path.join(restartDir, "master.hdf"), 'r') as fin:
            self.timeStep       = fin.attrs["timeStep"]
            self.simTime        = fin.attrs["simTime" ]
            self.SpeciesNames   = fin.attrs["SpeciesNames"]
            if "channelForcing" in fin.attrs:
               self.channelForcing = fin.attrs["channelForcing"]

            # Check that all files exist
            for x, y, z in np.ndindex(tuple(fin.attrs["tilesNumber"])):
               filename = self.filename(fin["partitions"][:][z,y,x][0], fin["partitions"][:][z,y,x][1])
               filename = os.path.join(restartDir, filename)
               assert os.path.exists(filename), "File {} does not exist!".format(filename)

      else:
         # Check that the object has been initialised properly
         assert (not self.reader), "HTRrestart object needs to be initialized with `config` in order to ba attached to an old restart file"
         for x, y, z in np.ndindex(self.tilesSpace):
            filename = os.path.join(restartDir, self.makeFilename(x, y, z))
            assert os.path.exists(filename), "File {} does not exist!".format(filename)
            with h5py.File(filename, 'r') as fin:
               if self.timeStep:
                  assert (self.timeStep == fin.attrs["timeStep"]), "Attribute timeStep differs in file {}".format(filename)
               else:
                  self.timeStep = fin.attrs["timeStep"]
               if self.simTime:
                  assert (self.simTime == fin.attrs["simTime"]), "Attribute simTime differs in file {}".format(filename)
               else:
                  self.simTime = fin.attrs["simTime"]
               if self.channelForcing:
                  assert (self.channelForcing == fin.attrs["channelForcing"]), "Attribute channelForcing differs in file {}".format(filename)
               else:
                  self.channelForcing = fin.attrs["channelForcing"]
               if self.SpeciesNames:
                  assert all(self.SpeciesNames == fin.attrs["SpeciesNames"]), "Attribute SpeciesNames differs in file {}".format(filename)
               else:
                  self.SpeciesNames = fin.attrs["SpeciesNames"]

      # Sort out where we should read the grid
      separate_grid = False
      grid_has_masterFile = False
      if (CCgridDir is not None) and os.path.exists(CCgridDir):
         separate_grid = True
         grid_has_masterFile = os.path.exists(os.path.join(CCgridDir, "master.hdf"))

         # Check that all grid files exist
         if grid_has_masterFile:
            with h5py.File(os.path.join(CCgridDir, "master.hdf"), 'r') as fin:
               for x, y, z in np.ndindex(tuple(fin.attrs["tilesNumber"])):
                  filename = self.filename(fin["partitions"][:][z,y,x][0], fin["partitions"][:][z,y,x][1])
                  filename = os.path.join(CCgridDir, filename)
                  assert os.path.exists(filename), "File {} does not exist!".format(filename)
         else:
            for x, y, z in np.ndindex(self.tilesSpace):
               filename = os.path.join(CCgridDir, self.makeFilename(x, y, z))
               assert os.path.exists(filename), "File {} does not exist!".format(filename)

      # Store useful variables
      self.restartDir = restartDir
      self.has_masterFile = has_masterFile
      self.CCgridDir = CCgridDir
      self.separate_grid = separate_grid
      self.grid_has_masterFile = grid_has_masterFile
      self.attached = True

   # This method loads a field from the attached resart
   # Parameters:
   # - fld: field name
   def load(self, fld):

      # Check that we are attached to a restart file
      assert self.attached, "You need to attach the HTR restart object to a folder before calling the load method"

      def loadFromMasterfile(d):
         cwd = os.getcwd()
         os.chdir(d)
         with h5py.File("master.hdf", 'r') as fin:
            assert (fld in fin), "Field {} is not present in restart".format(fld)
            f = fin[fld][:]
         os.chdir(cwd)
         return f

      def loadOld(d):
         # Check that the field exists in all tile files
         dtype = None
         for x, y, z in np.ndindex(self.tilesSpace):
            filename = os.path.join(self.restartDir, self.makeFilename(x, y, z))
            with h5py.File(filename, 'r') as fin:
               assert (fld in fin), "Field {} is not present in file {}".format(fld, filename)
               # Store/check field data type
               if dtype:
                  assert (fin[fld].dtype == dtype), "Field {} data type differs in file {}".format(fld, filename)
               else:
                  dtype = fin[fld].dtype

         # Global shape of the array
         shape = (self.hi_bound[-1,-1,-1,2] - self.lo_bound[0,0,0,2] + 1,
                  self.hi_bound[-1,-1,-1,1] - self.lo_bound[0,0,0,1] + 1,
                  self.hi_bound[-1,-1,-1,0] - self.lo_bound[0,0,0,0] + 1)

         # Load the data
         f = np.ndarray(shape, dtype=dtype)
         for x, y, z in np.ndindex(self.tilesSpace):
            filename = os.path.join(self.restartDir, self.makeFilename(x, y, z))
            with h5py.File(filename, 'r') as fin:
               f[self.lo_bound[x,y,z,2]:self.hi_bound[x,y,z,2]+1,
                 self.lo_bound[x,y,z,1]:self.hi_bound[x,y,z,1]+1,
                 self.lo_bound[x,y,z,0]:self.hi_bound[x,y,z,0]+1] = fin[fld][:]

         return f

      # Grid quantities might go through a different path
      if ((fld in ["centerCoordinates"]) and self.separate_grid):
         assert (self.CCgridDir is not None)
         if self.grid_has_masterFile:
            f = loadFromMasterfile(self.CCgridDir)
         else:
            # keep this branch for backward compatibility
            f = loadOld(self.CCgridDir)
         return f

      if self.has_masterFile:
         f = loadFromMasterfile(self.restartDir)
      else:
         # keep this branch for backward compatibility
         f = loadOld(self.restartDir)
      return f

