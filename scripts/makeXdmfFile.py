#!/usr/bin/env python3

import re
import os
import sys
import h5py
import glob
import shutil
import argparse
import subprocess
import numpy as np
from joblib import Parallel, delayed, dump, load

############################################################
# XMF MACROS
############################################################

XMF_HEADER = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="3.0">
   <Domain Name="Fluid">
      <Grid Name="FluidTimeSeries" GridType="Collection" CollectionType="Temporal">
"""

XMF_SOLUTION_HEADER = """
         <Grid Name="@NAME" GridType="Collection" CollectionType="Spatial">
            <Time Value="@TIMESTEP"/>
"""

XMF_TILE_HEADER = """
            <Grid Name="Tile @N" GridType="Uniform">
                  <Topology TopologyType="3DRectMesh" Dimensions="@XPOINTS @YPOINTS @ZPOINTS"></Topology>
                  <Geometry GeometryType="VXVYVZ">
                     <DataItem Dimensions="@ZPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/ZFaceCoordinates</DataItem>
                     <DataItem Dimensions="@YPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/YFaceCoordinates</DataItem>
                     <DataItem Dimensions="@XPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/XFaceCoordinates</DataItem>
                  </Geometry>
"""

XMF_SOLUTION_SCALAR= """
                  <Attribute Name="@NAME" AttributeType="Scalar" Center="Cell">
                     <DataItem Dimensions="@CELLS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@NAME</DataItem>
                  </Attribute>
"""

XMF_SOLUTION_SCALAR2= """
                  <Attribute Name="@NAME" AttributeType="Scalar" Center="Cell">
                     <DataItem Dimensions="@CELLS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/@NAME</DataItem>
                  </Attribute>
"""

XMF_SOLUTION_VECTOR= """
                  <Attribute Name="@NAME" AttributeType="Vector" Center="Cell">
                     <DataItem Dimensions="@CELLS 3" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/@NAME</DataItem>
                  </Attribute>
"""

XMF_TILE_FOOTER = """
            </Grid>
"""

XMF_SOLUTION_FOOTER = """
         </Grid>
"""

XMF_FOOTER = """
      </Grid>
   </Domain>
</Xdmf>
"""

############################################################
# UTILS
############################################################

def progressbar(it, prefix = "", suffix = "", decimals = 1, length = 50, fill = "-", file=sys.stdout):
   count = len(it)
   def show(j):
      percent = ("{0:." + str(decimals) + "f}").format(100 * (j / float(count)))
      filledLength = int(length * j // count)
      bar = fill * filledLength + ">" + " " * (length - filledLength)
      file.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
      file.flush()
   show(0)
   for i, item in enumerate(it):
      yield i, item
      show(i+1)
   file.write("\n")
   file.flush()

def getTileBoundsFromName(t):
   pat = r'([0-9]+),([0-9]+),([0-9]+)-([0-9]+),([0-9]+),([0-9]+).hdf'
   m = re.match(pat, t)
   assert(m is not None)
   lo = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
   hi = [int(m.group(4)), int(m.group(5)), int(m.group(6))]
   return lo, hi

def getTileGrid(CartesianGrid, nodes_grid_dir, path, lo, hi):
   if CartesianGrid:
      cwd = os.getcwd()
      os.chdir(os.path.join(nodes_grid_dir, path))
      with h5py.File("master.hdf", 'r') as fin:
         hi_grid = fin["position"][:].size
         Bnum = fin.attrs.get("Bnum")
         nlo = max(lo - Bnum, 0)
         nhi = max(hi - Bnum, 0)
         nodes = fin["position"][:][nlo:nhi+2]
         if Bnum > 0:
            if (lo == 0):
               # Add ghost point on Neg side
               if fin.attrs.get("NegStaggered"):
                  # Add node for staggered bc
                  nodes = np.insert(nodes, 0, nodes[0]-1e-12)
               else:
                  # Add node for collocated bc
                  nodes = np.insert(nodes, 0, 2*nodes[0]-nodes[1])
            if (hi == hi_grid):
               if fin.attrs.get("PosStaggered"):
                  # Add node for staggered bc
                  nodes = np.append(nodes, nodes[-1]+1e-12)
               else:
                  # Add node for collocated bc
                  nodes = np.append(nodes, 2*nodes[-1]-nodes[-2])
      os.chdir(cwd)
   return nodes

############################################################
# MAIN SCRIPT
############################################################

parser = argparse.ArgumentParser()
parser.add_argument('--sampledir', nargs='?', const='.', default='.',
                    help='directory with all the simulation output')
parser.add_argument('--debugOut', default=False, action='store_true',
                    help='flag to process debug output of the simulation')
parser.add_argument('--np', nargs='?', default=1, type=int,
                    help='number of cores')
args = parser.parse_args()
sample_dir = os.path.abspath(args.sampledir)

# Make directories for viz data
out_dir    = os.path.abspath("viz_data")
grid_file  = os.path.join(out_dir, "grid.hdf")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Check for solution snapshots
if (args.debugOut):
   snapshots = glob.glob(os.path.join(sample_dir,"debugOut"))
else:
   snapshots = glob.glob(os.path.join(sample_dir,"fluid_iter*"))
if (len(snapshots) == 0): assert False, "No solution files provided"

# sort files by timestep
snapshots.sort(key=lambda x: x[len(os.path.join(sample_dir, "fluid_iter")):])

# We still need to convert H5T_ARRAYs to numpy vectors...
# TODO: get rid of this step as soon as you find out how to read H5T_ARRAYs with XDMF
print("Converting snapshots:")
def ConvertSnapshot(i, sn):
   hdf_sol = None
   solutionfile = os.path.join(out_dir,sn.split('/')[-1])+".hdf"
   if not os.path.exists(solutionfile):
      hdf_sol = h5py.File(solutionfile, "w")
      tiles = glob.glob(os.path.join(sn,"*hdf"))
      masterFile = os.path.join(sn,"master.hdf")
      if os.path.exists(masterFile):
         hdf_master   = h5py.File(masterFile, 'r')
         SpeciesNames = hdf_master.attrs.get("SpeciesNames")
         hdf_master.close()
         tiles.remove(masterFile)
      for j, tl in enumerate(tiles):
         hdf_in = h5py.File(tl, 'r')
         if not os.path.exists(masterFile):
            SpeciesNames = hdf_in.attrs.get("SpeciesNames")
         group = tl.split('/')[-1][0:-4]
         hdf_sol[group+"/velocity"] = hdf_in["velocity"][:][:,:,:,:]
         for isp, sp in enumerate(SpeciesNames):
            hdf_sol[group+"/X_"+sp.decode()] = hdf_in['MolarFracs'][:][:,:,:,isp]
         if (args.debugOut):
            for isp, sp in enumerate(SpeciesNames):
               hdf_sol[group+"/rhoY_"+sp.decode()] = hdf_in['Conserved'][:][:,:,:,isp]
            hdf_sol[group+"/rhoU"] = hdf_in['Conserved'][:][:,:,:,len(SpeciesNames)]
            hdf_sol[group+"/rhoV"] = hdf_in['Conserved'][:][:,:,:,len(SpeciesNames)+1]
            hdf_sol[group+"/rhoW"] = hdf_in['Conserved'][:][:,:,:,len(SpeciesNames)+2]
            hdf_sol[group+"/rhoE"] = hdf_in['Conserved'][:][:,:,:,len(SpeciesNames)+3]
         hdf_in.close()
      hdf_sol.close()

Parallel(n_jobs=args.np, verbose=10)(delayed(ConvertSnapshot)(i, sn) for i, sn in enumerate(snapshots))
####################################################################################

# Grid directory
nodes_grid_dir = os.path.join(sample_dir, "nodes_grid")
# Check if this is Cartesian grid
CartesianGrid = os.path.exists(os.path.join(nodes_grid_dir, "xNodes"))
if CartesianGrid:
   assert os.path.exists(os.path.join(nodes_grid_dir, "yNodes")), "Could not find the yNodes folder"
   assert os.path.exists(os.path.join(nodes_grid_dir, "zNodes")), "Could not find the zNodes folder"
else:
   assert False, "Unsupported grid"

with open('out_fluid.xmf', 'w') as xmf_out:
   xmf_out.write(XMF_HEADER)

   for i, sn in progressbar(snapshots, "Writing Xdmf file:"):

      # We still need to convert H5T_ARRAYs to numpy vectors...
      # TODO: get rid of this step as soon as you find out how to read H5T_ARRAYs with XDMF
      solutionfile = os.path.join(out_dir,sn.split('/')[-1])+".hdf"
      ####################################################################################

      tiles = glob.glob(os.path.join(sn,"*hdf"))

      # Load attributes
      masterFile = os.path.join(sn,"master.hdf")
      if os.path.exists(masterFile):
         tiles.remove(masterFile)
         hdf_in = h5py.File(os.path.join(sn,"master.hdf"), 'r')
         simTime = hdf_in.attrs.get("simTime")
         SpeciesNames = hdf_in.attrs.get("SpeciesNames")
         hdf_in.close()
      else:
         hdf_in = h5py.File(tiles[0], 'r')
         simTime = hdf_in.attrs.get("simTime")
         SpeciesNames = hdf_in.attrs.get("SpeciesNames")
         hdf_in.close()

      xmf_out.write(XMF_SOLUTION_HEADER
                    .replace('@NAME','%s'% sn.split('/')[-1])
                    .replace('@TIMESTEP','%.8g'% simTime))

      # Set reference to each tile
      for j, tl in enumerate(tiles):
         hdf_in = h5py.File(tl, 'r')
         group = tl.split('/')[-1][0:-4]

         if not os.path.exists(masterFile):
            # Sanity check
            assert simTime == hdf_in.attrs.get("simTime")
            assert (SpeciesNames == hdf_in.attrs.get("SpeciesNames")).all()

         # Extract domain size.
         nx = hdf_in['pressure'].shape[0]
         ny = hdf_in['pressure'].shape[1]
         nz = hdf_in['pressure'].shape[2]

         # check if we need to store the grid
         with h5py.File(grid_file, "a") as hdf_grid:
            if group not in hdf_grid:
               lo, hi = getTileBoundsFromName(tl.split('/')[-1])
               hdf_grid[group+"/ZFaceCoordinates"] = getTileGrid(CartesianGrid, nodes_grid_dir, "xNodes", lo[0], hi[0])
               hdf_grid[group+"/YFaceCoordinates"] = getTileGrid(CartesianGrid, nodes_grid_dir, "yNodes", lo[1], hi[1])
               hdf_grid[group+"/XFaceCoordinates"] = getTileGrid(CartesianGrid, nodes_grid_dir, "zNodes", lo[2], hi[2])

         # Print reference to data
         xmf_out.write(XMF_TILE_HEADER
                       .replace("@N", "%s" % tl.split('/')[-1][0:-4])
                       .replace("@XPOINTS", "%s" % str(nx+1))
                       .replace("@YPOINTS", "%s" % str(ny+1))
                       .replace("@ZPOINTS", "%s" % str(nz+1))
                       .replace("@HDF_FILE", grid_file)
                       .replace("@GROUP", "%s" % group))

         xmf_out.write(XMF_SOLUTION_SCALAR
                       .replace("@NAME", "%s" % "pressure")
                       .replace("@HDF_FILE", "%s" % tl)
                       .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

         xmf_out.write(XMF_SOLUTION_SCALAR
                       .replace("@NAME","%s" % "rho")
                       .replace("@HDF_FILE", "%s" % tl)
                       .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

         xmf_out.write(XMF_SOLUTION_SCALAR
                       .replace("@NAME","%s" % "temperature")
                       .replace("@HDF_FILE", "%s" % tl)
                       .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

         if "electricPotential" in hdf_in:
            xmf_out.write(XMF_SOLUTION_SCALAR
                          .replace("@NAME","%s" % "electricPotential")
                          .replace("@HDF_FILE", "%s" % tl)
                          .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

         for sp in SpeciesNames:
            xmf_out.write(XMF_SOLUTION_SCALAR2
                          .replace("@NAME","%s" % "X_"+sp.decode())
                          .replace("@GROUP", "%s" % group)
                          .replace("@HDF_FILE", "%s" % solutionfile)
                          #.replace("@HDF_FILE", "%s" % tl)
                          .replace('@CELLS', '%s %s %s' % (nx,ny,nz)))

         xmf_out.write(XMF_SOLUTION_VECTOR
                       .replace("@NAME", "%s" % "velocity")
                       .replace("@GROUP", "%s" % group)
                       .replace("@HDF_FILE", "%s" % solutionfile)
                       #.replace("@HDF_FILE", "%s" % tl)
                       .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

         if (args.debugOut):
            xmf_out.write(XMF_SOLUTION_SCALAR
                          .replace("@NAME","%s" % "shockSensorX")
                          .replace("@HDF_FILE", "%s" % tl)
                          .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

            xmf_out.write(XMF_SOLUTION_SCALAR
                          .replace("@NAME","%s" % "shockSensorY")
                          .replace("@HDF_FILE", "%s" % tl)
                          .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))

            xmf_out.write(XMF_SOLUTION_SCALAR
                          .replace("@NAME","%s" % "shockSensorZ")
                          .replace("@HDF_FILE", "%s" % tl)
                          .replace("@CELLS", "%s %s %s" % (nx,ny,nz)))
            for sp in SpeciesNames:
               xmf_out.write(XMF_SOLUTION_SCALAR2
                             .replace("@NAME","%s" % "rhoY_"+sp.decode())
                             .replace("@GROUP", "%s" % group)
                             .replace("@HDF_FILE", "%s" % solutionfile)
                             #.replace("@HDF_FILE", "%s" % tl)
                             .replace('@CELLS', '%s %s %s' % (nx,ny,nz)))
            xmf_out.write(XMF_SOLUTION_SCALAR2
                          .replace("@NAME","%s" % "rhoU")
                          .replace("@GROUP", "%s" % group)
                          .replace("@HDF_FILE", "%s" % solutionfile)
                          #.replace("@HDF_FILE", "%s" % tl)
                          .replace('@CELLS', '%s %s %s' % (nx,ny,nz)))
            xmf_out.write(XMF_SOLUTION_SCALAR2
                          .replace("@NAME","%s" % "rhoV")
                          .replace("@GROUP", "%s" % group)
                          .replace("@HDF_FILE", "%s" % solutionfile)
                          #.replace("@HDF_FILE", "%s" % tl)
                          .replace('@CELLS', '%s %s %s' % (nx,ny,nz)))
            xmf_out.write(XMF_SOLUTION_SCALAR2
                          .replace("@NAME","%s" % "rhoW")
                          .replace("@GROUP", "%s" % group)
                          .replace("@HDF_FILE", "%s" % solutionfile)
                          #.replace("@HDF_FILE", "%s" % tl)
                          .replace('@CELLS', '%s %s %s' % (nx,ny,nz)))
            xmf_out.write(XMF_SOLUTION_SCALAR2
                          .replace("@NAME","%s" % "rhoE")
                          .replace("@GROUP", "%s" % group)
                          .replace("@HDF_FILE", "%s" % solutionfile)
                          #.replace("@HDF_FILE", "%s" % tl)
                          .replace('@CELLS', '%s %s %s' % (nx,ny,nz)))

         xmf_out.write(XMF_TILE_FOOTER)
         hdf_in.close()

      xmf_out.write(XMF_SOLUTION_FOOTER)

   xmf_out.write(XMF_FOOTER)
