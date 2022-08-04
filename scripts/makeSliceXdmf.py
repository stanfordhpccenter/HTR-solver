import os
import re
import shutil
import sys
import h5py
import subprocess
import argparse
import glob
import numpy as np

XMF_HEADER = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="3.0">
   <Domain Name="Fluid">
"""

XMF_SOLUTION_HEADER = """
      <Grid Name="@NAME" GridType="Collection" CollectionType="Spatial">
"""

XMF_TILE_HEADER = """
         <Grid Name="Tile @N" GridType="Uniform">
            <Topology TopologyType="2DRectMesh" Dimensions="@XPOINTS @YPOINTS"></Topology>
            <Geometry GeometryType="VXVY">
               <DataItem Dimensions="@YPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/YFaceCoordinates</DataItem>
               <DataItem Dimensions="@XPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@GROUP/XFaceCoordinates</DataItem>
            </Geometry>
"""

XMF_SOLUTION_SCALAR= """
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
   </Domain>
</Xdmf>
"""

parser = argparse.ArgumentParser()
parser.add_argument('--sampledir', nargs='?', const='.', default='.',
                    help='directory with all the simulation output')
parser.add_argument('--step', nargs=1, default=0, type=int,
                    help='timestep to be considered')
parser.add_argument('--yslice', nargs=1, default=0.0, type=float,
                    help='coordinate of the y plane to be processed')
args = parser.parse_args()
sample_dir = os.path.abspath(args.sampledir)

# Make directories for viz data
out_dir    = os.path.abspath("viz_data")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

outfile = os.path.join(out_dir, "slice.hdf")
hdf_out = h5py.File(outfile, "w")

resdir = os.path.join(sample_dir, "fluid_iter"+str(args.step).zfill(10))

# Grid directory
nodes_grid_dir = os.path.join(sample_dir, "nodes_grid")
# Check if this is Cartesian grid
CartesianGrid = os.path.exists(os.path.join(nodes_grid_dir, "xNodes"))
if CartesianGrid:
   assert os.path.exists(os.path.join(nodes_grid_dir, "yNodes")), "Could not find the yNodes folder"
   assert os.path.exists(os.path.join(nodes_grid_dir, "zNodes")), "Could not find the zNodes folder"
else:
   assert False, "Unsupported grid"

# Utilities for grid generation
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

with open('out_fluid.xmf', 'w') as xmf_out:
   xmf_out.write(XMF_HEADER)

   tiles = glob.glob(os.path.join(resdir, "*hdf"))
   assert len(tiles) > 1

   # Load attributes
   masterFile = os.path.join(resdir, "master.hdf")
   if os.path.exists(masterFile):
      tiles.remove(masterFile)
      hdf_master   = h5py.File(masterFile, 'r')
      simTime      = hdf_master.attrs.get("simTime")
      SpeciesNames = hdf_master.attrs.get("SpeciesNames")
      hdf_master.close()
   else:
      hdf_in       = h5py.File(tiles[0], 'r')
      simTime      = hdf_in.attrs.get("simTime")
      SpeciesNames = hdf_in.attrs.get("SpeciesNames")
      hdf_in.close()

   xmf_out.write(XMF_SOLUTION_HEADER.replace("@NAME", "slice"))

   for j, tl in enumerate(tiles):
      hdf_in = h5py.File(tl, 'r')
      lo, hi = getTileBoundsFromName(tl.split('/')[-1])

      if not os.path.exists(masterFile):
         # Sanity check
         assert simTime == hdf_in.attrs.get("simTime")
         assert (SpeciesNames == hdf_in.attrs.get("SpeciesNames")).all()

      # Extract domain size.
      nx = hdf_in['pressure'].shape[0]
      ny = hdf_in['pressure'].shape[1]
      nz = hdf_in['pressure'].shape[2]

      # check if we need any point in this slide
      ynodes = getTileGrid(CartesianGrid, nodes_grid_dir, "yNodes", lo[1], hi[1])
      if ((ynodes[ 0] < args.yslice) and
          (ynodes[-1] > args.yslice)):
         group = tl.split('/')[-1][0:-4]
         js = 0
         for jj, y in enumerate(ynodes):
            if ((ynodes[j  ] < args.yslice) and
                (ynodes[j+1] > args.yslice)):
               js = jj
               break

         # Grid
         hdf_out[group+"/XFaceCoordinates"] = getTileGrid(CartesianGrid, nodes_grid_dir, "zNodes", lo[2], hi[2])
         hdf_out[group+"/YFaceCoordinates"] = getTileGrid(CartesianGrid, nodes_grid_dir, "xNodes", lo[0], hi[0])

         # Data
         hdf_out[group+"/rho"        ] = hdf_in["rho"        ][:][:,js,:]
         hdf_out[group+"/pressure"   ] = hdf_in["pressure"   ][:][:,js,:]
         hdf_out[group+"/temperature"] = hdf_in["temperature"][:][:,js,:]
         hdf_out[group+"/velocity"   ] = hdf_in["velocity"   ][:][:,js,:,:]
         for isp, sp in enumerate(SpeciesNames):
            hdf_out[group+"/X_"+sp.decode()] = hdf_in['MolarFracs'][:][:,js,:,isp]

         xmf_out.write(XMF_TILE_HEADER
                       .replace("@N",       "%s" % group)
                       .replace("@XPOINTS", "%s" % str(nx+1))
                       .replace("@YPOINTS", "%s" % str(nz+1))
                       .replace("@GROUP",   "%s" % group)
                       .replace("@HDF_FILE","%s" % outfile))

         xmf_out.write(XMF_SOLUTION_SCALAR
                       .replace("@NAME",     "%s" % "rho")
                       .replace("@GROUP",    "%s" % group)
                       .replace("@HDF_FILE", "%s" % outfile)
                       .replace("@CELLS", "%s %s" % (nx,nz)))

         xmf_out.write(XMF_SOLUTION_SCALAR
                       .replace("@NAME",     "%s" % "pressure")
                       .replace("@GROUP",    "%s" % group)
                       .replace("@HDF_FILE", "%s" % outfile)
                       .replace("@CELLS", "%s %s" % (nx,nz)))

         xmf_out.write(XMF_SOLUTION_SCALAR
                       .replace("@NAME",     "%s" % "temperature")
                       .replace("@GROUP",    "%s" % group)
                       .replace("@HDF_FILE", "%s" % outfile)
                       .replace("@CELLS", "%s %s" % (nx,nz)))

         for sp in SpeciesNames:
            xmf_out.write(XMF_SOLUTION_SCALAR
                          .replace("@NAME",     "%s" % "X_"+sp.decode())
                          .replace("@GROUP",    "%s" % group)
                          .replace("@HDF_FILE", "%s" % outfile)
                          .replace("@CELLS", "%s %s" % (nx,nz)))

         xmf_out.write(XMF_SOLUTION_VECTOR
                       .replace("@NAME",     "%s" % "velocity")
                       .replace("@GROUP",    "%s" % group)
                       .replace("@HDF_FILE", "%s" % outfile)
                       .replace("@CELLS", "%s %s" % (nx,nz)))

      xmf_out.write(XMF_TILE_FOOTER)

   xmf_out.write(XMF_SOLUTION_FOOTER)
   xmf_out.write(XMF_FOOTER)

