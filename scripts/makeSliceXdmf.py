import os
import shutil
import sys
import h5py
import subprocess
import argparse
import glob
import numpy as np

# Load averages module
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import Averages

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
parser.add_argument('--resdir', nargs='?', const='.', default='.',
                    help='directory with all the simulation output')
parser.add_argument('--yslice', nargs=1, default=0.0, type=float,
                    help='coordinate of the y plane to be processed')
args = parser.parse_args()

# Make directories for viz data
out_dir    = os.path.abspath("viz_data")
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

outfile ="viz_data/slice.hdf"
#outfile = os.path.join(out_dir, "slice.hdf")
hdf_out = h5py.File(outfile, "w")

with open('out_fluid.xmf', 'w') as xmf_out:
   xmf_out.write(XMF_HEADER)

   tiles = glob.glob(os.path.join(args.resdir, "*hdf"))
   assert len(tiles) > 1

   # Load attributes
   hdf_in = h5py.File(tiles[0], 'r')
   simTime = hdf_in.attrs.get("simTime")
   SpeciesNames = hdf_in.attrs.get("SpeciesNames")
   hdf_in.close()

   xmf_out.write(XMF_SOLUTION_HEADER.replace("@NAME", "slice"))

   for j, tl in enumerate(tiles):
      hdf_in = h5py.File(tl, 'r')

      # Sanity check
      assert simTime == hdf_in.attrs.get("simTime")
      assert (SpeciesNames == hdf_in.attrs.get("SpeciesNames")).all()

      # Extract domain size.
      nx = hdf_in['pressure'].shape[0]
      ny = hdf_in['pressure'].shape[1]
      nz = hdf_in['pressure'].shape[2]

      # check if we need any point in this slide
      centerCoordinates = hdf_in["centerCoordinates"][:][:,:,:,:]
      if ((centerCoordinates[0, 0,0,1] < args.yslice) and
          (centerCoordinates[0,-1,0,1] > args.yslice)):
         group = tl.split('/')[-1][0:-4]
         js = (np.abs(centerCoordinates[0,:,0,1] - args.yslice).argmin())
#         print(group)
         cellWidth = hdf_in["cellWidth"][:][:,js,:,:]

         # grid
         XFaceCoordinates = []
         for i in range(nx):
            XFaceCoordinates.append(centerCoordinates[i,js,0,2]-0.5*cellWidth[i   ,0,2])
         XFaceCoordinates.append(centerCoordinates[nx-1,js,0,2]+0.5*cellWidth[nx-1,0,2])
         ZFaceCoordinates = []
         for i in range(nz):
            ZFaceCoordinates.append(centerCoordinates[0,js,i,0]-0.5*cellWidth[0,i   ,0])
         ZFaceCoordinates.append(centerCoordinates[0,js,nz-1,0]+0.5*cellWidth[0,nz-1,0])

         hdf_out[group+"/XFaceCoordinates"] = XFaceCoordinates
         hdf_out[group+"/YFaceCoordinates"] = ZFaceCoordinates

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

