import os
import shutil
import sys
import h5py
import subprocess
import argparse
import glob


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
                     <DataItem Dimensions="@ZPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/ZFaceCoordinates</DataItem>
                     <DataItem Dimensions="@YPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/YFaceCoordinates</DataItem>
                     <DataItem Dimensions="@XPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/XFaceCoordinates</DataItem>
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

def progressbar(it, prefix="", suffix = '', decimals = 1, length = 100, fill = '-', file=sys.stdout):
   count = len(it)
   def show(j):
      percent = ("{0:." + str(decimals) + "f}").format(100 * (j / float(count)))
      filledLength = int(length * j // count)
      bar = fill * filledLength + '-' * (length - filledLength)
      file.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
      file.flush()
   show(0)
   for i, item in enumerate(it):
      yield i, item
      show(i+1)
   file.write("\n")
   file.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--sampledir', nargs='?', const='.', default='.',
                    help='directory with all the simulation output')
parser.add_argument('--debugOut', default=False, action='store_true',
                    help='flag to process debug output of the simulation')
args = parser.parse_args()
sample_dir = os.path.abspath(args.sampledir)

# Make directories for viz data
out_dir    = os.path.abspath("viz_data")
grid_dir   = os.path.join(out_dir,"grid")

if not os.path.exists(grid_dir):
    os.makedirs(grid_dir)

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

with open('out_fluid.xmf', 'w') as xmf_out:
   xmf_out.write(XMF_HEADER)

   for i, sn in progressbar(snapshots, "Converting:"):

      # We still need to convert H5T_ARRAYs to numpy vectors...
      # TODO: get rid of this step as soon as you find out how to read H5T_ARRAYs with XDMF
      hdf_sol = None
      solutionfile = os.path.join(out_dir,sn.split('/')[-1])+".hdf"
      if not os.path.exists(solutionfile):
         hdf_sol = h5py.File(solutionfile, "w")
      ####################################################################################

      tiles = glob.glob(os.path.join(sn,"*hdf"))

      # Load attributes
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

         # Sanity check
         assert simTime == hdf_in.attrs.get("simTime")
         assert (SpeciesNames == hdf_in.attrs.get("SpeciesNames")).all()

         # Extract domain size.
         nx = hdf_in['pressure'].shape[0]
         ny = hdf_in['pressure'].shape[1]
         nz = hdf_in['pressure'].shape[2]

         # Generate grid file if it does not exist
         gridfile = os.path.join(grid_dir, tl.split('/')[-1])

         if not os.path.exists(gridfile):
            with h5py.File(gridfile, "w") as hdf_grid:
               centerCoordinates = hdf_in["centerCoordinates"][:][:,:,:,:]
               cellWidth = hdf_in["cellWidth"][:][:,:,:,:]
               XFaceCoordinates = []
               for i in range(nx):
                  XFaceCoordinates.append(centerCoordinates[i,0,0,2]-0.5*cellWidth[i,0,0,2])
               XFaceCoordinates.append(centerCoordinates[nx-1,0,0,2]+0.5*cellWidth[nx-1,0,0,2])
               YFaceCoordinates = []
               for i in range(ny):
                  YFaceCoordinates.append(centerCoordinates[0,i,0,1]-0.5*cellWidth[0,i,0,1])
               YFaceCoordinates.append(centerCoordinates[0,ny-1,0,1]+0.5*cellWidth[0,ny-1,0,1])
               ZFaceCoordinates = []
               for i in range(nz):
                  ZFaceCoordinates.append(centerCoordinates[0,0,i,0]-0.5*cellWidth[0,0,i,0])
               ZFaceCoordinates.append(centerCoordinates[0,0,nz-1,0]+0.5*cellWidth[0,0,nz-1,0])
               hdf_grid["XFaceCoordinates"] = XFaceCoordinates
               hdf_grid["YFaceCoordinates"] = YFaceCoordinates
               hdf_grid["ZFaceCoordinates"] = ZFaceCoordinates

         # We still need to convert H5T_ARRAYs to numpy vectors...
         # TODO: get rid of this step as soon as you find out how to read H5T_ARRAYs with XDMF
         group = tl.split('/')[-1][0:-4]
         if hdf_sol != None:
            hdf_sol[group+"/velocity"] = hdf_in["velocity"][:][:,:,:,:]
            for isp, sp in enumerate(SpeciesNames):
               hdf_sol[group+"/X_"+sp.decode()] = hdf_in['MolarFracs'][:][:,:,:,isp]
         ####################################################################################

         xmf_out.write(XMF_TILE_HEADER
                       .replace("@N", "%s" % tl.split('/')[-1][0:-4])
                       .replace("@XPOINTS", "%s" % str(nx+1))
                       .replace("@YPOINTS", "%s" % str(ny+1))
                       .replace("@ZPOINTS", "%s" % str(nz+1))
                       .replace("@HDF_FILE", gridfile))

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


         xmf_out.write(XMF_TILE_FOOTER)
         hdf_in.close()

      xmf_out.write(XMF_SOLUTION_FOOTER)
      if hdf_sol != None: hdf_sol.close()

   xmf_out.write(XMF_FOOTER)
