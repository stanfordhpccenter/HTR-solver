import os
import shutil
import sys
import h5py
import subprocess
import argparse
import glob

# Load averages module
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import Averages

XMF_HEADER = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="3.0">
   <Domain Name="Fluid">
"""

XMF_SOLUTION_HEADER = """
      <Grid Name="@NAME" GridType="Uniform">
         <Topology TopologyType="2DRectMesh" Dimensions="@XPOINTS @YPOINTS"></Topology>
         <Geometry GeometryType="VXVY">
            <DataItem Dimensions="@YPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/YFaceCoordinates</DataItem>
            <DataItem Dimensions="@XPOINTS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/XFaceCoordinates</DataItem>
         </Geometry>
"""

XMF_SOLUTION_SCALAR= """
         <Attribute Name="@NAME" AttributeType="Scalar" Center="Cell">
            <DataItem Dimensions="@CELLS" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@NAME</DataItem>
         </Attribute>
"""

XMF_SOLUTION_VECTOR= """
         <Attribute Name="@NAME" AttributeType="Vector" Center="Cell">
            <DataItem Dimensions="@CELLS 3" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@NAME</DataItem>
         </Attribute>
"""

XMF_SOLUTION_FOOTER = """
      </Grid>
"""

XMF_FOOTER = """
   </Domain>
</Xdmf>
"""

parser = argparse.ArgumentParser()
parser.add_argument('--avgdir', nargs='?', const='.', default='.',
                    help='directory with all the simulation output')
parser.add_argument('--nplane', default=0, type=int,
                    help='index of the average plane to be processed')
args = parser.parse_args()

# Make directories for viz data
out_dir    = os.path.abspath("avg_viz_data")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Check for solution snapshots
avg = Averages.avg1D(args.avgdir, args.nplane)

with open('out_fluid.xmf', 'w') as xmf_out:
   xmf_out.write(XMF_HEADER)

   solutionfile = os.path.join(out_dir, "data.hdf")

   # Load attributes
   SpeciesNames = avg.SpeciesNames

   # Extract domain size.
   nx = avg.centerCoordinates.shape[0]
   ny = avg.centerCoordinates.shape[1]

   # Generate grid file if it does not exist
   gridfile = os.path.join(out_dir, "grid.hdf")

   with h5py.File(gridfile, "w") as hdf_grid:
      cc = avg.centerCoordinates

      XFaceCoordinates = []
      if nx > 1:
         XFaceCoordinates.append(1.5*cc[0,0,1] - 0.5*cc[1,0,1])
         for i in range(1, nx):
            XFaceCoordinates.append(0.5*(cc[i-1,0,1]+cc[i,0,1]))
         XFaceCoordinates.append(2*cc[nx-1,0,1] - XFaceCoordinates[nx-1])
      else:
         XFaceCoordinates.append(cc[0,0,1] - 1e-13)
         XFaceCoordinates.append(cc[0,0,1] + 1e-13)

      YFaceCoordinates = []
      if ny > 1:
         YFaceCoordinates.append(1.5*cc[0,0,0] - 0.5*cc[0,1,0])
         for i in range(1, ny):
            YFaceCoordinates.append(0.5*(cc[0,i-1,0]+cc[0,i,0]))
         YFaceCoordinates.append(2*cc[0,ny-1,0] - YFaceCoordinates[ny-1])
      else:
         YFaceCoordinates.append(cc[0,0,0] - 1e-13)
         YFaceCoordinates.append(cc[0,0,0] + 1e-13)

      hdf_grid["XFaceCoordinates"] = XFaceCoordinates
      hdf_grid["YFaceCoordinates"] = YFaceCoordinates

   # Store solution
   with h5py.File(solutionfile, "w") as hdf_sol:
      hdf_sol["rho"] = avg.rho_avg
      hdf_sol["temperature"] = avg.temperature_avg
      hdf_sol["pressure"] = avg.pressure_avg
      hdf_sol["velocity"] = avg.velocity_avg
      for isp, sp in enumerate(SpeciesNames):
         hdf_sol["X_"+sp.decode()] = avg.MolarFracs_avg[:,:,isp]

   xmf_out.write(XMF_SOLUTION_HEADER
                 .replace("@NAME",    "%s" % args.avgdir.split("/")[0])
                 .replace("@XPOINTS", "%s" % str(nx+1))
                 .replace("@YPOINTS", "%s" % str(ny+1))
                 .replace("@HDF_FILE", gridfile))

   xmf_out.write(XMF_SOLUTION_SCALAR
                 .replace("@NAME","%s" % "rho")
                 .replace("@HDF_FILE", "%s" % solutionfile)
                 .replace("@CELLS", "%s %s" % (nx,ny)))

   xmf_out.write(XMF_SOLUTION_SCALAR
                 .replace("@NAME","%s" % "temperature")
                 .replace("@HDF_FILE", "%s" % solutionfile)
                 .replace("@CELLS", "%s %s" % (nx,ny)))


   xmf_out.write(XMF_SOLUTION_SCALAR
                 .replace("@NAME", "%s" % "pressure")
                 .replace("@HDF_FILE", "%s" % solutionfile)
                 .replace("@CELLS", "%s %s" % (nx,ny)))

   for sp in SpeciesNames:
      xmf_out.write(XMF_SOLUTION_SCALAR
                    .replace("@NAME","%s" % "X_"+sp.decode())
                    .replace("@HDF_FILE", "%s" % solutionfile)
                    .replace('@CELLS', '%s %s' % (nx,ny)))

   xmf_out.write(XMF_SOLUTION_VECTOR
                 .replace("@NAME", "%s" % "velocity")
                 .replace("@HDF_FILE", "%s" % solutionfile)
                 .replace("@CELLS", "%s %s" % (nx,ny)))

   xmf_out.write(XMF_SOLUTION_FOOTER)
   xmf_out.write(XMF_FOOTER)
