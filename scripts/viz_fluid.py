#!/usr/bin/env python3

import argparse
import h5py
import itertools
import json

XMF_HEADER = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="3.0">
  <Domain Name="Fluid">
    <Grid Name="FluidTimeSeries" GridType="Collection" CollectionType="Temporal">
"""

XMF_SOLUTION_HEADER = """
      <Grid Name="Fluid" GridType="Uniform">
        <Time Value="@TIMESTEP"/>
        <!-- Topology: orthonormal 3D grid -->
        <Topology TopologyType="3DRectMesh" Dimensions="@XPOINTS @YPOINTS @ZPOINTS"></Topology>
        <!-- Geometry: Node positions derived implicitly, based on grid origin and cell size -->
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
XMF_SOLUTION_VECTOR= """
        <Attribute Name="@NAME" AttributeType="Vector" Center="Cell">
          <DataItem Dimensions="@CELLS 3" NumberType="Float" Precision="8" Format="HDF">@HDF_FILE:/@NAME</DataItem>
        </Attribute>
"""
XMF_SOLUTION_FOOTER = """
      </Grid>
"""

XMF_FOOTER = """
    </Grid>
  </Domain>
</Xdmf>
"""

parser = argparse.ArgumentParser()
parser.add_argument('json_file',
                    help='original simulation configuration file')
parser.add_argument('-s', '--section', choices=['1','2'],
                    help='which section to visualize (if multi-section sim)')
parser.add_argument('hdf_file', nargs='+',
                    help='fluid restart file(s) to visualize')
args = parser.parse_args()

nx = None
ny = None
nz = None

simTime = {}
for (f, i) in zip(args.hdf_file, itertools.count()):
    hdf_in = h5py.File(f, 'r')
    # Extract domain size.
    if nx is None:
        nx = hdf_in['pressure'].shape[0]
        ny = hdf_in['pressure'].shape[1]
        nz = hdf_in['pressure'].shape[2]
    else:
        assert nx == hdf_in['pressure'].shape[0]
        assert ny == hdf_in['pressure'].shape[1]
        assert nz == hdf_in['pressure'].shape[2]
    hdf_out = h5py.File('out%010d.hdf' % i, 'w')
    # Load attributes
    simTime[i] = hdf_in.attrs.get("simTime")
    SpeciesNames = hdf_in.attrs.get("SpeciesNames")
    # Build face coordinates.
    centerCoordinates = hdf_in['centerCoordinates'][:][:,:,:,:]
    cellWidth = hdf_in['cellWidth'][:][:,:,:,:]
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
    hdf_out['XFaceCoordinates'] = XFaceCoordinates
    hdf_out['YFaceCoordinates'] = YFaceCoordinates
    hdf_out['ZFaceCoordinates'] = ZFaceCoordinates
    # Copy pressure over.
    hdf_out['pressure'] = hdf_in['pressure'][:]
    # Parse molar fractions in multiple fields giving the species name.
    MolarFracs = hdf_in['MolarFracs'][:][:,:,:,:]
    for isp, sp in enumerate(SpeciesNames):
       hdf_out['X_'+sp.decode()] = MolarFracs[:,:,:,isp]
    # Copy rho over.
    hdf_out['rho'] = hdf_in['rho'][:]
    # Copy temperature over.
    hdf_out['temperature'] = hdf_in['temperature'][:]
    # Convert velocity from an XxYxZ matrix of triples to an XxYxZx3 matrix.
    hdf_out['velocity'] = hdf_in['velocity'][:][:,:,:,:]
    hdf_out.close()
    hdf_in.close()

# NOTE: We flip the X and Z dimensions, because Legion dumps data in
# column-major order.
with open(args.json_file) as json_in:
    config = json.load(json_in)
    if args.section is not None:
        config = config['configs'][int(args.section)-1]
    # Compute number of boundary cells on each dimension.
    bx = nx - config['Grid']['zNum']
    by = ny - config['Grid']['yNum']
    bz = nz - config['Grid']['xNum']
    assert bx == 0 or bx == 2, 'Expected at most 1-cell boundary'
    assert by == 0 or by == 2, 'Expected at most 1-cell boundary'
    assert bz == 0 or bz == 2, 'Expected at most 1-cell boundary'

# NOTE: The XMF format expects grid dimensions in points, not cells, so we have
# to add 1 on each dimension.
with open('out_fluid.xmf', 'w') as xmf_out:
    xmf_out.write(XMF_HEADER)
    for i in range(len(args.hdf_file)):
        xmf_out.write(XMF_SOLUTION_HEADER
                      .replace('@TIMESTEP','%.8g'% simTime[i])
                      .replace('@XPOINTS', '%s' % str(nx+1))
                      .replace('@YPOINTS', '%s' % str(ny+1))
                      .replace('@ZPOINTS', '%s' % str(nz+1))
                      .replace('@CELLS', '%s %s %s' % (nx,ny,nz))
                      .replace('@HDF_FILE', 'out%010d.hdf' % i))
        xmf_out.write(XMF_SOLUTION_SCALAR
                      .replace('@NAME','%s'% 'pressure')
                      .replace('@CELLS', '%s %s %s' % (nx,ny,nz))
                      .replace('@HDF_FILE', 'out%010d.hdf' % i))
        xmf_out.write(XMF_SOLUTION_SCALAR
                      .replace('@NAME','%s'% 'rho')
                      .replace('@CELLS', '%s %s %s' % (nx,ny,nz))
                      .replace('@HDF_FILE', 'out%010d.hdf' % i))
        xmf_out.write(XMF_SOLUTION_SCALAR
                      .replace('@NAME','%s'% 'temperature')
                      .replace('@CELLS', '%s %s %s' % (nx,ny,nz))
                      .replace('@HDF_FILE', 'out%010d.hdf' % i))
        for sp in SpeciesNames:
            xmf_out.write(XMF_SOLUTION_SCALAR
                          .replace('@NAME','%s'% 'X_'+sp.decode())
                          .replace('@CELLS', '%s %s %s' % (nx,ny,nz))
                          .replace('@HDF_FILE', 'out%010d.hdf' % i))
        xmf_out.write(XMF_SOLUTION_VECTOR
                      .replace('@NAME','%s'% 'velocity')
                      .replace('@CELLS', '%s %s %s' % (nx,ny,nz))
                      .replace('@HDF_FILE', 'out%010d.hdf' % i))
        xmf_out.write(XMF_SOLUTION_FOOTER)
    xmf_out.write(XMF_FOOTER)
