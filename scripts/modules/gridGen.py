#!/usr/bin/env python3

import os
import HTRrestart
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint

##############################################################################
#                                 Compute grid                               #
##############################################################################
def GetNodes(Origin, Width, Num, Type, dx=1, xswitch=1, switch=1):
   if (Type["type"] == "Uniform"):
      x = np.linspace(0.0, 1.0, Num+1)
      x *= Width
      x += Origin
   elif (Type["type"] == "Cosine"):
      x = np.linspace(-1.0, 1.0, Num+1)
      x = -1*np.cos(PI*(x+1)/2)
      x = 0.5*Width*(x+1.0)+Origin
   elif (Type["type"] == "TanhMinus"):
      x = np.linspace(-1.0, 0.0, Num+1)
      x = np.tanh(Type["Stretching"]*x)/np.tanh(Type["Stretching"])
      x = Width*(x+1.0)+Origin
   elif (Type["type"] == "TanhPlus"):
      x = np.linspace( 0.0, 1.0, Num+1)
      x = np.tanh(Type["Stretching"]*x)/np.tanh(Type["Stretching"])
      x = Width*x+Origin
   elif (Type["type"] == "Tanh"):
      x = np.linspace(-1.0, 1.0, Num+1)
      x = np.tanh(Type["Stretching"]*x)/np.tanh(Type["Stretching"])
      x = 0.5*Width*(x+1.0)+Origin
   elif (Type["type"] == "SinhMinus"):
      x = np.linspace( 0.0, 1.0, Num+1)
      x = np.sinh(Type["Stretching"]*x)/np.sinh(Type["Stretching"])
      x = Width*x+Origin
   elif (Type["type"] == "SinhPlus"):
      x = np.linspace(-1.0, 0.0, Num+1)
      x = np.sinh(Type["Stretching"]*x)/np.sinh(Type["Stretching"])
      x = Width*(x+1.0)+Origin
   elif (Type["type"] == "Sinh"):
      x = np.linspace(-1.0, 1.0, Num+1)
      x = np.sinh(Type["Stretching"]*x)/np.sinh(Type["Stretching"])
      x = 0.5*Width*(x+1.0)+Origin
   elif (Type["type"] == "GeometricMinus"):
      assert dx<Width
      def GenGeomMinus(stretch):
         x = np.zeros(Num+1)
         x[0] = Origin
         for i in range(1,Num+1):
            x[i] = x[i-1] + dx*stretch**(i-1)
         return x
      def objectiveGeomMinus(Stretching):
         x = GenGeomMinus(Stretching)
         return x[Num]-(Width+Origin)
      stretch, = fsolve(objectiveGeomMinus, Type["Stretching"])
      x = GenGeomMinus(stretch)
   elif (Type["type"] == "GeometricPlus"):
      assert dx<Width
      def GenGeomPlus(stretch):
         x = np.zeros(Num+1)
         x[Num] = Origin+Width
         x[Num-1] = x[Num]-dx
         for i in range(1,Num+1):
            x[Num-i] = x[Num-i+1] - dx*stretch**(i-1)
         return x
      def objectiveGeomPlus(Stretching):
         x = GenGeomPlus(Stretching)
         return x[0]-Origin
      stretch, = fsolve(objectiveGeomPlus, Type["Stretching"])
      x = GenGeomPlus(stretch)
   elif (Type["type"] == "DoubleGeometricMinus"):
      assert dx<Width
      def GenDoubleGeomMinus(stretch):
         x = np.zeros(Num+1)
         my_dx = dx
         x[0] = Origin
         for i in range(1,Num+1):
            x[i] = x[i-1] + my_dx
            if x[i] > xswitch: my_dx *= stretch*switch
            else : my_dx *= stretch
         return x
      def objectiveDoubleGeomMinus(Stretching):
         x = GenDoubleGeomMinus(Stretching)
         return x[Num]-(Width+Origin)
      stretch, = fsolve(objectiveDoubleGeomMinus, Type["Stretching"])
      x = GenDoubleGeomMinus(stretch)
   else:
      assert False, "Unknown grid type {}".format(Type["type"])

   return x

def GetCellCenters(x, Num, periodic, StagMinus, StagPlus):
   if periodic:
      c = np.zeros(Num)
      d = np.zeros(Num)
      for i in range(0,Num):
         c[i] = 0.5*(x[i+1]+x[i])
         d[i] =     (x[i+1]-x[i])
   else:
      c = np.zeros(Num+2)
      d = np.zeros(Num+2)
      for i in range(1,Num+1):
         c[i] = 0.5*(x[i]+x[i-1])
         d[i] =     (x[i]-x[i-1])

      if StagMinus:
         c[0] = x[0]
         d[0] = 1e-12
      else:
         c[0] = c[1] - d[1]
         d[0] = d[1]

      if StagPlus:
         c[Num+1] = x[Num]
         d[Num+1] = 1e-12
      else:
         c[Num+1] = c[Num] + d[Num]
         d[Num+1] = d[Num]
   return c, d

def GetGrid(Origin, Width, Num, Type, periodic,
            StagMinus = False, StagPlus = False,
               dMinus = None ,    dPlus = None,
            dx=1, xswitch=1, switch=1):
   assert not(dMinus and dPlus), "You cannot specify the spacing on left and right at the same time"

   # Adapt stretching to match dMinus
   if dMinus:
      assert ((Type["type"] != "Uniform") or (Type["type"] != "Cosine")), "There isn't any stretching to be adapted for Uniform or Cosine grids"
      def objective(s):
         Type["Stretching"] = s[0]
         x = GetNodes(Origin, Width, Num, Type, dx=dx, xswitch=xswitch, switch=switch)
         return (x[1] - x[0]) - dMinus
      Type["Stretching"], = fsolve(objective, 1.0)

   # Adapt stretching to match dPlus
   if dPlus:
      assert ((Type["type"] != "Uniform") or (Type["type"] != "Cosine")), "There isn't any stretching to be adapted for Uniform or Cosine grids"
      def objective(s):
         Type["Stretching"] = s[0]
         x = GetNodes(Origin, Width, Num, Type, dx=dx, xswitch=xswitch, switch=switch)
         return (x[-1] - x[-2]) - dPlus
      Type["Stretching"], = fsolve(objective, 1.0)

   # Generate nodes grid
   x = GetNodes(Origin, Width, Num, Type, dx=dx, xswitch=xswitch, switch=switch)

   # Compute cell centers and cell widths
   return GetCellCenters(x, Num, periodic, StagMinus, StagPlus)

# This method writes a new grid for HTR
# Parameters:
# - config: config data structure from the json file
# - dxMinus: desired spacing on x minus side (optional)
# - dxPlus:  desired spacing on x plus  side (optional)
# - dyMinus: desired spacing on y minus side (optional)
# - dyPlus:  desired spacing on y plus  side (optional)
# - dzMinus: desired spacing on z minus side (optional)
# - dzPlus:  desired spacing on z plus  side (optional)
# - xNodes: function that provides nodes grid along x (i) -> double  (optional)
# - yNodes: function that provides nodes grid along y (i) -> double  (optional)
# - zNodes: function that provides nodes grid along z (i) -> double  (optional)
def getCellCenters(config,
            dxMinus = None, dxPlus = None,
            dyMinus = None, dyPlus = None,
            dzMinus = None, dzPlus = None,
            xNodes = None, yNodes = None, zNodes = None):

   def isStaggered(t):
      return ((t == "Dirichlet") or
              (t == "AdiabaticWall") or
              (t == "IsothermalWall") or
              (t == "SuctionAndBlowingWall"))

   # Check if grid is periodic in any direction
   if config["BC"]["xBCLeft"]["type"] == "Periodic":
      assert config["BC"]["xBCLeft"]["type"] == config["BC"]["xBCRight"]["type"]
      xPeriodic  = True
      xStagMinus = False
      xStagPlus  = False
   else:
      xPeriodic = False
      xStagMinus = isStaggered(config["BC"]["xBCLeft" ]["type"])
      xStagPlus  = isStaggered(config["BC"]["xBCRight"]["type"])

   if config["BC"]["yBCLeft"]["type"] == "Periodic":
      assert config["BC"]["yBCLeft"]["type"] == config["BC"]["yBCRight"]["type"]
      yPeriodic = True
      yStagMinus = False
      yStagPlus  = False
   else:
      yPeriodic = False
      yStagMinus = isStaggered(config["BC"]["yBCLeft" ]["type"])
      yStagPlus  = isStaggered(config["BC"]["yBCRight"]["type"])

   if config["BC"]["zBCLeft"]["type"] == "Periodic":
      assert config["BC"]["zBCLeft"]["type"] == config["BC"]["zBCRight"]["type"]
      zPeriodic = True
      zStagMinus = False
      zStagPlus  = False
   else:
      zPeriodic = False
      zStagMinus = isStaggered(config["BC"]["zBCLeft" ]["type"])
      zStagPlus  = isStaggered(config["BC"]["zBCRight"]["type"])

   if config["Grid"]["GridInput"]["type"] == "Cartesian":
      # Genearae a cartesian grid
      xGrid, dx = GetGrid(config["Grid"]["GridInput"]["origin"][0],
                          config["Grid"]["GridInput"]["width"][0],
                          config["Grid"]["xNum"],
                          config["Grid"]["GridInput"]["xType"],
                          xPeriodic,
                          StagMinus = xStagMinus,
                          StagPlus  = xStagPlus,
                          dMinus = dxMinus,
                          dPlus  = dxPlus)

      yGrid, dy = GetGrid(config["Grid"]["GridInput"]["origin"][1],
                          config["Grid"]["GridInput"]["width"][1],
                          config["Grid"]["yNum"],
                          config["Grid"]["GridInput"]["yType"],
                          yPeriodic,
                          StagMinus = yStagMinus,
                          StagPlus  = yStagPlus,
                          dMinus = dyMinus,
                          dPlus  = dyPlus)

      zGrid, dz = GetGrid(config["Grid"]["GridInput"]["origin"][2],
                          config["Grid"]["GridInput"]["width"][2],
                          config["Grid"]["zNum"],
                          config["Grid"]["GridInput"]["zType"],
                          zPeriodic,
                          StagMinus = zStagMinus,
                          StagPlus  = zStagPlus,
                          dMinus = dzMinus,
                          dPlus  = dzPlus)

   elif config["Grid"]["GridInput"]["type"] == "FromFile":
      assert config["Grid"]["GridInput"]["gridDir"], "gridDir has to be specified in Grid/GridInput input section"
      assert xNodes, "xNodes function has to be provided in FromFile mode"
      assert yNodes, "yNodes function has to be provided in FromFile mode"
      assert zNodes, "zNodes function has to be provided in FromFile mode"

      # Create the grid
      xN = [xNodes(i) for i in range(config["Grid"]["xNum"]+1)]
      yN = [yNodes(j) for j in range(config["Grid"]["yNum"]+1)]
      zN = [zNodes(k) for k in range(config["Grid"]["zNum"]+1)]

      # Define bounding box
      bBox = np.zeros(1, dtype=HTRrestart.bBoxType)
      bBox[0][0] = [xN[ 0], yN[ 0], zN[ 0]]
      bBox[0][1] = [xN[-1], yN[ 0], zN[ 0]]
      bBox[0][2] = [xN[-1], yN[-1], zN[ 0]]
      bBox[0][3] = [xN[ 0], yN[-1], zN[ 0]]
      bBox[0][4] = [xN[ 0], yN[ 0], zN[-1]]
      bBox[0][5] = [xN[-1], yN[ 0], zN[-1]]
      bBox[0][6] = [xN[-1], yN[-1], zN[-1]]
      bBox[0][7] = [xN[ 0], yN[-1], zN[-1]]

      # Dump grids to file
      # Create the directory if it does not exist
      if not os.path.exists(config["Grid"]["GridInput"]["gridDir"]):
         os.makedirs(config["Grid"]["GridInput"]["gridDir"])

      folder = os.path.join(config["Grid"]["GridInput"]["gridDir"], "xNodes")
      HTRrestart.HTROneDgrid(folder, xN, bBox).dump()
      folder = os.path.join(config["Grid"]["GridInput"]["gridDir"], "yNodes")
      HTRrestart.HTROneDgrid(folder, yN, bBox).dump()
      folder = os.path.join(config["Grid"]["GridInput"]["gridDir"], "zNodes")
      HTRrestart.HTROneDgrid(folder, zN, bBox).dump()

      # Define cell centers
      xGrid, dx = GetCellCenters(xN, config["Grid"]["xNum"], xPeriodic, xStagMinus, xStagPlus)
      yGrid, dy = GetCellCenters(yN, config["Grid"]["yNum"], yPeriodic, yStagMinus, yStagPlus)
      zGrid, dz = GetCellCenters(zN, config["Grid"]["zNum"], zPeriodic, zStagMinus, zStagPlus)

   else:
      assert False, "Unrecognized GridInput type: {}".format(config["Grid"]["GridInput"]["type"])

   return xGrid, yGrid, zGrid, dx, dy, dz

def GetGridBL(origin, n1, n2, n3, x1, x2, x3):
   nx = n1+n2+n3
   width = x1+x2+x3

   xf = np.zeros(nx+1)

   xf[n1:n1+n2+1] = GetGridFaces(origin+x1, x2, n2, 'Uniform', 1.0)
   xf[0:n1+1]     = GetGridFaces(origin, x1, n1, 'GeometricPlus', 1.1, (xf[n1+1]-xf[n1]))
   xf[n1+n2:nx+1] = GetGridFaces(origin+x1+x2, x3, n3, "GeometricMinus", 1.1, xf[n1+n2]-xf[n1+n2-1])

   dx = np.zeros(nx+2)
   for i in range(1,nx+1):
      dx[i] = xf[i]-xf[i-1]
   dx[0] = dx[1]
   dx[nx+1] = dx[nx]

   xc = np.zeros(nx+2)
   for i in range(1,nx+1):
      xc[i] = 0.5*(xf[i-1]+xf[i])
   xc[0] = xc[1] - dx[1]
   xc[nx+1] = xc[nx] + dx[nx]

   return xc, dx, nx, width

