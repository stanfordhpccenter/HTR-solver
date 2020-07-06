#!/usr/bin/env python2

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

##############################################################################
#                                 Compute grid                               #
##############################################################################
def GetGridFaces(Origin, Width, Num, Type, stretching, dx=1, xswitch=1, switch=1):
   if (Type == 'Uniform'):
      x = np.linspace(0.0, 1.0, Num+1)
      x *= Width
      x += Origin
   elif (Type == 'TanhMinus'):
      x = np.linspace(-1.0, 0.0, Num+1)
      x = np.tanh(stretching*x)/np.tanh(stretching)
      x = Width*(x+1.0)+Origin
   elif (Type == 'TanhPlus'):
      x = np.linspace( 0.0, 1.0, Num+1)
      x = np.tanh(stretching*x)/np.tanh(stretching)
      x = Width*x+Origin
   elif (Type == 'Tanh'):
      x = np.linspace(-1.0, 1.0, Num+1)
      x = np.tanh(stretching*x)/np.tanh(stretching)
      x = 0.5*Width*(x+1.0)+Origin
   elif (Type == 'SinhMinus'):
      x = np.linspace( 0.0, 1.0, Num+1)
      x = np.sinh(stretching*x)/np.sinh(stretching)
      x = Width*x+Origin
   elif (Type == 'SinhPlus'):
      x = np.linspace(-1.0, 0.0, Num+1)
      x = np.sinh(stretching*x)/np.sinh(stretching)
      x = Width*(x+1.0)+Origin
   elif (Type == 'Sinh'):
      x = np.linspace(-1.0, 1.0, Num+1)
      x = np.sinh(stretching*x)/np.sinh(stretching)
      x = 0.5*Width*(x+1.0)+Origin
   elif (Type == 'GeometricMinus'):
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
      stretch, = fsolve(objectiveGeomMinus, stretching)
      x = GenGeomMinus(stretch)
   elif (Type == 'GeometricPlus'):
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
      stretch, = fsolve(objectiveGeomPlus, stretching)
      x = GenGeomPlus(stretch)
   elif (Type == 'DoubleGeometricMinus'):
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
      stretch, = fsolve(objectiveDoubleGeomMinus, stretching)
      x = GenDoubleGeomMinus(stretch)
   return x

def GetGrid(Origin, Width, Num, Type, stretching, periodic, dx=1, xswitch=1, switch=1):
   x = GetGridFaces(Origin, Width, Num, Type, stretching, dx, xswitch, switch)
   if periodic:
      xc = np.zeros(Num)
      dx = np.zeros(Num)
      for i in range(0,Num):
         xc[i] = 0.5*(x[i+1]+x[i])
         dx[i] =     (x[i+1]-x[i])
   else:
      xc = np.zeros(Num+2)
      dx = np.zeros(Num+2)
      for i in range(1,Num+1):
         xc[i] = 0.5*(x[i]+x[i-1])
         dx[i] =     (x[i]-x[i-1])
      xc[0] = xc[1] - dx[1]
      dx[0] = dx[1]
      xc[Num+1] = xc[Num] + dx[Num]
      dx[Num+1] = dx[Num]
   return xc, dx

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

