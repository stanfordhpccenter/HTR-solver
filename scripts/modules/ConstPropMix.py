#!/usr/bin/env python2

def GetDensity(T, P, config):
   return P/(T*config["Flow"]["gasConstant"])

def GetViscosity(T, config):
   if (config["Flow"]["viscosityModel"] == "Constant"):
      viscosity = config["Flow"]["constantVisc"]
   elif (config["Flow"]["viscosityModel"] == "PowerLaw"):
      viscosity = config["Flow"]["powerlawViscRef"]*(T/config["Flow"]["powerlawTempRef"])**0.7
   elif (config["Flow"]["viscosityModel"] == "Sutherland"):
      viscosity = (config["Flow"]["sutherlandViscRef"]*(T/config["Flow"]["sutherlandTempRef"])**1.5)*(config["Flow"]["sutherlandTempRef"]+config["Flow"]["sutherlandSRef"])/(T+config["Flow"]["sutherlandSRef"])
   else: 
      assert False
   return viscosity

