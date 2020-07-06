#!/usr/bin/env python2

def GetDensity(T, P, config):
   return P/(T*config["Flow"]["gasConstant"])

def GetViscosity(T, config):
   if (config["Flow"]["viscosityModel"]["type"] == "Constant"):
      viscosity = config["Flow"]["viscosityModel"]["Visc"]
   elif (config["Flow"]["viscosityModel"]["type"] == "PowerLaw"):
      viscosity = config["Flow"]["viscosityModel"]["ViscRef"]*(T/config["Flow"]["viscosityModel"]["TempRef"])**0.7
   elif (config["Flow"]["viscosityModel"]["type"] == "Sutherland"):
      viscosity = (config["Flow"]["viscosityModel"]["ViscRef"]*(T/config["Flow"]["viscosityModel"]["TempRef"])**1.5)*(config["Flow"]["viscosityModel"]["TempRef"]+config["Flow"]["viscosityModel"]["SRef"])/(T+config["Flow"]["viscosityModel"]["SRef"])
   else: 
      assert False
   return viscosity

