#!/usr/bin/env python3

def GetDensity(T, P, config):
   return P/(T*config["Flow"]["mixture"]["gasConstant"])

def GetViscosity(T, config):
   if (config["Flow"]["mixture"]["viscosityModel"]["type"] == "Constant"):
      viscosity = config["Flow"]["mixture"]["viscosityModel"]["Visc"]
   elif (config["Flow"]["mixture"]["viscosityModel"]["type"] == "PowerLaw"):
      viscosity = config["Flow"]["mixture"]["viscosityModel"]["ViscRef"]*(T/config["Flow"]["mixture"]["viscosityModel"]["TempRef"])**0.7
   elif (config["Flow"]["mixture"]["viscosityModel"]["type"] == "Sutherland"):
      viscosity = (config["Flow"]["mixture"]["viscosityModel"]["ViscRef"]*(T/config["Flow"]["mixture"]["viscosityModel"]["TempRef"])**1.5)*(config["Flow"]["mixture"]["viscosityModel"]["TempRef"]+config["Flow"]["mixture"]["viscosityModel"]["SRef"])/(T+config["Flow"]["mixture"]["viscosityModel"]["SRef"])
   else: 
      assert False
   return viscosity

