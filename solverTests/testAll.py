#!/usr/bin/env python3

import os
import sys
import h5py
import glob
import json
import shutil
import unittest
import itertools
import subprocess
import numpy as np

executable = os.path.join(os.path.expandvars("$HTR_DIR"), "prometeo.sh")

# Leave a tollerance for the noise introduced by atomic
def checkSolutionFile(path, refFile, tol=1e-10):
   tiles = glob.glob(os.path.join(path,"*hdf"))
   if len(tiles) == 0: 
      print("ERROR: could not find solution file in {}".format(path))
      return False

   for tl in tiles:
      lb = list(map(int, tl.split("/")[-1].split("-")[0].split(",")))
      tile = h5py.File(tl, "r")
      for fld in tile:
         f   =    tile[fld][:]
         ref = refFile[fld][:]
         if  (len(f.shape) == 3):
            for (k,j,i), value in np.ndenumerate(f):
               rval = ref[k+lb[2],j+lb[1],i+lb[0]]
               if (abs(value - rval) > max(tol*abs(rval), 1e-12)):
                  print("Error on field: {}, delta = {}, tol = {}, expected = {}".format(fld, value - rval, max(tol*abs(rval), 1e-12), rval))
                  return False
         elif (len(f.shape) == 4):
            for (k,j,i,l), value in np.ndenumerate(f):
               rval = ref[k+lb[2],j+lb[1],i+lb[0],l]
               if (abs(value - rval) > max(tol*abs(rval), 1e-12)):
                  print("Error on field: {}, delta = {}, tol = {}, expected = {}".format(fld, value - rval, max(tol*abs(rval), 1e-12), rval))
                  return False
      tile.close()
   # if we arrive here, the solution is good
   return True

      # Load reference solution
def loadRef():
   if int(os.getenv("USE_CUDA", "0")):
#      ref = h5py.File("gpu_ref.hdf", "r")
      ref = h5py.File("cpu_ref.hdf", "r")
   else:
      ref = h5py.File("cpu_ref.hdf", "r")
   return ref

class TestTiledBase(object):

   def execute(self, name):
      return self.assertTrue(False, msg="{} should overwrite execute method".format(self.testName))
      
   def test(self):
      self.cwd = os.getcwd()
      os.chdir(os.path.expandvars("$HTR_DIR/solverTests/"+self.testName))

      try:
         ref = loadRef()
      except:
         self.fail(self.testName + "failed loading reference solution")

      self.runTiledTest(ref)

      try:
         if os.path.exists("InflowProfile"): shutil.rmtree("InflowProfile")
         if os.path.exists("InitialCase"): shutil.rmtree("InitialCase")
         for p in range(1, self.npart+1):
            fileList = glob.glob(str(p)+"*.json")
            for f in fileList: os.remove(f)
            dirList = glob.glob(str(p)+"*")
            for f in dirList: shutil.rmtree(f)
      except:
         self.fail(self.testName+" teardown failed")

      os.chdir(self.cwd)


class Test2DTiledBase(TestTiledBase):
   def runTiledTest(self, ref):
      for case in itertools.product(range(1, self.npart+1), repeat=2):
         name = "%dx%dx%d"%(case[0], case[1], 1)
         with self.subTest(name):
            #print(" > case: %s" % name)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = (case[0], case[1], 1)
               config["Mapping"]["tilesPerRank"] = (case[0], case[1], 1)
               nstep = int(config["Integrator"]["maxIter"])
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = self.execute(name)
            # run the check
            self.assertTrue(checkSolutionFile(os.path.join(name, "sample0/fluid_iter"+str(nstep).zfill(10)), ref),
                                              msg="Error on {} of {} test".format(name, self.testName))

class Test3DTiledBase(TestTiledBase):
   def runTiledTest(self, ref):
      for case in itertools.product(range(1, self.npart+1), repeat=3):
         name = "%dx%dx%d"%(case)
         with self.subTest(name):
            #print(" > case: %s" % name)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = case
               config["Mapping"]["tilesPerRank"] = case
               nstep = int(config["Integrator"]["maxIter"])
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = self.execute(name)
            # run the check
            self.assertTrue(checkSolutionFile(os.path.join(name, "sample0/fluid_iter"+str(nstep).zfill(10)), ref),
                                              msg="Error on {} of {} test".format(name, self.testName))

if __name__ == "__main__":
   suite = unittest.TestLoader().discover(os.path.expandvars("$HTR_DIR/solverTests/"), pattern = "test.py")
   result = 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
   sys.exit(result)
