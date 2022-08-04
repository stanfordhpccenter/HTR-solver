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

# load HTR modules
sys.path.insert(0, os.path.expandvars("$HTR_DIR/scripts/modules"))
import HTRrestart

os.environ["CI_RUN"] = "1"
executable = os.path.join(os.path.expandvars("$HTR_DIR"), "prometeo.sh")
ranks = int(os.getenv('RANKS', 1))

# Leave a tollerance for the noise introduced by atomic
def checkSolutionFile(solution, refFile, tol=1e-10):
   for fld in refFile:
      f = solution.load(fld)
      diff = np.absolute(f[:] - refFile[fld][:])
      if (diff > np.maximum(tol*abs(refFile[fld][:]), 1e-12)).any():
         max_diff = np.max(diff)
         max_loc = np.unravel_index(np.argmax(diff), diff.shape)
         refVal = refFile[fld][:][max_loc]
         tol = max(tol*abs(refVal), 1e-12)
         print("Error on field: {}, delta = {}, location = {}, tol = {}, expected = {}".format(
               fld, max_diff, max_loc, tol, refVal))
         return False
   # if we arrive here, the solution is good
   return True

# Download/update the repository of reference data
def downloadRefData():
   subprocess.check_call(['git', 'submodule', 'update', '--init'],
                         cwd=os.path.expandvars("$HTR_DIR"))

class TestTiledBase(object):

   # Load reference solution
   def loadRef(self):
      baseDir = os.path.join(os.path.expandvars("$HTR_DIR"), "solverTests/referenceData/Cartesian")
      baseDir = os.path.join(baseDir, self.testName)
      if int(os.getenv("USE_CUDA", "0")):
         # Check if we have specific reference for GPUs
         fileName = os.path.join(baseDir, "gpu_ref.hdf")
         if (not os.path.exists(fileName)):
            fileName = os.path.join(baseDir, "cpu_ref.hdf")
            self.assertTrue(os.path.exists(fileName),
                            msg="Could not find the reference data for {} test".format(self.testName))
      else:
         fileName = os.path.join(baseDir, "cpu_ref.hdf")
         self.assertTrue(os.path.exists(fileName),
                         msg="Could not find the reference data for {} test".format(self.testName))
      return h5py.File(fileName, "r")

   # Overloaded function that executes the test
   def execute(self, name):
      return self.assertTrue(False, msg="{} should overwrite execute method".format(self.testName))

   # Main driver of the test
   def test(self):
      self.cwd = os.getcwd()
      os.chdir(os.path.expandvars("$HTR_DIR/solverTests/"+self.testName))

      try:
         ref = self.loadRef()
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
      def runCase(t, tPR):
         name = "%dx%dx%d"%(t[0], t[1], 1)
         with self.subTest(name):
            #print(" > case: %s" % name)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = (  t[0],   t[1], 1)
               config["Mapping"]["tilesPerRank"] = (tPR[0], tPR[1], 1)
               nstep = int(config["Integrator"]["maxIter"])
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = self.execute(name)
            # run the check
            interpreter = HTRrestart.HTRrestart(config)
            interpreter.attach(sampleDir=os.path.join(name, "sample0"), step=nstep)
            self.assertTrue(checkSolutionFile(interpreter, ref),
                            msg="Error on {} of {} test".format(name, self.testName))
      if (ranks > 1):
         with open("base.json", "r") as fin:
            config = json.load(fin)
            gNum = [config["Grid"]["xNum"], config["Grid"]["yNum"]]
         tilesPerRank = [2 if (gNum[0]%2) == 0 else 1,
                         2 if (gNum[1]%2) == 0 else 1]
         # All ranks along each direction
         for i in range(2):
            if (gNum[i] == 1): continue
            r = ranks
            tiles = tilesPerRank[:]
            while ((gNum[i] % (r*tiles[i])) != 0): r -= 1
            tiles[i] *= r
            runCase(tiles, tilesPerRank)
         # Ranks on both directions (only if we have enough ranks)
         if (ranks > 2):
            r = ranks
            idx = 0
            t = tilesPerRank[:]
            while (r != 1):
               done = True
               for i in range(2):
                  my_idx = (idx+i)%2
                  if ((gNum[my_idx] > 1) and ((gNum[my_idx] % (2*t[my_idx])) == 0)):
                     t[my_idx] *= 2
                     r = int(r/2)
                     idx = (my_idx + 1)%2
                     done = False
                     break
               if done: break
            runCase(t, tilesPerRank)
      else:
         for tiles in itertools.product(range(1, self.npart+1), repeat=2):
            runCase(tiles, tiles)

class Test3DTiledBase(TestTiledBase):
   def runTiledTest(self, ref):
      def runCase(t, tPR):
         name = "%dx%dx%d"%(t[0], t[1], t[2])
         with self.subTest(name):
            #print(" > case: %s" % name)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = t
               config["Mapping"]["tilesPerRank"] = tPR
               nstep = int(config["Integrator"]["maxIter"])
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = self.execute(name)
            # run the check
            interpreter = HTRrestart.HTRrestart(config)
            interpreter.attach(sampleDir=os.path.join(name, "sample0"), step=nstep)
            self.assertTrue(checkSolutionFile(interpreter, ref),
                            msg="Error on {} of {} test".format(name, self.testName))
      if (ranks > 1):
         with open("base.json", "r") as fin:
            config = json.load(fin)
            gNum = [config["Grid"]["xNum"],
                    config["Grid"]["yNum"],
                    config["Grid"]["zNum"]]
         tilesPerRank = [2 if (gNum[0]%2) == 0 else 1,
                         2 if (gNum[1]%2) == 0 else 1,
                         2 if (gNum[2]%2) == 0 else 1]
         # All ranks along each direction
         for i in range(3):
            if (gNum[i] == 1): continue
            r = ranks
            tiles = tilesPerRank[:]
            while ((gNum[i] % (r*tiles[i])) != 0): r -= 1
            tiles[i] *= r
            runCase(tiles, tilesPerRank)

         # More complex configurations
         def splitRanks(avoid = None):
            idx = 0
            r = ranks
            t = tilesPerRank[:]
            while (r != 1):
               done = True
               for i in range(3):
                  my_idx = (idx + i)%3
                  if ((avoid != None) and (my_idx == avoid)): my_idx = (idx + 1)%3
                  if ((gNum[my_idx] > 1) and ((gNum[my_idx] % (2*t[my_idx])) == 0)):
                     t[my_idx] *= 2
                     r = int(r/2)
                     idx = (my_idx + 1)%3
                     if (avoid and (idx == avoid)): idx = (idx + 1)%3
                     done = False
                     break
               if done: break
            return t
         if (ranks > 2):
            # Ranks along X-Y directions  (only if we have enough ranks)
            t = splitRanks(avoid = 2)
            runCase(t, tilesPerRank)
            # Ranks along X-Z directions
            t = splitRanks(avoid = 1)
            runCase(t, tilesPerRank)
            # Ranks along Y-Z directions
            t = splitRanks(avoid = 0)
            runCase(t, tilesPerRank)
         if (ranks > 4):
            # Ranks along all directions (only if we have enough ranks)
            t = splitRanks()
            runCase(t, tilesPerRank)
      else:
         for tiles in itertools.product(range(1, self.npart+1), repeat=3):
            runCase(tiles, tiles)

if __name__ == "__main__":
   # Get reference data
   downloadRefData()
   # load the test suite
   suite = unittest.TestLoader().discover(os.path.expandvars("$HTR_DIR/solverTests/"), pattern = "test.py")
   # Remove tests that are not appropriate for our configuration
   for group in suite:
      for test in group:
         if (os.path.expandvars("$ELECTRIC_FIELD") != "1"):
            if (test._tests[0].testName == "Speelman_DV250"): suite._tests.remove(group)
   result = 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
   sys.exit(result)
