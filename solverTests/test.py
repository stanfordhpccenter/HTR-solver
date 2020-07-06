import unittest
import subprocess
import os
import shutil
import h5py
import glob
import numpy as np
import itertools
import json

src_dir = os.path.join(os.path.expandvars("$HTR_DIR"), "src")
executable = os.path.join(src_dir, "prometeo.sh")

# Leave a tollerance for the noise introduced by atomic
def checkSolutionFile(path, refFile, tol=1e-10):
   tiles = glob.glob(os.path.join(path,"*hdf"))
   for tl in tiles:
      lb = map(int, tl.split("/")[-1].split("-")[0].split(","))
      tile = h5py.File(tl, "r")
      for fld in tile:
         f   =    tile[fld][:]
         ref = refFile[fld][:]
         if  (len(f.shape) == 3):
            for (k,j,i), value in np.ndenumerate(f):
               rval = ref[k+lb[2],j+lb[1],i+lb[0]]
               if (abs(value - rval) > max(tol*abs(rval), 1e-12)):
                  print(fld, value - rval, max(tol*abs(rval), 1e-12), rval)
                  return False
         elif (len(f.shape) == 4):
            for (k,j,i,l), value in np.ndenumerate(f):
               rval = ref[k+lb[2],j+lb[1],i+lb[0],l]
               if (abs(value - rval) > max(tol*abs(rval), 1e-12)):
                  print(fld, value - rval, max(tol*abs(rval), 1e-12), rval)
                  return False
      tile.close()
   # if we arrive here, the solution is good
   return True

class solverTest(unittest.TestCase):

   def test_3DPeriodic(self):
      """
      Test tri-periodic domain.
      """
      #print("Running 3DPeriodic test:")
      baseDir = os.getcwd()
      os.chdir(os.path.abspath("3DPeriodic"))

      # Load reference solution
      if int(os.getenv("USE_CUDA", "0")):
         ref = h5py.File("gpu_ref.hdf", "r")
      else:
         ref = h5py.File("cpu_ref.hdf", "r")

      npart = 2
      try:
         for case in itertools.product(range(1, npart+1), repeat=3):
            name = "%dx%dx%d"%(case)
            #print(" > case: %s" % name)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = case
               config["Mapping"]["tilesPerRank"] = case
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = subprocess.check_output([executable, "-i", name+".json", "-o", name])
            # run the check
            self.assertTrue(checkSolutionFile(os.path.join(name, "sample0/fluid_iter0000000020"), ref),
                                              msg="Error on %s of 3DPeriodic test" % name)

      finally:
         ref.close()
         os.chdir(baseDir)

   def test_3DPeriodic_Air(self):
      """
      Test tri-periodic domain with AirMixture.
      """
      baseDir = os.getcwd()
      os.chdir(os.path.abspath("3DPeriodic_Air"))

      # Load reference solution
      if int(os.getenv("USE_CUDA", "0")):
         ref = h5py.File("gpu_ref.hdf", "r")
      else:
         ref = h5py.File("cpu_ref.hdf", "r")

      npart = 2
      try:
         for case in itertools.product(range(1, npart+1), repeat=3):
            name = "%dx%dx%d"%(case)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = case
               config["Mapping"]["tilesPerRank"] = case
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = subprocess.check_output([executable, "-i", name+".json", "-o", name])
            # run the check
            self.assertTrue(checkSolutionFile(os.path.join(name, "sample0/fluid_iter0000000020"), ref),
                                              msg="Error on %s of 3DPeriodic_Air test" % name)

      finally:
         ref.close()
         os.chdir(baseDir)

   def test_ChannelFlow(self):
      """
      Test a pseudo Channel Flow.
      """
      baseDir = os.getcwd()
      os.chdir(os.path.abspath("ChannelFlow"))

      # Load reference solution
      if int(os.getenv("USE_CUDA", "0")):
         ref = h5py.File("gpu_ref.hdf", "r")
      else:
         ref = h5py.File("cpu_ref.hdf", "r")

      npart = 2
      try:
         for case in itertools.product(range(1, npart+1), repeat=3):
            name = "%dx%dx%d"%(case)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = case
               config["Mapping"]["tilesPerRank"] = case
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = subprocess.check_output([executable, "-i", name+".json", "-o", name])
            # run the check
            self.assertTrue(checkSolutionFile(os.path.join(name, "sample0/fluid_iter0000000010"), ref),
                                              msg="Error on %s of ChannelFlow test" % name)

      finally:
         ref.close()
         os.chdir(baseDir)

   def test_BoundaryLayer(self):
      """
      Test a boundary layer.
      """
      baseDir = os.getcwd()
      os.chdir(os.path.abspath("BoundaryLayer"))

      # Load reference solution
      if int(os.getenv("USE_CUDA", "0")):
#         ref = h5py.File("gpu_ref.hdf", "r")
         ref = h5py.File("cpu_ref.hdf", "r")
      else:
         ref = h5py.File("cpu_ref.hdf", "r")

      npart = 2
      try:
         for case in itertools.product(range(1, npart+1), repeat=3):
            name = "%dx%dx%d"%(case)
            # create directory
            if os.path.exists(name): shutil.rmtree(name)
            os.makedirs(name)
            #create input file
            with open("base.json", "r") as fin:
               config = json.load(fin)
               config["Mapping"]["tiles"]        = case
               config["Mapping"]["tilesPerRank"] = case
               with open(name+".json", "w") as fout:
                  json.dump(config, fout, indent=3)
            # run the case
            MyOut = subprocess.check_output(["./MakeInput.py", name+".json"])
            MyOut = subprocess.check_output([executable, "-i", name+".json", "-o", name])
            # run the check
            self.assertTrue(checkSolutionFile(os.path.join(name, "sample0/fluid_iter0000000040"), ref),
                                              msg="Error on %s of BoundaryLayer test" % name)

      finally:
         ref.close()
         os.chdir(baseDir)

if __name__ == "__main__":
   unittest.main()
