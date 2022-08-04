#!/usr/bin/env python3

import os
import sys
import unittest
import subprocess
sys.path.insert(1, os.path.expandvars("$HTR_DIR/solverTests/"))
import testAll

class unitTest(unittest.TestCase, testAll.Test2DTiledBase):
   testName = "PlanarJet"
   npart = 2
   def execute(self, name):
      MyOut = subprocess.check_output(["./MakeInput.py", name+".json", name+"Run.json"])
      MyOut = subprocess.check_output([testAll.executable, "-i", name+"Run.json", "-o", name])
      return MyOut

if __name__ == "__main__":
   # Get reference data
   testAll.downloadRefData()
   # Run the test
   unittest.main()
