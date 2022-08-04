#!/usr/bin/env python

import os
import sys
import unittest
import subprocess

nThreads = 2
csize = int(0.2*(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1048576)) # 20% of the available memory in MB

class TestBase(object):
   def test(self):
      self.cwd = os.getcwd()
      os.chdir(os.path.expandvars("$HTR_DIR/unitTests/"+self.name))
      try:
         subprocess.check_output(["make", "clean"])
         subprocess.check_output(["make", "-j"])
      except:
         self.fail(self.name+" failed its setup")

      try:
         cmd = ["./"+self.name+".exec", "-ll:cpu", "1", "-ll:csize", str(csize)]
         if (os.path.expandvars("$USE_OPENMP") == "1"):
            cmd.extend(["-ll:ocpu", "1", "-ll:othr", str(nThreads)])
         if (os.path.expandvars("$USE_CUDA") == "1"):
            cmd.extend(["-ll:gpu", "1"])
         subprocess.check_output(cmd)
      except:
         self.fail("Failed " + self.name)

      try:
         subprocess.check_output(["make", "clean"])
      except:
         self.fail(self.name+" failed its teardown")
      os.chdir(self.cwd)

class MultiTestBase(object):
   def test(self):
      self.cwd = os.getcwd()
      os.chdir(os.path.expandvars("$HTR_DIR/unitTests/"+self.name))
      try:
         subprocess.check_output(["make", "clean"])
         subprocess.check_output(["make", "-j"])
      except:
         self.fail(self.name+" failed its setup")

      for i,t in enumerate(self.tests):
         with self.subTest(t):
            try:
               cmd = ["./"+t+".exec", "-ll:cpu", "1", "-ll:csize", str(csize)]
               if (os.path.expandvars("$USE_OPENMP") == "1"):
                  cmd.extend(["-ll:ocpu", "1", "-ll:othr", str(nThreads)])
               if (os.path.expandvars("$USE_CUDA") == "1"):
                  cmd.extend(["-ll:gpu", "1"])
               subprocess.check_output(cmd)
            except:
               self.fail("Failed " + t)

      try:
         subprocess.check_output(["make", "clean"])
      except:
         self.fail(self.name+" failed its teardown")
      os.chdir(self.cwd)

if __name__ == "__main__":
   suite = unittest.TestLoader().discover(os.path.expandvars("$HTR_DIR/unitTests/"), pattern = "test.py")
   # Remove tests that are not appropriate for our configuration
   for group in suite:
      for test in group:
         if (os.path.expandvars("$ELECTRIC_FIELD") != "1"):
            if (test._tests[0].name == "poissonTest"): suite._tests.remove(group)
   result = 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
   sys.exit(result)
