#!/usr/bin/env python

import os
import sys
import unittest
import subprocess

nThreads = 2

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
         subprocess.check_output(["./"+self.name+".exec", "-ll:ocpu", "1", "-ll:othr", str(nThreads)])
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
               subprocess.check_output(["./"+t+".exec", "-ll:ocpu", "1", "-ll:othr", str(nThreads)])
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
