import os
import sys
sys.path.insert(1, os.path.expandvars("$HTR_DIR/unitTests/"))
import unittest
import subprocess
import testAll

class unitTest(unittest.TestCase, testAll.TestBase):
   name = "laserTest"

if __name__ == "__main__":
   unittest.main()
