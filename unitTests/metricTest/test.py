import os
import sys
sys.path.insert(1, os.path.expandvars("$HTR_DIR/unitTests/"))
import unittest
import subprocess
import testAll

class unitTest(unittest.TestCase, testAll.MultiTestBase):
   name = "metricTest"
   tests = ["operatorsTest_Periodic", "operatorsTest_Collocated", "operatorsTest_Staggered",
               "metricTest_Periodic",    "metricTest_Collocated",    "metricTest_Staggered"]

if __name__ == "__main__":
   unittest.main()
