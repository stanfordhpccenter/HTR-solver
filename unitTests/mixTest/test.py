import os
import sys
sys.path.insert(1, os.path.expandvars("$HTR_DIR/unitTests/"))
import unittest
import subprocess
import testAll

class unitTest(unittest.TestCase, testAll.MultiTestBase):
   name = "mixTest"
   tests = ["mixTest_ConstPropMix", "mixTest_AirMix", "mixTest_CH41StMix",
            "mixTest_CH4_30SpMix", "mixTest_CH4_43SpIonsMix", "mixTest_CH4_26SpIonsMix",
            "mixTest_FFCM1Mix", "mixTest_BoivinMix", "mixTest_H2_UCSDMix"]

if __name__ == "__main__":
   unittest.main()
