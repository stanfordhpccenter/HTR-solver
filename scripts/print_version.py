#!/usr/bin/env python2

import argparse
import os
import glob
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

tiles = glob.glob(os.path.join(args.folder, "*hdf"))
version = None

for tl in tiles:
   fin = h5py.File(tl, "r")
   if version == None : version = fin.attrs.get("Versions")
   else : assert version == fin.attrs.get("Versions")
   fin.close()

print("Solution produced with")
print("  > Solver version: %s" % version[0])
print("  > Legion version: %s" % version[1])
