#!/usr/bin/env python3

import argparse
import os
import glob
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

version = None

if os.path.exists(os.path.join(args.folder, "master.hdf")):
   fin = h5py.File(os.path.join(args.folder, "master.hdf"), "r")
   version = fin.attrs.get("Versions")

else:
   # Keep for backward compatibility
   tiles = glob.glob(os.path.join(args.folder, "*hdf"))

   for t, tl in enumerate(tiles):
      fin = h5py.File(tl, "r")
      if (t == 0):
         version = fin.attrs.get("Versions")
      else:
         for i in range(len(version)):
            assert (version[i] == fin.attrs.get("Versions")[i])
      fin.close()

print("Solution produced with")
print("  > Solver version: %s" % version[0])
print("  > Legion version: %s" % version[1])
