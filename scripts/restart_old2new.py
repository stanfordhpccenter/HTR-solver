#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--np', nargs='?', default=1, type=int,
                     help='number of cores')
parser.add_argument('hdf_file', nargs='+')
args = parser.parse_args()

def processFile(f):
   print(f)
   with h5py.File(f, 'r+') as fout:
      dudtBoundary_old = fout["dudtBoundary"][:]
      del fout["dudtBoundary"]
      del fout["velocity_old_NSCBC"]
      del fout["temperature_old_NSCBC"]
      shape = dudtBoundary_old.shape
      fout.create_dataset("dudtBoundary", shape=shape, dtype = np.dtype("(3,)f8"))
      dudtBoundary      = np.ndarray(           shape, dtype = np.dtype('(3,)f8'))
      for (k,j,i), v in np.ndenumerate(dudtBoundary_old):
         dudtBoundary[k,j,i] = [v, 0.0, 0.0]
      fout["dudtBoundary"][:] = dudtBoundary

#for f in args.hdf_file: processFile(f)
Parallel(n_jobs=args.np)(delayed(processFile)(f) for f in args.hdf_file)
