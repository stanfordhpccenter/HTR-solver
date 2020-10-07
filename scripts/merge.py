#!/usr/bin/env python3

import argparse
import h5py
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('hdf_file', nargs='+')
args = parser.parse_args()

# Read input metadata
num_files = len(args.hdf_file)
lo_bound = [] # array(num_files, array(3,int))
hi_bound = [] # array(num_files, array(3,int))
for i in range(num_files):
   base = os.path.basename(args.hdf_file[i])
   pat = r'([0-9]+),([0-9]+),([0-9]+)-([0-9]+),([0-9]+),([0-9]+).hdf'
   m = re.match(pat, base)
   assert(m is not None)
   lo_bound.append([int(m.group(1)), int(m.group(2)), int(m.group(3))])
   hi_bound.append([int(m.group(4)), int(m.group(5)), int(m.group(6))])

# Sanity checks
all_lo = [None, None, None] # array(3, array(num_files,int))
all_hi = [None, None, None] # array(3, array(num_files,int))
for k in range(3):
   all_lo[k] = sorted(set([c[k] for c in lo_bound]))
   all_hi[k] = sorted(set([c[k] for c in hi_bound]))
   assert len(all_lo[k]) == len(all_hi[k])
   for (prev_hi,next_lo) in zip(all_hi[k][:-1],all_lo[k][1:]):
      assert prev_hi == next_lo - 1
for (x_lo,x_hi) in zip(all_lo[0],all_hi[0]):
   for (y_lo,y_hi) in zip(all_lo[1],all_hi[1]):
      for (z_lo,z_hi) in zip(all_lo[2],all_hi[2]):
         found = False
         for i in range(num_files):
            if (lo_bound[i][0] == x_lo and hi_bound[i][0] == x_hi and
                lo_bound[i][1] == y_lo and hi_bound[i][1] == y_hi and
                lo_bound[i][2] == z_lo and hi_bound[i][2] == z_hi):
               found = True
               break
         assert found

# Combine actual data into output HDF file
# NOTE: The X and Z dimensions are flipped in the actual data, because Legion
# dumps data in column-major order.
shape = (all_hi[2][-1] - all_lo[2][0] + 1,
         all_hi[1][-1] - all_lo[1][0] + 1,
         all_hi[0][-1] - all_lo[0][0] + 1)
name = ('%s,%s,%s-%s,%s,%s.hdf'
        % (all_lo[0][0],  all_lo[1][0],  all_lo[2][0],
           all_hi[0][-1], all_hi[1][-1], all_hi[2][-1]))
with h5py.File(name, 'w') as fout:
   with h5py.File(args.hdf_file[0], 'r') as fin:
      for key,val in fin.attrs.items():
         fout.attrs.create(key, val)
      for fld in fin:
         fout.create_dataset(fld, shape, dtype=fin[fld].dtype)
   for i in range(num_files):
      with h5py.File(args.hdf_file[i], 'r') as fin:
         for key,val in fin.attrs.items():
            assert (val == fin.attrs.get(key)).all()
         for fld in fout:
            fout[fld][lo_bound[i][2]:hi_bound[i][2]+1,
                      lo_bound[i][1]:hi_bound[i][1]+1,
                      lo_bound[i][0]:hi_bound[i][0]+1] = fin[fld][:]

