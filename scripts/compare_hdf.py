#!/usr/bin/env python3

import argparse
import h5py
import itertools
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('f1')
parser.add_argument('f2')
args = parser.parse_args()

f1 = h5py.File(args.f1, 'r')
f2 = h5py.File(args.f2, 'r')

for fld in f1:
    assert fld in f2 and f1[fld].shape == f2[fld].shape
for fld in f2:
    assert fld in f1 and f1[fld].shape == f2[fld].shape
for fld in f1:
    diff = np.absolute(f1[fld][:] - f2[fld][:])
    max_diff = np.max(diff)
    max_loc = np.unravel_index(np.argmax(diff), diff.shape)
    print('%s: max diff = %s at %s' % (fld, max_diff, max_loc))

f2.close()
f1.close()
