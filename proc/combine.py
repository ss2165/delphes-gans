"""combine.py helper file to combine small hdf image files in to one big one."""
import h5py
import numpy as np
import os

directory = os.path.abspath('./data')
files = ['qcd.hdf', 'qcd2.hdf', 'qcd3.hdf', 'qcd4.hdf']

files = [os.path.join(directory, f) for f in files]
arrays = []

for fname in files:
    print("Processing {}".format(fname))
    with h5py.File(fname, 'r') as f:
        for key in f.keys():
            arrays.append(f[key][:])

images = np.vstack(arrays)
output = os.path.join(directory, 'comb_qcd.hdf')
with h5py.File(output, 'w') as f:
    d_set = f.create_dataset('images', data=images, chunks=True)


