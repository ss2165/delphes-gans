"""
combine wprime and qcd datasets in to friendly form for training
"""
import h5py
import numpy as np
import os

def main():
    qfile = os.path.abspath('data/comb_qcd.hdf')
    wfile = os.path.abspath('data/comb_wprime.hdf')
    outfile = os.path.abspath('data/prepared.hdf')

    with h5py.File(qfile, 'r') as f:
        qims = f['images'][:]
    l1 = qims.shape[0]
    print("Images of each type: {}".format(l1))
    images = np.zeros((l1*2,25, 25))

    with h5py.File(wfile, 'r') as f:
        wims = f['images'][:l1]

    images[:l1, :, :] = qims
    images[l1:, :, :] = wims
    signal = np.zeros(l1*2)
    signal[l1:] += 1

    with h5py.File(outfile, 'w') as f:
        dset = f.create_dataset('image', data=images)
        sigs = f.create_dataset('signal', data=signal)
main()