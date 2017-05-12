"""
combine wprime and qcd datasets in to friendly form for train.py 
Usage:
    prepare.py <qcd_file>  <wprime_file> <out_file>
    prepare.py -h | --help

Arguments:
    <qcd_file>      HDF5 file of qcd images
    <wprime_file>   HDF5 file of wprime images
    <out_file>      HDF5 file to output to

Options:
    -h --help        Show this screen
"""
import h5py
import numpy as np
import os
from docopt import docopt


def main(qfile, wfile, outfile):
    qfile = os.path.abspath(qfile)
    wfile = os.path.abspath(wfile)
    outfile = os.path.abspath(outfile)

    with h5py.File(qfile, 'r') as f:
        qims = f['images'][:]
    l1 = qims.shape[0]
    print("Images of each type: {}".format(l1))
    images = np.zeros((l1*2, 25, 25))

    with h5py.File(wfile, 'r') as f:
        wims = f['images'][:l1]

    images[:l1, :, :] = qims
    images[l1:, :, :] = wims
    signal = np.zeros(l1*2)
    signal[l1:] += 1

    with h5py.File(outfile, 'w') as f:
        dset = f.create_dataset('image', data=images)
        sigs = f.create_dataset('signal', data=signal)

if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    print(arguments)

    main(arguments['<qcd_file>'], arguments['<wprime_file>'], arguments['<out_file>'])
