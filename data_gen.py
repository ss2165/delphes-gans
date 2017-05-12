"""data_gen.py converts Delphes ROOT files to jet images and outputs hdf files
Usage:
    data_gen.py <in_file> [-w] [-o <out_file>] [--ptmin=<ptmin>] [--ptmax=<ptmax>]
    data_gen.py -h | --help

Arguments:
    <in_file>  Root file to extract from

Options:
    -h --help        Show this screen
    -o <out_file>    file to save to
    --ptmin=<ptmin>  Minimum pT of jets in GeV [default: 250]
    --ptmax=<ptmax>  Maximum pT of jets in GeV [default: 300]

"""

from docopt import docopt
import numpy as np
import os
import h5py

from tqdm import tqdm
from jetimage.readjet import RootEvents, JetImage
from jetimage.analysis import plot_jet, average_image, image_set


def main(fname, output, ptmin, ptmax):
    """
    :param fname: ROOT file to extract from
    :param output: hdf file to output to
    :param ptmin: minimum pT
    :param ptmax: maximum pT
    :return: None
    """
    print("Reading ROOT file: {}".format(fname))
    jets = RootEvents(fname, ptmin=ptmin, ptmax=ptmax)
    images = []
    norm = 4000.0
    print("Pre-processing Jet images.")
    for jet in tqdm(jets):
        im0 = JetImage(jet)
        np.clip(im0.image_array, -1, norm, out=im0.image_array)
        im0.normalise(norm)
        im0.sj2_rotate()
        im0.flip()
        im0.normalise(1./norm)
        images.append(im0)

    im_ar = image_set(images)  # convert list of image objects to array of image arrays
    if output is not None:
        print("Saving.")
        if len(im_ar) > 0:
            savefile = os.path.abspath(output)
            with h5py.File(savefile, "a") as f:
                sname = set_name(f.keys())
                dset = f.create_dataset(sname, data=im_ar, chunks=True)
        else:
            print("No images read.")
    else:
        # if no output is specified just plot average image
        av = average_image(im_ar)
        plot_jet(av)


def set_name(keys):
    """examine dataset names (should be integer numbers) and provide the next"""
    if len(keys) == 0:
        return "1"
    ints = [int(key) for key in keys]
    m = max(ints)
    return str(m+1)

if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    main(arguments['<in_file>'], arguments['-o'],
         float(arguments['--ptmin']), float(arguments['--ptmax']))
