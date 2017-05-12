"""run_generator.py generates samples from a LAGAN generator
Usage:
    run_generator.py <N> <weights> <output> [--latent-space=<Z>]
    run_generator.py -h | --help

Arguments:
    <N>  Number of images to generate
    <weights>  Location of generator weights hdf5 file
    <output>   Location to save to 
Options:
    -h --help        Show this screen
    --latent-space=<Z>   Latent space (noise) vector size [default: 200]

"""
from docopt import docopt
import h5py
import numpy as np
import os
import time


def main(n_jets, outfile, latent_space, gen_weights=None):
    """
    :param n_jets: number of jet imags to generate
    :param gen_weights: weights to use
    :param outfile: hdf file to save to
    :param latent_space: latent space size
    :return: runtime
    """
    t0 = time.time()

    from models.networks.lagan import generator as build_generator
    g = build_generator(latent_space, return_intermediate=False)
    if gen_weights is not None:
        g.load_weights(os.path.abspath(gen_weights))

    noise = np.random.normal(0, 1, (n_jets, latent_space))
    sampled_labels = np.random.randint(0, 2, n_jets)
    generated_images = g.predict(
        [noise, sampled_labels.reshape(-1, 1)], verbose=False, batch_size=64)
    # Multiply to image scale and remove extraneous axis
    generated_images *= 100
    generated_images = np.squeeze(generated_images)

    # Save generated images
    with h5py.File(os.path.abspath(outfile), 'w') as f:
        dset = f.create_dataset('image', data=generated_images)
        sigs = f.create_dataset('signal', data=sampled_labels)

    return time.time() - t0  # return runtime


def performance(args, logfile='g_speed.txt'):
    """
    Helper function for running performance tests on generator
    :param args: command line arguments dictionary
    :param logfile: file to write times to
    :return: Returns nothing
    """
    # exponents = np.linspace(1, 5, 9)
    for N in np.logspace(1, 5, 9):
        # N = int(10**e)
        print(N)
        t = main(N, args['<weights>'], args['<output>'],
                 int(args['--latent-space']))
        with open(logfile, 'a') as f:
            f.write('{}\t{}\n'.format(N, t))  # record number of jets and runtime


if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    print(arguments)
    # performance(arguments)
    main(int(arguments['<N>']), arguments['<weights>'], arguments['<output>'],
         int(arguments['--latent-space']))
